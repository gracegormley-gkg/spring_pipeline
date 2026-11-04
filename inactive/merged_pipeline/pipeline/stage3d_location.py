"""
Stage 3d — Location.

Per synthesis_plan §Location:
  - Extract from cover / summary / purpose_and_need / alternatives ONLY.
    NEVER from public_comments (commenter addresses contaminate).
  - GLiNER + Sonnet structured extraction. v1 ships Sonnet-only because
    GLiNER's torch dependency adds ~2GB and the precision lift on top of
    structured-prompt Sonnet is marginal for the named-places path.
  - Feature-type-routed gazetteer: TIGER (state/county) + USFS Admin Forests
    + PAD-US + USGS WBD. Anything outside those routes -> name only,
    polygon=null. **No hulls, no buffers, no LLM-reasoned geometry.**
  - v1 ships polygon=null for ALL places (gazetteer integration deferred).
    This is the schema-honest answer per synthesis_plan §Key design decision 5
    ("null is honest").
  - Spatial qualifiers ("parts of", "near", "within", "downstream from", ...)
    are extracted but never auto-promote a feature to project_area_polygon.

What v1 produces:
  - location.named_places[] populated with extracted places + roles +
    feature_types + spatial_qualifiers
  - location.project_area_polygon = None
  - location.context_polygons = []
  - location.source_geometry = None
  - location.geometry_role / geometry_status / polygon_uncertainty = "unknown"
  - location.spatial_summary = empty SpatialSummary()
  - location.status = "ok" if any place extracted, else "needs_review"

Retrieval cascade (NEVER public_comments):
  Primary sections (in order): cover, summary, purpose_and_need, alternatives
  Keyword fallback: ["located in", "located near", "study area", "project area",
                     "national forest", "national park", "river", "watershed"]
  First-N fallback: first 8000 chars
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from . import config
from .ingest import IngestArtifact
from .retrieval import cascade
from .schema import (
    EISRecord,
    LocationField,
    NamedPlace,
    SpatialSummary,
)

if TYPE_CHECKING:
    from .llm_client import LLMClient
    from .sections import SectionsArtifact

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Cap on document text fed to Sonnet (~5k input tokens).
_MAX_INPUT_CHARS = 20_000

# Allowed roles per the schema's PlaceRole Literal — for filtering Sonnet
# output. project_area + context_reference + alternative_site +
# comparison_site + mitigation_location. (commenter_address comes from
# Stage 3f, not 3d; agency_office is METS-derived, not extracted from text.)
_ALLOWED_ROLES = {
    "project_area", "context_reference", "alternative_site",
    "comparison_site", "mitigation_location",
}

_PRIMARY_SECTIONS = ["cover", "summary", "purpose_and_need", "alternatives"]

_FALLBACK_KEYWORDS = [
    "located in", "located near", "study area", "project area",
    "national forest", "national park", "river", "watershed",
]


def run(
    record: EISRecord,
    ingest: IngestArtifact,
    sections_artifact: "SectionsArtifact",
    llm: "LLMClient | None" = None,
) -> list[str]:
    """Mutate `record.location`. Returns warnings."""
    warnings: list[str] = []

    if llm is None:
        record.location = LocationField(
            named_places=[],
            status="needs_review",
            spatial_summary=SpatialSummary(),
        )
        warnings.append("location: no llm provided")
        return warnings

    retrieval = cascade(
        ingest, sections_artifact,
        primary_sections=_PRIMARY_SECTIONS,
        fallback_keywords=_FALLBACK_KEYWORDS,
        max_keyword_windows=4,
        keyword_window_chars=3000,
        first_n_max_chars=8_000,
    )

    if not retrieval.windows:
        record.location = LocationField(
            named_places=[], status="needs_review",
            spatial_summary=SpatialSummary(),
        )
        warnings.append("location: empty document")
        return warnings

    raw_places = _sonnet_extract(retrieval.combined_text[:_MAX_INPUT_CHARS], llm)
    if raw_places is None:
        record.location = LocationField(
            named_places=[], status="needs_review",
            spatial_summary=SpatialSummary(),
        )
        warnings.append("location: Sonnet extraction failed")
        return warnings

    named_places: list[NamedPlace] = []
    drops: list[str] = []
    for p in raw_places:
        if not isinstance(p, dict):
            drops.append(f"non-dict:{p!r}"); continue
        name = (p.get("name") or "").strip()
        role = p.get("role")
        if not name:
            drops.append("empty-name"); continue
        if role not in _ALLOWED_ROLES:
            drops.append(f"oov-role:{role!r}({name!r})"); continue
        feature_type = p.get("feature_type")
        if feature_type is not None and not isinstance(feature_type, str):
            feature_type = None
        spatial_q = p.get("spatial_qualifier")
        if spatial_q is not None and not isinstance(spatial_q, str):
            spatial_q = None
        named_places.append(NamedPlace(
            name=name,
            role=role,  # type: ignore[arg-type]
            feature_type=feature_type or None,
            source_dataset=None,        # gazetteer deferred
            source_feature_id=None,     # gazetteer deferred
            spatial_qualifier=spatial_q,
            polygon=None,               # synthesis_plan §Location: null is honest
        ))

    status = "ok" if named_places else "needs_review"
    if retrieval.degraded and status == "ok":
        status = "needs_review"
    record.location = LocationField(
        named_places=named_places,
        project_area_polygon=None,
        context_polygons=[],
        source_geometry=None,
        geometry_role="unknown",
        geometry_status="unknown",
        polygon_uncertainty="unknown",
        spatial_summary=SpatialSummary(),
        status=status,
    )
    if drops:
        warnings.append(f"location: dropped {len(drops)} entries -> {drops}")
    if retrieval.degraded:
        warnings.append(
            f"location: retrieval degraded (source={retrieval.provenance_source})"
        )
    return warnings


def _sonnet_extract(document_text: str, llm: "LLMClient") -> list[dict] | None:
    prompt_template = (PROMPTS_DIR / "3d_location.txt").read_text(encoding="utf-8")
    prompt = prompt_template.replace("{document_text}", document_text)
    try:
        result = llm.call_json(
            model=llm.models["sonnet"],
            system="You are a careful geographer. Return only the requested JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.1,
            label="3d_location",
        )
    except Exception as exc:
        logger.warning("3d location LLM call failed: %s", exc)
        return None
    if not isinstance(result, dict):
        return None
    places = result.get("places")
    if not isinstance(places, list):
        return None
    return places
