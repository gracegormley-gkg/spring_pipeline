"""
Stage 4 — Critic (deterministic gates).

Per synthesis_plan.md §Critic:
  - 4a Deterministic gates: hard pass/fail checks. Run on every doc; never
    skipped.
  - 4b Tiered LLM critic (Haiku presence -> Sonnet grounded -> Opus retry):
    NOT YET IMPLEMENTED. v1 ships deterministic gates only. The schema's
    ValidationField.verdicts list stays empty in v1; the LLM critic is the
    natural next addition (synthesis_plan §Build order step 10).
  - 4c Aggregator: review_routing decision based on hard-gate results.

Hard gates (synthesis_plan §Critic — deterministic):
  - verbatim_quotes:    every Quote.text_raw must be a substring of
                        ingest.raw_text at its declared char_offset_raw.
  - year_range:         year, if populated, must fall in [NEPA_YEAR, MAX_YEAR].
  - theme_vocab:        every primary in config.ALL_PRIMARY_THEMES; every
                        subtheme in config.ALL_SUBTHEMES; every subtheme.parent
                        in chosen primaries AND in config.THEMES[parent].
                        (The schema's ThemesField validator already enforces
                         the parent-must-be-chosen-primary rule; this gate
                         double-checks vocab membership.)
  - geocoding_centroid: trivially passes in v1 (no polygons / centroids
                        produced). When Stage 3d gazetteer ships, this will
                        check that representative_point is inside the
                        project_area_polygon's bounding box.
  - schema_validation:  re-validate the EISRecord via Pydantic; pass iff
                        no ValidationError.

Routing rules (per ultraplan §Validation routing):
  - 0 hard-gate failures -> auto_approve
  - any hard-gate failure -> full_review
  - (When LLM critic ships: 1 'no' -> partial_review; 2+ 'no' -> full_review.)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from . import config
from .schema import (
    EISRecord,
    HardGates,
    ValidationField,
)

if TYPE_CHECKING:
    from .ingest import IngestArtifact

logger = logging.getLogger(__name__)


def run(record: EISRecord, ingest: "IngestArtifact") -> list[str]:
    """Run deterministic gates and populate `record.validation`. Mutates record.
    Returns warnings (each gate failure adds one warning string)."""
    warnings: list[str] = []

    quotes_pass, quote_warnings = _check_quotes(record, ingest)
    warnings.extend(quote_warnings)

    year_pass, year_warnings = _check_year(record)
    warnings.extend(year_warnings)

    themes_pass, theme_warnings = _check_themes(record)
    warnings.extend(theme_warnings)

    geocoding_pass, geo_warnings = _check_geocoding(record)
    warnings.extend(geo_warnings)

    schema_pass, schema_warnings = _check_schema(record)
    warnings.extend(schema_warnings)

    hard_gates = HardGates(
        verbatim_quotes="pass" if quotes_pass else "fail",
        year_range="pass" if year_pass else "fail",
        theme_vocab="pass" if themes_pass else "fail",
        geocoding_centroid="pass" if geocoding_pass else "fail",
        schema_validation="pass" if schema_pass else "fail",
    )

    routing, reasons = _decide_routing(hard_gates)

    record.validation = ValidationField(
        approach="rule_threshold_v1",
        verdicts=[],   # LLM critic deferred (Stage 4b)
        critic_pass_rate=None,
        field_level_confidence={},
        hard_gates=hard_gates,
        self_consistency={},
        review_routing=routing,
        routing_reasons=reasons,
    )

    return warnings


# ---------------------------------------------------------------------------
# Gate: verbatim quotes
# ---------------------------------------------------------------------------

def _check_quotes(record: EISRecord, ingest: "IngestArtifact") -> tuple[bool, list[str]]:
    """Every Stakeholder StanceRecord.quote.text_raw must be a substring of
    ingest.raw_text at its declared char_offset_raw."""
    warnings: list[str] = []
    all_pass = True
    raw = ingest.raw_text
    n = len(raw)

    for sh in record.stakeholders:
        for sr in sh.stance_records:
            q = sr.quote
            if q is None:
                continue
            s, e = q.char_offset_raw
            if s < 0 or e > n or e <= s:
                warnings.append(
                    f"hard_gate.verbatim_quotes: quote offsets ({s}, {e}) "
                    f"out of range for raw_text length {n} (stakeholder={sh.comment_author.name!r})"
                )
                all_pass = False
                continue
            actual = raw[s:e]
            if actual != q.text_raw:
                warnings.append(
                    f"hard_gate.verbatim_quotes: text_raw does not match raw_text "
                    f"at offsets ({s}, {e}); stakeholder={sh.comment_author.name!r}"
                )
                all_pass = False
    return all_pass, warnings


# ---------------------------------------------------------------------------
# Gate: year range
# ---------------------------------------------------------------------------

def _check_year(record: EISRecord) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    year = record.year.value
    if year is None:
        # Year missing isn't a hard-gate failure here; status downgrades
        # already happened in Stage 3a. Treat as pass (gate only checks
        # range when a value is present).
        return True, warnings
    if not (config.NEPA_YEAR <= year <= config.MAX_YEAR):
        warnings.append(
            f"hard_gate.year_range: year={year} outside [{config.NEPA_YEAR}, {config.MAX_YEAR}]"
        )
        return False, warnings
    return True, warnings


# ---------------------------------------------------------------------------
# Gate: theme vocab
# ---------------------------------------------------------------------------

def _check_themes(record: EISRecord) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    all_pass = True
    chosen_primaries = {p.value for p in record.themes.primary}
    for p in record.themes.primary:
        if p.value not in config.ALL_PRIMARY_THEMES:
            warnings.append(
                f"hard_gate.theme_vocab: primary {p.value!r} not in vocab"
            )
            all_pass = False
    for s in record.themes.subthemes:
        if s.value not in config.ALL_SUBTHEMES:
            warnings.append(
                f"hard_gate.theme_vocab: subtheme {s.value!r} not in vocab"
            )
            all_pass = False
            continue
        if s.parent not in chosen_primaries:
            warnings.append(
                f"hard_gate.theme_vocab: subtheme {s.value!r} parent {s.parent!r} "
                f"not in chosen primaries {sorted(chosen_primaries)}"
            )
            all_pass = False
            continue
        if s.value not in config.THEMES.get(s.parent, []):
            warnings.append(
                f"hard_gate.theme_vocab: subtheme {s.value!r} not listed under "
                f"parent {s.parent!r} in config.THEMES"
            )
            all_pass = False
    return all_pass, warnings


# ---------------------------------------------------------------------------
# Gate: geocoding centroid
# ---------------------------------------------------------------------------

def _check_geocoding(record: EISRecord) -> tuple[bool, list[str]]:
    """v1: no polygons produced -> trivially passes. When gazetteer ships,
    this will check representative_point ∈ project_area_polygon.bbox."""
    warnings: list[str] = []
    rp = record.location.spatial_summary.representative_point
    poly = record.location.project_area_polygon
    if rp is None and poly is None:
        return True, warnings
    if rp is not None and poly is None:
        warnings.append(
            "hard_gate.geocoding_centroid: representative_point set but no "
            "project_area_polygon to validate against"
        )
        return False, warnings
    # Both set: bounding-box containment check is a v2 concern (needs
    # shapely / GeoJSON parsing). For v1, accept if both present.
    return True, warnings


# ---------------------------------------------------------------------------
# Gate: schema validation
# ---------------------------------------------------------------------------

def _check_schema(record: EISRecord) -> tuple[bool, list[str]]:
    """Round-trip the record through Pydantic to catch any post-stage drift."""
    warnings: list[str] = []
    try:
        EISRecord.model_validate(record.model_dump())
        return True, warnings
    except ValidationError as exc:
        warnings.append(f"hard_gate.schema_validation: {exc}")
        return False, warnings


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _decide_routing(gates: HardGates) -> tuple[str, list[str]]:
    """Translate hard-gate outcomes into a review_routing value + reasons."""
    failures: list[str] = []
    if gates.verbatim_quotes == "fail":
        failures.append("verbatim_quotes failed")
    if gates.year_range == "fail":
        failures.append("year_range failed")
    if gates.theme_vocab == "fail":
        failures.append("theme_vocab failed")
    if gates.geocoding_centroid == "fail":
        failures.append("geocoding_centroid failed")
    if gates.schema_validation == "fail":
        failures.append("schema_validation failed")

    if failures:
        return "full_review", failures
    return "auto_approve", []
