"""
Stage 3c — Summary (two-pass).

Pass 1: Opus detailed summary (~180 words), evidence-grounded in document text.
Pass 2: Haiku layman rewrite (~80-120 words). Reads ONLY the detailed text
        from pass 1 — does not re-read the document. Saves cost; locks layman
        to detailed's factual content.

Per synthesis_plan §Summary. Adapted from
eis_pipeline/pipeline/stage2_fields/summary.py.

Retrieval cascade (synthesis_plan §Key design decision 0):
  Primary sections: summary -> purpose_and_need -> affected_environment
  Keyword fallback: "purpose", "proposed", "needed", "need for the action",
                    "project description", "would result", "alternatives considered"
  First-N fallback: first 10000 chars

When retrieval is degraded (no primary section detected), the field's
status downgrades to "needs_review" and provenance.source is set to
"fallback_keyword_search" or "fallback_first_n_chunks".

The schema's FieldWithStatus[str] holds the value; provenance points at the
union of retrieval windows so the critic can re-grounding-check against
that range.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from . import config
from .ingest import IngestArtifact
from .retrieval import (
    RetrievalResult,
    SOURCE_FALLBACK_FIRST_N,
    SOURCE_FALLBACK_KEYWORD,
    cascade,
)
from .schema import EISRecord, FieldWithStatus, Provenance

if TYPE_CHECKING:
    from .llm_client import LLMClient
    from .sections import SectionsArtifact

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# How much retrieval text to feed Opus. Caps cost; 20k chars ≈ ~5k input tokens.
_MAX_DETAILED_INPUT_CHARS = 20_000

# Primary section preference order for the summary.
_PRIMARY_SECTIONS = ["summary", "purpose_and_need", "affected_environment"]

# Keywords used when no primary section is detected.
_FALLBACK_KEYWORDS = [
    "purpose", "proposed", "needed", "need for", "project description",
    "affected environment", "environmental impact", "would result",
    "alternatives considered",
]


def run(
    record: EISRecord,
    ingest: IngestArtifact,
    sections_artifact: "SectionsArtifact",
    llm: "LLMClient | None" = None,
) -> list[str]:
    """Mutate `record.summary` and `record.layman_summary`. Returns warnings.

    `llm=None` -> both fields stay unpopulated with status='needs_review'.
    Detailed pass failure -> layman pass is skipped automatically (no detailed
    text to rewrite).
    """
    warnings: list[str] = []

    if llm is None:
        record.summary = FieldWithStatus[str](
            value=None, status="needs_review", provenance=None,
        )
        record.layman_summary = FieldWithStatus[str](
            value=None, status="needs_review", provenance=None,
        )
        warnings.append("summary: no llm provided")
        return warnings

    # --- Retrieval cascade ---
    retrieval = cascade(
        ingest, sections_artifact,
        primary_sections=_PRIMARY_SECTIONS,
        fallback_keywords=_FALLBACK_KEYWORDS,
        max_keyword_windows=6,
        keyword_window_chars=2000,
        first_n_max_chars=10_000,
    )

    if not retrieval.windows:
        warnings.append("summary: empty document; no retrieval windows")
        record.summary = FieldWithStatus[str](
            value=None, status="needs_review", provenance=None,
        )
        record.layman_summary = FieldWithStatus[str](
            value=None, status="needs_review", provenance=None,
        )
        return warnings

    # --- Pass 1: Opus detailed ---
    title = record.title.value or record.publication_id
    detailed_text = _opus_detailed(retrieval, title, llm)
    if detailed_text is None:
        warnings.append("summary: Opus detailed pass returned no usable summary")
        record.summary = FieldWithStatus[str](
            value=None,
            status="needs_review",
            provenance=_build_provenance(retrieval),
        )
        record.layman_summary = FieldWithStatus[str](
            value=None, status="needs_review", provenance=None,
        )
        return warnings

    summary_status = "needs_review" if retrieval.degraded else "ok"
    record.summary = FieldWithStatus[str](
        value=detailed_text,
        status=summary_status,
        provenance=_build_provenance(retrieval),
        confidence=0.85 if not retrieval.degraded else 0.65,
    )
    if retrieval.degraded:
        warnings.append(
            f"summary: retrieval degraded "
            f"(source={retrieval.provenance_source}); marked needs_review"
        )

    # --- Pass 2: Haiku layman ---
    layman_text = _haiku_layman(detailed_text, llm)
    if layman_text is None:
        warnings.append("summary: layman rewrite failed; layman_summary left empty")
        record.layman_summary = FieldWithStatus[str](
            value=None, status="needs_review", provenance=None,
        )
    else:
        # Layman provenance points at detailed_summary, not at the doc — it
        # was a rewrite, not an extraction. We mark source=haiku_classifier
        # (the layman pass uses Haiku) and leave section/char_offset_raw null
        # because the source isn't a doc span.
        record.layman_summary = FieldWithStatus[str](
            value=layman_text,
            status=summary_status,  # inherits detailed's status
            provenance=Provenance(source="haiku_classifier",
                                  note="rewrite of summary.value (no re-read)"),
            confidence=0.80 if not retrieval.degraded else 0.60,
        )

    return warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_provenance(retrieval: RetrievalResult) -> Provenance:
    """Build a Provenance covering the union of retrieval windows."""
    union = retrieval.union_offset()
    section: str | None = retrieval.section
    note = None
    if retrieval.provenance_source == SOURCE_FALLBACK_KEYWORD:
        note = f"keyword search produced {len(retrieval.windows)} window(s)"
    elif retrieval.provenance_source == SOURCE_FALLBACK_FIRST_N:
        note = f"first {retrieval.total_chars} chars of doc"

    # provenance_source is a runtime string — schema validates it against the
    # ProvenanceSource Literal at field-assignment time.
    return Provenance(
        source=retrieval.provenance_source,  # type: ignore[arg-type]
        page=None,
        char_offset_raw=union,
        section=section,  # type: ignore[arg-type]
        note=note,
    )


def _opus_detailed(retrieval: RetrievalResult, title: str, llm: "LLMClient") -> str | None:
    prompt_template = (PROMPTS_DIR / "3c_summary_detailed.txt").read_text(encoding="utf-8")
    document_text = retrieval.combined_text[:_MAX_DETAILED_INPUT_CHARS]
    prompt = (
        prompt_template
        .replace("{title}", title)
        .replace("{document_text}", document_text)
    )

    # NOTE: synthesis_plan §Summary specifies Opus for the detailed pass.
    # Temporarily routed through Sonnet because the user's Bedrock account
    # doesn't have an active inference profile for opus-4-7 yet. Sonnet
    # handles structured-summary prompts capably; revert to llm.models["opus"]
    # once Opus access lands. (Cost difference: ~5x cheaper on Sonnet, so
    # not a strict regression in the meantime.)
    model_id = llm.models["sonnet"]
    try:
        result = llm.call_json(
            model=model_id,
            system="You are a careful analyst. Return only the requested JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
            label="3c_summary_detailed",
        )
    except Exception as exc:
        logger.warning("3c detailed pass LLM call failed: %s", exc)
        return None

    if not isinstance(result, dict):
        return None
    if not result.get("sufficient_information"):
        return None
    summary = result.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None
    return summary.strip()


def _haiku_layman(detailed_text: str, llm: "LLMClient") -> str | None:
    prompt_template = (PROMPTS_DIR / "3c_summary_layman.txt").read_text(encoding="utf-8")
    prompt = prompt_template.replace("{detailed_summary}", detailed_text)

    try:
        result = llm.call_json(
            model=llm.models["haiku"],
            system="You are a careful rewriter. Return only the requested JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
            label="3c_summary_layman",
        )
    except Exception as exc:
        logger.warning("3c layman pass LLM call failed: %s", exc)
        return None

    if not isinstance(result, dict):
        return None
    layman = result.get("layman_summary")
    if not isinstance(layman, str) or not layman.strip():
        return None
    return layman.strip()
