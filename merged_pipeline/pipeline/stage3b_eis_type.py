"""
Stage 3b — EIS Type detection.

Two-pass:
  1. Regex on the cover section text (config.EIS_TYPE_REGEX).
     The regex matches both modern "Final Environmental Impact Statement" and
     NEPA-era 1970s "FINAL ENVIRONMENTAL STATEMENT" (Impact optional).
  2. Haiku 4.5 fallback on cover text if regex finds nothing.

Output is one of {Draft, Final, Supplemental, ROD, NOI, Unlabelled}, matching
schema's EISTypeValue Literal.

Per synthesis_plan.md §EIS type. Adapted from
v1_multiagent_pipeline/pipeline/stage3b_eis_type.py:
  - Drops the USFS-Final manifest assertion (full-NUL corpus).
  - Uses the merged sections_artifact API (DetectedSection.by_name()).
  - Updated regex tolerates the optional "Impact" word.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from . import config
from .ingest import IngestArtifact
from .schema import (
    EISRecord,
    FieldWithStatus,
    Provenance,
)

if TYPE_CHECKING:
    from .llm_client import LLMClient
    from .sections import SectionsArtifact

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def run(
    record: EISRecord,
    ingest: IngestArtifact,
    sections_artifact: "SectionsArtifact",
    llm: "LLMClient | None" = None,
) -> list[str]:
    """Mutate `record.eis_type`. Returns warnings.

    `llm` may be None -> Haiku fallback skipped; if regex misses, eis_type
    is set to "Unlabelled" / status="needs_review".
    """
    warnings: list[str] = []
    by_name = sections_artifact.by_name() if hasattr(sections_artifact, "by_name") else {}

    cover = by_name.get("cover")
    if cover is None or cover.char_span is None:
        warnings.append("eis_type: cover section missing; cannot detect EIS type")
        record.eis_type = FieldWithStatus(value="Unlabelled", status="needs_review")
        return warnings

    s, e = cover.char_span
    cover_text = ingest.raw_text[s:e]

    # Pass 1: regex
    eis_type = _regex_match(cover_text)
    if eis_type is not None:
        record.eis_type = FieldWithStatus(
            value=eis_type,
            status="extracted_from_cover",
            provenance=Provenance(
                source="regex",
                page=cover.pages[0] if cover.pages else None,
                section="cover",
                char_offset_raw=(s, e),
                note="EIS_TYPE_REGEX match",
            ),
            confidence=0.95,
        )
        logger.info("Stage 3b: eis_type=%s via regex", eis_type)
        return warnings

    # Pass 2: Haiku fallback
    if llm is None:
        record.eis_type = FieldWithStatus(
            value="Unlabelled",
            status="needs_review",
            provenance=Provenance(source="regex", section="cover", char_offset_raw=(s, e),
                                  note="EIS_TYPE_REGEX missed; no llm provided"),
            confidence=0.30,
        )
        warnings.append("eis_type: regex missed and no llm provided")
        return warnings

    logger.info("Stage 3b: regex miss; calling Haiku on cover text (%d chars)", len(cover_text))
    fallback_value = _haiku_fallback(cover_text, llm)
    if fallback_value in config.VALID_EIS_TYPES:
        record.eis_type = FieldWithStatus(
            value=fallback_value,
            status="extracted_from_cover" if fallback_value != "Unlabelled" else "needs_review",
            provenance=Provenance(
                source="haiku_classifier",
                page=cover.pages[0] if cover.pages else None,
                section="cover",
                char_offset_raw=(s, e),
                note="3b Haiku fallback",
            ),
            confidence=0.80 if fallback_value != "Unlabelled" else 0.40,
        )
        logger.info("Stage 3b: eis_type=%s via Haiku", fallback_value)
    else:
        record.eis_type = FieldWithStatus(
            value="Unlabelled",
            status="needs_review",
            provenance=Provenance(source="haiku_classifier", section="cover"),
            confidence=0.30,
        )
        warnings.append(
            f"eis_type: Haiku returned unexpected value {fallback_value!r}; defaulted to Unlabelled"
        )
    return warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regex_match(cover_text: str) -> str | None:
    """Return one of {Draft, Final, Supplemental} if regex hits."""
    m = config.EIS_TYPE_REGEX.search(cover_text)
    if not m:
        return None
    raw = m.group(1).strip()
    raw_lower = raw.lower()

    if raw_lower.startswith("supplemental"):
        return "Supplemental"
    if raw_lower.startswith("draft"):
        return "Draft"
    if raw_lower.startswith("final"):
        return "Final"
    return None


def _haiku_fallback(cover_text: str, llm: "LLMClient") -> str | None:
    """Call Haiku 4.5 with the 3b fallback prompt; return the parsed single-word answer."""
    prompt_template = (PROMPTS_DIR / "3b_eis_type_fallback.txt").read_text(encoding="utf-8")
    capped = cover_text[:6000]
    prompt = prompt_template.replace("{cover_text}", capped)

    try:
        raw = llm.call(
            model=llm.models["haiku"],
            system="You are a careful classifier. Return only the single word requested.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
            label="3b_eis_type_fallback",
        )
    except Exception as exc:
        logger.warning("3b Haiku fallback LLM call failed: %s", exc)
        return None

    if getattr(llm, "dry_run", False):
        return "Unlabelled"

    answer = raw.strip().strip("`").strip()
    answer = answer.split()[0] if answer else ""
    answer = answer.strip(".,:;!?\"'")
    return answer if answer in config.VALID_EIS_TYPES else None
