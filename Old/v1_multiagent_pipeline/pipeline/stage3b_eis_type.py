"""
Stage 3b — EIS Type detection.

Two-pass:
  1. Regex on the cover section text (free).
  2. Haiku 4.5 fallback if regex finds nothing.

Output is one of {Draft, Final, Supplemental, ROD, NOI, Unlabelled}.
For the v1 corpus filter (Final EIS only) anything other than 'Final' is a
manifest filter bug.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import config
from .ingest import IngestArtifact
from .llm_client import LLMClient
from .schema import (
    EISRecord,
    FieldWithStatus,
    Provenance,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def run(
    record: EISRecord,
    ingest: IngestArtifact,
    sections_artifact: Any,
    llm: LLMClient,
) -> list[str]:
    """
    Populate record.eis_type. Mutates record in place. Returns warnings.
    Requires sections_artifact to find the cover section's char span.
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

    # 'Supplemental Final' / 'Supplemental Draft' both → Supplemental
    if raw_lower.startswith("supplemental"):
        return "Supplemental"
    if raw_lower.startswith("draft"):
        return "Draft"
    if raw_lower.startswith("final"):
        return "Final"
    return None


def _haiku_fallback(cover_text: str, llm: LLMClient) -> str | None:
    """Call Haiku 4.5 with the 3b fallback prompt; return the parsed single-word answer."""
    prompt_template = (PROMPTS_DIR / "3b_eis_type_fallback.txt").read_text(encoding="utf-8")
    # Cap cover text at ~6000 chars so we don't accidentally feed an entire short doc.
    capped = cover_text[:6000]
    prompt = prompt_template.replace("{cover_text}", capped)

    raw = llm.call(
        model=config.MODELS["haiku"],
        system="You are a careful classifier. Return only the single word requested.",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8,
        temperature=0.0,
        label="3b_eis_type_fallback",
    )

    if llm.dry_run:
        return "Unlabelled"

    # Strip whitespace, punctuation, code fences
    answer = raw.strip().strip("`").strip()
    # Take just the first word
    answer = answer.split()[0] if answer else ""
    answer = answer.strip(".,:;!?\"'")
    return answer if answer in config.VALID_EIS_TYPES else None
