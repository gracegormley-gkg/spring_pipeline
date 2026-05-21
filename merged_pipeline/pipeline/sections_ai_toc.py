"""
Stage 2 fallback — AI-TOC (Haiku 3-sample) section detection.

Runs only when the regex pass missed a section that downstream stages need.
Synthesis_plan.md §Section detection: "AI-TOC pass salvages typewritten
1970s docs where regex fails."

Algorithm (ported from eis_pipeline/pipeline/stage1_chunking.py::_ai_toc()):
  1. Build a beginning + middle + end sample from raw_text (~15k chars total).
  2. Send to Haiku with a TOC-extraction prompt; request title +
     anchor_phrase per section.
  3. For each returned (title, anchor): map title -> CEQ section name via
     pipeline.sections.match_title_to_ceq, then locate the anchor in
     raw_text via verbatim -> case-insensitive -> whitespace-normalized
     regex search.
  4. Emit a DetectedSection with detection_method="ai_toc" for each hit
     whose CEQ name is in `missing_section_names`. Already-detected names
     are not overwritten (regex hits win over AI-TOC for the same name).

Failure modes (all soft, never abort):
  - LLM call fails / times out -> log + return [].
  - LLM returns non-list / no parseable sections -> return [].
  - Title doesn't map to any CEQ name -> drop that section.
  - Anchor phrase not located in raw_text -> drop that section.
"""

from __future__ import annotations

import logging
import re
import textwrap
from typing import TYPE_CHECKING

from . import config
from .ingest import IngestArtifact, page_range_for_span
from .sections import DetectedSection, match_title_to_ceq

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Total chars of doc text sent to Haiku, split across beginning/middle/end.
AI_TOC_SAMPLE_CHARS = 15_000

# Confidence stamped on AI-TOC-derived sections. Lower than regex (0.9) since
# anchor-phrase locating has more failure modes.
AI_TOC_CONFIDENCE = 0.85


_AI_TOC_SYSTEM = textwrap.dedent("""\
    You are identifying the major structural sections of a U.S. Environmental
    Impact Statement (EIS).

    I will give you three excerpts from one EIS document — the beginning, a
    sample from the middle, and the end. Use these excerpts to identify the
    document's natural section structure.

    Typical EIS sections include (but are not limited to):
      - Cover Sheet / Summary / Abstract
      - Purpose and Need / Project Description
      - Description of the Proposed Action
      - Alternatives Considered
      - Affected Environment / Existing Conditions
      - Environmental Consequences / Impact Analysis
      - Mitigation Measures
      - Consultation and Coordination
      - Comments and Responses
      - List of Preparers / References
      - Appendices

    For each major section, return:
      - "title": the section name AS IT APPEARS in the document text (preserve
                 original capitalization if it's a typeset heading).
      - "anchor_phrase": a distinctive 5–15 word phrase from the first sentence
                         or two of that section's body text. We will literally
                         string-search the document for this phrase to locate
                         the section's start, so it MUST be a verbatim substring
                         of the document text I showed you.

    Rules:
      - Skip subsections — only major top-level sections.
      - Skip pure boilerplate like page numbers, dates, repeating headers.
      - If the document has no clear structural sections (it's just a single
        flowing memo or letter with no real divisions), return {"sections": []}.

    Respond with ONLY valid JSON:
    {
      "sections": [
        {"title": "PURPOSE AND NEED", "anchor_phrase": "The proposed project addresses traffic congestion along..."},
        ...
      ]
    }
""")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def ai_toc_detect(
    missing_section_names: list[str],
    ingest: IngestArtifact,
    llm_client: "LLMClient",
) -> list[DetectedSection]:
    """Run one Haiku call; return DetectedSections for any anchor we located
    whose CEQ name is in `missing_section_names`."""
    if not missing_section_names:
        return []

    sample = _sample_for_toc(ingest.raw_text, AI_TOC_SAMPLE_CHARS)

    try:
        result = llm_client.call_json(
            model=llm_client.models["haiku"],
            system=_AI_TOC_SYSTEM,
            messages=[{"role": "user", "content": sample}],
            max_tokens=1024,
            temperature=0.1,
            label=f"ai_toc/{ingest.publication_id}",
        )
    except Exception as exc:
        logger.warning("AI-TOC LLM call failed for %s: %s", ingest.publication_id, exc)
        return []

    raw_sections = result.get("sections") if isinstance(result, dict) else None
    if not isinstance(raw_sections, list):
        logger.info("AI-TOC returned non-list for %s — ignoring", ingest.publication_id)
        return []

    missing_set = set(missing_section_names)
    detected: list[DetectedSection] = []
    seen_ceq_names: set[str] = set()

    for sec in raw_sections:
        if not isinstance(sec, dict):
            continue
        title = (sec.get("title") or "").strip()
        anchor = (sec.get("anchor_phrase") or "").strip()
        if not title or not anchor:
            continue

        ceq_name = match_title_to_ceq(title)
        if ceq_name is None:
            logger.debug("AI-TOC: title %r did not map to a CEQ section", title)
            continue
        if ceq_name not in missing_set:
            # The regex pass already found this section; don't overwrite.
            logger.debug(
                "AI-TOC: %r -> %s, but section already detected by regex; skipping",
                title, ceq_name,
            )
            continue
        if ceq_name in seen_ceq_names:
            # Multiple AI-TOC entries map to the same CEQ name; keep the first.
            continue

        offset = _find_anchor(ingest.raw_text, anchor)
        if offset is None:
            logger.debug("AI-TOC anchor not located in doc: %r", anchor[:60])
            continue

        page_range = page_range_for_span(ingest.pages, (offset, offset + len(anchor)))
        detected.append(DetectedSection(
            name=ceq_name,
            char_span=(offset, offset + len(anchor)),
            pages=page_range,
            confidence=AI_TOC_CONFIDENCE,
            status="ok",
            detection_method="ai_toc",
        ))
        seen_ceq_names.add(ceq_name)
        logger.info(
            "AI-TOC hit: %s (title=%r) @ offset=%d page=%s",
            ceq_name, title[:60], offset, page_range,
        )

    return detected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_for_toc(text: str, total_chars: int) -> str:
    """Build a beginning + middle + end sample, with separators."""
    if len(text) <= total_chars:
        return text

    per_segment = total_chars // 3
    first = text[:per_segment]
    mid_start = max(0, len(text) // 2 - per_segment // 2)
    middle = text[mid_start: mid_start + per_segment]
    last = text[-per_segment:]

    return (
        "=== BEGINNING OF DOCUMENT ===\n"
        f"{first}\n\n"
        "=== MIDDLE OF DOCUMENT (sampled) ===\n"
        f"{middle}\n\n"
        "=== END OF DOCUMENT ===\n"
        f"{last}"
    )


def _find_anchor(text: str, anchor: str) -> int | None:
    """Locate an anchor phrase in `text`. Tries verbatim, case-insensitive,
    then whitespace-normalized regex."""
    idx = text.find(anchor)
    if idx != -1:
        return idx

    lower_idx = text.lower().find(anchor.lower())
    if lower_idx != -1:
        return lower_idx

    # Whitespace-normalized match (OCR line breaks defeat exact match).
    normalized_anchor = re.sub(r"\s+", " ", anchor).strip()
    if not normalized_anchor:
        return None
    tokens = [re.escape(t) for t in normalized_anchor.split()]
    if not tokens:
        return None
    pattern = re.compile(r"\s+".join(tokens), re.IGNORECASE)
    m = pattern.search(text)
    return m.start() if m else None
