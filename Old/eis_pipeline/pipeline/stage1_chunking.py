"""
Stage 1 — Chunking + LLM chunk labeling.

Section discovery, in order of preference:
  1. AI-TOC (Haiku reads beginning/middle/end samples of doc, returns titles +
     anchor phrases; we search the full doc for each anchor to map to pages).
     This is what runs on docs whose structure isn't captured by Stage 0 regex.
  2. Stage 0 regex headings (good for typeset, well-formatted docs).
  3. Fixed FIXED_CHUNK_PAGES-page chunks (fallback for medium/long without
     either).
  4. Short docs (<SHORT_THRESHOLD words): one chunk = whole doc.

Each chunk gets: title, description, topic_tags via Haiku.
Chunks with median_confidence < OCR_EXCLUDE_THRESHOLD are marked used=False.
"""

from __future__ import annotations

import json
import logging
import statistics
import textwrap
from typing import TYPE_CHECKING

from .config import (
    CHUNK_TOPIC_TAGS,
    FIXED_CHUNK_PAGES,
    MODELS,
    OCR_EXCLUDE_THRESHOLD,
    SHORT_THRESHOLD,
)
from .io_layer import Document, PAGE_SEPARATOR, char_offset_to_page
from .schema import ChunkRecord, EISRecord, SectionInfo

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# Approximate max chars to send to the LLM for chunk labeling (~8k tokens * 4 chars/token)
_MAX_LABEL_CHARS = 32_000

# AI-TOC sample size: total chars sent to Haiku, split across beginning/middle/end.
_AI_TOC_SAMPLE_CHARS = 15_000
# Minimum sections AI-TOC must find before we trust its output over Stage 0 regex.
_AI_TOC_MIN_SECTIONS = 3


def run(doc: Document, record: EISRecord, client: "LLMClient") -> list[str]:
    """
    Build chunks from the document and label them. Mutates record.chunks.
    Returns warnings list.
    """
    warnings: list[str] = []

    # AI-TOC pass: try to discover real section structure before chunking.
    # Skipped for short docs (whole doc is one chunk regardless).
    if record.length_category != "short":
        _ai_toc(doc, record, client, warnings)

    raw_chunks = _split_into_chunks(doc, record)
    labeled: list[ChunkRecord] = []

    for i, raw in enumerate(raw_chunks):
        chunk_id = f"c{i + 1:02d}"
        conf = _chunk_confidence(doc, raw["pages"])
        used = conf is None or conf >= OCR_EXCLUDE_THRESHOLD

        if not used:
            logger.info(
                "Chunk %s excluded (confidence=%.3f < %.2f)",
                chunk_id, conf or 0.0, OCR_EXCLUDE_THRESHOLD,
            )

        labeled_chunk = _label_chunk(
            chunk_id=chunk_id,
            text=raw["text"],
            heading=raw.get("heading"),
            pages=raw["pages"],
            median_confidence=conf,
            used=used,
            client=client,
        )
        labeled.append(labeled_chunk)

    record.chunks = labeled
    logger.info("Stage 1 complete: %d chunks (%d used)", len(labeled), sum(c.used for c in labeled))
    return warnings


# ---------------------------------------------------------------------------
# AI-TOC pass (Haiku)
# ---------------------------------------------------------------------------

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


def _ai_toc(
    doc: Document,
    record: EISRecord,
    client: "LLMClient",
    warnings: list[str],
) -> None:
    """
    Use Haiku to identify the document's real section structure, then overwrite
    record.sections if the AI pass produced a stronger result than Stage 0 regex.

    Mutates record.sections and record.has_headings on success. Leaves them
    untouched on failure or empty result.
    """
    sample = _sample_for_toc(doc, _AI_TOC_SAMPLE_CHARS)

    try:
        result = client.call_json(
            model=MODELS["light"],
            system=_AI_TOC_SYSTEM,
            messages=[{"role": "user", "content": sample}],
            max_tokens=1024,
            temperature=0.1,
            label=f"ai_toc/{record.doc_id}",
        )
    except Exception as exc:
        logger.warning("AI-TOC failed for %s: %s — falling back to regex sections", record.doc_id, exc)
        warnings.append(f"ai_toc_error: {exc}")
        return

    raw_sections = result.get("sections") or []
    if not isinstance(raw_sections, list):
        logger.info("AI-TOC returned non-list for %s — ignoring", record.doc_id)
        return

    located: list[tuple[str, int]] = []  # (title, char_offset)
    for sec in raw_sections:
        if not isinstance(sec, dict):
            continue
        title = (sec.get("title") or "").strip()
        anchor = (sec.get("anchor_phrase") or "").strip()
        if not title or not anchor:
            continue
        offset = _find_anchor(doc.full_text, anchor)
        if offset is None:
            logger.debug("AI-TOC anchor not located in doc: %r", anchor)
            continue
        located.append((title, offset))

    # Sort by document order and dedupe near-duplicate positions
    located.sort(key=lambda t: t[1])
    deduped: list[tuple[str, int]] = []
    last_offset = -1
    for title, offset in located:
        if offset - last_offset < 200:
            continue  # same heading region — skip the dupe
        deduped.append((title, offset))
        last_offset = offset

    if len(deduped) < _AI_TOC_MIN_SECTIONS:
        logger.info(
            "AI-TOC found only %d locatable sections for %s — keeping Stage 0 regex sections",
            len(deduped), record.doc_id,
        )
        return

    # Build SectionInfo list with page ranges
    sections: list[SectionInfo] = []
    for i, (title, offset) in enumerate(deduped):
        start_page = char_offset_to_page(doc, offset) or 1
        if i + 1 < len(deduped):
            next_offset = deduped[i + 1][1]
            end_page = max(
                start_page,
                (char_offset_to_page(doc, next_offset) or start_page) - 1,
            )
        else:
            end_page = len(doc.pages)
        sections.append(SectionInfo(
            title=title[:120],
            start_page=start_page,
            end_page=end_page,
        ))

    logger.info(
        "AI-TOC: replaced %d regex sections with %d AI sections for %s",
        len(record.sections), len(sections), record.doc_id,
    )
    record.sections = sections
    record.has_headings = True


def _sample_for_toc(doc: Document, total_chars: int) -> str:
    """
    Build a beginning + middle + end sample of the doc, with separators so the
    LLM understands the structure of what it's seeing.
    """
    text = doc.full_text
    if len(text) <= total_chars:
        return text

    per_segment = total_chars // 3
    first = text[:per_segment]
    mid_start = max(0, len(text) // 2 - per_segment // 2)
    middle = text[mid_start : mid_start + per_segment]
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
    """
    Locate an anchor phrase in the document text. Tries:
      1. Exact substring match.
      2. Case-insensitive substring match.
      3. Whitespace-normalized match (collapses runs of whitespace including
         OCR line breaks).
    Returns the offset of the match in `text`, or None if not found.
    """
    idx = text.find(anchor)
    if idx != -1:
        return idx

    lower_idx = text.lower().find(anchor.lower())
    if lower_idx != -1:
        return lower_idx

    # Whitespace-normalized match (OCR line breaks defeat exact match)
    import re as _re
    normalized_anchor = _re.sub(r"\s+", " ", anchor).strip()
    if not normalized_anchor:
        return None
    # Find normalized_anchor by scanning normalized text and mapping back
    # to original offsets. Cheaper: use a regex with \s+ between tokens.
    tokens = [_re.escape(t) for t in normalized_anchor.split()]
    if not tokens:
        return None
    pattern = _re.compile(r"\s+".join(tokens), _re.IGNORECASE)
    m = pattern.search(text)
    return m.start() if m else None


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def _split_into_chunks(doc: Document, record: EISRecord) -> list[dict]:
    """Returns list of {text, pages, heading?} dicts."""

    # Short docs: one chunk
    if record.length_category == "short":
        return [{
            "text": doc.full_text,
            "pages": [p.page_num for p in doc.pages],
            "heading": record.title or "Full Document",
        }]

    # Medium/long with headings: one chunk per section
    if record.has_headings and record.sections:
        chunks = []
        for section in record.sections:
            pages_in_section = [
                p for p in doc.pages
                if section.start_page <= p.page_num <= section.end_page
            ]
            text = PAGE_SEPARATOR.join(p.text for p in pages_in_section)
            page_nums = [p.page_num for p in pages_in_section]
            chunks.append({
                "text": text,
                "pages": page_nums,
                "heading": section.title,
            })
        if chunks:
            return chunks

    # Medium/long without headings: fixed 30-page chunks
    return _fixed_page_chunks(doc, FIXED_CHUNK_PAGES)


def _fixed_page_chunks(doc: Document, pages_per_chunk: int) -> list[dict]:
    chunks = []
    pages = doc.pages
    for start_idx in range(0, len(pages), pages_per_chunk):
        page_slice = pages[start_idx : start_idx + pages_per_chunk]
        text = PAGE_SEPARATOR.join(p.text for p in page_slice)
        page_nums = [p.page_num for p in page_slice]
        chunks.append({
            "text": text,
            "pages": page_nums,
        })
    return chunks


# ---------------------------------------------------------------------------
# Confidence per chunk
# ---------------------------------------------------------------------------

def _chunk_confidence(doc: Document, page_nums: list[int]) -> float | None:
    confs = [
        p.median_confidence
        for p in doc.pages
        if p.page_num in set(page_nums) and p.median_confidence is not None
    ]
    if not confs:
        return None
    return statistics.median(confs)


# ---------------------------------------------------------------------------
# LLM chunk labeling
# ---------------------------------------------------------------------------

_SYSTEM = textwrap.dedent("""\
    You are an analyst reviewing excerpts from U.S. Environmental Impact Statements (EIS).
    For each excerpt, output ONLY valid JSON with these keys:
    - "title": a 1-line descriptive title (use the provided heading if given, else generate one)
    - "description": 2-3 factual sentences describing what this excerpt covers
    - "topic_tags": a list of 1-4 tags chosen ONLY from this fixed vocabulary:
      {tags}
    Do not include any text outside the JSON object.
""")

_USER_TMPL = textwrap.dedent("""\
    Heading (if any): {heading}

    Excerpt (pages {pages}):
    {text}
""")


def _label_chunk(
    chunk_id: str,
    text: str,
    heading: str | None,
    pages: list[int],
    median_confidence: float | None,
    used: bool,
    client: "LLMClient",
) -> ChunkRecord:
    system = _SYSTEM.format(tags=json.dumps(CHUNK_TOPIC_TAGS))
    user_text = _USER_TMPL.format(
        heading=heading or "(none)",
        pages=f"{pages[0]}–{pages[-1]}" if pages else "?",
        text=text[:_MAX_LABEL_CHARS],
    )

    try:
        result = client.call_json(
            model=MODELS["light"],
            system=system,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=512,
            temperature=0.1,
            label=f"chunk_label/{chunk_id}",
        )
        title = result.get("title") or heading or chunk_id
        description = result.get("description") or ""
        raw_tags = result.get("topic_tags") or []
        topic_tags = [t for t in raw_tags if t in CHUNK_TOPIC_TAGS]
    except Exception as exc:
        logger.warning("Chunk labeling failed for %s: %s", chunk_id, exc)
        title = heading or chunk_id
        description = ""
        topic_tags = []

    return ChunkRecord(
        chunk_id=chunk_id,
        title=title,
        description=description,
        topic_tags=topic_tags,
        pages=pages,
        text=text,
        median_confidence=round(median_confidence, 4) if median_confidence is not None else None,
        used=used,
    )
