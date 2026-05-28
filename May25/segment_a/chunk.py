"""
Chunking for flat OCR strings.

Per the v2 plan:
  1. PDF outline/bookmarks if present — NOT AVAILABLE (flat OCR, no PDFs).
  2. Else regex TOC + LLM to label section ranges.
  3. Else 50-page chunks with 2-page overlap (~CHARS_PER_PAGE chars/page).

We also expose a CEQ-chapter mapping: if we can detect chapter headings in
the OCR, we surface them as section labels mapped to canonical CEQ chapters.
That lets M2 hand the right chunk to the right extractor (e.g. Alternatives
gets the Alternatives chapter rather than a regex flag on the word).

`source_pages` everywhere = ESTIMATED page numbers from char offsets at
config.CHARS_PER_PAGE. Surface this caveat in the grading sheet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from config import (
    CHARS_PER_PAGE,
    CHAPTER_ALIASES,
    CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
)


@dataclass
class Chunk:
    """A slice of OCR text with estimated page span."""
    index: int
    start_char: int
    end_char: int
    text: str
    label: Optional[str] = None       # section name if section-mapped
    ceq_chapter: Optional[str] = None # canonical CEQ chapter if mapped

    @property
    def start_page(self) -> int:
        return char_to_page(self.start_char)

    @property
    def end_page(self) -> int:
        return char_to_page(self.end_char)

    @property
    def page_span(self) -> str:
        return f"{self.start_page}-{self.end_page}"


def char_to_page(offset: int) -> int:
    """Estimated 1-indexed page number for a char offset."""
    return max(1, (offset // CHARS_PER_PAGE) + 1)


def page_range_chars(start_page: int, end_page: int, text_len: int) -> tuple[int, int]:
    """Inverse of char_to_page: page span → char range. Clamped to valid bounds."""
    start = max(0, (start_page - 1) * CHARS_PER_PAGE)
    end = min(text_len, end_page * CHARS_PER_PAGE)
    if start >= text_len:
        return text_len, text_len
    if end < start:
        end = start
    return start, end


def estimated_pages(text: str) -> int:
    """Estimated total page count for a doc."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_PAGE + (1 if len(text) % CHARS_PER_PAGE else 0))


def first_pages(text: str, n_pages: int) -> str:
    return text[: n_pages * CHARS_PER_PAGE]


def last_pages(text: str, n_pages: int) -> str:
    return text[-n_pages * CHARS_PER_PAGE :]


def fixed_chunks(text: str) -> list[Chunk]:
    """Fallback: 50-page chunks with 2-page overlap."""
    if not text:
        return []
    chunks: list[Chunk] = []
    stride = CHUNK_CHARS - CHUNK_OVERLAP_CHARS
    i = 0
    idx = 0
    n = len(text)
    while i < n:
        end = min(i + CHUNK_CHARS, n)
        chunks.append(Chunk(
            index=idx,
            start_char=i,
            end_char=end,
            text=text[i:end],
        ))
        idx += 1
        if end >= n:
            break
        i += stride
    return chunks


# --- Regex-based chapter detection -------------------------------------------
#
# Heuristic: scan each newline-bounded line for ones that look like a CEQ
# chapter heading. A heading line:
#   - is short (<= 100 chars)
#   - is mostly uppercase OR has a clear marker prefix ("H.", "II.", "2.0",
#     "CHAPTER 3", "Section 4")
#   - does not end with sentence-flow punctuation
#   - after stripping any marker, starts with a CEQ alias
#
# We then collapse the matches per CEQ type to a single span (the longest one)
# and skip headings that appear in the first ~3000 chars (likely the TOC, not
# the real chapter break).

_LINE_RE = re.compile(r"(?:^|\n)([^\n]{1,100})\n")
_MARKER_PREFIX_RE = re.compile(
    r"^(?:\s*(?:CHAPTER|Chapter|SECTION|Section)\s+[\w\.]+|"
    r"\s*[A-Z]\.|\s*[IVXLC]+\.|\s*\d+(?:\.\d+)?\s*[:\.\-–])\s*",
)

_TOC_GUARD_CHARS = 3000  # ignore matches before this offset (likely TOC)


def _is_uppercase_dominant(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) >= 0.7


def _looks_like_heading_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 100:
        return False
    if s.endswith((".", ",", ";", ":")) and not _MARKER_PREFIX_RE.match(s):
        return False
    if _MARKER_PREFIX_RE.match(s):
        return True
    return _is_uppercase_dominant(s)


def detect_chapters(text: str) -> list[dict]:
    """
    Detect chapter headings in flat OCR, mapped to canonical CEQ chapters.

    Each CEQ chapter is represented at most once — we pick the heading whose
    span (this heading → next heading) is LONGEST.
    """
    raw: list[dict] = []
    for m in _LINE_RE.finditer(text):
        line_start = m.start(1)
        if line_start < _TOC_GUARD_CHARS:
            continue
        line = m.group(1)
        if not _looks_like_heading_line(line):
            continue
        canonical = _map_to_ceq(line)
        if not canonical:
            continue
        raw.append({
            "label": line.strip(),
            "ceq_chapter": canonical,
            "start_char": line_start,
        })

    if not raw:
        return []

    raw.sort(key=lambda h: h["start_char"])
    for i, h in enumerate(raw):
        h["end_char"] = raw[i + 1]["start_char"] if i + 1 < len(raw) else len(text)
        h["span"] = h["end_char"] - h["start_char"]

    by_ceq: dict[str, dict] = {}
    for h in raw:
        cur = by_ceq.get(h["ceq_chapter"])
        if cur is None or h["span"] > cur["span"]:
            by_ceq[h["ceq_chapter"]] = h

    final = sorted(by_ceq.values(), key=lambda h: h["start_char"])
    for h in final:
        h.pop("span", None)
    return final


def _map_to_ceq(heading: str) -> Optional[str]:
    """
    Map a heading string (possibly with leading marker) to a canonical CEQ chapter.
    The cleaned heading must START with a known alias.
    """
    h = heading.strip().lower()
    # Strip leading markers
    h = re.sub(r"^(chapter|section)\s+[\w\.]+\s*[:\.\-–]?\s*", "", h)
    h = re.sub(r"^[ivxlc]+\.\s*", "", h)
    h = re.sub(r"^[a-z]\.\s*", "", h)
    h = re.sub(r"^\d+(\.\d+)?\s*[:\.\-–]?\s*", "", h)
    h = h.strip()
    for canonical, aliases in CHAPTER_ALIASES.items():
        for alias in aliases:
            if h == alias or h.startswith(alias + " ") or h.startswith(alias + ":") or h.startswith(alias + " to ") or h.startswith(alias + " of "):
                return canonical
    return None


def chunks_for_doc(text: str) -> dict:
    """
    Decide chunking strategy for a doc and return both:
      - 'chapters': list of detected CEQ-chapter spans (may be empty)
      - 'chunks':  full-doc 50-page chunks with chapter labels attached when
                   a chunk falls inside a detected chapter

    The result is consumed by m2 to route each extractor to the right text.
    """
    chapters = detect_chapters(text)
    chunks = fixed_chunks(text)

    # Stamp chapter label on any chunk whose midpoint lies inside a chapter span
    for c in chunks:
        mid = (c.start_char + c.end_char) // 2
        for ch in chapters:
            if ch["start_char"] <= mid < ch["end_char"]:
                c.label = ch["label"]
                c.ceq_chapter = ch["ceq_chapter"]
                break

    return {
        "chapters": chapters,
        "chunks": chunks,
        "total_chars": len(text),
        "estimated_pages": estimated_pages(text),
    }


def text_for_ceq_chapter(text: str, chapters: list[dict], ceq: str) -> Optional[tuple[str, int, int]]:
    """Return (text, start_page, end_page) for the first detected chapter with given CEQ label."""
    for ch in chapters:
        if ch["ceq_chapter"] == ceq:
            seg = text[ch["start_char"]:ch["end_char"]]
            return seg, char_to_page(ch["start_char"]), char_to_page(ch["end_char"])
    return None
