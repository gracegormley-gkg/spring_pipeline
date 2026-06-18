"""
Chunking on real page boundaries.

Source is per-page JSON (see pages.py). Each Chunk carries the actual
start_page / end_page from the loader — no char/2500 estimation.

We also expose a CEQ-chapter mapping: if we can detect chapter headings in
the OCR, we surface them as section labels mapped to canonical CEQ chapters.
That lets M2 hand the right chunk to the right extractor (e.g. Alternatives
gets the Alternatives chapter rather than a regex flag on the word).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from config import (
    CHAPTER_ALIASES,
    CHUNK_PAGES,
    CHUNK_OVERLAP_PAGES,
)
from pages import Doc


@dataclass
class Chunk:
    """A slice of doc text aligned to real page boundaries."""
    index: int
    start_page: int
    end_page: int
    text: str
    label: Optional[str] = None       # section name if section-mapped
    ceq_chapter: Optional[str] = None # canonical CEQ chapter if mapped

    @property
    def page_span(self) -> str:
        return f"{self.start_page}-{self.end_page}"


def first_pages(doc: Doc, n_pages: int) -> str:
    if not doc.pages:
        return ""
    end = min(doc.pages[-1].page_num, doc.pages[0].page_num + n_pages - 1)
    return doc.text_for_pages(doc.pages[0].page_num, end)


def last_pages(doc: Doc, n_pages: int) -> str:
    if not doc.pages:
        return ""
    start = max(doc.pages[0].page_num, doc.pages[-1].page_num - n_pages + 1)
    return doc.text_for_pages(start, doc.pages[-1].page_num)


def fixed_chunks(doc: Doc) -> list[Chunk]:
    """50-page chunks with 2-page overlap, aligned to real pages."""
    if not doc.pages:
        return []
    chunks: list[Chunk] = []
    page_nums = [p.page_num for p in doc.pages]
    first, last = page_nums[0], page_nums[-1]
    stride = max(1, CHUNK_PAGES - CHUNK_OVERLAP_PAGES)
    idx = 0
    start = first
    while start <= last:
        end = min(start + CHUNK_PAGES - 1, last)
        text = doc.text_for_pages(start, end)
        chunks.append(Chunk(
            index=idx,
            start_page=start,
            end_page=end,
            text=text,
        ))
        idx += 1
        if end >= last:
            break
        start += stride
    return chunks


# --- Regex-based chapter detection -------------------------------------------
#
# Operates on doc.full_text (char offsets). We convert each detected heading's
# offset to a real page number via doc.page_at_offset.

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


def detect_chapters(doc: Doc) -> list[dict]:
    """
    Detect chapter headings in doc.full_text, mapped to canonical CEQ chapters.

    Each CEQ chapter is represented at most once — we pick the heading whose
    span (this heading → next heading) is LONGEST. Page numbers are real.
    """
    text = doc.full_text
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
        h["start_page"] = doc.page_at_offset(h["start_char"])
        h["end_page"] = doc.page_at_offset(max(h["start_char"], h["end_char"] - 1))
    return final


def _map_to_ceq(heading: str) -> Optional[str]:
    """
    Map a heading string (possibly with leading marker) to a canonical CEQ chapter.
    The cleaned heading must START with a known alias.
    """
    h = heading.strip().lower()
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


def chunks_for_doc(doc: Doc) -> dict:
    """
    Decide chunking strategy for a doc and return both:
      - 'chapters': list of detected CEQ-chapter spans (may be empty), each with
                    real start_page / end_page
      - 'chunks':  full-doc 50-page chunks (real pages) with chapter labels
                   attached when the chunk's page midpoint falls inside a chapter
    """
    chapters = detect_chapters(doc)
    chunks = fixed_chunks(doc)

    # Stamp chapter label on any chunk whose midpoint page lies inside a chapter
    for c in chunks:
        mid_page = (c.start_page + c.end_page) // 2
        for ch in chapters:
            if ch["start_page"] <= mid_page <= ch["end_page"]:
                c.label = ch["label"]
                c.ceq_chapter = ch["ceq_chapter"]
                break

    return {
        "chapters": chapters,
        "chunks": chunks,
        "total_pages": doc.n_pages,
        "total_chars": len(doc.full_text),
    }


def text_for_ceq_chapter(doc: Doc, chapters: list[dict], ceq: str) -> Optional[tuple[str, int, int]]:
    """Return (text, start_page, end_page) for the first detected chapter with given CEQ label."""
    for ch in chapters:
        if ch["ceq_chapter"] == ceq:
            seg = doc.full_text[ch["start_char"]:ch["end_char"]]
            return seg, ch["start_page"], ch["end_page"]
    return None
