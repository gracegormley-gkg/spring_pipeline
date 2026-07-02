"""Shared helpers for TOC extraction pipelines.

Both extract_toc_regex.py and extract_toc_llm.py use these utilities to load
per-page JSON files and detect which pages contain the document's TOC.
"""

import json
import re
from pathlib import Path

PAGE_NUM_RE = re.compile(
    r"^("
    r"\d+(?:-\d+)?"           # 1, 23-145
    r"|[A-Za-z]-\d+"          # S-1, A-12, P-3
    r"|[IVXLCDM]{1,5}"        # uppercase roman (short)
    r"|[ivxlcdm]{1,5}"        # lowercase roman (short)
    r")$"
)

TOC_HEADER_RE = re.compile(r"(?im)^(table of contents|contents)\b")
TOC_CONT_RE = re.compile(r"(?i)\(continued\)")


def load_pages(folder: Path):
    """Read all page_NNNN.json files in a folder, sorted by page number."""
    pages = []
    for p in sorted(folder.glob("page_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        pages.append({
            "page_number": d.get("page_number"),
            "text": d.get("text", ""),
        })
    return pages


CONTENTS_WORD_RE = re.compile(r"(?i)\bcontents\b")


def trim_to_toc(text: str) -> str:
    """Drop everything before the first 'Contents' / 'Table of Contents' line.

    The first TOC page often has title-page text above the header (document title,
    subtitle, agency name). Including it makes the regex pipeline misclassify
    those lines as TOC entries.
    """
    m = TOC_HEADER_RE.search(text)
    if not m:
        return text
    return text[m.start():]


def detect_toc_pages(pages, max_search=20):
    """Return indices (0-based) of pages that make up the TOC.

    Start: first page with 'Table of Contents' or 'Contents' header in first 300 chars.
    Continue: subsequent pages whose first 200 chars contain 'Contents' (catches
    'CONTENTS (Continued)' / 'TABLE OF CONTENTS (Continued)' but excludes adjacent
    list-of-figures / list-of-tables pages, which have their own header.
    """
    start = None
    for i, p in enumerate(pages[:max_search]):
        head = p["text"][:300]
        if TOC_HEADER_RE.search(head):
            start = i
            break
    if start is None:
        return []

    idxs = [start]
    for j in range(start + 1, min(len(pages), max_search + 10)):
        head = pages[j]["text"][:200]
        if CONTENTS_WORD_RE.search(head):
            idxs.append(j)
        else:
            break
    return idxs


def doc_id_from_folder(folder: Path) -> str:
    return folder.name
