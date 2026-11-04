"""Pure-regex/heuristic TOC extractor (no LLM, free, deterministic).

Usage:
    python extract_toc_regex.py <doc_folder> [--out FILE]

Example:
    python extract_toc_regex.py ../Documents/output/p1074_35556036806586

Input:  folder of page_NNNN.json files (output of extract_page_text.py).
Output: <doc_id>_toc_regex.json with nested {number, title, page, children}.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from _common import (
    PAGE_NUM_RE,
    load_pages,
    detect_toc_pages,
    doc_id_from_folder,
    trim_to_toc,
)

SECTION_NUM_RE = re.compile(
    r"^("
    r"\d+(?:\.\d+)*\.?"          # 1, 1.4, 1.4.1, 1.
    r"|[IVXLCDM]{1,5}\.?"        # I, II, IV.
    r"|[ivxlcdm]{1,5}\.?"        # i, ii.
    r"|[A-Z]\.?"                 # A, B.
    r"|\([a-zA-Z0-9]+\)"         # (a), (1)
    r")$"
)

NOISE_RE = re.compile(r"^(section|page(\s+no\.?)?|item|chapter)$", re.IGNORECASE)
HEADER_RE = re.compile(r"^(table of contents|contents)(\s*\(continued\))?$", re.IGNORECASE)


def classify(line: str):
    """Return 'page', 'section_num', 'title', 'noise', or 'skip'."""
    s = line.strip()
    if not s:
        return "skip"
    if NOISE_RE.match(s):
        return "noise"
    if PAGE_NUM_RE.match(s):
        # Single short roman numerals could be section labels, not pages.
        if re.fullmatch(r"[IVXLCDM]{1,3}", s):
            return "section_num"
        return "page"
    if SECTION_NUM_RE.match(s):
        return "section_num"
    return "title"


def depth_of(num: str) -> int:
    """Approximate hierarchy depth from a section number string."""
    if not num:
        return 0
    s = num.rstrip(".")
    if re.fullmatch(r"\d+(?:\.\d+)+", s):
        return s.count(".")
    if re.fullmatch(r"\d+", s):
        return 0
    if re.fullmatch(r"[IVXLCDM]+", s):
        return 0
    if re.fullmatch(r"[A-Z]", s):
        return 1
    if re.fullmatch(r"[ivxlcdm]+", s):
        return 2
    return 3


def extract_entries(toc_text: str):
    """Walk lines, produce flat list of (number, title, page) tuples.

    Handles two layouts:
      - inline: title and page on adjacent lines
      - column: a block of titles followed by a block of page numbers
        (pages get paired with titles in order)
    """
    tokens = []
    for ln in toc_text.splitlines():
        s = ln.strip()
        if not s or HEADER_RE.match(s):
            continue
        kind = classify(s)
        if kind == "noise" or kind == "skip":
            continue
        tokens.append((kind, s))

    entries = []
    title_buf = []         # list of (section_num or None, title) waiting for a page
    cur_section = None
    pending_pages = []

    def flush():
        while pending_pages and title_buf:
            num, tit = title_buf.pop(0)
            pg = pending_pages.pop(0)
            entries.append((num or "", tit, pg))

    for kind, v in tokens:
        if kind == "section_num":
            cur_section = v.rstrip(".")
        elif kind == "title":
            title_buf.append((cur_section, v))
            cur_section = None
        elif kind == "page":
            pending_pages.append(v)
            flush()

    for num, tit in title_buf:
        entries.append((num or "", tit, ""))

    return entries


def nest_entries(flat):
    """Convert flat entries into a nested tree based on section-number depth."""
    root = []
    stack = [(-1, root)]
    for num, title, page in flat:
        d = depth_of(num)
        while stack and stack[-1][0] >= d:
            stack.pop()
        node = {"number": num, "title": title, "page": page, "children": []}
        stack[-1][1].append(node)
        stack.append((d, node["children"]))
    return root


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc_folder", type=Path, help="folder of page_NNNN.json files")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: ./<doc_id>_toc_regex.json)")
    args = ap.parse_args()

    folder = args.doc_folder.resolve()
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    pages = load_pages(folder)
    if not pages:
        print(f"No page files in {folder}", file=sys.stderr)
        sys.exit(1)

    toc_idxs = detect_toc_pages(pages)
    doc_id = doc_id_from_folder(folder)

    result = {
        "doc_id": doc_id,
        "method": "regex",
        "toc_source_pages": [pages[i]["page_number"] for i in toc_idxs],
        "toc": [],
    }

    if not toc_idxs:
        result["note"] = "no TOC detected in first 20 pages"
    else:
        # Trim title-page text on the first TOC page so it doesn't pollute entries.
        page_texts = [pages[i]["text"] for i in toc_idxs]
        page_texts[0] = trim_to_toc(page_texts[0])
        toc_text = "\n".join(page_texts)
        flat = extract_entries(toc_text)
        result["toc"] = nest_entries(flat)

    out = args.out or Path.cwd() / f"{doc_id}_toc_regex.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
