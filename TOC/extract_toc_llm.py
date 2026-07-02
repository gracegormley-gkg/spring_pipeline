"""LLM-based TOC extractor using Claude Haiku (cheap, robust across formats).

Usage:
    export ANTHROPIC_API_KEY=...
    pip install anthropic
    python extract_toc_llm.py <doc_folder> [--out FILE] [--model MODEL]

Example:
    python extract_toc_llm.py ../Documents/output/p1074_35556036806586

Input:  folder of page_NNNN.json files (output of extract_page_text.py).
Output: <doc_id>_toc_llm.json with nested {number, title, page, children}.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from _common import load_pages, detect_toc_pages, doc_id_from_folder, trim_to_toc

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

PROMPT = """You are given raw text extracted from the Table of Contents pages of a document.
Convert it into structured JSON.

Output rules:
- Return ONLY a JSON array, no prose, no markdown fences.
- Each entry is an object with these fields:
  * "number": section/chapter label (e.g. "1", "1.4.1", "I", "A", "(a)") or "" if none.
  * "title": the heading text, cleaned of obvious OCR artifacts (e.g. "DESC | RIPTION" -> "DESCRIPTION").
  * "page": page number string exactly as it appears in the TOC (e.g. "1", "S-3", "iii", "1-9") or "" if none.
  * "children": array of nested entries with the same shape. Use [] if none.
- Infer hierarchy from numbering: "1.4.1" is a child of "1.4", which is a child of "1".
  Letters (A, B) are children of Roman numerals (I, II) when both appear.
- If section numbers and page numbers appear in separate columns/blocks (common after OCR),
  pair them by order, not by adjacency.
- Skip running headers like "Section", "Page", "Item", "Chapter", and "(Continued)" markers.

TOC text:
---
{toc_text}
---

JSON array:"""


def call_llm(toc_text: str, model: str) -> list:
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Install dependency: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": PROMPT.format(toc_text=toc_text)}],
    )
    raw = resp.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("doc_folder", type=Path, help="folder of page_NNNN.json files")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: ./<doc_id>_toc_llm.json)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Anthropic model (default: {DEFAULT_MODEL})")
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
        "method": "llm",
        "model": args.model,
        "toc_source_pages": [pages[i]["page_number"] for i in toc_idxs],
        "toc": [],
    }

    if not toc_idxs:
        result["note"] = "no TOC detected in first 20 pages"
    else:
        page_texts = [pages[i]["text"] for i in toc_idxs]
        page_texts[0] = trim_to_toc(page_texts[0])
        toc_text = "\n".join(page_texts)
        result["toc"] = call_llm(toc_text, args.model)

    out = args.out or Path.cwd() / f"{doc_id}_toc_llm.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
