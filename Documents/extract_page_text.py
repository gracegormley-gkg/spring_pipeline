"""
Reassemble Marker block-tree JSON (extracted_data) into readable text, per page,
and save one JSON file per page under a folder per document.

Output layout:
    output/<impulse_identifier>/page_0001.json

Usage:
    pip install pymongo beautifulsoup4
    export MONGO_URI='mongodb+srv://...'
    python extract_page_text.py                              # uses DEFAULT_IMPULSES
    python extract_page_text.py p0491_35556036091957         # one barcode
    python extract_page_text.py bc1 bc2 bc3                  # several barcodes
    python extract_page_text.py --all                        # process every document
"""

import os
import re
import json
import argparse
from pathlib import Path
from pymongo import MongoClient
from bs4 import BeautifulSoup

# --- config -----------------------------------------------------------------
OUTPUT_DIR = Path("output")          # where folders/files get written
DEFAULT_IMPULSES = ["p0491_35556036091957"]   # used when no barcodes passed on CLI
INCLUDE_HTML = False                 # True -> also store assembled HTML in each file

col = MongoClient(os.environ["MONGO_URI"])["praxis"]["colt"]

REF = re.compile(r"<content-ref\s+src=['\"]([^'\"]+)['\"][^>]*>\s*</content-ref>")


def as_blocks(children):
    """Normalize children (list, {'0':block,...} dict, or None) into an ordered list."""
    if not children:
        return []
    if isinstance(children, dict):
        try:
            keys = sorted(children, key=lambda k: int(k))
        except (ValueError, TypeError):
            keys = list(children)
        return [children[k] for k in keys]
    return list(children)


def index_blocks(block, table):
    if not isinstance(block, dict):
        return
    if "id" in block:
        table[block["id"]] = block
    for child in as_blocks(block.get("children")):
        index_blocks(child, table)


def resolve(block, table, seen=None):
    seen = seen or set()
    html = block.get("html") or ""

    def repl(m):
        ref_id = m.group(1)
        child = table.get(ref_id)
        if not child or ref_id in seen:
            return ""
        return resolve(child, table, seen | {ref_id})

    return REF.sub(repl, html)


def page_html(doc):
    table = {}
    children = as_blocks(doc.get("extracted_data", {}).get("children"))
    for top in children:
        index_blocks(top, table)
    pages = [b for b in children if isinstance(b, dict) and b.get("block_type") == "Page"] or children
    return "\n\n".join(resolve(p, table) for p in pages if isinstance(p, dict))


def page_text(doc):
    return BeautifulSoup(page_html(doc), "html.parser").get_text("\n").strip()


# --- write loop -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reassemble page text per impulse_identifier.")
    parser.add_argument(
        "barcodes",
        nargs="*",
        default=DEFAULT_IMPULSES,
        help=f"one or more impulse_identifiers to process (default: {DEFAULT_IMPULSES})",
    )
    parser.add_argument("--all", action="store_true", help="process all documents")
    args = parser.parse_args()

    if args.all:
        query = {}
    elif len(args.barcodes) == 1:
        query = {"impulse_identifier": args.barcodes[0]}
    else:
        query = {"impulse_identifier": {"$in": args.barcodes}}
    print(f"Querying: {query}")
    cursor = col.find(query).sort([("impulse_identifier", 1), ("page_number", 1)])

    count = 0
    for d in cursor:
        impulse_id = d.get("impulse_identifier", "unknown")
        page_no = d.get("page_number", 0)

        folder = OUTPUT_DIR / impulse_id
        folder.mkdir(parents=True, exist_ok=True)

        record = {
            "impulse_identifier": impulse_id,
            "page_number": page_no,
            "filename": d.get("filename"),
            "source_image": d.get("source_image"),
            "extraction_model": d.get("extraction_model"),
            "text": page_text(d),
        }
        if INCLUDE_HTML:
            record["html"] = page_html(d)

        out_path = folder / f"page_{page_no:04d}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        count += 1
        if count % 100 == 0:
            print(f"...wrote {count} pages")

    print(f"Done. Wrote {count} page files under {OUTPUT_DIR.resolve()}")
