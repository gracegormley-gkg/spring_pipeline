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

    # 10 documents not already present under output/, restricted to inventory barcodes:
    python extract_page_text.py --new --limit 10 \
        --from-csv '../eis-inventory-2nd-pass(eis inventory 2nd pass csv).csv'
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from pymongo import MongoClient
from bs4 import BeautifulSoup

# --- config -----------------------------------------------------------------
# Anchored to this file so output layout is stable regardless of the caller's cwd.
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_IMPULSES = ["p0491_35556036091957"]   # used when no barcodes passed on CLI
INCLUDE_HTML = False                 # True -> also store assembled HTML in each file

col = MongoClient(os.environ["MONGO_URI"])["praxis"]["colt"]

REF = re.compile(r"<content-ref\s+src=['\"]([^'\"]+)['\"][^>]*>\s*</content-ref>")
BARCODE = re.compile(r"(\d{10,})\s*$")


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


# --- document selection -----------------------------------------------------
def barcode_of(impulse_id):
    """'p1074_35556036806586' -> '35556036806586' (None if no trailing digit run)."""
    m = BARCODE.search(impulse_id or "")
    return m.group(1) if m else None


def existing_impulses():
    """impulse_identifiers that already have a non-empty folder under OUTPUT_DIR."""
    if not OUTPUT_DIR.is_dir():
        return set()
    return {p.name for p in OUTPUT_DIR.iterdir() if p.is_dir() and any(p.glob("page_*.json"))}


def csv_barcodes(path):
    """Barcodes from column 0 (955$b) of the inventory CSV."""
    out = set()
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if row and row[0].strip():
                out.add(row[0].strip())
    return out


def select_new(limit, csv_path=None):
    """Pick up to `limit` impulse_identifiers absent from OUTPUT_DIR (optionally in CSV)."""
    have = existing_impulses()
    print(f"Already extracted: {len(have)} document(s) under {OUTPUT_DIR}")

    allowed = None
    if csv_path:
        allowed = csv_barcodes(csv_path)
        print(f"Inventory CSV supplies {len(allowed)} unique barcodes")

    chosen, skipped_existing, skipped_csv = [], 0, 0
    for impulse_id in sorted(col.distinct("impulse_identifier")):
        if not impulse_id:
            continue
        if impulse_id in have:
            skipped_existing += 1
            continue
        if allowed is not None and barcode_of(impulse_id) not in allowed:
            skipped_csv += 1
            continue
        chosen.append(impulse_id)
        if len(chosen) >= limit:
            break

    print(f"Skipped {skipped_existing} already-extracted, {skipped_csv} not in CSV")
    print(f"Selected {len(chosen)}: {chosen}")
    return chosen


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
    parser.add_argument(
        "--new",
        action="store_true",
        help="auto-select documents that have no folder under output/ yet",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="max documents to select with --new (default: 10)"
    )
    parser.add_argument(
        "--from-csv",
        metavar="PATH",
        help="with --new, only pick documents whose barcode appears in this inventory CSV",
    )
    args = parser.parse_args()

    if args.new:
        selected = select_new(args.limit, args.from_csv)
        if not selected:
            raise SystemExit("Nothing new to extract.")
        query = {"impulse_identifier": {"$in": selected}}
    elif args.all:
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
