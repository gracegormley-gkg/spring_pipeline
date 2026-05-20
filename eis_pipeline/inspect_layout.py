#!/usr/bin/env python3
"""
Utility: inspect the real layout of a document folder before hardcoding parsing rules.

Run this FIRST on a real doc from the S3 bucket to understand the actual schemas.

Usage:
    python inspect_layout.py --doc-dir /path/to/P0491_35556036063543
    python inspect_layout.py --doc-dir /path/to/P0491_35556036063543 --full-json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect EIS document folder layout and schemas")
    p.add_argument("--doc-dir", required=True)
    p.add_argument(
        "--full-json",
        action="store_true",
        help="Print full JSON content of sample files (default: truncated)",
    )
    args = p.parse_args()

    doc_dir = Path(args.doc_dir)
    if not doc_dir.is_dir():
        print(f"ERROR: Not a directory: {doc_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"DOCUMENT DIRECTORY: {doc_dir}")
    print(f"{'='*60}\n")

    # Top-level contents
    top_level = sorted(doc_dir.iterdir())
    print("TOP-LEVEL CONTENTS:")
    for item in top_level:
        kind = "DIR" if item.is_dir() else "FILE"
        size = item.stat().st_size if item.is_file() else ""
        print(f"  [{kind}]  {item.name}  {size}")
    print()

    # METS files
    for mets_file in ["mets.xml", "mets.yaml", "mets.yml"]:
        path = doc_dir / mets_file
        if path.exists():
            _inspect_file(path, args.full_json, max_chars=3000)

    # TXT directory
    txt_dir = doc_dir / "TXT"
    if txt_dir.exists():
        txt_files = sorted(txt_dir.glob("*.txt"), key=lambda p: p.name)
        json_files = sorted(txt_dir.glob("*.json"), key=lambda p: p.name)
        print(f"TXT DIRECTORY: {len(txt_files)} .txt files, {len(json_files)} .json files")
        if txt_files:
            print(f"  First .txt: {txt_files[0].name}")
            print(f"  Last  .txt: {txt_files[-1].name}")
            _inspect_file(txt_files[0], args.full_json, max_chars=500, label="First page text")
        if json_files:
            print(f"\n  First per-page JSON: {json_files[0].name}")
            _inspect_json_schema(json_files[0], args.full_json)
        print()

    # CONFIDENCES directory
    conf_dir = doc_dir / "CONFIDENCES"
    if conf_dir.exists():
        conf_files = sorted(conf_dir.glob("*.json"))
        print(f"CONFIDENCES DIRECTORY: {len(conf_files)} .json files")
        if conf_files:
            print(f"  First: {conf_files[0].name}")
            _inspect_json_schema(conf_files[0], args.full_json)
        print()
    else:
        print("CONFIDENCES DIRECTORY: NOT FOUND\n")

    # Summary stats
    print("SUMMARY:")
    page_count = len(list((doc_dir / "TXT").glob("*.txt"))) if (doc_dir / "TXT").exists() else 0
    print(f"  Estimated page count: {page_count}")
    print()
    print("NOTE: Review the schemas above before running the pipeline.")
    print("If the shapes differ from what io_layer.py expects, update _load_confidence()")
    print("and/or _parse_mets_xml()/_parse_mets_yaml() in pipeline/io_layer.py.")
    print()


def _inspect_file(path: Path, full: bool, max_chars: int = 2000, label: str = "") -> None:
    label = label or path.name
    print(f"\n--- {label} ---")
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        if full or len(content) <= max_chars:
            print(content)
        else:
            print(content[:max_chars])
            print(f"\n  [... truncated — {len(content)} total chars. Use --full-json to see all ...]")
    except Exception as exc:
        print(f"  ERROR reading: {exc}")


def _inspect_json_schema(path: Path, full: bool) -> None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"  ERROR parsing JSON: {exc}")
        return

    print(f"  JSON type: {type(data).__name__}")

    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, list):
                print(f"    {k!r}: list of {len(v)} items")
                if v and isinstance(v[0], dict):
                    print(f"      First item keys: {list(v[0].keys())}")
                    if full:
                        print(f"      First item: {json.dumps(v[0], indent=6)}")
            elif isinstance(v, (int, float, str, bool)):
                print(f"    {k!r}: {v!r}")
    elif isinstance(data, list):
        print(f"  List of {len(data)} items")
        if data and isinstance(data[0], dict):
            print(f"  First item keys: {list(data[0].keys())}")
            if full:
                print(f"  First item: {json.dumps(data[0], indent=4)}")
            elif len(data) > 0:
                # Print first item truncated
                first_str = json.dumps(data[0], indent=2)
                print(f"  First item (truncated):\n{first_str[:300]}")


if __name__ == "__main__":
    main()
