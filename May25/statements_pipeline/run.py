"""
statements_pipeline orchestrator.

Per-doc flow:
    chunk (segment_a) → extract (people_pipeline) → verify (people_pipeline) →
    merge by (entity, stance) (people_pipeline) → find_statement (local) →
    write per-person folder + index.json (writer)

Doc source:
    By default, processes every doc found in segment_a's PAGES_DATA_DIR
    (`Documents/output/<doc_id>/page_NNNN.json`). Per-page JSONs are joined
    into a single full_text by `pages.load_doc`; chunking and the
    find_statement window slice contiguous spans from that joined text, so the
    LLM never sees raw per-page boundaries.

    title/work_id come from `inventory.lookup_work(doc_id)` when the doc is
    in the local MARC-shaped inventory CSV; otherwise they're left empty.

Subcommands:
    process              run pipeline against every doc in PAGES_DATA_DIR
    process --doc D      run on a specific doc_id (need not be in the inventory)
    process --limit N    process at most N docs
    process --force      ignore the raw_extract checkpoint
    status               progress + cost summary

Output:
    output/people/<doc_id>/
    ├── index.json
    ├── 001_sierra_club.json
    └── 002_john_smith.json
    ...

The extract step is checkpointed per-doc at output/raw_extract/<doc_id>.json so
reruns of find_statement don't re-call the (expensive) per-chunk extractor.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import settings  # configures sys.path for segment_a + people_pipeline

# segment_a imports
from chunk import chunks_for_doc
from pages import Doc, list_doc_ids, load_doc
from inventory import lookup_work

# people_pipeline imports (reused as-is)
from extract import extract_doc
from verify import verify_rows
from merge import merge_rows

# local imports
from find_statement import find_statements_for_doc
from writer import write_doc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("statements_pipeline")


# --- I/O helpers -------------------------------------------------------------

def _read_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _entry_for_doc(doc_id: str) -> dict:
    """Build a {doc_id, work_id, title} entry; fills work_id/title from inventory if available."""
    entry = {"doc_id": doc_id, "work_id": None, "title": ""}
    try:
        work = lookup_work(doc_id)
    except Exception as e:
        log.warning(f"  inventory lookup failed for {doc_id}: {e}")
        work = None
    if work:
        entry["work_id"] = work.get("id")
        entry["title"] = work.get("title") or ""
    return entry


def _enumerate_doc_ids() -> list[str]:
    """All doc_ids that have a per-page JSON dir under PAGES_DATA_DIR."""
    ids = list_doc_ids()
    if not ids:
        raise SystemExit(
            "No docs found under PAGES_DATA_DIR. Expected per-page JSONs at "
            "Documents/output/<doc_id>/page_NNNN.json."
        )
    return ids


# --- per-doc pipeline --------------------------------------------------------

def process_doc(selection_entry: dict, doc: Doc, force: bool = False) -> dict:
    doc_id = selection_entry["doc_id"]
    work_id = selection_entry.get("work_id")
    title = selection_entry.get("title") or ""
    log.info(f"=== {doc_id} | {title!r} ({doc.n_pages} pages, {len(doc.full_text):,} chars) ===")

    raw_path = settings.RAW_EXTRACT_DIR / f"{doc_id}.json"

    t0 = time.time()
    chunked = chunks_for_doc(doc)
    chunks = chunked["chunks"]
    log.info(f"Chunking: {len(chunks)} chunks, {len(chunked['chapters'])} CEQ chapters")

    # ---- Per-chunk extraction (checkpointed) ----
    raw = _read_json(raw_path) if not force else None
    if raw is None:
        log.info(f"Extracting from {len(chunks)} chunks (parallel={settings.EXTRACT_PARALLEL})...")
        per_chunk = extract_doc(chunks, doc_id=doc_id)
        raw = {
            "doc_id": doc_id,
            "work_id": work_id,
            "title": title,
            "n_chunks": len(chunks),
            "per_chunk": per_chunk,
        }
        _write_json(raw_path, raw)
    else:
        log.info(f"Raw extract: cached → {raw_path}")

    extract_usages = [rec.get("usage") for rec in raw["per_chunk"] if rec.get("usage")]
    extract_usage_summary = settings.aggregate_usages(extract_usages)

    # Flatten per-chunk results.
    flat_rows: list[dict] = []
    for rec in raw["per_chunk"]:
        flat_rows.extend(rec.get("entities") or [])
    log.info(f"Raw rows: {len(flat_rows)}")

    # ---- Verbatim quote verification ----
    verified = verify_rows(flat_rows, doc)
    n_verified = sum(1 for r in verified if r.get("quote_verified"))
    log.info(f"Quote verification: {n_verified}/{len(verified)} verbatim hits")

    # ---- Merge by (entity, stance) ----
    merged = merge_rows(verified)
    log.info(f"Merged into {len(merged)} (entity, stance) rows")

    # ---- Find statement + summarize (per row, parallel) ----
    log.info(f"Finding statements for {len(merged)} rows (parallel={settings.STATEMENT_PARALLEL})...")
    enriched = find_statements_for_doc(merged, doc)

    statement_usages: list[dict] = []
    for r in enriched:
        u = r.pop("_statement_usage", None)
        if u:
            statement_usages.append(u)
    statement_usage_summary = settings.aggregate_usages(statement_usages)

    total_usage_summary = settings.aggregate_usages(extract_usages + statement_usages)

    elapsed = round(time.time() - t0, 1)

    # ---- Write per-person files + index.json ----
    summary = write_doc(
        doc_id=doc_id,
        work_id=work_id,
        title=title,
        n_pages=doc.n_pages,
        n_chunks=len(chunks),
        n_raw_rows=len(flat_rows),
        rows=enriched,
        elapsed_sec=elapsed,
        usage_summary={
            "extract": extract_usage_summary,
            "find_statement": statement_usage_summary,
            "total": total_usage_summary,
        },
    )
    cost = total_usage_summary["total"]["cost_usd"]
    rc = summary.get("review_counts") or {}
    sc = summary.get("statement_counts") or {}
    log.info(
        f"Wrote {summary['out_dir']} ({summary['n_people']} people; "
        f"statement_present={sc.get('present', 0)} "
        f"needs_review={rc.get('needs_review', 0)}) "
        f"in {elapsed}s — est. cost ${cost:.4f}"
    )
    return summary


# --- subcommands -------------------------------------------------------------

def _load_doc_or_warn(doc_id: str) -> Doc | None:
    try:
        return load_doc(doc_id)
    except FileNotFoundError as e:
        log.warning(f"  No per-page JSON for {doc_id}: {e}")
        return None


def cmd_process(args) -> int:
    if args.doc:
        entry = _entry_for_doc(args.doc)
        doc = _load_doc_or_warn(args.doc)
        if doc is None:
            return 1
        process_doc(entry, doc, force=args.force)
        return 0

    doc_ids = _enumerate_doc_ids()
    limit = args.limit if args.limit is not None else len(doc_ids)
    todo = doc_ids[:limit]
    log.info(f"Processing {len(todo)}/{len(doc_ids)} doc(s) from PAGES_DATA_DIR.")

    summary: list[dict] = []
    for i, doc_id in enumerate(todo, 1):
        log.info(f"\n[{i}/{len(todo)}] {doc_id}")
        doc = _load_doc_or_warn(doc_id)
        if doc is None:
            log.warning("  Skipping: missing per-page JSON")
            continue
        entry = _entry_for_doc(doc_id)
        try:
            summary.append(process_doc(entry, doc, force=args.force))
        except Exception as e:
            log.exception(f"  Failed: {e}")
            summary.append({"doc_id": doc_id, "error": str(e)})

    _write_json(settings.RUN_SUMMARY_PATH, {"runs": summary})
    grand_cost = round(
        sum((r.get("usage") or {}).get("total", {}).get("cost_usd", 0) for r in summary),
        4,
    )
    log.info(
        f"Done. Wrote run summary for {len(summary)} doc(s) → {settings.RUN_SUMMARY_PATH}. "
        f"Grand-total estimated cost across this run: ${grand_cost:.4f}"
    )
    return 0


def cmd_status(args) -> int:
    doc_ids = list_doc_ids()
    print(f"Docs available in PAGES_DATA_DIR: {len(doc_ids)}")
    have_raw = sum(1 for d in doc_ids if (settings.RAW_EXTRACT_DIR / f"{d}.json").exists())
    have_people = sum(
        1 for d in doc_ids
        if (settings.PEOPLE_DIR / d / "index.json").exists()
    )
    print(f"Raw extract done: {have_raw}/{len(doc_ids)}")
    print(f"Per-person dirs: {have_people}/{len(doc_ids)}")

    # Aggregate counts across completed docs.
    totals = {"people": 0, "needs_review": 0, "statement_present": 0}
    cost_total = 0.0
    in_total = 0
    out_total = 0
    for d in doc_ids:
        p = settings.PEOPLE_DIR / d / "index.json"
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        totals["people"] += data.get("n_people", 0)
        rc = data.get("review_counts") or {}
        totals["needs_review"] += rc.get("needs_review", 0)
        sc = data.get("statement_counts") or {}
        totals["statement_present"] += sc.get("present", 0)
        usage_total = ((data.get("usage") or {}).get("total") or {}).get("total") or {}
        cost_total += usage_total.get("cost_usd", 0) or 0
        in_total += usage_total.get("input_tokens", 0) or 0
        out_total += usage_total.get("output_tokens", 0) or 0
    if totals["people"]:
        print(f"\nAcross completed docs: {totals['people']} (entity, stance) people files")
        print(
            f"  statement_present={totals['statement_present']}  "
            f"needs_review={totals['needs_review']}"
        )
        print(
            f"  Tokens: input={in_total:,}  output={out_total:,}  "
            f"est. cost so far: ${cost_total:.4f}"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="statements_pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_proc = sub.add_parser("process", help="Run pipeline against segment_a selection")
    p_proc.add_argument("--limit", type=int, default=None, help="process at most N docs")
    p_proc.add_argument("--doc", type=str, default=None, help="process a single doc_id")
    p_proc.add_argument("--force", action="store_true", help="ignore raw_extract checkpoint")
    p_proc.set_defaults(func=cmd_process)

    p_stat = sub.add_parser("status", help="Show progress")
    p_stat.set_defaults(func=cmd_status)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
