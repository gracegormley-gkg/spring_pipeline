"""
people_pipeline orchestrator.

Subcommands:
    process          run the full pipeline against the segment_a selection
    process --doc D  run on a specific doc_id
    process --limit N
    process --force  ignore checkpoints
    status           how many docs have completed each stage

Per-doc pipeline:
    chunk → extract (per-chunk Sonnet) → verify (verbatim quote) →
    merge by (entity, stance) → critic (per merged row, Sonnet) → write JSON

Output is one JSON file per doc at output/entries/<doc_id>.json. Per-chunk raw
extractions are checkpointed at output/raw_extract/<doc_id>.json so reruns
don't re-call the extractor.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import settings  # registers segment_a/ on sys.path

# segment_a imports
from chunk import chunks_for_doc
from nul import fetch_collection_works, load_docs

# local imports
from extract import extract_doc
from verify import verify_rows
from merge import merge_rows
from critic import critique_all


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("people_pipeline")


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


def _load_segment_a_selection() -> list[dict]:
    if not settings.SEGMENT_A_SELECTION_PATH.exists():
        raise SystemExit(
            f"segment_a selection not found at {settings.SEGMENT_A_SELECTION_PATH}. "
            "Run `python run.py select` in segment_a first."
        )
    with open(settings.SEGMENT_A_SELECTION_PATH) as f:
        data = json.load(f)
    return data["selected"]


# --- per-doc pipeline --------------------------------------------------------

def process_doc(work: dict, doc_id: str, text: str, force: bool = False) -> dict:
    work_id = work.get("id")
    title = work.get("title") or ""
    log.info(f"=== {doc_id} | {title!r} ({len(text):,} chars) ===")

    raw_path = settings.RAW_EXTRACT_DIR / f"{doc_id}.json"
    out_path = settings.ENTRIES_DIR / f"{doc_id}.json"

    t0 = time.time()
    chunked = chunks_for_doc(text)
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

    # Collect extract usage from the per-chunk records (works for both fresh
    # and cached runs since usage is persisted in the raw extract file).
    extract_usages = [rec.get("usage") for rec in raw["per_chunk"] if rec.get("usage")]
    extract_usage_summary = settings.aggregate_usages(extract_usages)

    # Flatten per-chunk results into one list of mention rows.
    flat_rows: list[dict] = []
    for rec in raw["per_chunk"]:
        flat_rows.extend(rec.get("entities") or [])
    log.info(f"Raw rows: {len(flat_rows)}")

    # ---- Verbatim quote verification ----
    verified = verify_rows(flat_rows, text)
    n_verified = sum(1 for r in verified if r.get("quote_verified"))
    log.info(f"Quote verification: {n_verified}/{len(verified)} verbatim hits")

    # ---- Merge by (entity, stance) ----
    merged = merge_rows(verified)
    log.info(f"Merged into {len(merged)} (entity, stance) rows")

    # ---- Critic (per-row, parallel) ----
    log.info(f"Running critic on {len(merged)} rows (parallel={settings.CRITIC_PARALLEL})...")
    critiqued = critique_all(merged, text)

    # Pull critic usage off the rows (added by critic.critique_row), then strip
    # the private `_critic_usage` key from rows before they're persisted.
    critic_usages: list[dict] = []
    for r in critiqued:
        u = r.pop("_critic_usage", None)
        if u:
            critic_usages.append(u)
    critic_usage_summary = settings.aggregate_usages(critic_usages)

    # Total usage = extract + critic (any cached extract usage is already in the file)
    total_usage_summary = settings.aggregate_usages(extract_usages + critic_usages)

    # Verdict tally for the run summary.
    verdict_counts: dict = {"PASS": 0, "PASS_WITH_NOTE": 0, "RE_EXTRACT": 0, "HUMAN_REVIEW": 0}
    for r in critiqued:
        v = r.get("critic", {}).get("verdict", "HUMAN_REVIEW")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    stance_counts: dict = {}
    for r in critiqued:
        s = r.get("stance", "?")
        stance_counts[s] = stance_counts.get(s, 0) + 1

    elapsed = round(time.time() - t0, 1)

    output = {
        "doc_id": doc_id,
        "work_id": work_id,
        "title": title,
        "estimated_pages": chunked["estimated_pages"],
        "n_chunks": len(chunks),
        "n_raw_rows": len(flat_rows),
        "n_entries": len(critiqued),
        "verdict_counts": verdict_counts,
        "stance_counts": stance_counts,
        "elapsed_sec": elapsed,
        "usage": {
            "extract": extract_usage_summary,
            "critic": critic_usage_summary,
            "total": total_usage_summary,
        },
        "schema": {
            "stance_vocabulary": list(settings.STANCES),
            "kind_vocabulary": list(settings.KINDS),
            "verdicts": ["PASS", "PASS_WITH_NOTE", "RE_EXTRACT", "HUMAN_REVIEW"],
            "page_numbers": "ESTIMATED from char offsets at 2500 chars/page.",
            "merge_rule": "One row per (entity, stance) — stance changes produce separate rows.",
            "cost_note": "USD costs are ESTIMATES from settings.PRICES_USD_PER_M; verify against AWS Bedrock invoice.",
        },
        "entries": critiqued,
    }
    _write_json(out_path, output)
    cost = total_usage_summary["total"]["cost_usd"]
    log.info(
        f"Wrote {out_path} ({len(critiqued)} entries; "
        f"PASS={verdict_counts['PASS']} "
        f"PWN={verdict_counts['PASS_WITH_NOTE']} "
        f"RE={verdict_counts['RE_EXTRACT']} "
        f"HR={verdict_counts['HUMAN_REVIEW']}) in {elapsed}s — est. cost ${cost:.4f}"
    )
    return {
        "doc_id": doc_id,
        "work_id": work_id,
        "title": title,
        "n_entries": len(critiqued),
        "verdict_counts": verdict_counts,
        "stance_counts": stance_counts,
        "elapsed_sec": elapsed,
        "usage": total_usage_summary,
        "out_path": str(out_path),
    }


# --- subcommands -------------------------------------------------------------

def cmd_process(args) -> int:
    selection = _load_segment_a_selection()
    docs = load_docs()
    works_by_id = {w.get("id"): w for w in fetch_collection_works()}

    if args.doc:
        # Find the doc in selection (preferred) or fall back to docs/works lookup.
        sel_entry = next((s for s in selection if s["doc_id"] == args.doc), None)
        if sel_entry is None:
            log.warning(f"{args.doc} not in segment_a selection — running anyway if text/work resolvable.")
            text = docs.get(args.doc)
            if text is None:
                log.error(f"doc_id {args.doc} not in docs_with_digits.json")
                return 1
            # Best-effort work lookup
            from nul import find_ocr
            work = next(
                (w for w in works_by_id.values() if find_ocr(w, {args.doc: text})),
                None,
            )
            if work is None:
                log.error(f"No NUL work matches {args.doc}")
                return 1
        else:
            work = works_by_id.get(sel_entry["work_id"])
            text = docs.get(args.doc)
            if work is None or text is None:
                log.error("Missing work or text for selected doc.")
                return 1
        process_doc(work, args.doc, text, force=args.force)
        return 0

    limit = args.limit if args.limit is not None else len(selection)
    todo = selection[:limit]
    log.info(f"Processing {len(todo)}/{len(selection)} doc(s) from segment_a selection.")

    summary: list[dict] = []
    for i, s in enumerate(todo, 1):
        log.info(f"\n[{i}/{len(todo)}] {s['doc_id']}")
        work = works_by_id.get(s["work_id"])
        text = docs.get(s["doc_id"])
        if work is None or text is None:
            log.warning("  Skipping: missing work or text")
            continue
        try:
            summary.append(process_doc(work, s["doc_id"], text, force=args.force))
        except Exception as e:
            log.exception(f"  Failed: {e}")
            summary.append({"doc_id": s["doc_id"], "error": str(e)})

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
    selection = _load_segment_a_selection()
    print(f"Segment A selection: {len(selection)} docs")
    have_raw = sum(1 for s in selection if (settings.RAW_EXTRACT_DIR / f"{s['doc_id']}.json").exists())
    have_entries = sum(1 for s in selection if (settings.ENTRIES_DIR / f"{s['doc_id']}.json").exists())
    print(f"Raw extract done: {have_raw}/{len(selection)}")
    print(f"Final entries:    {have_entries}/{len(selection)}")

    # Aggregate counts across completed entries.
    totals = {"entries": 0, "PASS": 0, "PASS_WITH_NOTE": 0, "RE_EXTRACT": 0, "HUMAN_REVIEW": 0}
    cost_total = 0.0
    in_total = 0
    out_total = 0
    for s in selection:
        p = settings.ENTRIES_DIR / f"{s['doc_id']}.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        totals["entries"] += d.get("n_entries", 0)
        for k, v in (d.get("verdict_counts") or {}).items():
            totals[k] = totals.get(k, 0) + v
        usage_total = ((d.get("usage") or {}).get("total") or {}).get("total") or {}
        cost_total += usage_total.get("cost_usd", 0) or 0
        in_total += usage_total.get("input_tokens", 0) or 0
        out_total += usage_total.get("output_tokens", 0) or 0
    if totals["entries"]:
        print(f"\nAcross completed docs: {totals['entries']} (entity, stance) rows")
        print(
            f"  PASS={totals['PASS']}  PASS_WITH_NOTE={totals['PASS_WITH_NOTE']}  "
            f"RE_EXTRACT={totals['RE_EXTRACT']}  HUMAN_REVIEW={totals['HUMAN_REVIEW']}"
        )
        print(
            f"  Tokens: input={in_total:,}  output={out_total:,}  "
            f"est. cost so far: ${cost_total:.4f}"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="people_pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_proc = sub.add_parser("process", help="Run pipeline against segment_a selection")
    p_proc.add_argument("--limit", type=int, default=None, help="process at most N docs")
    p_proc.add_argument("--doc", type=str, default=None, help="process a single doc_id")
    p_proc.add_argument("--force", action="store_true", help="ignore checkpoints")
    p_proc.set_defaults(func=cmd_process)

    p_stat = sub.add_parser("status", help="Show progress")
    p_stat.set_defaults(func=cmd_status)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
