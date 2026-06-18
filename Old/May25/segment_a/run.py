"""
Segment A orchestrator.

Usage:
    # Build candidate pool + write the 20-doc selection
    python run.py select

    # Process N docs from the selection (defaults to all 20).
    # Per-doc outputs are checkpointed; rerun resumes where you left off.
    python run.py process            # all 20
    python run.py process --limit 1  # smoke test: just the first selected doc
    python run.py process --doc P0491_35556036806768  # one specific doc

    # Inspect what was produced
    python run.py status

Notes:
    - Requires ANTHROPIC_API_KEY in env (run from opencode or `export ANTHROPIC_API_KEY=...`).
    - Doc text is loaded from PAGES_DATA_DIR/<doc_id>/<page_num>.json. Page numbers
      in outputs are real (from the source files), not estimated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from config import (
    CRITIC_DIR,
    GRADING_DIR,
    M1_DIR,
    M2_DIR,
    SELECTION_PATH,
)
from chunk import chunks_for_doc
from critic import run_critic
from grading import write_grading_sheet
from llm import aggregate_usages, end_usage_session, start_usage_session
from m1 import run_m1
from m2 import run_m2
from inventory import lookup_work
from pages import Doc, list_doc_ids, load_doc
from selection import select as run_select, load_selection


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("segment_a")


# --- helpers -----------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _resolve_work_and_doc(doc_id: str) -> tuple[Optional[dict], Optional[Doc]]:
    try:
        doc = load_doc(doc_id)
    except FileNotFoundError:
        log.error(f"doc_id {doc_id} has no pages_data/<doc_id>/ directory")
        return None, None
    work = lookup_work(doc_id)
    if work is None:
        log.warning(f"No inventory CSV row matches doc_id {doc_id}")
        return None, doc
    return work, doc


# --- pipeline per doc --------------------------------------------------------

def process_doc(work: dict, doc_id: str, doc: Doc, force: bool = False) -> dict:
    """Run M1, M2, Critic, grading sheet for one doc. Checkpointed per stage.

    Token usage and estimated cost are tracked across all LLM calls in this doc
    via llm.start_usage_session / end_usage_session. The returned dict includes
    a `usage` key with the per-model breakdown and total cost. Stages that hit
    the per-stage cache do not generate new calls; cost reflects only the work
    actually performed in this run.
    """
    work_id = work.get("id")
    title = work.get("title") or ""
    log.info(f"=== {doc_id} | {title!r} ({doc.n_pages} pages, {len(doc.full_text):,} chars) ===")

    m1_path = M1_DIR / f"{doc_id}.json"
    m2_path = M2_DIR / f"{doc_id}.json"
    crit_path = CRITIC_DIR / f"{doc_id}.json"

    t0 = time.time()
    chunked = chunks_for_doc(doc)
    log.info(f"Chunking: {len(chunked['chunks'])} chunks, {len(chunked['chapters'])} CEQ chapters")

    start_usage_session()
    try:
        # ---- M1 ----
        m1 = _read_json(m1_path) if not force else None
        if m1 is None:
            log.info("Running M1...")
            m1 = run_m1(work, doc)
            _write_json(m1_path, m1)
        else:
            log.info(f"M1: cached → {m1_path}")

        # ---- M2 ----
        m2 = _read_json(m2_path) if not force else None
        if m2 is None:
            log.info("Running M2...")
            m2 = run_m2(doc, chunked=chunked)
            _write_json(m2_path, m2)
        else:
            log.info(f"M2: cached → {m2_path}")

        # ---- Critic ----
        crit = _read_json(crit_path) if not force else None
        if crit is None:
            log.info("Running Critic...")
            crit = run_critic(doc, m1, m2)
            _write_json(crit_path, crit)
        else:
            log.info(f"Critic: cached → {crit_path}")

        # ---- Grading sheet ----
        sheet_path = write_grading_sheet(GRADING_DIR, doc_id, work_id, title, m1, m2, crit)
        log.info(f"Grading sheet → {sheet_path}")
    finally:
        usages = end_usage_session()

    usage_summary = aggregate_usages(usages)
    cost = usage_summary["total"]["cost_usd"]
    n_calls = usage_summary["total"]["calls"]
    elapsed = round(time.time() - t0, 1)
    log.info(
        f"{doc_id} done in {elapsed}s — {n_calls} LLM call(s), est. cost ${cost:.4f}"
    )

    return {
        "doc_id": doc_id,
        "work_id": work_id,
        "title": title,
        "elapsed_sec": elapsed,
        "m1_path": str(m1_path),
        "m2_path": str(m2_path),
        "critic_path": str(crit_path),
        "grading_sheet": str(sheet_path),
        "usage": usage_summary,
    }


# --- subcommands -------------------------------------------------------------

def cmd_select(args) -> int:
    sel = run_select()
    log.info(f"Selected {len(sel)} docs. Buckets: " + ", ".join(
        f"{b}={sum(1 for s in sel if (s['estimated_pages'] < 200) == (b == 'short') and (s['estimated_pages'] >= 800) == (b == 'long') and (200 <= s['estimated_pages'] < 800) == (b == 'medium'))}"
        for b in ("short", "medium", "long")
    ))
    return 0


def cmd_process(args) -> int:
    if args.doc:
        work, doc = _resolve_work_and_doc(args.doc)
        if work is None or doc is None:
            return 1
        result = process_doc(work, args.doc, doc, force=args.force)
        _write_run_summary([result])
        return 0

    selection = load_selection()
    if selection is None:
        log.info("No selection yet — running `select` first.")
        selection = run_select()

    limit = args.limit if args.limit is not None else len(selection)
    todo = selection[:limit]
    log.info(f"Processing {len(todo)} doc(s)...")

    summary: list[dict] = []
    for i, s in enumerate(todo, 1):
        log.info(f"\n[{i}/{len(todo)}] {s['doc_id']}")
        # Inventory CSV is the metadata source (NUL API was retired earlier in
        # favour of inventory.lookup_work). selection.json's work_id is no
        # longer authoritative; we re-resolve from the doc_id.
        work = lookup_work(s["doc_id"])
        if work is None:
            log.warning(f"  Skipping: no inventory CSV row for {s['doc_id']}")
            continue
        try:
            doc = load_doc(s["doc_id"])
        except FileNotFoundError as e:
            log.warning(f"  Skipping {s['doc_id']}: {e}")
            continue
        try:
            summary.append(process_doc(work, s["doc_id"], doc, force=args.force))
        except Exception as e:
            log.exception(f"  Failed: {e}")
            summary.append({"doc_id": s["doc_id"], "error": str(e)})

    _write_run_summary(summary)
    grand_cost = round(
        sum(((r.get("usage") or {}).get("total") or {}).get("cost_usd", 0) or 0 for r in summary),
        4,
    )
    log.info(
        f"Done. Wrote run summary for {len(summary)} doc(s). "
        f"Grand-total estimated cost across this run: ${grand_cost:.4f}"
    )
    return 0


def _write_run_summary(summary: list[dict]) -> None:
    """Write output/run_summary.json with the runs from this invocation.

    Overwrites — this reflects only the most recent `run.py process` call,
    same as people_pipeline / statements_pipeline. For per-doc cost history
    across runs, look at each doc's individual stage outputs.
    """
    _write_json(Path("output/run_summary.json"), {"runs": summary})


def cmd_status(args) -> int:
    sel = load_selection()
    if sel is None:
        print("No selection yet. Run: python run.py select")
        return 0
    print(f"Selection: {len(sel)} docs at {SELECTION_PATH}")
    have_m1 = sum(1 for s in sel if (M1_DIR / f"{s['doc_id']}.json").exists())
    have_m2 = sum(1 for s in sel if (M2_DIR / f"{s['doc_id']}.json").exists())
    have_crit = sum(1 for s in sel if (CRITIC_DIR / f"{s['doc_id']}.json").exists())
    have_sheet = sum(1 for s in sel if (GRADING_DIR / f"{s['doc_id']}.csv").exists())
    print(f"M1 done:        {have_m1}/{len(sel)}")
    print(f"M2 done:        {have_m2}/{len(sel)}")
    print(f"Critic done:    {have_crit}/{len(sel)}")
    print(f"Grading sheets: {have_sheet}/{len(sel)}")
    return 0


def cmd_grade(args) -> int:
    """Regenerate grading sheets from on-disk M1/M2/Critic JSONs. No LLM calls.

    Useful when M1/M2/Critic exist but the CSV is stale or missing — e.g. after
    editing a stage JSON by hand, or just to refresh CSVs without re-running
    the (expensive) extraction stages.

    Scope:
        --doc D     rebuild only doc D's CSV
        (no flag)   rebuild every doc that has all three of M1/M2/Critic on disk
    """
    if args.doc:
        targets = [args.doc]
    else:
        # Every doc with all three stage JSONs present on disk.
        m1_ids  = {p.stem for p in M1_DIR.glob("*.json")}
        m2_ids  = {p.stem for p in M2_DIR.glob("*.json")}
        crit_ids = {p.stem for p in CRITIC_DIR.glob("*.json")}
        targets = sorted(m1_ids & m2_ids & crit_ids)

    if not targets:
        log.info("No docs found with M1+M2+Critic on disk.")
        return 0

    log.info(f"Regenerating grading sheets for {len(targets)} doc(s)...")
    written = 0
    skipped: list[tuple[str, str]] = []
    for doc_id in targets:
        m1_path = M1_DIR / f"{doc_id}.json"
        m2_path = M2_DIR / f"{doc_id}.json"
        crit_path = CRITIC_DIR / f"{doc_id}.json"

        m1 = _read_json(m1_path)
        m2 = _read_json(m2_path)
        crit = _read_json(crit_path)
        if m1 is None or m2 is None or crit is None:
            missing = [n for n, x in [("m1", m1), ("m2", m2), ("critic", crit)] if x is None]
            skipped.append((doc_id, f"missing: {', '.join(missing)}"))
            continue

        # Title / work_id come from the inventory CSV. Use empty fallbacks so
        # docs not in the CSV still get a refreshed grading sheet.
        try:
            work = lookup_work(doc_id)
        except Exception as e:
            log.warning(f"  inventory lookup failed for {doc_id}: {e}")
            work = None
        title = (work.get("title") if work else "") or ""
        work_id = (work.get("id") if work else None)

        sheet_path = write_grading_sheet(GRADING_DIR, doc_id, work_id, title, m1, m2, crit)
        log.info(f"  ✓ {doc_id} → {sheet_path}")
        written += 1

    if skipped:
        log.info(f"Skipped {len(skipped)} doc(s):")
        for doc_id, reason in skipped:
            log.info(f"  - {doc_id}: {reason}")
    log.info(f"Done. Wrote {written} grading sheet(s).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="segment_a")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sel = sub.add_parser("select", help="Build candidate pool and pick 20 docs")
    p_sel.set_defaults(func=cmd_select)

    p_proc = sub.add_parser("process", help="Run M1 → M2 → Critic → grading on selected docs")
    p_proc.add_argument("--limit", type=int, default=None, help="process at most N docs (smoke test: --limit 1)")
    p_proc.add_argument("--doc", type=str, default=None, help="process a single specific doc_id")
    p_proc.add_argument("--force", action="store_true", help="ignore existing checkpoints")
    p_proc.set_defaults(func=cmd_process)

    p_grade = sub.add_parser(
        "grade",
        help="Rebuild grading sheets from on-disk M1/M2/Critic JSONs (no LLM calls)",
    )
    p_grade.add_argument("--doc", type=str, default=None, help="rebuild a single doc's CSV")
    p_grade.set_defaults(func=cmd_grade)

    p_stat = sub.add_parser("status", help="Show how much of the run is done")
    p_stat.set_defaults(func=cmd_status)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
