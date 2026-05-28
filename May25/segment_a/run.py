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
    - Page numbers in outputs are ESTIMATED from char offsets at 2500 chars/page.
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
from m1 import run_m1
from m2 import run_m2
from nul import fetch_collection_works, find_ocr, load_docs
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


def _resolve_work_and_text(doc_id: str) -> tuple[Optional[dict], Optional[str]]:
    docs = load_docs()
    text = docs.get(doc_id)
    if text is None:
        log.error(f"doc_id {doc_id} not present in docs_with_digits.json")
        return None, None
    works = fetch_collection_works()
    for w in works:
        match = find_ocr(w, {doc_id: text})
        if match:
            return w, text
    log.warning(f"No NUL work matches doc_id {doc_id}")
    return None, text


# --- pipeline per doc --------------------------------------------------------

def process_doc(work: dict, doc_id: str, text: str, force: bool = False) -> dict:
    """Run M1, M2, Critic, grading sheet for one doc. Checkpointed per stage."""
    work_id = work.get("id")
    title = work.get("title") or ""
    log.info(f"=== {doc_id} | {title!r} ({len(text):,} chars) ===")

    m1_path = M1_DIR / f"{doc_id}.json"
    m2_path = M2_DIR / f"{doc_id}.json"
    crit_path = CRITIC_DIR / f"{doc_id}.json"

    t0 = time.time()
    chunked = chunks_for_doc(text)
    log.info(f"Chunking: {len(chunked['chunks'])} chunks, {len(chunked['chapters'])} CEQ chapters")

    # ---- M1 ----
    m1 = _read_json(m1_path) if not force else None
    if m1 is None:
        log.info("Running M1...")
        m1 = run_m1(work, text)
        _write_json(m1_path, m1)
    else:
        log.info(f"M1: cached → {m1_path}")

    # ---- M2 ----
    m2 = _read_json(m2_path) if not force else None
    if m2 is None:
        log.info("Running M2...")
        m2 = run_m2(text, chunked=chunked)
        _write_json(m2_path, m2)
    else:
        log.info(f"M2: cached → {m2_path}")

    # ---- Critic ----
    crit = _read_json(crit_path) if not force else None
    if crit is None:
        log.info("Running Critic...")
        crit = run_critic(text, m1, m2)
        _write_json(crit_path, crit)
    else:
        log.info(f"Critic: cached → {crit_path}")

    # ---- Grading sheet ----
    sheet_path = write_grading_sheet(GRADING_DIR, doc_id, work_id, title, m1, m2, crit)
    log.info(f"Grading sheet → {sheet_path}")

    return {
        "doc_id": doc_id,
        "work_id": work_id,
        "title": title,
        "elapsed_sec": round(time.time() - t0, 1),
        "m1_path": str(m1_path),
        "m2_path": str(m2_path),
        "critic_path": str(crit_path),
        "grading_sheet": str(sheet_path),
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
        work, text = _resolve_work_and_text(args.doc)
        if work is None or text is None:
            return 1
        process_doc(work, args.doc, text, force=args.force)
        return 0

    selection = load_selection()
    if selection is None:
        log.info("No selection yet — running `select` first.")
        selection = run_select()

    limit = args.limit if args.limit is not None else len(selection)
    todo = selection[:limit]
    log.info(f"Processing {len(todo)} doc(s)...")

    docs = load_docs()
    works_by_id = {w.get("id"): w for w in fetch_collection_works()}

    summary: list[dict] = []
    for i, s in enumerate(todo, 1):
        log.info(f"\n[{i}/{len(todo)}] {s['doc_id']}")
        work = works_by_id.get(s["work_id"])
        text = docs.get(s["doc_id"])
        if work is None or text is None:
            log.warning(f"  Skipping: missing work or text")
            continue
        try:
            summary.append(process_doc(work, s["doc_id"], text, force=args.force))
        except Exception as e:
            log.exception(f"  Failed: {e}")
            summary.append({"doc_id": s["doc_id"], "error": str(e)})

    _write_json(Path("output/run_summary.json"), {"runs": summary})
    log.info(f"Done. Wrote run summary for {len(summary)} doc(s).")
    return 0


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

    p_stat = sub.add_parser("status", help="Show how much of the run is done")
    p_stat.set_defaults(func=cmd_status)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
