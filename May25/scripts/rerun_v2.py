"""
Rerun the 20 calibration documents under extractor-v2.1, through Bedrock.

Run from May25/. Four steps, each its own command, so you see the cost of each
before paying for the next:

    python scripts/rerun_v2.py check     # free: is everything in place?
    python scripts/rerun_v2.py ping      # ~$0.01: can this key reach Sonnet and Opus?
    python scripts/rerun_v2.py smoke     # ~$3-5: one 123-page document end to end
    python scripts/rerun_v2.py all       # ~$50-100: the other 19, capped by --max-usd

Credentials come from spring_pipeline/.env, which mcal.settings loads on import:

    AWS_BEARER_TOKEN_BEDROCK=...   # a Bedrock API key (AWS console > Bedrock > API keys)
    AWS_REGION=us-east-1           # optional; must be a US region (the model ids
                                   # are us.anthropic.* cross-region profiles)

What a run does to the repo: every document's folder under segment_b/output/ is
regenerated in place (--force), including fresh blind/ and reveal/ grading
sheets. The v1 outputs are tracked in git, so nothing is lost; `git diff` shows
what changed. batch_summary.json for v1 is kept as batch_summary.v1.json.

After `all` finishes, grade the blind sheets: segment_b/output/<doc_id>/blind/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MAY25 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAY25))

from mcal import settings  # noqa: E402  (loads .env, bridges segment_a)

import llm  # noqa: E402  (segment_a)
from pages import load_doc  # noqa: E402
from segment_b import run as segb  # noqa: E402

EXTRACTOR_TAG = "extractor-v2.1"

# The 8 seed documents (graded in Evaluation - Sheet1.csv, against prose that has
# since been rewritten) and the 12 documents from the v1 Segment B batch.
# Listed explicitly so the calibration set is fixed, not whatever is on disk.
SEED_DOCS = [
    "p1074_35556036058550", "p1074_35556036105336", "p1074_35556036546182",
    "p1074_35556036806586", "p1074_35556036811230", "p1074_35556036861797",
    "p1074_35556038322269", "p1074_35556039563135",
]
SEGMENT_B_V1_DOCS = [
    "p0491_35556036854362", "p1074_35556036108041", "p1074_35556036522308",
    "p1074_35556036525772", "p1074_35556036525913", "p1074_35556036535615",
    "p1074_35556036543080", "p1074_35556036796613", "p1074_35556036854040",
    "p1074_35556036855435", "p1074_35556036861656", "p1074_35556038062691",
]
ALL_DOCS = SEED_DOCS + SEGMENT_B_V1_DOCS

# 123 pages, three chunks: big enough to exercise map, reduce, retries and every
# Critic call, small enough to be a cheap first spend.
SMOKE_DOC = "p1074_35556036105336"

OUT = segb.OUTPUT_DIR


# --- check ------------------------------------------------------------------


def check() -> bool:
    """Free preflight. Prints one line per check; True if all pass."""
    ok = True

    def report(passed: bool, label: str, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'ok' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    print("Preflight")

    # Code: the tag is in HEAD's history and the pipeline code is unmodified.
    tagged = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXTRACTOR_TAG, "HEAD"],
        cwd=MAY25, capture_output=True,
    ).returncode == 0
    report(tagged, f"HEAD contains {EXTRACTOR_TAG}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "segment_a", "segment_b", "mcal"],
        cwd=MAY25, capture_output=True, text=True,
    ).stdout
    # Run outputs and the rolling null-tag monitor change on every run.
    dirty = [
        l for l in dirty.splitlines()
        if "segment_b/output/" not in l and "null_tag_monitor" not in l
    ]
    report(not dirty, "pipeline code has no uncommitted changes",
           f"{len(dirty)} changed file(s)" if dirty else "")

    # Credentials: present, never printed.
    has_bedrock = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
    has_api = bool(os.environ.get("ANTHROPIC_API_KEY"))
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    report(has_bedrock or has_api, "model credentials set",
           "Bedrock key" if has_bedrock else ("Anthropic API key" if has_api
           else "add AWS_BEARER_TOKEN_BEDROCK to spring_pipeline/.env"))
    if has_bedrock:
        report(region.startswith("us-"), "AWS region is a US region", region)

    # Extractor settings the calibration depends on.
    import config as segment_a_config
    report(segment_a_config.SUMMARY_MAP_MODEL == "sonnet", "summary map model is sonnet")
    stack = settings.geocoder_precheck()["stack"]
    report(stack == "full", "geocoder stack is full", stack)

    # Inputs: every document loads.
    missing = []
    for d in ALL_DOCS:
        try:
            load_doc(d)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{d}: {e}")
    report(not missing, f"all {len(ALL_DOCS)} documents load", "; ".join(missing[:3]))

    free_gb = shutil.disk_usage(MAY25).free / 1e9
    report(free_gb >= 2, "at least 2 GB free disk", f"{free_gb:.1f} GB")

    print("\nAll checks passed." if ok else "\nFix the FAIL lines before spending anything.")
    return ok


# --- ping -------------------------------------------------------------------


def ping() -> bool:
    """One tiny call per model: proves the key can reach both. About $0.01."""
    llm.start_usage_session()
    ok = True
    for name, model in (("sonnet", llm.MODEL_SONNET), ("opus", llm.MODEL_OPUS)):
        try:
            got, _ = llm.call_json_with_usage(
                model,
                "Reply with JSON only.",
                'Return exactly {"ok": true}.',
                max_tokens=50,
            )
            print(f"  [ok] {name} ({model}) -> {got}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [FAIL] {name} ({model}) -> {e}")
    usage = llm.aggregate_usages(llm.end_usage_session())
    print(f"  cost ${((usage.get('total') or {}).get('cost_usd') or 0):.4f}")
    return ok


# --- runs -------------------------------------------------------------------


def _archive_v1_summary() -> None:
    src, dst = OUT / "batch_summary.json", OUT / "batch_summary.v1.json"
    if src.exists() and not dst.exists():
        src.rename(dst)
        print(f"Kept the v1 batch summary as {dst.name}")


def _run(doc_ids: list[str], batch_id: str, max_usd: float | None) -> dict:
    summary = segb.process_batch(
        doc_ids, force=True, batch_id=batch_id, max_usd=max_usd
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "docs"}, indent=2))
    for d in summary["docs"]:
        cost = ((d.get("usage") or {}).get("total") or {}).get("cost_usd", 0) or 0
        line = f"  {d['doc_id']}  {d.get('n_pages', '?'):>4}pp  ${cost:6.2f}"
        if d.get("error"):
            line += f"  ERROR: {d['error']}"
        elif d.get("warnings"):
            line += f"  {len(d['warnings'])} warning(s)"
        print(line)
    return summary


def smoke() -> dict:
    _archive_v1_summary()
    summary = _run([SMOKE_DOC], "v2-rerun-smoke", max_usd=None)
    (OUT / "batch_summary.json").rename(OUT / "batch_summary.v2-smoke.json")
    print(
        f"\nLook at segment_b/output/{SMOKE_DOC}/ (run_manifest.json, blind/) "
        f"before running `all`."
    )
    return summary


def run_all(max_usd: float) -> dict:
    smoke_path = OUT / "batch_summary.v2-smoke.json"
    if not smoke_path.exists():
        raise SystemExit("Run `smoke` first and check its output.")
    _archive_v1_summary()
    rest = [d for d in ALL_DOCS if d != SMOKE_DOC]
    summary = _run(rest, "v2-rerun", max_usd=max_usd)

    # One summary for all 20, so the cost table and gate rate cover the set.
    smoke_summary = json.loads(smoke_path.read_text(encoding="utf-8"))
    docs = smoke_summary["docs"] + summary["docs"]
    ok = [d for d in docs if not d.get("error")]
    total = sum(((d.get("usage") or {}).get("total") or {}).get("cost_usd", 0) or 0 for d in ok)
    n_fields = sum((d.get("rollup") or {}).get("n_fields", 0) for d in ok)
    n_gated = sum((d.get("rollup") or {}).get("n_gated", 0) for d in ok)
    combined = {
        "stage": summary["stage"],
        "extractor": EXTRACTOR_TAG,
        "batch_id": "v2-rerun",
        "n_docs": len(docs),
        "n_ok": len(ok),
        "n_failed": len(docs) - len(ok),
        "total_cost_usd": round(total, 4),
        "per_doc_avg_usd": round(total / len(ok), 4) if ok else None,
        "n_fields": n_fields,
        "n_gated": n_gated,
        "overall_gate_rate": round(n_gated / n_fields, 4) if n_fields else None,
        "docs": docs,
    }
    (OUT / "batch_summary.json").write_text(json.dumps(combined, indent=2, default=str) + "\n")
    print(f"\nAll 20: {len(ok)} ok, ${total:.2f} total. Summary in batch_summary.json.")
    if len(ok) < len(docs):
        print("Rerun failed documents one at a time with:")
        print("  python -m segment_b.run process --doc <doc_id> --force")
    return combined


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="free preflight")
    sub.add_parser("ping", help="one tiny call to Sonnet and Opus (~$0.01)")
    sub.add_parser("smoke", help=f"run {SMOKE_DOC} end to end (~$3-5)")
    a = sub.add_parser("all", help="run the other 19 documents")
    a.add_argument("--max-usd", type=float, default=120.0,
                   help="stop starting new documents past this spend (default 120)")
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.cmd == "check":
        return 0 if check() else 1
    if not check():
        return 1
    if args.cmd == "ping":
        return 0 if ping() else 1
    if args.cmd == "smoke":
        return 0 if smoke()["n_failed"] == 0 else 1
    return 0 if run_all(args.max_usd)["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
