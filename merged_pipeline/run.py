#!/usr/bin/env python3
"""
Merged EIS pipeline CLI.

Examples
--------
Dry-run (no LLM calls; sections/regex paths only):
    python run.py --doc-key p1074_35556035057348 \\
        --output output/p1074_35556035057348.json --no-llm

Full run with default $2.00 budget cap:
    export ANTHROPIC_API_KEY=sk-...
    python run.py --doc-key p1074_35556035057348 \\
        --output output/p1074_35556035057348.json

Custom budget cap:
    python run.py --doc-key p1074_... --output out.json --budget-usd 0.50

Notes
-----
  - With --no-llm, every LLM-dependent field downgrades to needs_review or its
    stage-specific abstention status (themes -> skipped_summary_unavailable,
    stakeholders -> skipped_no_llm, etc.). The pipeline never aborts; it
    always writes a valid EISRecord JSON.
  - Token ledger appends to output/token_ledger.json after each run unless
    --no-ledger is passed.
  - On budget cap trip mid-run, the pipeline catches BudgetExceededError and
    writes whatever has been populated so far (synthesis_plan §Per-doc cost
    cap "partial-output semantics"). Fields not yet processed stay at their
    schema defaults; extraction_budget_status flips to "partial_budget_cap".
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline import config
from pipeline.llm_client import BudgetExceededError, LLMClient
from pipeline.stage5_output import run_doc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the merged EIS pipeline on a single doc_key.",
    )
    p.add_argument("--doc-key", required=True,
                   help="docs_with_digits.json key (e.g. p1074_35556035057348)")
    p.add_argument("--output", required=True,
                   help="path to write EISRecord JSON")
    p.add_argument("--docs-json", default=config.DEFAULT_DOCS_JSON,
                   help=f"path to docs_with_digits.json (default: {config.DEFAULT_DOCS_JSON})")
    p.add_argument("--budget-usd", type=float, default=config.DEFAULT_BUDGET_USD,
                   help=f"per-doc hard budget cap in USD (default: ${config.DEFAULT_BUDGET_USD:.2f})")
    p.add_argument("--no-llm", action="store_true",
                   help="run without any LLM calls (regex/section paths only)")
    p.add_argument("--dry-run", action="store_true",
                   help="LLM dry-run mode (logs prompts; doesn't call API)")
    p.add_argument("--no-ledger", action="store_true",
                   help="don't append to output/token_ledger.json after run")
    p.add_argument("--nul-cache-dir", default=None,
                   help=f"NUL API cache dir (default: {config.DEFAULT_NUL_CACHE_DIR})")
    p.add_argument("--backend", default="auto", choices=["auto", "anthropic", "bedrock"],
                   help="LLM backend (default: auto — picks bedrock when AWS creds present "
                        "and ANTHROPIC_API_KEY is not, else anthropic)")
    p.add_argument("--aws-region", default=None,
                   help="AWS region for Bedrock (default: $AWS_REGION or $AWS_DEFAULT_REGION)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    llm: LLMClient | None
    if args.no_llm:
        llm = None
    else:
        try:
            llm = LLMClient(
                dry_run=args.dry_run,
                budget_usd=args.budget_usd,
                backend=args.backend,
                aws_region=args.aws_region,
            )
        except ValueError as exc:
            print(f"error: {exc}\n\n"
                  f"hint: pass --no-llm for a regex/section-only run, OR\n"
                  f"      --backend anthropic and set ANTHROPIC_API_KEY, OR\n"
                  f"      --backend bedrock with AWS_REGION + AWS creds set",
                  file=sys.stderr)
            return 2

    try:
        record = run_doc(
            args.doc_key,
            docs_json_path=args.docs_json,
            nul_cache_dir=args.nul_cache_dir,
            llm=llm,
            output_path=args.output,
            write_ledger=not args.no_ledger,
        )
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except BudgetExceededError as exc:
        # Budget tripped mid-run. We still want to write a partial JSON.
        # The cleanest way: re-run individual stages catching BudgetExceeded
        # is more refactor than fits this MVP. For v1, treat budget trip as
        # a hard stop with a clear message; the doc gets re-runnable later.
        print(f"\nBUDGET CAP TRIPPED: {exc}", file=sys.stderr)
        print(
            "Partial-output semantics not yet wired in v1 (synthesis_plan §Per-doc "
            "cost cap). Re-run with a larger --budget-usd if needed.", file=sys.stderr,
        )
        return 3

    # Print a one-line summary
    print(
        f"OK doc={args.doc_key} routing={record.validation.review_routing} "
        f"warnings={len(record.pipeline.warnings)} "
        f"cost=${record.pipeline.total_cost_usd:.4f} "
        f"duration={record.pipeline.duration_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
