#!/usr/bin/env python3
"""
CLI entrypoint: run the full EIS metadata pipeline on a single document folder.

Usage:
    python run.py --doc-dir ./data/P0491_35556036063543 --output ./output/result.json

Flags:
    --dry-run          Skip LLM calls; log prompts that would have been sent
    --skip-stages 2,3  Run only specified stages (comma-separated: 0,1,2,3)
    --only-fields summary,themes  Extract a subset of Stage 2 fields
    --budget-usd 5.00  Abort if total estimated cost exceeds this
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EIS metadata extraction pipeline (v1)")

    # Input: either a full S3-style doc folder, or a docs_with_digits.json entry
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--doc-dir", help="Path to document folder (P0491_<DOC_ID>/)")
    input_group.add_argument("--json-file", help="Path to docs_with_digits.json")

    p.add_argument("--doc-key", help="Key in docs_with_digits.json, e.g. P0491_35556036063543 (required with --json-file)")
    p.add_argument("--output", required=True, help="Output JSON file path")
    p.add_argument("--dry-run", action="store_true", help="Skip LLM calls")
    p.add_argument(
        "--skip-stages",
        default="",
        help="Comma-separated stage numbers to SKIP (e.g. '2,3')",
    )
    p.add_argument(
        "--only-fields",
        default="",
        help="Comma-separated Stage 2 fields to run (others skipped). "
             "Valid: summary,themes,location,alternatives,key_people,"
             "historical_internal,historical_external,current_status",
    )
    p.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="Abort if projected cost exceeds this amount (USD)",
    )
    p.add_argument(
        "--token-ledger",
        default="output/token_ledger.json",
        help="Path to the persistent token usage ledger (set empty to disable)",
    )
    return p.parse_args()


ALL_STAGE2_FIELDS = [
    "summary",
    "themes",
    "location",
    "alternatives",
    "key_people",
    "historical_internal",
    "historical_external",
    "current_status",
]


def main() -> None:
    args = parse_args()

    skip_stages = {int(s.strip()) for s in args.skip_stages.split(",") if s.strip()}
    only_fields = (
        {f.strip() for f in args.only_fields.split(",") if f.strip()}
        if args.only_fields
        else set(ALL_STAGE2_FIELDS)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Lazy imports (keep startup fast; spaCy load is slow)
    # -----------------------------------------------------------------------
    from pipeline.io_layer import load_document, load_from_digits_json
    from pipeline.llm_client import LLMClient, BudgetExceededError
    from pipeline import stage0_triage, stage1_chunking, stage3_critic
    from pipeline.stage2_fields import (
        summary as s2_summary,
        themes as s2_themes,
        location as s2_location,
        alternatives as s2_alternatives,
        key_people as s2_key_people,
        historical_internal as s2_hist_internal,
        historical_external as s2_hist_external,
        current_status as s2_current_status,
    )
    from pipeline.config import MODELS
    from pipeline.schema import EISRecord, OCRInfo, PipelineMetadata

    client = LLMClient(dry_run=args.dry_run, budget_usd=args.budget_usd)
    warnings: list[str] = []
    run_start = time.monotonic()

    # -----------------------------------------------------------------------
    # Load document
    # -----------------------------------------------------------------------
    if args.json_file:
        if not args.doc_key:
            logger.error("--doc-key is required when using --json-file")
            sys.exit(1)
        logger.info("Loading from docs_with_digits.json: key=%s", args.doc_key)
        doc = load_from_digits_json(args.json_file, args.doc_key)
    else:
        logger.info("Loading document from: %s", args.doc_dir)
        doc = load_document(args.doc_dir)
    logger.info("Loaded: doc_id=%s project_id=%s pages=%d", doc.doc_id, doc.project_id, len(doc.pages))

    # Bootstrap record with required fields
    record = EISRecord(
        doc_id=doc.doc_id,
        project_id=doc.project_id,
        ocr=OCRInfo(median_confidence=None, page_count=len(doc.pages), unclear_document_flag=False),
        pipeline_metadata=PipelineMetadata(
            models_used=dict(MODELS),
        ),
    )

    # -----------------------------------------------------------------------
    # Stage 0 — Deterministic triage
    # -----------------------------------------------------------------------
    if 0 not in skip_stages:
        logger.info("=== Stage 0: Triage ===")
        try:
            w = stage0_triage.run(doc, record)
            warnings.extend(w)
        except Exception as exc:
            logger.error("Stage 0 failed: %s", exc)
            raise
    else:
        logger.info("Skipping Stage 0")

    # -----------------------------------------------------------------------
    # Stage 1 — Chunking
    # -----------------------------------------------------------------------
    if 1 not in skip_stages:
        logger.info("=== Stage 1: Chunking ===")
        try:
            w = stage1_chunking.run(doc, record, client)
            warnings.extend(w)
        except Exception as exc:
            logger.error("Stage 1 failed: %s", exc)
            raise
    else:
        logger.info("Skipping Stage 1")

    # -----------------------------------------------------------------------
    # Stage 2 — Per-field extraction
    # -----------------------------------------------------------------------
    if 2 not in skip_stages:
        logger.info("=== Stage 2: Field extraction ===")

        field_runners = {
            "summary":           s2_summary.run,
            "themes":            s2_themes.run,
            "location":          s2_location.run,
            "alternatives":      s2_alternatives.run,
            "key_people":        s2_key_people.run,
            "historical_internal": s2_hist_internal.run,
            "historical_external": s2_hist_external.run,
            "current_status":    s2_current_status.run,
        }

        for field_name, runner in field_runners.items():
            if field_name not in only_fields:
                logger.info("  Skipping field: %s", field_name)
                continue
            logger.info("  Extracting: %s", field_name)
            try:
                runner(record, client)
            except BudgetExceededError:
                logger.error("Budget exceeded — stopping after field: %s", field_name)
                warnings.append(f"budget_exceeded_at: {field_name}")
                break
            except Exception as exc:
                logger.error("Field '%s' failed: %s", field_name, exc)
                warnings.append(f"field_error: {field_name}: {exc}")
    else:
        logger.info("Skipping Stage 2")

    # -----------------------------------------------------------------------
    # Stage 3 — Critic
    # -----------------------------------------------------------------------
    if 3 not in skip_stages:
        logger.info("=== Stage 3: Critic ===")
        try:
            w = stage3_critic.run(record, client)
            warnings.extend(w)
        except Exception as exc:
            logger.error("Stage 3 failed: %s", exc)
            warnings.append(f"critic_stage_error: {exc}")
    else:
        logger.info("Skipping Stage 3")

    # -----------------------------------------------------------------------
    # Finalize pipeline metadata
    # -----------------------------------------------------------------------
    record.pipeline_metadata.total_tokens = (
        client.usage.total_input_tokens + client.usage.total_output_tokens
    )
    record.pipeline_metadata.total_cost_usd = round(client.usage.total_cost_usd, 6)
    record.pipeline_metadata.warnings = warnings

    # -----------------------------------------------------------------------
    # Validate with Pydantic and write output
    # -----------------------------------------------------------------------
    try:
        validated = EISRecord.model_validate(record.model_dump())
    except Exception as exc:
        logger.error("Pydantic validation error: %s", exc)
        warnings.append(f"schema_validation_error: {exc}")
        validated = record  # write anyway — don't silently drop data

    output_json = validated.model_dump_json(indent=2)
    output_path.write_text(output_json, encoding="utf-8")

    run_duration = time.monotonic() - run_start

    # Persistent token ledger — append this run's usage.
    if args.token_ledger:
        from pipeline.token_ledger import append_run
        try:
            append_run(
                ledger_path=args.token_ledger,
                doc_id=doc.doc_id,
                usage=client.usage,
                duration_seconds=run_duration,
                pipeline_version=record.pipeline_metadata.version,
            )
        except Exception as exc:
            logger.warning("Could not update token ledger: %s", exc)

    logger.info(
        "Done. Output written to %s | cost=$%.4f | warnings=%d | duration=%.1fs",
        output_path,
        client.usage.total_cost_usd,
        len(warnings),
        run_duration,
    )

    if warnings:
        logger.warning("Warnings: %s", warnings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
