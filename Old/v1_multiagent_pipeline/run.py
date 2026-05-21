#!/usr/bin/env python3
"""
v1_multiagent_pipeline — CLI entrypoint.

Runs Stage 1 → 1.5 → 2 → 3a → 3b on a single doc_key from docs_with_digits.json.
Group A + B scope. Later stages (3c–6) added by future sessions.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run.py \\
      --json-file "/path/to/docs_with_digits.json" \\
      --doc-key p1074_35556036099737 \\
      --through-stage 3b \\
      --budget-usd 0.50

Acceptance gate (Group B): the produced output/stage3_fields/<doc_key>.json
contains title, year, lead_agency, publication date, eis_type, sections.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Make `pipeline.*` importable when run from the package dir
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config
from pipeline.llm_client import BudgetExceededError, LLMClient
from pipeline.nul_client import NULClient
from pipeline.schema import EISRecord, PipelineMeta
from pipeline import ingest as stage1
from pipeline import grouping as stage1_5
from pipeline import sections as stage2
from pipeline import stage3a_mets_fields, stage3b_eis_type
from pipeline import token_ledger

# ---------------------------------------------------------------------------
# Stage ordering / parsing
# ---------------------------------------------------------------------------

STAGE_ORDER: list[str] = ["1", "1.5", "2", "3a", "3b"]
LLM_STAGES: set[str] = {"3b"}  # stages that may make LLM calls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v1_multiagent_pipeline (Group A+B)")

    p.add_argument(
        "--json-file",
        default=config.DEFAULT_DOCS_JSON,
        help="Path to docs_with_digits.json",
    )
    p.add_argument(
        "--doc-key",
        required=True,
        help="Key in docs_with_digits.json, e.g. p1074_35556036099737",
    )
    p.add_argument(
        "--through-stage",
        default="3b",
        choices=STAGE_ORDER,
        help="Run all stages up to and including this one",
    )
    p.add_argument(
        "--budget-usd",
        type=float,
        default=config.DEFAULT_BUDGET_USD,
        help="Hard cap on LLM cost (USD); raises BudgetExceededError if exceeded",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; log prompts that would have been sent",
    )
    p.add_argument(
        "--no-embedding-fallback",
        action="store_true",
        help="Skip Stage 2 embedding fallback (regex-only sections)",
    )
    p.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "output"),
        help="Root output directory",
    )
    return p.parse_args()


def stages_to_run(through_stage: str) -> list[str]:
    idx = STAGE_ORDER.index(through_stage)
    return STAGE_ORDER[: idx + 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "runs" / f"{int(time.time())}_{args.doc_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("run")
    logger.info("=" * 70)
    logger.info("v1_multiagent_pipeline | doc_key=%s | through=%s | dry_run=%s",
                args.doc_key, args.through_stage, args.dry_run)
    logger.info("Output root: %s", output_root)

    run_stages = stages_to_run(args.through_stage)
    needs_llm = bool(set(run_stages) & LLM_STAGES) and not args.dry_run

    # Don't bother instantiating LLMClient if we're not making any calls
    llm = LLMClient(dry_run=args.dry_run, budget_usd=args.budget_usd) if (
        needs_llm or args.dry_run
    ) else None

    nul = NULClient(cache_dir=output_root / "nul_cache")

    # Initialize record with mandatory fields
    record = EISRecord(
        publication_id=args.doc_key,
        physical_record_ids=[args.doc_key],
    )
    record.pipeline = PipelineMeta(
        pipeline_version=config.PIPELINE_VERSION,
        stage_versions={k: v for k, v in config.STAGE_VERSIONS.items() if _stage_ran(k, run_stages)},
        model_ids=dict(config.MODELS),
    )

    warnings: list[str] = []
    t_start = time.time()

    # -----------------------------------------------------------------------
    # Stage 1 — Ingest
    # -----------------------------------------------------------------------
    logger.info("=== Stage 1: Ingest ===")
    ingest_artifact = stage1.run(args.json_file, args.doc_key, nul)
    stage1.write_artifact(ingest_artifact, output_root / "stage1_assembled")

    if "1.5" in run_stages:
        logger.info("=== Stage 1.5: Grouping (passthrough) ===")
        grouping_artifact = stage1_5.run(ingest_artifact)
        stage1_5.write_artifact(grouping_artifact, output_root / "stage1_5_grouped")
        record.is_supplemental = grouping_artifact.is_supplemental
        record.physical_record_ids = grouping_artifact.physical_record_ids
        # components → record.components
        from pipeline.schema import Component
        record.components = [
            Component(record_id=c.record_id, role=c.role, confidence=c.confidence)  # type: ignore[arg-type]
            for c in grouping_artifact.components
        ]

    sections_artifact = None
    if "2" in run_stages:
        logger.info("=== Stage 2: Section detection ===")
        sections_artifact = stage2.run(
            ingest_artifact,
            allow_embedding_fallback=not args.no_embedding_fallback,
        )
        stage2.write_artifact(sections_artifact, output_root / "stage2_sections")

        from pipeline.schema import SectionRecord
        record.sections = [
            SectionRecord(
                name=s.name,  # type: ignore[arg-type]
                char_span=s.char_span,
                pages=s.pages,
                confidence=s.confidence,
                status=s.status,  # type: ignore[arg-type]
                detection_method=s.detection_method,  # type: ignore[arg-type]
            )
            for s in sections_artifact.sections
        ]
        logger.info(
            "Sections: %s",
            ", ".join(
                f"{s.name}={s.status}" + (f"@{s.pages}" if s.pages else "")
                for s in sections_artifact.sections
            ),
        )

    if "3a" in run_stages:
        logger.info("=== Stage 3a: NUL-sourced fields (title, year, agency, date) ===")
        try:
            w = stage3a_mets_fields.run(record, ingest_artifact, sections_artifact)
            warnings.extend(w)
        except Exception as exc:
            logger.error("Stage 3a failed: %s", exc, exc_info=True)
            warnings.append(f"stage_3a_error: {exc}")

    if "3b" in run_stages:
        logger.info("=== Stage 3b: EIS type ===")
        if sections_artifact is None:
            warnings.append("stage_3b: sections_artifact missing — cannot detect EIS type")
        else:
            try:
                if llm is None:
                    raise RuntimeError("Stage 3b requires LLMClient (or --dry-run)")
                w = stage3b_eis_type.run(record, ingest_artifact, sections_artifact, llm)
                warnings.extend(w)
            except BudgetExceededError as exc:
                logger.error("Budget exceeded during Stage 3b: %s", exc)
                warnings.append(f"budget_exceeded_at: 3b — {exc}")
                record.extraction_budget_status = "partial_budget_cap"
            except Exception as exc:
                logger.error("Stage 3b failed: %s", exc, exc_info=True)
                warnings.append(f"stage_3b_error: {exc}")

    # -----------------------------------------------------------------------
    # Finalize pipeline metadata
    # -----------------------------------------------------------------------
    duration = time.time() - t_start
    record.pipeline.duration_seconds = round(duration, 2)
    if llm is not None:
        record.pipeline.total_tokens = (
            llm.usage.total_input_tokens + llm.usage.total_output_tokens
        )
        record.pipeline.total_cost_usd = round(llm.usage.total_cost_usd, 6)
    record.pipeline.warnings = warnings

    # Update per-field status reflecting what got populated
    record.per_field_status = {
        "title":       record.title.status,
        "year":        record.year.status,
        "lead_agency": record.agency.lead_agency.status,
        "eis_type":    record.eis_type.status,
        "sections":    "ok" if record.sections else "not_run",
    }

    # -----------------------------------------------------------------------
    # Validate + write
    # -----------------------------------------------------------------------
    out_path = output_root / "stage3_fields" / f"{args.doc_key}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        validated = EISRecord.model_validate(record.model_dump())
        out_json = validated.model_dump_json(indent=2)
        out_path.write_text(out_json, encoding="utf-8")
        logger.info("Schema validation: OK")
    except Exception as exc:
        logger.error("Schema validation FAILED: %s", exc)
        warnings.append(f"schema_validation_error: {exc}")
        # Write the unvalidated record so we can inspect what went wrong
        out_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    # Update token ledger
    if llm is not None and llm.usage.calls > 0:
        token_ledger.append_run(
            ledger_path=output_root / "token_ledger.json",
            doc_id=args.doc_key,
            usage=llm.usage,
            duration_seconds=duration,
            pipeline_version=config.PIPELINE_VERSION,
        )

    logger.info(
        "Done. Output: %s | duration=%.1fs | cost=$%.4f | warnings=%d",
        out_path,
        duration,
        llm.usage.total_cost_usd if llm else 0.0,
        len(warnings),
    )
    if warnings:
        logger.warning("Warnings (%d):", len(warnings))
        for w in warnings:
            logger.warning("  - %s", w)


def _stage_ran(stage_key: str, run_stages: list[str]) -> bool:
    """Map config.STAGE_VERSIONS keys (stage1, stage2, ...) to STAGE_ORDER tokens."""
    mapping = {
        "stage1": "1",
        "stage1_5": "1.5",
        "stage2": "2",
        "stage3a": "3a",
        "stage3b": "3b",
    }
    return mapping.get(stage_key, stage_key) in run_stages


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(1)
