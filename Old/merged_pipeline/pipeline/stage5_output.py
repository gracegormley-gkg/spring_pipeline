"""
Stage 5 — Output writer + orchestration.

Wires Stage 1 -> 1.5 -> 2 -> 3a -> 3b -> 3c -> 3g -> 3e -> 3d -> 3f -> 4
into a single `run_doc()` entrypoint, then writes the final EISRecord JSON
to the requested output path.

Per synthesis_plan §Build order step 11.

Idempotent stage caching (synthesis_plan §Caching) is NOT yet wired here;
each run re-executes every stage. The cache module is the natural next
addition; it's listed in the §Critical files §New code to write list.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from . import config
from . import grouping as grouping_mod
from . import ingest as ingest_mod
from . import sections as sections_mod
from . import stage3a_mets_fields
from . import stage3b_eis_type
from . import stage3c_summary
from . import stage3d_location
from . import stage3e_alternatives
from . import stage3f_stakeholders
from . import stage3g_themes
from . import stage4_critic
from . import token_ledger
from .nul_client import NULClient
from .schema import (
    Component,
    EISRecord,
    PipelineMeta,
)

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def run_doc(
    doc_key: str,
    *,
    docs_json_path: str | Path = config.DEFAULT_DOCS_JSON,
    nul_cache_dir: str | Path | None = None,
    llm: "LLMClient | None" = None,
    output_path: str | Path | None = None,
    write_ledger: bool = True,
) -> EISRecord:
    """Run all stages on a single doc_key. Returns the final EISRecord.

    If `output_path` is provided, also writes the record JSON to that path.

    `llm=None` -> all LLM-dependent fields skip gracefully and degrade their
    statuses ("needs_review" or stage-specific abstention codes). The
    pipeline still produces a valid EISRecord JSON.
    """
    t_start = time.monotonic()
    all_warnings: list[str] = []

    # ---- Stage 1: Ingest ----
    nul_client = NULClient(cache_dir=nul_cache_dir)
    ingest_artifact = ingest_mod.run(docs_json_path, doc_key, nul_client)
    logger.info("Stage 1 done: %d chars, %d pages",
                len(ingest_artifact.raw_text), len(ingest_artifact.pages))

    # ---- Stage 1.5: Grouping (passthrough in v1) ----
    grouping_artifact = grouping_mod.run(ingest_artifact)

    # ---- Stage 2: Section detection cascade ----
    sections_artifact = sections_mod.run(
        ingest_artifact, llm_client=llm,
        allow_ai_toc=llm is not None,
        allow_embedding_fallback=True,
    )
    logger.info("Stage 2 done: %d sections detected",
                sum(1 for s in sections_artifact.sections if s.status == "ok"))

    # ---- Build the EISRecord skeleton ----
    record = EISRecord(
        publication_id=grouping_artifact.publication_id,
        physical_record_ids=grouping_artifact.physical_record_ids,
        components=[
            Component(record_id=c.record_id, role=c.role, confidence=c.confidence)  # type: ignore[arg-type]
            for c in grouping_artifact.components
        ],
        sections=[s.to_schema_record() for s in sections_artifact.sections],
        is_supplemental=grouping_artifact.is_supplemental,
    )

    # ---- Stage 3a: METS fields ----
    all_warnings.extend(
        stage3a_mets_fields.run(record, ingest_artifact, sections_artifact, llm=llm)
    )

    # ---- Stage 3b: EIS type ----
    all_warnings.extend(
        stage3b_eis_type.run(record, ingest_artifact, sections_artifact, llm=llm)
    )

    # ---- Stage 3c: Summary (gates 3g) ----
    all_warnings.extend(
        stage3c_summary.run(record, ingest_artifact, sections_artifact, llm=llm)
    )

    # ---- Stage 3g: Themes (downstream of 3c) ----
    all_warnings.extend(stage3g_themes.run(record, llm=llm))

    # ---- Stage 3e: Alternatives ----
    all_warnings.extend(
        stage3e_alternatives.run(record, ingest_artifact, sections_artifact, llm=llm)
    )

    # ---- Stage 3d: Location ----
    all_warnings.extend(
        stage3d_location.run(record, ingest_artifact, sections_artifact, llm=llm)
    )

    # ---- Stage 3f: Stakeholders ----
    all_warnings.extend(
        stage3f_stakeholders.run(record, ingest_artifact, sections_artifact, llm=llm)
    )

    # ---- Stage 4: Critic (deterministic gates) ----
    all_warnings.extend(stage4_critic.run(record, ingest_artifact))

    # ---- Pipeline metadata ----
    duration = time.monotonic() - t_start
    usage_total_tokens = 0
    usage_total_cost = 0.0
    by_model_ids: dict[str, str] = {}
    if llm is not None:
        usage_total_tokens = llm.usage.total_input_tokens + llm.usage.total_output_tokens
        usage_total_cost = round(llm.usage.total_cost_usd, 6)
        by_model_ids = {k: k for k in llm.usage.by_model.keys()}

    record.pipeline = PipelineMeta(
        pipeline_version=config.PIPELINE_VERSION,
        stage_versions=dict(config.STAGE_VERSIONS),
        model_ids=by_model_ids if by_model_ids else dict(config.MODELS),
        gazetteer_versions={},  # v1: no gazetteer wired
        calibration_model_id=None,
        total_tokens=usage_total_tokens,
        total_cost_usd=usage_total_cost,
        duration_seconds=round(duration, 2),
        warnings=all_warnings,
    )

    # ---- Write output JSON ----
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use Pydantic's model_dump_json for stable serialization
        output_path.write_text(
            record.model_dump_json(indent=2, exclude_none=False),
            encoding="utf-8",
        )
        logger.info("Wrote output: %s", output_path)

    # ---- Token ledger (append run) ----
    if write_ledger and llm is not None:
        try:
            token_ledger.append_run(
                ledger_path=config.DEFAULT_TOKEN_LEDGER_PATH,
                doc_id=doc_key,
                usage=llm.usage,
                duration_seconds=duration,
                pipeline_version=config.PIPELINE_VERSION,
            )
        except Exception as exc:
            logger.warning("Token ledger append failed: %s", exc)

    logger.info(
        "run_doc(%s) complete: %.2fs, %d warnings, review_routing=%s",
        doc_key, duration, len(all_warnings), record.validation.review_routing,
    )
    return record
