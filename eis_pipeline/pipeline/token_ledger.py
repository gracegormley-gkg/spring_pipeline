"""
Persistent token usage ledger.

After each pipeline run, append a record of the run's LLM usage to a JSON file.
The ledger tracks tokens per model + cost + duration so we can:

  - Audit total environmental impact of the project (tokens ≈ compute ≈ energy)
  - Diagnose which models drive cost over time
  - Spot regressions when prompt changes balloon token counts

The ledger file is human-readable JSON. Concurrent writes are NOT safe — this
assumes one pipeline run at a time. (Adding a lockfile would be straightforward
if we ever parallelize.)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_client import UsageAccumulator

logger = logging.getLogger(__name__)


def append_run(
    ledger_path: str | Path,
    doc_id: str,
    usage: "UsageAccumulator",
    duration_seconds: float,
    pipeline_version: str = "v1.0",
) -> None:
    """
    Append this run's usage to the ledger. Creates the file if it doesn't exist.
    Recomputes lifetime totals on every write so they're always in sync.
    """
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    by_model = {
        model: {
            "input_tokens": mu.input_tokens,
            "output_tokens": mu.output_tokens,
            "calls": mu.calls,
            "cost_usd": round(mu.cost_usd, 6),
        }
        for model, mu in usage.by_model.items()
    }

    run_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "doc_id": doc_id,
        "pipeline_version": pipeline_version,
        "duration_seconds": round(duration_seconds, 2),
        "total_input_tokens": usage.total_input_tokens,
        "total_output_tokens": usage.total_output_tokens,
        "total_tokens": usage.total_input_tokens + usage.total_output_tokens,
        "total_cost_usd": round(usage.total_cost_usd, 6),
        "calls": usage.calls,
        "by_model": by_model,
    }

    # Load existing ledger (or start fresh)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "runs" not in data:
                logger.warning("Ledger at %s has unexpected shape — starting fresh", path)
                data = {"runs": []}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read ledger %s (%s) — starting fresh", path, exc)
            data = {"runs": []}
    else:
        data = {"runs": []}

    data["runs"].append(run_entry)
    data["lifetime_totals"] = _compute_lifetime_totals(data["runs"])

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(
        "Token ledger updated: %s | run_total=%d tokens ($%.4f) | lifetime=%d runs (%d tokens, $%.2f)",
        path,
        run_entry["total_tokens"],
        run_entry["total_cost_usd"],
        data["lifetime_totals"]["runs"],
        data["lifetime_totals"]["total_tokens"],
        data["lifetime_totals"]["total_cost_usd"],
    )


def _compute_lifetime_totals(runs: list[dict]) -> dict:
    totals = {
        "runs": len(runs),
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "calls": 0,
        "by_model": {},
    }
    for r in runs:
        totals["total_input_tokens"] += r.get("total_input_tokens", 0)
        totals["total_output_tokens"] += r.get("total_output_tokens", 0)
        totals["total_tokens"] += r.get("total_tokens", 0)
        totals["total_cost_usd"] += r.get("total_cost_usd", 0.0)
        totals["calls"] += r.get("calls", 0)

        for model, mu in (r.get("by_model") or {}).items():
            agg = totals["by_model"].setdefault(model, {
                "input_tokens": 0,
                "output_tokens": 0,
                "calls": 0,
                "cost_usd": 0.0,
            })
            agg["input_tokens"] += mu.get("input_tokens", 0)
            agg["output_tokens"] += mu.get("output_tokens", 0)
            agg["calls"] += mu.get("calls", 0)
            agg["cost_usd"] += mu.get("cost_usd", 0.0)

    totals["total_cost_usd"] = round(totals["total_cost_usd"], 6)
    for mu in totals["by_model"].values():
        mu["cost_usd"] = round(mu["cost_usd"], 6)
    return totals
