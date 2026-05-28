"""
people_pipeline settings.

Named `settings.py` (not `config.py`) on purpose: segment_a/config.py is added
to sys.path at runtime, and Python caches modules by name. Using a different
name here means our local pipeline constants and segment_a's constants can
coexist without one shadowing the other.

Paths point at outputs scoped to this pipeline only. Scope settings control
which docs we run on (defaults to the same 20 docs as segment_a's calibration).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
ENTRIES_DIR = OUTPUT_DIR / "entries"          # final merged + critiqued JSON, one per doc
RAW_EXTRACT_DIR = OUTPUT_DIR / "raw_extract"  # per-chunk extractions (checkpointed)
CRITIC_DIR = OUTPUT_DIR / "critic"            # raw critic outputs (kept for debugging)
RUN_SUMMARY_PATH = OUTPUT_DIR / "run_summary.json"

# Reuse segment_a's selection so grading lines up with segment_a's results.
SEGMENT_A_DIR = ROOT.parent / "segment_a"
SEGMENT_A_SELECTION_PATH = SEGMENT_A_DIR / "output" / "selection.json"

# Make segment_a's modules importable (chunk, llm, nul, config, selection).
# IMPORTANT: append (not insert) so local modules win when names collide.
# Both directories have a `critic.py` and a `run.py` — we want the local ones.
_seg_a = str(SEGMENT_A_DIR)
if _seg_a in sys.path:
    sys.path.remove(_seg_a)
sys.path.append(_seg_a)

# --- Scope -------------------------------------------------------------------
# How many chunks per doc to feed the extractor in parallel. We don't cap by
# default — exhaustiveness is the goal — but parallelism is bounded.
EXTRACT_PARALLEL = 4
CRITIC_PARALLEL = 4

# Cap chunk text fed to Sonnet (in chars) to keep one extract call cheap.
# A 50-page chunk is ~125k chars; Sonnet handles it but the prompt is denser
# than necessary. 80k keeps headroom for the system prompt + JSON output.
EXTRACT_CHAR_CAP = 80_000

# --- Closed stance vocabulary ------------------------------------------------
# Per the design: drop entries whose stance is not one of these.
STANCES = ("in_favor", "opposed", "conditional", "neutral")

# --- Entity kinds (open-ish; the model picks one) ----------------------------
KINDS = (
    "individual",      # named person
    "official",        # named person acting for an org/agency
    "organization",    # advocacy group, NGO, business
    "agency",          # federal/state/local government agency
    "tribe",           # tribal nation or indigenous community
    "government",      # state/county/municipal government as a body
    "other",
)

# --- Pricing (USD per 1M tokens) ---------------------------------------------
# Approximate Bedrock on-demand prices for Claude 4-tier models. These are
# defaults so we can ESTIMATE cost from usage logs; verify against current AWS
# pricing for invoicing.
#
# Match by substring of the model id. First matching key wins.
PRICES_USD_PER_M = {
    # Claude Sonnet 4 / 4.5 / 4.6 family on Bedrock
    "sonnet-4": {"input": 3.00, "output": 15.00},
    # Claude Opus 4 / 4.7 family on Bedrock
    "opus-4":   {"input": 15.00, "output": 75.00},
    # Claude Haiku 4-5 (when accessible)
    "haiku":    {"input": 1.00, "output": 5.00},
}


def price_for_model(model: str) -> dict:
    """Look up per-1M-token rates for a model id. Returns zeros if unknown."""
    m = (model or "").lower()
    for key, rates in PRICES_USD_PER_M.items():
        if key in m:
            return rates
    return {"input": 0.0, "output": 0.0}


def cost_for_usage(usage: dict) -> float:
    """USD cost estimate for one usage dict (the shape returned by llm.call_with_usage)."""
    rates = price_for_model(usage.get("model", ""))
    inp = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    out = usage.get("output_tokens", 0)
    # Cache reads are typically billed at 10% of input rate; treat conservatively.
    return (inp * rates["input"] + cache_read * rates["input"] * 0.1 + out * rates["output"]) / 1_000_000


def aggregate_usages(usages: list[dict]) -> dict:
    """Sum a list of usage dicts. Per-model breakdown + total."""
    by_model: dict[str, dict] = {}
    for u in usages:
        if not u:
            continue
        m = u.get("model", "?")
        agg = by_model.setdefault(m, {
            "model": m,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cost_usd": 0.0,
        })
        agg["calls"] += 1
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            agg[k] += u.get(k, 0)
        agg["cost_usd"] += cost_for_usage(u)
    total = {
        "calls": sum(v["calls"] for v in by_model.values()),
        "input_tokens": sum(v["input_tokens"] for v in by_model.values()),
        "output_tokens": sum(v["output_tokens"] for v in by_model.values()),
        "cache_creation_input_tokens": sum(v["cache_creation_input_tokens"] for v in by_model.values()),
        "cache_read_input_tokens": sum(v["cache_read_input_tokens"] for v in by_model.values()),
        "cost_usd": round(sum(v["cost_usd"] for v in by_model.values()), 4),
    }
    return {
        "by_model": [
            {**v, "cost_usd": round(v["cost_usd"], 4)} for v in by_model.values()
        ],
        "total": total,
    }
