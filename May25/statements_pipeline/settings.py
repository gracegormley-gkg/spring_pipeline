"""
statements_pipeline settings.

Reuses segment_a/ (chunk, pages, llm, nul, config) and people_pipeline/
(extract, verify, merge) via sys.path. Local modules win on name collisions
because we APPEND (not insert) the external paths.

Constants here intentionally mirror people_pipeline/settings.py for the keys
that people_pipeline's modules read (STANCES, KINDS, EXTRACT_PARALLEL,
EXTRACT_CHAR_CAP, OUTPUT_DIR). When `import settings` runs from the imported
people_pipeline modules, Python returns THIS module — so those names need to
resolve correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
PEOPLE_DIR = OUTPUT_DIR / "people"            # per-doc folder of per-person JSONs + index.json
RAW_EXTRACT_DIR = OUTPUT_DIR / "raw_extract"  # per-chunk extractions (checkpointed)
RUN_SUMMARY_PATH = OUTPUT_DIR / "run_summary.json"

# Reuse segment_a's calibration selection.
SEGMENT_A_DIR = ROOT.parent / "segment_a"
PEOPLE_PIPELINE_DIR = ROOT.parent / "people_pipeline"
SEGMENT_A_SELECTION_PATH = SEGMENT_A_DIR / "output" / "selection.json"

# Make segment_a's and people_pipeline's modules importable. Local modules win
# on name collisions because we APPEND (not insert).
for _ext in (SEGMENT_A_DIR, PEOPLE_PIPELINE_DIR):
    _p = str(_ext)
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.append(_p)

# --- Parallelism / caps ------------------------------------------------------
EXTRACT_PARALLEL = 4
STATEMENT_PARALLEL = 4
CRITIC_PARALLEL = 4

EXTRACT_CHAR_CAP = 80_000

# Pages of margin on each side of the entity's evidence pages when building
# the doc-text window we feed to find_statement.
WINDOW_MARGIN_PAGES = 10

# Cap on the doc-text window we send to Sonnet for statement finding.
WINDOW_CHAR_CAP = 60_000

# Cap on the extracted statement.text length.
STATEMENT_CHAR_CAP = 40_000

# --- Closed stance vocabulary ------------------------------------------------
STANCES = ("in_favor", "opposed", "conditional", "neutral")
KINDS = (
    "individual",
    "official",
    "organization",
    "agency",
    "tribe",
    "government",
    "other",
)

# Forms the statement-finder may report.
STATEMENT_FORMS = (
    "letter",                # contiguous comment letter signed by/attributed to the entity
    "testimony",             # spoken statement at a hearing / town hall
    "written_comment",       # other contiguous written statement (memo, position paper)
    "narrator_paraphrase",   # narrator describes the position; no contiguous statement
    "sectional",             # entity appears in a list/table under a stance heading
    "none",                  # no statement, no clear paraphrase block
)

# --- Pricing (USD per 1M tokens) — same defaults as people_pipeline ----------
PRICES_USD_PER_M = {
    "sonnet-4": {"input": 3.00, "output": 15.00},
    "opus-4":   {"input": 15.00, "output": 75.00},
    "haiku":    {"input": 1.00, "output": 5.00},
}


def price_for_model(model: str) -> dict:
    m = (model or "").lower()
    for key, rates in PRICES_USD_PER_M.items():
        if key in m:
            return rates
    return {"input": 0.0, "output": 0.0}


def cost_for_usage(usage: dict) -> float:
    rates = price_for_model(usage.get("model", ""))
    inp = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    out = usage.get("output_tokens", 0)
    return (inp * rates["input"] + cache_read * rates["input"] * 0.1 + out * rates["output"]) / 1_000_000


def aggregate_usages(usages: list[dict]) -> dict:
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
