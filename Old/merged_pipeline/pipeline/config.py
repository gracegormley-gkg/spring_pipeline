"""
Central configuration for the merged EIS pipeline.

Locks model IDs, controlled vocabularies, regex patterns, and thresholds.

Source: ported verbatim from v1_multiagent_pipeline/pipeline/config.py with
two deltas keyed to barbie_claude/synthesis_plan.md:

  1. DEFAULT_BUDGET_USD: 1.00 -> 2.00
     The synthesis plan §Per-doc cost cap raised the default for the full-NUL
     corpus (vs. v1_multi's USFS-narrowed $1.00) because OCR variance and
     stakeholder-heavy comment appendices push p99 cost higher. Treated as a
     measurement, not a target; configurable per run via --budget-usd.

  2. DEFAULT_NUL_CACHE_DIR: new constant, points at output/nul_cache/ relative
     to the merged_pipeline/ root. Lets nul_client construct paths without
     callers having to know the convention.

Pipeline + stage versions are kept in sync with PipelineMeta defaults in
pipeline/schema.py. Bump these when stage logic or prompts change so the
cache key (per synthesis_plan §Caching) invalidates.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Pipeline version
# ---------------------------------------------------------------------------
PIPELINE_VERSION = "v1.0.0"
STAGE_VERSIONS: dict[str, str] = {
    "stage1": "1.0.0",
    "stage1_5": "1.0.0",
    "stage2": "1.0.0",
    "stage3a": "1.0.0",
    "stage3b": "1.0.0",
    # 3c–4 added by future groups
}

# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------
# "anthropic" -> direct Anthropic API (env: ANTHROPIC_API_KEY)
# "bedrock"   -> Amazon Bedrock via anthropic.AnthropicBedrock (env: AWS_REGION
#                + standard AWS credentials chain or AWS_BEARER_TOKEN_BEDROCK)
# "auto"      -> pick bedrock if AWS creds present, else anthropic
import os as _os
LLM_BACKEND: str = _os.environ.get("LLM_BACKEND", "auto")

# ---------------------------------------------------------------------------
# Model IDs — direct Anthropic
# ---------------------------------------------------------------------------
ANTHROPIC_MODELS: dict[str, str] = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",  # retries only
}

# ---------------------------------------------------------------------------
# Model IDs — Bedrock
# ---------------------------------------------------------------------------
# Bedrock requires inference profile IDs (not raw model IDs) for these
# models — raw IDs like 'anthropic.claude-sonnet-4-6' fail with
# "Invocation of model ID ... with on-demand throughput isn't supported.
#  Retry your request with the ID or ARN of an inference profile that
#  contains this model."
# In us-east-1 / us-west-2, the cross-region inference profile prefix is
# 'us.'. Other regions use different prefixes:
#   us-east-1, us-west-2  -> us.<model>
#   eu-* (Frankfurt etc.) -> eu.<model>
#   ap-* (Tokyo etc.)     -> apac.<model>
# Override via env BEDROCK_HAIKU_MODEL / BEDROCK_SONNET_MODEL /
# BEDROCK_OPUS_MODEL to use a region-specific profile or a custom ARN.
BEDROCK_MODELS: dict[str, str] = {
    "haiku":  _os.environ.get(
        "BEDROCK_HAIKU_MODEL",  "us.anthropic.claude-haiku-4-5-20251001"
    ),
    "sonnet": _os.environ.get(
        "BEDROCK_SONNET_MODEL", "us.anthropic.claude-sonnet-4-6"
    ),
    "opus":   _os.environ.get(
        "BEDROCK_OPUS_MODEL",   "us.anthropic.claude-opus-4-7"
    ),
}

# Resolved model dict — what the rest of the pipeline reads.
# (Selection happens at LLMClient construction time based on the actual
# backend used; this dict is the default-best-guess and can be overridden
# in callers via llm.models["haiku"] etc.)
def _resolve_models(backend: str) -> dict[str, str]:
    if backend == "bedrock":
        return dict(BEDROCK_MODELS)
    return dict(ANTHROPIC_MODELS)

# Default MODELS dict reads the static LLM_BACKEND env. LLMClient will
# re-resolve at construction time if it ends up picking a different backend
# under "auto".
MODELS: dict[str, str] = _resolve_models(LLM_BACKEND if LLM_BACKEND != "auto" else "anthropic")

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
DEFAULT_DOCS_JSON = "/Users/gracegormley/Desktop/Y2/Q2/Knight Lab/docs_with_digits.json"
CHARS_PER_FAKE_PAGE = 2500

# Resolved relative to the merged_pipeline/ project root (two parents up
# from this file: merged_pipeline/pipeline/config.py -> merged_pipeline/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output"
DEFAULT_NUL_CACHE_DIR = DEFAULT_OUTPUT_DIR / "nul_cache"
DEFAULT_STAGE_CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
DEFAULT_TOKEN_LEDGER_PATH = DEFAULT_OUTPUT_DIR / "token_ledger.json"

# ---------------------------------------------------------------------------
# NUL API
# ---------------------------------------------------------------------------
NUL_API_BASE = "https://api.dc.library.northwestern.edu/api/v2"
NUL_REQUEST_TIMEOUT_S = 10
NUL_POLITENESS_DELAY_S = 0.3

# ---------------------------------------------------------------------------
# Date constraints
# ---------------------------------------------------------------------------
NEPA_YEAR = 1969
MAX_YEAR = datetime.date.today().year

# ---------------------------------------------------------------------------
# Agency vocabulary (canonical → abbreviations + variants)
# ---------------------------------------------------------------------------
AGENCY_VOCAB: dict[str, dict] = {
    "USFS": {
        "canonical": "U.S. Forest Service",
        "variants": [
            "United States Forest Service", "Forest Service", "US Forest Service",
            "USDA Forest Service", "U.S.D.A. Forest Service",
        ],
    },
    "BLM": {
        "canonical": "Bureau of Land Management",
        "variants": ["Bureau of Land Mgmt", "BLM Field Office"],
    },
    "USACE": {
        "canonical": "U.S. Army Corps of Engineers",
        "variants": [
            "Army Corps of Engineers", "Corps of Engineers",
            "United States Army Corps of Engineers", "COE",
        ],
    },
    "EPA": {
        "canonical": "Environmental Protection Agency",
        "variants": ["U.S. Environmental Protection Agency", "US EPA"],
    },
    "NPS": {
        "canonical": "National Park Service",
        "variants": ["U.S. National Park Service", "US National Park Service"],
    },
    "FWS": {
        "canonical": "U.S. Fish and Wildlife Service",
        "variants": [
            "Fish and Wildlife Service", "US Fish and Wildlife Service",
            "United States Fish and Wildlife Service", "USFWS",
        ],
    },
    "NASA": {
        "canonical": "National Aeronautics and Space Administration",
        "variants": ["NASA"],
    },
    "DOE": {
        "canonical": "U.S. Department of Energy",
        "variants": ["Department of Energy", "US DOE"],
    },
    "DOT": {
        "canonical": "U.S. Department of Transportation",
        "variants": ["Department of Transportation"],
    },
    "FHWA": {
        "canonical": "Federal Highway Administration",
        "variants": ["FHWA"],
    },
    "FERC": {
        "canonical": "Federal Energy Regulatory Commission",
        "variants": [],
    },
    "BOEM": {
        "canonical": "Bureau of Ocean Energy Management",
        "variants": ["Minerals Management Service", "MMS"],
    },
    "BIA": {
        "canonical": "Bureau of Indian Affairs",
        "variants": ["U.S. Bureau of Indian Affairs"],
    },
    "BOR": {
        "canonical": "Bureau of Reclamation",
        "variants": ["U.S. Bureau of Reclamation", "USBR", "Reclamation"],
    },
    "FAA": {
        "canonical": "Federal Aviation Administration",
        "variants": ["U.S. Federal Aviation Administration"],
    },
    "NRC": {
        "canonical": "Nuclear Regulatory Commission",
        "variants": ["U.S. Nuclear Regulatory Commission", "Atomic Energy Commission", "AEC"],
    },
    "DOD": {
        "canonical": "U.S. Department of Defense",
        "variants": [
            "Department of Defense", "DoD",
            "Department of the Army", "Department of the Navy",
        ],
    },
    "USDA": {
        "canonical": "U.S. Department of Agriculture",
        "variants": ["Department of Agriculture"],
    },
    "DOI": {
        "canonical": "U.S. Department of the Interior",
        "variants": ["Department of the Interior", "Department of Interior"],
    },
    "FRA": {
        "canonical": "Federal Railroad Administration",
        "variants": [],
    },
    "HUD": {
        "canonical": "U.S. Department of Housing and Urban Development",
        "variants": ["Housing and Urban Development"],
    },
}

# Build reverse lookup: any string → abbreviation key
_AGENCY_FLAT: dict[str, str] = {}
for _abbr, _data in AGENCY_VOCAB.items():
    _AGENCY_FLAT[_abbr.lower()] = _abbr
    _AGENCY_FLAT[_data["canonical"].lower()] = _abbr
    for _var in _data["variants"]:
        _AGENCY_FLAT[_var.lower()] = _abbr
AGENCY_FLAT = _AGENCY_FLAT


def lookup_agency(text: str) -> str | None:
    """Return canonical abbreviation key (e.g. 'USFS') or None."""
    if not text:
        return None
    return AGENCY_FLAT.get(text.strip().lower())


# ---------------------------------------------------------------------------
# Theme taxonomy (per inter_agent_plan §1.2 Stage 3g; 13 primary + scoped subs)
# ---------------------------------------------------------------------------
THEMES: dict[str, list[str]] = {
    "energy_infrastructure": [
        "nuclear_power", "hydroelectric", "wind", "solar",
        "oil_and_gas", "electric_transmission", "pipelines",
    ],
    "transportation": [
        "highway", "rail", "transit", "aviation",
        "port_and_harbor", "bridge",
    ],
    "land_management": [
        "public_lands_planning", "grazing", "recreation",
        "timber", "fire_management", "designation",
    ],
    "water_resources": [
        "dam_and_reservoir", "water_supply", "irrigation",
        "flood_control", "wetlands", "watershed_restoration",
    ],
    "defense_and_military": [
        "base_realignment", "training_range", "weapons_testing", "munitions",
    ],
    "urban_development": [
        "housing", "redevelopment", "commercial", "mixed_use",
    ],
    "mining_and_extraction": [
        "coal", "hardrock_mining", "oil_and_gas_leasing", "sand_and_gravel",
    ],
    "agriculture_and_forestry": [
        "forest_plan", "timber_harvest", "rangeland", "pest_management",
    ],
    "wildlife_and_habitat": [
        "endangered_species", "habitat_restoration", "fisheries", "predator_management",
    ],
    "cultural_heritage": [
        "historic_preservation", "archaeological", "tribal_sacred_sites",
    ],
    "waste_and_remediation": [
        "superfund", "landfill", "hazardous_waste", "cleanup",
    ],
    "aerospace_and_space_exploration": [
        "launch_facility", "satellite", "crewed_mission",
        "research_program", "planetary_science",
    ],
    "other": [],
}

ALL_PRIMARY_THEMES: list[str] = list(THEMES.keys())
ALL_SUBTHEMES: list[str] = [s for subs in THEMES.values() for s in subs]

# Per-primary keyword set for Stage 4a self-consistency check (themes ↔ summary).
THEME_KEYWORDS: dict[str, list[str]] = {
    "energy_infrastructure": ["energy", "power", "transmission", "pipeline", "wind", "solar", "nuclear"],
    "transportation":        ["highway", "road", "transit", "rail", "bridge", "corridor", "interstate"],
    "land_management":       ["forest", "grazing", "recreation", "timber", "wilderness", "public land"],
    "water_resources":       ["dam", "reservoir", "watershed", "wetland", "flood", "irrigation", "water supply"],
    "defense_and_military":  ["military", "training", "range", "munitions", "base", "armed forces"],
    "urban_development":     ["housing", "redevelopment", "commercial", "downtown", "mixed-use"],
    "mining_and_extraction": ["mine", "coal", "lease", "ore", "extraction", "drilling"],
    "agriculture_and_forestry": ["agriculture", "forest", "timber", "rangeland", "pesticide"],
    "wildlife_and_habitat":  ["wildlife", "habitat", "species", "fishery", "endangered"],
    "cultural_heritage":     ["historic", "archaeological", "cultural", "tribal", "preservation"],
    "waste_and_remediation": ["superfund", "landfill", "hazardous", "cleanup", "remediation"],
    "aerospace_and_space_exploration": ["space", "launch", "satellite", "spacecraft", "mission"],
    "other":                 [],
}

# ---------------------------------------------------------------------------
# CEQ section taxonomy (Stage 2)
# ---------------------------------------------------------------------------
CEQ_SECTIONS: list[str] = [
    "cover",
    "summary",
    "purpose_and_need",
    "alternatives",
    "affected_environment",
    "environmental_consequences",
    "public_comments",
    "response_to_comments",
    "rod",
]

# Section regex patterns (multiline, case-insensitive). Note: avoid crossing
# newlines inside a "heading" to dodge prose matches.
SECTION_PATTERNS: dict[str, list[re.Pattern]] = {
    "summary": [
        re.compile(r"^\s*(SUMMARY|EXECUTIVE\s+SUMMARY|ABSTRACT)\s*$",
                   re.MULTILINE | re.IGNORECASE),
    ],
    "purpose_and_need": [
        re.compile(r"^\s*(PURPOSE\s+(AND|&)\s+NEED|NEED\s+FOR\s+(THE\s+)?(ACTION|PROJECT))\s*$",
                   re.MULTILINE | re.IGNORECASE),
    ],
    "alternatives": [
        re.compile(
            r"^\s*(ALTERNATIVES?(\s+(CONSIDERED|ANALYZED|TO\s+THE\s+PROPOSED\s+ACTION))?"
            r"|DESCRIPTION\s+OF\s+ALTERNATIVES)\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ],
    "affected_environment": [
        re.compile(r"^\s*(AFFECTED\s+ENVIRONMENT|EXISTING\s+CONDITIONS)\s*$",
                   re.MULTILINE | re.IGNORECASE),
    ],
    "environmental_consequences": [
        re.compile(r"^\s*(ENVIRONMENTAL\s+(CONSEQUENCES|EFFECTS|IMPACTS))\s*$",
                   re.MULTILINE | re.IGNORECASE),
    ],
    "public_comments": [
        re.compile(
            r"^\s*(PUBLIC\s+(COMMENTS?|INVOLVEMENT|PARTICIPATION)"
            r"|COMMENTS\s+ON\s+THE\s+DRAFT)\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ],
    "response_to_comments": [
        re.compile(r"^\s*(RESPONSE\s+TO\s+COMMENTS|COMMENTS?\s+AND\s+RESPONSES)\s*$",
                   re.MULTILINE | re.IGNORECASE),
    ],
    "rod": [
        re.compile(r"^\s*(RECORD\s+OF\s+DECISION|ROD)\s*$",
                   re.MULTILINE | re.IGNORECASE),
    ],
}

# Which downstream stages need which sections (for Stage 2 abstention semantics).
# Per synthesis_plan §Key design decision 0: missing sections are NOT a hard gate;
# downstream fields fall back to keyword search / first-N-chunks. This map is
# advisory metadata for Stage 4 routing, not a kill-switch.
SECTION_REQUIRED_BY: dict[str, list[str]] = {
    "summary":              ["stage3c_summary", "stage3g_themes"],
    "purpose_and_need":     ["stage3d_location"],
    "alternatives":         ["stage3e_alternatives"],
    "affected_environment": ["stage3d_location"],
    "public_comments":      ["stage3f_stakeholders"],
    "response_to_comments": ["stage3f_stakeholders"],
}

# Reject section-name regex matches that look like legal citations or addresses.
SECTION_REJECT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\d{5}(?:-\d{4})?"),                          # ZIP code
    re.compile(r"\bSection\s+\d+\(", re.IGNORECASE),          # "Section 4(f)" legal citation
    re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+\s+(St|Ave|Blvd|Rd)"), # street address
]

# Embedding fallback (Stage 2)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_THRESHOLD = 0.55
EMBEDDING_MARGIN = 0.10  # best must beat second-best by this much

# Canonical descriptors for embedding fallback.
SECTION_DESCRIPTORS: dict[str, str] = {
    "summary":              "executive summary of the project, what it does, why, what the impacts are",
    "purpose_and_need":     "the reason this project is needed, the goals it serves, what problem it addresses",
    "alternatives":         "alternatives considered including the no-action alternative and other options the agency evaluated",
    "public_comments":      "letters and comments from members of the public, agencies, and tribes submitted during the public review",
    "response_to_comments": "the agency's written responses to public comments",
}

# ---------------------------------------------------------------------------
# EIS type detection (Stage 3b)
# ---------------------------------------------------------------------------
# Note: "Impact" is optional. NEPA-era 1970s docs (e.g. the Castaic-Haskell
# fixture, p1074_35556035057348) use "FINAL ENVIRONMENTAL STATEMENT" without
# "Impact"; modern docs use "Final Environmental Impact Statement". The
# leading qualifier (Draft/Final/Supplemental) is mandatory, so this stays
# tight against false positives like "Programmatic Environmental Statement".
EIS_TYPE_REGEX = re.compile(
    r"\b(Draft|Final|Supplemental(?:\s+(?:Draft|Final))?)"
    r"\s+Environmental(?:\s+Impact)?\s+Statement\b",
    re.IGNORECASE,
)
VALID_EIS_TYPES = {"Draft", "Final", "Supplemental", "ROD", "NOI", "Unlabelled"}

# ---------------------------------------------------------------------------
# Cost cap defaults
# ---------------------------------------------------------------------------
# synthesis_plan §Per-doc cost cap: full-NUL default is $2.00/doc (vs. v1_multi's
# USFS-narrowed $1.00). Treated as measurement, not a target. Configurable per
# run via run.py --budget-usd.
DEFAULT_BUDGET_USD = 2.00

# ---------------------------------------------------------------------------
# Quote / verbatim
# ---------------------------------------------------------------------------
QUOTE_MIN_WORDS = 8
QUOTE_MAX_WORDS = 40
