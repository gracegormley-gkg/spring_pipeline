"""
Central configuration for v1_multiagent_pipeline.

Locks model IDs, controlled vocabularies, regex patterns, and thresholds.
Source: v1_multiagent_plan.md + inter_agent_plan.md §1.2.
"""

from __future__ import annotations

import datetime
import re

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
# Model IDs (locked per build brief §9)
# ---------------------------------------------------------------------------
MODELS: dict[str, str] = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",  # retries only
}

# ---------------------------------------------------------------------------
# I/O paths
# ---------------------------------------------------------------------------
DEFAULT_DOCS_JSON = "/Users/gracegormley/Desktop/Y2/Q2/Knight Lab/docs_with_digits.json"
CHARS_PER_FAKE_PAGE = 2500

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
# Ported from existing eis_pipeline/pipeline/config.py
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
# Theme taxonomy (per inter_agent_plan.md §1.2 Stage 3g)
# 13 primary themes; per-primary subthemes.
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
# Small, illustrative; expand after first batch.
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
# Used by Stage 4 to decide if a missing section is fatal vs. acceptable.
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
EIS_TYPE_REGEX = re.compile(
    r"\b(Draft|Final|Supplemental(?:\s+(?:Draft|Final))?)"
    r"\s+Environmental\s+Impact\s+Statement\b",
    re.IGNORECASE,
)
VALID_EIS_TYPES = {"Draft", "Final", "Supplemental", "ROD", "NOI", "Unlabelled"}

# ---------------------------------------------------------------------------
# Cost cap defaults
# ---------------------------------------------------------------------------
DEFAULT_BUDGET_USD = 1.00  # per-doc hard cap (build brief §1)

# ---------------------------------------------------------------------------
# Quote / verbatim
# ---------------------------------------------------------------------------
QUOTE_MIN_WORDS = 8
QUOTE_MAX_WORDS = 40
