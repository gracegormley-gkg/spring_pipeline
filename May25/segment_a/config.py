"""
Segment A: Calibration Run config.

Source data: per-page JSON files at PAGES_DATA_DIR/<doc_id>/page_NNNN.json
(one file per page; each carries {page_number, text, ...}). Page numbers come
straight from the file, no estimation.
"""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"

# Project-level per-page JSON source: spring_pipeline/Documents/output/<doc_id>/page_NNNN.json
PROJECT_ROOT = ROOT.parent.parent
PAGES_DATA_DIR = PROJECT_ROOT / "Documents" / "output"

NUL_CACHE_PATH = CACHE_DIR / "nul_works.json"
# Local CSV inventory ("eis-inventory-2nd-pass") used as work-metadata source
# instead of the NUL API. Keyed off the 955$b accession (e.g. 35556036091957);
# see inventory.py for parsing details.
INVENTORY_CSV_PATH = PROJECT_ROOT / "eis-inventory-2nd-pass(eis inventory 2nd pass csv).csv"
SELECTION_PATH = OUTPUT_DIR / "selection.json"
M1_DIR = OUTPUT_DIR / "m1"
M2_DIR = OUTPUT_DIR / "m2"
CRITIC_DIR = OUTPUT_DIR / "critic"
GRADING_DIR = OUTPUT_DIR / "grading_sheets"

# --- NUL API ---
COLLECTION_ID = "f2fc1bd8-c37f-4486-b28a-509f0e0362e1"
NUL_API_BASE = "https://api.dc.library.northwestern.edu/api/v2"

# --- Sample selection (per Pipeline v2 plan) ---
N_SHORT = 5    # < 200 pages
N_MEDIUM = 10  # 200–800 pages
N_LONG = 5     # > 800 pages
SHORT_MAX_PAGES = 200
LONG_MIN_PAGES = 800
MAX_PER_BUREAU = 4
EIS_TYPES = ["Draft", "Final", "Supplemental", "ROD"]
RANDOM_SEED = 20260525

# --- Chunking (in real pages) ---
CHUNK_PAGES = 50
CHUNK_OVERLAP_PAGES = 2

# Pages that windowed extractors read (in real pages).
FIRST_PAGE = 1
FIRST_2_PAGES = 2
FIRST_3_PAGES = 3
FIRST_4_PAGES = 4
FIRST_30_PAGES = 30

# --- Year bounds ---
YEAR_MIN = 1969
YEAR_MAX = 2026

# --- Models ---
# Bedrock inference-profile IDs.
#
# The "haiku" tier now has ZERO call sites. Its only consumer was the title
# fallback in m1.extract_title, which was removed: titles come from the
# inventory index deterministically, because measuring all 54,105 inventory rows
# showed none lack a title and the fallback's only real trigger was an
# over-LONG one (155 rows, bound-volume aggregates) -- a case where the correct
# title was in hand and was being discarded to re-derive it from OCR.
#
# MODEL_HAIKU is kept pointing at Sonnet so that `llm.haiku()` stays safe if
# something calls it, but nothing does. Note also that the original reason for
# the stand-in ("not accessible on this account") is false: Haiku 4.5 answers at
# the fully-versioned id below. It has no bare alias, unlike Sonnet and Opus --
# `us.anthropic.claude-haiku-4-5` returns "model identifier is invalid", which is
# probably how it came to look inaccessible.
#
# If you want Haiku for genuinely high-volume cheap work, the real candidates at
# 2000 docs are the location scope classifier and the key_people role tagger.
# Point those at MODEL_HAIKU_REAL explicitly, and do it BETWEEN calibration
# stages -- changing an extractor's model mid-stage shifts the score
# distribution the frozen thresholds were fitted to.
MODEL_HAIKU = "us.anthropic.claude-sonnet-4-6"
MODEL_SONNET = "us.anthropic.claude-sonnet-4-6"
MODEL_OPUS = "us.anthropic.claude-opus-4-7"

# Real Haiku, verified reachable on this account. Unused by default; see above.
MODEL_HAIKU_REAL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# --- Closed vocabularies ---
EIS_TYPE_PATTERNS = {
    # Order matters: more specific first so "Final Supplemental" hits Supplemental.
    "Supplemental": r"\b(supplement(?:al)?(?:\s+(?:draft|final))?(?:\s+environmental\s+impact\s+statement)?)\b",
    "ROD": r"\b(record\s+of\s+decision|\bROD\b)\b",
    "Draft": r"\b(draft\s+environmental\s+(?:impact\s+)?statement|DEIS)\b",
    "Final": r"\b(final\s+environmental\s+(?:impact\s+)?statement|FEIS)\b",
}

# --- Theme taxonomy (frozen — copied from V1) ---
THEMES = {
    "Transportation Infrastructure": [
        "Mobility Networks and Connectivity",
        "Infrastructure Impacts on Landscapes",
    ],
    "Energy Systems": [
        "Energy Extraction and Production",
        "Energy Distribution and Consumption",
    ],
    "Wildlife and Natural Areas": [
        "Habitat Conservation and Biodiversity",
        "Human-Wildlife Interactions",
    ],
    "Water Systems": [
        "Water Infrastructure and Management",
        "Water Scarcity and Environmental Change",
    ],
    "Urban Development": [
        "Urban Expansion and Land Use Change",
        "Housing, Planning, and Built Environment",
    ],
    "Industrial Production and Materials": [
        "Resource Extraction and Material Flows",
        "Industrial Manufacturing and Pollution",
    ],
    "Climate and Weather Modification": [
        "Climate Engineering and Intervention",
        "Adaptation to Climate Variability",
    ],
    "Governance and Institutional Control": [
        "Environmental Regulation and Policy",
        "Institutional Power and Resource Management",
    ],
    "Place Based Development Conflicts": [
        "Community Resistance and Activism",
        "Land Rights and Displacement",
    ],
    "Indigenous Narratives and Sovereignty": [
        "Indigenous Knowledge and Environmental Stewardship",
        "Sovereignty, Rights, and Self-Determination",
    ],
}

# --- CEQ §1502 standard chapter labels (used for section-mapping) ---
CEQ_CHAPTERS = [
    "Purpose and Need",
    "Alternatives",
    "Affected Environment",
    "Environmental Consequences",
    "Mitigation",
    "Consultation",
]

# Common alternate headings we should map to canonical CEQ chapters when found.
CHAPTER_ALIASES = {
    "Purpose and Need": ["purpose and need", "purpose of and need for", "background and need"],
    "Alternatives": ["alternatives", "alternatives considered", "proposed action and alternatives"],
    "Affected Environment": ["affected environment", "existing environment", "environmental setting"],
    "Environmental Consequences": ["environmental consequences", "environmental impacts", "impacts"],
    "Mitigation": ["mitigation", "mitigation measures"],
    "Consultation": ["consultation", "consultation and coordination", "list of preparers", "preparers"],
}
