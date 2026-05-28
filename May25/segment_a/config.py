"""
Segment A: Calibration Run config.

Source data: docs_with_digits.json (flat OCR strings, 203 docs keyed by accession).
No PDF outline available — page citations are estimated from char offsets at
CHARS_PER_PAGE = 2500 (matches V1 pipeline heuristic).
"""

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"

DOCS_WITH_DIGITS_PATH = Path("/Users/gracegormley/Desktop/Y2/Q2/Knight Lab/docs_with_digits.json")
NUL_CACHE_PATH = CACHE_DIR / "nul_works.json"
SELECTION_PATH = OUTPUT_DIR / "selection.json"
M1_DIR = OUTPUT_DIR / "m1"
M2_DIR = OUTPUT_DIR / "m2"
CRITIC_DIR = OUTPUT_DIR / "critic"
GRADING_DIR = OUTPUT_DIR / "grading_sheets"

# --- NUL API ---
COLLECTION_ID = "f2fc1bd8-c37f-4486-b28a-509f0e0362e1"
NUL_API_BASE = "https://api.dc.library.northwestern.edu/api/v2"

# --- Page estimation ---
# OCR was produced from scanned PDFs; V1 pipeline calibrated this at 2500.
CHARS_PER_PAGE = 2500

# --- Sample selection (per Pipeline v2 plan) ---
N_SHORT = 5    # < 200 pages
N_MEDIUM = 10  # 200–800 pages
N_LONG = 5     # > 800 pages
SHORT_MAX_PAGES = 200
LONG_MIN_PAGES = 800
MAX_PER_BUREAU = 4
EIS_TYPES = ["Draft", "Final", "Supplemental", "ROD"]
RANDOM_SEED = 20260525

# --- Chunking ---
CHUNK_PAGES = 50
CHUNK_OVERLAP_PAGES = 2
CHUNK_CHARS = CHUNK_PAGES * CHARS_PER_PAGE          # 125_000
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_PAGES * CHARS_PER_PAGE  # 5_000

# Pages that windowed extractors read.
FIRST_PAGE_CHARS = 1 * CHARS_PER_PAGE
FIRST_2_PAGES = 2 * CHARS_PER_PAGE
FIRST_3_PAGES = 3 * CHARS_PER_PAGE
FIRST_4_PAGES = 4 * CHARS_PER_PAGE
FIRST_30_PAGES = 30 * CHARS_PER_PAGE

# --- Year bounds ---
YEAR_MIN = 1969
YEAR_MAX = 2026

# --- Models ---
# Bedrock inference-profile IDs. Haiku 4-5 isn't accessible on this account,
# so the "haiku" tier falls back to Sonnet here. Cost impact is small for
# Segment A's volume.
MODEL_HAIKU = "us.anthropic.claude-sonnet-4-6"
MODEL_SONNET = "us.anthropic.claude-sonnet-4-6"
MODEL_OPUS = "us.anthropic.claude-opus-4-7"

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
