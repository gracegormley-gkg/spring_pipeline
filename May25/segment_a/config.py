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

# --- Which model runs the M2 summary MAP step -------------------------------
# "sonnet" (default since extractor-v2) or "opus".
#
# The map step asks for a short summary of one 50-page chunk plus verbatim quotes
# copied out of it. That is extraction and transcription; the synthesis happens in
# the reduce step, which stays on Opus. The map step is ~40% of Opus input tokens,
# so this is the pipeline's largest single cost lever.
#
# It was Opus through stage v1 and moved to Sonnet at the v1 -> v2 boundary,
# which is the only safe place for it: changing an extractor's model shifts the
# score distribution the conformal thresholds were fitted to. The v1 thresholds
# were saturated or force-gated in six of seven buckets, so there was little
# calibration to lose, and every graded doc is rerun before v2 is built.
#
# The quality risk is specific: copying quotes verbatim out of 1970s OCR is where
# a cheaper model would degrade, and it would show up as quotes failing
# verification. Compare the quote-verified rate per doc before and after.
#
# Every M2 artifact records the model as `_summary_map_model`. `segment_b/run.py`
# refuses to reuse a checkpoint made under a different map model, and
# `mcal/build.py` refuses to calibrate on one. Artifacts from before the stamp
# existed are read as Opus (`mcal.settings.M2_LEGACY_SUMMARY_MAP_MODEL`).
SUMMARY_MAP_MODEL = "sonnet"

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
# Heading -> canonical CEQ chapter.
#
# The original table used only the modern CEQ vocabulary ("Purpose and Need",
# "Affected Environment", "Environmental Consequences"), which no document in this
# corpus uses, because they predate it. NEPA 102(2)(C) documents from the 1970s are
# organized as a numbered five-point response, and their headings read:
#
#     1. DESCRIPTION OF PROPOSED ACTION
#     2. DESCRIPTION OF THE EXISTING ENVIRONMENT
#     II. DESCRIPTION OF THE ENVIRONMENT
#     PROBABLE ENVIRONMENTAL IMPACTS
#     9. STAFF DISCUSSION OF SIGNIFICANT ENVIRONMENTAL MATTERS
#     MITIGATING MEASURES DURING MINING OPERATIONS
#     EVALUATION AND DISPOSITION OF COMMENTS RECEIVED FROM DRAFT ENVIRONMENTAL
#         STATEMENT
#
# Measured over the 21 materialized documents, the modern-only table detected
# `Purpose and Need` 0 times, `Affected Environment` once and `Consultation`
# twice. The additions below are taken from the actual headings those documents
# contain -- collected by dumping every line that passes
# `chunk._looks_like_heading_line` and fails `chunk._map_to_ceq` -- rather than
# from the CEQ regulations. `tests/test_detect_chapters.py` records the before and
# after.
#
# `_map_to_ceq` matches the LONGEST alias first, so entries here may safely
# overlap: "proposed action and alternatives" resolves to Alternatives even though
# "proposed action" is a Purpose and Need alias.
CHAPTER_ALIASES = {
    "Purpose and Need": [
        "purpose and need",
        "purpose of and need for",
        "background and need",
        # 1970s 102(2)(C) point 1: the project is described before it is justified,
        # and the description IS the purpose-and-need section.
        "description of proposed action",
        "description of the proposed action",
        "description of the proposed project",
        "description of the proposed improvement",
        "description of the action",
        "proposed action",
        "project description",
        "purpose of the project",
        "purpose and description",
        "need for the project",
        "need for the proposed action",
    ],
    "Alternatives": [
        "alternatives",
        "alternatives considered",
        "alternatives to the proposed action",
        "proposed action and alternatives",
        "alternative",
        "alternative actions",
        "alternative corridor",
        "alternative freeway corridor locations",
        "recommended alternative",
        "bus alternatives",
        "transit alternatives",
        "no action alternative",
    ],
    "Affected Environment": [
        "affected environment",
        "existing environment",
        "environmental setting",
        # 1970s point 2.
        "description of the existing environment",
        "description of the environment",
        "description of existing environment",
        "existing conditions",
        "environment affected",
        "the environment",
        "environmental inventory",
    ],
    "Environmental Consequences": [
        "environmental consequences",
        "environmental impacts",
        "impacts",
        # 1970s point 3, and the EPA staff-review formulation.
        "probable environmental impacts",
        "probable impact",
        "probable impacts",
        "environmental impact of the proposed action",
        "environmental impacts of the proposed action",
        "adverse environmental effects",
        "adverse impacts",
        "unavoidable adverse impacts",
        "staff discussion of significant environmental matters",
        "significant environmental matters",
        "impact of the proposed action",
        "environmental effects",
    ],
    "Mitigation": [
        "mitigation",
        "mitigation measures",
        "mitigating measures",
        "measures to mitigate",
        "measures to minimize",
    ],
    "Consultation": [
        "consultation",
        "consultation and coordination",
        "list of preparers",
        "preparers",
        # 1970s point 5 / comment-response sections.
        "evaluation and disposition of comments",
        "comments and responses",
        "comments received",
        "response to comments",
        "coordination and consultation",
        "coordination",
        "distribution list",
        "agencies consulted",
    ],
}
