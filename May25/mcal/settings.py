"""
M-Cal configuration.

Single source of truth for: artifact paths and stage-versioning, the 7 CP
buckets and the field -> bucket map, alpha / degeneracy thresholds, confidence
signal weights, judge-model routing, and geocoder asset discovery.

Import this module before anything that touches segment_a -- it installs the
sys.path bridge described below.

segment_a is NOT a package (no __init__.py); its modules import each other
flat (`from config import ...`, `from pages import Doc`). To reuse them we
append segment_a to sys.path. We *append* rather than insert so that installed
packages and stdlib win any name collision -- segment_a owns several generic
module names (config, chunk, grading, evidence, critic, pages, llm) and we do
not want them shadowing anything.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# --- Paths ------------------------------------------------------------------

MCAL_ROOT = Path(__file__).resolve().parent          # May25/mcal
MAY25_ROOT = MCAL_ROOT.parent                        # May25
PROJECT_ROOT = MAY25_ROOT.parent                     # spring_pipeline
SEGMENT_A_DIR = MAY25_ROOT / "segment_a"
SEGMENT_B_DIR = MAY25_ROOT / "segment_b"

ARTIFACTS_DIR = MCAL_ROOT / "artifacts"
TEMPLATES_DIR = MCAL_ROOT / "templates"
RUBRICS_DIR = TEMPLATES_DIR / "rubrics"

# Human grades. MCAL_PLAN 3.1 points at segment_a/output/grading_sheets/*.csv,
# but those sheets are 100% unfilled (0 of 333 rows have `your_grade`). The
# real grades live in the transposed, free-text Evaluation sheet. mcal/grades.py
# reads BOTH: the Evaluation sheet is the seed-v1 source, and the per-doc
# grading sheets become the source from v2 onward once they are filled in.
EVALUATION_CSV = MAY25_ROOT / "Evaluation - Sheet1.csv"
GRADING_SHEETS_DIR = SEGMENT_A_DIR / "output" / "grading_sheets"

# segment_a outputs consumed as M-Cal inputs.
SEGMENT_A_OUTPUT = SEGMENT_A_DIR / "output"
M1_DIR = SEGMENT_A_OUTPUT / "m1"
M2_DIR = SEGMENT_A_OUTPUT / "m2"
CRITIC_DIR = SEGMENT_A_OUTPUT / "critic"

# Per-page OCR JSON: Documents/output/<doc_id>/page_NNNN.json
PAGES_DATA_DIR = PROJECT_ROOT / "Documents" / "output"
INVENTORY_CSV_PATH = PROJECT_ROOT / "eis-inventory-2nd-pass(eis inventory 2nd pass csv).csv"

# --- M2 prompt-version marker (MCAL_PLAN 3.7 step-0 precheck) ---------------

M2_PROMPT_VERSION_MARKER = M2_DIR / "_prompt_version.txt"
M2_PROMPT_VERSION_REQUIRED = "v1_plain_language"
M2_PRE_AMENDMENT_DIR = SEGMENT_A_OUTPUT / "m2_pre_amendment"


# --- sys.path bridge --------------------------------------------------------

def bridge_segment_a() -> None:
    """Make segment_a's flat modules importable. Idempotent."""
    p = str(SEGMENT_A_DIR)
    if p not in sys.path:
        sys.path.append(p)


bridge_segment_a()


# --- .env -------------------------------------------------------------------

def load_env() -> None:
    """
    Load spring_pipeline/.env if present.

    segment_a read os.environ directly and had no .env layer. The geocoder
    stack (MCAL_PLAN 3.9a) needs three per-machine values that do not belong
    in source control, so we add one here. Never overrides an already-set
    environment variable.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency is pinned
        return
    for candidate in (PROJECT_ROOT / ".env", MAY25_ROOT / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)


load_env()


# --- Models -----------------------------------------------------------------
# Mirrors segment_a/config.py. Duplicated deliberately: mcal must be able to
# route to Opus for judging independently of whatever segment_a decides.

MODEL_SONNET = "us.anthropic.claude-sonnet-4-6"
MODEL_OPUS = "us.anthropic.claude-opus-4-7"


# --- Field vocabulary -------------------------------------------------------
# Canonical field keys. These are the keys used in thresholds, confidence
# scoring, critic prompt filenames, and run_manifest.json.

M1_FIELDS = ("title", "year", "eis_type", "lead_agency")

# The six summary subfields as emitted by segment_a/m2.py SUMMARY_SCHEMA_KEYS.
SUMMARY_OVERVIEW = "summary.overview"
SUMMARY_SUBFIELDS = (
    "summary.project_description",
    "summary.affected_community",
    "summary.alternatives_overview",
    "summary.environmental_impact",
    "summary.public_response",
)
SUMMARY_FIELDS = (SUMMARY_OVERVIEW,) + SUMMARY_SUBFIELDS

SUMMARY_OF_INTEREST = "summary_of_interest"

STRUCTURED_FIELDS = ("alternatives", "themes", "location", "key_people")

ALL_FIELDS = M1_FIELDS + SUMMARY_FIELDS + (SUMMARY_OF_INTEREST,) + STRUCTURED_FIELDS

# Fields that atomic_verify.py decomposes (MCAL_PLAN 3.4).
# Note: summary.overview is deliberately EXCLUDED -- it is a roll-up whose
# evidence is carried forward from the five subfields (m2.py:321-336), so
# decomposing it would double-count. It is still bucketed and gated.
ATOMIC_VERIFY_FIELDS = SUMMARY_SUBFIELDS + (SUMMARY_OF_INTEREST,)


# --- CP buckets (MCAL_PLAN 3.3) ---------------------------------------------
# Seven buckets, frozen per MCAL_PLAN 7.5.
#
# Plan-inconsistency resolved here: 3.3 defines summary_narrative as
# {project_description, affected_community, alternatives_overview,
# public_response} and omits `overview` entirely, but 1 ("Fields with no
# observed failures") explicitly assigns overview to summary_narrative. We
# follow 1 -- otherwise summary.overview has no bucket and could never be
# gated, which is strictly worse. Recorded in calibration_report as a
# documented deviation.

BUCKETS = {
    "M1": list(M1_FIELDS),
    "summary_narrative": [
        SUMMARY_OVERVIEW,
        "summary.project_description",
        "summary.affected_community",
        "summary.alternatives_overview",
        "summary.public_response",
    ],
    "summary_numeric": ["summary.environmental_impact"],
    "summary_of_interest": [SUMMARY_OF_INTEREST],
    "alternatives+themes": ["alternatives", "themes"],
    "location": ["location"],
    "key_people": ["key_people"],
}

BUCKET_ORDER = (
    "M1",
    "summary_narrative",
    "summary_numeric",
    "summary_of_interest",
    "alternatives+themes",
    "location",
    "key_people",
)

# The six buckets that predate summary_of_interest. MCAL_PLAN 6 acceptance
# criterion 3 counts degeneracy over these only -- summary_of_interest starts
# with zero graded examples by construction, so including it would make the
# criterion unreachable at v2.
ORIGINAL_BUCKETS = tuple(b for b in BUCKET_ORDER if b != "summary_of_interest")

FIELD_TO_BUCKET = {f: b for b, fields in BUCKETS.items() for f in fields}


def bucket_for_field(field: str) -> str:
    """Map a canonical field key to its CP bucket. Raises on unknown fields."""
    try:
        return FIELD_TO_BUCKET[field]
    except KeyError:
        raise KeyError(
            f"No CP bucket for field {field!r}. Bucket definitions are frozen "
            f"(MCAL_PLAN 7.5); a genuinely new field needs a plan amendment. "
            f"Known fields: {sorted(FIELD_TO_BUCKET)}"
        ) from None


# --- Conformal prediction parameters (MCAL_PLAN 3.3, 7 Q4) ------------------

ALPHA = 0.15
ALPHA_EFFECTIVE_DEGENERATE = 0.25

# N_wrong_docs below this at ALPHA -> degenerate, retry at ALPHA_EFFECTIVE.
DEGENERATE_MIN_WRONG_DOCS = 6
# N_wrong_docs below this at ALPHA_EFFECTIVE -> degenerate_severe, gate all.
DEGENERATE_SEVERE_MIN_WRONG_DOCS = 3
# Full-scale Segment B unlock threshold (MCAL_PLAN 6, 7 Q1).
FULL_SCALE_MIN_WRONG_DOCS = 15


# --- Confidence signals (MCAL_PLAN 3.3) -------------------------------------
# Frozen at 0.5/0.5 through at least stage v3. The other four signals are
# computed and logged at weight 0 so that weight validation has data to work
# with when n reaches ~60.

SIGNAL_WEIGHTS = {
    "s_quote": 0.5,
    "s_critic": 0.5,
    "s_source": 0.0,
    "s_citation": 0.0,
    "s_shard": 0.0,
    "s_acronym": 0.0,
}

WEIGHT_VALIDATION_MIN_N = 60

CRITIC_VERDICT_SCORES = {
    "PASS": 1.0,
    "PASS_WITH_NOTE": 0.7,
    "RE_EXTRACT": 0.3,
    "HUMAN_REVIEW": 0.0,
}

QUOTE_VERDICT_SCORES = {"yes": 1.0, "mixed": 0.5, "no": 0.0}

# M1 values are not verbatim quotes, so s_quote is undefined for them and
# defaults to 1.0 (MCAL_PLAN 3.3). This makes the M1 composite
# 0.5*s_critic + 0.5 -- a 0.5 floor plus half the Critic verdict.
S_QUOTE_DEFAULT_M1 = 1.0


# --- quote_check thresholds (MCAL_PLAN 3.2) ---------------------------------

QUOTE_PAGE_TOLERANCE = 2
QUOTE_RATIO_YES = 90.0
QUOTE_RATIO_MIXED = 60.0
# Below QUOTE_RATIO_MIXED -> "no".

# Second, orthogonal gate: content-token coverage.
#
# rapidfuzz partial_ratio alone is not safe at the plan's thresholds. Measured
# on the 8 graded docs (444 verified quotes, each also scored against a
# different document's pages as a negative control):
#
#     true positives   : median 100.0, min 100.0
#     foreign quotes   : median  49.5, p95  58.9, and up to 67.7 for quotes
#                        under 40 chars
#
# So the plan's 60.0 "mixed" floor sits *below* the chance ceiling for short
# strings -- partial_ratio slides a window over a long page and finds a decent
# match by luck. That is exactly why the Lincoln Hwy fabricated clause "or
# important wildlife habitats are affected" scores 62.8 against its own cited
# pages despite being absent from them. Short strings are the norm for atomic
# claims (MCAL_PLAN 3.4 asks for one subject-predicate-object per atom), so
# this would have been a systematic false-accept in atomic verification.
#
# Content-token coverage -- the fraction of a quote's content words (>=4 chars,
# non-stopword, NEPA boilerplate excluded) that appear on the page -- separates
# far more cleanly, because it is insensitive to window position:
#
#     true positives   : median 1.00, min  1.00
#     foreign quotes   : median 0.10, p95  0.33, max 0.67
#     coverage >= 0.70 : 0.0% false-negative, 0.0% false-positive
#
# The wildlife clause scores 0.00 here, i.e. unambiguously rejected. Both gates
# must pass; coverage is the binding one in practice. Measurements are
# reproducible via tests/test_quote_check.py::TestAgainstCorpus.
QUOTE_COVERAGE_YES = 0.70
# The "mixed" floor sits below the observed foreign-quote maximum (0.67) on
# purpose: "mixed" is a half-credit signal feeding s_quote, not an accept, and
# keeping it permissive preserves the distinction between "partially supported"
# and "absent" that the Critic and atomic verifier both act on.
QUOTE_COVERAGE_MIXED = 0.40
# Quotes with fewer than this many content tokens cannot be scored on coverage
# and fall back to the ratio gate alone.
QUOTE_COVERAGE_MIN_TOKENS = 3
#
# Note on the +/-2 page tolerance: MCAL_PLAN 0 justifies it as compensation
# for page numbers estimated via char_offset/2500. That estimation does not
# exist in the current code -- segment_a chunks in real pages (config.py
# CHUNK_PAGES=50) and page numbers are read exactly from per-page JSON. The
# tolerance is retained on different and still-valid grounds: OCR noise, and
# extractors citing a page adjacent to the one a quote actually lands on when
# a passage straddles a page seam (pages.find_quote deliberately refuses to
# match across seams, pages.py:89-90).


# --- Judge-model routing (MCAL_PLAN 3.11, 7 Q2) -----------------------------
# Opus for the five summary subfields plus summary_of_interest; Sonnet
# otherwise. summary.overview stays on Sonnet -- it is a roll-up of already
# Opus-judged subfields, so paying twice buys little.

OPUS_JUDGED_FIELDS = frozenset(SUMMARY_SUBFIELDS) | {SUMMARY_OF_INTEREST}


def judge_model_for_field(field: str) -> str:
    return MODEL_OPUS if field in OPUS_JUDGED_FIELDS else MODEL_SONNET


def default_judge_model_map() -> dict[str, str]:
    """The judge_model_by_field block written into confidence_config.v(N).json."""
    return {f: ("opus" if f in OPUS_JUDGED_FIELDS else "sonnet") for f in ALL_FIELDS}


# --- Dependent fields (MCAL_PLAN 3.10 era gate) -----------------------------
# If `year` is not trustworthy, key_people cannot be era-gated, so it cascades
# to HUMAN_REVIEW.

DEPENDENT_FIELDS = {"year": ["key_people"]}


# --- Geocoder stack (MCAL_PLAN 3.9a) ----------------------------------------

CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
MAPBOX_GEOCODER_URL = "https://api.mapbox.com/geocoding/v5/mapbox.places"
NOMINATIM_USER_AGENT = "eis_pipeline_mcal"
NOMINATIM_MIN_INTERVAL_SEC = 1.1


def _env_path(name: str) -> Optional[Path]:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


def padus_path() -> Optional[Path]:
    return _env_path("PADUS_GEODATABASE_PATH")


def gnis_path() -> Optional[Path]:
    return _env_path("GNIS_TSV_PATH")


def mapbox_token() -> Optional[str]:
    return os.environ.get("MAPBOX_TOKEN", "").strip() or None


def geocoder_precheck() -> dict:
    """
    Check the three user-supplied geocoder assets (MCAL_PLAN 3.7, 3.9a).

    Returns {"stack": "full"|"reduced", "missing": [...], "checklist": [...]}.
    Never raises -- build.py decides whether to halt, and MCAL_PLAN 3.9a says
    to continue in reduced mode rather than block the calibration loop.
    """
    missing: list[str] = []
    checklist: list[str] = []

    padus = padus_path()
    if padus is None:
        missing.append("PADUS_GEODATABASE_PATH")
        checklist.append(
            "PADUS_GEODATABASE_PATH unset. Download PAD-US (Protected Areas "
            "Database of the US) from USGS, unzip, and point this at the .gdb "
            "directory. ~1GB. Enables federal-lands geocoding (hop 3), the "
            "biggest expected quality win for this corpus."
        )
    elif not padus.exists():
        missing.append("PADUS_GEODATABASE_PATH")
        checklist.append(f"PADUS_GEODATABASE_PATH points at a missing path: {padus}")

    gnis = gnis_path()
    if gnis is None:
        missing.append("GNIS_TSV_PATH")
        checklist.append(
            "GNIS_TSV_PATH unset. Download the USGS GNIS domestic-names file "
            "from The National Map and point this at the .txt/.tsv. ~2GB. "
            "Enables named natural/cultural feature geocoding (hop 2)."
        )
    elif not gnis.exists():
        missing.append("GNIS_TSV_PATH")
        checklist.append(f"GNIS_TSV_PATH points at a missing path: {gnis}")

    if mapbox_token() is None:
        missing.append("MAPBOX_TOKEN")
        checklist.append(
            "MAPBOX_TOKEN unset. Register a free Mapbox account and put the "
            "token in .env as MAPBOX_TOKEN. 100k requests/month free tier "
            "comfortably covers ~10 calls/doc x 2000 docs. Enables POI and "
            "named-highway geocoding (hop 4)."
        )

    return {
        "stack": "reduced" if missing else "full",
        "missing": missing,
        "checklist": checklist,
    }


# --- Stage versioning (MCAL_PLAN 2) -----------------------------------------
# Artifacts are stage-versioned: taxonomy.v1.json, thresholds.v2.json, ...
# Everything is written into artifacts/v(N)-draft/ first; ratifying the
# taxonomy promotes the whole directory to artifacts/v(N)/.

STAGE_PATTERN = "v"  # stages are "v1", "v2", ...


def normalize_stage(stage: str) -> str:
    """Accept 'v1' or '1'; return 'v1'. Rejects anything else."""
    s = str(stage).strip().lower()
    if s.startswith("v"):
        s = s[1:]
    if not s.isdigit() or int(s) < 1:
        raise ValueError(f"Bad stage {stage!r}; expected v1, v2, ... ")
    return f"v{int(s)}"


def stage_number(stage: str) -> int:
    return int(normalize_stage(stage)[1:])


def prior_stage(stage: str) -> Optional[str]:
    n = stage_number(stage)
    return f"v{n - 1}" if n > 1 else None


def stage_dir(stage: str, *, draft: bool = False) -> Path:
    """artifacts/v1/ or artifacts/v1-draft/."""
    s = normalize_stage(stage)
    return ARTIFACTS_DIR / (f"{s}-draft" if draft else s)


def artifact_path(name: str, stage: str, *, draft: bool = False) -> Path:
    """
    Stage-versioned artifact path.

    `name` is the bare artifact name from MCAL_PLAN 2 without the stage
    suffix, e.g. "taxonomy.json" -> artifacts/v1/taxonomy.v1.json.
    """
    s = normalize_stage(stage)
    d = stage_dir(stage, draft=draft)
    if "." in name:
        base, _, ext = name.rpartition(".")
        return d / f"{base}.{s}.{ext}"
    return d / f"{name}.{s}"


def latest_stage() -> Optional[str]:
    """Highest promoted (non-draft) stage on disk, or None."""
    if not ARTIFACTS_DIR.exists():
        return None
    stages = []
    for p in ARTIFACTS_DIR.iterdir():
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit():
            stages.append(int(p.name[1:]))
    return f"v{max(stages)}" if stages else None


# --- Unversioned / rolling artifacts (MCAL_PLAN 2) --------------------------
# These are explicitly not stage-suffixed in the plan.

NULL_TAG_MONITOR_PATH = ARTIFACTS_DIR / "null_tag_monitor.json"
NEXT_BATCH_PATH = ARTIFACTS_DIR / "next_batch.csv"

# Rolling monitor threshold: null failure_tag rate per bucket above which the
# taxonomy needs a v(N+1) refresh with new T19+ codes (MCAL_PLAN 6).
NULL_TAG_REFRESH_THRESHOLD = 0.15

# summary_of_interest non-empty rate above which the field is presumed to be
# manufacturing salience rather than detecting it (MCAL_PLAN 3.15, 6).
SOI_NONEMPTY_RATE_CEILING = 0.60

# atomic_verify false-negative rate on correctly-graded subfields. Advisory at
# v1 (atom sample too small), gating from v2 (MCAL_PLAN 3.4, 6).
ATOMIC_FALSE_NEGATIVE_CEILING = 0.10
ATOMIC_FALSE_NEGATIVE_GATING_FROM_STAGE = 2


# --- Active selection (MCAL_PLAN 3.6) ---------------------------------------

NEXT_BATCH_SIZE = 10


# --- doc_id normalization ---------------------------------------------------
# doc_ids are inconsistently cased on disk (segment_a/output/critic/ has both
# p0491_... and P0491_...), and pages.load_doc does a case-sensitive directory
# lookup. Normalize on the way in.

def normalize_doc_id(doc_id: str) -> str:
    return (doc_id or "").strip().lower()


def resolve_doc_dir(doc_id: str) -> Optional[Path]:
    """Find a doc's page directory case-insensitively."""
    if not PAGES_DATA_DIR.exists():
        return None
    target = normalize_doc_id(doc_id)
    for p in PAGES_DATA_DIR.iterdir():
        if p.is_dir() and normalize_doc_id(p.name) == target:
            return p
    return None


def available_doc_ids() -> list[str]:
    """doc_ids with materialized per-page OCR JSON on this machine."""
    if not PAGES_DATA_DIR.exists():
        return []
    return sorted(p.name for p in PAGES_DATA_DIR.iterdir() if p.is_dir())
