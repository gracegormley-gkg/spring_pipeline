"""
Central configuration: agency vocabulary, theme taxonomy, regex patterns, model IDs.
"""

import re

# ---------------------------------------------------------------------------
# Model IDs
# Note: plan spec listed claude-opus-4-7 which predates current releases.
# Using the latest available models per the current Anthropic model lineup.
# ---------------------------------------------------------------------------
MODELS = {
    "heavy": "claude-opus-4-6",          # summary, themes, key_people, alternatives, historical
    "light": "claude-haiku-4-5-20251001", # chunk labeling, location, simple checks
    "critic": "claude-sonnet-4-6",        # per-field critics
}

# ---------------------------------------------------------------------------
# Agency vocabulary
# Each entry: canonical_name -> {abbreviations: [...], variants: [...]}
# ---------------------------------------------------------------------------
AGENCY_VOCAB: dict[str, dict] = {
    "Bureau of Land Management": {
        "abbreviations": ["BLM"],
        "variants": ["Bureau of Land Mgmt", "BLM Field Office"],
    },
    "U.S. Forest Service": {
        "abbreviations": ["USFS", "FS"],
        "variants": ["United States Forest Service", "Forest Service", "US Forest Service"],
    },
    "U.S. Army Corps of Engineers": {
        "abbreviations": ["USACE", "COE"],
        "variants": [
            "Army Corps of Engineers", "Corps of Engineers",
            "United States Army Corps of Engineers",
        ],
    },
    "Environmental Protection Agency": {
        "abbreviations": ["EPA"],
        "variants": ["U.S. Environmental Protection Agency", "US EPA"],
    },
    "National Park Service": {
        "abbreviations": ["NPS"],
        "variants": ["U.S. National Park Service", "US National Park Service"],
    },
    "U.S. Fish and Wildlife Service": {
        "abbreviations": ["FWS", "USFWS"],
        "variants": [
            "Fish and Wildlife Service", "US Fish and Wildlife Service",
            "United States Fish and Wildlife Service",
        ],
    },
    "NASA": {
        "abbreviations": ["NASA"],
        "variants": ["National Aeronautics and Space Administration"],
    },
    "Department of Energy": {
        "abbreviations": ["DOE"],
        "variants": ["U.S. Department of Energy", "US DOE"],
    },
    "Department of Transportation": {
        "abbreviations": ["DOT"],
        "variants": [
            "U.S. Department of Transportation", "Federal Highway Administration",
            "FHWA", "Federal Transit Administration", "FTA",
        ],
    },
    "Federal Energy Regulatory Commission": {
        "abbreviations": ["FERC"],
        "variants": [],
    },
    "Bureau of Ocean Energy Management": {
        "abbreviations": ["BOEM", "MMS"],
        "variants": ["Minerals Management Service", "Bureau of Ocean Energy"],
    },
    "Bureau of Indian Affairs": {
        "abbreviations": ["BIA"],
        "variants": ["U.S. Bureau of Indian Affairs"],
    },
    "Bureau of Reclamation": {
        "abbreviations": ["BOR", "Reclamation"],
        "variants": ["U.S. Bureau of Reclamation", "USBR"],
    },
    "Federal Aviation Administration": {
        "abbreviations": ["FAA"],
        "variants": ["U.S. Federal Aviation Administration"],
    },
    "Nuclear Regulatory Commission": {
        "abbreviations": ["NRC"],
        "variants": ["U.S. Nuclear Regulatory Commission", "Atomic Energy Commission", "AEC"],
    },
    "Department of Defense": {
        "abbreviations": ["DOD", "DoD"],
        "variants": ["U.S. Department of Defense", "Department of the Army", "Department of the Navy"],
    },
    "Department of Agriculture": {
        "abbreviations": ["USDA"],
        "variants": ["U.S. Department of Agriculture"],
    },
    "Department of Interior": {
        "abbreviations": ["DOI"],
        "variants": ["U.S. Department of the Interior", "Department of the Interior"],
    },
    "Federal Railroad Administration": {
        "abbreviations": ["FRA"],
        "variants": [],
    },
    "Urban Mass Transportation Administration": {
        "abbreviations": ["UMTA"],
        "variants": ["Urban Mass Transit Administration"],
    },
    "Housing and Urban Development": {
        "abbreviations": ["HUD"],
        "variants": ["U.S. Department of Housing and Urban Development"],
    },
}

# Build a flat lookup: any variant/abbreviation -> canonical name
_AGENCY_FLAT: dict[str, str] = {}
for _canonical, _data in AGENCY_VOCAB.items():
    _AGENCY_FLAT[_canonical.lower()] = _canonical
    for _abbr in _data["abbreviations"]:
        _AGENCY_FLAT[_abbr.lower()] = _canonical
    for _var in _data["variants"]:
        _AGENCY_FLAT[_var.lower()] = _canonical

AGENCY_FLAT = _AGENCY_FLAT


def lookup_agency(text: str) -> str | None:
    """Exact lookup of agency from any known variant. Returns canonical name or None."""
    return AGENCY_FLAT.get(text.strip().lower())


# ---------------------------------------------------------------------------
# Theme taxonomy
# ---------------------------------------------------------------------------
THEMES: dict[str, list[str]] = {
    "energy_infrastructure": [
        "fossil_fuel_extraction",
        "renewable_energy",
        "transmission_lines",
        "pipelines",
        "nuclear_power",
        "energy_storage",
        "offshore_energy",
        "energy_efficiency",
    ],
    "transportation": [
        "highways_and_roads",
        "rail_and_transit",
        "airports_and_aviation",
        "ports_and_waterways",
        "bridges_and_tunnels",
        "pedestrian_and_cycling",
    ],
    "land_management": [
        "public_lands",
        "grazing_and_range",
        "forestry",
        "wilderness_and_conservation",
        "recreation",
        "tribal_consultation",
    ],
    "water_resources": [
        "dams_and_reservoirs",
        "flood_control",
        "water_supply",
        "wetlands",
        "coastal_management",
        "irrigation",
    ],
    "defense_and_military": [
        "military_installations",
        "weapons_testing",
        "base_realignment",
        "training_ranges",
    ],
    "urban_development": [
        "housing",
        "urban_renewal",
        "commercial_development",
        "community_facilities",
        "brownfield_redevelopment",
    ],
    "mining_and_extraction": [
        "hard_rock_mining",
        "coal_mining",
        "oil_and_gas",
        "sand_and_gravel",
        "remediation",
    ],
    "agriculture_and_forestry": [
        "crop_production",
        "timber_harvesting",
        "pesticides",
        "irrigation_agriculture",
        "livestock",
    ],
    "wildlife_and_habitat": [
        "endangered_species",
        "habitat_conservation",
        "fisheries",
        "migratory_birds",
        "marine_mammals",
    ],
    "cultural_heritage": [
        "historic_properties",
        "archaeological_resources",
        "indigenous_cultural_sites",
        "section_106",
    ],
    "waste_and_remediation": [
        "hazardous_waste",
        "nuclear_waste",
        "landfill",
        "superfund",
        "wastewater",
    ],
    "other": [
        "unclassified",
    ],
}

ALL_THEMES = list(THEMES.keys())
ALL_SUBTHEMES: list[str] = [s for subs in THEMES.values() for s in subs]


# ---------------------------------------------------------------------------
# Chunk topic tags
# ---------------------------------------------------------------------------
CHUNK_TOPIC_TAGS = [
    "purpose_and_need",
    "affected_environment",
    "proposed_action",
    "alternatives",
    "mitigation",
    "consultation",
    "cumulative_impacts",
    "comments_and_responses",
    "appendix",
    "references",
    "other",
]

# ---------------------------------------------------------------------------
# EIS type regex patterns
# ---------------------------------------------------------------------------
EIS_TYPE_PATTERNS = {
    "Draft": re.compile(r"\b(draft|DEIS)\b", re.IGNORECASE),
    "Final": re.compile(r"\b(final|FEIS)\b", re.IGNORECASE),
    "Supplemental": re.compile(r"\b(supplemental|supplement|SEIS)\b", re.IGNORECASE),
}

# ---------------------------------------------------------------------------
# Date regex patterns
# Year filter: NEPA enacted 1969 — anything earlier is noise
# ---------------------------------------------------------------------------
NEPA_YEAR = 1969
import datetime
MAX_YEAR = datetime.date.today().year

MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)
DATE_PATTERNS = [
    re.compile(
        rf"\b(?:{MONTH_NAMES})\s+\d{{1,2}},?\s+(\d{{4}})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d{4})-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/(\d{4})\b"),
    re.compile(r"\b(\d{4})\b"),  # bare year — used as last resort
]

# ---------------------------------------------------------------------------
# Heading regex patterns
#
# Both patterns use [ \t]+ (not \s+) so they cannot cross line boundaries —
# this prevents OCR letterhead fragments like "5 F\nUNITED STATES DEPARTMENT"
# from being treated as a single heading.
#
# Pattern 1 captures the rest of the heading line so the post-match filter
# (_is_real_heading in stage0_triage) can reject bare legal citations like
# "Section 4(f)" by word count.
#
# Pattern 2 requires at least one decimal (e.g. "2.1") to distinguish real
# numbered headings from leading-digit noise like street addresses ("143 SOUTH
# THIRD STREET").
# ---------------------------------------------------------------------------
HEADING_PATTERNS = [
    re.compile(
        r"^[ \t]*(?:CHAPTER|SECTION|APPENDIX)[ \t]+[IVXLC\d]+(?:\.\d+)*[^\n]*",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"^[ \t]*\d+(?:\.\d+){1,3}[ \t]+[A-Z][A-Z\s][^\n]*",
        re.MULTILINE,
    ),
]
TOC_MARKER_PATTERN = re.compile(r"^\s*Table\s+of\s+Contents\s*$", re.MULTILINE | re.IGNORECASE)

# Heading-filter regexes: applied to candidate heading text in
# stage0_triage._is_real_heading to reject addresses and ZIP codes.
HEADING_ADDRESS_KEYWORDS = re.compile(
    r"\b(STREET|AVENUE|\bAVE\b|BLVD|BOULEVARD|ROAD|\bRD\b|DRIVE|\bDR\b|"
    r"LANE|\bLN\b|PARKWAY|PKWY|HIGHWAY|HWY|"
    r"P\.?\s?O\.?\s?BOX|SUITE|FLOOR)\b",
    re.IGNORECASE,
)
HEADING_ZIP_PATTERN = re.compile(r"\b\d{5}(?:-\d{4})?\b")

# ---------------------------------------------------------------------------
# Length thresholds (words)
# ---------------------------------------------------------------------------
SHORT_THRESHOLD = 10_000
LONG_THRESHOLD = 60_000

# ---------------------------------------------------------------------------
# OCR quality thresholds
# ---------------------------------------------------------------------------
OCR_UNCLEAR_THRESHOLD = 0.8   # doc-level — flag if below
OCR_EXCLUDE_THRESHOLD = 0.7   # chunk-level — exclude from retrieval if below

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------
FIXED_CHUNK_PAGES = 30  # pages per chunk when no headings

# ---------------------------------------------------------------------------
# Key-people extraction parameters
#
# MAX_KEY_PEOPLE bumped from 30 → 50 per 5/14 meeting: "cut it down less" —
# the triage + stance steps already filter noise downstream, so let more
# candidates through.
# ---------------------------------------------------------------------------
MIN_ENTITY_FREQUENCY = 2
MAX_KEY_PEOPLE = 50
QUOTE_MIN_WORDS = 8
QUOTE_MAX_WORDS = 40

# Particles that can appear between first and last name (van der, de la, etc.).
# Used by stage0_triage._is_full_name to decide whether a token is a real name
# component or a connector.
NAME_PARTICLES = frozenset({
    "de", "del", "la", "van", "von", "der", "den", "du", "dos", "da", "ten", "ter",
})

# Middle-initial pattern: single uppercase letter, optional period (e.g. "J.").
MIDDLE_INITIAL_PATTERN = re.compile(r"^[A-Z]\.?$")

# A real-name token: starts uppercase, contains at least one lowercase letter,
# letters / apostrophe / hyphen only. Rejects all-caps tokens like "UNITED".
NAME_TOKEN_PATTERN = re.compile(r"^[A-Z][A-Za-z'-]*[a-z][A-Za-z'-]*$")

# spaCy model preference order. The loader falls back if the preferred model
# is not installed locally.
SPACY_MODELS_PREFERENCE = ["en_core_web_trf", "en_core_web_lg"]

# Haiku gap-filler budget: max number of LLM calls per document to fill in
# stakeholders missed by spaCy + dictionary lookups.
NER_GAPFILL_MAX_CALLS = 6
# Trigger thresholds for conditional gap-fill (in addition to always running on
# chunks tagged comments_and_responses).
NER_GAPFILL_MIN_WORDS = 2000
NER_GAPFILL_MAX_EXISTING_ENTITIES = 3
