r"""
Deterministic acronym glossary + first-use rewriter (MCAL_PLAN 3.8, 4 Q1,
build item #3).

MCAL_PLAN 1(11): undefined acronyms are the only failure mode present in 8/8
graded docs. MCAL_PLAN 4 Q1 is explicit about the remedy -- "This is a
post-processor, not a prompt instruction ... prompt-only enforcement failed 8/8
in the Evaluation CSV. Determinism beats persuasion." So there is deliberately
NO LLM CALL anywhere in this module, and there must never be one: an LLM asked
to expand an unknown acronym will invent a plausible expansion, which is the
exact failure (T03-style fabrication) the rest of M-Cal exists to prevent.

Two passes:

  PRE-PASS  build_glossary(doc)   once per doc. Harvests (token, expansion)
            pairs from parenthetical co-locations in both directions and from
            any front/back-matter Abbreviations/Acronyms/Glossary section.
  POST-PASS annotate_field(text, glossary)  once per output field. Rewrites the
            FIRST occurrence of each known acronym to "Full Name (FN)" and
            leaves later occurrences alone. Unknown acronyms are TAGGED
            (T04_undefined_acronym -> PASS_WITH_NOTE) and never rewritten.

Priority of sources, highest first: an explicit doc glossary section, then
parenthetical definitions in the doc body, then the curated
`acronym_commons.v(N).json` seed. Plan order is "doc glossary takes priority
over commons"; the section-over-parenthetical refinement is ours -- a document
that ships a glossary table has told us its own house style, while a
parenthetical harvest can be misled by OCR noise.

Empirical findings on the 21 materialized docs (see
tests/test_acronyms.py::TestAgainstCorpus, which recomputes these):

  * ZERO of the 21 docs contain an Abbreviations/Acronyms/Glossary heading.
    These are 1969-1980 statements and they predate that convention; the only
    "List of ..." headings present are LIST OF TABLES / FIGURES / COMMENTERS.
    The section parser is therefore dead weight on today's corpus but is kept
    because MCAL_PLAN 3.8 requires it and Segment B runs on ~2,000 docs, many
    post-1990, where the convention is near-universal.
  * The plan's candidate regex `\b([A-Z][A-Z0-9]{1,}[A-Z0-9])\b` matches 5,696
    distinct strings across those 21 docs, and the overwhelming majority are
    ordinary words in ALL-CAPS headings ("THE" 2,493 hits, "AND" 760,
    "PROPERTY" 1,108). A static denylist cannot cover that tail, so the
    ordinary-word test is evidence-based: a candidate whose lowercase form
    occurs >= LOWERCASE_EVIDENCE_MIN times as a standalone word in the same
    document is ordinary English, not an acronym. Measured separation is
    clean -- EPA/EIS/HUD/CDBG/ORV/NHTSA/GVWR all have exactly 0 lowercase
    occurrences, while THE/AND/AREA/STATE have thousands.
  * That test is applied ONLY to tokens with no known expansion, i.e. only to
    the decision "should I tag this as undefined?". A token that IS in the doc
    glossary or in commons is always treated as an acronym. Otherwise "LOS"
    (level of service) would be silently demoted in the Los Angeles Transit
    doc, where "los" appears in lowercase inside "Los Angeles" hundreds of
    times.

Known limits, called out so they are not mistaken for bugs:
  * The plan's regex requires >= 3 characters, so 2-letter acronyms are
    invisible to it. Rather than widen it -- 2-letter all-caps strings in OCR
    are dominated by state postal codes, roman numerals and column headers, so
    widening trades one false negative for a flood of false positives -- short
    tokens are detected ONLY when they are curated in the commons seed. `EA`
    therefore works; an uncurated 2-letter acronym is never detected and never
    tagged. Same mechanism as the dotted/mixed-case forms below.
  * Detection is case-sensitive by construction. An acronym that OCR
    lowercased is not recovered.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field as dc_field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from rapidfuzz import fuzz

from mcal import settings
from mcal.quote_check import normalize

# segment_a bridge is installed by the mcal.settings import above.
from pages import Doc  # noqa: E402


# --- Taxonomy / verdict vocabulary ------------------------------------------

TAG_UNDEFINED_ACRONYM = "T04_undefined_acronym"

# MCAL_PLAN 3.8: an undefined acronym degrades the field to PASS_WITH_NOTE. It
# is not a hard failure -- the extracted prose is still correct, it is just
# unglossed -- so it must not cascade to RE_EXTRACT or HUMAN_REVIEW.
VERDICT_UNDEFINED = "PASS_WITH_NOTE"
VERDICT_CLEAN = "PASS"


# --- Candidate detection ----------------------------------------------------

# Verbatim from MCAL_PLAN 3.8. Minimum length is 3 (one lead letter, >=1 middle,
# one trailing); see the module docstring on why we do not widen it.
ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,}[A-Z0-9])\b")

# Tokens the plan's regex structurally cannot express: dotted and mixed-case
# chemical/pollutant symbols (PM2.5, NOx) and 2-letter tokens (EA). Derived from
# the commons seed rather than hand-listed twice, so the two can never drift
# apart, and deliberately closed: only CURATED odd-form tokens are detectable,
# so this path cannot generate unknown-acronym noise. Matched BEFORE ACRONYM_RE
# and allowed to mask it -- otherwise "PM2.5" is detected as the meaningless
# token "PM2".
def _special_form_pattern(tokens: Iterable[str]) -> re.Pattern:
    odd = sorted(
        (t for t in tokens if not ACRONYM_RE.fullmatch(t)),
        key=len,
        reverse=True,  # longest-first so PM2.5 wins over PM10-style prefixes
    )
    if not odd:
        return re.compile(r"(?!x)x")  # never matches
    alt = "|".join(re.escape(t) for t in odd)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alt})(?![A-Za-z0-9])")


# --- Denylist ---------------------------------------------------------------
# MCAL_PLAN 3.8 names three classes: roman numerals I-XX, section markers, and
# 2-letter state postal codes co-located with place names. The fourth class
# (ordinary all-caps English) is not in the plan but is unavoidable in
# practice -- see the module docstring measurements.

_ROMAN_NUMERALS = frozenset(
    [
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
        "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
    ]
)

# Structural/reference markers. These behave like acronyms to the regex but
# gloss to nothing useful ("SEC." -> "Section" adds no information).
_SECTION_MARKERS = frozenset(
    """
    SEC SECT SECS NO NOS PP PG PGS FIG FIGS TBL TBLS APP APPX CHAP CHAPS
    VOL VOLS PT PTS PARA PARAS ART ARTS ITEM ITEMS EXH EXHS ATT ATTS
    TABLE TABLES FIGURE FIGURES APPENDIX APPENDICES SECTION SECTIONS
    CHAPTER CHAPTERS VOLUME PAGE PAGES EXHIBIT PLATE PLATES CONTENTS
    """.split()
)

# Ordinary English (and NEPA-boilerplate) words that appear in ALL CAPS in
# headings, table columns and cover pages. Seeded from the 200 most frequent
# regex candidates across the 21 materialized docs. This is the static half of
# the ordinary-word filter; the evidence-based half lives in
# Glossary.ordinary_words and catches the long tail.
_ORDINARY_CAPS = frozenset(
    """
    THE AND NOT ALL ANY MAY SHALL FOR WITH FROM THIS THAT THESE THOSE WHICH
    WHO WHOM WHOSE ARE WAS WERE BEEN BEING HAVE HAS HAD WILL WOULD CAN COULD
    SHOULD MUST BUT NOR ITS OUR THEIR THEM THEY YOU YOUR ALSO SUCH BOTH EACH
    MORE MOST SOME ONLY SAME THAN THEN THERE HERE INTO OVER UNDER ABOUT ABOVE
    BELOW BETWEEN DURING AFTER BEFORE WITHIN WITHOUT PER VIA UPON
    AREA AREAS STATE STATES CITY COUNTY TOWN LAND LANDS WATER AIR NOISE
    PROJECT PROJECTS PROPOSED ACTION ACTIONS ALTERNATIVE ALTERNATIVES IMPACT
    IMPACTS ENVIRONMENT ENVIRONMENTAL STATEMENT SUMMARY DRAFT FINAL REVIEW
    COMMENT COMMENTS RESPONSE RESPONSES AGENCY AGENCIES FEDERAL LOCAL PUBLIC
    REGIONAL NATIONAL URBAN RURAL COST COSTS TOTAL LEVEL LEVELS QUALITY
    NUMBER NAME NAMED ADDRESS DATE TERM TERMS TYPE TYPES SOURCE SOURCES USE
    USES LOCATION SITE SITES NORTH SOUTH EAST WEST LONG SHORT HIGH LOW
    PROPERTY STREET SQUARE POINT POINTS CODE CODES PARTY PARTIES PERSON
    PERSONS PEOPLE DEPARTMENT DIVISION OFFICE BUREAU COMMISSION BOARD
    DESCRIPTION EVALUATION ANALYSIS ANALYSES CONTROL CATEGORY CLASS GENERAL
    SPECIAL FUTURE PRESENT EXISTING ADVERSE BENEFICIAL PROGRAM PROGRAMS
    PLANNING DEVELOPMENT CONSTRUCTION OPERATION OPERATING MITIGATION
    POPULATION HOUSING EMPLOYMENT ECONOMIC SOCIAL TRAFFIC TRANSIT TRANSPORT
    TRANSPORTATION HIGHWAY RAIL BUS ROAD ROADS ENERGY FUEL VEHICLE VEHICLES
    RESOURCE RESOURCES SPECIES HABITAT SOIL SOILS CLIMATE WEATHER
    RECEIVED PREDICTED ESTIMATED REQUIRED PROVIDED INCLUDED CONTINUED
    MEASURES EFFECTS EFFECT RELATIONSHIP PARTICIPATION PRODUCTIVITY
    DISPOSITION OCCUPATION CONTRACTOR SECURITY DISTANCE HOUR HOURS YEAR YEARS
    RATE RATES REGION REGIONS CORE BUILD BUILDING BUILDINGS SUBJECT
    ADMINISTRATION SECRETARY DIRECTOR GOVERNOR MAYOR CHAIRMAN
    NONE YES NEW OLD ONE TWO THREE FOUR FIVE SIX TEN
    """.split()
)

# 2-letter USPS state codes. MCAL_PLAN 3.8 asks us to deny these when
# co-located with a place name. Note this rule is nearly inert given the
# plan's own >=3-char regex: "IL" is never a candidate in the first place. It
# is implemented anyway because commons/glossary-section entries are not
# regex-gated, and because a later stage may widen the regex.
_STATE_CODES = frozenset(
    """
    AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS
    MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV
    WI WY DC PR VI GU AS MP
    """.split()
)

# "Springfield, IL" / "Springfield, Ill." -- a capitalized word plus a comma
# immediately before the token is the place-name co-location signal.
_PLACE_CONTEXT_RE = re.compile(r"[A-Z][a-z]+\s*,\s*$")


def is_denylisted(token: str, left_context: str = "") -> bool:
    """
    True when `token` must never be treated as an acronym (MCAL_PLAN 3.8).

    `left_context` is the text immediately preceding the token; it is only
    consulted for the state-postal-code rule, which is conditional on place-name
    co-location ("Ontario, OR" is Oregon, but "OR" in "AND/OR" is not).
    """
    t = (token or "").strip().upper()
    if not t:
        return True
    if t in _ROMAN_NUMERALS:
        return True
    if t in _SECTION_MARKERS:
        return True
    if t in _ORDINARY_CAPS:
        return True
    if t in _STATE_CODES and _PLACE_CONTEXT_RE.search(left_context or ""):
        return True
    return False


# --- Expansion plausibility -------------------------------------------------
# "Validate that a candidate expansion's initials actually match the acronym
# letters" is the single most important guard in this module: without it, the
# direction-A harvest ("<words> (FN)") happily annexes whatever words happen to
# precede a parenthesis.

# Words a token letter may skip over. All are also legal MATCH targets -- "LOS"
# = "Level Of Service" takes its O from a stopword -- so the matcher tries both.
_SKIPPABLE = frozenset(
    """
    of and the for a an to in on at or as by with from de la del les des du
    u.s. u.s us usa united states
    """.split()
)

# A candidate expansion may never START with one of these. "u.s." and "united"
# are deliberately absent -- "U.S. Army Corps of Engineers" is a real expansion.
_LEADING_STOPWORDS = frozenset(
    "of and the for a an to in on at or as by with from de la del les des du".split()
)

# A single word may supply at most this many acronym letters ("Rulemaking" ->
# R,M in NPRM; "greenhouse" -> G,H in GHG). Uncapped, the matcher would accept
# almost any long word for almost any acronym.
MAX_LETTERS_PER_WORD = 3
# ...and at most this many words may supply more than one letter.
MAX_MULTI_LETTER_WORDS = 2
# Guard against a candidate expansion that is merely a long run of prose.
MAX_EXPANSION_WORDS_FACTOR = 2
MAX_EXPANSION_WORDS_SLACK = 2
MAX_EXPANSION_CHARS = 120

# Relaxed floor used for glossary-section table rows, where the table layout is
# itself strong structural evidence and the expansion often carries trailing
# qualifiers the initials cannot account for.
GLOSSARY_MIN_COVERAGE = 0.5

_WORD_SPLIT_RE = re.compile(r"[\s\-\u2013\u2014/(),]+")


def _expansion_words(expansion: str) -> list[str]:
    """
    Split an expansion into initial-bearing words.

    Hyphens and slashes split: "right-of-way" must yield R, O, W for ROW, and
    "high-occupancy vehicle" must yield H, O, V for HOV.
    """
    return [w for w in _WORD_SPLIT_RE.split(expansion or "") if w]


def _letters(token: str) -> list[str]:
    """
    Acronym letters to account for, lowercased, digits dropped.

    Digits are dropped rather than matched: in "PM2.5" and "SO2" the digit is
    part of a chemical symbol, not an initial, and no expansion word supplies
    it.
    """
    return [c.lower() for c in (token or "") if c.isalpha()]


def _word_letters(word: str) -> str:
    return "".join(c.lower() for c in word if c.isalpha())


def initials_match(token: str, expansion: str, *, max_letters_per_word: int = MAX_LETTERS_PER_WORD) -> bool:
    """
    Strict check: every acronym letter is accounted for, in order, and every
    unaccounted-for word is skippable.

    With `max_letters_per_word > 1`, a word may supply several consecutive
    letters as long as the first of them is the word's own first letter --
    required by real NEPA usage such as "Notice of Proposed Rulemaking" (NPRM),
    where M comes from mid-word. That permissiveness has a cost: it lets a word
    be silently dropped from the front of an expansion ("average annual daily
    traffic" -> "average daily traffic" for AADT, because "average" can supply
    both As). Callers that search over candidate expansions therefore try
    `max_letters_per_word=1` first; see `_expansion_before`.
    """
    letters = _letters(token)
    words = _expansion_words(expansion)
    if not letters or not words:
        return False
    if len(expansion) > MAX_EXPANSION_CHARS:
        return False
    if len(words) > MAX_EXPANSION_WORDS_FACTOR * len(letters) + MAX_EXPANSION_WORDS_SLACK:
        return False

    wl = [_word_letters(w) for w in words]
    skippable = [w.lower().strip(".,;:") in _SKIPPABLE or not wl[i] for i, w in enumerate(words)]
    n, m = len(letters), len(words)

    @lru_cache(maxsize=None)
    def go(i: int, j: int, multi: int) -> bool:
        if i == n:
            return all(skippable[k] for k in range(j, m))
        if j == m:
            return False
        w = wl[j]
        if w and w[0] == letters[i]:
            # Consume k letters from this word: the first is the word initial,
            # the rest must appear in order inside the remainder of the word.
            pos = 0
            k = 1
            while True:
                if go(i + k, j + 1, multi + (1 if k > 1 else 0)):
                    return True
                if k >= max_letters_per_word or i + k >= n:
                    break
                if multi + 1 > MAX_MULTI_LETTER_WORDS:
                    break
                nxt = w.find(letters[i + k], pos + 1)
                if nxt < 0:
                    break
                pos = nxt
                k += 1
        if skippable[j] and go(i, j + 1, multi):
            return True
        return False

    try:
        return go(0, 0, 0)
    finally:
        go.cache_clear()


def initials_coverage(token: str, expansion: str) -> float:
    """
    Fraction of the acronym's letters recoverable, in order, from word initials.

    Longest-common-subsequence between the acronym's letters and the expansion's
    word initials. Used for the relaxed glossary-table gate, where a strict
    match is too demanding ("ADT  average daily traffic count, 1975" is a real
    row shape).
    """
    letters = _letters(token)
    inits = [wl[0] for wl in (_word_letters(w) for w in _expansion_words(expansion)) if wl]
    if not letters or not inits:
        return 0.0
    prev = [0] * (len(inits) + 1)
    for a in letters:
        cur = [0]
        for j, b in enumerate(inits):
            cur.append(prev[j] + 1 if a == b else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1] / len(letters)


def plausible_expansion(
    token: str,
    expansion: str,
    *,
    strict: bool = True,
    max_letters_per_word: int = MAX_LETTERS_PER_WORD,
) -> bool:
    """
    Gate a harvested (token, expansion) pair.

    `strict=True` (parenthetical harvest, commons validation) demands a full
    initials match. `strict=False` (glossary-section rows) demands only that the
    first letter line up and that GLOSSARY_MIN_COVERAGE of the letters be
    recoverable.
    """
    exp = (expansion or "").strip()
    if not exp or len(exp) > MAX_EXPANSION_CHARS:
        return False
    words = _expansion_words(exp)
    if not words:
        return False
    # An expansion never begins with a function word. Without this rule the
    # shortest-first search in `_expansion_before` returns "of Planning
    # Coordination" for OPC, because O can legally match "of".
    if words[0].lower().strip(".,") in _LEADING_STOPWORDS:
        return False
    # An expansion must contain lowercase prose or be multi-word; "EIS (FEIS)"
    # is a cross-reference, not a definition.
    if len(words) == 1 and (words[0].isupper() or len(_word_letters(words[0])) < 4):
        return False
    if initials_match(token, exp, max_letters_per_word=max_letters_per_word):
        return True
    if strict:
        return False
    # Relaxed path. Note the first-initial requirement is why this is a fallback
    # rather than the primary test: "U.S. Environmental Protection Agency" starts
    # with a skippable prefix, and only the strict matcher above knows that.
    letters = _letters(token)
    first = _word_letters(words[0])
    if not letters or not first or first[0] != letters[0]:
        return False
    return initials_coverage(token, exp) >= GLOSSARY_MIN_COVERAGE


# --- Commons seed (MCAL_PLAN 3.8 fallback, artifact acronym_commons.v(N).json)

# The ~40 NEPA entries named in MCAL_PLAN 3.8, in plan order. Expansions are
# written the way a general reader should see them, because they are inserted
# verbatim into user-facing summaries. Agency names keep the "U.S." prefix where
# that is the agency's own style.
COMMONS_SEED: dict[str, str] = {
    "EIS": "Environmental Impact Statement",
    "NEPA": "National Environmental Policy Act",
    "CEQ": "Council on Environmental Quality",
    "EPA": "U.S. Environmental Protection Agency",
    "USACE": "U.S. Army Corps of Engineers",
    "NOAA": "National Oceanic and Atmospheric Administration",
    "USFWS": "U.S. Fish and Wildlife Service",
    "USFS": "U.S. Forest Service",
    "BLM": "Bureau of Land Management",
    "DOT": "U.S. Department of Transportation",
    "FHWA": "Federal Highway Administration",
    "FAA": "Federal Aviation Administration",
    "ROD": "Record of Decision",
    "FONSI": "Finding of No Significant Impact",
    "DEIS": "Draft Environmental Impact Statement",
    "FEIS": "Final Environmental Impact Statement",
    "SEIS": "Supplemental Environmental Impact Statement",
    "EA": "Environmental Assessment",
    "LEDPA": "Least Environmentally Damaging Practicable Alternative",
    "NHPA": "National Historic Preservation Act",
    "ESA": "Endangered Species Act",
    "CWA": "Clean Water Act",
    "CAA": "Clean Air Act",
    "NAAQS": "National Ambient Air Quality Standards",
    "PM2.5": "particulate matter under 2.5 micrometers in diameter",
    "VOC": "volatile organic compounds",
    "NOx": "nitrogen oxides",
    "SO2": "sulfur dioxide",
    "MSAT": "mobile source air toxics",
    "GHG": "greenhouse gases",
    "CO2": "carbon dioxide",
    "VMT": "vehicle miles traveled",
    "HOV": "high-occupancy vehicle",
    "LOS": "level of service",
    "ADT": "average daily traffic",
    "ROW": "right-of-way",
    "DBE": "Disadvantaged Business Enterprise",
    "MBE": "Minority Business Enterprise",
    "SHPO": "State Historic Preservation Officer",
    "THPO": "Tribal Historic Preservation Officer",
}

SPECIAL_FORM_RE = _special_form_pattern(COMMONS_SEED)

COMMONS_ARTIFACT_NAME = "acronym_commons.json"

SOURCE_GLOSSARY_SECTION = "glossary_section"
SOURCE_PARENTHETICAL = "parenthetical"
SOURCE_COMMONS = "commons"

# Higher wins when the same token is defined twice.
_SOURCE_RANK = {
    SOURCE_GLOSSARY_SECTION: 3,
    SOURCE_PARENTHETICAL: 2,
    SOURCE_COMMONS: 1,
}


# --- Entry / Glossary types -------------------------------------------------


@dataclass(frozen=True)
class AcronymEntry:
    """One (token -> expansion) binding and where it came from."""

    token: str
    expansion: str
    source: str
    page: Optional[int] = None
    n_observed: int = 1

    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "expansion": self.expansion,
            "source": self.source,
            "page": self.page,
            "n_observed": self.n_observed,
        }

    def first_use_form(self, *, inside_parens: bool = False) -> str:
        """
        Rendering of a first use.

        MCAL_PLAN 3.8 specifies `"Full Name (FN)"`. When the occurrence is
        itself already parenthesised -- "the statement (EIS) says" -- that form
        would nest parentheses, so we emit "Full Name, FN" inside the existing
        pair instead. Both forms are recognised by `_already_defined`, which is
        what makes the post-pass idempotent.
        """
        if inside_parens:
            return f"{self.expansion}, {self.token}"
        return f"{self.expansion} ({self.token})"


@dataclass
class Glossary:
    """
    Per-document acronym glossary, plus the commons fallback it defers to.

    `entries` is doc-derived and authoritative. `commons` is the curated seed.
    `rejected` retains discarded (token, expansion, reason) triples so a bad
    harvest can be diagnosed without re-running the pre-pass.
    """

    doc_id: str
    entries: dict[str, AcronymEntry] = dc_field(default_factory=dict)
    commons: dict[str, AcronymEntry] = dc_field(default_factory=dict)
    rejected: list[dict] = dc_field(default_factory=list)
    # Candidates whose lowercase form is well attested in this document, i.e.
    # ordinary words in an ALL-CAPS heading rather than acronyms.
    ordinary_words: frozenset = frozenset()
    pages_scanned: int = 0
    sections_found: list[dict] = dc_field(default_factory=list)

    # -- lookup --

    def entry_for(self, token: str) -> Optional[AcronymEntry]:
        t = canonical_token(token)
        if not t:
            return None
        return _resolve(self.entries, t) or _resolve(self.commons, t)

    def expand(self, token: str) -> Optional[str]:
        """Expansion for `token`, doc glossary first, else commons, else None."""
        e = self.entry_for(token)
        return e.expansion if e else None

    def source_for(self, token: str) -> Optional[str]:
        e = self.entry_for(token)
        return e.source if e else None

    def known(self, token: str) -> bool:
        return self.entry_for(token) is not None

    def is_probably_ordinary(self, token: str) -> bool:
        """
        Evidence-based ordinary-English test (see module docstring).

        Only meaningful for UNKNOWN tokens: a token with an expansion is an
        acronym regardless of how often its letters appear in lowercase.
        """
        return canonical_token(token).lower() in self.ordinary_words

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, token: object) -> bool:
        return isinstance(token, str) and self.known(token)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "pages_scanned": self.pages_scanned,
            "n_doc_entries": len(self.entries),
            "n_commons_entries": len(self.commons),
            "sections_found": self.sections_found,
            "entries": [self.entries[t].to_dict() for t in sorted(self.entries)],
            "rejected": self.rejected,
        }


def canonical_token(token: str) -> str:
    """
    Canonical lookup key for a token.

    Strips a possessive/plural apostrophe-s ("NPA's" -> "NPA"), which 1970s
    typewriter style uses for plain plurals. Case is preserved because
    mixed-case forms are meaningful ("NOx" is not "NOX"); lookup falls back to
    the uppercase form for robustness.
    """
    t = (token or "").strip()
    t = re.sub(r"['\u2019]s$", "", t)
    return t


def _resolve(mapping: Mapping[str, AcronymEntry], token: str) -> Optional[AcronymEntry]:
    """Exact lookup, then an uppercase retry ("NOX" finds the "NOx" entry)."""
    if token in mapping:
        return mapping[token]
    return mapping.get(token.upper())


# --- Commons artifact I/O ---------------------------------------------------


def commons_entries() -> dict[str, AcronymEntry]:
    """The built-in seed as AcronymEntry objects."""
    return {
        tok: AcronymEntry(token=tok, expansion=exp, source=SOURCE_COMMONS)
        for tok, exp in COMMONS_SEED.items()
    }


def write_commons_seed(stage: str, *, draft: bool = False) -> Path:
    """
    Write `acronym_commons.v(N).json` (MCAL_PLAN 2).

    Schema is exactly the one the artifact table specifies:
    `{"acronyms": [{"token", "expansion", "sources": []}]}`. `sources` is empty
    for every seed row on purpose -- these are curated NEPA-domain defaults with
    no per-document provenance, and Segment B must be able to tell them apart
    from doc-derived bindings.
    """
    path = settings.artifact_path(COMMONS_ARTIFACT_NAME, stage, draft=draft)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "acronyms": [
            {"token": tok, "expansion": COMMONS_SEED[tok], "sources": []}
            for tok in sorted(COMMONS_SEED)
        ]
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_commons(stage: Optional[str] = None, *, draft: bool = False) -> dict[str, AcronymEntry]:
    """
    Load the commons artifact for `stage`, falling back to the built-in seed.

    The fallback is deliberate rather than an error: the post-pass must never be
    the reason a Segment B run halts, and an out-of-date commons file degrades
    output quality far less than a missing one. `stage=None` uses the latest
    promoted stage on disk.
    """
    if stage is None:
        stage = settings.latest_stage()
    if stage is None:
        return commons_entries()
    path = settings.artifact_path(COMMONS_ARTIFACT_NAME, stage, draft=draft)
    if not path.exists():
        return commons_entries()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return commons_entries()
    out: dict[str, AcronymEntry] = {}
    for row in payload.get("acronyms") or []:
        tok = (row.get("token") or "").strip()
        exp = (row.get("expansion") or "").strip()
        if tok and exp:
            out[tok] = AcronymEntry(token=tok, expansion=exp, source=SOURCE_COMMONS)
    return out or commons_entries()


# --- Occurrence scanning ----------------------------------------------------


@dataclass(frozen=True)
class Occurrence:
    """One candidate acronym occurrence in a piece of text."""

    token: str
    start: int
    end: int


def iter_occurrences(text: str) -> list[Occurrence]:
    """
    All candidate acronym occurrences in `text`, left to right, non-overlapping.

    Special forms (PM2.5, NOx) are matched first and mask the general regex, so
    "PM2.5" is never reported as "PM2".
    """
    if not text:
        return []
    spans: list[Occurrence] = []
    taken: list[tuple[int, int]] = []
    for m in SPECIAL_FORM_RE.finditer(text):
        spans.append(Occurrence(m.group(0), m.start(), m.end()))
        taken.append((m.start(), m.end()))
    for m in ACRONYM_RE.finditer(text):
        if any(m.start() < e and s < m.end() for s, e in taken):
            continue
        spans.append(Occurrence(m.group(1), m.start(), m.end()))
    spans.sort(key=lambda o: o.start)
    return spans


# --- Pre-pass: parenthetical harvest ----------------------------------------

# "Full Name (FN)". The optional 's / 's inside the parens covers "(NPA's)".
_PAREN_TOKEN_RE = re.compile(
    r"\(\s*([A-Z][A-Z0-9]{1,}[A-Z0-9]|" + "|".join(
        re.escape(t) for t in sorted(COMMONS_SEED, key=len, reverse=True)
        if not ACRONYM_RE.fullmatch(t)
    ) + r")['\u2019]?s?\s*\)"
)

# "FN (Full Name)".
_TOKEN_PAREN_RE = re.compile(
    r"\b([A-Z][A-Z0-9]{1,}[A-Z0-9])['\u2019]?s?\s*\(\s*([^()\n]{4,120}?)\s*\)"
)

# Where a candidate expansion may start. A period ends it, unless it belongs to
# an abbreviation ("U.S. Army Corps of Engineers (USACE)"), in which case it is
# preceded by a single capital letter.
_EXPANSION_CUT_RE = re.compile(r"[;:!?\)\]\"\u201c\u201d]|(?<![A-Z])\.(?=\s|$)")

_MAX_WORDS_BEFORE = 14


def _clean_expansion(raw: str) -> str:
    """Whitespace-collapse and strip framing punctuation from a raw expansion."""
    s = re.sub(r"\s+", " ", (raw or "").replace("\u00ad", "")).strip()
    s = s.strip(" \t,;:\u2013-")
    s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.IGNORECASE)
    # A possessive belongs to the sentence, not to the agency's name:
    # "FEDERAL HIGHWAY ADMINISTRATION'S (FHWA)".
    s = re.sub(r"['\u2019][sS]$", "", s)
    return s.strip()


def _decase_expansion(expansion: str) -> str:
    """
    Title-case an ALL-CAPS expansion.

    Definitions on 1970s cover pages are frequently set in full caps
    ("BUREAU OF LAND MANAGEMENT (BLM)"). Inserting that verbatim into a summary
    reads as shouting, so multi-word all-caps expansions are down-cased, keeping
    skippable words ("of", "and") lowercase except in first position. Mixed-case
    expansions are left exactly as the document wrote them.
    """
    words = expansion.split()
    if len(words) < 2:
        return expansion
    letters = [c for c in expansion if c.isalpha()]
    if not letters or any(c.islower() for c in letters):
        return expansion
    out: list[str] = []
    for i, w in enumerate(words):
        low = w.lower()
        if i > 0 and low.strip(".,") in _SKIPPABLE:
            out.append(low)
        else:
            out.append(low[:1].upper() + low[1:])
    return " ".join(out)


def _words_before(text: str, end: int) -> list[str]:
    """Candidate expansion words immediately preceding `end`, in order."""
    head = text[max(0, end - 400) : end]
    cuts = [m.end() for m in _EXPANSION_CUT_RE.finditer(head)]
    if cuts:
        head = head[cuts[-1] :]
    words = [w for w in re.split(r"\s+", head.replace("\u00ad", "")) if w.strip(" \t")]
    return words[-_MAX_WORDS_BEFORE:]


def _expansion_before(text: str, start: int, token: str) -> Optional[str]:
    """
    Shortest run of words ending at `start` whose initials match `token`.

    Two nested searches, and the order matters empirically:

      1. shortest-first over candidate lengths, with each word allowed to supply
         exactly ONE letter;
      2. only if that finds nothing, the same sweep allowing intra-word letters.

    Shortest-first stops "the Bureau of Land Management (BLM)" from annexing
    "the". One-letter-per-word first stops the intra-word rule from swallowing a
    leading word: on the Airport Spur doc the text reads "annual average daily
    traffic (AADT)", and a multi-letter-first search returns "average daily
    traffic" because "average" can supply both As. Measured on the 8 graded docs
    this ordering fixes AADT, AASHO and EES without breaking NPRM (which has no
    one-letter-per-word solution and is found on the second sweep).
    """
    words = _words_before(text, start)
    if not words:
        return None
    letters = _letters(token)
    if not letters:
        return None
    limit = min(len(words), MAX_EXPANSION_WORDS_FACTOR * len(letters) + MAX_EXPANSION_WORDS_SLACK)
    for per_word in (1, MAX_LETTERS_PER_WORD):
        for k in range(1, limit + 1):
            cand = _clean_expansion(" ".join(words[-k:]))
            if cand and plausible_expansion(
                token, cand, strict=True, max_letters_per_word=per_word
            ):
                return _decase_expansion(cand)
    return None


def harvest_parentheticals(text: str, page: Optional[int] = None) -> tuple[list[AcronymEntry], list[dict]]:
    """
    Both parenthetical directions from one chunk of text (MCAL_PLAN 3.8).

    Returns (accepted entries, rejected diagnostics). Rejections are kept
    because a silently-dropped definition is indistinguishable from a document
    that never defined the acronym, and the two call for different fixes.
    """
    accepted: list[AcronymEntry] = []
    rejected: list[dict] = []
    if not text:
        return accepted, rejected

    # Direction A: "Full Name (FN)".
    for m in _PAREN_TOKEN_RE.finditer(text):
        token = canonical_token(m.group(1))
        left = text[max(0, m.start() - 40) : m.start()]
        if is_denylisted(token, left):
            continue
        exp = _expansion_before(text, m.start(), token)
        if exp:
            accepted.append(
                AcronymEntry(token=token, expansion=exp, source=SOURCE_PARENTHETICAL, page=page)
            )
        else:
            rejected.append(
                {
                    "token": token,
                    "expansion": _clean_expansion(" ".join(_words_before(text, m.start()))),
                    "reason": "no_initials_match_before_paren",
                    "page": page,
                }
            )

    # Direction B: "FN (Full Name)".
    for m in _TOKEN_PAREN_RE.finditer(text):
        token = canonical_token(m.group(1))
        left = text[max(0, m.start() - 40) : m.start()]
        if is_denylisted(token, left):
            continue
        exp = _clean_expansion(m.group(2))
        if plausible_expansion(token, exp, strict=True):
            accepted.append(
                AcronymEntry(
                    token=token,
                    expansion=_decase_expansion(exp),
                    source=SOURCE_PARENTHETICAL,
                    page=page,
                )
            )
        elif exp:
            rejected.append(
                {
                    "token": token,
                    "expansion": exp,
                    "reason": "no_initials_match_inside_paren",
                    "page": page,
                }
            )
    return accepted, rejected


# --- Pre-pass: glossary / abbreviations section -----------------------------

GLOSSARY_HEADINGS = (
    "abbreviations",
    "acronyms",
    "glossary",
    "list of acronyms",
    "list of abbreviations",
    "acronyms and abbreviations",
    "abbreviations and acronyms",
    "glossary of terms",
    "acronyms and initialisms",
)

# rapidfuzz ratio floor for a heading match, on OCR-normalized text. 88 is
# tight enough to reject "LIST OF FIGURES" (~71 against "list of acronyms")
# while absorbing a character or two of OCR damage.
HEADING_MATCH_RATIO = 88

# How far past a matched heading to keep parsing rows.
GLOSSARY_MAX_LINES = 200

_HEADING_PREFIX_RE = re.compile(
    r"^\s*(?:(?:CHAPTER|Chapter|SECTION|Section|APPENDIX|Appendix)\s+[\w.\-]+\s*[:.\-]?\s*"
    r"|[A-Z]\.\s+|[IVXLC]+\.\s+|\d+(?:\.\d+)*\s*[:.\-]?\s*)",
)
_HEADING_TAIL_RE = re.compile(r"[\s.\u2013\-]*(?:\d+|[ivxlc]+)?\s*$")
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]{0,20}>")

# Table row: token, then a column separator (tab, 2+ spaces, dot leaders, dash
# run, or a spaced dash/colon), then the expansion.
_TABLE_ROW_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9./\-]{1,15})"
    r"(?:\t+|[ ]{2,}|\s*\.{2,}\s*|\s*[-\u2013]{2,}\s*|\s+[-\u2013:=]\s+)"
    r"\s*([^\s].{2,110})$"
)
# Single-space fallback, only used on pages where no table rows were found.
_LOOSE_ROW_RE = re.compile(r"^\s*([A-Z][A-Z0-9./]{2,15})\s+([A-Za-z][^\s].{3,110})$")


def _heading_score(line: str) -> float:
    """Best fuzzy ratio of `line` against the glossary-heading vocabulary."""
    s = _TAG_RE.sub("", line or "").strip()
    if not s or len(s) > 60:
        return 0.0
    s = _HEADING_PREFIX_RE.sub("", s)
    s = _HEADING_TAIL_RE.sub("", s)
    n = normalize(s)
    if not n:
        return 0.0
    return max(fuzz.ratio(n, normalize(h)) for h in GLOSSARY_HEADINGS)


def looks_like_glossary_heading(line: str) -> bool:
    return _heading_score(line) >= HEADING_MATCH_RATIO


def parse_glossary_rows(lines: Sequence[str], page: Optional[int] = None) -> tuple[list[AcronymEntry], list[dict]]:
    """
    Parse rows of an acronyms/abbreviations table.

    MCAL_PLAN 3.8 says to "parse table-formatted entries preferentially", which
    we implement literally: if any 2+-space / tab / dot-leader row parses, the
    single-space fallback is not attempted at all. Mixing the two on one page
    is how "SHALL BE construed as" style prose sneaks in as an entry.
    """
    accepted: list[AcronymEntry] = []
    rejected: list[dict] = []
    table_hit = False
    loose: list[tuple[str, str]] = []

    for raw in lines:
        line = _TAG_RE.sub("", raw or "").rstrip()
        if not line.strip():
            continue
        m = _TABLE_ROW_RE.match(line)
        loose_only = False
        if not m:
            m = _LOOSE_ROW_RE.match(line)
            loose_only = m is not None
        if not m:
            continue
        token = canonical_token(m.group(1))
        if not (ACRONYM_RE.fullmatch(token) or SPECIAL_FORM_RE.fullmatch(token)):
            continue
        if is_denylisted(token):
            continue
        exp = _clean_expansion(re.sub(r"[\s.]*\d*\s*$", "", m.group(2)))
        if not exp:
            continue
        if loose_only:
            loose.append((token, exp))
            continue
        table_hit = True
        if plausible_expansion(token, exp, strict=False):
            accepted.append(
                AcronymEntry(
                    token=token,
                    expansion=_decase_expansion(exp),
                    source=SOURCE_GLOSSARY_SECTION,
                    page=page,
                )
            )
        else:
            rejected.append(
                {"token": token, "expansion": exp, "reason": "implausible_glossary_row", "page": page}
            )

    if not table_hit:
        for token, exp in loose:
            if plausible_expansion(token, exp, strict=True):
                accepted.append(
                    AcronymEntry(
                        token=token,
                        expansion=_decase_expansion(exp),
                        source=SOURCE_GLOSSARY_SECTION,
                        page=page,
                    )
                )
    return accepted, rejected


def _section_windows(doc: Doc, front_pages: int, back_pages: int) -> list:
    """Pages to scan for a glossary heading: front matter plus the last N pages."""
    if not doc.pages:
        return []
    first = doc.pages[0].page_num
    last = doc.pages[-1].page_num
    front_end = min(last, first + front_pages - 1)
    back_start = max(first, last - back_pages + 1)
    keep = set(range(first, front_end + 1)) | set(range(back_start, last + 1))
    return [p for p in doc.pages if p.page_num in keep]


def harvest_glossary_sections(
    doc: Doc,
    *,
    front_pages: int = 30,
    back_pages: int = 30,
) -> tuple[list[AcronymEntry], list[dict], list[dict]]:
    """
    Find Abbreviations/Acronyms/Glossary sections and parse their rows.

    Returns (entries, rejected, sections_found). The window is the plan's:
    front matter (~first 30pp, roman-numbered in modern EISs) plus the last
    30pp, where back-matter glossaries live.
    """
    entries: list[AcronymEntry] = []
    rejected: list[dict] = []
    sections: list[dict] = []
    pages = _section_windows(doc, front_pages, back_pages)
    by_num = {p.page_num: p for p in doc.pages}

    for page in pages:
        lines = page.text.splitlines()
        for i, line in enumerate(lines):
            score = _heading_score(line)
            if score < HEADING_MATCH_RATIO:
                continue
            sections.append(
                {"page": page.page_num, "heading": line.strip()[:80], "score": round(score, 1)}
            )
            # Rows may continue onto the following pages; a glossary is often
            # 2-3 pages long.
            block = lines[i + 1 : i + 1 + GLOSSARY_MAX_LINES]
            got, bad = parse_glossary_rows(block, page=page.page_num)
            entries.extend(got)
            rejected.extend(bad)
            nxt = by_num.get(page.page_num + 1)
            if nxt is not None and got:
                more, more_bad = parse_glossary_rows(
                    nxt.text.splitlines()[:GLOSSARY_MAX_LINES], page=nxt.page_num
                )
                entries.extend(more)
                rejected.extend(more_bad)
            break  # one heading per page is enough
    return entries, rejected, sections


# --- Pre-pass: ordinary-word evidence ---------------------------------------

LOWERCASE_EVIDENCE_MIN = 3

_LOWER_WORD_RE = re.compile(r"\b[a-z][a-z0-9]{1,14}\b")


def _ordinary_word_evidence(page_texts: Iterable[str], candidates: Iterable[str]) -> frozenset:
    """
    Candidates whose lowercase form is well attested in the same document.

    See the module docstring for the measured separation. This is the general
    replacement for an unmaintainable static stoplist: "PROPERTY" appears 1,108
    times in ALL CAPS in this corpus and 382 times in lowercase, so the document
    itself tells us it is a word.
    """
    wanted = {c.lower() for c in candidates}
    if not wanted:
        return frozenset()
    counts: Counter = Counter()
    for text in page_texts:
        for w in _LOWER_WORD_RE.findall(text or ""):
            if w in wanted:
                counts[w] += 1
    return frozenset(w for w, n in counts.items() if n >= LOWERCASE_EVIDENCE_MIN)


# --- Pre-pass entry point ---------------------------------------------------


def build_glossary(
    doc: Doc,
    *,
    commons: Optional[Mapping[str, AcronymEntry]] = None,
    stage: Optional[str] = None,
    front_pages: int = 30,
    back_pages: int = 30,
) -> Glossary:
    """
    Build a document's acronym glossary (MCAL_PLAN 3.8 pre-pass). No LLM calls.

    Parenthetical definitions are harvested from the WHOLE document, not just
    the front matter: NEPA authors define an acronym at its first use, which is
    routinely deep inside a chapter. It is one regex pass over the OCR text.
    The heading scan is restricted to the plan's front/back windows.

    When one token is defined more than once, the winner is chosen by
    (source rank, times observed, earliest page) -- an explicit glossary row
    beats a parenthetical, a repeated definition beats a one-off, and an earlier
    page beats a later one.
    """
    commons_map = dict(commons) if commons is not None else load_commons(stage)

    entries: list[AcronymEntry] = []
    rejected: list[dict] = []

    sec_entries, sec_rejected, sections = harvest_glossary_sections(
        doc, front_pages=front_pages, back_pages=back_pages
    )
    entries.extend(sec_entries)
    rejected.extend(sec_rejected)

    seen_tokens: Counter = Counter()
    for page in doc.pages:
        got, bad = harvest_parentheticals(page.text, page=page.page_num)
        entries.extend(got)
        rejected.extend(bad)
        for occ in iter_occurrences(page.text):
            seen_tokens[canonical_token(occ.token)] += 1

    # Fold duplicates: count observations per (token, normalized expansion).
    grouped: dict[str, dict[str, list[AcronymEntry]]] = {}
    for e in entries:
        grouped.setdefault(e.token, {}).setdefault(normalize(e.expansion), []).append(e)

    best: dict[str, AcronymEntry] = {}
    for token, variants in grouped.items():
        scored = []
        for group in variants.values():
            head = min(group, key=lambda e: (e.page is None, e.page or 0))
            rank = max(_SOURCE_RANK.get(e.source, 0) for e in group)
            scored.append((rank, len(group), -(head.page or 0), head, len(group)))
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        rank, n, _, head, n_obs = scored[0]
        best[token] = AcronymEntry(
            token=head.token,
            expansion=head.expansion,
            source=head.source,
            page=head.page,
            n_observed=n_obs,
        )
        for _, _, _, other, _ in scored[1:]:
            rejected.append(
                {
                    "token": token,
                    "expansion": other.expansion,
                    "reason": f"superseded_by:{head.source}:{head.expansion}",
                    "page": other.page,
                }
            )

    unknown = [t for t in seen_tokens if t not in best and t not in commons_map]
    ordinary = _ordinary_word_evidence((p.text for p in doc.pages), unknown)

    return Glossary(
        doc_id=getattr(doc, "doc_id", ""),
        entries=best,
        commons=commons_map,
        rejected=rejected,
        ordinary_words=ordinary,
        pages_scanned=doc.n_pages,
        sections_found=sections,
    )


# --- Post-pass --------------------------------------------------------------

# How much text before/after an occurrence to inspect when deciding whether the
# acronym is already defined there. The expansion plus a little slack for
# punctuation and an OCR-inserted line break.
_DEFINED_CONTEXT_SLACK = 24


def _already_defined(text: str, occ: Occurrence, expansion: str) -> bool:
    """
    True when the text at `occ` already carries its definition.

    Recognises both plan-specified orders -- "Full Name (FN)" / "Full Name, FN"
    to the left, "FN (Full Name)" to the right -- via `quote_check.normalize`,
    so an OCR-damaged or differently-punctuated definition still counts. Using
    the same normalizer as the quote verifier is deliberate: acronym rewriting
    must not create text whose definition the quote checker cannot see.

    This is the function that makes `annotate_field` idempotent.
    """
    nexp = normalize(expansion)
    if not nexp:
        return False
    span = len(expansion) + _DEFINED_CONTEXT_SLACK
    left = normalize(text[max(0, occ.start - span) : occ.start])
    if left.endswith(nexp):
        return True
    right = normalize(text[occ.end : occ.end + span])
    return right.startswith(nexp)


def _is_parenthesised(text: str, occ: Occurrence) -> bool:
    """True when the occurrence is alone inside its own parentheses: "(EIS)"."""
    before = text[:occ.start].rstrip()
    after = text[occ.end:].lstrip()
    return before.endswith("(") and after.startswith(")")


def annotate_field(
    text: str,
    glossary: Glossary,
    *,
    rewrite: bool = True,
) -> tuple[str, list[str], dict]:
    """
    Rewrite first uses and tag undefined acronyms in one output field.

    Returns `(rewritten_text, tags, stats)`:

      * the first occurrence of each KNOWN acronym becomes "Full Name (FN)";
        later occurrences are untouched, per MCAL_PLAN 3.8;
      * an occurrence that already reads as a definition is counted as
        `already_defined` and left alone -- so running this on its own output is
        a no-op;
      * an UNKNOWN acronym is recorded in `stats["undefined"]` and contributes
        `T04_undefined_acronym` to `tags`. It is NEVER rewritten. MCAL_PLAN 3.8
        and 4 Q1 both forbid inventing an expansion, and no expansion is worse
        than a wrong one only from a copy-editing point of view -- from a
        research-integrity point of view it is the whole ballgame.

    Rewriting walks the ORIGINAL text left to right and appends replacements to
    an output buffer, so inserted expansions are never themselves rescanned.
    `rewrite=False` runs the same accounting without editing, which is what the
    confidence signal needs when a caller only wants to score a field.
    """
    stats = {
        "n_occurrences": 0,
        "n_distinct": 0,
        "rewritten": [],
        "already_defined": [],
        "undefined": [],
        "skipped_ordinary": [],
        "sources": {},
    }
    if not text:
        stats["defined_first_use_rate"] = 1.0
        stats["suggested_verdict"] = VERDICT_CLEAN
        return text, [], stats

    out: list[str] = []
    cursor = 0
    handled: set[str] = set()
    distinct: set[str] = set()
    ordinary_seen: set[str] = set()

    for occ in iter_occurrences(text):
        token = canonical_token(occ.token)
        left = text[max(0, occ.start - 40) : occ.start]
        if is_denylisted(token, left):
            continue
        entry = glossary.entry_for(token)
        # An unknown token the document itself uses in lowercase is an ordinary
        # word in an ALL-CAPS heading, not an acronym. Checked before counting so
        # it cannot drag down defined_first_use_rate. Never applied to KNOWN
        # tokens -- see the module docstring on "LOS" vs "Los Angeles".
        if entry is None and glossary.is_probably_ordinary(token):
            if token not in ordinary_seen:
                ordinary_seen.add(token)
                stats["skipped_ordinary"].append(token)
            continue

        stats["n_occurrences"] += 1
        distinct.add(token)
        if token in handled:
            continue
        handled.add(token)

        if entry is None:
            stats["undefined"].append(token)
            continue

        stats["sources"][token] = entry.source
        if _already_defined(text, occ, entry.expansion):
            stats["already_defined"].append(token)
            continue
        if not rewrite:
            stats["rewritten"].append(token)
            continue
        replacement = entry.first_use_form(inside_parens=_is_parenthesised(text, occ))
        out.append(text[cursor : occ.start])
        out.append(replacement)
        cursor = occ.end
        stats["rewritten"].append(token)

    out.append(text[cursor:])
    rewritten_text = "".join(out) if rewrite else text

    stats["n_distinct"] = len(distinct)
    stats["defined_first_use_rate"] = _rate(stats)
    stats["suggested_verdict"] = (
        VERDICT_UNDEFINED if stats["undefined"] else VERDICT_CLEAN
    )
    tags = [TAG_UNDEFINED_ACRONYM] if stats["undefined"] else []
    return rewritten_text, tags, stats


def annotate_record(
    fields: Mapping[str, str],
    glossary: Glossary,
    *,
    rewrite: bool = True,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, dict]]:
    """
    Run the post-pass over a mapping of field name -> prose.

    Per-field rather than whole-record, because MCAL_PLAN 3.8 defines first use
    "per output field": a reader of `summary.public_response` should not have to
    have read `summary.project_description` to know what LEDPA means. The cost
    is that a doc-level view repeats definitions, which is the correct tradeoff
    for independently-consumed fields.
    """
    texts: dict[str, str] = {}
    tags: dict[str, list[str]] = {}
    stats: dict[str, dict] = {}
    for name, value in (fields or {}).items():
        if not isinstance(value, str):
            texts[name] = value
            continue
        new_text, field_tags, field_stats = annotate_field(value, glossary, rewrite=rewrite)
        texts[name] = new_text
        tags[name] = field_tags
        stats[name] = field_stats
    return texts, tags, stats


# --- Confidence signal (MCAL_PLAN 3.3: s_acronym) ---------------------------

# A field containing no acronyms at all is vacuously fully-defined. Scoring it
# 0.0 would penalise plain-language prose, which MCAL_PLAN 3.14 explicitly
# wants more of.
S_ACRONYM_NO_ACRONYMS = 1.0


def _rate(stats: Mapping) -> float:
    defined = len(stats.get("rewritten") or []) + len(stats.get("already_defined") or [])
    undefined = len(stats.get("undefined") or [])
    total = defined + undefined
    if total == 0:
        return S_ACRONYM_NO_ACRONYMS
    return defined / total


def defined_first_use_rate(stats) -> float:
    """
    `s_acronym` (MCAL_PLAN 3.3): the defined-first-use rate, in [0, 1].

    Denominator is the number of DISTINCT acronyms whose first use we could act
    on; numerator is how many of those ended up defined at first use, whether by
    the document itself (`already_defined`) or by our rewrite (`rewritten`).
    Occurrences after the first are excluded by construction -- MCAL_PLAN 3.8
    wants them left alone, so counting them would be scoring something the
    pipeline deliberately does not do.

    Accepts a single stats dict from `annotate_field` or an iterable of them
    (e.g. every field of a doc). Aggregation is over pooled acronym counts, not
    a mean of per-field rates, so a field with eight acronyms weighs more than a
    field with one.
    """
    if stats is None:
        return S_ACRONYM_NO_ACRONYMS
    if isinstance(stats, Mapping):
        return _rate(stats)
    pooled = {"rewritten": [], "already_defined": [], "undefined": []}
    for s in stats:
        if not isinstance(s, Mapping):
            continue
        for key in pooled:
            pooled[key].extend(s.get(key) or [])
    return _rate(pooled)


def suggested_verdict(tags: Iterable[str]) -> str:
    """
    Critic verdict floor implied by the acronym tags (MCAL_PLAN 3.8).

    An undefined acronym is a PASS_WITH_NOTE, never worse: the extraction is
    correct, it is only unglossed.
    """
    return VERDICT_UNDEFINED if TAG_UNDEFINED_ACRONYM in set(tags or ()) else VERDICT_CLEAN
