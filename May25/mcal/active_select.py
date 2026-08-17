"""
Uncertainty sampling for the next grading batch (MCAL_PLAN 3.6, build item #13).

Emits `artifacts/next_batch.csv` with `doc_id, uncertainty_score,
dominant_predicted_failure_tags` -- ~10 rows, matching the MCAL_PLAN 7.5
recalibration cadence. Runnable as `python -m mcal.active_select --n 10`.

NO LLM calls, by requirement. This runs between every calibration round, and a
selector that costs a dollar per candidate would not get run. Everything below is
regex, page counts, and `chunk.detect_chapters`, so the whole 21-document corpus
scores in about 2 seconds and the result is deterministic given the corpus.

--------------------------------------------------------------------------
What this is, and what it is not: a cold-start heuristic
--------------------------------------------------------------------------

MCAL_PLAN 3.6 asks to "rank candidate docs by predicted composite variance
across fields". That phrasing presumes a predictor of composite, and there is no
such predictor at this stage, for a structural reason: composite is
`0.5*s_quote + 0.5*s_critic` (MCAL_PLAN 3.3), and both inputs are properties of
an EXTRACTION. The candidate pool is precisely the documents that have never been
extracted. So there is nothing to compute a variance of.

Two honest options: extract every candidate first (which costs roughly what
grading the batch costs, and would have to be redone under the next artifact
stage anyway), or predict from features of the raw document. This module does the
second. Concretely:

  1. For each field, take the graded corpus's own error rate as a prior, Laplace
     smoothed. MCAL_PLAN 1 makes the smoothing necessary rather than decorative:
     `title`, `themes`, `lead_agency` and `summary.overview` are 8/8 correct, and
     the plan itself notes "the Wilson interval on 8/8 still admits a true error
     rate around 30%". An unsmoothed 0.0 would make those fields contribute
     exactly zero uncertainty forever, which is the opposite of what a sampler
     that is supposed to explore should do.
  2. Shift that prior in LOG-ODDS space by feature-driven adjustments, each one
     traceable to a specific MCAL_PLAN 1 finding (pre-1978 scans break `year`, a
     missing comment-response chapter breaks `summary.public_response`, and so
     on). Log-odds rather than raw probability so that a "+1 doubling of odds"
     means the same thing whether the prior is 0.1 or 0.5, and so the result
     cannot leave [0, 1].
  3. Bernoulli variance `p(1-p)`, averaged over fields and rescaled by its 0.25
     maximum. This is the uncertainty-sampling half: it peaks at p = 0.5, i.e. on
     the documents whose outcome we genuinely cannot call, and it is low both for
     documents we expect to be clean AND for documents we are confident will fail
     -- a document certain to fail teaches almost as little as one certain to
     pass, because the grade is predictable either way.
  4. Blend with a tag-rarity term that prefers documents predicted to exercise
     failure modes the graded set has few or no examples of. This is what the
     Critic prompt builder is waiting on: `critic_prompt.select_few_shots` does
     greedy set-cover over observed tags per field, and at seed v1 most of
     T08/T09/T10/T13/T14 have zero exemplars, so those prompts are running on
     positive controls alone.

What this is NOT: a calibrated variance estimate. The feature adjustments below
are prior beliefs written down from MCAL_PLAN 1's failure analysis, not
coefficients fitted to anything, and they cannot be fitted until several rounds of
(features, observed grades) pairs exist. They are stated as explicit numbers in
`FIELD_ADJUSTMENTS` so that when that data does exist, the fit has something to
replace. Two known limitations worth stating plainly:

  * `n_distinct_states` is a proxy for multi-site scope and it is a noisy one:
     an EIS that lists every state in a distribution table scores high without
     being multi-site. Measured on this corpus it separates the national
     rulemakings (Off-road vehicles: 30+ states; Endangered species: 30+) from
     the corridor projects (Airport Spur: 6), which is the discrimination we
     want, but it will misfire.
  * The pool is whatever has materialized OCR on this machine, not the ~2000-doc
     corpus. Selection quality is bounded by that; the module reports the pool
     size so the number is never invisible.

--------------------------------------------------------------------------
Candidate pool
--------------------------------------------------------------------------

Materialized docs (per-page OCR JSON under `Documents/output/`) that are NOT yet
graded. As of writing: 21 materialized, 8 graded (the Evaluation sheet), so 13
candidates -- including `p0491_35556036091957`, which has OCR and an (unfilled)
grading sheet but no grades, and `P0491_35556036854362`, whose directory name is
capitalized. Casing on disk is inconsistent and `pages.load_doc` does a
case-SENSITIVE directory lookup, so `doc_id` in the emitted CSV is the on-disk
directory name (directly usable by `load_doc`) while all comparisons against the
grade set go through `settings.normalize_doc_id`.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional, Sequence

from . import grades as grades_mod
from . import settings

# segment_a bridge is installed by the settings import.
from chunk import detect_chapters  # noqa: E402
from pages import Doc, load_doc  # noqa: E402

log = logging.getLogger(__name__)


# --- Cheap document features ------------------------------------------------

# Chapter-heading detection is `chunk.detect_chapters`, which maps OCR'd headings
# onto the six CEQ chapters via `config.CHAPTER_ALIASES`. Those aliases do not
# include a comment-response chapter (it is not a CEQ 1502 chapter) or a
# glossary, so those two get their own regexes.

_COMMENT_RESPONSE_RE = re.compile(
    r"\b(?:comments?\s+and\s+responses?"
    r"|response[s]?\s+to\s+comments?"
    r"|comment\s+letters?"
    r"|letters?\s+of\s+comment"
    r"|coordination\s+and\s+review\s+comments?"
    r"|summary\s+of\s+comments?)\b",
    re.IGNORECASE,
)

_GLOSSARY_RE = re.compile(
    r"\b(?:glossary"
    r"|list\s+of\s+(?:abbreviations|acronyms)"
    r"|abbreviations\s+and\s+acronyms"
    r"|acronyms\s+and\s+abbreviations"
    r"|definition\s+of\s+terms)\b",
    re.IGNORECASE,
)

# Deliberately NOT including the bare word "national": "National Register of
# Historic Places", "National Park Service" and "National Environmental Policy
# Act" appear in nearly every document in this corpus and would make the national
# -scope signal fire everywhere. These phrases are specific to a
# rulemaking/programmatic action, which is the MCAL_PLAN 1(9d) failure (the Fuel
# Economy CAFE document graded "no location").
_NATIONAL_SCOPE_RE = re.compile(
    r"\b(?:nationwide"
    r"|programmatic"
    r"|rulemaking"
    r"|Federal\s+Register"
    r"|notice\s+of\s+proposed\s+rule"
    r"|all\s+(?:fifty|50)\s+states"
    r"|throughout\s+the\s+United\s+States"
    r"|national\s+(?:standard|policy|program|scale)s?)\b",
    re.IGNORECASE,
)

_ROD_RE = re.compile(r"\brecord\s+of\s+decision\b", re.IGNORECASE)

STATE_NAMES = (
    "Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware "
    "Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky "
    "Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi "
    "Missouri Montana Nebraska Nevada Ohio Oklahoma Oregon Pennsylvania "
    "Tennessee Texas Utah Vermont Virginia Washington Wisconsin Wyoming"
).split() + [
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
    "North Dakota", "Rhode Island", "South Carolina", "South Dakota",
    "West Virginia",
]
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(STATE_NAMES, key=len, reverse=True)) + r")\b")

_YEAR_RE = re.compile(r"\b(19[4-9]\d|20[0-2]\d)\b")

# The 1978 CEQ regulations (40 CFR 1500-1508) are the dividing line MCAL_PLAN
# 1(10) / T13 turns on: before them there is no "cooperating agency" category as
# the extractor defines it.
NEPA_REGULATIONS_YEAR = 1978
# All three wrong `year` grades were pre-1980 (MCAL_PLAN 1(1)).
YEAR_OCR_RISK_BEFORE = 1980
# Above this page count a document is chunked into enough map-reduce shards that
# the (entity <-> figure) decoupling MCAL_PLAN 1(3) describes has room to happen.
# 50-page chunks with 2-page overlap (config.CHUNK_PAGES), so ~5+ shards.
LONG_DOC_PAGES = 250
MULTI_SITE_MIN_STATES = 3
NATIONAL_SCOPE_MIN_HITS_PER_100PP = 5.0


@dataclass
class DocFeatures:
    """Cheap, LLM-free observables of one candidate document."""

    doc_id: str                 # on-disk directory name (case as found)
    normalized_doc_id: str
    n_pages: int = 0
    n_chars: int = 0
    year: Optional[int] = None
    year_source: str = "none"   # "inventory" | "front_matter_regex" | "none"
    title: str = ""
    ceq_chapters: list[str] = dc_field(default_factory=list)
    has_comment_response_chapter: bool = False
    has_glossary: bool = False
    n_distinct_states: int = 0
    states: list[str] = dc_field(default_factory=list)
    national_scope_hits: int = 0
    rod_mentions: int = 0

    @property
    def has_alternatives_chapter(self) -> bool:
        return "Alternatives" in self.ceq_chapters

    @property
    def has_consultation_chapter(self) -> bool:
        return "Consultation" in self.ceq_chapters

    @property
    def is_pre_regulations(self) -> Optional[bool]:
        """None when the year is unknown -- do not guess an era."""
        return None if self.year is None else self.year < NEPA_REGULATIONS_YEAR

    @property
    def national_scope_rate(self) -> float:
        """National-scope phrase hits per 100 pages. Length-normalized."""
        if not self.n_pages:
            return 0.0
        return 100.0 * self.national_scope_hits / self.n_pages

    @property
    def looks_national(self) -> bool:
        return self.national_scope_rate >= NATIONAL_SCOPE_MIN_HITS_PER_100PP

    @property
    def looks_multi_site(self) -> bool:
        return self.n_distinct_states >= MULTI_SITE_MIN_STATES

    @property
    def is_long(self) -> bool:
        return self.n_pages >= LONG_DOC_PAGES

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "n_pages": self.n_pages,
            "n_chars": self.n_chars,
            "year": self.year,
            "year_source": self.year_source,
            "title": self.title,
            "ceq_chapters": self.ceq_chapters,
            "has_alternatives_chapter": self.has_alternatives_chapter,
            "has_consultation_chapter": self.has_consultation_chapter,
            "has_comment_response_chapter": self.has_comment_response_chapter,
            "has_glossary": self.has_glossary,
            "n_distinct_states": self.n_distinct_states,
            "national_scope_hits": self.national_scope_hits,
            "national_scope_rate_per_100pp": round(self.national_scope_rate, 2),
            "looks_national": self.looks_national,
            "looks_multi_site": self.looks_multi_site,
            "is_long": self.is_long,
            "is_pre_1978": self.is_pre_regulations,
        }


def _inventory_year(doc_id: str) -> tuple[Optional[int], str]:
    """
    Year from the inventory CSV (260$cg), which is free -- one cached CSV read.

    Deliberately NOT the year adjudicator: that costs a Sonnet call per document
    and this module must stay LLM-free. The inventory year is also the SAME source
    M1 already trusts, so a wrong inventory year does not bias the selector
    relative to what the pipeline will actually do with the document.
    """
    try:
        from inventory import lookup_work

        work = lookup_work(doc_id)
    except Exception as e:  # noqa: BLE001 - a missing inventory must not stop selection
        log.debug("inventory lookup failed for %s: %s", doc_id, e)
        return None, "none"
    if not work:
        return None, "none"
    raw = work.get("create_date")
    if raw:
        m = _YEAR_RE.search(str(raw))
        if m:
            return int(m.group(1)), "inventory"
    return None, "none"


def _front_matter_year(doc: Doc, n_pages: int = 5) -> Optional[int]:
    """
    Modal year in the first few pages. Fallback only.

    Modal rather than first-seen: cover pages of these scans routinely carry a
    project number that looks like a year, and the real date repeats.
    """
    if not doc.pages:
        return None
    head = doc.pages[: max(1, n_pages)]
    years = [int(y) for p in head for y in _YEAR_RE.findall(p.text or "")]
    if not years:
        return None
    counts: dict[int, int] = {}
    for y in years:
        counts[y] = counts.get(y, 0) + 1
    # Ties break towards the later year: a transmittal date postdates a cited one.
    return max(counts, key=lambda y: (counts[y], y))


def extract_features(doc_id: str, doc: Optional[Doc] = None) -> DocFeatures:
    """
    Compute one candidate's features. No LLM, no network.

    `doc` may be injected (tests, callers that already loaded it); otherwise it is
    loaded from the resolved -- case-insensitively -- page directory.
    """
    resolved = settings.resolve_doc_dir(doc_id)
    on_disk = resolved.name if resolved is not None else doc_id
    if doc is None:
        if resolved is None:
            raise FileNotFoundError(
                f"No materialized OCR for doc_id {doc_id!r} under "
                f"{settings.PAGES_DATA_DIR}. Candidates come from "
                f"settings.available_doc_ids(); if that list is empty the corpus "
                f"is not mounted on this machine."
            )
        doc = load_doc(on_disk, settings.PAGES_DATA_DIR)

    full = doc.full_text or ""
    year, year_source = _inventory_year(on_disk)
    if year is None:
        y = _front_matter_year(doc)
        if y is not None:
            year, year_source = y, "front_matter_regex"

    title = ""
    try:
        from inventory import lookup_work

        title = (lookup_work(on_disk) or {}).get("title") or ""
    except Exception:  # noqa: BLE001
        title = ""

    states = sorted({m.group(1) for m in _STATE_RE.finditer(full)})

    return DocFeatures(
        doc_id=on_disk,
        normalized_doc_id=settings.normalize_doc_id(on_disk),
        n_pages=doc.n_pages,
        n_chars=len(full),
        year=year,
        year_source=year_source,
        title=title,
        ceq_chapters=[c["ceq_chapter"] for c in detect_chapters(doc)],
        has_comment_response_chapter=bool(_COMMENT_RESPONSE_RE.search(full)),
        has_glossary=bool(_GLOSSARY_RE.search(full)),
        n_distinct_states=len(states),
        states=states,
        national_scope_hits=len(_NATIONAL_SCOPE_RE.findall(full)),
        rod_mentions=len(_ROD_RE.findall(full)),
    )


# --- Priors -----------------------------------------------------------------

# Laplace smoothing. `(k+1)/(n+2)` on an 8-document field: 0/8 -> 0.10,
# 3/8 -> 0.40, 5/8 -> 0.60. See the module docstring for why the 0/8 fields must
# not be pinned to zero.
LAPLACE_ALPHA = 1.0

# Prior used when a field has no graded items at all -- `summary_of_interest`,
# which is new by construction (MCAL_PLAN 3.15) and starts with zero examples.
# 0.5 is the maximum-uncertainty value, which is the honest encoding of "we have
# never seen this field graded" and also maximizes its variance contribution, so
# candidates are not penalized for the field's novelty.
UNKNOWN_FIELD_PRIOR = 0.5


def field_error_priors(grade_set: "grades_mod.GradeSet") -> dict[str, float]:
    """Smoothed per-field error rate over the graded corpus."""
    out: dict[str, float] = {}
    for field in settings.ALL_FIELDS:
        items = grade_set.for_field(field)
        if not items:
            out[field] = UNKNOWN_FIELD_PRIOR
            continue
        k = sum(1 for i in items if not i.correct)
        n = len(items)
        out[field] = (k + LAPLACE_ALPHA) / (n + 2 * LAPLACE_ALPHA)
    return out


# --- Feature adjustments ----------------------------------------------------
# Log-odds shifts applied to the prior. Each entry is
# `(feature_name, field, delta_log_odds, rationale)`.
#
# `delta = 1.0` roughly triples the odds; `0.5` multiplies them by ~1.65. These
# are PRIOR BELIEFS from MCAL_PLAN 1, not fitted coefficients -- see the module
# docstring. Written as data so the eventual fit has a table to replace and so
# `explain()` can quote the rationale to the reviewer.

FIELD_ADJUSTMENTS: tuple[tuple[str, str, float, str], ...] = (
    # --- era ---
    ("pre_1978", "key_people", 1.2,
     "MCAL_PLAN 1(10)/T13: no 40 CFR 1501.8 cooperating-agency category exists "
     "before the 1978 regulations, so the extractor's core distinction is "
     "undefined."),
    ("pre_1980", "year", 1.2,
     "MCAL_PLAN 1(1): all 3 wrong `year` grades are pre-1980 scans, where the "
     "real date sits on a signature or transmittal page, not the cover."),
    ("pre_1980", "eis_type", 0.6,
     "MCAL_PLAN 1(2): old cover pages are lightly textual, which is how the "
     "Lincoln Hwy Final was read as a ROD."),
    # --- structure ---
    ("no_alternatives_chapter", "alternatives", 1.5,
     "MCAL_PLAN 1(8): the Buffalo Light Rail `alternatives[0]` was empty because "
     "structural identification of the Alternatives chapter failed."),
    ("no_alternatives_chapter", "summary.alternatives_overview", 0.8,
     "The summary subfield reads the same chapter; if it cannot be located the "
     "subfield is synthesized from scattered mentions."),
    ("no_comment_response_chapter", "summary.public_response", 1.2,
     "MCAL_PLAN 1(5), the most common failure (4/8): comment-response tables are "
     "structurally unlike body chapters and citations get dropped."),
    ("long_doc", "summary.project_description", 0.5,
     "MCAL_PLAN 1(3): more map-reduce shards is more opportunity for the reduce "
     "step to pair a figure with the wrong entity."),
    ("long_doc", "summary.environmental_impact", 0.5,
     "MCAL_PLAN 1(4): same mechanism, and this subfield carries the most figures."),
    ("long_doc", "summary.affected_community", 0.3,
     "Population and demographic figures spread across more shards."),
    ("rod_in_body", "eis_type", 0.8,
     "MCAL_PLAN 1(2): the Lincoln Hwy misclassification came from 'Record of "
     "Decision' appearing in body or citation text rather than as a heading."),
    # --- scope ---
    ("multi_site", "location", 1.0,
     "MCAL_PLAN 1(9c): multi-site documents geocoded 1 of 3 primary sites."),
    ("national_scope", "location", 1.5,
     "MCAL_PLAN 1(9d): the Fuel Economy CAFE rulemaking was graded 'no location' "
     "because a national action with no site was treated as absent-location."),
    ("no_glossary", "summary.overview", 0.3,
     "MCAL_PLAN 1(11): acronyms undefined in 8/8. A document with no glossary "
     "gives the acronym post-processor nothing to work from, so unglossed jargon "
     "leaks into the roll-up."),
    ("no_glossary", "summary_of_interest", 0.3,
     "Same, and this field has no graded baseline at all."),
    ("consultation_chapter", "key_people", 0.5,
     "MCAL_PLAN 1(10): the Consultation chapter is exactly where cooperating "
     "agencies, consulted agencies, recipients and commenters get bundled "
     "together and all labeled 'cooperator'."),
)


def active_feature_names(f: DocFeatures) -> list[str]:
    """Which adjustment features this document exhibits."""
    names: list[str] = []
    if f.year is not None:
        if f.year < NEPA_REGULATIONS_YEAR:
            names.append("pre_1978")
        if f.year < YEAR_OCR_RISK_BEFORE:
            names.append("pre_1980")
    if not f.has_alternatives_chapter:
        names.append("no_alternatives_chapter")
    if not f.has_comment_response_chapter:
        names.append("no_comment_response_chapter")
    if not f.has_glossary:
        names.append("no_glossary")
    if f.has_consultation_chapter:
        names.append("consultation_chapter")
    if f.is_long:
        names.append("long_doc")
    if f.looks_multi_site:
        names.append("multi_site")
    if f.looks_national:
        names.append("national_scope")
    if f.rod_mentions:
        names.append("rod_in_body")
    return names


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predicted_error_rates(
    f: DocFeatures, priors: dict[str, float]
) -> dict[str, float]:
    """Per-field predicted error probability, prior shifted in log-odds space."""
    active = set(active_feature_names(f))
    out: dict[str, float] = {}
    for field in settings.ALL_FIELDS:
        z = _logit(priors.get(field, UNKNOWN_FIELD_PRIOR))
        for feat, fld, delta, _why in FIELD_ADJUSTMENTS:
            if fld == field and feat in active:
                z += delta
        out[field] = _sigmoid(z)
    return out


# --- Tag prediction ---------------------------------------------------------
# `(feature_name, tag, weight)`. Weight is confidence that the feature implies
# the tag, in [0, 1]; it scales the rarity term, so a speculative prediction
# contributes less than a near-certain one.

TAG_PREDICTIONS: tuple[tuple[str, str, float], ...] = (
    ("pre_1978", "T13_pre_1978_nepa_format", 0.9),
    ("pre_1980", "T11_year_ocr_error", 0.6),
    ("rod_in_body", "T12_eis_type_confused_with_rod", 0.5),
    ("no_alternatives_chapter", "T10_alternatives_chapter_missed", 0.8),
    ("multi_site", "T09_multi_site_partial_geocode", 0.7),
    ("national_scope", "T08_scope_misclassified_national", 0.8),
    ("regional_scope", "T14_regional_scope_underspecified", 0.5),
    ("no_comment_response_chapter", "T01_missing_citation", 0.7),
    ("long_doc", "T01_missing_citation", 0.4),
    ("long_doc", "T02_numeric_hallucination", 0.5),
    ("no_glossary", "T04_undefined_acronym", 0.8),
    ("no_glossary", "T15_jargon_without_gloss", 0.5),
    ("consultation_chapter", "T05_commenter_mislabeled_as_cooperator", 0.6),
)

DEFAULT_DOMINANT_TAGS = 3


def predicted_tags(f: DocFeatures) -> dict[str, float]:
    """
    `{tag: weight}` predicted for this document. Max weight wins on collision.

    `regional_scope` is derived here rather than in `active_feature_names`
    because it is the complement of `multi_site` at exactly two states -- the
    MCAL_PLAN 1(9)/T14 condition "scope is regional but fewer than two primary
    sites are named" -- and it must not also fire the multi-site adjustment.
    """
    active = set(active_feature_names(f))
    if f.n_distinct_states == 2:
        active.add("regional_scope")
    out: dict[str, float] = {}
    for feat, tag, weight in TAG_PREDICTIONS:
        if feat in active:
            out[tag] = max(out.get(tag, 0.0), weight)
    return out


def tag_counts_in_grades(grade_set: "grades_mod.GradeSet") -> dict[str, int]:
    """
    Observed exemplar count per failure tag in the graded corpus.

    One correction on top of `GradeSet.tag_counts()`. `mcal/grades.py` decision
    (B) deliberately does NOT emit `T04_undefined_acronym` as a failure tag: the
    doc-level note "includes undefined acronyms" is present on 8/8 docs and
    folding it into `y_i` would mark every summary field of every document wrong.
    It is recorded as a per-item `acronym_issue` flag instead. Correct for
    calibration; wrong here, because it would leave T04 looking like a
    zero-exemplar tag at maximum rarity when it is in fact the most thoroughly
    observed failure in the corpus (MCAL_PLAN 1(11), 8/8). So we fold
    `acronym_issue` back in, for rarity purposes only.

    Two deliberate narrowings of that correction:

    * **Counted per DOCUMENT, not per item.** `acronym_issue` is a doc-level note
      copied onto every item of the document, so an item-level count returns 111
      here versus 10 for `T01_missing_citation` -- implying acronyms are eleven
      times better evidenced than missing citations, when the truth is 8
      documents versus 10 items. Document count is the granularity the
      observation was actually made at.

    * **`T15_jargon_without_gloss` is NOT included.** It is a new code introduced
      with the plain-language clause (build item #4), which has never run, so it
      genuinely has zero exemplars. Unglossed domain jargon is a different defect
      from an undefined acronym -- MCAL_PLAN 3.14 separates them explicitly
      ("Acronym expansion is handled by a separate post-processor ... but you
      MUST NOT use undefined domain jargon"). Crediting T15 with the acronym
      observations would drive its rarity to near zero and suppress selection of
      exactly the documents needed to observe it.
    """
    counts = dict(grade_set.tag_counts())
    n_acronym_docs = len(
        {i.doc_id for i in grade_set.items if i.acronym_issue}
    )
    if n_acronym_docs:
        counts["T04_undefined_acronym"] = max(
            counts.get("T04_undefined_acronym", 0), n_acronym_docs
        )
    return counts


def tag_rarity(tag: str, counts: dict[str, int]) -> float:
    """
    `1 / (1 + n_observed)`. Unobserved tag -> 1.0, seen 5 times -> 0.17.

    Reciprocal rather than a hard "zero-exemplar only" filter because the goal is
    to grow thin evidence as well as to fill holes: `critic_prompt` needs 3
    exemplars per tag for its set-cover, so going from 1 example to 2 is real
    progress even though the tag is not unobserved.
    """
    return 1.0 / (1.0 + counts.get(tag, 0))


# --- Scoring ----------------------------------------------------------------

# Bernoulli variance maxes at 0.25; rescaling makes the variance term [0, 1] so
# the two blended terms are on the same scale.
_MAX_BERNOULLI_VAR = 0.25

VARIANCE_WEIGHT = 0.6
RARITY_WEIGHT = 0.4

# Below this pool size the prevalence correction is skipped: with one or two
# candidates every predicted tag is "universal" by arithmetic and the correction
# would zero the rarity term for no reason.
MIN_POOL_FOR_PREVALENCE = 3


def pool_tag_prevalence(features: Sequence[DocFeatures]) -> dict[str, float]:
    """
    Fraction of the candidate pool each tag is predicted for.

    Used to discount tags that cannot discriminate. Measured on the current
    13-candidate pool, `no_glossary` fires on 12 of 13 documents, so
    `T04_undefined_acronym` and `T15_jargon_without_gloss` are predicted for
    almost everything -- and a tag predicted for every candidate carries no
    information about WHICH candidate to grade next, however rare it is in the
    graded set. Without this correction those two tags occupied two of the three
    `dominant_predicted_failure_tags` slots on 9 of the top 10 rows, which makes
    the column that the reviewer actually reads useless.

    This is a property of the POOL, not of a document, so it is computed once in
    `rank_candidates` and passed down. It does mean the score of a document
    depends on which other documents are in the pool -- true of any
    diversity-aware sampler, and the pool is fixed and reported, so the result
    stays reproducible.
    """
    n = len(features)
    if n < MIN_POOL_FOR_PREVALENCE:
        return {}
    counts: dict[str, int] = {}
    for f in features:
        for tag in predicted_tags(f):
            counts[tag] = counts.get(tag, 0) + 1
    return {t: c / n for t, c in counts.items()}


def effective_rarity(
    tag: str, counts: dict[str, int], prevalence: dict[str, float]
) -> float:
    """Graded-set rarity, discounted by how common the tag is in the pool."""
    return tag_rarity(tag, counts) * (1.0 - prevalence.get(tag, 0.0))


@dataclass
class Candidate:
    """One scored candidate document."""

    features: DocFeatures
    predicted_error_rates: dict[str, float] = dc_field(default_factory=dict)
    predicted_tags: dict[str, float] = dc_field(default_factory=dict)
    variance_term: float = 0.0
    rarity_term: float = 0.0
    uncertainty_score: float = 0.0
    dominant_tags: list[str] = dc_field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.features.doc_id

    def explain(self) -> list[str]:
        """Human-readable reasons, for the CLI table and the calibration report."""
        f = self.features
        active = active_feature_names(f)
        lines = [
            f"{f.n_pages} pp, year {f.year or '?'} ({f.year_source}), "
            f"chapters {f.ceq_chapters or 'none detected'}",
            f"features: {', '.join(active) or 'none'}",
            f"variance term {self.variance_term:.3f} x {VARIANCE_WEIGHT} + "
            f"rarity term {self.rarity_term:.3f} x {RARITY_WEIGHT} "
            f"= {self.uncertainty_score:.3f}",
        ]
        for feat, fld, delta, why in FIELD_ADJUSTMENTS:
            if feat in active:
                lines.append(f"  {feat} -> {fld} +{delta} log-odds: {why}")
        return lines

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "uncertainty_score": round(self.uncertainty_score, 6),
            "dominant_predicted_failure_tags": list(self.dominant_tags),
            "variance_term": round(self.variance_term, 6),
            "rarity_term": round(self.rarity_term, 6),
            "predicted_error_rates": {
                k: round(v, 4) for k, v in self.predicted_error_rates.items()
            },
            "predicted_tags": {k: round(v, 3) for k, v in self.predicted_tags.items()},
            "features": self.features.to_dict(),
            "active_features": active_feature_names(self.features),
        }


def score_candidate(
    f: DocFeatures,
    priors: dict[str, float],
    tag_counts: dict[str, int],
    *,
    pool_prevalence: Optional[dict[str, float]] = None,
    n_dominant_tags: int = DEFAULT_DOMINANT_TAGS,
) -> Candidate:
    """Pure function of (features, priors, tag_counts, pool_prevalence)."""
    prevalence = pool_prevalence or {}
    rates = predicted_error_rates(f, priors)
    variance = (
        sum(p * (1.0 - p) for p in rates.values()) / len(rates) / _MAX_BERNOULLI_VAR
        if rates
        else 0.0
    )

    tags = predicted_tags(f)
    if tags:
        num = sum(
            w * effective_rarity(t, tag_counts, prevalence) for t, w in tags.items()
        )
        den = sum(tags.values())
        rarity = num / den if den else 0.0
    else:
        rarity = 0.0

    # Dominant tags are ranked by weight x effective rarity: the tags this
    # document is both likely to exercise AND that the graded set is short of AND
    # that distinguish it from the rest of the pool. Ties break lexicographically
    # so the CSV is reproducible. Tags whose effective rarity is 0 (universal in
    # the pool) are dropped rather than shown -- reporting a tag that every
    # candidate shares is noise in the column the reviewer reads.
    ranked = sorted(
        (
            (t, w)
            for t, w in tags.items()
            if effective_rarity(t, tag_counts, prevalence) > 0.0
        ),
        key=lambda kv: (-(kv[1] * effective_rarity(kv[0], tag_counts, prevalence)), kv[0]),
    )
    dominant = [t for t, _ in ranked[:n_dominant_tags]]

    return Candidate(
        features=f,
        predicted_error_rates=rates,
        predicted_tags=tags,
        variance_term=min(1.0, variance),
        rarity_term=min(1.0, rarity),
        uncertainty_score=(
            VARIANCE_WEIGHT * min(1.0, variance) + RARITY_WEIGHT * min(1.0, rarity)
        ),
        dominant_tags=dominant,
    )


def candidate_doc_ids(grade_set: Optional["grades_mod.GradeSet"] = None) -> list[str]:
    """
    Materialized-but-ungraded doc_ids, in on-disk casing, sorted.

    Sorted case-insensitively so `P0491_...` and `p0491_...` sit together rather
    than having every capitalized directory float to the top of the batch.
    """
    gs = grade_set if grade_set is not None else grades_mod.load_grades()
    graded = {settings.normalize_doc_id(d) for d in gs.doc_ids}
    return sorted(
        (
            d
            for d in settings.available_doc_ids()
            if settings.normalize_doc_id(d) not in graded
        ),
        key=lambda d: settings.normalize_doc_id(d),
    )


def rank_candidates(
    *,
    grade_set: Optional["grades_mod.GradeSet"] = None,
    doc_ids: Optional[Sequence[str]] = None,
    features: Optional[Sequence[DocFeatures]] = None,
    n_dominant_tags: int = DEFAULT_DOMINANT_TAGS,
) -> list[Candidate]:
    """
    Score and rank every candidate, best first.

    Deterministic: the ordering key is `(-uncertainty_score, normalized_doc_id)`,
    so a tie in score is broken by doc_id and repeated runs over an unchanged
    corpus produce a byte-identical `next_batch.csv`. That matters more than it
    sounds -- the file is an artifact a reviewer works from, and a batch that
    silently reshuffles between runs is a batch nobody trusts.

    `features=` injects pre-computed features so tests need no corpus.
    """
    gs = grade_set if grade_set is not None else grades_mod.load_grades()
    priors = field_error_priors(gs)
    counts = tag_counts_in_grades(gs)

    if features is None:
        ids = list(doc_ids) if doc_ids is not None else candidate_doc_ids(gs)
        feats = []
        for did in ids:
            try:
                feats.append(extract_features(did))
            except Exception as e:  # noqa: BLE001 - one unreadable doc must not stop selection
                log.warning("skipping candidate %s: %s", did, e)
    else:
        feats = list(features)

    prevalence = pool_tag_prevalence(feats)
    scored = [
        score_candidate(
            f, priors, counts,
            pool_prevalence=prevalence,
            n_dominant_tags=n_dominant_tags,
        )
        for f in feats
    ]
    scored.sort(key=lambda c: (-c.uncertainty_score, c.features.normalized_doc_id))
    return scored


# --- Artifact ---------------------------------------------------------------

NEXT_BATCH_COLUMNS = ("doc_id", "uncertainty_score", "dominant_predicted_failure_tags")
TAG_SEPARATOR = "|"


def write_next_batch(
    candidates: Sequence[Candidate],
    *,
    n: int = settings.NEXT_BATCH_SIZE,
    path: Optional[Path] = None,
) -> Path:
    """
    Write `artifacts/next_batch.csv` (MCAL_PLAN 2).

    Exactly the three columns the plan specifies, and nothing else -- no comment
    preamble, unlike the grading sheets. The plan lists this file's schema
    verbatim and a reviewer or UI reading it positionally must not have to know
    about M-Cal's commenting conventions. Everything diagnostic lives in the
    `Candidate` objects and in the CLI output instead.

    Tags are `|`-joined rather than comma-joined: the csv module would quote a
    comma-joined cell correctly, but the file gets opened in a spreadsheet and
    re-saved by humans, and a quoted comma inside a cell is the single most common
    way that survives as a column split.

    Unversioned by design (MCAL_PLAN 2 lists `next_batch.csv` without a stage
    suffix) -- it is a rolling worklist, not a frozen calibration artifact.
    """
    p = path or settings.NEXT_BATCH_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(NEXT_BATCH_COLUMNS)
        for c in candidates[: max(0, n)]:
            w.writerow(
                [
                    c.doc_id,
                    f"{c.uncertainty_score:.4f}",
                    TAG_SEPARATOR.join(c.dominant_tags),
                ]
            )
    return p


def read_next_batch(path: Optional[Path] = None) -> list[dict]:
    """Read back `next_batch.csv`, splitting the tag column."""
    p = path or settings.NEXT_BATCH_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"No next_batch.csv at {p}. Generate it with "
            f"`python -m mcal.active_select --n {settings.NEXT_BATCH_SIZE}`."
        )
    out = []
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(
                {
                    "doc_id": (row.get("doc_id") or "").strip(),
                    "uncertainty_score": float(row.get("uncertainty_score") or 0.0),
                    "dominant_predicted_failure_tags": [
                        t
                        for t in (
                            row.get("dominant_predicted_failure_tags") or ""
                        ).split(TAG_SEPARATOR)
                        if t
                    ],
                }
            )
    return out


def selection_report(
    candidates: Sequence[Candidate],
    *,
    n: int,
    grade_set: Optional["grades_mod.GradeSet"] = None,
) -> dict:
    """Diagnostics for `calibration_report.v(N).md` and the CLI."""
    gs = grade_set
    chosen = list(candidates[: max(0, n)])
    counts = tag_counts_in_grades(gs) if gs is not None else {}
    covered: dict[str, int] = {}
    for c in chosen:
        for t in c.dominant_tags:
            covered[t] = covered.get(t, 0) + 1
    zero_exemplar = sorted(t for t in covered if counts.get(t, 0) == 0)
    return {
        "method": "cold_start_feature_heuristic",
        "calibrated": False,
        "caveat": (
            "Predicted composite variance is NOT calibrated. Candidates have no "
            "extractions, so composite has no inputs; the ranking is a smoothed "
            "per-field error prior shifted by document features drawn from "
            "MCAL_PLAN 1. See mcal/active_select.py's module docstring."
        ),
        "n_pool": len(candidates),
        "n_selected": len(chosen),
        "requested_n": n,
        "variance_weight": VARIANCE_WEIGHT,
        "rarity_weight": RARITY_WEIGHT,
        "selected": [c.to_dict() for c in chosen],
        "tags_covered_by_batch": dict(sorted(covered.items())),
        "zero_exemplar_tags_covered": zero_exemplar,
        "graded_tag_counts": counts,
    }


# --- CLI --------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """`python -m mcal.active_select --n 10`."""
    ap = argparse.ArgumentParser(
        prog="mcal.active_select",
        description=(
            "Pick the next grading batch by uncertainty sampling (MCAL_PLAN 3.6). "
            "No LLM calls; deterministic given the corpus."
        ),
    )
    ap.add_argument(
        "--n",
        type=int,
        default=settings.NEXT_BATCH_SIZE,
        help=(
            f"batch size (default {settings.NEXT_BATCH_SIZE}, "
            f"settings.NEXT_BATCH_SIZE, matching the MCAL_PLAN 7.5 cadence)"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output CSV (default {settings.NEXT_BATCH_PATH})",
    )
    ap.add_argument(
        "--doc",
        action="append",
        default=None,
        dest="docs",
        help="restrict the pool to these doc_ids (repeatable)",
    )
    ap.add_argument(
        "--include-graded",
        action="store_true",
        help="do not exclude already-graded docs (diagnostic only)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.add_argument("--verbose", action="store_true", help="show per-doc reasoning")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    gs = grades_mod.load_grades()
    if args.docs:
        ids = list(args.docs)
    elif args.include_graded:
        ids = settings.available_doc_ids()
    else:
        ids = candidate_doc_ids(gs)

    if not ids:
        print(
            "No candidate documents. Either every materialized document is "
            f"already graded, or {settings.PAGES_DATA_DIR} has no per-page OCR "
            "on this machine.",
            file=sys.stderr,
        )
        return 1

    print(
        f"pool: {len(ids)} ungraded of {len(settings.available_doc_ids())} "
        f"materialized ({gs.n_docs} graded)"
    )
    candidates = rank_candidates(grade_set=gs, doc_ids=ids)
    if not candidates:
        print("No candidate features could be computed.", file=sys.stderr)
        return 1

    chosen = candidates[: max(0, args.n)]
    width = max(len(c.doc_id) for c in chosen)
    print()
    print(f"{'#':>2}  {'doc_id':<{width}}  score  pp    yr    tags")
    for i, c in enumerate(chosen, 1):
        f = c.features
        print(
            f"{i:>2}  {c.doc_id:<{width}}  {c.uncertainty_score:.3f}  "
            f"{f.n_pages:<5} {f.year or '?':<5} "
            f"{TAG_SEPARATOR.join(c.dominant_tags)}"
        )
        if args.verbose:
            for line in c.explain():
                print(f"      {line}")

    report = selection_report(candidates, n=args.n, grade_set=gs)
    if report["zero_exemplar_tags_covered"]:
        print(
            "\nbatch covers tags with ZERO graded exemplars: "
            + ", ".join(report["zero_exemplar_tags_covered"])
        )

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    path = write_next_batch(candidates, n=args.n, path=args.out)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
