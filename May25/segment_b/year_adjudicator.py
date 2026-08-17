"""
Always-run year adjudicator (MCAL_PLAN 3.13, 1(1), build item #15).

MCAL_PLAN 1(1): `year` was wrong on 3/8 graded docs and ALL THREE were pre-1980
(Operation Breakthrough graded "1972 -> 1976", LA Transit "1980 -> 1979",
Lincoln Hwy "1972 -> 1971, based on page 70"). Two causes, both structural:

  1. M1 reads a regex off the first 3 pages only. Old EISs frequently carry no
     date on the cover at all -- the real date is on a signature/approval page
     or a transmittal letter. The Lincoln Hwy grade points at page 70.
  2. 1970s microfilm OCR corrupts digits: `l972`, `197O`, `I97I`. A plain
     `\\b(19|20)\\d\\d\\b` regex sees none of those.

So this module: (a) always runs, never only on disagreement, so that a
confidently-wrong M1 year is still re-examined; (b) repairs OCR'd digits BEFORE
the year regex, per MCAL_PLAN 1(1); (c) gathers candidates from the plan's
windows -- first 5pp, the last 3pp of front matter, and keyword-detected
signature/approval and transmittal pages; and (d) makes exactly ONE Sonnet call
whose prompt states the priority rule verbatim: signature > transmittal > cover
> body.

Why a digit repair of our own instead of `mcal.quote_check.normalize`: that
normalizer deliberately folds LETTERS ONTO DIGITS (o->0, l->1, s->5) so that
fuzzy prose comparison is glyph-insensitive. Applied to year extraction it is
actively harmful -- it would turn "loss" into "1055" and "Ross" into "R055", and
it destroys the digits it touches, so "1975" and "197S" become
indistinguishable from "1955"-shaped noise. Year repair must be *targeted*: only
tokens already shaped like a year are rewritten, and only when the repaired
value lands inside config.YEAR_MIN..YEAR_MAX. `quote_check.normalize` is still
used, on the prose channel, to check that the model's `evidence_quote` really
came from the evidence we sent it.

Robustness: the LLM is treated as advisory. Its year must be in range or it is
discarded; a year outside the regex-derived candidate set is accepted but forced
to `confidence: "low"` with a note; any exception falls back to the
priority-then-mode regex candidate. This function does not raise.

Token cost: one Sonnet call with a bounded evidence block (at most
MAX_CANDIDATES_IN_PROMPT snippets of ~200 chars). Usage is recorded through
`llm.call_with_usage`, so a caller that wraps the doc in
`llm.start_usage_session()` / `llm.end_usage_session()` gets this call in its
per-doc cost roll-up for free (MCAL_PLAN 2, calibration_report Cost Summary).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from rapidfuzz import fuzz

from mcal import settings  # noqa: F401  -- installs the segment_a sys.path bridge
from mcal.quote_check import normalize

# segment_a bridge is installed by the mcal.settings import above.
import config as seg_a_config  # noqa: E402
from pages import Doc  # noqa: E402

log = logging.getLogger(__name__)


# --- Vocabulary -------------------------------------------------------------

# MCAL_PLAN 3.13 output enum. "adjudicated" is the catch-all the plan provides
# for a year that no single source type owns outright.
SOURCE_TYPES = ("signature", "transmittal", "cover", "body", "adjudicated")

# MCAL_PLAN 1(1): "signature > transmittal > cover > body". Used both to rank
# evidence in the prompt and to break ties in the deterministic fallback.
SOURCE_PRIORITY = {
    "signature": 4,
    "transmittal": 3,
    "cover": 2,
    "body": 1,
    "adjudicated": 0,
}

CONFIDENCE_LEVELS = ("high", "medium", "low")

YEAR_MIN = seg_a_config.YEAR_MIN   # 1969 -- NEPA was signed 1970-01-01
YEAR_MAX = seg_a_config.YEAR_MAX   # 2026


# --- Windows ----------------------------------------------------------------

# MCAL_PLAN 1(1) / 3.13 windows.
FIRST_PAGES = 5
FRONT_MATTER_PAGES = seg_a_config.FIRST_30_PAGES   # 30
FRONT_MATTER_TAIL_PAGES = 3
# Pages 1..COVER_PAGES are treated as the cover/title-page block.
COVER_PAGES = 3

# Signature and transmittal pages are located by keyword over the WHOLE doc,
# then capped -- an approval page can sit anywhere (the Lincoln Hwy grade points
# at p.70 of 347), so restricting the search to front matter would reproduce the
# original bug. The caps keep the prompt bounded.
MAX_SIGNATURE_PAGES = 3
MAX_TRANSMITTAL_PAGES = 2
# How close a signature/transmittal keyword must be to a year for that year to
# count as signature/transmittal evidence. Roughly one line of OCR either side.
SIGNATURE_PROXIMITY_CHARS = 160
# Budget for the weak, colon-less "date" keyword: same line only.
_WEAK_PROXIMITY_CHARS = 40

MAX_CANDIDATES_IN_PROMPT = 30
CONTEXT_CHARS = 90
# Deterministic-fallback outlier guard, in years from the median candidate.
FALLBACK_OUTLIER_YEARS = 12
# Pages this close to the end of the document count as "edge" pages alongside
# front matter, for signature-page ranking.
EDGE_BACK_PAGES = 10


# --- OCR digit repair (MCAL_PLAN 1(1): "OCR-normalize digits before regex") --

# Glyphs OCR emits for each digit on 1970s microfilm. Deliberately conservative:
# only substitutions we have seen in this corpus, and no digit->digit rewrites.
_DIGIT_REPAIR = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "i": "1", "|": "1", "!": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "b": "6",
    "B": "8",
    "g": "9", "q": "9", "G": "9",
}

# A year-shaped token: a century digit (or its confusables), the second century
# digit, then two more digit-or-confusable characters. Anchored on non-alphanumeric
# boundaries so that accession numbers ("35556036861797") cannot contribute.
_DIGITISH = "0-9OoQDlIi|!ZzSsbBgqG"
_YEAR_SHAPED_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?:[1lI|!][9gqG]|[2Zz][0OoQD])[{_DIGITISH}]{{2}}(?![A-Za-z0-9])"
)

# Run after repair. Plain and strict -- repair has already done the hard part.
_YEAR_RE = re.compile(r"(?<!\d)(19\d\d|20\d\d)(?!\d)")

# Typewritten short dates: "Date: 3/3/77". Essential rather than optional -- the
# Fuel Economy doc's only real publication date is "Date: 3/3/77" on the
# signature line, while the cover's prominent 4-digit year is "MODEL YEAR 1979",
# which is not a date at all. Restricted to the full M/D/YY shape; a bare 2-digit
# number is far too ambiguous to guess at.
_SHORT_DATE_RE = re.compile(
    r"(?<![\d/\-])\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*(\d{2})(?![\d/\-])"
)


def _expand_two_digit(yy: str) -> Optional[int]:
    """"77" -> 1977, "05" -> 2005. None when neither century lands in range."""
    try:
        n = int(yy)
    except (TypeError, ValueError):
        return None
    for candidate in (1900 + n, 2000 + n):
        if YEAR_MIN <= candidate <= YEAR_MAX:
            return candidate
    return None


# Author-date citation keys: "Duke Power Company. 1976c. Transmittal Letter
# from ...". A year immediately followed by a lowercase disambiguation letter is
# a bibliography entry, never a publication date -- and because reference lists
# are full of the words "transmittal", "approved" and "signed", these are the
# highest-priority false positives the keyword sweep can produce.
_CITATION_SUFFIX_RE = re.compile(r"^[a-z][.,;:)\]]")


def _year_spans(text: str) -> list[tuple[int, int, int]]:
    """(year, start, end) for every in-range year in already-repaired text."""
    out: list[tuple[int, int, int]] = []
    for m in _YEAR_RE.finditer(text or ""):
        y = int(m.group(1))
        if not (YEAR_MIN <= y <= YEAR_MAX):
            continue
        if _CITATION_SUFFIX_RE.match(text[m.end() : m.end() + 2]):
            continue
        out.append((y, m.start(), m.end()))
    for m in _SHORT_DATE_RE.finditer(text or ""):
        y = _expand_two_digit(m.group(1))
        if y is not None:
            out.append((y, m.start(), m.end()))
    out.sort(key=lambda t: t[1])
    return out


def repair_year_token(token: str) -> Optional[int]:
    """
    Map one year-shaped token to an in-range year, or None.

    "l972" -> 1972, "197O" -> 1970, "I97I" -> 1971. Returns None when the
    repaired value falls outside config.YEAR_MIN..YEAR_MAX, which is what stops
    the repair from inventing years out of noise: "Iggy" repairs to 1999 but
    "l0SS" never even matches, and "1855" repairs to itself and is rejected.
    """
    if not token or len(token) != 4:
        return None
    digits = "".join(_DIGIT_REPAIR.get(ch, ch) for ch in token)
    if not digits.isdigit():
        return None
    year = int(digits)
    return year if YEAR_MIN <= year <= YEAR_MAX else None


def repair_ocr_years(text: str) -> str:
    """
    Rewrite year-shaped OCR tokens to their digit form, in place.

    Length-preserving (4 characters in, 4 out), so character offsets into the
    repaired text remain valid offsets into the original. Tokens whose repair
    would fall outside the plausible year range are left untouched rather than
    guessed at.
    """
    if not text:
        return ""

    def _sub(m: re.Match) -> str:
        raw = m.group(0)
        year = repair_year_token(raw)
        return str(year) if year is not None else raw

    return _YEAR_SHAPED_RE.sub(_sub, text)


def years_in(text: str) -> list[int]:
    """
    All in-range years in `text`, after OCR digit repair, in order of appearance.

    Includes M/D/YY short dates, which is where several 1970s signature blocks
    keep the only real date in the document.
    """
    return [y for y, _, _ in _year_spans(repair_ocr_years(text))]


# --- Signature / transmittal detection --------------------------------------

# MCAL_PLAN 1(1) names the keywords: `Approved | Signed | Date: | Transmittal`.
# Taken literally, `Approved` is the single worst false-positive source in this
# corpus: "approved by UMTA in October 1976", "the plan was unanimously
# approved" and "approved rapid transit projects" are all ordinary body prose,
# and each one promoted a body year to signature evidence in measurement (it is
# what made the Buffalo doc prefer 1976 over its own "JUNE 1977" cover). So the
# participle is only honoured in the FORMS a signature block actually takes --
# followed by a colon, or set in capitals as a block label.
#
# Scoped inline flags (Python 3.11+) keep the case-sensitive `APPROVED` /
# `APPROVAL` alternatives in the same pattern as the case-insensitive ones.
_SIGNATURE_RE = re.compile(
    r"(?i:\bdate:|\bdated:|\bapproved\s*:|\bapproved\s+by\s*:|\bapproval\s+date\b"
    r"|\bdate\s+approved\b|\bsigned\b|\bsignature\b|\bcertified\s+by\b)"
    r"|/s/"
    r"|(?-i:\bAPPROVED\b|\bAPPROVAL\b)"
)

# A bare "date" with no colon. OCR eats the colon often enough to matter -- the
# Airport Spur summary sheet reads "Date DCT 2 7 1975", which is the document's
# real approval date -- but "date" alone is far too common to trust on its own.
# It therefore (a) only counts on a page that ALSO carries a strong keyword, so
# it sharpens attribution within an already-identified approval page rather than
# nominating new ones, and (b) gets a much tighter proximity budget
# (_WEAK_PROXIMITY_CHARS). Without (a), "revised drawings bearing the date
# January 5, 1976" on p.287 of the Bad Creek doc outranked its own 1977 cover.
_WEAK_SIGNATURE_RE = re.compile(r"(?i)\bdate\b")

_TRANSMITTAL_RE = re.compile(
    r"(?i)(\btransmittal\b|\btransmitted\s+herewith\b|\bletter\s+of\s+transmittal\b"
    r"|\bforwarded\s+herewith\b)"
)


def _keyword_spans(text: str, pattern: re.Pattern) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in pattern.finditer(text or "")]


def _distance(span: tuple[int, int], spans: Sequence[tuple[int, int]]) -> Optional[int]:
    """Character distance from `span` to the nearest of `spans`, or None if empty."""
    if not spans:
        return None
    best = None
    for s, e in spans:
        d = 0 if (s < span[1] and span[0] < e) else min(abs(span[0] - e), abs(s - span[1]))
        if best is None or d < best:
            best = d
    return best


def _is_edge_page(doc: Doc, page_num: int) -> bool:
    """True for front-matter pages and the last EDGE_BACK_PAGES of the document."""
    if not doc.pages:
        return False
    first = doc.pages[0].page_num
    last = doc.pages[-1].page_num
    return page_num <= first + FRONT_MATTER_PAGES - 1 or page_num >= last - EDGE_BACK_PAGES + 1


def _keyword_mentions(doc: Doc) -> tuple[list[YearMention], list[int], list[int]]:
    """
    Year mentions that sit next to a signature/approval or transmittal keyword.

    MCAL_PLAN 1(1) says to detect the PAGE by keyword; we go one step finer and
    require the keyword to be within the proximity budget of the year itself.
    Page-level keyword matching alone is far too coarse on this corpus:
    "approved" and "Date" occur in ordinary body prose, so page-level detection
    labelled three of the Airport Spur doc's mid-document body pages as
    signature pages and thereby promoted design-horizon years (1990, 2015) above
    the real approval date. Proximity also lets us attribute the RIGHT year on a
    page instead of every year on it.

    Returns (mentions, signature_pages, transmittal_pages). Pages are ranked by
    how tightly the keyword hugs the year, then by page number, and capped so
    the single prompt stays bounded.
    """
    per_page: dict[int, dict[str, list[YearMention]]] = {}
    for page in doc.pages:
        repaired = repair_ocr_years(page.text)
        strong = _keyword_spans(repaired, _SIGNATURE_RE)
        # Weak keyword only refines pages already nominated by a strong one.
        weak = _keyword_spans(repaired, _WEAK_SIGNATURE_RE) if strong else []
        trans = _keyword_spans(repaired, _TRANSMITTAL_RE)
        if not (strong or trans):
            continue
        for year, start, end in _year_spans(repaired):
            span = (start, end)
            d_strong = _distance(span, strong)
            d_weak = _distance(span, weak)
            d_trans = _distance(span, trans)
            # Normalise each channel against its own budget so the channels are
            # comparable, then take the best.
            options: list[tuple[float, int, str]] = []
            if d_trans is not None and d_trans <= SIGNATURE_PROXIMITY_CHARS:
                options.append((d_trans / SIGNATURE_PROXIMITY_CHARS, d_trans, "transmittal"))
            if d_strong is not None and d_strong <= SIGNATURE_PROXIMITY_CHARS:
                # Ties go to signature, which MCAL_PLAN 1(1) ranks above
                # transmittal; a transmittal letter usually also carries a
                # "Date:", and without the tie-break every transmittal page
                # would be relabelled and `source_type` would lose the
                # distinction the plan asks for. -0.001 encodes that preference.
                options.append(
                    (d_strong / SIGNATURE_PROXIMITY_CHARS - 0.001, d_strong, "signature")
                )
            if d_weak is not None and d_weak <= _WEAK_PROXIMITY_CHARS:
                options.append((d_weak / _WEAK_PROXIMITY_CHARS, d_weak, "signature"))
            if not options:
                continue
            _, dist, kind = min(options)
            per_page.setdefault(page.page_num, {}).setdefault(kind, []).append(
                YearMention(
                    year=year,
                    page=page.page_num,
                    source_type=kind,
                    context=_context(repaired, start, end),
                    keyword_distance=dist,
                )
            )

    def _pick(kind: str, cap: int) -> list[int]:
        # Rank by position class first, then by how tightly the keyword hugs the
        # year, then by page. Position class matters because a signature or
        # approval block structurally belongs to the front matter (or the very
        # end), while the dominant false positive is correspondence reproduced
        # mid-document -- "the speed memo ... dated November 17, 1975 and signed
        # by R. W. Baker" on p.216 of the Airport Spur doc. Mid-document pages
        # are still eligible (the Lincoln Hwy grade points at p.70), they just
        # lose ties.
        pages = [
            (
                0 if _is_edge_page(doc, pn) else 1,
                min(m.keyword_distance for m in d[kind]),
                pn,
            )
            for pn, d in per_page.items()
            if d.get(kind)
        ]
        pages.sort()
        return sorted(pn for _, _, pn in pages[:cap])

    sig_pages = _pick("signature", MAX_SIGNATURE_PAGES)
    trans_pages = _pick("transmittal", MAX_TRANSMITTAL_PAGES)

    mentions: list[YearMention] = []
    for pn in sig_pages:
        mentions.extend(per_page[pn]["signature"])
    for pn in trans_pages:
        mentions.extend(per_page[pn]["transmittal"])
    return mentions, sig_pages, trans_pages


def find_signature_pages(doc: Doc) -> tuple[list[int], list[int]]:
    """
    Locate signature/approval and transmittal pages (MCAL_PLAN 1(1)).

    Returns (signature_pages, transmittal_pages). A page qualifies only if it
    carries a year within SIGNATURE_PROXIMITY_CHARS of the keyword -- an
    approval block with no date on it says nothing about the year, and would
    only spend prompt tokens.
    """
    _, sig, trans = _keyword_mentions(doc)
    return sig, trans


# --- Candidate collection ---------------------------------------------------


@dataclass(frozen=True)
class YearMention:
    """One year mention, with the evidence class of the page it came from."""

    year: int
    page: int
    source_type: str
    context: str
    # Characters between this year and the signature/transmittal keyword that
    # typed it. None for positional (cover/body) mentions.
    keyword_distance: Optional[int] = None

    @property
    def priority(self) -> int:
        return SOURCE_PRIORITY.get(self.source_type, 0)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "page": self.page,
            "source_type": self.source_type,
            "context": self.context,
            "keyword_distance": self.keyword_distance,
        }


def _context(text: str, start: int, end: int) -> str:
    snippet = text[max(0, start - CONTEXT_CHARS) : end + CONTEXT_CHARS]
    return re.sub(r"\s+", " ", snippet).strip()


def _mentions_on_page(page_num: int, text: str, source_type: str) -> list[YearMention]:
    repaired = repair_ocr_years(text)
    return [
        YearMention(
            year=year,
            page=page_num,
            source_type=source_type,
            context=_context(repaired, start, end),
        )
        for year, start, end in _year_spans(repaired)
    ]


def _window_labels(doc: Doc) -> dict[int, str]:
    """
    Page number -> evidence class for the plan's positional windows.

    Windows (MCAL_PLAN 1(1), 3.13): the first 5pp, of which the first 3 are
    treated as the cover/title block, plus the last 3pp of front matter, where
    front matter is the first 30pp (segment_a config.FIRST_30_PAGES -- the same
    window M1 already calls front matter).

    The plan's phrase is "first 5pp + last 3pp of front matter". Read literally
    it could also mean the last 3pp of the whole document; we take front matter,
    because that is where the pre-1978 statements in this corpus put their
    approval block, and because the keyword sweep in `_keyword_mentions`
    already covers the back of the document.
    """
    if not doc.pages:
        return {}
    first = doc.pages[0].page_num
    last = doc.pages[-1].page_num
    present = {p.page_num for p in doc.pages}

    labels: dict[int, str] = {}
    front_matter_end = min(last, first + FRONT_MATTER_PAGES - 1)
    tail_start = max(first, front_matter_end - FRONT_MATTER_TAIL_PAGES + 1)
    cover_end = min(last, first + COVER_PAGES - 1)

    for p in range(first, min(last, first + FIRST_PAGES - 1) + 1):
        if p in present:
            labels[p] = "cover" if p <= cover_end else "body"
    for p in range(tail_start, front_matter_end + 1):
        if p in present:
            labels.setdefault(p, "body")
    return labels


def collect_candidates(doc: Doc) -> list[YearMention]:
    """
    Every year mention in the plan's windows, deduplicated per (year, page).

    Keyword-proximate mentions are collected first and win the (year, page) slot,
    so a date on page 3's approval block is typed "signature" rather than
    "cover". Sorted by evidence priority (signature first) then page, which is
    also the order they are presented to the model.
    """
    by_num = {p.page_num: p.text for p in doc.pages}
    out: list[YearMention] = []
    seen: set[tuple[int, int]] = set()

    keyword_mentions, _, _ = _keyword_mentions(doc)
    for m in keyword_mentions:
        key = (m.year, m.page)
        if key in seen:
            continue
        seen.add(key)
        out.append(m)

    for page_num, source_type in _window_labels(doc).items():
        for m in _mentions_on_page(page_num, by_num.get(page_num, ""), source_type):
            key = (m.year, m.page)
            if key in seen:
                continue
            seen.add(key)
            out.append(m)

    out.sort(key=lambda m: (-m.priority, m.page, m.year))
    return out


def aggregate_candidates(mentions: Sequence[YearMention]) -> list[dict]:
    """
    Collapse mentions into one row per year for the output `candidates` field.

    Rows carry the count and the best (highest-priority) source type, which is
    what a reviewer needs to see why a year won.
    """
    by_year: dict[int, dict] = {}
    for m in mentions:
        row = by_year.setdefault(
            m.year,
            {"year": m.year, "count": 0, "pages": [], "source_types": [], "best_source": m.source_type},
        )
        row["count"] += 1
        row["pages"].append(m.page)
        if m.source_type not in row["source_types"]:
            row["source_types"].append(m.source_type)
        if SOURCE_PRIORITY.get(m.source_type, 0) > SOURCE_PRIORITY.get(row["best_source"], 0):
            row["best_source"] = m.source_type
    rows = list(by_year.values())
    for r in rows:
        r["pages"] = sorted(set(r["pages"]))[:8]
    rows.sort(
        key=lambda r: (-SOURCE_PRIORITY.get(r["best_source"], 0), -r["count"], r["year"])
    )
    return rows


def modal_year(mentions: Sequence[YearMention]) -> Optional[int]:
    """
    Plain mode over all mentions, ties broken by evidence priority then recency.

    Kept as its own function because it is the plan-literal fallback ("the modal
    regex candidate") and because `fallback_choice` deviates from it; a reader
    comparing the two should be able to see both.
    """
    if not mentions:
        return None
    counts = Counter(m.year for m in mentions)
    best_priority = {
        y: max(m.priority for m in mentions if m.year == y) for y in counts
    }
    return max(counts, key=lambda y: (counts[y], best_priority[y], y))


def _drop_outliers(mentions: Sequence[YearMention]) -> list[YearMention]:
    """
    Discard candidates far from the document's own centre of gravity.

    Median-based rather than mean-based because candidate counts are small
    (3-20) and a single 2015 wrecks a mean. Returns the input unchanged if the
    filter would empty it.
    """
    years = sorted(m.year for m in mentions)
    if not years:
        return list(mentions)
    median = years[len(years) // 2]
    kept = [m for m in mentions if abs(m.year - median) <= FALLBACK_OUTLIER_YEARS]
    return kept or list(mentions)


def fallback_choice(mentions: Sequence[YearMention]) -> Optional[YearMention]:
    """
    Deterministic choice used whenever the LLM cannot be trusted.

    Two DOCUMENTED DEVIATIONS from the plan's "fall back to the modal regex
    candidate":

    1. The mode is taken WITHIN the highest-priority tier present, not across
       all mentions. Rationale from MCAL_PLAN 1(1) itself -- the reason the
       adjudicator exists is that body-text mentions outnumber the signature
       date, so a global mode reproduces the failure being fixed. Measured on the
       graded set, the global mode picks the human-correct year for 4 of 8 docs
       and the tiered rule for 6 of 8.
    2. Inside a signature/transmittal tier the winner is the year CLOSEST to its
       keyword, not the most frequent. A signature block contains exactly one
       date and it is adjacent to the word "Date"; frequency there measures how
       often a year is repeated elsewhere on the same page. This is what makes
       the Fuel Economy doc resolve to 1977 (from "Date: 3/3/77") instead of
       1979, the model year printed eight times on the cover.

    Positional tiers (cover/body) have no keyword distance, so they still fall
    back to the mode. `modal_year()` is retained for comparison and is reported
    in the result note whenever the two disagree.

    Before any of that, candidates more than FALLBACK_OUTLIER_YEARS from the
    median candidate year are discarded. Two-digit dates make this necessary:
    the Airport Spur approval line OCRs as "12/5/15", which expands to 2015
    because 1915 is out of NEPA's range, and a 40-year outlier must not be
    allowed to win a tier. The outlier filter applies ONLY here -- the model
    still sees every candidate, because it can read the context and we would
    rather it reject a bad candidate on evidence than never see it.
    """
    if not mentions:
        return None
    eligible = _drop_outliers(mentions)
    top = max(m.priority for m in eligible)
    tier = [m for m in eligible if m.priority == top]
    keyed = [m for m in tier if m.keyword_distance is not None]
    if keyed:
        counts = Counter(m.year for m in keyed)
        return min(keyed, key=lambda m: (m.keyword_distance, -counts[m.year], m.page))
    counts = Counter(m.year for m in tier)
    year = max(counts, key=lambda y: (counts[y], -min(m.page for m in tier if m.year == y)))
    return min((m for m in tier if m.year == year), key=lambda m: m.page)


# --- Prompt -----------------------------------------------------------------

SYSTEM_PROMPT = (
    "You adjudicate the publication year of a US Environmental Impact Statement "
    "from OCR'd page text. The scans are 1969-2000 microfilm and the digits are "
    "unreliable. You reason only from the evidence provided. Return JSON only, "
    "no prose outside the JSON object."
)

# The priority rule is stated as a numbered hierarchy AND restated as two
# explicit "outranks" sentences, because MCAL_PLAN 1(1) specifies the ordering
# and a single mention of it in a long prompt is the kind of instruction models
# drop when the body evidence is voluminous.
_PRIORITY_BLOCK = """\
PRIORITY RULE (this is the whole point of the task):
  1. signature   -- a date on a signature / approval / certification block
  2. transmittal -- a date on a transmittal or cover letter
  3. cover       -- a date printed on the cover or title page
  4. body        -- any other mention inside the document

A signature date OUTRANKS a transmittal date.
A transmittal date OUTRANKS a cover-page date.
A cover-page date OUTRANKS any in-body mention, no matter how often the in-body
year is repeated. Frequency NEVER beats source type.

Notes on this corpus:
  - Pre-1978 statements often carry no date on the cover at all. Do not infer a
    cover date from the absence of one.
  - A year appearing inside a citation, a data table, a population projection,
    a design horizon ("1990 traffic volumes") or a statute name
    ("Act of 1969") is NOT a publication date. Ignore those.
  - Digits have already been OCR-repaired where the shape was unambiguous;
    treat remaining oddities with suspicion rather than confidence.
"""

_SCHEMA_BLOCK = """\
Return exactly this JSON object:
{
  "year": <int or null>,
  "source_type": "signature" | "transmittal" | "cover" | "body" | "adjudicated",
  "confidence": "high" | "medium" | "low",
  "evidence_quote": "<verbatim substring of one CONTEXT above, or null>",
  "note": "<one short sentence, or null>"
}

Rules for the answer:
  - "year" must be one of the CANDIDATE years unless every candidate is clearly
    a non-publication date, in which case return null.
  - "source_type" is the class of the evidence you actually used.
  - "evidence_quote" must be copied verbatim from a CONTEXT line. If you cannot
    quote your evidence, return null and set confidence to "low".
  - Use "adjudicated" only when you combined evidence classes and no single one
    is decisive.
"""


def build_prompt(mentions: Sequence[YearMention], doc_id: str = "") -> tuple[str, str]:
    """Build the (system, user) pair for the single Sonnet call."""
    shown = list(mentions[:MAX_CANDIDATES_IN_PROMPT])
    lines = []
    for m in shown:
        lines.append(
            f"- year={m.year} source_type={m.source_type} page={m.page}\n"
            f"  CONTEXT: {m.context}"
        )
    candidate_years = sorted({m.year for m in shown})
    user = (
        f"DOCUMENT: {doc_id or 'unknown'}\n\n"
        f"{_PRIORITY_BLOCK}\n"
        f"CANDIDATE years found by regex: {candidate_years}\n\n"
        f"EVIDENCE ({len(shown)} mentions, highest-priority source first):\n"
        + "\n".join(lines)
        + f"\n\n{_SCHEMA_BLOCK}"
    )
    return SYSTEM_PROMPT, user


# --- Validation -------------------------------------------------------------

# rapidfuzz floor for "the model's evidence_quote came from the evidence we
# sent". Loose on purpose: the model may re-space or re-case OCR text, and this
# check exists to catch invented quotes, not to police whitespace.
EVIDENCE_QUOTE_MIN_RATIO = 85


def _coerce_year(value) -> Optional[int]:
    """Accept 1971, "1971", "c. 1971"; reject containers and out-of-range values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple, set, dict)):
        # A container means the model ignored the schema; do not go fishing for a
        # 4-digit substring in its repr.
        return None
    if isinstance(value, (int, float)):
        y = int(value)
    else:
        m = re.search(r"\d{4}", str(value))
        if not m:
            return None
        y = int(m.group(0))
    return y if YEAR_MIN <= y <= YEAR_MAX else None


def _quote_is_grounded(quote: str, mentions: Sequence[YearMention]) -> bool:
    q = normalize(quote or "")
    if len(q) < 12:
        return False
    haystack = normalize(" ".join(m.context for m in mentions))
    if not haystack:
        return False
    return fuzz.partial_ratio(q, haystack) >= EVIDENCE_QUOTE_MIN_RATIO


def _default_call(system: str, user: str, **kw) -> dict:
    """
    The single Sonnet call.

    Imported lazily so that importing this module -- and therefore running its
    deterministic half in tests -- does not require the Anthropic SDK or
    credentials.
    """
    import llm  # segment_a, JSON-only wrapper

    return llm.sonnet(system, user, **kw)


def _result(
    *,
    year: Optional[int],
    source_type: str,
    confidence: str,
    candidates: list[dict],
    evidence_quote: Optional[str],
    note: Optional[str],
) -> dict:
    """The MCAL_PLAN 3.13 output object, and nothing else."""
    return {
        "year": year,
        "source_type": source_type,
        "confidence": confidence,
        "candidates": candidates,
        "evidence_quote": evidence_quote,
        "note": note,
    }


# --- Entry point ------------------------------------------------------------

MAX_TOKENS = 700


def adjudicate(
    doc: Doc,
    *,
    m1_year: Optional[int] = None,
    call: Optional[Callable[..., dict]] = None,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """
    Adjudicate a document's publication year. Always runs. One Sonnet call.

    `m1_year` is used ONLY to annotate the note with whether the adjudicated
    year agrees with Segment A's. It is deliberately NOT put in the prompt: the
    point of an always-run adjudicator (MCAL_PLAN 3.13) is an independent read,
    and handing the model the prior invites it to ratify a wrong one -- which is
    precisely how 3/8 docs got their wrong year.

    `call` injects the LLM function for tests; the default is `llm.sonnet`.

    Returns the MCAL_PLAN 3.13 object:
    `{year, source_type, confidence, candidates, evidence_quote, note}`.
    """
    mentions = collect_candidates(doc)
    candidates = aggregate_candidates(mentions)

    if not mentions:
        # No evidence to weigh. We skip the LLM call here rather than send an
        # empty prompt: "the adjudicator always runs" (MCAL_PLAN 3.13) is about
        # not gating on M1 disagreement, and asking a model to date a document
        # from no dates is an invitation to hallucinate one.
        return _result(
            year=None,
            source_type="adjudicated",
            confidence="low",
            candidates=[],
            evidence_quote=None,
            note="no year candidates in first 5pp, front-matter tail, or any "
            "signature/transmittal page",
        )

    fallback = fallback_choice(mentions)
    mode = modal_year(mentions)
    notes: list[str] = []

    system, user = build_prompt(mentions, getattr(doc, "doc_id", ""))
    fn = call or _default_call
    raw: Optional[dict] = None
    try:
        raw = fn(system, user, max_tokens=max_tokens)
    except Exception as e:  # noqa: BLE001 - a dating failure must not stop a run
        log.warning("year adjudicator LLM call failed for %s: %s", getattr(doc, "doc_id", "?"), e)
        notes.append(f"llm_call_failed:{type(e).__name__}")
        raw = None
    else:
        if not isinstance(raw, dict):
            notes.append(f"llm_returned_{type(raw).__name__}_not_object")
            raw = None

    year = _coerce_year((raw or {}).get("year"))
    source_type = str((raw or {}).get("source_type") or "").strip().lower()
    confidence = str((raw or {}).get("confidence") or "").strip().lower()
    quote = (raw or {}).get("evidence_quote")
    quote = quote.strip() if isinstance(quote, str) and quote.strip() else None
    llm_note = (raw or {}).get("note")
    llm_note = llm_note.strip() if isinstance(llm_note, str) and llm_note.strip() else None

    if raw is not None and year is None:
        # Either the model returned null (all candidates are non-publication
        # dates, per its own reading) or it returned junk. Either way we prefer
        # the deterministic candidate over an empty answer, and say so.
        notes.append("llm_year_missing_or_out_of_range")

    candidate_years = {m.year for m in mentions}
    if year is None:
        year = fallback.year if fallback else None
        source_type = fallback.source_type if fallback else "adjudicated"
        confidence = "low"
        quote = fallback.context if fallback else None
        notes.append("fell_back_to_regex_candidate")
    else:
        if year not in candidate_years:
            # Accepted, because the model reads text our repair may have missed,
            # but never at high confidence and never silently.
            notes.append(f"llm_year_not_in_regex_candidates:{sorted(candidate_years)}")
            confidence = "low"
        if source_type not in SOURCE_TYPES:
            notes.append(f"invalid_source_type:{source_type or 'missing'}")
            source_type = _best_source_for_year(mentions, year) or "adjudicated"
        if confidence not in CONFIDENCE_LEVELS:
            notes.append(f"invalid_confidence:{confidence or 'missing'}")
            confidence = "low"
        if quote is not None and not _quote_is_grounded(quote, mentions):
            notes.append("evidence_quote_not_grounded_replaced")
            quote = _context_for_year(mentions, year)
        elif quote is None:
            quote = _context_for_year(mentions, year)
            confidence = "low"
            notes.append("no_evidence_quote_returned")

    # A single body-text mention is weak evidence however confident the model is.
    if year is not None and source_type == "body" and confidence == "high":
        confidence = "medium"
        notes.append("downgraded_high_to_medium_for_body_only_evidence")

    if mode is not None and year is not None and mode != year:
        notes.append(f"modal_candidate_was_{mode}")
    if m1_year is not None:
        m1 = _coerce_year(m1_year)
        notes.append(
            f"agrees_with_m1:{m1}" if m1 == year else f"disagrees_with_m1:{m1_year}"
        )
    if llm_note:
        notes.append(llm_note)

    return _result(
        year=year,
        source_type=source_type,
        confidence=confidence,
        candidates=candidates,
        evidence_quote=quote,
        note="; ".join(notes) if notes else None,
    )


def _best_source_for_year(mentions: Sequence[YearMention], year: int) -> Optional[str]:
    hits = [m for m in mentions if m.year == year]
    if not hits:
        return None
    return max(hits, key=lambda m: m.priority).source_type


def _context_for_year(mentions: Sequence[YearMention], year: Optional[int]) -> Optional[str]:
    """Highest-priority context we have for `year`, used when the model gives none."""
    if year is None:
        return None
    hits = [m for m in mentions if m.year == year]
    if not hits:
        return None
    return max(hits, key=lambda m: (m.priority, -m.page)).context


def s_source_agreement(result: dict, m1_year: Optional[int]) -> float:
    """
    Agreement component for `s_source` on the `year` field (MCAL_PLAN 3.3).

    MCAL_PLAN 3.3 defines `s_source` for M1 fields as NUL/regex/Sonnet
    agreement on a {all: 1.0, 2/3: 0.5, disagree: 0.0} scale. This module owns
    two of those three voices -- the OCR-repaired regex candidate and the Sonnet
    adjudication -- so it exposes their agreement here and leaves the inventory
    (NUL) voice to the caller. Weight is 0 through stage v3, so this is logged
    rather than acted on.
    """
    year = result.get("year")
    if year is None:
        return 0.0
    votes = [year]
    cands = result.get("candidates") or []
    if cands:
        votes.append(cands[0].get("year"))
    m1 = _coerce_year(m1_year)
    if m1 is not None:
        votes.append(m1)
    agree = sum(1 for v in votes if v == year)
    return agree / len(votes) if votes else 0.0
