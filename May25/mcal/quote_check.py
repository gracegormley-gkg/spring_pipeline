"""
OCR-normalized fuzzy quote verification (MCAL_PLAN 3.2, build item #1).

Upgrade of segment_a/pages.py:find_quote, which does a whitespace-collapsed
EXACT substring search on a single page. That is too brittle for 1970s OCR
scans: `rn`/`m`, `l`/`1`/`I`, `O`/`0`, `S`/`5` confusions and stray hyphenation
make a genuinely-present quote unfindable, which manifests downstream as a
false "unsupported claim".

This module is the single point of truth for quote verification in M-Cal and
Segment B. It intentionally mirrors segment_a/evidence.py's advice -- callers
should NOT roll their own substring search.

Three-valued verdict, per MCAL_PLAN 3.2:
    partial_ratio >= 90  -> "yes"    (s_quote 1.0)
    60 <= ratio  <  90   -> "mixed"  (s_quote 0.5)
    ratio        <  60   -> "no"     (s_quote 0.0)

Page tolerance is +/-2. MCAL_PLAN 0 justifies that as compensation for page
numbers estimated via char_offset/2500; that estimation does not exist in the
current code (pages are exact, read from per-page JSON, and segment_a chunks in
real pages). The tolerance is retained on different grounds:
  * pages.find_quote deliberately refuses to match across page seams
    (pages.py:89-90), so a passage straddling a seam is cited on one page but
    only partially present there;
  * extractors cite the page a section starts on rather than the page a
    specific sentence lands on;
  * OCR page-order glitches on microfilm scans.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field as dc_field
from typing import Iterable, Optional, Sequence

from rapidfuzz import fuzz

from . import settings

# segment_a bridge is installed by settings import.
from pages import Doc  # noqa: E402


# --- OCR normalization ------------------------------------------------------
# Applied identically to both sides of every comparison. Order matters: we
# fold case and confusable glyphs onto a single canonical form, so `rn` -> `m`
# must run before `m` is used in any later rule.

# Ligatures and typographic characters that OCR emits inconsistently.
_UNICODE_FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00ad": "",   # soft hyphen
    "\u2026": "...",
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u2009": " ",
}

# Confusable-glyph classes. Every member of a class collapses to the class
# representative, so `Modoc`/`M0doc`/`M0d0c` all normalize alike.
_CONFUSABLES = {
    "o": "0",   # O <-> 0
    "0": "0",
    "l": "1",   # l <-> 1 <-> I
    "i": "1",
    "1": "1",
    "s": "5",   # S <-> 5
    "5": "5",
    "b": "8",   # B <-> 8
    "8": "8",
    "g": "9",   # G <-> 9  (common on low-DPI microfilm)
    "9": "9",
    "z": "2",   # Z <-> 2
    "2": "2",
}

_RN_M = re.compile(r"rn")
_VV_W = re.compile(r"vv")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
# Hyphen followed by a line break = OCR'd hyphenation; join the word halves.
_HYPHEN_BREAK = re.compile(r"-\s*\n\s*")


def normalize(text: str, *, strip_punct: bool = True, confusables: bool = True) -> str:
    """
    OCR-normalize a string for comparison.

    Steps, in order:
      1. NFKC unicode normalization + explicit ligature/quote/dash folding
      2. de-hyphenate across line breaks
      3. lowercase
      4. `rn` -> `m`, `vv` -> `w` (the two most common OCR splits)
      5. strip punctuation
      6. collapse confusable glyph classes (o/0, l/1/i, s/5, b/8, g/9, z/2)
      7. collapse whitespace

    Steps 5 and 6 are toggleable because numeric verification needs to compare
    digits faithfully -- see `numeric_tokens`.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    for a, b in _UNICODE_FOLD.items():
        s = s.replace(a, b)
    s = _HYPHEN_BREAK.sub("", s)
    s = s.lower()
    s = _RN_M.sub("m", s)
    s = _VV_W.sub("w", s)
    if strip_punct:
        s = _PUNCT.sub(" ", s)
    if confusables:
        s = "".join(_CONFUSABLES.get(ch, ch) for ch in s)
    s = _WS.sub(" ", s)
    return s.strip()


# --- Content-token coverage -------------------------------------------------
# Second gate, orthogonal to partial_ratio. See the long note in
# settings.QUOTE_COVERAGE_YES for the measurements motivating it: partial_ratio
# has a ~50-68 chance floor on short strings, so a character-ratio-only gate
# systematically false-accepts short fabricated atoms.

# Function words plus the NEPA boilerplate that appears on nearly every page of
# nearly every EIS. Including the domain filler is deliberate: "environmental",
# "impact" and "project" carry almost no discriminating information in this
# corpus, so counting them as evidence of a match inflates coverage.
#
# NOTE: these must be stored in NORMALIZED form. content_tokens() runs on the
# output of normalize(), where confusable folding has already turned
# "environmental" into "env1r0nmenta1"; a plain-text stopword set would never
# match anything and the filter would be silently inert.
_STOPWORDS_RAW = """
    the a an and or of to in for on at by with from as is are was were be been
    being that this these those it its would will may can could shall should
    not no nor which who whom whose than then there their them they he she his
    her have has had do does did but if when while into over under about above
    below between during after before such other any all both each more most
    some only same so too very own also however therefore thus within
    environmental impact statement project proposed action alternative
    alternatives area areas page section chapter table figure
""".split()

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_TOKEN_LEN = 4

# Folded at import time with the same function that will process the inputs, so
# the two sides are guaranteed to agree.
_STOPWORDS = frozenset(normalize(w) for w in _STOPWORDS_RAW)


def content_tokens(normalized: str) -> list[str]:
    """
    Content words of an already-normalized string.

    Expects `normalize()` output. Keeps tokens of >=4 chars that are not
    stopwords, plus any all-digit token regardless of length so that figures
    ("7", "5", "1200") still count as evidence.

    The numeric test is `str.isdigit()`, deliberately NOT "contains a digit":
    confusable folding rewrites letters as digits ("environmental" ->
    "env1r0nmenta1"), so a contains-a-digit test would classify most folded
    words as numbers and bypass the stopword filter entirely. Real numbers fold
    to pure-digit tokens because punctuation is stripped ("7.5" -> "7", "5"),
    while folded words stay mixed alpha-digit.
    """
    out = []
    for t in _TOKEN_RE.findall(normalized or ""):
        if t.isdigit():
            out.append(t)
        elif len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS:
            out.append(t)
    return out


def content_coverage(normalized_quote: str, normalized_page: str) -> Optional[float]:
    """
    Fraction of the quote's content tokens present in the page.

    Returns None when the quote has too few content tokens to score, in which
    case the caller falls back to the ratio gate alone.
    """
    q_tokens = content_tokens(normalized_quote)
    if len(q_tokens) < settings.QUOTE_COVERAGE_MIN_TOKENS:
        return None
    page_set = set(content_tokens(normalized_page))
    if not page_set:
        return 0.0
    hits = sum(1 for t in q_tokens if t in page_set)
    return hits / len(q_tokens)


# --- Numeric handling -------------------------------------------------------
# Confusable folding would make "7.5" and "7.S" identical, which is desirable
# for prose but catastrophic for MCAL_PLAN 1(4) (Magnitude 7.5 vs 7.0). Numeric
# claims are therefore compared on a separate, unfolded channel.

_NUM_RE = re.compile(
    r"""
    (?<![\w.])
    (?:\$\s*)?                    # optional currency
    \d{1,3}(?:,\d{3})+(?:\.\d+)?  # 1,200  or 1,234.5
    | (?<![\w.])(?:\$\s*)?\d+(?:\.\d+)?   # 47  or 7.5  or $369
    """,
    re.VERBOSE,
)

# Magnitude words that scale a bare number; kept so "$369 million" != "$369".
_SCALE_WORDS = ("trillion", "billion", "million", "thousand", "hundred")


def numeric_tokens(text: str) -> list[str]:
    """
    Extract comparable numeric tokens, preserving digits exactly.

    Runs on a punctuation-preserving, confusable-free normalization so that
    7.5 and 7.0 stay distinct. Thousands separators are stripped and a
    following scale word is appended, so "$369 million" -> "369million" and
    "1,200 acres" -> "1200".
    """
    if not text:
        return []
    s = unicodedata.normalize("NFKC", text)
    for a, b in _UNICODE_FOLD.items():
        s = s.replace(a, b)
    s = _HYPHEN_BREAK.sub("", s).lower()

    out: list[str] = []
    for m in _NUM_RE.finditer(s):
        raw = m.group(0)
        num = raw.replace("$", "").replace(",", "").strip()
        if not num:
            continue
        num = num.rstrip(".")
        tail = s[m.end() : m.end() + 24]
        for w in _SCALE_WORDS:
            if re.match(rf"\s*{w}\b", tail):
                num = f"{num}{w}"
                break
        out.append(num)
    return out


# --- Result type ------------------------------------------------------------


@dataclass
class QuoteCheck:
    """Outcome of verifying one quote against a page range."""

    verified: str                      # "yes" | "mixed" | "no"
    score: float                       # best partial_ratio observed, 0-100
    coverage: Optional[float] = None   # content-token coverage on matched page
    matched_page: Optional[int] = None
    normalized_match_span: Optional[tuple[int, int, int]] = None  # (page, start, end)
    pages_searched: list[int] = dc_field(default_factory=list)
    # Numeric agreement, only meaningful when the quote carries figures.
    numeric_ok: Optional[bool] = None
    quote_numbers: list[str] = dc_field(default_factory=list)
    missing_numbers: list[str] = dc_field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.verified == "yes"

    @property
    def s_quote(self) -> float:
        """Confidence signal contribution (MCAL_PLAN 3.3)."""
        return settings.QUOTE_VERDICT_SCORES[self.verified]

    def to_dict(self) -> dict:
        return {
            "verified": self.verified,
            "score": round(self.score, 2),
            "coverage": round(self.coverage, 3) if self.coverage is not None else None,
            "matched_page": self.matched_page,
            "normalized_match_span": list(self.normalized_match_span)
            if self.normalized_match_span
            else None,
            "pages_searched": self.pages_searched,
            "numeric_ok": self.numeric_ok,
            "quote_numbers": self.quote_numbers,
            "missing_numbers": self.missing_numbers,
            "reason": self.reason,
        }


# --- Page-range helpers -----------------------------------------------------


def coerce_pages(source_pages) -> list[int]:
    """
    Normalize a source_pages value into a sorted list of ints.

    segment_a emits `list[str]` (evidence.py:60 -> ["27"]) while MCAL_PLAN
    specs `list[int]`, and grading sheets carry span strings like "12-14".
    Accept all three shapes.
    """
    if source_pages is None:
        return []
    if isinstance(source_pages, (int, float)):
        return [int(source_pages)]
    if isinstance(source_pages, str):
        source_pages = [source_pages]

    out: set[int] = set()
    for item in source_pages:
        if item is None:
            continue
        if isinstance(item, (int, float)):
            out.add(int(item))
            continue
        s = str(item).strip()
        if not s:
            continue
        m = re.fullmatch(r"(\d+)\s*[-\u2013]\s*(\d+)", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            # Guard against a malformed span swallowing the whole doc.
            if b - a <= 200:
                out.update(range(a, b + 1))
            else:
                out.update({a, b})
            continue
        m = re.search(r"\d+", s)
        if m:
            out.add(int(m.group(0)))
    return sorted(out)


def expand_with_tolerance(
    pages: Sequence[int], tolerance: int = settings.QUOTE_PAGE_TOLERANCE
) -> list[int]:
    """Widen a page list by +/-tolerance (MCAL_PLAN 3.2). Never returns page < 1."""
    out: set[int] = set()
    for p in pages:
        for q in range(p - tolerance, p + tolerance + 1):
            if q >= 1:
                out.add(q)
    return sorted(out)


# --- Core verification ------------------------------------------------------


def check_quote(
    quote: str,
    source_pages,
    doc: Doc,
    *,
    tolerance: int = settings.QUOTE_PAGE_TOLERANCE,
    require_numeric: bool = False,
    search_whole_doc_if_no_pages: bool = False,
) -> QuoteCheck:
    """
    Verify `quote` against `doc` within `source_pages` +/- tolerance.

    `require_numeric`: when True, every numeric token in the quote must also
    appear in the matched page text or the verdict is downgraded. Used for
    atomic claims typed `numeric` (MCAL_PLAN 3.4 step 2). Note that a numeric
    downgrade caps the verdict at "mixed" rather than forcing "no" -- the prose
    really is present, only the figure is unconfirmed, and conflating the two
    would hide which of the two failure modes occurred.

    An empty quote is "no", never "yes". MCAL_PLAN 3.5 requires the Critic to
    emit a >=20-char evidence_quote or set it null and RE_EXTRACT; returning
    "yes" for an empty string would silently defeat that.
    """
    q = (quote or "").strip()
    if not q:
        return QuoteCheck(verified="no", score=0.0, reason="empty_quote")

    norm_q = normalize(q)
    if not norm_q:
        return QuoteCheck(
            verified="no", score=0.0, reason="quote_normalized_to_empty"
        )

    cited = coerce_pages(source_pages)
    if cited:
        candidates = expand_with_tolerance(cited, tolerance)
    elif search_whole_doc_if_no_pages:
        candidates = [p.page_num for p in doc.pages]
    else:
        return QuoteCheck(
            verified="no", score=0.0, reason="no_source_pages_cited"
        )

    by_num = {p.page_num: p.text for p in doc.pages}
    searched = [p for p in candidates if p in by_num]
    if not searched:
        return QuoteCheck(
            verified="no",
            score=0.0,
            pages_searched=[],
            reason=f"cited_pages_not_in_doc:{cited}",
        )

    # Score every candidate page; keep the best. Cited pages are preferred over
    # tolerance-expanded neighbours on ties so that matched_page reports the
    # page the extractor actually claimed when both match equally well.
    #
    # Ranking is on (ratio, coverage) jointly rather than ratio alone: a page
    # that shares a lot of vocabulary with the quote is a better candidate than
    # one that happens to yield a lucky sliding-window alignment.
    cited_set = set(cited)
    best_score = -1.0
    best_cov: Optional[float] = None
    best_page: Optional[int] = None
    best_norm_page = ""
    for pnum in searched:
        norm_page = normalize(by_num[pnum])
        if not norm_page:
            continue
        score = float(fuzz.partial_ratio(norm_q, norm_page))
        cov = content_coverage(norm_q, norm_page)
        rank = (score, cov if cov is not None else 0.0)
        best_rank = (best_score, best_cov if best_cov is not None else 0.0)
        better = rank > best_rank
        tie_prefers_cited = (
            rank == best_rank
            and pnum in cited_set
            and best_page is not None
            and best_page not in cited_set
        )
        if better or tie_prefers_cited:
            best_score, best_cov, best_page, best_norm_page = score, cov, pnum, norm_page

    if best_page is None:
        return QuoteCheck(
            verified="no",
            score=0.0,
            pages_searched=searched,
            reason="all_candidate_pages_empty",
        )

    # Both gates must pass. Coverage is the binding constraint in practice and
    # is what rejects short fabrications that clear the ratio floor by chance.
    ratio_yes = best_score >= settings.QUOTE_RATIO_YES
    ratio_mixed = best_score >= settings.QUOTE_RATIO_MIXED
    if best_cov is None:
        cov_yes = cov_mixed = True  # too few content tokens to score
    else:
        cov_yes = best_cov >= settings.QUOTE_COVERAGE_YES
        cov_mixed = best_cov >= settings.QUOTE_COVERAGE_MIXED

    if ratio_yes and cov_yes:
        verdict = "yes"
    elif ratio_mixed and cov_mixed:
        verdict = "mixed"
    else:
        verdict = "no"

    cov_str = "n/a" if best_cov is None else f"{best_cov:.2f}"
    span = _match_span(norm_q, best_norm_page)
    res = QuoteCheck(
        verified=verdict,
        score=best_score,
        coverage=best_cov,
        matched_page=best_page,
        normalized_match_span=(best_page, span[0], span[1]) if span else None,
        pages_searched=searched,
        reason=f"partial_ratio={best_score:.1f} coverage={cov_str}",
    )

    # --- numeric channel ---
    nums = numeric_tokens(q)
    if nums:
        res.quote_numbers = nums
        page_nums = set(numeric_tokens(by_num[best_page]))
        # Also accept a bare figure when the page writes it with a scale word,
        # and vice versa: "369million" should satisfy a claim of "369".
        expanded = set(page_nums)
        for n in page_nums:
            for w in _SCALE_WORDS:
                if n.endswith(w):
                    expanded.add(n[: -len(w)])
        missing = [n for n in nums if n not in expanded]
        res.missing_numbers = missing
        res.numeric_ok = not missing
        if missing and require_numeric and verdict == "yes":
            res.verified = "mixed"
            res.reason = (
                f"partial_ratio={best_score:.1f} coverage={cov_str} but numbers "
                f"absent from page {best_page}: {missing}"
            )

    return res


def _match_span(norm_quote: str, norm_page: str) -> Optional[tuple[int, int]]:
    """
    Best-effort character span of the match within the normalized page.

    Exact hit first (cheap and precise); otherwise use rapidfuzz's alignment.
    """
    idx = norm_page.find(norm_quote)
    if idx >= 0:
        return idx, idx + len(norm_quote)
    try:
        al = fuzz.partial_ratio_alignment(norm_quote, norm_page)
    except Exception:  # pragma: no cover - alignment is best-effort only
        return None
    if al is None:
        return None
    return al.dest_start, al.dest_end


# --- Batch helpers ----------------------------------------------------------


def check_evidence_list(evidence_list: Iterable[dict], doc: Doc, **kw) -> list[QuoteCheck]:
    """Verify a segment_a-shaped evidence list (`{quote, source_pages, ...}`)."""
    return [
        check_quote(ev.get("quote", ""), ev.get("source_pages"), doc, **kw)
        for ev in (evidence_list or [])
    ]


def aggregate_verdict(checks: Sequence[QuoteCheck]) -> str:
    """
    Collapse several quote checks into one field-level verdict.

    All yes -> "yes"; all no (or empty) -> "no"; anything else -> "mixed".
    Matches the vocabulary grading.py already renders in the `quote_verified`
    column, so the two stay legible side by side.
    """
    if not checks:
        return "no"
    verdicts = {c.verified for c in checks}
    if verdicts == {"yes"}:
        return "yes"
    if verdicts == {"no"}:
        return "no"
    return "mixed"


def s_quote_for(checks: Sequence[QuoteCheck]) -> float:
    """
    Mean s_quote across a field's quotes.

    Averaged rather than min-ed: a summary with four supported claims and one
    unsupported one is materially better than one with five unsupported
    claims, and the CP threshold is what decides whether that difference is
    good enough. An empty list scores 0.0 -- no evidence is not the same as
    verified evidence.
    """
    if not checks:
        return 0.0
    return sum(c.s_quote for c in checks) / len(checks)
