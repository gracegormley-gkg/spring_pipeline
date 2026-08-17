"""
Role-restricted key_people extraction (MCAL_PLAN 3.10 / 4 Q3, build item #7).

Replaces segment_a/m2.py:extract_key_people, whose three buckets
(agency_preparers / cooperating_agencies / public_commenters) failed on 5 of the
8 graded docs with a single mechanism: it fed the "Consultation and Coordination"
chapter to Sonnet and labelled everything it returned a cooperating agency.

(The Evaluation sheet actually grades 6 of 8 as defective: five read "all
commenters = cooperators" and Fuel Economy reads "nearly empty". Those are two
different failures with two different fixes -- the heading whitelist for the
first, the era gate plus the new consulted_entities bucket for the second, since
a 1977 rulemaking has no 1501.8 cooperators to find and the old code had nowhere
to put the entities it did find. Lincoln Hwy is ungraded on this field.)

Real NEPA documents use that chapter as a CATCH-ALL. One chapter routinely
contains, in order: the handful of formally designated cooperating agencies, the
much longer list of consulted agencies, tribal governments, the entire
distribution list of draft-EIS recipients (libraries, NGOs, elected officials),
and the commenters themselves. 40 CFR 1501.8 defines "cooperating agency"
narrowly -- an agency with jurisdiction by law or special expertise that the lead
agency has formally designated -- and the old extractor had no filter
corresponding to that definition. So a library that received a copy of the draft
came out labelled a cooperator, which is both wrong and unfalsifiable from the
output alone.

The fix, per MCAL_PLAN 3.10 and 4 Q3, is to stop treating chapter membership as
evidence of role:

  * cooperating agencies come ONLY from a subsection whose heading
    OCR-normalized-fuzzy-matches {"cooperating agencies", "joint lead agencies",
    "assisting agencies"}. A bare "Consultation and Coordination" heading does
    NOT match -- that is the 5/8 bug, and `test_key_people_pipeline.py` pins it.
  * no such heading -> ONE Sonnet call asking specifically about formal
    designation under 1501.8, and the bucket is routed to HUMAN_REVIEW either
    way (4 Q3). Uncertainty yields an EMPTY list, never a guess.
  * everyone else in the Consultation chapter goes to a NEW `consulted_entities`
    bucket, role-tagged {consulted_agency, tribe, recipient_of_draft, other}.
    Nothing in that bucket can be labelled a cooperator; the enum has no such
    value.
  * commenters come ONLY from Comment/Response chapters. Absent chapter -> empty
    list, not a scan of whatever text mentions "comment".

Two cross-field dependencies are enforced here rather than left to the Critic:

  * the dependent-field cascade (settings.DEPENDENT_FIELDS): if `year` is not
    trustworthy, key_people cannot be era-gated, so the whole field goes to
    HUMAN_REVIEW. Fuel Economy is the live example -- its year Critic verdict is
    HUMAN_REVIEW (NUL says 1977, the regex says 1979).
  * pre-1978 documents predate the 1501.8 schema, so `cooperating_agencies` is
    routed to HUMAN_REVIEW with T13 instead of being extracted against a
    definition that did not exist yet.

Private-individual handling is policy, not calibration (MCAL_PLAN 3.5, 3.11,
5 "Explicitly deferred": "Removing private-individual -> HUMAN_REVIEW (policy
call, permanent)"). `classify_capacity` implements the operational definition and
binds capacity to the passage cited AT THE POINT OF STANCE ATTRIBUTION, so a
mayor commenting officially in chapter 5 and as a resident in chapter 7 is judged
twice, independently; an ambiguous passage is HUMAN_REVIEW regardless of any
Critic verdict.

Every entity carries `evidence` from `verify_and_locate`, and every entity also
carries the inputs the Critic's per-entity role-check needs (MCAL_PLAN 3.10
step 6) so a mislabel becomes RE_EXTRACT + T05 rather than silent output.

Graceful degradation throughout: a missing chapter, an unreadable heading, or
Sonnet returning junk produces an empty bucket plus a note and a tag, never an
exception.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable, Optional, Sequence

from rapidfuzz import fuzz

from mcal import settings
from mcal.quote_check import normalize

# segment_a's flat modules; the sys.path bridge is installed by mcal.settings.
from chunk import detect_chapters, first_pages, text_for_ceq_chapter  # noqa: E402
from evidence import Evidence, verify_and_locate  # noqa: E402
from llm import sonnet  # noqa: E402
from pages import Doc  # noqa: E402

log = logging.getLogger(__name__)


# --- Vocabulary -------------------------------------------------------------

# MCAL_PLAN 3.10 step 3. This whitelist is the whole fix; it must stay exactly
# this narrow. "consultation and coordination", "list of persons consulted" and
# "agencies consulted" are deliberately NOT here.
COOPERATING_HEADINGS = ("cooperating agencies", "joint lead agencies", "assisting agencies")

# MCAL_PLAN 3.10 step 5.
COMMENT_HEADINGS = ("comments received", "response to comments", "public hearing transcripts")

# Used to LOCATE the consultation chapter for the consulted_entities bucket --
# not to license any cooperating-agency claim.
CONSULTATION_HEADINGS = (
    "consultation and coordination",
    "list of persons consulted",
    "agencies consulted",
    "persons and agencies consulted",
    "coordination and consultation",
)

PREPARERS_HEADINGS = ("list of preparers", "preparers", "list of contributors")

CONSULTED_ROLES = ("consulted_agency", "tribe", "recipient_of_draft", "other")
STANCES = ("support", "oppose", "conditional", "neutral")
CAPACITIES = ("private", "non_private", "ambiguous")

# Failure tags, matching the codes mcal/grades.py already infers from human notes.
T_COMMENTER_AS_COOPERATOR = "T05_commenter_mislabeled_as_cooperator"
T_PRE_1978 = "T13_pre_1978_nepa_format"

# 40 CFR 1501.8's predecessor guidance (CEQ 1973 guidelines) postdates 1970 but
# the modern cooperating-agency schema arrives with the 1978 regulations.
NEPA_1978 = 1978

CRITIC_VERDICTS_UNTRUSTWORTHY = ("RE_EXTRACT", "HUMAN_REVIEW")
CRITIC_VERDICTS_TRUSTWORTHY = ("PASS", "PASS_WITH_NOTE")

# `key_people` depends on `year` (settings.DEPENDENT_FIELDS). Read from settings
# rather than hardcoded so the cascade cannot silently disagree with the config
# that gate.py and the manifest also read.
YEAR_DEPENDENTS = tuple(settings.DEPENDENT_FIELDS.get("year", ()))
KEY_PEOPLE_DEPENDS_ON_YEAR = "key_people" in YEAR_DEPENDENTS

MAX_SECTION_CHARS = 40_000
MAX_PROMPT_CHARS = 60_000
# Offsets below this are almost certainly the table of contents (same guard
# chunk.py:93 uses for chapter detection).
TOC_GUARD_CHARS = 3_000
PASSAGE_WINDOW_CHARS = 700


# --- Heading matching -------------------------------------------------------
#
# MCAL_PLAN 3.10 step 3 says "OCR-normalized-fuzzy-matches". Measured on the
# heading strings this corpus actually contains (normalize() from
# mcal/quote_check.py, then rapidfuzz.ratio):
#
#     "COOPERATING AGENCIES"                  vs whitelist -> 100.0   accept
#     "5.2 Cooperating Agencies"              vs whitelist ->  90.9   accept
#     "Cooperating Agency" (singular)         vs whitelist ->  89.5   accept
#     "Coordination with Cooperating Agencies" vs whitelist -> 69.0   accept*
#     "Cooperating and Consulted Agencies"    vs whitelist ->  74.1   REJECT
#     "Consultation and Coordination"         vs whitelist ->  44.9   REJECT
#     "List of Persons Consulted"             vs whitelist ->  41.9   REJECT
#     "Agencies Consulted"                    vs whitelist ->  44.4   REJECT
#
# (*) accepted by the substring rule, not the ratio rule.
#
# So two rules, both required to be conservative:
#   1. the normalized whitelist phrase appears as a substring of the normalized
#      heading (handles numbering, "Coordination with ...", trailing words);
#   2. else rapidfuzz.ratio >= 88 on the whole normalized heading.
#
# token_set_ratio is deliberately NOT used: it scores both "Coordination with
# Cooperating Agencies" and "Cooperating and Consulted Agencies" at 100 because
# it ignores extra tokens, and the second of those is precisely the bundled
# heading we must not treat as authority for a 1501.8 designation. Rejecting it
# costs one Sonnet fallback call and a HUMAN_REVIEW; accepting it re-creates the
# 5/8 failure.
HEADING_RATIO_MIN = 88.0

# Section-number / chapter markers stripped before matching. Roman numerals and
# bare letters must be followed by punctuation AND whitespace: without that
# requirement `[ivxlcdm]+` happily eats the leading "C" of "Cooperating", which
# silently drops the ratio below threshold and re-opens the 5/8 bug from the
# other direction.
_MARKER_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"(?:chapter|section|part|appendix|volume)\s+[\w.\-]+[\s:.\-\u2013)]*"
    r"|\d+(?:\.\d+)*\s*[.:)\-\u2013]?\s+"
    r"|[ivxlcdm]{1,6}[.:)]\s+"
    r"|[a-z][.:)]\s+"
    r")",
    re.IGNORECASE,
)
# "Cooperating Agencies ................ 5-3" -- a TOC row, not a section start.
_DOT_LEADER_RE = re.compile(r"\.{3,}\s*[\dixv\-]+\s*$", re.IGNORECASE)
_TRAILING_PAGENO_RE = re.compile(r"[\s.\u2026]+\d{1,4}(?:[-\u2013]\d{1,4})?\s*$")


@dataclass
class HeadingMatch:
    """Why a heading was accepted, kept for the audit trail."""

    heading: str
    matched_phrase: str
    rule: str          # "substring" | "ratio"
    score: float

    def to_dict(self) -> dict:
        return {
            "heading": self.heading,
            "matched_phrase": self.matched_phrase,
            "rule": self.rule,
            "score": round(self.score, 1),
        }


def strip_heading_decoration(line: str) -> str:
    """Drop numbering, dot leaders and trailing page numbers from a heading."""
    s = (line or "").strip()
    s = _DOT_LEADER_RE.sub("", s)
    s = _TRAILING_PAGENO_RE.sub("", s)
    s = _MARKER_PREFIX_RE.sub("", s)
    return s.strip(" \t.-\u2013:")


def match_heading(line: str, phrases: Sequence[str]) -> Optional[HeadingMatch]:
    """
    Does `line` name a section of one of `phrases`? (MCAL_PLAN 3.10 step 3.)

    Returns the best match or None. See the block comment above for the two
    rules and the measurements behind the threshold.
    """
    raw = (line or "").strip()
    if not raw or len(raw) > 120:
        # Long lines are prose. A heading that long is a false positive risk far
        # bigger than the recall it buys.
        return None
    cleaned = strip_heading_decoration(raw)
    n_head = normalize(cleaned)
    if not n_head:
        return None

    best: Optional[HeadingMatch] = None
    for phrase in phrases:
        n_phrase = normalize(phrase)
        if not n_phrase:
            continue
        if n_phrase in n_head:
            cand = HeadingMatch(raw, phrase, "substring", 100.0)
        else:
            score = float(fuzz.ratio(n_head, n_phrase))
            if score < HEADING_RATIO_MIN:
                continue
            cand = HeadingMatch(raw, phrase, "ratio", score)
        if best is None or cand.score > best.score:
            best = cand
    return best


def _looks_like_toc_row(line: str) -> bool:
    return bool(_DOT_LEADER_RE.search(line or ""))


def looks_like_heading_line(line: str) -> bool:
    """
    Is this line a HEADING at all, independent of what it says?

    `match_heading` answers "does this heading name X"; this answers "is this a
    heading". Both are needed, because prose sentences can score well against a
    whitelist phrase: the Fuel Economy scan contains the sentence "Comments to
    the proposed fuel economy have been received and ..." which fuzzy-matches
    "comments received" but is body text, and treating it as a comment-chapter
    heading would resurrect the "scan wherever the string occurs" behaviour this
    module removes.

    Accepts a line with a section marker, or one that is uppercase-dominant, or
    one whose substantive words are title-cased. Same family of heuristics as
    chunk.py's detector, kept local because chunk.py's is private and tuned for
    CEQ chapter titles rather than subsection headings.
    """
    s = (line or "").strip()
    if not s or len(s) > 120:
        return False
    if s.endswith((",", ";")):
        return False
    # A period followed by a lowercase word is prose, not a heading.
    if re.search(r"\.\s+[a-z]", s):
        return False
    if _MARKER_PREFIX_RE.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.7:
        return True
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'\u2019\-]*", s) if len(w) >= 4]
    if not words:
        return False
    return sum(1 for w in words if w[:1].isupper()) / len(words) >= 0.8


@dataclass
class Section:
    """A located subsection of the document."""

    heading: str
    matched_phrase: str
    match_rule: str
    match_score: float
    start_char: int
    end_char: int
    start_page: int
    end_page: int
    text: str

    def to_dict(self) -> dict:
        return {
            "heading": self.heading,
            "matched_phrase": self.matched_phrase,
            "match_rule": self.match_rule,
            "match_score": round(self.match_score, 1),
            "pages": f"{self.start_page}-{self.end_page}",
            "n_chars": len(self.text),
        }


# A numbered/marked heading is the only reliable section terminator in this
# corpus. Terminating on any ALL-CAPS line would truncate at the first list item,
# because the bodies of these sections ARE lists of agency names in caps. A
# numbered line must ALSO read like a heading once its marker is stripped,
# otherwise "1.5 million dollars in mitigation" at the start of a line would end
# the section.
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:chapter|section|part|appendix)\s+[\w.\-]+|^\s*\d+(?:\.\d+){0,3}\s+\S",
    re.IGNORECASE,
)


def _is_section_terminator(line: str) -> bool:
    if not _NUMBERED_HEADING_RE.match(line or ""):
        return False
    return looks_like_heading_line(strip_heading_decoration(line))


def find_sections(
    doc: Doc,
    phrases: Sequence[str],
    *,
    max_chars: int = MAX_SECTION_CHARS,
    skip_toc_chars: int = TOC_GUARD_CHARS,
    limit: int = 4,
) -> list[Section]:
    """
    Locate body sections whose heading matches one of `phrases`.

    TOC rows (dot leaders) and the front-matter offset window are skipped: a
    "Cooperating Agencies ..... 5-3" line in the contents proves a section
    exists somewhere, but it contains no agencies, and treating it as the
    section would hand the extractor the whole table of contents.
    """
    text = doc.full_text or ""
    out: list[Section] = []
    covered_until = 0
    for m in re.finditer(r"[^\n]+", text):
        if m.start() < skip_toc_chars:
            continue
        # A matching heading inside an already-captured section is almost always
        # the same heading repeated as a running header, or a sub-heading of the
        # section we already hold. Skipping it avoids paying for a duplicate
        # extraction over overlapping text.
        if m.start() < covered_until:
            continue
        line = m.group(0)
        if _looks_like_toc_row(line) or not looks_like_heading_line(line):
            continue
        hit = match_heading(line, phrases)
        if hit is None:
            continue
        body_start = m.end()
        end = _section_end(text, body_start, max_chars)
        section_text = text[body_start:end]
        if not section_text.strip():
            continue
        out.append(
            Section(
                heading=line.strip(),
                matched_phrase=hit.matched_phrase,
                match_rule=hit.rule,
                match_score=hit.score,
                start_char=m.start(),
                end_char=end,
                start_page=doc.page_at_offset(m.start()),
                end_page=doc.page_at_offset(max(m.start(), end - 1)),
                text=section_text,
            )
        )
        covered_until = end
        if len(out) >= limit:
            break
    return out


def _section_end(text: str, body_start: int, max_chars: int) -> int:
    """
    End offset of a section body: the next numbered heading, or a length cap.

    The cap exists because most 1970s scans have no numbering at all; capping
    over-captures, which is the safer error, since the extractor is instructed to
    work only from the passage it is given and a truncated cooperating-agency list
    would silently under-report. A terminator is only honoured once some
    non-whitespace body has been consumed, so a numbered line immediately under
    the heading cannot produce an empty section.
    """
    hard_stop = min(len(text), body_start + max_chars)
    cursor = body_start
    while cursor < hard_stop:
        nl = text.find("\n", cursor)
        if nl < 0 or nl >= hard_stop:
            break
        line = text[cursor:nl]
        if text[body_start:cursor].strip() and _is_section_terminator(line):
            return cursor
        cursor = nl + 1
    return hard_stop


# --- Era gate (MCAL_PLAN 3.10 step 2) ---------------------------------------


@dataclass
class EraGate:
    """
    Outcome of the dependent-field cascade + pre-1978 era check.

    Four fields rather than one flag, because the plan's two rules have different
    scopes and different remedies:
      * `field_human_review` -- the whole key_people field is untrustworthy
        because `year` is untrustworthy (settings.DEPENDENT_FIELDS). Nothing
        downstream can repair that, so it is unconditional.
      * `cooperating_human_review` -- only the cooperating_agencies bucket is
        unsafe. Preparers, consulted entities and commenters are still perfectly
        extractable from a 1971 document.
      * `skip_cooperating_extraction` -- pre-1978 only. The bucket is defined
        against 40 CFR 1501.8, which did not exist yet, so MCAL_PLAN 3.10 step 2
        routes it to HUMAN_REVIEW with T13 instead of extracting against a schema
        the document cannot have followed.
      * `allow_cooperating_fallback` -- whether the Sonnet designation-check
        fallback may run. Suppressed when the era is unknown: that call asks the
        model to judge "formally designated under 1501.8 or its predecessor
        guidance", which is unanswerable if we do not know which regime applies.
        A WHITELISTED HEADING is still honoured in that case -- "COOPERATING
        AGENCIES" over a list is era-independent evidence, and suppressing it
        would throw away the candidate answer a reviewer of a gated field needs
        (MCAL_PLAN 3.12, 7 Q8).
    """

    year: Optional[int] = None
    year_verdict: Optional[str] = None
    field_human_review: bool = False
    cooperating_human_review: bool = False
    skip_cooperating_extraction: bool = False
    allow_cooperating_fallback: bool = True
    tags: list[str] = dc_field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "year_critic_verdict": self.year_verdict,
            "field_human_review": self.field_human_review,
            "cooperating_human_review": self.cooperating_human_review,
            "skip_cooperating_extraction": self.skip_cooperating_extraction,
            "allow_cooperating_fallback": self.allow_cooperating_fallback,
            "tags": list(self.tags),
            "reason": self.reason,
            "dependent_fields_config": {"year": list(YEAR_DEPENDENTS)},
        }


def apply_era_gate(
    year: Optional[int] = None, year_critic_verdict: Optional[str] = None
) -> EraGate:
    """
    MCAL_PLAN 3.10 step 2.

    Ordering matters: the dependent-field cascade is checked FIRST, because
    "year is 1974" is only actionable if we believe the year. A doc whose year
    verdict is HUMAN_REVIEW must not be era-gated on the strength of that same
    unreliable year -- it goes to HUMAN_REVIEW wholesale, and the pre-1978 test
    is not even attempted.

    Three cases the plan does not spell out, resolved conservatively:
      * verdict missing/unknown -> treated as untrustworthy. An absent verdict
        means the Critic never ran on `year`, which is strictly less evidence
        than a RE_EXTRACT.
      * year missing but verdict good -> the field is kept, but the era test
        cannot be performed, so cooperating_agencies is routed to HUMAN_REVIEW
        WITHOUT T13 (we do not know it is a pre-1978 document).
      * settings.DEPENDENT_FIELDS not declaring the dependency -> the cascade is
        skipped and the reason says so, so config and code cannot disagree
        silently.
    """
    verdict = (year_critic_verdict or "").strip().upper() or None
    year_int: Optional[int]
    try:
        year_int = int(year) if year is not None else None
    except (TypeError, ValueError):
        year_int = None

    gate = EraGate(year=year_int, year_verdict=verdict)

    if not KEY_PEOPLE_DEPENDS_ON_YEAR:
        gate.reason = (
            "settings.DEPENDENT_FIELDS does not list key_people as dependent on "
            "year; dependent-field cascade skipped."
        )
        return gate

    if verdict is None or verdict not in CRITIC_VERDICTS_TRUSTWORTHY:
        gate.field_human_review = True
        gate.cooperating_human_review = True
        gate.allow_cooperating_fallback = False
        gate.reason = (
            f"year Critic verdict {verdict or 'MISSING'} is not in "
            f"{CRITIC_VERDICTS_TRUSTWORTHY}; key_people -> HUMAN_REVIEW "
            f"unconditionally via the dependent-field cascade "
            f"(settings.DEPENDENT_FIELDS). The era is unknown, so the "
            f"designation-check fallback is suppressed; a whitelisted heading is "
            f"still honoured."
        )
        return gate

    if year_int is None:
        gate.cooperating_human_review = True
        gate.allow_cooperating_fallback = False
        gate.reason = (
            f"year Critic verdict {verdict} is trustworthy but no year value is "
            f"available, so the pre-1978 era test cannot run; "
            f"cooperating_agencies -> HUMAN_REVIEW."
        )
        return gate

    if year_int < NEPA_1978:
        gate.cooperating_human_review = True
        gate.skip_cooperating_extraction = True
        gate.allow_cooperating_fallback = False
        gate.tags.append(T_PRE_1978)
        gate.reason = (
            f"year={year_int} predates the 1978 CEQ regulations that define a "
            f"cooperating agency (40 CFR 1501.8); cooperating_agencies -> "
            f"HUMAN_REVIEW with {T_PRE_1978}."
        )
        return gate

    gate.reason = f"year={year_int} (verdict {verdict}); post-1978 schema applies."
    return gate


def year_signal_from_artifacts(
    m1: Optional[dict] = None, critic: Optional[dict] = None
) -> tuple[Optional[int], Optional[str]]:
    """
    Pull (year, year_critic_verdict) out of Segment A's on-disk artifacts.

    m1[year][value] and critic[year][verdict] are the shapes segment_a/m1.py and
    segment_a/critic.py actually write; both are tolerated as absent.
    """
    year: Optional[int] = None
    verdict: Optional[str] = None
    if isinstance(m1, dict):
        entry = m1.get("year")
        raw = entry.get("value") if isinstance(entry, dict) else entry
        try:
            year = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            year = None
    if isinstance(critic, dict):
        entry = critic.get("year")
        if isinstance(entry, dict):
            v = entry.get("verdict")
            verdict = str(v).strip().upper() if v else None
    return year, verdict


# --- Capacity classification (MCAL_PLAN 3.5 / 3.11) -------------------------
#
# "A named person is a private individual iff the cited passage does not identify
# them with a government agency, elected office, tribal/nation role, incorporated
# organization, or a professional/expert role relevant to their stance."
#
# Implemented as cue detection over the cited passage only. World knowledge is
# explicitly not consulted -- that is the same anti-hallucination discipline
# MCAL_PLAN 4 Q2 applies to quotes, and it is what makes the classification
# auditable: every decision points at a cue string a reviewer can look up.

_TITLE_CUES = (
    r"dr\.", r"prof\.", r"professor", r"mayor", r"governor", r"senator",
    r"representative", r"congressman", r"congresswoman", r"councilman",
    r"councilwoman", r"council member", r"councilmember", r"commissioner",
    r"supervisor", r"alderman", r"chairman", r"chairwoman", r"chairperson",
    r"chair", r"vice[\- ]chair", r"secretary", r"assistant secretary",
    r"under ?secretary", r"administrator", r"deputy administrator", r"director",
    r"deputy director", r"executive director", r"superintendent", r"chief",
    r"regional forester", r"district ranger", r"state engineer", r"attorney",
    r"colonel", r"lieutenant", r"general", r"captain", r"sheriff", r"judge",
    r"president of", r"vice president", r"manager", r"engineer for",
    r"planner for", r"spokesman for", r"spokesperson for",
)

_AFFILIATION_CUES = (
    r"on behalf of", r"representing", r"speaking for", r"testif\w+ for",
    r"employed by", r"staff of", r"member of the \w+ (?:commission|board|council)",
)

_ORGANIZATION_CUES = (
    r"department of", r"bureau of", r"office of", r"agency", r"administration",
    r"commission", r"authority", r"district", r"board of", r"council of",
    r"corps of engineers", r"service\b", r"\binc\.", r"\bincorporated\b",
    r"\bcorp\b", r"\bcorporation\b", r"\bcompany\b", r"\bco\.", r"\bllc\b",
    r"\bltd\b", r"association", r"institute", r"society", r"foundation",
    r"university", r"college", r"chamber of commerce", r"league", r"union",
    r"federation", r"club\b", r"coalition", r"alliance", r"trust\b",
    r"cooperative", r"railroad", r"utilit\w+", r"school district",
)

_TRIBAL_CUES = (
    r"\btribe\b", r"\btribal\b", r"\bnation\b", r"\bband\b", r"\bpueblo\b",
    r"\brancheria\b", r"\bindian community\b", r"\breservation\b",
    r"tribal council", r"\bchairman of the\b",
)

# Cues that affirmatively mark someone as speaking privately. Their presence
# ALONGSIDE a non-private cue is the dual-capacity case MCAL_PLAN 3.5 sends to
# HUMAN_REVIEW: the passage names both roles and does not settle which one the
# stance belongs to.
_PRIVATE_CUES = (
    r"private citizen", r"private individual", r"as an individual",
    r"in his personal capacity", r"in her personal capacity",
    r"resident of", r"local resident", r"area resident", r"nearby resident",
    r"homeowner", r"home ?owner", r"property owner", r"landowner",
    r"member of the public", r"concerned citizen", r"citizen of",
    r"a farmer", r"a rancher", r"a neighbor",
)

_TITLE_RE = re.compile(r"(?:^|\W)(" + "|".join(_TITLE_CUES) + r")(?:\W|$)", re.IGNORECASE)
_AFFILIATION_RE = re.compile("|".join(_AFFILIATION_CUES), re.IGNORECASE)
_ORGANIZATION_RE = re.compile("|".join(_ORGANIZATION_CUES), re.IGNORECASE)
_TRIBAL_RE = re.compile("|".join(_TRIBAL_CUES), re.IGNORECASE)
_PRIVATE_RE = re.compile("|".join(_PRIVATE_CUES), re.IGNORECASE)

# Names of organizations are not "named persons" at all, so the private/
# non-private test does not apply to them.
_PERSON_NAME_RE = re.compile(
    # Honorifics are matched case-sensitively as written ("Dr.", not "dr.") because
    # the rest of the pattern relies on real capitalization to tell a person's name
    # from a lowercase prose fragment.
    r"^(?:(?:Dr|Mr|Mrs|Ms|Miss|Prof|Rev|Hon)\.?\s+)?"
    r"[A-Z][A-Za-z'\u2019\-]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][A-Za-z'\u2019\-]+){0,2}$"
)


@dataclass
class Capacity:
    """Result of `classify_capacity`."""

    capacity: str                  # private | non_private | ambiguous
    basis: str
    non_private_cues: list[str] = dc_field(default_factory=list)
    private_cues: list[str] = dc_field(default_factory=list)
    name_located: bool = False
    window: str = ""

    @property
    def is_private(self) -> bool:
        return self.capacity == "private"

    @property
    def requires_human_review(self) -> bool:
        """
        Policy, not calibration (MCAL_PLAN 3.11, 5).

        A private individual's stance is HUMAN_REVIEW permanently; an ambiguous
        (dual-capacity) passage is HUMAN_REVIEW "regardless of Critic verdict"
        per MCAL_PLAN 3.5. Only an unambiguously non-private capacity avoids it.
        """
        return self.capacity in ("private", "ambiguous")

    def to_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "basis": self.basis,
            "non_private_cues": list(self.non_private_cues),
            "private_cues": list(self.private_cues),
            "name_located_in_passage": self.name_located,
            "requires_human_review": self.requires_human_review,
        }


def looks_like_person_name(name: str) -> bool:
    """Heuristic person-vs-organization test; organizations are never private."""
    s = (name or "").strip()
    if not s or len(s) > 60:
        return False
    if _ORGANIZATION_RE.search(s) or _TRIBAL_RE.search(s):
        return False
    return bool(_PERSON_NAME_RE.match(s))


def _name_index(name: str, passage: str) -> int:
    """
    Character offset of the first mention of `name` in `passage`, or -1.

    Three attempts, cheapest first: the raw string, the surname alone (comment
    sections write "Mr. Johnson" where the roster wrote "Robert A. Johnson"), and
    finally an OCR-normalized search whose offset is scaled back onto the raw
    string. The scaled offset is approximate, which is fine: it only has to land
    inside the right sentence.
    """
    if not name or not passage:
        return -1
    idx = passage.find(name)
    if idx >= 0:
        return idx
    parts = [p for p in re.split(r"[\s.,]+", name) if len(p) > 2]
    if parts:
        idx = passage.find(parts[-1])
        if idx >= 0:
            return idx
    n_pass, n_name = normalize(passage), normalize(name)
    n_idx = n_pass.find(n_name) if n_name else -1
    if n_idx < 0 and parts:
        n_idx = n_pass.find(normalize(parts[-1]))
    if n_idx < 0:
        return -1
    return min(len(passage) - 1, int(n_idx * (len(passage) / max(1, len(n_pass)))))


_SENTENCE_BOUNDARY = ".;!?\n"
# A title sits immediately before the name and immediately AFTER a sentence
# boundary of its own ("Dr." ends in a period), so the sentence slice is padded
# to the left far enough to keep it.
_ATTRIBUTION_LEFT_PAD = 48


def _attribution_sentence(name: str, passage: str) -> tuple[bool, str]:
    """
    The clause of `passage` where the stance is attributed to `name`.

    MCAL_PLAN 3.5 binds capacity to what the passage states "at the point of
    stance attribution", so cue detection runs on the SENTENCE naming the person,
    not on a fixed character window. A window is actively wrong in a
    comments-received section, where the next sentence names a different
    commenter with a title -- widening the window would make every private
    commenter look non-private, which is the mislabeling failure in a different
    costume.

    Erring narrow is the safe direction: fewer cues means the private default,
    and a private stance is HUMAN_REVIEW anyway.
    """
    idx = _name_index(name, passage)
    if idx < 0:
        return False, ""

    end = len(passage)
    for i in range(idx, min(len(passage), idx + PASSAGE_WINDOW_CHARS)):
        if passage[i] in _SENTENCE_BOUNDARY:
            end = i + 1
            break
    start = 0
    for i in range(idx - 1, max(-1, idx - PASSAGE_WINDOW_CHARS), -1):
        if passage[i] in _SENTENCE_BOUNDARY:
            start = i + 1
            break
    start = max(0, start - _ATTRIBUTION_LEFT_PAD)
    return True, passage[start:end]


def _cue_hits(regex: re.Pattern, text: str, label: str) -> list[str]:
    hits: list[str] = []
    for m in regex.finditer(text or ""):
        hit = (m.group(0) or "").strip().lower()
        if hit and hit not in hits:
            hits.append(f"{label}:{hit}")
        if len(hits) >= 6:
            break
    return hits


def classify_capacity(
    name: str, passage: str, *, kind: Optional[str] = None
) -> Capacity:
    """
    Is this named person speaking privately? (MCAL_PLAN 3.5 operational definition.)

    Decision order:
      1. no name, or no locatable name in the passage -> `ambiguous`. We cannot
         bind a capacity to a passage that does not mention the person, and
         guessing is exactly what the definition forbids.
      2. the "name" is an organization -> `non_private`.
      3. non-private cues AND private cues in the window -> `ambiguous`
         (dual capacity; HUMAN_REVIEW regardless of Critic verdict).
      4. non-private cues only -> `non_private`.
      5. otherwise -> `private`, which is the definition's default: absence of an
         institutional identification IS the private case.

    `kind` (the extractor's own guess) is recorded as a signal but never
    overrides the passage, because the extractor's label is what we are auditing.
    """
    nm = (name or "").strip()
    if not nm:
        return Capacity(capacity="ambiguous", basis="no_name_supplied")

    if not looks_like_person_name(nm):
        cues = (
            _cue_hits(_ORGANIZATION_RE, nm, "org")
            + _cue_hits(_TRIBAL_RE, nm, "tribal")
        )
        return Capacity(
            capacity="non_private",
            basis="entity_name_is_an_organization",
            non_private_cues=cues or ["org:name_shape"],
            name_located=True,
        )

    located, window = _attribution_sentence(nm, passage)
    if not located:
        return Capacity(
            capacity="ambiguous",
            basis="name_not_found_in_cited_passage",
            name_located=False,
        )

    non_private = (
        _cue_hits(_TITLE_RE, window, "title")
        + _cue_hits(_AFFILIATION_RE, window, "affiliation")
        + _cue_hits(_ORGANIZATION_RE, window, "org")
        + _cue_hits(_TRIBAL_RE, window, "tribal")
    )
    private = _cue_hits(_PRIVATE_RE, window, "private")

    if non_private and private:
        return Capacity(
            capacity="ambiguous",
            basis="dual_capacity_cues_in_cited_passage",
            non_private_cues=non_private,
            private_cues=private,
            name_located=True,
            window=window[:400],
        )
    if non_private:
        return Capacity(
            capacity="non_private",
            basis="institutional_identification_in_cited_passage",
            non_private_cues=non_private,
            name_located=True,
            window=window[:400],
        )
    return Capacity(
        capacity="private",
        basis="no_institutional_identification_in_cited_passage",
        private_cues=private,
        name_located=True,
        window=window[:400],
    )


# --- Evidence / passage helpers ---------------------------------------------


def _evidence_for(quote: str, doc: Doc) -> Evidence:
    q = (quote or "").strip()
    if not q:
        return {
            "quote": "",
            "source_pages": [],
            "quote_verified": False,
            "note": "No quote returned by extractor.",
        }
    return verify_and_locate(q, doc)


def cited_passage(doc: Doc, evidence: Evidence, *, fallback: str = "") -> str:
    """
    Text of the page a quote was verified on -- the "cited passage" every
    capacity and role judgement is bound to (MCAL_PLAN 3.5, 3.10 step 6).

    Falls back to the quote itself when verification failed: the Critic still
    needs SOMETHING to look at, and an unverified quote is already flagged.
    """
    pages = (evidence or {}).get("source_pages") or []
    for p in pages:
        try:
            pnum = int(str(p).strip())
        except (TypeError, ValueError):
            continue
        text = doc.text_for_pages(pnum, pnum)
        if text.strip():
            return text
    return fallback or (evidence or {}).get("quote", "") or ""


def _sonnet_json(system: str, user: str, *, max_tokens: int = 2000) -> Optional[dict]:
    """One Sonnet call that must return a JSON object, or None. Never raises."""
    try:
        out = sonnet(system=system, user=user, max_tokens=max_tokens)
    except Exception as e:
        log.warning(f"Sonnet call failed: {e}")
        return None
    if not isinstance(out, dict):
        log.warning(f"Sonnet returned {type(out).__name__}, expected object")
        return None
    return out


def _entity_list(out: Optional[dict], key: str) -> list[dict]:
    if not isinstance(out, dict):
        return []
    raw = out.get(key)
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _clean(value: Any) -> str:
    return str(value or "").strip()


# --- Bucket 1: agency preparers (MCAL_PLAN 3.10 step 1) ---------------------

PREPARERS_SYSTEM = (
    "You list the agency staff who PREPARED this Environmental Impact Statement.\n"
    "Respond ONLY with JSON:\n"
    '{"agency_preparers": [{"name": "<full name>", "role": "<role/title as given>",\n'
    '                       "organization": "<agency/firm or null>",\n'
    '                       "quote": "<verbatim phrase from the excerpt naming this person>"}]}\n'
    "Rules:\n"
    "- Only people identified as preparers, contributors, or reviewers of THIS document.\n"
    "- Do NOT include commenters, consulted agencies, or recipients of the draft.\n"
    "- Do NOT attribute stances.\n"
    "- Quotes MUST be copied character-for-character from the excerpt."
)


def extract_agency_preparers(
    doc: Doc, chapters: Optional[list[dict]] = None
) -> tuple[list[dict], dict]:
    """
    MCAL_PLAN 3.10 step 1: "unchanged -- deterministic from Preparers chapter".

    "Deterministic" refers to the SECTION SELECTION, not to the extraction: the
    list of names still needs a model, because these sections are free-form
    rosters. What is deterministic (and what was already correct in Segment A,
    which is why the plan says leave it alone) is that preparers are read only
    from a Preparers/Consultation section, in that order of preference.
    """
    sections = find_sections(doc, PREPARERS_HEADINGS, limit=2)
    meta: dict = {"source": None, "sections": [s.to_dict() for s in sections]}
    if sections:
        text = "\n\n".join(s.text for s in sections)
        meta["source"] = "preparers_heading"
    else:
        consultation = text_for_ceq_chapter(doc, chapters or [], "Consultation")
        if consultation:
            text = consultation[0]
            meta["source"] = "consultation_chapter"
        else:
            text = first_pages(doc, min(30, doc.n_pages or 30))
            meta["source"] = "first_30_pages_fallback"

    out = _sonnet_json(
        PREPARERS_SYSTEM,
        f"PREPARERS / CONSULTATION EXCERPT:\n{text[:MAX_PROMPT_CHARS]}",
    )
    meta["extractor_ok"] = out is not None
    entries: list[dict] = []
    for item in _entity_list(out, "agency_preparers"):
        name = _clean(item.get("name"))
        if not name:
            continue
        ev = _evidence_for(_clean(item.get("quote")), doc)
        entries.append(
            {
                "name": name,
                "role": _clean(item.get("role")),
                "organization": _clean(item.get("organization")) or None,
                "bucket": "agency_preparers",
                "evidence": [ev],
            }
        )
    meta["n"] = len(entries)
    return entries, meta


# --- Bucket 2: cooperating agencies (MCAL_PLAN 3.10 step 3) -----------------

COOPERATING_SYSTEM = (
    "You list COOPERATING AGENCIES from a section of an EIS whose heading names\n"
    "cooperating / joint lead / assisting agencies.\n"
    "Respond ONLY with JSON:\n"
    '{"cooperating_agencies": [{"name": "<agency or tribal nation>",\n'
    '                           "designation_phrase": "<how the text describes the role>",\n'
    '                           "quote": "<verbatim phrase from the excerpt naming this agency>"}]}\n'
    "Rules:\n"
    "- Include an entity ONLY if this excerpt states it is a cooperating, joint\n"
    "  lead, or assisting agency for this document.\n"
    "- Do NOT include agencies merely consulted, agencies that commented, or\n"
    "  recipients of the draft, even if they appear in this excerpt.\n"
    "- If the excerpt names none, return an empty list.\n"
    "- Quotes MUST be copied character-for-character from the excerpt."
)

# MCAL_PLAN 3.10 step 3 dictates this question verbatim.
COOPERATING_FALLBACK_SYSTEM = (
    "Question: Is any entity described in this document as a formally designated\n"
    "COOPERATING AGENCY under NEPA 40 CFR 1501.8 or its predecessor CEQ guidance?\n"
    "Respond ONLY with JSON:\n"
    '{"answer": "yes|no|uncertain",\n'
    ' "cooperating_agencies": [{"name": "...", "quote": "<verbatim phrase>"}],\n'
    ' "reasoning": "<one sentence>"}\n'
    "Rules:\n"
    "- Answer yes ONLY if the text states a formal designation. Appearing in a\n"
    "  consultation list, a distribution list, or a comment letter is NOT a\n"
    "  designation.\n"
    "- If the text is unclear about formal designation, answer uncertain and\n"
    "  return an empty list. Do not guess.\n"
    "- Quotes MUST be copied character-for-character from the excerpt."
)


def extract_cooperating_agencies(
    doc: Doc,
    chapters: Optional[list[dict]] = None,
    *,
    era_gate: Optional[EraGate] = None,
) -> tuple[list[dict], dict]:
    """
    MCAL_PLAN 3.10 step 3 -- the core fix for the 5/8 failure.

    Extraction is licensed by a HEADING, never by chapter membership. When no
    whitelisted heading exists we make exactly one Sonnet call about formal
    designation and route the bucket to HUMAN_REVIEW either way (MCAL_PLAN 4 Q3),
    because a model's reading of "formally designated" on a 1970s scan is a
    hypothesis, not an extraction. `uncertain` returns an empty list -- an empty
    bucket that a reviewer fills in is recoverable; a fabricated cooperator that
    looks authoritative is not.
    """
    meta: dict = {
        "source": None,
        "sections": [],
        "human_review": False,
        "fallback_answer": None,
        "fallback_reasoning": None,
    }
    if era_gate is not None and era_gate.skip_cooperating_extraction:
        meta["source"] = "skipped_by_era_gate"
        meta["human_review"] = True
        meta["note"] = era_gate.reason
        return [], meta

    sections = find_sections(doc, COOPERATING_HEADINGS, limit=3)
    meta["sections"] = [s.to_dict() for s in sections]

    if sections:
        meta["source"] = "heading_whitelist"
        entries: list[dict] = []
        seen: set[str] = set()
        for section in sections:
            out = _sonnet_json(
                COOPERATING_SYSTEM,
                f"SECTION {section.heading!r} (pages {section.start_page}-"
                f"{section.end_page}):\n{section.text[:MAX_PROMPT_CHARS]}",
            )
            if out is None:
                meta["extractor_ok"] = False
                continue
            meta.setdefault("extractor_ok", True)
            for item in _entity_list(out, "cooperating_agencies"):
                name = _clean(item.get("name"))
                if not name or normalize(name) in seen:
                    continue
                seen.add(normalize(name))
                ev = _evidence_for(_clean(item.get("quote")), doc)
                entries.append(
                    {
                        "name": name,
                        "bucket": "cooperating_agencies",
                        "designation_phrase": _clean(item.get("designation_phrase")),
                        "authority": "heading_whitelist",
                        "heading": section.heading,
                        "heading_pages": f"{section.start_page}-{section.end_page}",
                        "evidence": [ev],
                    }
                )
        meta["n"] = len(entries)
        return entries, meta

    # No whitelisted heading anywhere -> single fallback call + HUMAN_REVIEW.
    meta["human_review"] = True
    if era_gate is not None and not era_gate.allow_cooperating_fallback:
        # The fallback asks about designation "under 1501.8 or its predecessor
        # guidance", which cannot be answered when the era itself is in doubt.
        meta["source"] = "fallback_suppressed_by_era_gate"
        meta["note"] = era_gate.reason
        meta["n"] = 0
        return [], meta

    meta["source"] = "sonnet_fallback"
    excerpt = _consultation_text(doc, chapters)[0] or first_pages(
        doc, min(30, doc.n_pages or 30)
    )
    out = _sonnet_json(
        COOPERATING_FALLBACK_SYSTEM,
        f"DOCUMENT EXCERPT:\n{excerpt[:MAX_PROMPT_CHARS]}",
        max_tokens=1200,
    )
    meta["extractor_ok"] = out is not None
    if out is None:
        meta["fallback_answer"] = "unavailable"
        meta["n"] = 0
        return [], meta

    answer = _clean(out.get("answer")).lower()
    meta["fallback_answer"] = answer or "uncertain"
    meta["fallback_reasoning"] = _clean(out.get("reasoning"))
    if answer != "yes":
        # "no" and "uncertain" both yield an empty list. MCAL_PLAN 3.10 step 3.
        meta["n"] = 0
        return [], meta

    entries = []
    seen = set()
    for item in _entity_list(out, "cooperating_agencies"):
        name = _clean(item.get("name"))
        if not name or normalize(name) in seen:
            continue
        seen.add(normalize(name))
        ev = _evidence_for(_clean(item.get("quote")), doc)
        entries.append(
            {
                "name": name,
                "bucket": "cooperating_agencies",
                "designation_phrase": "",
                "authority": "sonnet_fallback",
                "evidence": [ev],
            }
        )
    meta["n"] = len(entries)
    return entries, meta


# --- Bucket 3: consulted entities (MCAL_PLAN 3.10 step 4) -------------------

CONSULTED_SYSTEM = (
    "You role-tag the entities named in the consultation/coordination section of\n"
    "an EIS. Respond ONLY with JSON:\n"
    '{"consulted_entities": [{"name": "<entity>",\n'
    '                         "role": "consulted_agency|tribe|recipient_of_draft|other",\n'
    '                         "quote": "<verbatim phrase from the excerpt naming this entity>"}]}\n'
    "Role definitions:\n"
    "- consulted_agency: an agency the lead agency consulted or coordinated with.\n"
    "- tribe: a tribal government, nation, band, pueblo or tribal organization.\n"
    "- recipient_of_draft: on the distribution list for the draft EIS (libraries,\n"
    "  clearinghouses, organizations, officials sent a copy).\n"
    "- other: named in this section but none of the above.\n"
    "There is NO cooperating-agency role in this task. Do not invent one.\n"
    "Quotes MUST be copied character-for-character from the excerpt."
)


def _consultation_text(
    doc: Doc, chapters: Optional[list[dict]] = None
) -> tuple[str, str]:
    """(text, provenance) of the consultation chapter, by CEQ map then heading."""
    consultation = text_for_ceq_chapter(doc, chapters or [], "Consultation")
    if consultation and consultation[0].strip():
        return consultation[0], "ceq_consultation_chapter"
    sections = find_sections(doc, CONSULTATION_HEADINGS, limit=2)
    if sections:
        return "\n\n".join(s.text for s in sections), "consultation_heading"
    return "", "not_found"


def extract_consulted_entities(
    doc: Doc,
    chapters: Optional[list[dict]] = None,
    *,
    exclude_names: Iterable[str] = (),
) -> tuple[list[dict], dict]:
    """
    MCAL_PLAN 3.10 step 4: the NEW bucket that absorbs everyone the old code
    mislabelled.

    Cooperating agencies already extracted are excluded by OCR-normalized name
    match, so the same agency does not appear in both buckets. The role enum
    contains no "cooperator" value and anything off-enum is coerced to "other" --
    the label cannot be reintroduced by a model that ignores instructions.
    """
    text, provenance = _consultation_text(doc, chapters)
    meta: dict = {"source": provenance, "n": 0, "extractor_ok": None}
    if not text.strip():
        meta["note"] = "No consultation chapter or heading found."
        return [], meta

    out = _sonnet_json(
        CONSULTED_SYSTEM,
        f"CONSULTATION SECTION:\n{text[:MAX_PROMPT_CHARS]}",
        max_tokens=3000,
    )
    meta["extractor_ok"] = out is not None
    if out is None:
        return [], meta

    excluded = {normalize(n) for n in exclude_names if n}
    entries: list[dict] = []
    seen: set[str] = set()
    for item in _entity_list(out, "consulted_entities"):
        name = _clean(item.get("name"))
        if not name:
            continue
        key = normalize(name)
        if key in excluded or key in seen:
            continue
        seen.add(key)
        role = _clean(item.get("role")).lower()
        if role not in CONSULTED_ROLES:
            role = "other"
        ev = _evidence_for(_clean(item.get("quote")), doc)
        entries.append(
            {
                "name": name,
                "bucket": "consulted_entities",
                "role": role,
                "evidence": [ev],
            }
        )
    meta["n"] = len(entries)
    meta["n_excluded_as_cooperating"] = len(excluded)
    return entries, meta


# --- Bucket 4: public commenters (MCAL_PLAN 3.10 step 5) --------------------

COMMENTERS_SYSTEM = (
    "You list PUBLIC COMMENTERS and their stances from a comments/responses or\n"
    "hearing-transcript section of an EIS. Respond ONLY with JSON:\n"
    '{"public_commenters": [{"name": "<name as written; surname only for private individuals>",\n'
    '                        "kind": "private|organization|official|tribal",\n'
    '                        "stance": "support|oppose|conditional|neutral",\n'
    '                        "affiliation": "<as stated in the passage, or null>",\n'
    '                        "quote": "<verbatim quote attributed to this commenter>"}]}\n'
    "Rules:\n"
    "- Include a commenter ONLY if this excerpt clearly attributes a stance to them.\n"
    "- `affiliation` must come from the passage; null if the passage gives none.\n"
    "- Do not infer a stance from a topic being discussed.\n"
    "- Quotes MUST be copied character-for-character from the excerpt."
)


def extract_public_commenters(
    doc: Doc, chapters: Optional[list[dict]] = None
) -> tuple[list[dict], dict]:
    """
    MCAL_PLAN 3.10 step 5: commenters ONLY from Comment/Response chapters.

    The old code regex-searched the whole document for "response to comments"
    and then read 60k chars from wherever it landed, which is how commenters
    ended up mixed in with the consultation chapter. Here, no matching heading
    means an EMPTY list -- for a Draft EIS with no comment chapter yet, empty is
    the correct answer, not a reason to go looking elsewhere.

    Capacity is classified per commenter against the passage their stance was
    cited from, and a private (or ambiguous) capacity forces HUMAN_REVIEW on the
    whole bucket as a matter of policy (MCAL_PLAN 3.11).
    """
    sections = find_sections(doc, COMMENT_HEADINGS, limit=3)
    meta: dict = {
        "source": "comment_heading" if sections else "no_comment_chapter",
        "sections": [s.to_dict() for s in sections],
        "human_review": False,
        "n": 0,
        "extractor_ok": None,
    }
    if not sections:
        meta["note"] = (
            "No comments-received / response-to-comments / hearing-transcript "
            "heading found; empty list per MCAL_PLAN 3.10 step 5."
        )
        return [], meta

    entries: list[dict] = []
    seen: set[str] = set()
    for section in sections:
        out = _sonnet_json(
            COMMENTERS_SYSTEM,
            f"SECTION {section.heading!r} (pages {section.start_page}-"
            f"{section.end_page}):\n{section.text[:MAX_PROMPT_CHARS]}",
            max_tokens=3000,
        )
        if out is None:
            meta["extractor_ok"] = False
            continue
        if meta["extractor_ok"] is None:
            meta["extractor_ok"] = True
        for item in _entity_list(out, "public_commenters"):
            name = _clean(item.get("name"))
            if not name:
                continue
            stance = _clean(item.get("stance")).lower()
            if stance not in STANCES:
                # A stance we cannot place in the enum is not a stance; keep the
                # commenter but say so, since MCAL_PLAN 3.5 Q4 asks whether
                # stances are attributed at all.
                stance = ""
            quote = _clean(item.get("quote"))
            ev = _evidence_for(quote, doc)
            passage = cited_passage(doc, ev, fallback=section.text[:4000])
            capacity = classify_capacity(name, passage, kind=_clean(item.get("kind")))
            key = f"{normalize(name)}|{stance}"
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "name": name,
                    "bucket": "public_commenters",
                    "kind": _clean(item.get("kind")).lower() or None,
                    "affiliation": _clean(item.get("affiliation")) or None,
                    "stance": stance or None,
                    "capacity": capacity.to_dict(),
                    "human_review": capacity.requires_human_review,
                    "heading": section.heading,
                    "heading_pages": f"{section.start_page}-{section.end_page}",
                    "evidence": [ev],
                }
            )
    meta["n"] = len(entries)
    meta["human_review"] = any(e["human_review"] for e in entries)
    meta["n_private_or_ambiguous"] = sum(1 for e in entries if e["human_review"])
    return entries, meta


# --- Critic role-check hook (MCAL_PLAN 3.10 step 6) -------------------------

ROLE_CHECK_QUESTION = (
    "Is this entity described in the cited passage as (a) a formally-designated "
    "cooperating agency under NEPA 40 CFR 1501.8 (or its predecessor CEQ "
    "guidance), (b) a public commenter, or (c) neither?"
)

ROLE_CHECK_OPTIONS = {
    "a": "formally-designated cooperating agency",
    "b": "public commenter",
    "c": "neither",
}

# The answer that must come back for the bucket we placed the entity in.
# Preparers and consulted entities are both "neither" by construction, which is
# the point: the check is specifically looking for the cooperator/commenter
# confusion that produced the 5/8 failure.
ROLE_CHECK_EXPECTED = {
    "cooperating_agencies": "a",
    "public_commenters": "b",
    "consulted_entities": "c",
    "agency_preparers": "c",
}


def role_check_items(result: dict, doc: Doc) -> list[dict]:
    """
    Build the per-entity payloads the Critic needs for its role check.

    Everything the Critic must see is inlined -- the entity, the bucket we
    claimed, the verbatim quote, the cited pages, and the passage text itself --
    so the check can be run without re-opening the document, matching the
    self-contained-manifest requirement in MCAL_PLAN 3.12.
    """
    items: list[dict] = []
    for bucket in ("cooperating_agencies", "consulted_entities", "public_commenters",
                   "agency_preparers"):
        for i, entity in enumerate(result.get(bucket) or []):
            ev_list = entity.get("evidence") or []
            ev: Evidence = ev_list[0] if ev_list else {}
            items.append(
                {
                    "item_id": f"{bucket}[{i}]",
                    "bucket": bucket,
                    "entity": entity.get("name", ""),
                    "claimed_role": _claimed_role(bucket, entity),
                    "question": ROLE_CHECK_QUESTION,
                    "options": dict(ROLE_CHECK_OPTIONS),
                    "expected_answer": ROLE_CHECK_EXPECTED[bucket],
                    "evidence_quote": ev.get("quote", ""),
                    "source_pages": list(ev.get("source_pages") or []),
                    "quote_verified": bool(ev.get("quote_verified")),
                    "cited_passage": cited_passage(doc, ev)[:6000],
                }
            )
    return items


def _claimed_role(bucket: str, entity: dict) -> str:
    if bucket == "cooperating_agencies":
        return "cooperating_agency"
    if bucket == "public_commenters":
        return f"public_commenter/{entity.get('stance') or 'unspecified_stance'}"
    if bucket == "consulted_entities":
        return entity.get("role") or "other"
    return "agency_preparer"


def apply_role_check_answers(result: dict, answers: dict[str, str]) -> dict:
    """
    Fold Critic role-check answers back into the result (MCAL_PLAN 3.10 step 6).

    A mismatch is RE_EXTRACT + T05. Note the asymmetry: an entity we called a
    cooperating agency that the Critic reads as a commenter is the exact 1(10)
    failure, so it is reported with T05 in either direction of confusion -- the
    tag names the confusion, not which side of it we landed on. Answers we did
    not ask for are ignored; items we asked about but got no answer for are
    reported as unanswered rather than silently passed.
    """
    mismatches: list[dict] = []
    unanswered: list[str] = []
    checked = 0
    for item in result.get("role_check_items") or []:
        got = str(answers.get(item["item_id"], "")).strip().lower()[:1]
        if got not in ROLE_CHECK_OPTIONS:
            unanswered.append(item["item_id"])
            continue
        checked += 1
        if got != item["expected_answer"]:
            mismatches.append(
                {
                    "item_id": item["item_id"],
                    "bucket": item["bucket"],
                    "entity": item["entity"],
                    "claimed": item["expected_answer"],
                    "critic_says": got,
                }
            )

    out = {
        "n_checked": checked,
        "n_unanswered": len(unanswered),
        "unanswered": unanswered,
        "mismatches": mismatches,
        "verdict": "RE_EXTRACT" if mismatches else "PASS",
        "tags": [T_COMMENTER_AS_COOPERATOR] if mismatches else [],
    }
    result["role_check_result"] = out
    if mismatches:
        for t in out["tags"]:
            if t not in result["tags"]:
                result["tags"].append(t)
    return out


# --- Top level --------------------------------------------------------------


def run_key_people_pipeline(
    doc: Doc,
    chapters: Optional[list[dict]] = None,
    *,
    year: Optional[int] = None,
    year_critic_verdict: Optional[str] = None,
    m1: Optional[dict] = None,
    critic: Optional[dict] = None,
) -> dict:
    """
    Full role-restricted key_people pipeline (MCAL_PLAN 3.10, build item #7).

    `year` / `year_critic_verdict` may be passed directly, or inferred from
    Segment A's `m1` / `critic` artifacts.

    Output contract:
        {
          agency_preparers, cooperating_agencies, consulted_entities,
          public_commenters,        # four buckets, each a list of entities
          era_gate,                 # the dependent-field + pre-1978 decision
          human_review,             # field-level
          human_review_fields,      # per-bucket
          role_check_items,         # MCAL_PLAN 3.10 step 6 inputs
          tags, notes, sources
        }

    The field-level HUMAN_REVIEW from the era gate does NOT suppress extraction.
    MCAL_PLAN 3.12/7 Q8 require the raw extraction to be emitted alongside the
    gate decision -- a reviewer looking at a gated field needs the candidate
    answer in front of them, and suppressing it would make the multi-round
    protocol (7.5) unable to collect grades on gated fields.
    """
    if year is None and year_critic_verdict is None and (m1 or critic):
        year, year_critic_verdict = year_signal_from_artifacts(m1, critic)

    # Normally handed in by the M2 driver (chunk.chunks_for_doc). Recovered here
    # so the module works standalone: without chapters, the consulted_entities
    # bucket loses its CEQ-chapter route and falls back to a heading scan only.
    if chapters is None:
        try:
            chapters = detect_chapters(doc)
        except Exception as e:  # pragma: no cover - detector is pure regex
            log.warning(f"chapter detection failed: {e}")
            chapters = []

    gate = apply_era_gate(year, year_critic_verdict)
    tags: list[str] = list(gate.tags)
    notes: list[str] = [gate.reason] if gate.reason else []

    preparers, preparers_meta = extract_agency_preparers(doc, chapters)
    cooperating, cooperating_meta = extract_cooperating_agencies(
        doc, chapters, era_gate=gate
    )
    consulted, consulted_meta = extract_consulted_entities(
        doc, chapters, exclude_names=[e["name"] for e in cooperating]
    )
    commenters, commenters_meta = extract_public_commenters(doc, chapters)

    if cooperating_meta.get("source") == "sonnet_fallback":
        notes.append(
            "No heading matched the cooperating-agency whitelist; used the single "
            "Sonnet designation-check fallback and routed the bucket to "
            "HUMAN_REVIEW (MCAL_PLAN 3.10 step 3, 4 Q3). Answer: "
            f"{cooperating_meta.get('fallback_answer')}."
        )
    elif cooperating_meta.get("source") == "fallback_suppressed_by_era_gate":
        notes.append(
            "No heading matched the cooperating-agency whitelist and the era gate "
            "suppressed the designation-check fallback; empty list + HUMAN_REVIEW."
        )
    if commenters_meta.get("source") == "no_comment_chapter":
        notes.append(str(commenters_meta.get("note") or ""))
    if commenters_meta.get("n_private_or_ambiguous"):
        notes.append(
            f"{commenters_meta['n_private_or_ambiguous']} commenter stance(s) are "
            f"attributed to a private individual or an ambiguous capacity -> "
            f"mandatory HUMAN_REVIEW (MCAL_PLAN 3.5, 3.11; policy, not calibrated)."
        )

    human_review_fields = {
        "agency_preparers": False,
        "cooperating_agencies": bool(
            gate.cooperating_human_review or cooperating_meta.get("human_review")
        ),
        "consulted_entities": False,
        "public_commenters": bool(commenters_meta.get("human_review")),
    }
    field_human_review = gate.field_human_review or any(human_review_fields.values())

    result: dict = {
        "agency_preparers": preparers,
        "cooperating_agencies": cooperating,
        "consulted_entities": consulted,
        "public_commenters": commenters,
        "era_gate": gate.to_dict(),
        "human_review": field_human_review,
        "human_review_fields": human_review_fields,
        "tags": _dedupe(tags),
        "notes": [n for n in notes if n],
        "sources": {
            "agency_preparers": preparers_meta,
            "cooperating_agencies": cooperating_meta,
            "consulted_entities": consulted_meta,
            "public_commenters": commenters_meta,
        },
        "counts": {
            "agency_preparers": len(preparers),
            "cooperating_agencies": len(cooperating),
            "consulted_entities": len(consulted),
            "public_commenters": len(commenters),
        },
    }
    result["role_check_items"] = role_check_items(result, doc)
    return result


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for i in items:
        if i:
            seen.setdefault(i, None)
    return list(seen.keys())


def as_m2_key_people_field(result: dict) -> dict:
    """
    Adapt the pipeline output to the `key_people` field shape M2 emits, so the
    Critic, grading sheets and gate.py keep working unchanged.

    The three Segment A keys keep their names and meanings; `consulted_entities`
    is additive. `comment_response_present` is retained because grading.py
    renders it, and it now means what it says -- a comment/response HEADING was
    found -- rather than "the string 'response to comments' occurs somewhere".
    """
    commenters_meta = (result.get("sources") or {}).get("public_commenters") or {}
    return {
        "value": {
            "agency_preparers": result.get("agency_preparers", []),
            "cooperating_agencies": result.get("cooperating_agencies", []),
            "consulted_entities": result.get("consulted_entities", []),
            "public_commenters": result.get("public_commenters", []),
            "comment_response_present": commenters_meta.get("source")
            == "comment_heading",
        },
        "confidence": "low" if result.get("human_review") else "high",
        "tags": result.get("tags", []),
        "human_review": result.get("human_review", False),
        "human_review_fields": result.get("human_review_fields", {}),
        "era_gate": result.get("era_gate", {}),
        "note": " ".join(result.get("notes", []))[:1000],
    }
