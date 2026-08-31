"""
M1: Easily Gathered (NUL-first, not LLM-first).

Fields:
  - title          (inventory index only -- deterministic, no LLM)
  - year           (NUL primary, regex on first 3 pages fallback)
  - eis_type       (regex on first page primary, Sonnet on first 2 pages verifier)
  - lead_agency    (NUL primary, Sonnet on first 4 pages fallback, list-valued)

Each field returns {value, confidence, sources}.
"""

from __future__ import annotations

import re
from typing import Optional

from config import (
    EIS_TYPE_PATTERNS,
    FIRST_2_PAGES,
    FIRST_3_PAGES,
    FIRST_4_PAGES,
    FIRST_PAGE,
    YEAR_MAX,
    YEAR_MIN,
)
from chunk import first_pages
from llm import sonnet
from nul import get_contributors, get_year
from pages import Doc


# --- Title -------------------------------------------------------------------

# Longest title we pass through untouched. Beyond this the record is almost
# always a bound-volume aggregate (see _normalize_index_title).
TITLE_MAX_CHARS = 500
TITLE_MIN_CHARS = 5


def _normalize_index_title(raw: str) -> dict:
    """
    Normalize a catalogue title from the inventory index (MARC 245$ab).

    Measured over all 54,105 inventory rows: **zero** are empty or shorter than
    5 characters, and 155 (0.29%) exceed 500 characters. Length distribution is
    min 23 / median 91 / max 2809. So a missing title is not a real failure mode
    in this corpus; an over-long one is.

    The over-long records are bound-volume aggregates -- one catalogue record
    covering a stack of technical reports, semicolon-separated:

        "Alaska OCS (Outer Continental Shelf) socioeconomic studies program:
         Prudhoe Bay case study, technical report B1#4; Beaufort Sea region
         petroleum development scenarios, ... B1#6a; ... " (2809 chars)

    The informative title is right there at the front; what follows is a volume
    manifest. So we truncate at the first `;` and keep the full string under
    `full_title`, rather than discarding a correct title in order to re-derive
    one from OCR.

    Returns `{value, full_title, truncated, n_parts, reason}`.
    """
    s = " ".join((raw or "").split())
    if not s:
        return {"value": "", "full_title": "", "truncated": False, "reason": "absent"}

    if len(s) <= TITLE_MAX_CHARS:
        return {"value": s, "full_title": s, "truncated": False, "reason": "ok"}

    # Bound-volume aggregate: keep the head, count what we dropped.
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if len(parts) > 1 and len(parts[0]) <= TITLE_MAX_CHARS:
        return {
            "value": parts[0],
            "full_title": s,
            "truncated": True,
            "n_parts": len(parts),
            "reason": "bound_volume_aggregate",
        }

    # Single very long title: cut on a word boundary so we never emit a
    # half-word, which would look like OCR damage to a downstream reader.
    cut = s[:TITLE_MAX_CHARS]
    if " " in cut:
        cut = cut[: cut.rindex(" ")]
    return {
        "value": cut.rstrip(" ,;:-") + "\u2026",
        "full_title": s,
        "truncated": True,
        "reason": "over_length",
    }


def extract_title(work: dict, doc: Doc) -> dict:
    """
    Title, taken from the inventory index. Deterministic -- no LLM call.

    The previous implementation fell back to an LLM read of the first OCR page
    whenever the index title was absent or outside the 5-500 char window. That
    fallback was removed:

      * It never executed. Titles resolve from the index on 9/9 graded docs and
        12/12 of the next batch, and no inventory row lacks a title at all.
      * Its only real trigger was an over-LONG title -- i.e. a case where the
        correct title was in hand and was being thrown away. Truncating is
        strictly better than re-deriving.
      * A guessed title is worse than no title for a research index. This field
        is what people search and cite on, and a plausible-but-wrong title is
        the hardest kind of error for a reader to notice.
      * Dead code whose first real execution would be at 2000-doc scale, on the
        hardest inputs (no metadata + 1970s OCR), is a liability rather than a
        safety net.

    `doc` is retained in the signature for call-site stability and is unused.

    A genuinely absent title returns an explicit status for the Critic to route
    to HUMAN_REVIEW, following the never-return-empty-silently rule that
    MCAL_PLAN 1(8) applies to `alternatives`.
    """
    nul_title = work.get("title") or work.get("nul_metadata", {}).get("title")
    if isinstance(nul_title, list):
        nul_title = nul_title[0] if nul_title else None

    norm = _normalize_index_title(nul_title or "")

    if not norm["value"] or len(norm["value"]) < TITLE_MIN_CHARS:
        return {
            "value": "",
            "confidence": "low",
            "sources": ["inventory index (no usable title)"],
            "status": "title_not_in_index",
            "note": (
                "No usable title in the inventory index for this accession. "
                "Not inferred from OCR by design; route to human review."
            ),
        }

    out = {
        "value": norm["value"],
        "confidence": "medium" if norm["truncated"] else "high",
        "sources": ["inventory index (MARC 245$ab)"],
    }
    if norm["truncated"]:
        out["full_title"] = norm["full_title"]
        out["truncated"] = True
        out["note"] = (
            f"Index title is {len(norm['full_title'])} chars"
            + (
                f" listing {norm['n_parts']} bound volumes"
                if norm.get("n_parts")
                else ""
            )
            + f"; truncated to {TITLE_MAX_CHARS} ({norm['reason']}). "
            "Full string preserved in `full_title`."
        )
    return out


# --- Year --------------------------------------------------------------------

YEAR_RE = re.compile(r"\b(19[6-9]\d|20[0-2]\d)\b")


def extract_year(work: dict, doc: Doc) -> dict:
    nul_year = get_year(work)
    head = first_pages(doc, FIRST_3_PAGES)
    regex_years = [
        int(y) for y in YEAR_RE.findall(head)
        if YEAR_MIN <= int(y) <= YEAR_MAX
    ]
    regex_year = _modal_year(regex_years)

    if nul_year and regex_year and nul_year == regex_year:
        return {"value": nul_year, "confidence": "high", "sources": ["NUL", "regex (first 3 pages)"]}
    if nul_year and YEAR_MIN <= nul_year <= YEAR_MAX:
        # NUL present; regex either missing or disagrees
        if regex_year and regex_year != nul_year:
            return {
                "value": nul_year,
                "confidence": "low",
                "sources": ["NUL", "regex (first 3 pages)"],
                "note": f"NUL={nul_year} disagrees with regex={regex_year} — flag",
            }
        return {"value": nul_year, "confidence": "high", "sources": ["NUL"]}
    if regex_year:
        return {"value": regex_year, "confidence": "medium", "sources": ["regex (first 3 pages)"]}
    return {"value": None, "confidence": "low", "sources": []}


def _modal_year(years: list[int]) -> Optional[int]:
    if not years:
        return None
    from collections import Counter
    return Counter(years).most_common(1)[0][0]


# --- EIS Type ----------------------------------------------------------------

def eis_type_regex_hits(head: str) -> list[dict]:
    """
    Every EIS-type pattern that matches `head`, EARLIEST MATCH FIRST.

    Replaces a `break` on the first pattern that matched, which made
    `config.EIS_TYPE_PATTERNS`' dict order the decider rather than where the
    phrase appears. The order is `Supplemental, ROD, Draft, Final`, so a Final EIS
    whose cover page said "record of decision" anywhere was labelled ROD, and one
    that mentioned a "supplement" was labelled Supplemental.

    Measured on the seven realistic cover-page phrasings in
    `tests/test_eis_type.py`, dict order gets 3 wrong and position gets 0. On the
    21 materialized documents the two disagree on `p1074_35556036535615`, where
    "Final" appears at offset 163 and "Draft" at 520: dict order returns Draft.

    The specificity the config comment was protecting survives, because the
    patterns are phrase-anchored rather than word-anchored. In "FINAL SUPPLEMENTAL
    ENVIRONMENTAL IMPACT STATEMENT" the Final pattern requires "final" followed by
    "environmental", which "SUPPLEMENTAL" interrupts, so Final does not match at
    all and Supplemental wins on position as well as on order.
    """
    hits: list[dict] = []
    for label, pat in EIS_TYPE_PATTERNS.items():
        m = re.search(pat, head, re.IGNORECASE)
        if m:
            hits.append({"label": label, "offset": m.start(), "matched": m.group(0)})
    hits.sort(key=lambda h: h["offset"])
    return hits


def extract_eis_type(doc: Doc) -> dict:
    """
    EIS type from a position-scored cover-page regex, verified by Sonnet.

    On disagreement the field is flagged `review_required` instead of silently
    preferring the regex. The regex was the losing source in the one observed
    failure (Lincoln Hwy, graded `wrong: "ROD", correct: "Final"`), so defaulting
    to it is defaulting to the source that was wrong.

    The Sonnet verifier is NOT skipped when the regex is unambiguous, though the
    review suggested it. Two measured reasons: 6 of the 21 materialized documents
    produce NO regex hit at all, so Sonnet is the only source on 29% of them; and
    position scoring is still fallible on phrasings like "comments on the draft EIS
    are addressed in this final EIS", where the earliest match is the wrong one. At
    ~1.3k input tokens and 300 output on Sonnet the call is ~$0.008/doc -- about
    $17 across 2,000 documents, or 0.13% of the projected corpus cost. That is not
    a lever worth pulling at the price of the only cross-check this field has.
    """
    head = first_pages(doc, FIRST_PAGE)
    hits = eis_type_regex_hits(head)
    regex_label = hits[0]["label"] if hits else None

    verifier_text = first_pages(doc, FIRST_2_PAGES)
    out = sonnet(
        system=(
            "You determine the type of an Environmental Impact Statement. "
            "Choose ONE from: Draft, Final, Supplemental, ROD. "
            "If none of these clearly applies, return \"Unknown\". "
            "Respond ONLY with JSON: {\"eis_type\": \"<one of Draft|Final|Supplemental|ROD|Unknown>\", "
            "\"evidence\": \"<short phrase from the text supporting your choice>\"}."
        ),
        user=f"OCR (first 2 pages):\n{verifier_text}",
        max_tokens=300,
    )
    sonnet_label = (out.get("eis_type") or "").strip()

    # Kept on every return so a reviewer can see what the page actually said,
    # including the runner-up labels the position rule discarded.
    regex_audit = {
        "regex_hits": hits,
        "regex_label": regex_label,
        "selection_rule": "earliest match in first page",
    }

    if regex_label and sonnet_label and regex_label == sonnet_label:
        return {
            "value": regex_label,
            "confidence": "high",
            "sources": ["regex (first page)", "Sonnet (first 2 pages)"],
            "evidence": out.get("evidence", ""),
            **regex_audit,
        }
    if regex_label and not sonnet_label:
        return {
            "value": regex_label,
            "confidence": "medium",
            "sources": ["regex (first page)"],
            **regex_audit,
        }
    if sonnet_label and not regex_label:
        return {
            "value": sonnet_label,
            "confidence": "medium",
            "sources": ["Sonnet (first 2 pages)"],
            "evidence": out.get("evidence", ""),
            **regex_audit,
        }
    if regex_label and sonnet_label and regex_label != sonnet_label:
        # Both sources are carried and NEITHER is preferred. The value has to be
        # populated -- MCAL_PLAN 7 Q8 requires the manifest to carry the raw
        # extraction so the reviewer can grade it -- but `review_required` makes
        # the gate route it to a human rather than letting the regex win by
        # default. In the one observed failure the regex was the wrong source.
        return {
            "value": regex_label,
            "confidence": "low",
            "sources": ["regex (first page)", "Sonnet (first 2 pages)"],
            "note": f"regex={regex_label} disagrees with Sonnet={sonnet_label}",
            "evidence": out.get("evidence", ""),
            "review_required": True,
            "candidates": [
                {"value": regex_label, "source": "regex (first page)"},
                {"value": sonnet_label, "source": "Sonnet (first 2 pages)"},
            ],
            **regex_audit,
        }
    return {"value": "Unknown", "confidence": "low", "sources": [], **regex_audit}


# --- Lead Agency / Contributor ----------------------------------------------

def extract_lead_agency(work: dict, doc: Doc) -> dict:
    nul_contribs = get_contributors(work)
    # Strip role parens
    nul_contribs = [re.sub(r"\s*\([^)]+\)\s*$", "", c).strip() for c in nul_contribs]
    nul_contribs = [c for c in nul_contribs if c]

    if nul_contribs:
        result = {
            "value": nul_contribs,
            "confidence": "high",
            "sources": ["NUL"],
        }
        if len(nul_contribs) > 2:
            result["note"] = "More than 2 agencies — likely joint-lead; verify."
        return result

    # Fallback: Sonnet on first 4 pages
    head = first_pages(doc, FIRST_4_PAGES)
    out = sonnet(
        system=(
            "You extract the lead and cooperating federal agencies listed on the "
            "cover/title pages of an Environmental Impact Statement. "
            "Respond ONLY with JSON: {\"agencies\": [\"<agency name>\", ...]}. "
            "Return an empty list if you cannot find named agencies."
        ),
        user=f"OCR (first 4 pages):\n{head}",
        max_tokens=400,
    )
    agencies = out.get("agencies") or []
    if not isinstance(agencies, list):
        agencies = []
    agencies = [a for a in (str(x).strip() for x in agencies) if a]
    return {
        "value": agencies,
        "confidence": "medium" if agencies else "low",
        "sources": ["Sonnet (first 4 pages)"],
        "note": "More than 2 agencies — likely joint-lead; verify." if len(agencies) > 2 else None,
    }


# --- Top-level ---------------------------------------------------------------

def run_m1(work: dict, doc: Doc) -> dict:
    return {
        "title": extract_title(work, doc),
        "year": extract_year(work, doc),
        "eis_type": extract_eis_type(doc),
        "lead_agency": extract_lead_agency(work, doc),
    }
