"""
Shared evidence helpers.

The per-field rework (M1, M2) attaches a list of evidence objects to every
extracted value:

    {
      "quote":          "<verbatim text from the doc>",
      "source_pages":   ["<page_num>"],   # exact page(s) where the quote was found
      "quote_verified": true | false,     # false → forces HUMAN_REVIEW
      "note":           "<short>",        # only set when verified is false
    }

`verify_and_locate(quote, doc)` is the single point of truth for verification:
it whitespace-normalizes, searches the doc page-by-page, and returns the page
number where the quote actually lives. Callers should NOT do their own
substring search; that's how we previously ended up with inconsistent
behavior across fields.

A failed verification is NEVER silently dropped — it returns an Evidence
with `quote_verified=False` and an empty `source_pages`, which downstream
rubrics convert into `HUMAN_REVIEW`.

`evidence_source` marks provenance. Two fields — `summary.overview` and
`themes` — are roll-ups: they may present quotes that were verified against a
DIFFERENT field's claim. Such entries are tagged
`evidence_source="carried_from_subfields"` plus `carried_from="<field>"`, so
that a reviewer can see the quote is second-hand and, more importantly, so the
confidence machinery can exclude them from `s_quote`. Untagged evidence is
first-party by default; nothing has to be rewritten for the tag to mean what it
says.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from pages import Doc

# `evidence_source` values.
EVIDENCE_OWN = "own"
EVIDENCE_CARRIED = "carried_from_subfields"


class Evidence(TypedDict, total=False):
    quote: str
    source_pages: list[str]
    quote_verified: bool
    note: str
    evidence_source: str
    carried_from: str


def is_carried(ev: dict) -> bool:
    """True if this evidence entry was borrowed from another field."""
    return (ev or {}).get("evidence_source") == EVIDENCE_CARRIED


def own_evidence(evidence_list) -> list[dict]:
    """
    The subset of an evidence list that belongs to the field itself.

    Absence of `evidence_source` means first-party: every artifact written before
    the tag existed is therefore treated exactly as it was, so this cannot
    retroactively change a score that has already been calibrated on.
    """
    return [ev for ev in (evidence_list or []) if not is_carried(ev)]


def carried_evidence(evidence_list) -> list[dict]:
    return [ev for ev in (evidence_list or []) if is_carried(ev)]


def mark_carried(ev: dict, source_field: str) -> Evidence:
    """A copy of `ev` tagged as carried from `source_field`."""
    out = dict(ev or {})
    out["evidence_source"] = EVIDENCE_CARRIED
    out["carried_from"] = source_field
    return out  # type: ignore[return-value]


def verify_and_locate(quote: str, doc: Doc) -> Evidence:
    """Verify a quote verbatim against `doc` and return an Evidence dict."""
    q = (quote or "").strip()
    if not q:
        return {
            "quote": "",
            "source_pages": [],
            "quote_verified": False,
            "note": "Empty quote.",
        }
    hit = doc.find_quote(q)
    if hit is None:
        return {
            "quote": q,
            "source_pages": [],
            "quote_verified": False,
            "note": "Quote not found verbatim in doc — forced HUMAN_REVIEW.",
        }
    page_num, _offset = hit
    return {
        "quote": q,
        "source_pages": [str(page_num)],
        "quote_verified": True,
    }


def evidence_for_quotes(quotes: list[str], doc: Doc) -> list[Evidence]:
    """Verify a list of quotes; keep the order, mark each verified/unverified."""
    out: list[Evidence] = []
    for q in quotes or []:
        out.append(verify_and_locate(q, doc))
    return out


def union_pages(evidences: list[Evidence]) -> list[str]:
    """Dedupe page citations across a list of evidences (preserves first-seen order)."""
    seen: dict[str, None] = {}
    for ev in evidences or []:
        for p in ev.get("source_pages", []) or []:
            seen.setdefault(p, None)
    return list(seen.keys())


def any_unverified(evidences: list[Evidence]) -> bool:
    return any(not ev.get("quote_verified") for ev in evidences or [])
