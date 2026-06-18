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
"""

from __future__ import annotations

from typing import Optional, TypedDict

from pages import Doc


class Evidence(TypedDict, total=False):
    quote: str
    source_pages: list[str]
    quote_verified: bool
    note: str


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
