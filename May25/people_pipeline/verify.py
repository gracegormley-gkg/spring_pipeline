"""
Verbatim quote checking for extracted entity rows.

Same approach as segment_a/m2.py: normalize whitespace, then check whether the
quote appears anywhere in the (full) doc text. We can't pin to an exact page
because page numbers are estimated from char offsets — but we can refuse to
mark a quote as verified unless the literal string is present somewhere.

A row whose quote is not found verbatim keeps the quote (so the human grader
can see it) but `quote_verified` is False, which forces HUMAN_REVIEW in the
critic step.
"""

from __future__ import annotations

import re
from typing import Iterable


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def verify_in_text(quote: str, full_text: str, _normalized_text_cache: dict | None = None) -> bool:
    """True iff `quote` appears verbatim in `full_text` (whitespace-normalized)."""
    if not quote:
        return False
    nq = _normalize(quote)
    if not nq:
        return False
    # Cache the normalized full-text since this is called many times per doc.
    if _normalized_text_cache is not None:
        nt = _normalized_text_cache.get("text")
        if nt is None:
            nt = _normalize(full_text)
            _normalized_text_cache["text"] = nt
    else:
        nt = _normalize(full_text)
    return nq in nt


def verify_rows(rows: Iterable[dict], full_text: str) -> list[dict]:
    """Annotate each row with `quote_verified` (bool). Does not drop rows."""
    cache: dict = {}
    out: list[dict] = []
    for row in rows:
        verified = verify_in_text(row.get("quote", ""), full_text, _normalized_text_cache=cache)
        row = dict(row)
        row["quote_verified"] = verified
        if not verified:
            row["verify_note"] = "Quote not found verbatim in document — HUMAN_REVIEW will be forced."
        out.append(row)
    return out
