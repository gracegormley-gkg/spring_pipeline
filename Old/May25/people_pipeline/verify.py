"""
Verbatim quote checking for extracted entity rows.

Uses `Doc.find_quote` (whitespace-normalized, per-page search). When a quote is
found verbatim we override `evidence_pages` with the single verified page so
downstream rows cite exact pages rather than the chunk's wide page span. A row
whose quote is not found verbatim keeps the LLM-reported pages but
`quote_verified` is False, which forces HUMAN_REVIEW in the critic step.
"""

from __future__ import annotations

from typing import Iterable

from pages import Doc


def verify_rows(rows: Iterable[dict], doc: Doc) -> list[dict]:
    """Annotate each row with `quote_verified` (bool). Does not drop rows.

    For verified rows, replace `evidence_pages` with the single verified page.
    Original LLM-reported pages are preserved under `evidence_pages_reported`.
    """
    out: list[dict] = []
    for row in rows:
        quote = (row.get("quote") or "").strip()
        verified_page: int | None = None
        if quote:
            hit = doc.find_quote(quote)
            if hit is not None:
                verified_page = hit[0]
        new_row = dict(row)
        if verified_page is not None:
            new_row["quote_verified"] = True
            new_row["evidence_pages_reported"] = row.get("evidence_pages") or []
            new_row["evidence_pages"] = [str(verified_page)]
        else:
            new_row["quote_verified"] = False
            new_row["verify_note"] = "Quote not found verbatim in document — HUMAN_REVIEW will be forced."
        out.append(new_row)
    return out
