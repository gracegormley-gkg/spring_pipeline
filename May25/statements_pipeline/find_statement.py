"""
Per-row statement-finding for the statements pipeline.

For each merged (entity, stance) row we ask Sonnet to:
  - classify whether a contiguous statement by the entity exists in the doc
    (letter / testimony / written_comment / narrator_paraphrase / sectional / none)
  - if yes, return verbatim opening + closing anchors that bound the statement
  - always return a 2-3 sentence summary of the entity's opinion

Anchors are verified against the doc text with whitespace-tolerant matching.
When both verify, the full statement text is sliced from the doc between them.
When they don't, statement.text is null but summary is always present.

A note on windowing: we don't send the whole doc — we send a page-window around
the entity's evidence pages (WINDOW_MARGIN_PAGES on each side, capped at
WINDOW_CHAR_CAP chars). This both keeps the call cheap and keeps anchor search
local, so a generic anchor like "Sincerely yours," doesn't accidentally match a
different letter on the other side of the doc.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import settings

from config import MODEL_SONNET             # from segment_a/
from llm import call_json_with_usage        # from segment_a/
from pages import Doc                        # from segment_a/

log = logging.getLogger(__name__)


_SYSTEM = (
    "You analyze ONE stance-bearing entity in an Environmental Impact Statement "
    "and locate their full statement (if one exists) within a window of doc text.\n\n"
    "INPUTS: the entity's name/role/kind/stance, an exemplar quote, and a window "
    "of doc text around the pages where the entity was found.\n\n"
    "OUTPUT a single JSON object:\n"
    "{\n"
    '  "statement_form": "letter|testimony|written_comment|narrator_paraphrase|sectional|none",\n'
    '  "statement_present": true|false,\n'
    '  "opening_anchor": "<verbatim first ~50-150 chars of the statement, or empty>",\n'
    '  "closing_anchor": "<verbatim last ~50-150 chars of the statement, or empty>",\n'
    '  "summary": "<2-3 sentence summary of the entity\'s opinion in your own words>"\n'
    "}\n\n"
    "FORM DEFINITIONS:\n"
    "- letter: a contiguous comment letter signed by or attributed to the entity. "
    "Usually opens with a salutation (e.g. 'Dear ...') and closes with a signature.\n"
    "- testimony: a contiguous spoken statement, typically introduced by a speaker "
    "tag (e.g. 'MR. SMITH:', 'CHAIRWOMAN JONES:') and ending where the next speaker "
    "begins.\n"
    "- written_comment: a contiguous written statement that isn't a formal letter "
    "(memo, position paper, attachment).\n"
    "- narrator_paraphrase: the doc narrator describes the entity's position without "
    "a contiguous statement block. Use this for one-off mentions like 'The Sierra "
    "Club argued that ORV use should be halted.'\n"
    "- sectional: the entity appears in a list/table grouped under a stance heading. "
    "No individual statement; only the group label.\n"
    "- none: no statement and no clear paraphrase block.\n\n"
    "ANCHOR RULES:\n"
    "- Anchors MUST be copied verbatim from the window. Exact wording, exact "
    "punctuation. Do not paraphrase, summarize, or join across newlines.\n"
    "- opening_anchor: the first ~50-150 chars of the statement (e.g. salutation + "
    "first phrase, or speaker tag + first sentence).\n"
    "- closing_anchor: the last ~50-150 chars (e.g. closing phrase + signature, or "
    "the final sentence before the next speaker).\n"
    "- Pick anchors that are DISTINCTIVE within the window — avoid generic strings "
    "like 'Sincerely,' or 'Dear Sir,' alone; include enough surrounding words to "
    "uniquely identify the position.\n"
    "- If statement_form is narrator_paraphrase / sectional / none, set both anchors "
    "to '' and statement_present to false.\n\n"
    "SUMMARY RULES:\n"
    "- 2-3 sentences in your own words describing what the entity thinks about the "
    "project (or the specific aspect they take a position on).\n"
    "- Always include a summary, even when no contiguous statement is present — "
    "summarize from the available evidence.\n"
    "- Do not invent positions that aren't supported by the text."
)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _find_anchor(window: str, anchor: str) -> Optional[tuple[int, int]]:
    """Whitespace-tolerant substring search. Returns (start, end) in window, or None.

    Builds a regex from the anchor's normalized tokens, separated by \\s+. This
    lets us match anchors that the model copied with collapsed whitespace
    against the doc text, which often has line breaks mid-sentence from OCR.
    """
    norm = _normalize_ws(anchor)
    if not norm:
        return None
    tokens = norm.split(" ")
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    m = re.search(pattern, window)
    if m is None:
        return None
    return m.start(), m.end()


def _parse_evidence_pages(evidence_pages: list) -> list[int]:
    nums: list[int] = []
    for span in evidence_pages or []:
        if not isinstance(span, str):
            continue
        try:
            if "-" in span:
                a, b = span.split("-", 1)
                nums.append(int(a.strip()))
                nums.append(int(b.strip()))
            else:
                nums.append(int(span.strip()))
        except ValueError:
            continue
    return nums


def _build_window(doc: Doc, evidence_pages: list) -> tuple[str, int, int]:
    """Build a doc-text window around the entity's evidence pages.

    Returns (window_text, start_page, end_page). If evidence pages can't be
    parsed, falls back to a head-of-doc window.
    """
    if not doc.pages:
        return "", 1, 1
    nums = _parse_evidence_pages(evidence_pages)
    if not nums:
        text = doc.full_text[:settings.WINDOW_CHAR_CAP]
        return text, doc.pages[0].page_num, doc.pages[-1].page_num
    first_doc_page = doc.pages[0].page_num
    last_doc_page = doc.pages[-1].page_num
    start_page = max(min(nums) - settings.WINDOW_MARGIN_PAGES, first_doc_page)
    end_page = min(max(nums) + settings.WINDOW_MARGIN_PAGES, last_doc_page)
    text = doc.text_for_pages(start_page, end_page)
    if len(text) > settings.WINDOW_CHAR_CAP:
        text = text[:settings.WINDOW_CHAR_CAP]
    return text, start_page, end_page


def _slice_statement(window: str, opening: str, closing: str) -> tuple[Optional[str], bool, bool]:
    """Slice window between opening and closing anchors.

    Returns (text, opening_ok, closing_ok). text is None if either anchor doesn't
    verify or if closing precedes opening.
    """
    open_loc = _find_anchor(window, opening) if opening else None
    close_loc = _find_anchor(window, closing) if closing else None
    opening_ok = open_loc is not None
    closing_ok = close_loc is not None
    if not (opening_ok and closing_ok):
        return None, opening_ok, closing_ok
    start = open_loc[0]
    end = close_loc[1]
    if end <= start:
        # Closing precedes opening — likely matched the wrong occurrence.
        return None, True, True
    text = window[start:end]
    if len(text) > settings.STATEMENT_CHAR_CAP:
        text = text[:settings.STATEMENT_CHAR_CAP]
    return text, True, True


def _ask_model(row: dict, window: str, start_page: int, end_page: int) -> tuple[dict, Optional[dict]]:
    payload = {
        "entity": row.get("entity"),
        "kind": row.get("kind"),
        "role": row.get("role"),
        "stance": row.get("stance"),
        "summary_quote": row.get("summary_quote"),
        "evidence_pages": row.get("evidence_pages"),
    }
    user = (
        f"ENTITY:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"DOC WINDOW (pages {start_page}-{end_page}):\n"
        f"--- BEGIN ---\n{window}\n--- END ---"
    )
    try:
        out, usage = call_json_with_usage(
            MODEL_SONNET, _SYSTEM, user, max_tokens=1500,
        )
        return out, usage
    except Exception as e:
        log.warning(
            f"find_statement call failed for entity {row.get('entity')!r}: {e}"
        )
        return {
            "statement_form": "none",
            "statement_present": False,
            "opening_anchor": "",
            "closing_anchor": "",
            "summary": "",
            "_error": str(e),
        }, None


def find_statement_for_row(row: dict, doc: Doc) -> dict:
    """Per-row statement finder.

    Returns the row with `statement` (dict) and `summary` (str) fields added,
    plus `_statement_usage` for cost aggregation (stripped before final write).
    """
    window, start_page, end_page = _build_window(doc, row.get("evidence_pages") or [])
    raw, usage = _ask_model(row, window, start_page, end_page)

    if not isinstance(raw, dict):
        raw = {}
    form = raw.get("statement_form")
    if form not in settings.STATEMENT_FORMS:
        form = "none"
    summary = (raw.get("summary") or "").strip()
    opening = (raw.get("opening_anchor") or "").strip()
    closing = (raw.get("closing_anchor") or "").strip()
    present_claim = bool(raw.get("statement_present"))

    text: Optional[str] = None
    opening_verified = False
    closing_verified = False
    if present_claim and opening and closing:
        text, opening_verified, closing_verified = _slice_statement(window, opening, closing)

    statement = {
        "present": text is not None,
        "form": form,
        "text": text,
        "opening_anchor": opening,
        "closing_anchor": closing,
        "opening_anchor_verified": opening_verified,
        "closing_anchor_verified": closing_verified,
        "window_pages": [start_page, end_page],
    }
    if raw.get("_error"):
        statement["error"] = raw["_error"]

    out = dict(row)
    out["statement"] = statement
    out["summary"] = summary
    if usage is not None:
        out["_statement_usage"] = usage
    return out


def find_statements_for_doc(
    rows: list[dict],
    doc: Doc,
    parallel: int = settings.STATEMENT_PARALLEL,
) -> list[dict]:
    """Run find_statement_for_row over all rows in parallel; preserves sequence order."""
    if not rows:
        return []
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(find_statement_for_row, r, doc): r for r in rows}
        for fut in as_completed(futures):
            try:
                out.append(fut.result())
            except Exception as e:
                r = futures[fut]
                log.exception(f"find_statement crashed for entity {r.get('entity')!r}: {e}")
                fallback = dict(r)
                fallback["statement"] = {
                    "present": False,
                    "form": "none",
                    "text": None,
                    "opening_anchor": "",
                    "closing_anchor": "",
                    "opening_anchor_verified": False,
                    "closing_anchor_verified": False,
                    "error": str(e),
                }
                fallback["summary"] = ""
                out.append(fallback)
    out.sort(key=lambda r: r.get("sequence", 10**9))
    return out
