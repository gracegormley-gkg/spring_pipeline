"""
Per-row statement-finding AND stance evaluation.

After the split: extract is discovery-only and merge collapses by entity.
This stage is where stance is decided. For each merged row we ask Sonnet to:
  - locate the entity's contiguous statement in a window of doc text (if any)
    and classify its form
  - return verbatim opening + closing anchors that bound the statement
  - judge the entity's STANCE off whatever evidence is available (full
    statement, paraphrase block, sectional heading)
  - assign a stance_confidence label so downstream review can prioritize the
    risky calls
  - always return a 2-3 sentence summary of the entity's opinion

Anchors are verified against the doc text with whitespace-tolerant matching,
biased toward the entity's evidence pages so generic anchors lock onto the
right occurrence. When both anchors verify, the full statement text is sliced
from the doc between them. When they don't, statement.text is null but
summary and stance are always present.

Confidence policy (capped after the model returns):
  - statement_form == "none"                      → forced "low"
  - narrator_paraphrase / sectional / no anchors  → capped at "medium"
  - else                                          → as model said (high|medium|low)

A note on windowing: we don't send the whole doc — we send a page-window around
the entity's evidence pages (WINDOW_MARGIN_PAGES on each side, capped at
WINDOW_CHAR_CAP chars). This keeps the call cheap and keeps anchor search
local.
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
    "You analyze ONE candidate stakeholder in an Environmental Impact Statement, "
    "locate their full statement in a window of doc text (if one exists), AND "
    "judge their stance on the project from the available evidence. You also "
    "look for a RESPONSE to their concern (e.g. an agency reply, preparer note, "
    "or rebuttal block) elsewhere in the same window.\n\n"
    "INPUTS: the entity's name/role/kind, an exemplar mention quote (which may "
    "be a direct quote, a narrator paraphrase, or a sectional heading), and a "
    "window of doc text around the pages where the entity was found.\n\n"
    "OUTPUT a single JSON object:\n"
    "{\n"
    '  "statement_form": "letter|testimony|written_comment|narrator_paraphrase|sectional|none",\n'
    '  "statement_present": true|false,\n'
    '  "opening_anchor": "<verbatim first ~50-150 chars of the statement, or empty>",\n'
    '  "closing_anchor": "<verbatim last ~50-150 chars of the statement, or empty>",\n'
    '  "stance": "in_favor|opposed|conditional|neutral",\n'
    '  "stance_confidence": "high|medium|low",\n'
    '  "stance_basis": "<one short phrase: what evidence justifies this stance>",\n'
    '  "summary": "<2-3 sentence summary of the entity\'s opinion in your own words>",\n'
    '  "response_present": true|false,\n'
    '  "response_form": "agency_response|preparer_reply|discussion|none",\n'
    '  "response_opening_anchor": "<verbatim first ~50-150 chars of the response, or empty>",\n'
    '  "response_closing_anchor": "<verbatim last ~50-150 chars of the response, or empty>",\n'
    '  "response_summary": "<1-2 sentence summary of the response, or empty if no response>"\n'
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
    "ANCHOR RULES (for both statement and response):\n"
    "- Anchors MUST be copied verbatim from the window. Exact wording, exact "
    "punctuation. Do not paraphrase, summarize, or join across newlines.\n"
    "- opening_anchor: the first ~50-150 chars (e.g. salutation + first phrase, "
    "speaker tag + first sentence, or 'Response:' label + first sentence).\n"
    "- closing_anchor: the last ~50-150 chars (e.g. closing phrase + signature, "
    "the final sentence before the next speaker, or the last sentence of the "
    "response block).\n"
    "- Pick anchors that are DISTINCTIVE within the window — avoid generic strings "
    "like 'Sincerely,' or 'We agree.' alone; include enough surrounding words to "
    "uniquely identify the location.\n"
    "- If a section is absent, set both of its anchors to '' and its _present "
    "flag to false.\n\n"
    "STANCE RULES:\n"
    "- stance MUST be one of:\n"
    "  - in_favor: supports the proposal or an aspect of it.\n"
    "  - opposed: objects to the proposal or an aspect of it.\n"
    "  - conditional: supports only if specific conditions are met (mitigation, "
    "alternative selection, scope changes).\n"
    "  - neutral: has an attributed view but neither supports nor opposes "
    "(e.g. raises concerns without taking a side, asks procedural questions).\n"
    "- stance_confidence MUST be one of:\n"
    "  - high: a contiguous statement (letter / testimony / written_comment) that "
    "states the position clearly. The text spells it out.\n"
    "  - medium: stance is reasonably inferable from a paraphrase or a sectional "
    "heading. Real evidence, but no direct statement to verify against.\n"
    "  - low: stance is guessed from sparse / ambiguous evidence (e.g. one short "
    "narrator mention, conflicting signals across the window, or you couldn't find "
    "the statement and the only evidence is the exemplar quote).\n"
    "- stance_basis: one short phrase pointing at the evidence (e.g. 'opens letter "
    "calling project unacceptable', 'listed under PRO REGULATIONS heading').\n\n"
    "RESPONSE RULES:\n"
    "- A response is the agency's, preparer's, or document author's REPLY to the "
    "entity's concern — typically appears AFTER the entity's letter/paraphrase, "
    "often labeled 'Response:', 'Reply:', 'Discussion:', or formatted as a "
    "follow-up paragraph addressing the comment.\n"
    "- response_form options:\n"
    "  - agency_response: an explicit response from the lead/cooperating agency "
    "(usually labeled, often paired one-to-one with comments in a "
    "Comments-and-Responses section).\n"
    "  - preparer_reply: a less formal reply from the doc's preparers, embedded "
    "in narrative or a discussion block.\n"
    "  - discussion: a discussion paragraph addressing the comment without an "
    "explicit response label.\n"
    "  - none: no response is present in the window.\n"
    "- Set response_present=false and response_form='none' when no response is "
    "in the window. Do NOT invent a response.\n"
    "- If the response references the entity's concern in a clearly directed "
    "way (e.g. naming them, quoting them, or appearing immediately after their "
    "comment block), include it. If it's a generic discussion that doesn't "
    "address this entity's concern specifically, set response_present=false.\n\n"
    "SUMMARY RULES:\n"
    "- summary: 2-3 sentences in your own words describing what the entity thinks "
    "about the project. Always include, even when no contiguous statement is "
    "present — summarize from the available evidence.\n"
    "- response_summary: 1-2 sentences summarizing the response's substance. "
    "Empty when response_present=false.\n"
    "- Do not invent positions or responses not supported by the text."
)


# stance_confidence is ranked so we can compare and cap.
_CONF_RANK = {"low": 0, "medium": 1, "high": 2}
_CONF_VALUES = ("low", "medium", "high")


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _find_anchor(
    window: str,
    anchor: str,
    *,
    near_offset: Optional[int] = None,
    max_distance: Optional[int] = None,
) -> Optional[tuple[int, int]]:
    """Whitespace-tolerant substring search. Returns (start, end) in window, or None.

    Builds a regex from the anchor's normalized tokens, separated by \\s+. This
    lets us match anchors that the model copied with collapsed whitespace
    against the doc text, which often has line breaks mid-sentence from OCR.

    If `near_offset` and `max_distance` are given, picks the match closest to
    `near_offset` within `max_distance` chars (rather than the first match).
    """
    norm = _normalize_ws(anchor)
    if not norm:
        return None
    tokens = norm.split(" ")
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    if near_offset is None:
        m = re.search(pattern, window)
        if m is None:
            return None
        return m.start(), m.end()
    best: Optional[tuple[int, int]] = None
    best_dist = None
    for m in re.finditer(pattern, window):
        dist = abs(m.start() - near_offset)
        if max_distance is not None and dist > max_distance:
            continue
        if best_dist is None or dist < best_dist:
            best = (m.start(), m.end())
            best_dist = dist
    if best is None:
        # Fall back to the first match anywhere in the window if nothing was
        # within range — better to verify with a far match than to fail outright.
        m = re.search(pattern, window)
        if m is None:
            return None
        return m.start(), m.end()
    return best


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


def _last_page_within_chars(doc: Doc, start_page: int, char_cap: int) -> int:
    """Walk pages forward from start_page, returning the highest page_num whose
    cumulative text (joined with PAGE_SEP) still fits in char_cap. Used to make
    `window_pages` honest after we hard-cap the window text."""
    from pages import PAGE_SEP
    cum = 0
    last = start_page
    sep_len = len(PAGE_SEP)
    started = False
    for p in doc.pages:
        if p.page_num < start_page:
            continue
        add = len(p.text) + (sep_len if started else 0)
        if cum + add > char_cap and started:
            break
        cum += add
        last = p.page_num
        started = True
    return last


def _build_window(
    doc: Doc, evidence_pages: list
) -> tuple[str, int, int, int]:
    """Build a doc-text window around the entity's evidence pages.

    Returns (window_text, start_page, end_page, evidence_offset). evidence_offset
    is the char offset within window_text where the first evidence page starts —
    used to bias anchor matching toward the right occurrence. If evidence pages
    can't be parsed, falls back to a head-of-doc window with end_page set to
    the last page that actually fits in the window.
    """
    if not doc.pages:
        return "", 1, 1, 0
    nums = _parse_evidence_pages(evidence_pages)
    if not nums:
        log.debug(
            "No parseable evidence_pages (%r); falling back to head of doc.",
            evidence_pages,
        )
        text = doc.full_text[:settings.WINDOW_CHAR_CAP]
        first = doc.pages[0].page_num
        last = _last_page_within_chars(doc, first, len(text))
        return text, first, last, 0
    first_doc_page = doc.pages[0].page_num
    last_doc_page = doc.pages[-1].page_num
    start_page = max(min(nums) - settings.WINDOW_MARGIN_PAGES, first_doc_page)
    end_page = min(max(nums) + settings.WINDOW_MARGIN_PAGES, last_doc_page)
    text = doc.text_for_pages(start_page, end_page)
    if len(text) > settings.WINDOW_CHAR_CAP:
        text = text[:settings.WINDOW_CHAR_CAP]
        end_page = _last_page_within_chars(doc, start_page, len(text))
    # Char offset of the first evidence page within `text`.
    evidence_offset = 0
    target_page = min(nums)
    cursor = 0
    from pages import PAGE_SEP
    sep_len = len(PAGE_SEP)
    started = False
    for p in doc.pages:
        if p.page_num < start_page or p.page_num > end_page:
            continue
        if started:
            cursor += sep_len
        if p.page_num >= target_page:
            evidence_offset = cursor
            break
        cursor += len(p.text)
        started = True
    return text, start_page, end_page, evidence_offset


def _slice_statement(
    window: str,
    opening: str,
    closing: str,
    evidence_offset: int = 0,
) -> tuple[Optional[str], bool, bool]:
    """Slice window between opening and closing anchors.

    Returns (text, opening_ok, closing_ok). text is None if either anchor doesn't
    verify or if closing precedes opening.
    """
    open_loc = (
        _find_anchor(
            window,
            opening,
            near_offset=evidence_offset,
            max_distance=settings.ANCHOR_PROXIMITY_CHARS,
        )
        if opening
        else None
    )
    if open_loc is not None:
        # Prefer a closing anchor that occurs *after* the opening, near the
        # opening — closes are typically within a letter-length of the open.
        close_loc = (
            _find_anchor(
                window,
                closing,
                near_offset=open_loc[1],
                max_distance=settings.ANCHOR_PROXIMITY_CHARS,
            )
            if closing
            else None
        )
    else:
        close_loc = (
            _find_anchor(
                window,
                closing,
                near_offset=evidence_offset,
                max_distance=settings.ANCHOR_PROXIMITY_CHARS,
            )
            if closing
            else None
        )
    opening_ok = open_loc is not None
    closing_ok = close_loc is not None
    if not (opening_ok and closing_ok):
        return None, opening_ok, closing_ok
    start = open_loc[0]
    end = close_loc[1]
    if end <= start:
        return None, True, True
    text = window[start:end]
    if len(text) > settings.STATEMENT_CHAR_CAP:
        text = text[:settings.STATEMENT_CHAR_CAP]
    return text, True, True


def _slice_response(
    window: str,
    opening: str,
    closing: str,
    *,
    near_offset: int,
) -> tuple[Optional[str], bool, bool]:
    """Slice window between response opening and closing anchors.

    Same proximity-biased strategy as `_slice_statement` but `near_offset` is
    chosen by the caller — typically the END of the statement when we have
    one (responses follow comments), otherwise the entity's evidence pages.
    """
    open_loc = (
        _find_anchor(
            window,
            opening,
            near_offset=near_offset,
            max_distance=settings.ANCHOR_PROXIMITY_CHARS,
        )
        if opening
        else None
    )
    if open_loc is not None:
        close_loc = (
            _find_anchor(
                window,
                closing,
                near_offset=open_loc[1],
                max_distance=settings.ANCHOR_PROXIMITY_CHARS,
            )
            if closing
            else None
        )
    else:
        close_loc = (
            _find_anchor(
                window,
                closing,
                near_offset=near_offset,
                max_distance=settings.ANCHOR_PROXIMITY_CHARS,
            )
            if closing
            else None
        )
    opening_ok = open_loc is not None
    closing_ok = close_loc is not None
    if not (opening_ok and closing_ok):
        return None, opening_ok, closing_ok
    start = open_loc[0]
    end = close_loc[1]
    if end <= start:
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
        "exemplar_quote": row.get("summary_quote"),
        "exemplar_attribution_mode": row.get("attribution_mode"),
        "evidence_pages": row.get("evidence_pages"),
    }
    user = (
        f"ENTITY:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"DOC WINDOW (pages {start_page}-{end_page}):\n"
        f"--- BEGIN ---\n{window}\n--- END ---"
    )
    try:
        out, usage = call_json_with_usage(
            MODEL_SONNET, _SYSTEM, user, max_tokens=2000,
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
            "stance": "neutral",
            "stance_confidence": "low",
            "stance_basis": "find_statement call failed",
            "summary": "",
            "response_present": False,
            "response_form": "none",
            "response_opening_anchor": "",
            "response_closing_anchor": "",
            "response_summary": "",
            "_error": str(e),
        }, None


def _cap_confidence(
    raw_conf: str,
    *,
    form: str,
    anchors_verified: bool,
) -> str:
    """Apply policy caps so confidence reflects the structural evidence too,
    not just what the model claimed."""
    conf = raw_conf if raw_conf in _CONF_VALUES else "low"
    if form == "none":
        return "low"
    if form in ("narrator_paraphrase", "sectional") or not anchors_verified:
        # Cap at medium — there's no contiguous statement to verify against.
        if _CONF_RANK[conf] > _CONF_RANK["medium"]:
            conf = "medium"
    return conf


def find_statement_for_row(row: dict, doc: Doc) -> dict:
    """Per-row statement + stance + summary.

    Returns the row with `statement` (dict), `stance`, `stance_confidence`,
    `stance_basis`, `summary` fields added, plus `_statement_usage` for cost
    aggregation (stripped before the final write).
    """
    window, start_page, end_page, evidence_offset = _build_window(
        doc, row.get("evidence_pages") or []
    )
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
    statement_end_offset = evidence_offset  # used as the proximity anchor for response if we don't find a statement
    if present_claim and opening and closing:
        text, opening_verified, closing_verified = _slice_statement(
            window, opening, closing, evidence_offset=evidence_offset
        )
        # If the statement verified, prefer searching for the response right
        # after it — responses to comment letters appear inline below them.
        if text is not None:
            stmt_loc = _find_anchor(
                window, closing,
                near_offset=evidence_offset,
                max_distance=settings.ANCHOR_PROXIMITY_CHARS,
            )
            if stmt_loc is not None:
                statement_end_offset = stmt_loc[1]
    anchors_verified = opening_verified and closing_verified and text is not None

    # Stance: validate against the closed vocabulary; otherwise force low conf
    # and default to neutral (don't drop the row — let a human grade it).
    raw_stance = (raw.get("stance") or "").strip().lower()
    raw_conf = (raw.get("stance_confidence") or "").strip().lower()
    stance_basis = (raw.get("stance_basis") or "").strip()
    if raw_stance not in settings.STANCES:
        log.debug(
            "Invalid stance %r from model for entity %r; defaulting to neutral/low.",
            raw_stance, row.get("entity"),
        )
        stance = "neutral"
        stance_confidence = "low"
        if not stance_basis:
            stance_basis = "model returned no recognized stance"
    else:
        stance = raw_stance
        stance_confidence = _cap_confidence(
            raw_conf, form=form, anchors_verified=anchors_verified
        )

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

    # ---- Response (agency reply / preparer note / discussion block) ----
    response_form = raw.get("response_form")
    if response_form not in settings.RESPONSE_FORMS:
        response_form = "none"
    response_summary = (raw.get("response_summary") or "").strip()
    response_opening = (raw.get("response_opening_anchor") or "").strip()
    response_closing = (raw.get("response_closing_anchor") or "").strip()
    response_present_claim = bool(raw.get("response_present"))

    response_text: Optional[str] = None
    response_opening_verified = False
    response_closing_verified = False
    if (
        response_present_claim
        and response_form != "none"
        and response_opening
        and response_closing
    ):
        response_text, response_opening_verified, response_closing_verified = _slice_response(
            window,
            response_opening,
            response_closing,
            near_offset=statement_end_offset,
        )

    response = {
        "present": response_text is not None,
        "form": response_form,
        "text": response_text,
        "summary": response_summary,
        "opening_anchor": response_opening,
        "closing_anchor": response_closing,
        "opening_anchor_verified": response_opening_verified,
        "closing_anchor_verified": response_closing_verified,
    }

    out = dict(row)
    out["statement"] = statement
    out["stance"] = stance
    out["stance_confidence"] = stance_confidence
    out["stance_basis"] = stance_basis
    out["summary"] = summary
    out["response"] = response
    if usage is not None:
        out["_statement_usage"] = usage
    return out


def find_statements_for_doc(
    rows: list[dict],
    doc: Doc,
    parallel: Optional[int] = None,
) -> list[dict]:
    """Run find_statement_for_row over all rows in parallel; preserves sequence order."""
    if not rows:
        return []
    workers = parallel if parallel is not None else settings.STATEMENT_PARALLEL
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
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
                fallback["stance"] = "neutral"
                fallback["stance_confidence"] = "low"
                fallback["stance_basis"] = f"find_statement crashed: {e}"
                fallback["summary"] = ""
                fallback["response"] = {
                    "present": False,
                    "form": "none",
                    "text": None,
                    "summary": "",
                    "opening_anchor": "",
                    "closing_anchor": "",
                    "opening_anchor_verified": False,
                    "closing_anchor_verified": False,
                }
                out.append(fallback)
    out.sort(key=lambda r: r.get("sequence", 10**9))
    return out
