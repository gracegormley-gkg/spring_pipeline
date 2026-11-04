"""
Per-row statement-finding, stance evaluation, and response capture.

After the architecture refurbish:
- One LLM call per merged ENTITY.
- The model returns a `complaints` array — one entry per distinct mention of
  this entity in the doc-text window. A single Sierra Club row may produce
  TWO complaints when the doc has both a paraphrase summary on page 34
  ("Sierra Club expressed concern that...") AND the full letter on page 142.
- Each complaint carries its own form, anchors, evidence pages, and an
  optional `response` (agency reply nearby in the doc).
- Stance / stance_confidence / stance_basis / summary stay at the ENTITY
  level — unified across complaints. An entity has one stance toward the
  project, even if the doc paraphrases their letter once and reproduces it
  in full elsewhere.

The writer fans out the complaints array into per-complaint and per-response
JSON files; this module just produces the structured row.

Anchors (statement and response) are matched whitespace-tolerantly and
biased toward the entity's evidence pages so generic anchors lock onto the
right occurrence within the window.

Confidence policy (capped after the model returns):
  - all complaints have form == "none" (or no complaints)  → forced "low"
  - best complaint form is narrator_paraphrase / sectional / no anchors verified
                                                             → capped at "medium"
  - else (at least one verified contiguous statement)        → as model said
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
    "You analyze ONE candidate stakeholder in an Environmental Impact Statement "
    "and locate ALL of their distinct complaint instances within a window of "
    "doc text. For each complaint, you also look for a nearby agency / preparer "
    "RESPONSE (e.g. a 'Response:' block, an inline rebuttal, or a discussion "
    "paragraph addressing the comment). Finally you judge the entity's overall "
    "stance toward the project.\n\n"
    "INPUTS: the entity's name/role/kind, an exemplar mention quote, and a "
    "window of doc text around the pages where the entity was found.\n\n"
    "OUTPUT a single JSON object:\n"
    "{\n"
    '  "stance": "in_favor|opposed|conditional|neutral",\n'
    '  "stance_confidence": "high|medium|low",\n'
    '  "stance_basis": "<one short phrase: what evidence justifies this stance>",\n'
    '  "summary": "<2-3 sentence ENTITY-level summary covering all mentions>",\n'
    '  "complaints": [\n'
    "    {\n"
    '      "form": "letter|testimony|written_comment|narrator_paraphrase|sectional|none",\n'
    '      "opening_anchor": "<verbatim first ~50-150 chars of this complaint, or empty>",\n'
    '      "closing_anchor": "<verbatim last ~50-150 chars of this complaint, or empty>",\n'
    '      "evidence_pages": ["<p>"|"<p>-<p>"],\n'
    '      "complaint_summary": "<1 sentence describing what THIS specific mention says>",\n'
    '      "response": {\n'
    '        "present": true|false,\n'
    '        "form": "agency_response|preparer_reply|discussion|none",\n'
    '        "opening_anchor": "<verbatim first ~50-150 chars of the response, or empty>",\n'
    '        "closing_anchor": "<verbatim last ~50-150 chars of the response, or empty>",\n'
    '        "agency": "<name of the responding agency, or empty>",\n'
    '        "agency_kind": "agency|government|other",\n'
    '        "summary": "<1-2 sentence summary of the response, or empty>"\n'
    "      }\n"
    "    }\n"
    "  ]\n"
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
    "a contiguous statement block. Use this for comment-summary lines like 'The "
    "Sierra Club argued that ORV use should be halted.'\n"
    "- sectional: the entity appears in a list/table grouped under a stance heading. "
    "No individual statement; only the group label.\n"
    "- none: a mention exists but it does not state a position (rare — usually "
    "you'd just omit such a complaint).\n\n"
    "MULTI-COMPLAINT RULES:\n"
    "- Return EVERY distinct mention of this entity in the window where they take "
    "or are described as taking a position. The same Sierra Club may appear as "
    "BOTH a paraphrase in a Comments-and-Responses section AND a full letter in "
    "an appendix — return TWO complaints in that case.\n"
    "- Do NOT duplicate the same physical mention as two complaints.\n"
    "- Pure rosters (alphabetical lists of recipients with no stance heading) are "
    "out of scope; do not list those.\n"
    "- If the entity is in the window but never expresses a position, return an "
    "empty complaints array.\n"
    "- Order complaints by their position in the window (earliest first).\n\n"
    "ANCHOR RULES (for both complaint and response):\n"
    "- Anchors MUST be copied verbatim from the window. Exact wording, exact "
    "punctuation. Do not paraphrase, summarize, or join across newlines.\n"
    "- Anchors should be DISTINCTIVE — avoid generic strings like 'Sincerely,' "
    "alone; include enough surrounding words to uniquely identify the location.\n"
    "- For paraphrase / sectional / none forms, set both complaint anchors to '' "
    "(no contiguous text to slice).\n"
    "- For an absent response, set response.present=false and both response "
    "anchors to ''.\n\n"
    "STANCE RULES (entity-level — SAME across all complaints):\n"
    "- stance MUST be one of: in_favor / opposed / conditional / neutral.\n"
    "  - in_favor: supports the proposal or an aspect of it.\n"
    "  - opposed: objects to the proposal or an aspect of it.\n"
    "  - conditional: supports only if specific conditions are met.\n"
    "  - neutral: has an attributed view but neither supports nor opposes.\n"
    "- stance_confidence MUST be one of:\n"
    "  - high: at least one contiguous statement (letter / testimony / written_comment) "
    "states the position clearly.\n"
    "  - medium: stance is reasonably inferable from a paraphrase or sectional "
    "heading. Real evidence, but no direct statement.\n"
    "  - low: stance is guessed from sparse / ambiguous evidence.\n"
    "- stance_basis: one short phrase pointing at the evidence (e.g. 'opens "
    "letter calling project unacceptable').\n\n"
    "RESPONSE RULES:\n"
    "- A response is the agency's, preparer's, or document author's REPLY to "
    "this specific complaint — typically appears immediately AFTER the entity's "
    "letter/paraphrase, often labeled 'Response:', 'Reply:', 'Discussion:'.\n"
    "- response.agency: who issued the response (e.g. 'Bureau of Land Management', "
    "'Forest Service'). Empty when the response isn't attributed or is just the "
    "doc's preparer.\n"
    "- response.agency_kind: agency / government / other. Use 'other' when "
    "unattributed or unclear.\n"
    "- Set response.present=false when there's no response, or when the nearby "
    "discussion isn't directed at THIS complaint.\n\n"
    "SUMMARY RULES:\n"
    "- summary (entity-level): 2-3 sentences covering the entity's overall "
    "position and key concerns across all their mentions in this window.\n"
    "- complaint_summary: 1 sentence describing what THIS specific mention says. "
    "May differ between two complaints from the same entity (the paraphrase "
    "version may emphasize different points than the full letter).\n"
    "- response.summary: 1-2 sentences capturing the response's substance. "
    "Empty when response.present=false.\n"
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
    is the char offset within window_text where the first evidence page starts.
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


def _slice_anchored(
    window: str,
    opening: str,
    closing: str,
    *,
    near_offset: int,
) -> tuple[Optional[str], bool, bool, int]:
    """Slice window between opening and closing anchors using proximity bias.

    Returns (text, opening_ok, closing_ok, end_offset). end_offset is the char
    position in `window` where the matched closing anchor ends (or 0 if not
    matched) — useful for chaining: a downstream search can use this as its
    `near_offset` (e.g. searching for a response near the end of a complaint).
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
        return None, opening_ok, closing_ok, 0
    start = open_loc[0]
    end = close_loc[1]
    if end <= start:
        return None, True, True, 0
    text = window[start:end]
    if len(text) > settings.STATEMENT_CHAR_CAP:
        text = text[:settings.STATEMENT_CHAR_CAP]
    return text, True, True, end


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
        # 3000 max tokens — multi-complaint output can be larger than the old
        # single-statement output. ~150-200 tokens per complaint, plus the
        # entity-level header.
        out, usage = call_json_with_usage(
            MODEL_SONNET, _SYSTEM, user, max_tokens=3000,
        )
        return out, usage
    except Exception as e:
        log.warning(
            f"find_statement call failed for entity {row.get('entity')!r}: {e}"
        )
        return {
            "stance": "neutral",
            "stance_confidence": "low",
            "stance_basis": "find_statement call failed",
            "summary": "",
            "complaints": [],
            "_error": str(e),
        }, None


def _validate_complaint(c: dict) -> dict:
    """Coerce a complaint dict from the model into a known shape."""
    if not isinstance(c, dict):
        c = {}
    form = c.get("form")
    if form not in settings.STATEMENT_FORMS:
        form = "none"
    opening = (c.get("opening_anchor") or "").strip()
    closing = (c.get("closing_anchor") or "").strip()
    evidence_pages = c.get("evidence_pages") or []
    if not isinstance(evidence_pages, list):
        evidence_pages = []
    evidence_pages = [str(p).strip() for p in evidence_pages if str(p).strip()]
    complaint_summary = (c.get("complaint_summary") or "").strip()

    raw_response = c.get("response") or {}
    if not isinstance(raw_response, dict):
        raw_response = {}
    resp_form = raw_response.get("form")
    if resp_form not in settings.RESPONSE_FORMS:
        resp_form = "none"
    response = {
        "present_claim": bool(raw_response.get("present")),
        "form": resp_form,
        "opening_anchor": (raw_response.get("opening_anchor") or "").strip(),
        "closing_anchor": (raw_response.get("closing_anchor") or "").strip(),
        "agency": (raw_response.get("agency") or "").strip(),
        "agency_kind": (raw_response.get("agency_kind") or "").strip().lower() or "other",
        "summary": (raw_response.get("summary") or "").strip(),
    }
    return {
        "form": form,
        "opening_anchor": opening,
        "closing_anchor": closing,
        "evidence_pages": evidence_pages,
        "complaint_summary": complaint_summary,
        "_response": response,
    }


def _cap_confidence(
    raw_conf: str,
    *,
    complaints: list[dict],
) -> str:
    """Cap confidence based on the BEST complaint form across the entity.

    - all complaints have form == 'none' (or no complaints at all)  → forced low
    - best form is narrator_paraphrase / sectional / no anchors verified → cap at medium
    - else (at least one verified contiguous statement) → as model said
    """
    conf = raw_conf if raw_conf in _CONF_VALUES else "low"
    if not complaints:
        return "low"
    has_verified_contiguous = any(
        c["form"] in ("letter", "testimony", "written_comment")
        and c.get("_text") is not None
        for c in complaints
    )
    only_none = all(c["form"] == "none" for c in complaints)
    if only_none:
        return "low"
    if not has_verified_contiguous:
        if _CONF_RANK[conf] > _CONF_RANK["medium"]:
            conf = "medium"
    return conf


def _process_complaints(
    raw_complaints: list,
    window: str,
    evidence_offset: int,
) -> list[dict]:
    """Validate each complaint, slice its statement and response from the window."""
    out: list[dict] = []
    for raw_c in raw_complaints or []:
        c = _validate_complaint(raw_c)

        # ---- Slice the complaint statement ----
        statement_text: Optional[str] = None
        opening_ok = closing_ok = False
        statement_end_offset = evidence_offset
        if c["form"] != "none" and c["opening_anchor"] and c["closing_anchor"]:
            statement_text, opening_ok, closing_ok, end_offset = _slice_anchored(
                window,
                c["opening_anchor"],
                c["closing_anchor"],
                near_offset=evidence_offset,
            )
            if statement_text is not None:
                statement_end_offset = end_offset

        c["_text"] = statement_text
        c["_opening_verified"] = opening_ok
        c["_closing_verified"] = closing_ok

        # ---- Slice the response (if any) ----
        resp = c["_response"]
        resp_text: Optional[str] = None
        resp_opening_ok = resp_closing_ok = False
        if (
            resp["present_claim"]
            and resp["form"] != "none"
            and resp["opening_anchor"]
            and resp["closing_anchor"]
        ):
            resp_text, resp_opening_ok, resp_closing_ok, _ = _slice_anchored(
                window,
                resp["opening_anchor"],
                resp["closing_anchor"],
                near_offset=statement_end_offset,
            )
        resp["_text"] = resp_text
        resp["_opening_verified"] = resp_opening_ok
        resp["_closing_verified"] = resp_closing_ok

        out.append(c)

    return out


def find_statement_for_row(row: dict, doc: Doc) -> dict:
    """Per-row find_statement.

    Returns the row with `complaints` (list of dicts), `stance`,
    `stance_confidence`, `stance_basis`, `summary`, plus `_statement_usage`.
    Each complaint carries its statement (text + anchors + pages) and an
    embedded `response` dict.
    """
    window, start_page, end_page, evidence_offset = _build_window(
        doc, row.get("evidence_pages") or []
    )
    raw, usage = _ask_model(row, window, start_page, end_page)

    if not isinstance(raw, dict):
        raw = {}
    summary = (raw.get("summary") or "").strip()
    raw_complaints = raw.get("complaints") or []
    if not isinstance(raw_complaints, list):
        raw_complaints = []

    complaints = _process_complaints(raw_complaints, window, evidence_offset)

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
        stance_confidence = _cap_confidence(raw_conf, complaints=complaints)

    # Shape complaints into the final on-disk schema.
    final_complaints: list[dict] = []
    for c in complaints:
        resp = c["_response"]
        final_complaints.append({
            "form": c["form"],
            "evidence_pages": c["evidence_pages"],
            "complaint_summary": c["complaint_summary"],
            "statement": {
                "text": c["_text"],
                "opening_anchor": c["opening_anchor"],
                "closing_anchor": c["closing_anchor"],
                "opening_anchor_verified": c["_opening_verified"],
                "closing_anchor_verified": c["_closing_verified"],
            },
            "response": {
                "present": resp["_text"] is not None,
                "form": resp["form"],
                "agency": resp["agency"],
                "agency_kind": resp["agency_kind"],
                "summary": resp["summary"],
                "text": resp["_text"],
                "opening_anchor": resp["opening_anchor"],
                "closing_anchor": resp["closing_anchor"],
                "opening_anchor_verified": resp["_opening_verified"],
                "closing_anchor_verified": resp["_closing_verified"],
            },
        })

    out = dict(row)
    out["stance"] = stance
    out["stance_confidence"] = stance_confidence
    out["stance_basis"] = stance_basis
    out["summary"] = summary
    out["complaints"] = final_complaints
    out["window_pages"] = [start_page, end_page]
    if raw.get("_error"):
        out["_find_statement_error"] = raw["_error"]
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
                fallback["stance"] = "neutral"
                fallback["stance_confidence"] = "low"
                fallback["stance_basis"] = f"find_statement crashed: {e}"
                fallback["summary"] = ""
                fallback["complaints"] = []
                fallback["_find_statement_error"] = str(e)
                out.append(fallback)
    out.sort(key=lambda r: r.get("sequence", 10**9))
    return out
