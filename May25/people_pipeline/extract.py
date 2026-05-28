"""
Per-chunk extraction of stance-bearing entities.

For each 50-page chunk, ask Sonnet to return every named entity (person, org,
agency, tribe, government body) that takes a position on the project or any
specific aspect of it. Stance is a closed set; entries without a clear stance
are dropped at parse time.

Output schema per chunk (raw — before merge/verify/critic):

  {
    "chunk_index": int,
    "start_page": int,
    "end_page": int,
    "entities": [
      {
        "entity": str,           # name as it appears
        "kind":   str,           # individual|official|organization|agency|tribe|government|other
        "role":   str,           # short free-text role/title (e.g. "Sierra Club staff", "preparer")
        "stance": str,           # in_favor|opposed|conditional|neutral
        "quote":  str,           # verbatim quote attributed to this entity
        "evidence_pages": [str], # page spans like "12" or "12-13"
        "stance_basis": str      # short note: why this stance is attributed (1 phrase)
      }
    ]
  }
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import settings  # registers segment_a/ on sys.path

from chunk import Chunk        # from segment_a/
from config import MODEL_SONNET  # from segment_a/
from llm import call_with_usage  # from segment_a/  (raw text + token usage)

log = logging.getLogger(__name__)


# Where to dump raw Sonnet responses when JSON parsing fails.
DEBUG_DIR = settings.OUTPUT_DIR / "debug"


# Extractor token budget. With sectional mode, a single chunk can produce
# dozens of entities. 4000 was too small (responses got truncated mid-list,
# leaving no closing brace and an un-parseable JSON object). 8000 gives ~32k
# chars of JSON output, enough for ~80 entities per chunk.
EXTRACT_MAX_TOKENS = 8000


_EXTRACT_SYSTEM = (
    "You extract every named entity in an Environmental Impact Statement excerpt "
    "whose POSITION on the project (or any specific aspect of it) is attributable "
    "from the text. Entities can be individuals, named officials (speaking for an "
    "org/agency), organizations, agencies, tribal nations, or governments.\n\n"
    "Be EXHAUSTIVE. Older EIS docs often do not contain per-letter quotes; instead "
    "they group commenters under a stance heading or a position-labeled table. "
    "Capture those entities too.\n\n"
    "Respond ONLY with JSON of the form:\n"
    "{\n"
    '  "entities": [\n'
    '    {\n'
    '      "entity": "<name as written>",\n'
    '      "kind":   "individual|official|organization|agency|tribe|government|other",\n'
    '      "role":   "<short role/affiliation if known, e.g. \'Senator (D-NM)\', \'cooperating agency\', \'tribal council\', \'commenter\'>",\n'
    '      "stance": "in_favor|opposed|conditional|neutral",\n'
    '      "attribution_mode": "direct_quote|paraphrased|sectional",\n'
    '      "quote":  "<verbatim text from the excerpt — see attribution_mode rules below>",\n'
    '      "evidence_pages": ["<p>"|"<p>-<p>"],\n'
    '      "stance_basis": "<one short phrase: why this stance is attributed>"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "ATTRIBUTION MODES — three valid ways to attribute a stance:\n"
    "  1. direct_quote: the entity is quoted directly making the position. `quote` "
    "is the verbatim sentence the entity said.\n"
    "  2. paraphrased: the document narrator paraphrases the entity's position "
    "without a direct quote, but the attribution is unambiguous "
    "(e.g. \"The Sierra Club argued that ORV use should be halted.\"). `quote` is "
    "the verbatim narrator sentence that names the entity and states their position.\n"
    "  3. sectional: the entity appears in a list, table, or roster grouped under a "
    "STANCE HEADING (e.g. a \"PRO REGULATIONS\" / \"CON REGULATIONS\" table; an "
    "\"Organizations supporting X\" list; a comment-summary section keyed by "
    "sequence number where the position is stated for the group). `quote` is the "
    "verbatim heading or label sentence that establishes the stance for the group "
    "(e.g. the heading \"PRO REGULATIONS\" together with the list intro line). "
    "The entity name itself must appear verbatim in the listed group.\n\n"
    "STRICT RULES:\n"
    "- The `quote` MUST be copied verbatim from the excerpt — exact wording and "
    "punctuation. Do not paraphrase. For sectional mode, the heading or label "
    "sentence IS the quote; copy it exactly as written.\n"
    "- `stance` MUST be one of:\n"
    "  - in_favor: supports the proposal or an aspect of it.\n"
    "  - opposed: objects to the proposal or an aspect of it.\n"
    "  - conditional: supports only if specific conditions are met (mitigation, "
    "alternative selection, scope changes, etc.).\n"
    "  - neutral: has an attributed view but neither supports nor opposes (e.g. "
    "raises concerns without taking a side, asks procedural questions, requests "
    "information).\n"
    "  If you cannot confidently choose one of these four, OMIT the entity.\n"
    "- A pure roster (a list of commenters with NO stance heading and NO position "
    "info anywhere near it — e.g. an alphabetical \"List of Recipients\" appendix) "
    "is OUT of scope. But a list that sits under a position heading or whose intro "
    "describes the group's position IS in scope (use sectional mode).\n"
    "- Authorial / agency-narrator voice describing the project's own purpose is "
    "NOT a stance. But a narrator sentence like \"The Forest Service believes the "
    "regulations are necessary\" IS a stance attributed to the Forest Service.\n"
    "- `evidence_pages` must reference pages within the excerpt's stated page span.\n"
    "- If the same entity takes the same stance multiple times in this excerpt, "
    "include it once with the strongest evidence."
)


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        if s.startswith("json"):
            s = s[4:].strip()
    return s


def _first_balanced_object(s: str) -> Optional[str]:
    """Find the first balanced {...} JSON object in s, ignoring braces inside strings."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _repair_truncated_entities_object(raw: str) -> Optional[dict]:
    """
    Attempt to recover a partial response that looks like:
        {"entities": [{...}, {...}, {...   <-- cut off mid-element

    Strategy: find the outer "entities": [ ... opening, walk the array element
    by element keeping only balanced objects, then synthesize a closing `]}`.
    Returns None if nothing salvageable.
    """
    text = _strip_fences(raw)
    m = re.search(r'"entities"\s*:\s*\[', text)
    if not m:
        return None
    arr_start = m.end()  # index just after `[`
    elements: list[str] = []
    i = arr_start
    n = len(text)
    while i < n:
        # Skip whitespace and commas between elements
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        if text[i] != "{":
            # Garbage / partial token after the last good `}` — stop salvaging.
            break
        # Walk a balanced { ... } object starting at i
        depth = 0
        in_str = False
        esc = False
        elem_start = i
        elem_end: Optional[int] = None
        while i < n:
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        elem_end = i + 1
                        break
            i += 1
        if elem_end is None:
            # Truncated mid-element: stop, keep what we have.
            break
        elements.append(text[elem_start:elem_end])
        i = elem_end
    if not elements:
        return None
    rebuilt = "{\"entities\": [" + ",".join(elements) + "]}"
    try:
        return json.loads(rebuilt)
    except json.JSONDecodeError:
        return None


def _parse_extract_response(raw: str) -> dict:
    """Tolerant JSON parser for extractor responses. Raises ValueError if nothing recoverable."""
    cleaned = _strip_fences(raw)
    # 1. Plain json.loads on the stripped string
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 2. First balanced JSON object anywhere in the response
    candidate = _first_balanced_object(raw) or _first_balanced_object(cleaned)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # 3. Repair a truncated `{"entities": [ ... ]}` response
    repaired = _repair_truncated_entities_object(raw)
    if repaired is not None:
        return repaired
    raise ValueError("no parseable JSON object in response")


def _save_debug(doc_id_or_chunk: str, raw: str) -> str:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DIR / f"{doc_id_or_chunk}.txt"
    with open(path, "w") as f:
        f.write(raw)
    return str(path)


def extract_one_chunk(chunk: Chunk, char_cap: int = settings.EXTRACT_CHAR_CAP, doc_id: str = "doc") -> dict:
    """Run the extractor against a single chunk; returns the raw model JSON + chunk metadata + usage."""
    text = chunk.text[:char_cap]
    span = f"{chunk.start_page}-{chunk.end_page}"
    label_hint = f", section: {chunk.label}" if chunk.label else ""
    user = (
        f"Excerpt from chunk #{chunk.index} (estimated pages {span}{label_hint}).\n"
        f"All page numbers in `evidence_pages` MUST fall within {chunk.start_page}-{chunk.end_page}.\n\n"
        f"--- BEGIN EXCERPT ---\n{text}\n--- END EXCERPT ---"
    )
    raw_response: Optional[str] = None
    usage: Optional[dict] = None
    try:
        raw_response, usage = call_with_usage(
            MODEL_SONNET, _EXTRACT_SYSTEM, user,
            max_tokens=EXTRACT_MAX_TOKENS,
        )
        out = _parse_extract_response(raw_response)
    except Exception as e:
        # Save the raw response (if we got one) so the failure is diagnosable.
        debug_path: Optional[str] = None
        if raw_response is not None:
            debug_path = _save_debug(f"{doc_id}__chunk_{chunk.index}", raw_response)
        log.warning(
            f"extract: chunk {chunk.index} failed: {e}"
            + (f" (raw response saved → {debug_path})" if debug_path else " (no raw response captured)")
        )
        return {
            "chunk_index": chunk.index,
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "ceq_chapter": chunk.ceq_chapter,
            "entities": [],
            "error": str(e),
            "debug_path": debug_path,
            "usage": usage,  # may have been recorded even if parse failed
        }

    raw_entities = out.get("entities") or []
    cleaned = _clean_entities(raw_entities, chunk)
    return {
        "chunk_index": chunk.index,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "ceq_chapter": chunk.ceq_chapter,
        "entities": cleaned,
        "n_dropped": len(raw_entities) - len(cleaned),
        "usage": usage,
    }


def _clean_entities(entities: list[dict], chunk: Chunk) -> list[dict]:
    """Validate stance, fill defaults, drop anything malformed or out-of-vocab."""
    valid_modes = {"direct_quote", "paraphrased", "sectional"}
    out: list[dict] = []
    for e in entities:
        if not isinstance(e, dict):
            continue
        stance = (e.get("stance") or "").strip().lower()
        if stance not in settings.STANCES:
            continue  # drop entries without a recognized stance
        entity = (e.get("entity") or "").strip()
        quote = (e.get("quote") or "").strip()
        if not entity or not quote:
            continue
        kind = (e.get("kind") or "other").strip().lower()
        if kind not in settings.KINDS:
            kind = "other"
        mode = (e.get("attribution_mode") or "direct_quote").strip().lower()
        if mode not in valid_modes:
            mode = "direct_quote"
        evidence_pages = e.get("evidence_pages") or [f"{chunk.start_page}-{chunk.end_page}"]
        if not isinstance(evidence_pages, list) or not evidence_pages:
            evidence_pages = [f"{chunk.start_page}-{chunk.end_page}"]
        out.append({
            "entity": entity,
            "kind": kind,
            "role": (e.get("role") or "").strip(),
            "stance": stance,
            "attribution_mode": mode,
            "quote": quote,
            "evidence_pages": [str(p) for p in evidence_pages],
            "stance_basis": (e.get("stance_basis") or "").strip(),
            "chunk_index": chunk.index,
        })
    return out


def extract_doc(chunks: list[Chunk], parallel: int = settings.EXTRACT_PARALLEL, doc_id: str = "doc") -> list[dict]:
    """Run extraction over all chunks in parallel. Returns one raw record per chunk."""
    if not chunks:
        return []
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(extract_one_chunk, c, doc_id=doc_id): c for c in chunks}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                c = futures[fut]
                log.exception(f"extract chunk {c.index} crashed: {e}")
                results.append({
                    "chunk_index": c.index,
                    "start_page": c.start_page,
                    "end_page": c.end_page,
                    "entities": [],
                    "error": str(e),
                })
    results.sort(key=lambda r: r["chunk_index"])
    return results
