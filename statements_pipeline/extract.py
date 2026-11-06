"""
Per-chunk DISCOVERY-only extraction for the statements pipeline.

Differs from `people_pipeline/extract.py` in one critical way: this module does
NOT classify stance. Stance is judged later, at find_statement time, off the
full statement text (or paraphrase block) — the model has much more context
there than it does inside one 50-page chunk.

Extract here is therefore a simple "name + role + verbatim mention" pass. We
keep `attribution_mode` because find_statement uses it as a hint for what kind
of statement to look for (and because the writer's review flagging cares about
verbatim-vs-paraphrase).

This file SHADOWS people_pipeline/extract.py because settings.py appends
external paths to sys.path (so local modules win on name collisions). The
upstream extract is left untouched for people_pipeline's own use.

Output schema per chunk (raw — before merge/verify/find_statement):

  {
    "chunk_index": int,
    "start_page": int,
    "end_page": int,
    "ceq_chapter": str | None,
    "entities": [
      {
        "entity": str,             # name as written
        "kind":   str,             # individual|official|organization|agency|tribe|government|other
        "role":   str,             # short free-text role/title
        "attribution_mode": str,   # direct_quote|paraphrased|sectional
        "quote":  str,             # verbatim text — see attribution_mode rules
        "evidence_pages": [str],   # "12" or "12-13"
        "chunk_index": int
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
from llm import call_with_usage  # from segment_a/

log = logging.getLogger(__name__)

DEBUG_DIR = settings.OUTPUT_DIR / "debug"

# Same budget as upstream — sectional roster chunks can produce dozens of entities.
EXTRACT_MAX_TOKENS = 8000


_EXTRACT_SYSTEM = (
    "You extract every named entity in an Environmental Impact Statement excerpt "
    "whose POSITION on the project (or any specific aspect of it) is attributable "
    "from the text. Entities can be individuals, named officials (speaking for an "
    "org/agency), organizations, agencies, tribal nations, or governments.\n\n"
    "Be EXHAUSTIVE. Older EIS docs often do not contain per-letter quotes; instead "
    "they group commenters under a stance heading or a position-labeled table. "
    "Capture those entities too.\n\n"
    "DO NOT classify the stance. A separate downstream step does that with the "
    "full statement in view. Your job is just discovery + a verbatim mention.\n\n"
    "Respond ONLY with JSON of the form:\n"
    "{\n"
    '  "entities": [\n'
    '    {\n'
    '      "entity": "<name as written>",\n'
    '      "kind":   "individual|official|organization|agency|tribe|government|other",\n'
    '      "role":   "<short role/affiliation, e.g. \'Senator (D-NM)\', \'tribal council\', \'commenter\'>",\n'
    '      "attribution_mode": "direct_quote|paraphrased|sectional",\n'
    '      "quote":  "<verbatim text from the excerpt — see attribution_mode rules>",\n'
    '      "evidence_pages": ["<p>"|"<p>-<p>"]\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "ATTRIBUTION MODES — three valid ways an entity can be flagged as having a position:\n"
    "  1. direct_quote: the entity is quoted directly. `quote` is the verbatim "
    "sentence the entity said.\n"
    "  2. paraphrased: the document narrator paraphrases the entity's position "
    "without a direct quote, but the attribution is unambiguous (e.g. \"The "
    "Sierra Club argued that ORV use should be halted.\"). `quote` is the "
    "verbatim narrator sentence that names the entity and states their position.\n"
    "  3. sectional: the entity appears in a list, table, or roster grouped under "
    "a STANCE HEADING (e.g. a \"PRO REGULATIONS\" / \"CON REGULATIONS\" table; "
    "an \"Organizations supporting X\" list). `quote` is the verbatim heading or "
    "label sentence that establishes the group's position. The entity name itself "
    "must appear verbatim in the listed group.\n\n"
    "STRICT RULES:\n"
    "- The `quote` MUST be copied verbatim from the excerpt. Exact wording and "
    "punctuation. No paraphrasing.\n"
    "- A pure roster (a list of commenters with NO stance heading and NO position "
    "info anywhere near it — e.g. an alphabetical \"List of Recipients\" appendix) "
    "is OUT of scope.\n"
    "- Authorial / agency-narrator voice describing the project's own purpose is "
    "NOT a position. But a narrator sentence like \"The Forest Service believes "
    "the regulations are necessary\" IS a position attributed to the Forest Service.\n"
    "- `evidence_pages` must reference pages within the excerpt's stated page span.\n"
    "- If the same entity is mentioned multiple times in this excerpt, include "
    "the strongest single mention; the merger collapses cross-chunk duplicates."
)


# --- JSON parsing (copied from upstream so we're robust to truncation) -------

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
    """Recover a partial response cut off mid-element inside `entities: [...]`."""
    text = _strip_fences(raw)
    m = re.search(r'"entities"\s*:\s*\[', text)
    if not m:
        return None
    arr_start = m.end()
    elements: list[str] = []
    i = arr_start
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        if text[i] != "{":
            break
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
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    candidate = _first_balanced_object(raw) or _first_balanced_object(cleaned)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    repaired = _repair_truncated_entities_object(raw)
    if repaired is not None:
        return repaired
    raise ValueError("no parseable JSON object in response")


def _save_debug(name: str, raw: str) -> str:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DIR / f"{name}.txt"
    with open(path, "w") as f:
        f.write(raw)
    return str(path)


# --- per-chunk extraction ----------------------------------------------------

def extract_one_chunk(
    chunk: Chunk,
    char_cap: int = settings.EXTRACT_CHAR_CAP,
    doc_id: str = "doc",
) -> dict:
    """Run the discovery extractor against a single chunk."""
    text = chunk.text[:char_cap]
    span = f"{chunk.start_page}-{chunk.end_page}"
    label_hint = f", section: {chunk.label}" if chunk.label else ""
    user = (
        f"Excerpt from chunk #{chunk.index} (pages {span}{label_hint}).\n"
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
            "ceq_chapter": getattr(chunk, "ceq_chapter", None),
            "entities": [],
            "error": str(e),
            "debug_path": debug_path,
            "usage": usage,
        }

    raw_entities = out.get("entities") or []
    cleaned = _clean_entities(raw_entities, chunk)
    return {
        "chunk_index": chunk.index,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "ceq_chapter": getattr(chunk, "ceq_chapter", None),
        "entities": cleaned,
        "n_dropped": len(raw_entities) - len(cleaned),
        "usage": usage,
    }


def _clean_entities(entities: list[dict], chunk: Chunk) -> list[dict]:
    """Validate fields, drop malformed rows. No stance check (stance is downstream)."""
    valid_modes = {"direct_quote", "paraphrased", "sectional"}
    out: list[dict] = []
    for e in entities:
        if not isinstance(e, dict):
            continue
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
            "attribution_mode": mode,
            "quote": quote,
            "evidence_pages": [str(p) for p in evidence_pages],
            "chunk_index": chunk.index,
        })
    return out


def extract_doc(
    chunks: list[Chunk],
    parallel: Optional[int] = None,
    doc_id: str = "doc",
) -> list[dict]:
    """Run discovery extraction over all chunks in parallel."""
    if not chunks:
        return []
    workers = parallel if parallel is not None else settings.EXTRACT_PARALLEL
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
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
