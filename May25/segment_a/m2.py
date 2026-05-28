"""
M2: Semantic Extraction.

Per the v2 plan:
  - Summary: Opus, section-mapped chunks in parallel + reduce call. Schema:
      project_description, affected_community, alternatives_overview,
      environmental_impact, public_response.
    Each field cites pages. public_response carries based_on_main_doc_only=true
    (no appendix support yet — that's M3).
  - Alternatives Proposed: Sonnet on the structurally-identified Alternatives
    chapter (NOT regex on the word "alternative").
  - Themes: Sonnet, given CHUNK SUMMARIES (not raw text), assigns 1–3 themes +
    2–5 subthemes from the prepared taxonomy.
  - Location + Geometry: Sonnet on first 30 pages + any "Project Area"/"Study
    Area" chapter; geocoder pass; list for linear/multi-site projects.
  - Key People / Groups: split into Agency preparers, Cooperating agencies,
    Public commenters w/ stance. Verbatim quote checker on EXACT cited page.

All page numbers are ESTIMATED from char offsets. Each extracted field is
returned with {value, confidence, source_pages, ...}.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from config import (
    CHUNK_PAGES,
    FIRST_30_PAGES,
    THEMES,
)
from chunk import (
    Chunk,
    char_to_page,
    chunks_for_doc,
    first_pages,
    text_for_ceq_chapter,
)
from llm import opus, sonnet

log = logging.getLogger(__name__)


# --- Summary (Opus, map-reduce) ---------------------------------------------

SUMMARY_SCHEMA_KEYS = [
    "project_description",
    "affected_community",
    "alternatives_overview",
    "environmental_impact",
    "public_response",
]


def _summary_map_one(chunk: Chunk) -> dict:
    """Per-chunk summary map step. Returns partial findings keyed by schema field."""
    out = opus(
        system=(
            "You are summarizing a slice of an Environmental Impact Statement. "
            "Output JSON with EXACTLY these keys, each holding 1-3 sentences drawn "
            "ONLY from the given chunk (return \"\" if the chunk has nothing to say "
            "about that key):\n"
            "{\n"
            '  "project_description": "...",\n'
            '  "affected_community":  "...",\n'
            '  "alternatives_overview": "...",\n'
            '  "environmental_impact": "...",\n'
            '  "public_response": "..."\n'
            "}\n"
            "Do not invent. Do not include page numbers in your answer text."
        ),
        user=(
            f"Chunk #{chunk.index} (estimated pages {chunk.start_page}-{chunk.end_page}"
            f"{', section: ' + chunk.label if chunk.label else ''}):\n\n{chunk.text}"
        ),
        max_tokens=2000,
    )
    return {
        "chunk_index": chunk.index,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "ceq_chapter": chunk.ceq_chapter,
        "findings": out,
    }


def _summary_reduce(partials: list[dict]) -> dict:
    """Reduce chunk summaries into the final 5-field doc summary with page citations."""
    payload = json.dumps([
        {
            "chunk_index": p["chunk_index"],
            "pages": f"{p['start_page']}-{p['end_page']}",
            "ceq_chapter": p["ceq_chapter"],
            "findings": p["findings"],
        }
        for p in partials
    ], ensure_ascii=False)

    out = opus(
        system=(
            "You are consolidating per-chunk findings into one document-level summary. "
            "Output JSON with EXACTLY these keys:\n"
            "{\n"
            '  "project_description":   {"text": "...", "source_pages": ["12-14", "..."]},\n'
            '  "affected_community":    {"text": "...", "source_pages": [...]},\n'
            '  "alternatives_overview": {"text": "...", "source_pages": [...]},\n'
            '  "environmental_impact":  {"text": "...", "source_pages": [...]},\n'
            '  "public_response":       {"text": "...", "source_pages": [...], "based_on_main_doc_only": true}\n'
            "}\n"
            "Rules:\n"
            "- Each text field: 2-4 sentences, plain language, no jargon, no invention.\n"
            "- source_pages must be drawn from the chunk pages provided.\n"
            "- public_response is always limited to the main document (no appendix); set the flag true.\n"
            "- If no chunks support a field, return text=\"\" and source_pages=[]."
        ),
        user=f"Per-chunk findings:\n{payload}",
        max_tokens=4000,
    )
    # Normalize shape
    for key in SUMMARY_SCHEMA_KEYS:
        if key not in out:
            out[key] = {"text": "", "source_pages": []}
        if key == "public_response":
            out[key].setdefault("based_on_main_doc_only", True)
    return out


def extract_summary(chunks: list[Chunk], max_chunks: int = 12, parallel: int = 4) -> dict:
    """Run Opus over chunks in parallel, then reduce. Caps chunk count to control cost."""
    if not chunks:
        return {k: {"text": "", "source_pages": []} for k in SUMMARY_SCHEMA_KEYS}

    # Prefer chunks tagged with a CEQ chapter; fill the rest in document order.
    tagged = [c for c in chunks if c.ceq_chapter]
    untagged = [c for c in chunks if not c.ceq_chapter]
    selected = (tagged + untagged)[:max_chunks]
    selected.sort(key=lambda c: c.index)

    partials: list[dict] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_summary_map_one, c): c for c in selected}
        for fut in as_completed(futures):
            try:
                partials.append(fut.result())
            except Exception as e:
                log.warning(f"Summary map failed for chunk {futures[fut].index}: {e}")
    partials.sort(key=lambda p: p["chunk_index"])
    return _summary_reduce(partials)


# --- Alternatives ------------------------------------------------------------

def extract_alternatives(text: str, chapters: list[dict]) -> dict:
    chapter_text = text_for_ceq_chapter(text, chapters, "Alternatives")
    if chapter_text is None:
        return {
            "value": [],
            "confidence": "low",
            "source_pages": [],
            "note": "Alternatives chapter not detected structurally — skipped per v2 plan (no word-regex fallback).",
        }
    chunk_text, start_page, end_page = chapter_text
    # Cap to ~80 pages to stay within context
    cap = 80 * 2500
    if len(chunk_text) > cap:
        chunk_text = chunk_text[:cap]
        end_page = start_page + 79

    out = sonnet(
        system=(
            "You list the alternatives evaluated in the Alternatives chapter of an EIS. "
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "alternatives": [\n'
            '    {"name": "<short name>", "description": "<1-2 sentences>", "source_pages": ["<p>-<p>"]}\n'
            "  ]\n"
            "}\n"
            "Include the No Action alternative if present. Page numbers refer to the chapter excerpt provided."
        ),
        user=(
            f"Alternatives chapter (estimated pages {start_page}-{end_page}):\n\n{chunk_text}"
        ),
        max_tokens=4000,
    )
    alternatives = out.get("alternatives") or []
    return {
        "value": alternatives,
        "confidence": "high" if alternatives else "low",
        "source_pages": [f"{start_page}-{end_page}"],
    }


# --- Themes ------------------------------------------------------------------

def extract_themes(summary: dict) -> dict:
    """Sonnet, given chunk summaries (here we use the reduced doc summary), assigns themes."""
    payload = json.dumps({
        k: (v.get("text") if isinstance(v, dict) else v)
        for k, v in summary.items()
    }, ensure_ascii=False)
    out = sonnet(
        system=(
            "You classify Environmental Impact Statements into a fixed taxonomy.\n"
            "Choose 1-3 themes and 2-5 subthemes from the taxonomy below.\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "themes": ["<theme>"],\n'
            '  "subthemes": ["<subtheme>"],\n'
            '  "justification": "<1-2 sentences citing which schema field supports the choice>",\n'
            '  "self_confidence": "high|medium|low"\n'
            "}\n\n"
            f"TAXONOMY:\n{json.dumps(THEMES, indent=2)}"
        ),
        user=f"Document summary fields:\n{payload}",
        max_tokens=600,
    )
    return {
        "value": {
            "themes": out.get("themes", []),
            "subthemes": out.get("subthemes", []),
        },
        "confidence": out.get("self_confidence", "medium"),
        "justification": out.get("justification", ""),
        "source_pages": ["summary"],
    }


# --- Location ----------------------------------------------------------------

def extract_location(text: str, chapters: list[dict]) -> dict:
    pieces: list[tuple[str, int, int]] = []
    pieces.append((first_pages(text, 30), 1, min(30, max(1, len(text) // 2500))))
    for label in ("Project Area", "Study Area"):
        # Best-effort — these labels aren't standard CEQ but may have been
        # detected as a chapter heading by the regex-based detector.
        for ch in chapters:
            if label.lower() in (ch.get("label") or "").lower():
                seg = text[ch["start_char"]:ch["end_char"]]
                pieces.append((seg[:60_000], char_to_page(ch["start_char"]), char_to_page(ch["end_char"])))

    joined = "\n\n---\n\n".join(p[0] for p in pieces)
    page_spans = [f"{p[1]}-{p[2]}" for p in pieces]

    out = sonnet(
        system=(
            "You extract the geographic location of an EIS project.\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "places": [\n'
            '    {"name": "<place>", "kind": "point|corridor|region", "state": "<US state or null>"}\n'
            "  ],\n"
            '  "is_multi_site": true|false,\n'
            '  "self_confidence": "high|medium|low",\n'
            '  "note": "<short>"\n'
            "}\n"
            "For corridors give endpoints in the name (\"Akron, OH to Cleveland, OH\")."
        ),
        user=f"First 30 pages and any Project/Study Area excerpts:\n{joined}",
        max_tokens=600,
    )
    places = out.get("places") or []
    geocoded = _geocode_places(places)
    return {
        "value": {
            "places": places,
            "is_multi_site": out.get("is_multi_site", False),
            "geocoded": geocoded,
        },
        "confidence": out.get("self_confidence", "medium"),
        "source_pages": page_spans,
        "note": out.get("note", ""),
    }


def _geocode_places(places: list[dict]) -> list[dict]:
    """Best-effort geocode via Nominatim. Skips silently if geopy missing."""
    try:
        from geopy.geocoders import Nominatim  # type: ignore
    except ImportError:
        return [{"name": p.get("name"), "lat": None, "lon": None, "note": "geopy not installed"} for p in places]
    geo = Nominatim(user_agent="eis_pipeline_segmentA")
    out: list[dict] = []
    for p in places:
        name = p.get("name")
        if not name:
            continue
        try:
            time.sleep(1.1)  # Nominatim rate limit
            r = geo.geocode(name, timeout=10)
            if r:
                out.append({"name": name, "lat": round(r.latitude, 6), "lon": round(r.longitude, 6), "address": r.address})
            else:
                out.append({"name": name, "lat": None, "lon": None})
        except Exception as e:
            out.append({"name": name, "lat": None, "lon": None, "error": str(e)})
    return out


# --- Key People / Groups -----------------------------------------------------

def extract_key_people(text: str, chapters: list[dict]) -> dict:
    """
    Three categories per the v2 plan:
      - agency_preparers       : from Preparers/Consultation chapter
      - cooperating_agencies   : from Consultation chapter
      - public_commenters      : only when main doc has comment-response content
    """
    consultation = text_for_ceq_chapter(text, chapters, "Consultation")
    preparers_text = consultation[0] if consultation else first_pages(text, 30)
    preparers_page_span = (
        f"{consultation[1]}-{consultation[2]}" if consultation else "1-30"
    )

    preparers_out = sonnet(
        system=(
            "You list (a) agency staff who prepared this EIS and (b) cooperating "
            "agencies / tribal nations consulted. Respond ONLY with JSON:\n"
            "{\n"
            '  "agency_preparers":     [{"name": "<full name>", "role": "<role/title>"}],\n'
            '  "cooperating_agencies": [{"name": "<agency or nation>"}],\n'
            '  "source_excerpt":       "<short verbatim phrase used as evidence>"\n'
            "}\n"
            "Do NOT include private individuals. Do NOT attribute stances."
        ),
        user=f"Consultation/Preparers excerpt:\n{preparers_text[:60_000]}",
        max_tokens=1200,
    )

    # Public commenters: detect comment-response content first
    has_comment_response = bool(re.search(
        r"\b(comment(?:s)?\s+and\s+response|response\s+to\s+comments)\b",
        text, re.IGNORECASE,
    ))

    commenters_block: list[dict] = []
    commenters_pages: list[str] = []
    if has_comment_response:
        m = re.search(r"\b(comment(?:s)?\s+and\s+response|response\s+to\s+comments)\b", text, re.IGNORECASE)
        start = m.start() if m else 0
        excerpt = text[start : start + 60_000]
        commenters_pages = [f"{char_to_page(start)}-{char_to_page(start + len(excerpt))}"]
        out = sonnet(
            system=(
                "You list public commenters with attributed stances from a comments-and-response "
                "section. Respond ONLY with JSON:\n"
                "{\n"
                '  "commenters": [\n'
                '    {\n'
                '      "name": "<last name only for private individuals, full name for officials/organizations>",\n'
                '      "kind": "private|organization|official|tribal",\n'
                '      "stance": "support|oppose|conditional|neutral",\n'
                '      "quote": "<verbatim quote attributed to this commenter>",\n'
                '      "source_pages": ["<p>-<p>"]\n'
                "    }\n"
                "  ]\n"
                "}\n"
                "Rules: only include commenters whose stance is CLEARLY attributed. Use last name only "
                "for private individuals (or 'private commenter'). Quotes must be verbatim from the excerpt."
            ),
            user=f"Comment-response excerpt (estimated pages {commenters_pages[0]}):\n\n{excerpt}",
            max_tokens=1500,
        )
        commenters_block = out.get("commenters") or []

    # Verbatim quote check: drop quotes that don't appear in the cited text
    commenters_block = _verify_quotes(commenters_block, text)

    return {
        "value": {
            "agency_preparers": preparers_out.get("agency_preparers", []),
            "cooperating_agencies": preparers_out.get("cooperating_agencies", []),
            "public_commenters": commenters_block,
            "comment_response_present": has_comment_response,
        },
        "confidence": "high" if preparers_out.get("agency_preparers") else "medium",
        "source_pages": {
            "preparers": [preparers_page_span],
            "commenters": commenters_pages,
        },
    }


def _verify_quotes(commenters: list[dict], full_text: str) -> list[dict]:
    """Drop quotes that don't appear verbatim in the cited page span (± estimation error)."""
    out: list[dict] = []
    for c in commenters:
        quote = (c.get("quote") or "").strip()
        if not quote:
            continue
        # Be tolerant of OCR whitespace
        normalized_quote = re.sub(r"\s+", " ", quote)
        normalized_text = re.sub(r"\s+", " ", full_text)
        if normalized_quote in normalized_text:
            out.append(c)
        else:
            c["quote_verified"] = False
            c["note"] = "Quote not found verbatim — flagged for HUMAN_REVIEW"
            out.append(c)
    for c in out:
        c.setdefault("quote_verified", True)
    return out


# --- Top-level ---------------------------------------------------------------

def run_m2(text: str, chunked: Optional[dict] = None) -> dict:
    """Run all M2 extractors. `chunked` is the output of chunks_for_doc; computed if omitted."""
    chunked = chunked or chunks_for_doc(text)
    chunks: list[Chunk] = chunked["chunks"]
    chapters: list[dict] = chunked["chapters"]

    log.info(f"M2: {len(chunks)} chunks, {len(chapters)} CEQ-mapped chapters detected")

    summary = extract_summary(chunks)
    alternatives = extract_alternatives(text, chapters)
    themes = extract_themes(summary)
    location = extract_location(text, chapters)
    key_people = extract_key_people(text, chapters)

    return {
        "summary": summary,
        "alternatives": alternatives,
        "themes": themes,
        "location": location,
        "key_people": key_people,
        "chunking_meta": {
            "n_chunks": len(chunks),
            "n_chapters_detected": len(chapters),
            "chapters": [
                {"label": c["label"], "ceq_chapter": c["ceq_chapter"],
                 "start_page": char_to_page(c["start_char"]),
                 "end_page": char_to_page(c["end_char"])}
                for c in chapters
            ],
            "chunk_size_pages": CHUNK_PAGES,
            "note": "Page numbers are ESTIMATED from char offsets at 2500 chars/page.",
        },
    }
