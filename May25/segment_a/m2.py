"""
M2: Semantic Extraction.

Every extracted field carries an `evidence` list — each item is a verbatim
quote that has been located on a real page of the doc via
`evidence.verify_and_locate`. Quotes that cannot be found verbatim keep
`quote_verified=false` so the critic forces HUMAN_REVIEW.

Fields:
  - Summary: Opus, map-reduce. Each subfield's evidence is a list of verified
    verbatim quotes selected from the per-chunk findings.
  - Alternatives: Sonnet on the structurally-identified Alternatives chapter.
    Each alternative carries a verbatim quote naming/describing it.
  - Themes: Sonnet from the summary. Evidence is the union of summary evidence.
  - Location: Sonnet on first 30 pages + Project/Study Area. Each place carries
    a verbatim quote.
  - Key People: agency_preparers, cooperating_agencies, public_commenters —
    each entry carries a verbatim quote.
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
    chunks_for_doc,
    first_pages,
    text_for_ceq_chapter,
)
from evidence import Evidence, evidence_for_quotes, union_pages, verify_and_locate
from llm import opus, sonnet
from pages import Doc

log = logging.getLogger(__name__)


# --- Summary (Opus, map-reduce) ---------------------------------------------

SUMMARY_SCHEMA_KEYS = [
    "overview",
    "project_description",
    "affected_community",
    "alternatives_overview",
    "environmental_impact",
    "public_response",
]

# Keys whose evidence the `overview` field unions over.
SUMMARY_SUBFIELDS = [
    "project_description",
    "affected_community",
    "alternatives_overview",
    "environmental_impact",
    "public_response",
]


def _summary_map_one(chunk: Chunk) -> dict:
    """Per-chunk summary map step. Returns text + verbatim supporting quotes per field.

    Each field returns a `quotes` LIST so the reducer has enough candidates to
    cite every claim it carries forward — earlier versions returned a single
    quote per field per chunk, which made the final per-doc summary
    under-cited (you'd see 3 claims in `text` backed by only 1-2 quotes).
    """
    out = opus(
        system=(
            "You are summarizing a slice of an Environmental Impact Statement.\n"
            "Output JSON with EXACTLY these five keys; for each key return BOTH a "
            "short text and a LIST of verbatim supporting quotes drawn from THIS CHUNK:\n"
            "{\n"
            '  "project_description":   {"text": "1-3 sentences", "quotes": ["...", "..."]},\n'
            '  "affected_community":    {"text": "...",            "quotes": ["..."]},\n'
            '  "alternatives_overview": {"text": "...",            "quotes": ["..."]},\n'
            '  "environmental_impact":  {"text": "...",            "quotes": ["..."]},\n'
            '  "public_response":       {"text": "...",            "quotes": ["..."]}\n'
            "}\n"
            "Rules:\n"
            "- If the chunk says nothing about a field, return text=\"\" and quotes=[].\n"
            "- Each entry in `quotes` MUST be copied character-for-character from the "
            "chunk. Pick short sentences or phrases (10-200 chars). A downstream verifier "
            "substring-matches each one against the document; quotes that aren't present "
            "verbatim are flagged for human review.\n"
            "- Provide ONE quote per substantive claim in `text`. If your text has 4 "
            "claims, return 4 quotes. Do not write a sentence in `text` that no quote "
            "supports.\n"
            "- Do not invent. Do not paraphrase inside the `quotes` field."
        ),
        user=(
            f"Chunk #{chunk.index} (pages {chunk.start_page}-{chunk.end_page}"
            f"{', section: ' + chunk.label if chunk.label else ''}):\n\n{chunk.text}"
        ),
        max_tokens=4000,
    )
    return {
        "chunk_index": chunk.index,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "ceq_chapter": chunk.ceq_chapter,
        "findings": out,
    }


def _summary_reduce(partials: list[dict], doc: Doc) -> dict:
    """
    Reduce chunk summaries into the final 6-field doc summary.

    The reducer is asked to:
      - synthesize 2-4 sentence text per subfield
      - return ONE verbatim quote PER SUBSTANTIVE CLAIM in that text (drawn
        from the per-chunk `quotes` lists). Earlier versions capped this at
        1-2 quotes per field, which left later sentences uncited.
      - write a 3-5 sentence overview that draws on the others, similarly
        cited claim-by-claim

    After the model responds, every quote is verified against `doc` and the
    verified page numbers are written into `evidence`.
    """
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
            "You consolidate per-chunk findings into one document-level summary.\n"
            "Output JSON with EXACTLY these keys:\n"
            "{\n"
            '  "overview":              {"text": "3-5 sentences", "quotes": ["..."]},\n'
            '  "project_description":   {"text": "2-4 sentences", "quotes": ["..."]},\n'
            '  "affected_community":    {"text": "...",           "quotes": ["..."]},\n'
            '  "alternatives_overview": {"text": "...",           "quotes": ["..."]},\n'
            '  "environmental_impact":  {"text": "...",           "quotes": ["..."]},\n'
            '  "public_response":       {"text": "...",           "quotes": ["..."], "based_on_main_doc_only": true}\n'
            "}\n"
            "Rules:\n"
            "- overview text encapsulates the other five — what the project is, where, "
            "the alternatives evaluated, the main environmental impacts, and the public "
            "response. Write so a non-expert can understand the document at a glance. "
            "Derive it from the other five fields; do not introduce new facts.\n"
            "- For EVERY key (including overview), provide ONE verbatim quote per "
            "substantive claim in your text. If your text has four claims, return four "
            "quotes. Do not write a sentence in `text` that no quote supports. Pick "
            "quotes from the per-chunk `quotes` arrays VERBATIM (do not edit, paraphrase, "
            "or join). Each quote you return must appear character-for-character "
            "somewhere in the input.\n"
            "- public_response is always limited to the main document; set the flag true.\n"
            "- If no chunks support a field, return text=\"\" and quotes=[]."
        ),
        user=f"Per-chunk findings:\n{payload}",
        max_tokens=6000,
    )

    # Normalize shape and verify quotes.
    final: dict = {}
    for key in SUMMARY_SCHEMA_KEYS:
        entry = out.get(key) or {}
        text = (entry.get("text") if isinstance(entry, dict) else entry) or ""
        quotes = entry.get("quotes") if isinstance(entry, dict) else []
        if not isinstance(quotes, list):
            quotes = []
        evidence_list = evidence_for_quotes([q for q in quotes if isinstance(q, str)], doc)
        final[key] = {"text": text, "evidence": evidence_list}
        if key == "public_response":
            final[key]["based_on_main_doc_only"] = bool(
                (entry.get("based_on_main_doc_only") if isinstance(entry, dict) else True)
                if entry else True
            )

    # Overview evidence: union of evidence from the five subfields when overview's own
    # quotes are missing or unverified, so the overview row in the grading sheet always
    # points at real pages.
    overview_ev = final["overview"]["evidence"]
    if not any(ev.get("quote_verified") for ev in overview_ev):
        unioned: list[Evidence] = []
        seen_pages: set[str] = set()
        for sub in SUMMARY_SUBFIELDS:
            for ev in final[sub]["evidence"]:
                if not ev.get("quote_verified"):
                    continue
                key_id = (ev.get("quote", ""), tuple(ev.get("source_pages", []) or []))
                if any(key_id == (e2.get("quote", ""), tuple(e2.get("source_pages", []) or [])) for e2 in unioned):
                    continue
                unioned.append(ev)
                if len(unioned) >= 3:
                    break
            if len(unioned) >= 3:
                break
        if unioned:
            final["overview"]["evidence"] = unioned
    return final


def extract_summary(chunks: list[Chunk], doc: Doc, max_chunks: int = 12, parallel: int = 4) -> dict:
    """Run Opus over chunks in parallel, then reduce. Caps chunk count to control cost."""
    if not chunks:
        return {k: {"text": "", "evidence": []} for k in SUMMARY_SCHEMA_KEYS}

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
    return _summary_reduce(partials, doc)


# --- Alternatives ------------------------------------------------------------

def extract_alternatives(doc: Doc, chapters: list[dict]) -> dict:
    chapter_text = text_for_ceq_chapter(doc, chapters, "Alternatives")
    if chapter_text is None:
        return {
            "value": [],
            "confidence": "low",
            "evidence": [],
            "note": "Alternatives chapter not detected structurally — skipped per v2 plan (no word-regex fallback).",
        }
    chunk_text, start_page, end_page = chapter_text
    # Cap to ~80 pages to stay within context (rough char cap on the chapter slice)
    cap = 80 * 2500
    if len(chunk_text) > cap:
        chunk_text = chunk_text[:cap]
        end_page = min(end_page, start_page + 79)

    out = sonnet(
        system=(
            "You list the alternatives evaluated in the Alternatives chapter of an EIS. "
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "alternatives": [\n'
            '    {\n'
            '      "name": "<short name>",\n'
            '      "description": "<1-2 sentences, your own words>",\n'
            '      "quote": "<verbatim phrase from the excerpt that names/describes this alternative>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Include the No Action alternative if present. The `quote` MUST be copied "
            "character-for-character from the excerpt; the downstream verifier will "
            "substring-match it against the source document."
        ),
        user=(
            f"Alternatives chapter (pages {start_page}-{end_page}):\n\n{chunk_text}"
        ),
        max_tokens=4000,
    )
    raw = out.get("alternatives") or []
    enriched: list[dict] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        quote = (a.get("quote") or "").strip()
        ev = verify_and_locate(quote, doc) if quote else {
            "quote": "", "source_pages": [], "quote_verified": False,
            "note": "No quote returned by extractor.",
        }
        enriched.append({
            "name": (a.get("name") or "").strip(),
            "description": (a.get("description") or "").strip(),
            "evidence": [ev],
        })
    return {
        "value": enriched,
        "confidence": "high" if enriched and any(e["evidence"][0].get("quote_verified") for e in enriched) else "low",
        "chapter_pages": f"{start_page}-{end_page}",
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
    # Themes are derived from the summary; carry the summary's verified evidence forward
    # so the grading sheet can point at concrete pages.
    carried_evidence: list[Evidence] = []
    seen_pages: set[str] = set()
    for key in SUMMARY_SUBFIELDS:
        sf = summary.get(key) or {}
        for ev in sf.get("evidence", []) or []:
            if not ev.get("quote_verified"):
                continue
            pages = tuple(ev.get("source_pages", []) or [])
            key_id = (ev.get("quote", ""), pages)
            if any(key_id == (e2.get("quote", ""), tuple(e2.get("source_pages", []) or [])) for e2 in carried_evidence):
                continue
            carried_evidence.append(ev)
            if len(carried_evidence) >= 3:
                break
        if len(carried_evidence) >= 3:
            break
    return {
        "value": {
            "themes": out.get("themes", []),
            "subthemes": out.get("subthemes", []),
        },
        "confidence": out.get("self_confidence", "medium"),
        "justification": out.get("justification", ""),
        "evidence": carried_evidence,
    }


# --- Location ----------------------------------------------------------------

def extract_location(doc: Doc, chapters: list[dict]) -> dict:
    pieces: list[tuple[str, int, int]] = []
    first_n = min(FIRST_30_PAGES, doc.n_pages)
    pieces.append((first_pages(doc, first_n), 1, first_n))
    for label in ("Project Area", "Study Area"):
        # Best-effort — these labels aren't standard CEQ but may have been
        # detected as a chapter heading by the regex-based detector.
        for ch in chapters:
            if label.lower() in (ch.get("label") or "").lower():
                seg = doc.full_text[ch["start_char"]:ch["end_char"]]
                pieces.append((seg[:60_000], ch["start_page"], ch["end_page"]))

    joined = "\n\n---\n\n".join(p[0] for p in pieces)

    out = sonnet(
        system=(
            "You extract the geographic location of an EIS project.\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "places": [\n'
            '    {\n'
            '      "name": "<place>",\n'
            '      "kind": "point|corridor|region",\n'
            '      "state": "<US state or null>",\n'
            '      "quote": "<verbatim phrase from the excerpts that establishes this place>"\n'
            "    }\n"
            "  ],\n"
            '  "is_multi_site": true|false,\n'
            '  "self_confidence": "high|medium|low",\n'
            '  "note": "<short>"\n'
            "}\n"
            "For corridors give endpoints in the name (\"Akron, OH to Cleveland, OH\"). "
            "The `quote` MUST be copied character-for-character from the excerpts."
        ),
        user=f"First 30 pages and any Project/Study Area excerpts:\n{joined}",
        max_tokens=800,
    )
    raw_places = out.get("places") or []
    enriched_places: list[dict] = []
    for p in raw_places:
        if not isinstance(p, dict):
            continue
        quote = (p.get("quote") or "").strip()
        ev = verify_and_locate(quote, doc) if quote else {
            "quote": "", "source_pages": [], "quote_verified": False,
            "note": "No quote returned by extractor.",
        }
        enriched_places.append({
            "name": (p.get("name") or "").strip(),
            "kind": (p.get("kind") or "").strip(),
            "state": p.get("state"),
            "evidence": [ev],
        })
    geocoded = _geocode_places(enriched_places)
    return {
        "value": {
            "places": enriched_places,
            "is_multi_site": out.get("is_multi_site", False),
            "geocoded": geocoded,
        },
        "confidence": out.get("self_confidence", "medium"),
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

def extract_key_people(doc: Doc, chapters: list[dict]) -> dict:
    """
    Three categories per the v2 plan:
      - agency_preparers       : from Preparers/Consultation chapter
      - cooperating_agencies   : from Consultation chapter
      - public_commenters      : only when main doc has comment-response content

    Every entry carries a verbatim quote + verified page (via evidence helpers).
    """
    text = doc.full_text
    consultation = text_for_ceq_chapter(doc, chapters, "Consultation")
    preparers_text = consultation[0] if consultation else first_pages(doc, FIRST_30_PAGES)

    preparers_out = sonnet(
        system=(
            "You list (a) agency staff who prepared this EIS and (b) cooperating "
            "agencies / tribal nations consulted. Respond ONLY with JSON:\n"
            "{\n"
            '  "agency_preparers": [\n'
            '    {\n'
            '      "name": "<full name>",\n'
            '      "role": "<role/title>",\n'
            '      "quote": "<verbatim phrase from the excerpt naming this person>"\n'
            '    }\n'
            "  ],\n"
            '  "cooperating_agencies": [\n'
            '    {\n'
            '      "name": "<agency or nation>",\n'
            '      "quote": "<verbatim phrase from the excerpt naming this agency>"\n'
            '    }\n'
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Do NOT include private individuals.\n"
            "- Do NOT attribute stances.\n"
            "- Quotes MUST be copied character-for-character from the excerpt; the "
            "downstream verifier will substring-match them against the document."
        ),
        user=f"Consultation/Preparers excerpt:\n{preparers_text[:60_000]}",
        max_tokens=2000,
    )

    preparers = _enrich_named_list(preparers_out.get("agency_preparers", []), doc)
    cooperating = _enrich_named_list(preparers_out.get("cooperating_agencies", []), doc)

    # Public commenters: detect comment-response content first
    has_comment_response = bool(re.search(
        r"\b(comment(?:s)?\s+and\s+response|response\s+to\s+comments)\b",
        text, re.IGNORECASE,
    ))

    commenters_block: list[dict] = []
    if has_comment_response:
        m = re.search(r"\b(comment(?:s)?\s+and\s+response|response\s+to\s+comments)\b", text, re.IGNORECASE)
        start = m.start() if m else 0
        excerpt = text[start : start + 60_000]
        excerpt_start_page = doc.page_at_offset(start)
        excerpt_end_page = doc.page_at_offset(start + len(excerpt))
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
                '      "quote": "<verbatim quote attributed to this commenter>"\n'
                "    }\n"
                "  ]\n"
                "}\n"
                "Rules: only include commenters whose stance is CLEARLY attributed. Use last name only "
                "for private individuals (or 'private commenter'). Quotes MUST be verbatim from the excerpt."
            ),
            user=(
                f"Comment-response excerpt (pages {excerpt_start_page}-{excerpt_end_page}):\n\n{excerpt}"
            ),
            max_tokens=2000,
        )
        raw_commenters = out.get("commenters") or []
        for c in raw_commenters:
            if not isinstance(c, dict):
                continue
            quote = (c.get("quote") or "").strip()
            ev = verify_and_locate(quote, doc) if quote else {
                "quote": "", "source_pages": [], "quote_verified": False,
                "note": "No quote returned by extractor.",
            }
            commenters_block.append({
                "name": (c.get("name") or "").strip(),
                "kind": (c.get("kind") or "").strip(),
                "stance": (c.get("stance") or "").strip(),
                "evidence": [ev],
            })

    return {
        "value": {
            "agency_preparers": preparers,
            "cooperating_agencies": cooperating,
            "public_commenters": commenters_block,
            "comment_response_present": has_comment_response,
        },
        "confidence": "high" if preparers else "medium",
    }


def _enrich_named_list(items: list[dict], doc: Doc) -> list[dict]:
    """For each {name, role?, quote} item, verify the quote and attach evidence."""
    out: list[dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        quote = (item.get("quote") or "").strip()
        ev = verify_and_locate(quote, doc) if quote else {
            "quote": "", "source_pages": [], "quote_verified": False,
            "note": "No quote returned by extractor.",
        }
        cleaned = {k: v for k, v in item.items() if k != "quote"}
        cleaned["evidence"] = [ev]
        out.append(cleaned)
    return out


# --- Top-level ---------------------------------------------------------------

def run_m2(doc: Doc, chunked: Optional[dict] = None) -> dict:
    """Run all M2 extractors. `chunked` is the output of chunks_for_doc; computed if omitted."""
    chunked = chunked or chunks_for_doc(doc)
    chunks: list[Chunk] = chunked["chunks"]
    chapters: list[dict] = chunked["chapters"]

    log.info(f"M2: {len(chunks)} chunks, {len(chapters)} CEQ-mapped chapters detected")

    summary = extract_summary(chunks, doc)
    alternatives = extract_alternatives(doc, chapters)
    themes = extract_themes(summary)
    location = extract_location(doc, chapters)
    key_people = extract_key_people(doc, chapters)

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
                 "start_page": c["start_page"],
                 "end_page": c["end_page"]}
                for c in chapters
            ],
            "chunk_size_pages": CHUNK_PAGES,
        },
    }
