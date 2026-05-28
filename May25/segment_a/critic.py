"""
M2 Check: Critic (Sonnet).

For each M2 field the Critic receives:
  - The extracted value
  - The cited page span (string like "12-14")
  - THE CITED PAGES PULLED IN (not just the page numbers — actual text)

Returns per field: PASS | PASS_WITH_NOTE | RE_EXTRACT | HUMAN_REVIEW
plus model_confidence (low/medium/high) and short notes.

Hard override: public_commenters whose `kind == "private"` always get
HUMAN_REVIEW regardless of Critic verdict.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from config import CHARS_PER_PAGE
from chunk import page_range_chars
from llm import sonnet

log = logging.getLogger(__name__)

VERDICTS = ("PASS", "PASS_WITH_NOTE", "RE_EXTRACT", "HUMAN_REVIEW")


def _resolve_cited_text(text: str, source_pages: list[str]) -> str:
    """Pull the actual cited pages' text from the doc."""
    pieces: list[str] = []
    for span in source_pages or []:
        if not isinstance(span, str) or "-" not in span:
            continue
        try:
            a, b = span.split("-", 1)
            sp = int(a.strip())
            ep = int(b.strip())
        except ValueError:
            continue
        s, e = page_range_chars(sp, ep, len(text))
        pieces.append(text[s:e])
    return "\n\n[...]\n\n".join(pieces)[:80_000]


def _ask_critic(field: str, rubric: str, extracted: object, cited_text: str) -> dict:
    out = sonnet(
        system=(
            "You are a Critic verifying an extraction from an Environmental Impact Statement.\n"
            f"Field under review: {field}\n\n"
            f"RUBRIC — answer each check yes/no, then give a verdict:\n{rubric}\n\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "rubric_results": [{"check": "<short>", "result": "yes|no|n/a", "note": "<short>"}],\n'
            '  "verdict": "PASS|PASS_WITH_NOTE|RE_EXTRACT|HUMAN_REVIEW",\n'
            '  "model_confidence": "low|medium|high",\n'
            '  "notes": "<2-3 sentence summary>"\n'
            "}"
        ),
        user=(
            f"EXTRACTED VALUE:\n{json.dumps(extracted, ensure_ascii=False, indent=2)}\n\n"
            f"CITED PAGES (text pulled in):\n{cited_text or '(no cited pages provided)'}"
        ),
        max_tokens=2000,
    )
    verdict = out.get("verdict")
    if verdict not in VERDICTS:
        out["verdict"] = "HUMAN_REVIEW"
        out["notes"] = (out.get("notes") or "") + " [Critic returned unknown verdict — forcing HUMAN_REVIEW.]"
    return out


# --- Per-field rubrics -------------------------------------------------------

RUBRIC_SUMMARY = (
    "- Each schema field's text supported by content visible on the cited pages?\n"
    "- Numbers, places, and dates in the text match what the cited pages say?\n"
    "- public_response is limited to the main doc (based_on_main_doc_only=true)?\n"
    "- No invention or hallucinated detail beyond cited content?"
)

RUBRIC_ALTERNATIVES = (
    "- Each listed alternative is actually discussed on its cited page?\n"
    "- The No Action alternative is included if the cited text includes it?\n"
    "- Names and descriptions match terminology used in the cited text?\n"
    "- No outside-knowledge alternatives added that aren't in the chapter?"
)

RUBRIC_THEMES = (
    "- Each chosen theme is supported by the doc summary fields?\n"
    "- Theme names come from the fixed taxonomy verbatim?\n"
    "- Subthemes belong to the chosen theme(s) per the taxonomy?\n"
    "- 1–3 themes and 2–5 subthemes selected (within range)?"
)

RUBRIC_LOCATION = (
    "- Place names appear in the cited pages?\n"
    "- is_multi_site set correctly (true if doc spans multiple sites or a corridor)?\n"
    "- For corridor projects: endpoints named, not just a midpoint?\n"
    "- State abbreviations match U.S. conventions where applicable?"
)

RUBRIC_KEY_PEOPLE = (
    "- Each agency preparer named in the cited Preparers/Consultation text?\n"
    "- Cooperating agencies / tribal nations actually listed in the cited text?\n"
    "- For public_commenters: stance present on cited page? Quote verbatim? Speaker clear?\n"
    "- Private individuals: last-name-only or 'private commenter' convention followed?"
)

RUBRIC_TITLE = (
    "- Title is non-empty and plausibly the document's title?\n"
    "- Length is reasonable (not a sentence fragment, not the whole abstract)?\n"
    "- If two sources (NUL + Haiku) — do they roughly agree?"
)

RUBRIC_YEAR = (
    "- Year falls within 1969 ≤ year ≤ current year?\n"
    "- Year present in NUL metadata OR found on first 3 pages?\n"
    "- If NUL and regex disagree, value reflects this with a low-confidence flag?"
)

RUBRIC_EIS_TYPE = (
    "- Type is one of: Draft, Final, Supplemental, ROD, Unknown?\n"
    "- Regex hit on first page and Sonnet on first 2 pages agree?\n"
    "- Evidence phrase supports the chosen type?"
)

RUBRIC_LEAD_AGENCY = (
    "- All listed agencies appear in NUL contributors OR on the cited title pages?\n"
    "- Joint-lead (>2 agencies) flagged with a note for human review?\n"
    "- No role text (e.g., '(issuing body)') left in the agency names?"
)


RUBRIC_BY_FIELD = {
    "title": (RUBRIC_TITLE, False),
    "year": (RUBRIC_YEAR, False),
    "eis_type": (RUBRIC_EIS_TYPE, False),
    "lead_agency": (RUBRIC_LEAD_AGENCY, False),
    "summary": (RUBRIC_SUMMARY, True),
    "alternatives": (RUBRIC_ALTERNATIVES, True),
    "themes": (RUBRIC_THEMES, False),       # cited text is just "summary"; pass [] for pages
    "location": (RUBRIC_LOCATION, True),
    "key_people": (RUBRIC_KEY_PEOPLE, True),
}


def _source_pages_for_field(field: str, m1: dict, m2: dict) -> list[str]:
    """Get the cited page spans for a given field."""
    if field in ("title", "year", "eis_type", "lead_agency"):
        return ["1-3"]  # M1 reads the front matter
    if field == "summary":
        spans: list[str] = []
        for f in m2["summary"].values():
            if isinstance(f, dict):
                spans.extend(f.get("source_pages", []) or [])
        return list(dict.fromkeys(spans))[:8]  # dedupe, cap
    if field == "alternatives":
        return m2["alternatives"].get("source_pages", []) or []
    if field == "themes":
        return []  # themes is derived from summary; nothing to pull
    if field == "location":
        return m2["location"].get("source_pages", []) or []
    if field == "key_people":
        sp = m2["key_people"].get("source_pages", {})
        return list(dict.fromkeys((sp.get("preparers", []) or []) + (sp.get("commenters", []) or [])))
    return []


def _extracted_for_field(field: str, m1: dict, m2: dict) -> object:
    if field in ("title", "year", "eis_type", "lead_agency"):
        return m1.get(field)
    return m2.get(field)


def _apply_private_commenter_override(field: str, extracted: object, critic_result: dict) -> dict:
    """Stakeholder stance for private individuals → HUMAN_REVIEW regardless of verdict."""
    if field != "key_people":
        return critic_result
    if not isinstance(extracted, dict):
        return critic_result
    value = extracted.get("value", extracted)
    commenters = []
    if isinstance(value, dict):
        commenters = value.get("public_commenters", [])
    has_private_stance = any(
        (c.get("kind") == "private") and c.get("stance") for c in commenters or []
    )
    if has_private_stance and critic_result["verdict"] != "HUMAN_REVIEW":
        critic_result["verdict"] = "HUMAN_REVIEW"
        critic_result["notes"] = (critic_result.get("notes") or "") + (
            " [Forced HUMAN_REVIEW: private-individual stance present (policy override).]"
        )
    return critic_result


def run_critic(text: str, m1: dict, m2: dict) -> dict:
    """Run the Critic across all fields. Returns {field: critic_result}."""
    results: dict = {}
    for field, (rubric, needs_text) in RUBRIC_BY_FIELD.items():
        extracted = _extracted_for_field(field, m1, m2)
        source_pages = _source_pages_for_field(field, m1, m2)
        cited_text = _resolve_cited_text(text, source_pages) if needs_text else ""
        try:
            result = _ask_critic(field, rubric, extracted, cited_text)
        except Exception as e:
            log.warning(f"Critic failed on {field}: {e}")
            result = {
                "verdict": "HUMAN_REVIEW",
                "model_confidence": "low",
                "notes": f"Critic call failed: {e}",
                "rubric_results": [],
            }
        result["source_pages"] = source_pages
        result = _apply_private_commenter_override(field, extracted, result)
        results[field] = result
    return results
