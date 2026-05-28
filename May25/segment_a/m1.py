"""
M1: Easily Gathered (NUL-first, not LLM-first).

Fields:
  - title          (NUL primary, Haiku fallback on first page)
  - year           (NUL primary, regex on first 3 pages fallback)
  - eis_type       (regex on first page primary, Sonnet on first 2 pages verifier)
  - lead_agency    (NUL primary, Sonnet on first 4 pages fallback, list-valued)

Each field returns {value, confidence, sources}.
"""

from __future__ import annotations

import re
from typing import Optional

from config import (
    EIS_TYPE_PATTERNS,
    FIRST_2_PAGES,
    FIRST_3_PAGES,
    FIRST_4_PAGES,
    FIRST_PAGE_CHARS,
    YEAR_MAX,
    YEAR_MIN,
)
from chunk import first_pages
from llm import haiku, sonnet
from nul import get_contributors, get_year


# --- Title -------------------------------------------------------------------

def extract_title(work: dict, text: str) -> dict:
    nul_title = work.get("title") or work.get("nul_metadata", {}).get("title")
    if isinstance(nul_title, list):
        nul_title = nul_title[0] if nul_title else None
    if nul_title and 5 <= len(nul_title) <= 500:
        return {
            "value": nul_title.strip(),
            "confidence": "high",
            "sources": ["NUL"],
        }
    # Fallback: Haiku on first page
    first = text[:FIRST_PAGE_CHARS]
    out = haiku(
        system=(
            "You extract the title of an Environmental Impact Statement from OCR text. "
            "Respond ONLY with JSON: {\"title\": \"<full title>\"}. "
            "If you cannot determine a plausible title, return an empty string."
        ),
        user=f"OCR (first page):\n{first}",
        max_tokens=300,
    )
    title = (out.get("title") or "").strip()
    return {
        "value": title,
        "confidence": "medium" if title else "low",
        "sources": ["Haiku (first page)"],
    }


# --- Year --------------------------------------------------------------------

YEAR_RE = re.compile(r"\b(19[6-9]\d|20[0-2]\d)\b")


def extract_year(work: dict, text: str) -> dict:
    nul_year = get_year(work)
    head = text[:FIRST_3_PAGES]
    regex_years = [
        int(y) for y in YEAR_RE.findall(head)
        if YEAR_MIN <= int(y) <= YEAR_MAX
    ]
    regex_year = _modal_year(regex_years)

    if nul_year and regex_year and nul_year == regex_year:
        return {"value": nul_year, "confidence": "high", "sources": ["NUL", "regex (first 3 pages)"]}
    if nul_year and YEAR_MIN <= nul_year <= YEAR_MAX:
        # NUL present; regex either missing or disagrees
        if regex_year and regex_year != nul_year:
            return {
                "value": nul_year,
                "confidence": "low",
                "sources": ["NUL", "regex (first 3 pages)"],
                "note": f"NUL={nul_year} disagrees with regex={regex_year} — flag",
            }
        return {"value": nul_year, "confidence": "high", "sources": ["NUL"]}
    if regex_year:
        return {"value": regex_year, "confidence": "medium", "sources": ["regex (first 3 pages)"]}
    return {"value": None, "confidence": "low", "sources": []}


def _modal_year(years: list[int]) -> Optional[int]:
    if not years:
        return None
    from collections import Counter
    return Counter(years).most_common(1)[0][0]


# --- EIS Type ----------------------------------------------------------------

def extract_eis_type(text: str) -> dict:
    head = text[:FIRST_PAGE_CHARS]
    regex_hits: list[str] = []
    # Order matters: try Supplemental and ROD first (per config) so they win
    # against the Draft/Final patterns when both appear.
    for label, pat in EIS_TYPE_PATTERNS.items():
        if re.search(pat, head, re.IGNORECASE):
            regex_hits.append(label)
            break  # first hit wins by config ordering

    regex_label = regex_hits[0] if regex_hits else None

    verifier_text = text[:FIRST_2_PAGES]
    out = sonnet(
        system=(
            "You determine the type of an Environmental Impact Statement. "
            "Choose ONE from: Draft, Final, Supplemental, ROD. "
            "If none of these clearly applies, return \"Unknown\". "
            "Respond ONLY with JSON: {\"eis_type\": \"<one of Draft|Final|Supplemental|ROD|Unknown>\", "
            "\"evidence\": \"<short phrase from the text supporting your choice>\"}."
        ),
        user=f"OCR (first 2 pages):\n{verifier_text}",
        max_tokens=300,
    )
    sonnet_label = (out.get("eis_type") or "").strip()

    if regex_label and sonnet_label and regex_label == sonnet_label:
        return {
            "value": regex_label,
            "confidence": "high",
            "sources": ["regex (first page)", "Sonnet (first 2 pages)"],
            "evidence": out.get("evidence", ""),
        }
    if regex_label and not sonnet_label:
        return {"value": regex_label, "confidence": "medium", "sources": ["regex (first page)"]}
    if sonnet_label and not regex_label:
        return {"value": sonnet_label, "confidence": "medium", "sources": ["Sonnet (first 2 pages)"]}
    if regex_label and sonnet_label and regex_label != sonnet_label:
        return {
            "value": regex_label,
            "confidence": "low",
            "sources": ["regex (first page)", "Sonnet (first 2 pages)"],
            "note": f"regex={regex_label} disagrees with Sonnet={sonnet_label}",
            "evidence": out.get("evidence", ""),
        }
    return {"value": "Unknown", "confidence": "low", "sources": []}


# --- Lead Agency / Contributor ----------------------------------------------

def extract_lead_agency(work: dict, text: str) -> dict:
    nul_contribs = get_contributors(work)
    # Strip role parens
    nul_contribs = [re.sub(r"\s*\([^)]+\)\s*$", "", c).strip() for c in nul_contribs]
    nul_contribs = [c for c in nul_contribs if c]

    if nul_contribs:
        result = {
            "value": nul_contribs,
            "confidence": "high",
            "sources": ["NUL"],
        }
        if len(nul_contribs) > 2:
            result["note"] = "More than 2 agencies — likely joint-lead; verify."
        return result

    # Fallback: Sonnet on first 4 pages
    head = text[:FIRST_4_PAGES]
    out = sonnet(
        system=(
            "You extract the lead and cooperating federal agencies listed on the "
            "cover/title pages of an Environmental Impact Statement. "
            "Respond ONLY with JSON: {\"agencies\": [\"<agency name>\", ...]}. "
            "Return an empty list if you cannot find named agencies."
        ),
        user=f"OCR (first 4 pages):\n{head}",
        max_tokens=400,
    )
    agencies = out.get("agencies") or []
    if not isinstance(agencies, list):
        agencies = []
    agencies = [a for a in (str(x).strip() for x in agencies) if a]
    return {
        "value": agencies,
        "confidence": "medium" if agencies else "low",
        "sources": ["Sonnet (first 4 pages)"],
        "note": "More than 2 agencies — likely joint-lead; verify." if len(agencies) > 2 else None,
    }


# --- Top-level ---------------------------------------------------------------

def run_m1(work: dict, text: str) -> dict:
    return {
        "title": extract_title(work, text),
        "year": extract_year(work, text),
        "eis_type": extract_eis_type(text),
        "lead_agency": extract_lead_agency(work, text),
    }
