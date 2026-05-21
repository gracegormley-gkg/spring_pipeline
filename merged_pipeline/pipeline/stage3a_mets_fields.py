"""
Stage 3a — METS-equivalent fields: title, year, lead_agency, date.publication.

Strategy (synthesis_plan.md §Title/Year/Agency):
  - METS-first: pull from ingest.nul_metadata when populated.
  - Extractive cross-check on the cover when NUL is empty or unmatched:
      * title:        Haiku gap-fill on cover (regex too brittle).
      * lead_agency:  AGENCY_VOCAB substring scan on cover -> Haiku gap-fill.
      * year:         regex on first 5 fake pages (no Haiku — regex is robust
                      for 4-digit years in the NEPA range).
  - Provenance is per-field. Sources cascade through:
      nul_api -> regex / controlled_vocab_match -> haiku_gapfill.
  - Status reflects path: "ok" or "extracted_from_mets" if NUL was the source,
    "needs_review" if the field needed cross-check, "deferred_v1" for fields
    we don't extract yet (office_or_region).

Empirical signal: fixture p1074_35556035057348 has NUL.title populated but
NUL.creator=[] and NUL.date_created=[]. Cross-check is mandatory for
agency+year on this doc; the title path lifts straight from NUL.

Per synthesis_plan §No corpus filter shortcut: lead_agency must always be
extracted; never asserted_by_corpus_filter.

Adapted from v1_multiagent_pipeline/pipeline/stage3a_mets_fields.py:
  - Drops the USFS-only manifest-check warning (full-NUL corpus).
  - Adds title and agency Haiku gap-fill paths.
  - Provenance.section/page/char_offset_raw set when cover is the source so
    downstream (Stage 4 critic) can locate the evidence.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from . import config
from .ingest import IngestArtifact
from .schema import (
    EISRecord,
    FieldWithStatus,
    PublicationDate,
    Provenance,
)

if TYPE_CHECKING:
    from .llm_client import LLMClient
    from .sections import SectionsArtifact

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Year regex (only used for cover/page-1-5 fallback; NUL date parsing has its own).
_YEAR_PATTERN = re.compile(r"\b(19[7-9]\d|20\d{2})\b")

# How many chars of cover text to feed to Haiku gap-fill prompts.
_COVER_TEXT_CAP = 6000


def run(
    record: EISRecord,
    ingest: IngestArtifact,
    sections_artifact: "SectionsArtifact | None" = None,
    llm: "LLMClient | None" = None,
) -> list[str]:
    """Mutate `record` in place. Returns a list of warnings.

    `sections_artifact` is required if you want cover-anchored provenance on
    fallback fields. If None, fallbacks still work but provenance.section
    is left null. (The schema's _provenance_section_in_sections validator
    only checks values that are set.)

    `llm` may be None -> Haiku gap-fill silently skipped; cross-check stops
    at the regex layer. Field status downgrades to "needs_review" when this
    happens for a field that didn't have a NUL source.
    """
    warnings: list[str] = []
    nul = ingest.nul_metadata or {}

    cover_span = _cover_span(sections_artifact)
    cover_text = (
        ingest.raw_text[cover_span[0]:cover_span[1]]
        if cover_span is not None else ingest.raw_text[:_COVER_TEXT_CAP]
    )

    # --- title ---
    _populate_title(record, nul, cover_text, cover_span, llm, warnings)

    # --- date.publication + year ---
    _populate_year_and_date(record, nul, ingest, warnings)

    # --- lead_agency ---
    _populate_lead_agency(record, nul, cover_text, cover_span, llm, warnings)

    # --- agency.office_or_region: deferred_v1 ---
    record.agency.office_or_region = FieldWithStatus[str](
        value=None,
        status="deferred_v1",
        provenance=None,
    )

    return warnings


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

def _populate_title(
    record: EISRecord,
    nul: dict[str, Any],
    cover_text: str,
    cover_span: tuple[int, int] | None,
    llm: "LLMClient | None",
    warnings: list[str],
) -> None:
    nul_title = nul.get("title")
    if nul_title:
        record.title = FieldWithStatus[str](
            value=str(nul_title),
            status="extracted_from_mets",
            provenance=Provenance(source="nul_api"),
            confidence=0.99,
        )
        logger.info("Stage 3a: title from NUL — %r", nul_title)
        return

    # NUL had no title — try Haiku gap-fill on cover.
    if llm is None:
        record.title = FieldWithStatus[str](value=None, status="needs_review", provenance=None)
        warnings.append("title: NUL had no title; no llm provided — left as needs_review")
        return

    gapfill_title = _haiku_title_gapfill(cover_text, llm)
    if gapfill_title:
        prov_kwargs: dict[str, Any] = {"source": "haiku_gapfill"}
        if cover_span is not None:
            prov_kwargs["section"] = "cover"
            prov_kwargs["char_offset_raw"] = cover_span
        record.title = FieldWithStatus[str](
            value=gapfill_title,
            status="needs_review",  # Haiku-extracted titles always cross-check
            provenance=Provenance(**prov_kwargs),
            confidence=0.75,
        )
        logger.info("Stage 3a: title from Haiku gap-fill — %r", gapfill_title)
    else:
        record.title = FieldWithStatus[str](value=None, status="needs_review")
        warnings.append("title: NUL empty and Haiku gap-fill returned no title")


# ---------------------------------------------------------------------------
# Year + date.publication
# ---------------------------------------------------------------------------

def _populate_year_and_date(
    record: EISRecord,
    nul: dict[str, Any],
    ingest: IngestArtifact,
    warnings: list[str],
) -> None:
    nul_date = nul.get("date_created")
    pub_year, pub_value, pub_precision = _parse_publication_date(nul_date)

    if pub_year is not None:
        record.date.publication = PublicationDate(
            value=pub_value,
            precision=pub_precision,
            status="ok",
            provenance=Provenance(source="nul_api"),
        )
        record.year = FieldWithStatus[int](
            value=pub_year,
            status="extracted_from_mets",
            provenance=Provenance(source="nul_api"),
            confidence=0.98,
        )
        logger.info("Stage 3a: year=%d (precision=%s) from NUL", pub_year, pub_precision)
    else:
        # Fallback: scan first 5 fake pages for the most-frequent valid year.
        fallback_year = _extract_year_fallback(ingest)
        if fallback_year is not None:
            record.year = FieldWithStatus[int](
                value=fallback_year,
                status="needs_review",
                provenance=Provenance(source="regex", note="year regex on first 5 fake pages"),
                confidence=0.80,
            )
            record.date.publication = PublicationDate(
                value=str(fallback_year),
                precision="year",
                status="needs_review",
                provenance=Provenance(source="regex"),
            )
            logger.info("Stage 3a: fallback year=%d from raw text", fallback_year)
            warnings.append(
                f"year: NUL had no parseable date; extracted {fallback_year} from raw text"
            )
        else:
            record.year = FieldWithStatus[int](value=None, status="needs_review")
            record.date.publication = PublicationDate(value=None, status="needs_review")
            warnings.append("year: NUL had no parseable date and regex fallback failed")

    # Year range gate (NEPA_YEAR..MAX_YEAR). Failures don't blank the value;
    # they downgrade status so Stage 4 catches them.
    if record.year.value is not None:
        if not (config.NEPA_YEAR <= record.year.value <= config.MAX_YEAR):
            record.year.status = "needs_review"
            warnings.append(
                f"year: {record.year.value} outside [{config.NEPA_YEAR}, {config.MAX_YEAR}]"
            )


# ---------------------------------------------------------------------------
# Lead agency
# ---------------------------------------------------------------------------

def _populate_lead_agency(
    record: EISRecord,
    nul: dict[str, Any],
    cover_text: str,
    cover_span: tuple[int, int] | None,
    llm: "LLMClient | None",
    warnings: list[str],
) -> None:
    creator = nul.get("creator") or []
    contributor = nul.get("contributor") or []
    matched_abbr, matched_label = _match_agency_in_strings(creator + contributor)

    if matched_abbr is not None:
        record.agency.lead_agency = FieldWithStatus[str](
            value=matched_abbr,
            status="extracted_from_mets",
            provenance=Provenance(source="nul_api", note=f"matched {matched_label!r}"),
            confidence=0.99,
        )
        logger.info("Stage 3a: lead_agency=%s (from NUL %r)", matched_abbr, matched_label)
        return

    # NUL had no usable creator — try cover-text vocab scan, then Haiku.
    cover_abbr, cover_label, cover_offset = _scan_cover_for_agency(cover_text)
    if cover_abbr is not None:
        prov_kwargs: dict[str, Any] = {
            "source": "controlled_vocab_match",
            "note": f"matched {cover_label!r} in cover",
        }
        if cover_span is not None and cover_offset is not None:
            # cover_offset is relative to cover_text; shift to raw_text coords.
            absolute = cover_span[0] + cover_offset
            prov_kwargs["section"] = "cover"
            prov_kwargs["char_offset_raw"] = (
                absolute, absolute + len(cover_label or ""),
            )
        record.agency.lead_agency = FieldWithStatus[str](
            value=cover_abbr,
            status="needs_review",
            provenance=Provenance(**prov_kwargs),
            confidence=0.85,
        )
        logger.info(
            "Stage 3a: lead_agency=%s via cover vocab scan (matched %r)",
            cover_abbr, cover_label,
        )
        return

    if llm is None:
        # NUL had something but didn't match controlled vocab; or had nothing
        # at all. Without an LLM, fall back to whatever NUL gave us, marked
        # for review.
        if creator or contributor:
            first = (creator or contributor)[0]
            record.agency.lead_agency = FieldWithStatus[str](
                value=str(first),
                status="needs_review",
                provenance=Provenance(source="nul_api", note="unmatched against AGENCY_VOCAB"),
                confidence=0.60,
            )
            warnings.append(
                f"lead_agency: NUL gave {first!r}, not in controlled vocab; "
                f"no llm provided to gap-fill"
            )
        else:
            record.agency.lead_agency = FieldWithStatus[str](
                value=None, status="needs_review", provenance=None,
            )
            warnings.append("lead_agency: NUL empty; no llm to gap-fill from cover")
        return

    # Haiku gap-fill on cover.
    gapfill_agency = _haiku_agency_gapfill(cover_text, llm)
    if gapfill_agency:
        canonical = config.lookup_agency(gapfill_agency)
        prov_kwargs = {"source": "haiku_gapfill", "note": f"Haiku returned {gapfill_agency!r}"}
        if cover_span is not None:
            prov_kwargs["section"] = "cover"
            prov_kwargs["char_offset_raw"] = cover_span
        if canonical:
            record.agency.lead_agency = FieldWithStatus[str](
                value=canonical,
                status="needs_review",
                provenance=Provenance(**prov_kwargs),
                confidence=0.70,
            )
            logger.info(
                "Stage 3a: lead_agency=%s via Haiku gap-fill (canonical of %r)",
                canonical, gapfill_agency,
            )
        else:
            # Haiku returned a name that didn't canonicalize; keep raw, low confidence.
            record.agency.lead_agency = FieldWithStatus[str](
                value=gapfill_agency,
                status="needs_review",
                provenance=Provenance(**prov_kwargs),
                confidence=0.45,
            )
            warnings.append(
                f"lead_agency: Haiku returned {gapfill_agency!r} but it did not "
                f"canonicalize against AGENCY_VOCAB"
            )
    else:
        record.agency.lead_agency = FieldWithStatus[str](value=None, status="needs_review")
        warnings.append("lead_agency: NUL empty, cover scan empty, Haiku returned nothing")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cover_span(sections_artifact: "SectionsArtifact | None") -> tuple[int, int] | None:
    if sections_artifact is None:
        return None
    by_name = sections_artifact.by_name() if hasattr(sections_artifact, "by_name") else {}
    cover = by_name.get("cover")
    if cover is None or cover.char_span is None:
        return None
    return cover.char_span


def _parse_publication_date(value: Any) -> tuple[int | None, str | None, str | None]:
    """Parse NUL's date_created. Returns (year, full_value, precision).
    NUL typically returns just a year string ("2017"), occasionally a full date."""
    if value is None:
        return None, None, None
    s = str(value).strip()
    if not s:
        return None, None, None

    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if iso_match:
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return int(iso_match.group(1)), s, "day"
        except ValueError:
            pass

    ym_match = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if ym_match:
        return int(ym_match.group(1)), s, "month"

    year_match = _YEAR_PATTERN.search(s)
    if year_match:
        return int(year_match.group(1)), year_match.group(1), "year"

    return None, None, None


def _extract_year_fallback(ingest: IngestArtifact) -> int | None:
    """Most-frequent valid year in first 5 fake pages."""
    if not ingest.pages:
        return None

    end_char = ingest.pages[min(4, len(ingest.pages) - 1)].char_end_raw
    text = ingest.raw_text[:end_char]

    counts: dict[int, int] = {}
    for m in _YEAR_PATTERN.finditer(text):
        y = int(m.group(1))
        if config.NEPA_YEAR <= y <= config.MAX_YEAR:
            counts[y] = counts.get(y, 0) + 1

    if not counts:
        return None
    return max(counts, key=lambda y: (counts[y], y))


def _match_agency_in_strings(candidates: list[str]) -> tuple[str | None, str | None]:
    """Match any string in `candidates` against AGENCY_VOCAB. Direct hit first,
    then substring scan against any vocab variant."""
    for label in candidates:
        if not label:
            continue
        s = str(label).strip()
        abbr = config.lookup_agency(s)
        if abbr:
            return abbr, s
        s_lower = s.lower()
        for variant_lower, abbr_match in config.AGENCY_FLAT.items():
            if variant_lower in s_lower:
                return abbr_match, s
    return None, None


def _scan_cover_for_agency(cover_text: str) -> tuple[str | None, str | None, int | None]:
    """Find the first AGENCY_VOCAB variant occurrence in cover_text.
    Returns (abbr, matched_substring, offset_in_cover) or (None, None, None).

    Iterates variants longest-first so 'United States Forest Service' matches
    before the bare 'Forest Service' substring inside it.
    """
    if not cover_text:
        return None, None, None
    cover_lower = cover_text.lower()
    variants_sorted = sorted(config.AGENCY_FLAT.items(), key=lambda kv: -len(kv[0]))
    for variant_lower, abbr in variants_sorted:
        # Skip 1-3 char abbreviations to avoid false positives ('DOE' inside
        # words, 'EPA' as initials in unrelated names). Rely on canonical names
        # + multi-word variants for cover-text matching.
        if len(variant_lower) <= 3:
            continue
        idx = cover_lower.find(variant_lower)
        if idx != -1:
            return abbr, cover_text[idx:idx + len(variant_lower)], idx
    return None, None, None


# ---------------------------------------------------------------------------
# Haiku gap-fill calls
# ---------------------------------------------------------------------------

def _haiku_title_gapfill(cover_text: str, llm: "LLMClient") -> str | None:
    prompt_template = (PROMPTS_DIR / "3a_title_gapfill.txt").read_text(encoding="utf-8")
    prompt = prompt_template.replace("{cover_text}", cover_text[:_COVER_TEXT_CAP])
    try:
        result = llm.call_json(
            model=llm.models["haiku"],
            system="You are a careful extractor. Return only the requested JSON object.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
            label="3a_title_gapfill",
        )
    except Exception as exc:
        logger.warning("3a title gap-fill LLM call failed: %s", exc)
        return None

    if not isinstance(result, dict):
        return None
    if not result.get("found"):
        return None
    title = result.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    return title.strip()


def _haiku_agency_gapfill(cover_text: str, llm: "LLMClient") -> str | None:
    prompt_template = (PROMPTS_DIR / "3a_agency_gapfill.txt").read_text(encoding="utf-8")
    prompt = prompt_template.replace("{cover_text}", cover_text[:_COVER_TEXT_CAP])
    try:
        result = llm.call_json(
            model=llm.models["haiku"],
            system="You are a careful classifier. Return only the requested JSON object.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
            label="3a_agency_gapfill",
        )
    except Exception as exc:
        logger.warning("3a agency gap-fill LLM call failed: %s", exc)
        return None

    if not isinstance(result, dict):
        return None
    if not result.get("found"):
        return None
    agency = result.get("agency")
    if not isinstance(agency, str) or not agency.strip():
        return None
    return agency.strip()
