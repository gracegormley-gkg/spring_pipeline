"""
Stage 3a — METS-equivalent fields from NUL.

Populates: title, year, lead_agency, agency.office_or_region (deferred_v1),
date.publication. Lead agency is matched against the controlled vocab in
config.AGENCY_VOCAB; for the v1 corpus filter (USFS Final ≥2000) we expect
USFS — anything else is a manifest filter bug and gets a warning.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from . import config
from .ingest import IngestArtifact
from .schema import (
    EISRecord,
    FieldWithStatus,
    PublicationDate,
    Provenance,
)

logger = logging.getLogger(__name__)

# Year regexes (used only if NUL date_created can't be parsed).
_YEAR_PATTERNS = [
    re.compile(r"\b(19[7-9]\d|20\d{2})\b"),  # 1970-2099 — covers NEPA era forward
]


def run(record: EISRecord, ingest: IngestArtifact, sections_artifact: Any | None = None) -> list[str]:
    """
    Populate NUL-sourced fields on `record`. Mutates the record in place.
    Returns a list of warnings.
    """
    warnings: list[str] = []
    nul = ingest.nul_metadata or {}

    # --- title ---
    nul_title = nul.get("title")
    if nul_title:
        record.title = FieldWithStatus[str](
            value=str(nul_title),
            status="ok",
            provenance=Provenance(source="nul_api"),
            confidence=0.99,
        )
        logger.info("Stage 3a: title from NUL — %r", nul_title)
    else:
        record.title = FieldWithStatus[str](
            value=None,
            status="needs_review",
            provenance=None,
        )
        warnings.append("title: NUL had no title; extractive fallback not implemented in 3a")

    # --- date.publication + year ---
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
            status="ok",
            provenance=Provenance(source="nul_api"),
            confidence=0.98,
        )
        logger.info("Stage 3a: year=%d (precision=%s) from NUL", pub_year, pub_precision)
    else:
        # Extractive fallback: scan first 5 fake pages for a plausible year
        fallback_year = _extract_year_fallback(ingest)
        if fallback_year is not None:
            record.year = FieldWithStatus[int](
                value=fallback_year,
                status="needs_review",
                provenance=Provenance(source="regex", note="extracted from first 5 pages"),
                confidence=0.80,
            )
            record.date.publication = PublicationDate(
                value=str(fallback_year),
                precision="year",
                status="needs_review",
                provenance=Provenance(source="regex"),
            )
            logger.info("Stage 3a: fallback year=%d from raw text", fallback_year)
            warnings.append(f"year: NUL had no parseable date; extracted {fallback_year} from raw text")
        else:
            record.year = FieldWithStatus[int](value=None, status="needs_review")
            record.date.publication = PublicationDate(value=None, status="needs_review")
            warnings.append("year: NUL had no parseable date and regex fallback failed")

    # Year range gate
    if record.year.value is not None:
        if not (config.NEPA_YEAR <= record.year.value <= config.MAX_YEAR):
            record.year.status = "needs_review"
            warnings.append(
                f"year: {record.year.value} outside [{config.NEPA_YEAR}, {config.MAX_YEAR}]"
            )

    # --- lead_agency ---
    creator = nul.get("creator") or []
    contributor = nul.get("contributor") or []
    matched_abbr, matched_label = _match_agency(creator + contributor)

    if matched_abbr is not None:
        record.agency.lead_agency = FieldWithStatus[str](
            value=matched_abbr,
            status="extracted_from_mets",
            provenance=Provenance(source="nul_api", note=f"matched {matched_label!r}"),
            confidence=0.99,
        )
        logger.info("Stage 3a: lead_agency=%s (from %r)", matched_abbr, matched_label)
        if matched_abbr != "USFS":
            warnings.append(
                f"lead_agency: {matched_abbr} — not USFS; manifest filter may be misconfigured"
            )
    elif creator or contributor:
        # Took a creator but didn't match the controlled vocab
        first = (creator or contributor)[0]
        record.agency.lead_agency = FieldWithStatus[str](
            value=str(first),
            status="needs_review",
            provenance=Provenance(source="nul_api", note="unmatched against AGENCY_VOCAB"),
            confidence=0.60,
        )
        warnings.append(f"lead_agency: NUL gave {first!r}, not in controlled vocab")
    else:
        record.agency.lead_agency = FieldWithStatus[str](
            value=None,
            status="needs_review",
            provenance=None,
        )
        warnings.append("lead_agency: NUL had no creator/contributor; field empty")

    # --- agency.office_or_region: deferred_v1 ---
    record.agency.office_or_region = FieldWithStatus[str](
        value=None,
        status="deferred_v1",
        provenance=None,
    )

    return warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_publication_date(value: Any) -> tuple[int | None, str | None, str | None]:
    """
    Parse NUL's date_created. Returns (year, full_value, precision).
    NUL typically returns just a year string ("2017"), occasionally a full date.
    """
    if value is None:
        return None, None, None
    s = str(value).strip()
    if not s:
        return None, None, None

    # ISO date: YYYY-MM-DD
    iso_match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if iso_match:
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return int(iso_match.group(1)), s, "day"
        except ValueError:
            pass

    # Year-month: YYYY-MM
    ym_match = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if ym_match:
        return int(ym_match.group(1)), s, "month"

    # Bare year, possibly inside a string with extra junk
    year_match = re.search(r"\b(19[7-9]\d|20\d{2})\b", s)
    if year_match:
        return int(year_match.group(1)), year_match.group(1), "year"

    return None, None, None


def _extract_year_fallback(ingest: IngestArtifact) -> int | None:
    """Scan first 5 fake pages for the most-frequent valid year."""
    if not ingest.pages:
        return None

    end_char = ingest.pages[min(4, len(ingest.pages) - 1)].char_end_raw
    text = ingest.raw_text[:end_char]

    counts: dict[int, int] = {}
    for pat in _YEAR_PATTERNS:
        for m in pat.finditer(text):
            y = int(m.group(1))
            if config.NEPA_YEAR <= y <= config.MAX_YEAR:
                counts[y] = counts.get(y, 0) + 1

    if not counts:
        return None
    return max(counts, key=lambda y: (counts[y], y))


def _match_agency(candidates: list[str]) -> tuple[str | None, str | None]:
    """
    Match NUL creator/contributor strings against AGENCY_VOCAB.
    Returns (abbreviation, original_label) or (None, None).
    """
    for label in candidates:
        if not label:
            continue
        s = str(label).strip()
        # Direct lookup first
        abbr = config.lookup_agency(s)
        if abbr:
            return abbr, s
        # Substring scan: if any vocab variant appears within the label, accept it
        s_lower = s.lower()
        for variant_lower, abbr_match in config.AGENCY_FLAT.items():
            if variant_lower in s_lower:
                return abbr_match, s
    return None, None
