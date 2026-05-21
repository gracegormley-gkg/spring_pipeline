"""
Stage 2 — Section detection.

Two-pass strategy per build brief §6 Stage 2:
  1. Regex pass over the CEQ taxonomy.
  2. Embedding-similarity fallback (sentence-transformers/all-MiniLM-L6-v2)
     for any required-for-downstream section the regex missed.

Reject patterns filter out matches inside legal citations ('Section 4(f)'),
ZIP-code lines, and street addresses.

The cover section is page-based: defaults to pages 1–3 if at least 3 pages exist.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .ingest import FakePage, IngestArtifact, page_range_for_span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

@dataclass
class DetectedSection:
    name: str
    char_span: tuple[int, int] | None = None
    pages: tuple[int, int] | None = None
    confidence: float = 1.0
    status: str = "ok"   # ok | not_found | needs_review | ambiguous
    detection_method: str = "regex"  # regex | embedding_fallback | default_pages

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "char_span": list(self.char_span) if self.char_span else None,
            "pages": list(self.pages) if self.pages else None,
            "confidence": self.confidence,
            "status": self.status,
            "detection_method": self.detection_method,
        }


@dataclass
class SectionsArtifact:
    publication_id: str
    sections: list[DetectedSection] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "publication_id": self.publication_id,
            "sections": [s.to_json() for s in self.sections],
        }

    def by_name(self) -> dict[str, DetectedSection]:
        return {s.name: s for s in self.sections}


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run(ingest: IngestArtifact, *, allow_embedding_fallback: bool = True) -> SectionsArtifact:
    """Detect CEQ sections in the raw text. Pass-1 regex, pass-2 embedding fallback."""
    raw = ingest.raw_text
    pages = ingest.pages

    detected: dict[str, DetectedSection] = {}

    # Cover: default to first 3 pages
    detected["cover"] = _detect_cover(pages)

    # Regex pass for the rest
    for section_name in config.CEQ_SECTIONS:
        if section_name == "cover":
            continue
        sec = _regex_detect(section_name, raw, pages)
        if sec is not None:
            detected[section_name] = sec

    found_names = set(detected.keys())
    logger.info("Stage 2 regex pass: found %s", sorted(found_names))

    # Embedding fallback for any *required* section that's missing
    missing_required = []
    for section_name, required_by in config.SECTION_REQUIRED_BY.items():
        if section_name not in found_names:
            missing_required.append(section_name)

    if missing_required and allow_embedding_fallback:
        logger.info(
            "Stage 2 embedding fallback for missing required sections: %s",
            missing_required,
        )
        try:
            from .sections_embedding import embedding_detect  # lazy import
            embed_results = embedding_detect(missing_required, raw, pages)
            for sec in embed_results:
                detected[sec.name] = sec
        except Exception as exc:
            logger.warning(
                "Embedding fallback unavailable (%s); marking missing sections not_found",
                exc,
            )
    elif missing_required:
        logger.info("Skipping embedding fallback (allow_embedding_fallback=False)")

    # Fill in not_found stubs for any CEQ section we have nothing on (so the
    # output always lists all 9 sections with explicit status).
    for section_name in config.CEQ_SECTIONS:
        if section_name not in detected:
            detected[section_name] = DetectedSection(
                name=section_name,
                status="not_found",
                detection_method="regex",
            )

    # Assemble in canonical order
    artifact = SectionsArtifact(
        publication_id=ingest.publication_id,
        sections=[detected[name] for name in config.CEQ_SECTIONS],
    )
    return artifact


def write_artifact(artifact: SectionsArtifact, output_dir: str | Path) -> Path:
    path = Path(output_dir) / f"{artifact.publication_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_json(), indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Regex pass
# ---------------------------------------------------------------------------

def _detect_cover(pages: list[FakePage]) -> DetectedSection:
    """Cover = pages 1..min(3, len(pages))."""
    if not pages:
        return DetectedSection(name="cover", status="not_found")
    end_page = min(3, len(pages))
    end_char = pages[end_page - 1].char_end_raw
    return DetectedSection(
        name="cover",
        char_span=(0, end_char),
        pages=(1, end_page),
        confidence=1.0,
        status="ok",
        detection_method="default_pages",
    )


def _regex_detect(
    section_name: str,
    raw: str,
    pages: list[FakePage],
) -> DetectedSection | None:
    """
    Find the first valid regex hit for `section_name`.
    Section spans from the match start to the start of the next-detected
    section (computed in a second post-pass below) — but here we just record
    the heading position. Span resolution is finalized after all sections are
    found, in `_finalize_spans`.
    """
    patterns = config.SECTION_PATTERNS.get(section_name, [])
    for pat in patterns:
        for m in pat.finditer(raw):
            if _is_rejected_match(m, raw):
                continue
            heading_start = m.start()
            page_range = page_range_for_span(pages, (heading_start, heading_start + 1))
            return DetectedSection(
                name=section_name,
                char_span=(heading_start, heading_start + len(m.group(0))),
                pages=page_range,
                confidence=0.9,
                status="ok",
                detection_method="regex",
            )
    return None


def _is_rejected_match(m: re.Match, raw: str) -> bool:
    """Reject matches that look like ZIP codes, legal citations, or addresses."""
    # Look at the surrounding ~120 chars on each side
    start = max(0, m.start() - 60)
    end = min(len(raw), m.end() + 60)
    window = raw[start:end]
    for rej in config.SECTION_REJECT_PATTERNS:
        if rej.search(window):
            return True
    return False
