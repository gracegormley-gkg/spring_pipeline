"""
Stage 2 — Section detection cascade.

Order (per synthesis_plan.md §Section detection):
  1. Cover  — default to first ~3 fake pages (deterministic; always populated).
  2. Regex  — scan text_normalized for CEQ section headings using
              config.SECTION_PATTERNS, with config.SECTION_REJECT_PATTERNS
              filtering ZIP-code / legal-citation / address false positives.
  3. AI-TOC — Haiku reads beginning/middle/end samples (~15k chars total),
              returns titles + verbatim anchor phrases; we string-search to
              locate each anchor and map to a CEQ section name. Salvages
              typewritten 1970s docs whose headings don't match the regex.
              (Lazy import; no-op if llm_client is None.)
  4. Embedding — sentence-transformers/all-MiniLM-L6-v2 against canonical
              section descriptors; last-resort for required-by-downstream
              sections still missing. (Lazy import; no-op if package absent.)
  5. Stub `not_found` for any CEQ section nothing has produced.

Sections are a precision booster, NOT a hard gate (synthesis_plan §Key design
decision 0). When a target section isn't found, downstream Stage 3 fields
fall back to per-field retrieval (keyword search → first N chunks) and mark
provenance with `source: "fallback_keyword_search"`. The pipeline never goes
silent on a missing section; it just downgrades context confidence.

The DetectedSection dataclass mirrors schema.SectionRecord but stays a
dataclass at this stage to avoid Pydantic-validator coupling inside the
detection loop. Conversion to SectionRecord happens in Stage 5 (output writer).

Ported from v1_multiagent_pipeline/pipeline/sections.py with the AI-TOC pass
inserted between regex and embedding (per synthesis_plan §Section detection).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import config
from .ingest import FakePage, IngestArtifact, page_range_for_span

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

@dataclass
class DetectedSection:
    """Stage-2 dataclass; converts to schema.SectionRecord at Stage 5.

    `name` should always be a value from config.CEQ_SECTIONS (matches the
    schema's SectionName Literal). Kept as `str` here so the dataclass
    doesn't carry a forward-ref to the Pydantic Literal.

    `detection_method` should be one of: regex, ai_toc, embedding_fallback,
    default_pages, manual.
    """
    name: str
    char_span: tuple[int, int] | None = None
    pages: tuple[int, int] | None = None
    confidence: float = 1.0
    status: str = "ok"   # ok | not_found | needs_review | ambiguous
    detection_method: str = "regex"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "char_span": list(self.char_span) if self.char_span else None,
            "pages": list(self.pages) if self.pages else None,
            "confidence": self.confidence,
            "status": self.status,
            "detection_method": self.detection_method,
        }

    def to_schema_record(self):
        """Convert to a schema.SectionRecord (Pydantic model). Used at Stage 5
        to populate EISRecord.sections; also used by Stage 3a/3b orchestration
        to seed the record's sections list before fields with section-aware
        provenance are written (cross-field validators check that
        provenance.section refers to a name in record.sections)."""
        from .schema import SectionRecord  # local import to avoid cycle
        return SectionRecord(
            name=self.name,  # type: ignore[arg-type]  # Literal[SectionName] enforced by schema
            char_span=self.char_span,
            pages=self.pages,
            confidence=self.confidence,
            status=self.status,  # type: ignore[arg-type]
            detection_method=self.detection_method,  # type: ignore[arg-type]
        )


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

def run(
    ingest: IngestArtifact,
    *,
    llm_client: "LLMClient | None" = None,
    allow_ai_toc: bool = True,
    allow_embedding_fallback: bool = True,
) -> SectionsArtifact:
    """
    Run the cascade. Each fallback only fires for sections in
    config.SECTION_REQUIRED_BY that the prior pass didn't find — keeps cost
    bounded and avoids hallucinating non-required sections.

    Args:
      ingest:                   Stage 1 artifact (raw_text + pages).
      llm_client:               If None, AI-TOC silently skipped.
      allow_ai_toc:             Override to disable AI-TOC even when client present
                                (useful for pure-deterministic runs / tests).
      allow_embedding_fallback: Override to disable embedding fallback (useful
                                in environments without sentence-transformers).
    """
    raw = ingest.raw_text
    pages = ingest.pages

    detected: dict[str, DetectedSection] = {}

    # 1. Cover: default to first 3 pages (always populated).
    detected["cover"] = _detect_cover(pages)

    # 2. Regex pass for the rest.
    for section_name in config.CEQ_SECTIONS:
        if section_name == "cover":
            continue
        sec = _regex_detect(section_name, raw, pages)
        if sec is not None:
            detected[section_name] = sec

    found_names = set(detected.keys())
    logger.info("Stage 2 regex pass: found %s", sorted(found_names))

    # Identify required-by-downstream sections still missing.
    def _missing_required() -> list[str]:
        return [s for s in config.SECTION_REQUIRED_BY if s not in detected]

    # 3. AI-TOC pass (if LLM available + enabled).
    missing = _missing_required()
    if missing and allow_ai_toc and llm_client is not None:
        logger.info("Stage 2 AI-TOC pass for missing required sections: %s", missing)
        try:
            from .sections_ai_toc import ai_toc_detect  # lazy import
            ai_results = ai_toc_detect(missing, ingest, llm_client)
            for sec in ai_results:
                detected[sec.name] = sec
        except Exception as exc:
            # AI-TOC is a soft helper: never let it kill the pipeline.
            logger.warning(
                "AI-TOC pass failed (%s); continuing to embedding fallback", exc,
            )
    elif missing and allow_ai_toc and llm_client is None:
        logger.info(
            "Stage 2: AI-TOC skipped (no llm_client provided) — sections still "
            "missing: %s", missing,
        )

    # 4. Embedding fallback for any required section still missing.
    missing = _missing_required()
    if missing and allow_embedding_fallback:
        logger.info("Stage 2 embedding fallback for missing required sections: %s", missing)
        try:
            from .sections_embedding import embedding_detect  # lazy import
            embed_results = embedding_detect(missing, raw, pages)
            for sec in embed_results:
                detected[sec.name] = sec
        except Exception as exc:
            logger.warning(
                "Embedding fallback unavailable (%s); marking missing sections not_found",
                exc,
            )
    elif missing:
        logger.info("Skipping embedding fallback (allow_embedding_fallback=False)")

    # 5. Stub not_found for any CEQ section we have nothing on.
    for section_name in config.CEQ_SECTIONS:
        if section_name not in detected:
            detected[section_name] = DetectedSection(
                name=section_name,
                status="not_found",
                detection_method="regex",
            )

    # Assemble in canonical order.
    return SectionsArtifact(
        publication_id=ingest.publication_id,
        sections=[detected[name] for name in config.CEQ_SECTIONS],
    )


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
        return DetectedSection(name="cover", status="not_found", detection_method="default_pages")
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
    """First valid regex hit for `section_name`; rejects address/citation context."""
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


# ---------------------------------------------------------------------------
# CEQ-name normalizer (used by AI-TOC pass to map LLM-returned titles)
# ---------------------------------------------------------------------------

def match_title_to_ceq(title: str) -> str | None:
    """Map a free-form section title to a CEQ section name.

    Strategy:
      1. Run each CEQ regex against the title — if it matches as a heading,
         return that CEQ name.
      2. Heuristic keyword lookup as a backup (handles cases where the AI-TOC
         returns 'Comments and Responses' without the regex's $-anchored shape).

    Returns None if nothing maps cleanly. AI-TOC drops un-mappable titles.
    """
    if not title:
        return None
    t = title.strip()

    # Pass 1: try the canonical regex against the bare title.
    for section_name, patterns in config.SECTION_PATTERNS.items():
        for pat in patterns:
            if pat.search(t):
                return section_name

    # Pass 2: relaxed keyword match (case-insensitive; matches anywhere in the
    # title, with no $ anchor). Ordered roughly most-specific-first.
    tl = t.lower()
    keyword_map: list[tuple[str, list[str]]] = [
        ("response_to_comments",      ["response to comment", "comments and response", "agency response"]),
        ("public_comments",           ["public comment", "public involvement", "public participation",
                                       "comments on the draft", "letters received"]),
        ("environmental_consequences",["environmental consequence", "environmental effect",
                                       "environmental impact", "impact analysis"]),
        ("affected_environment",      ["affected environment", "existing condition", "existing environment"]),
        ("alternatives",              ["alternative"]),
        ("purpose_and_need",          ["purpose and need", "purpose & need", "need for the action",
                                       "need for action", "need for the project"]),
        ("summary",                   ["summary", "abstract"]),
        ("rod",                       ["record of decision"]),
    ]
    for section_name, keywords in keyword_map:
        for kw in keywords:
            if kw in tl:
                return section_name

    return None
