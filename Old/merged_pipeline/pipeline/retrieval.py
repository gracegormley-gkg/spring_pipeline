"""
Stage 3 retrieval cascade.

Per synthesis_plan.md §Key design decision 0: sections are a precision booster,
not a hard gate. When a target section isn't found, every Stage 3 field falls
back through:
  1. Target section text (highest precision; provenance.source from the section's
     detection_method on the originating SectionRecord).
  2. Keyword search across the full normalized text (`fallback_keyword_search`).
  3. First-N usable chars from the start of the doc (`fallback_first_n_chunks`).

Each fallback downgrades the Provenance.source marker so downstream consumers
(Stage 4 critic, output writer) know the context was weaker — but the field
still produces a value when evidence exists anywhere in the doc. This is the
"never go silent" contract.

Both keyword search and first-N return RetrievalWindow lists so callers can
optionally feed multiple distinct windows to an LLM (prevents one big keyword
hit from drowning out the others).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .ingest import IngestArtifact

if TYPE_CHECKING:
    from .sections import SectionsArtifact

logger = logging.getLogger(__name__)


# Locked literals from schema.ProvenanceSource. We don't import the schema's
# Literal directly (avoid coupling) but the strings here must match it.
SOURCE_FALLBACK_KEYWORD = "fallback_keyword_search"
SOURCE_FALLBACK_FIRST_N = "fallback_first_n_chunks"

# Section detection methods that map cleanly to a ProvenanceSource. When the
# section was found by an LLM-assisted pass we still mark provenance against
# the originating method (regex / ai_toc / embedding_fallback / default_pages)
# rather than treating section retrieval as a fallback.
_SECTION_TO_PROVENANCE_SOURCE = {
    "regex":              "regex",
    "ai_toc":             "haiku_classifier",      # AI-TOC anchor was Haiku-derived
    "embedding_fallback": "embedding_fallback",
    "default_pages":      "regex",                 # cover; deterministic, regex-equivalent
    "manual":             "regex",                 # manual annotation -> treat as deterministic
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RetrievalWindow:
    """A single span of text plus its raw-text offset."""
    text: str
    char_offset_raw: tuple[int, int]
    source_label: str = ""  # human-readable hint for logging (not provenance)

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class RetrievalResult:
    """The combined output of a retrieval cascade.

    `windows` are ordered most-to-least preferred. `provenance_source` is the
    schema.ProvenanceSource string Stage 3 should stamp on its field. `section`
    is the schema.SectionName the text came from (None if fallback).

    `degraded` is True iff retrieval did not find the requested section AND
    fell through to keyword search or first-N. Callers can use this to set
    field status to "needs_review" automatically, or to add a status note.
    """
    windows: list[RetrievalWindow]
    provenance_source: str
    section: str | None
    degraded: bool

    @property
    def combined_text(self) -> str:
        return "\n\n--- WINDOW BOUNDARY ---\n\n".join(w.text for w in self.windows)

    @property
    def total_chars(self) -> int:
        return sum(len(w) for w in self.windows)

    def union_offset(self) -> tuple[int, int] | None:
        """Min-start to max-end across all windows; useful for char_offset_raw
        on a single Provenance object pointing at the retrieval region."""
        if not self.windows:
            return None
        starts = [w.char_offset_raw[0] for w in self.windows]
        ends = [w.char_offset_raw[1] for w in self.windows]
        return (min(starts), max(ends))


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def get_section_text(
    ingest: IngestArtifact,
    sections_artifact: "SectionsArtifact",
    section_name: str,
) -> RetrievalWindow | None:
    """Return text+offset for a single named section, or None if not detected."""
    by_name = sections_artifact.by_name() if hasattr(sections_artifact, "by_name") else {}
    sec = by_name.get(section_name)
    if sec is None or sec.char_span is None or sec.status != "ok":
        return None
    s, e = sec.char_span
    return RetrievalWindow(
        text=ingest.raw_text[s:e],
        char_offset_raw=(s, e),
        source_label=f"section:{section_name}",
    )


def get_keyword_windows(
    ingest: IngestArtifact,
    keywords: list[str],
    *,
    max_windows: int = 8,
    window_chars: int = 2000,
    case_insensitive: bool = True,
) -> list[RetrievalWindow]:
    """Find keyword hits in raw_text and return surrounding windows.

    Windows are coalesced when they overlap (so a paragraph with several
    keyword hits becomes one window, not three).
    """
    if not keywords:
        return []
    flags = re.IGNORECASE if case_insensitive else 0
    # Word-boundary anchored alternation for cheap multi-keyword scan.
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b", flags,
    )

    half = window_chars // 2
    raw = ingest.raw_text
    n = len(raw)

    raw_windows: list[tuple[int, int]] = []
    for m in pattern.finditer(raw):
        s = max(0, m.start() - half)
        e = min(n, m.end() + half)
        raw_windows.append((s, e))
        if len(raw_windows) > max_windows * 4:  # generous cap before coalescing
            break

    if not raw_windows:
        return []

    # Coalesce overlapping windows
    raw_windows.sort()
    merged: list[tuple[int, int]] = [raw_windows[0]]
    for s, e in raw_windows[1:]:
        ms, me = merged[-1]
        if s <= me:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))

    out: list[RetrievalWindow] = []
    for s, e in merged[:max_windows]:
        out.append(RetrievalWindow(
            text=raw[s:e],
            char_offset_raw=(s, e),
            source_label=f"keyword_window[{s}:{e}]",
        ))
    return out


def get_first_n_chunks(
    ingest: IngestArtifact,
    *,
    max_chars: int = 10_000,
) -> list[RetrievalWindow]:
    """Return the first `max_chars` of the raw text as a single window.
    Used as last-resort fallback when section + keyword search both miss."""
    n = min(len(ingest.raw_text), max_chars)
    if n <= 0:
        return []
    return [RetrievalWindow(
        text=ingest.raw_text[:n],
        char_offset_raw=(0, n),
        source_label="first_n_chunks",
    )]


# ---------------------------------------------------------------------------
# Cascade orchestrator
# ---------------------------------------------------------------------------

def cascade(
    ingest: IngestArtifact,
    sections_artifact: "SectionsArtifact",
    *,
    primary_sections: list[str],
    fallback_keywords: list[str],
    max_keyword_windows: int = 6,
    keyword_window_chars: int = 2000,
    first_n_max_chars: int = 10_000,
) -> RetrievalResult:
    """The retrieval cascade used by every Stage 3 field producer.

    Tries each section in `primary_sections` (in order). On the first hit,
    returns a single-window result with provenance.source mapped from the
    section's detection_method.

    If no primary section is detected `ok`, falls back to keyword search.
    If keyword search returns nothing, falls back to first-N chunks.

    Always returns a RetrievalResult with at least one window unless the doc
    is literally empty (in which case windows=[] and degraded=True).
    """
    # Pass 1: primary sections (first match wins)
    by_name = sections_artifact.by_name() if hasattr(sections_artifact, "by_name") else {}
    for section_name in primary_sections:
        sec = by_name.get(section_name)
        if sec is None or sec.char_span is None or sec.status != "ok":
            continue
        win = get_section_text(ingest, sections_artifact, section_name)
        if win is None:
            continue
        prov_source = _SECTION_TO_PROVENANCE_SOURCE.get(sec.detection_method, "regex")
        logger.info(
            "Retrieval cascade: hit section %r via %s (chars=%d)",
            section_name, sec.detection_method, len(win),
        )
        return RetrievalResult(
            windows=[win],
            provenance_source=prov_source,
            section=section_name,
            degraded=False,
        )

    # Pass 2: keyword search across full doc
    if fallback_keywords:
        windows = get_keyword_windows(
            ingest, fallback_keywords,
            max_windows=max_keyword_windows,
            window_chars=keyword_window_chars,
        )
        if windows:
            logger.info(
                "Retrieval cascade: %d primary section(s) %s missed; keyword search "
                "returned %d window(s)",
                len(primary_sections), primary_sections, len(windows),
            )
            return RetrievalResult(
                windows=windows,
                provenance_source=SOURCE_FALLBACK_KEYWORD,
                section=None,
                degraded=True,
            )

    # Pass 3: first-N chunks
    windows = get_first_n_chunks(ingest, max_chars=first_n_max_chars)
    logger.info(
        "Retrieval cascade: section + keyword passes empty; using first-%d-chars fallback",
        first_n_max_chars,
    )
    return RetrievalResult(
        windows=windows,
        provenance_source=SOURCE_FALLBACK_FIRST_N,
        section=None,
        degraded=True,
    )
