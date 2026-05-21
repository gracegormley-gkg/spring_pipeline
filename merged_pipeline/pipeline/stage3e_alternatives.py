"""
Stage 3e — Alternatives extraction.

Two-stage approach (synthesis_plan §Alternatives):
  Stage A: Regex-extract labels from the alternatives section text.
           Free; handles common label patterns (Alternative N, No Action,
           Alignment, Variation, Option, Route, Corridor).
  Stage B: ONE Sonnet call describes ALL labels at once. Cheaper than per-label
           calls and produces consistent style. Each label's context is bounded
           to first ~500 tokens (~2000 chars) after the label occurrence to cap
           cost (synthesis_plan §Alternatives "capped to first 500 tokens after
           label").

Retrieval cascade for the source text:
  Primary section: alternatives
  Keyword fallback: "Alternative", "No Action", "Alignment", ...
  First-N fallback: first 10000 chars

When the alternatives section is missing AND keyword fallback finds nothing,
record.alternatives ends up [] with no warning beyond the retrieval log entry —
that's the honest answer for a doc with no alternatives discussion (rare in
practice; almost always means the doc is a memo/letter, not a real EIS).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from . import config
from .ingest import IngestArtifact
from .retrieval import (
    SOURCE_FALLBACK_FIRST_N,
    SOURCE_FALLBACK_KEYWORD,
    cascade,
)
from .schema import Alternative, EISRecord, Provenance

if TYPE_CHECKING:
    from .llm_client import LLMClient
    from .sections import SectionsArtifact

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Stage A label regex. Catches the common label words, optionally followed by
# a number/letter or compound identifier. Anchored at word boundary; case
# preserved so we keep the doc's capitalization. Also matches "No Action"
# and "No Build" as standalone labels.
_LABEL_PATTERNS = [
    # "Alternative A", "Alternative 1", "Alternative A-1"
    re.compile(r"\bAlternative\s+([A-Z]|\d+)(?:[-–][A-Z\d]+)?\b"),
    # "Alignment 1", "Alignment A"
    re.compile(r"\bAlignment\s+([A-Z]|\d+)\b"),
    # "Variation A", "Variation 1", "Variation A-1"
    re.compile(r"\bVariation\s+([A-Z]|\d+)(?:[-–][A-Z\d]+)?\b"),
    # "Option A", "Option 1", "Plan A"
    re.compile(r"\b(?:Option|Plan)\s+([A-Z]|\d+)\b"),
    # "Route A", "Corridor 1"
    re.compile(r"\b(?:Route|Corridor)\s+([A-Z]|\d+)\b"),
    # "No Action" / "No Build" / "Status Quo"
    re.compile(r"\bNo[-\s]+Action\b|\bNo[-\s]+Build\b|\bStatus\s+Quo\b"),
    # "Preferred Alternative" / "Recommended Alternative" / "Build Alternative"
    re.compile(r"\b(?:Preferred|Recommended|Build)\s+Alternative\b"),
]

# Per-label context window for Sonnet (~500 tokens ≈ 2000 chars).
_LABEL_CONTEXT_CHARS = 2000

# Cap total labels we ship to Sonnet (one big call).
_MAX_LABELS_PER_CALL = 12

# Keywords for cascade fallback when 'alternatives' section is missing.
_FALLBACK_KEYWORDS = [
    "Alternative", "No Action", "No Build", "Alignment", "Variation",
    "Option", "Route A", "Corridor", "Preferred Alternative",
    "alternatives considered",
]


def run(
    record: EISRecord,
    ingest: IngestArtifact,
    sections_artifact: "SectionsArtifact",
    llm: "LLMClient | None" = None,
) -> list[str]:
    """Mutate `record.alternatives`. Returns warnings."""
    warnings: list[str] = []

    retrieval = cascade(
        ingest, sections_artifact,
        primary_sections=["alternatives"],
        fallback_keywords=_FALLBACK_KEYWORDS,
        max_keyword_windows=4,
        keyword_window_chars=4000,
        first_n_max_chars=10_000,
    )

    if not retrieval.windows:
        record.alternatives = []
        warnings.append("alternatives: empty document; no retrieval windows")
        return warnings

    # Stage A: extract labels via regex. Each label keeps its absolute char
    # offset in the raw text so we can locate context windows for Stage B.
    labels = _extract_labels(retrieval, ingest)
    if not labels:
        record.alternatives = []
        if retrieval.degraded:
            warnings.append(
                "alternatives: retrieval degraded "
                f"({retrieval.provenance_source}) and no labels found"
            )
        else:
            logger.info("Stage 3e: no alternative labels found in alternatives section")
        return warnings

    logger.info("Stage 3e: extracted %d label(s): %s",
                len(labels), [lbl["label"] for lbl in labels])

    if llm is None:
        # Without an LLM, ship label-only entries with empty descriptions and
        # status="needs_review".
        record.alternatives = [
            Alternative(
                label=lbl["label"],
                description="",
                provenance=_label_provenance(lbl, retrieval),
                status="needs_review",
            )
            for lbl in labels
        ]
        warnings.append("alternatives: no llm provided; descriptions left empty")
        return warnings

    # Stage B: one Sonnet call for all label descriptions.
    descriptions = _sonnet_describe(labels, ingest, llm)
    if descriptions is None:
        # Sonnet failed entirely; ship label-only with needs_review.
        record.alternatives = [
            Alternative(
                label=lbl["label"],
                description="",
                provenance=_label_provenance(lbl, retrieval),
                status="needs_review",
            )
            for lbl in labels
        ]
        warnings.append("alternatives: Sonnet description pass failed")
        return warnings

    alternatives: list[Alternative] = []
    for lbl in labels:
        desc_entry = descriptions.get(lbl["label"], {})
        desc = desc_entry.get("description") or ""
        found = bool(desc_entry.get("found")) and bool(desc)
        # Even degraded retrieval shouldn't auto-mark every alt as needs_review
        # — the label was extracted deterministically. Mark needs_review only
        # if Sonnet itself said it couldn't describe.
        status = "ok" if found else "needs_review"
        # If the section was missing (degraded retrieval), still bump status
        # down because the LLM may have described from a noisy keyword window.
        if retrieval.degraded and status == "ok":
            status = "needs_review"
        alternatives.append(Alternative(
            label=lbl["label"],
            description=desc,
            provenance=_label_provenance(lbl, retrieval),
            status=status,
        ))

    record.alternatives = alternatives
    if retrieval.degraded:
        warnings.append(
            f"alternatives: retrieval degraded "
            f"(source={retrieval.provenance_source}); {len(alternatives)} labels with "
            f"description statuses downgraded to needs_review"
        )
    return warnings


# ---------------------------------------------------------------------------
# Stage A — label extraction
# ---------------------------------------------------------------------------

def _extract_labels(retrieval, ingest: IngestArtifact) -> list[dict]:
    """Scan retrieval windows for labels. Returns dicts with:
      label             — the matched label string (verbatim from doc)
      char_offset_raw   — absolute offset in ingest.raw_text
      window_start      — start of the originating retrieval window
    De-dupes labels by their normalized label string (e.g. 'Alternative A'
    appearing 3 times only emits once)."""
    seen_labels: set[str] = set()
    out: list[dict] = []

    for win in retrieval.windows:
        win_offset = win.char_offset_raw[0]
        for pat in _LABEL_PATTERNS:
            for m in pat.finditer(win.text):
                label = _normalize_label(m.group(0))
                if label in seen_labels:
                    continue
                abs_offset = win_offset + m.start()
                out.append({
                    "label": label,
                    "char_offset_raw": (abs_offset, abs_offset + len(m.group(0))),
                    "window_start": win_offset,
                })
                seen_labels.add(label)
                if len(out) >= _MAX_LABELS_PER_CALL:
                    return out
    return out


def _normalize_label(raw_label: str) -> str:
    """Light normalization: collapse whitespace, title-case canonical labels.
    Preserves ID suffixes verbatim (e.g. 'Alternative A-1' stays 'Alternative A-1')."""
    s = re.sub(r"\s+", " ", raw_label.strip())
    # Canonicalize "No-Action" / "No Action" / "NO ACTION" -> "No Action"
    if re.fullmatch(r"No[-\s]+Action", s, re.IGNORECASE):
        return "No Action"
    if re.fullmatch(r"No[-\s]+Build", s, re.IGNORECASE):
        return "No Build"
    if re.fullmatch(r"Status\s+Quo", s, re.IGNORECASE):
        return "Status Quo"
    return s


def _label_provenance(label: dict, retrieval) -> Provenance:
    return Provenance(
        source=retrieval.provenance_source,  # type: ignore[arg-type]
        char_offset_raw=label["char_offset_raw"],
        section=retrieval.section,  # type: ignore[arg-type]
        note=f"regex label match",
    )


# ---------------------------------------------------------------------------
# Stage B — Sonnet description
# ---------------------------------------------------------------------------

def _sonnet_describe(
    labels: list[dict],
    ingest: IngestArtifact,
    llm: "LLMClient",
) -> dict[str, dict] | None:
    """Build per-label context (capped to ~500 tokens after each label),
    send all to Sonnet in one call, return mapping label -> {description, found}.
    Returns None if the LLM call fails entirely.
    """
    # Build one big context string with each label and its 2000-char window.
    contexts: list[str] = []
    for lbl in labels:
        s = lbl["char_offset_raw"][0]
        e = min(len(ingest.raw_text), s + _LABEL_CONTEXT_CHARS)
        contexts.append(
            f"=== {lbl['label']} ===\n{ingest.raw_text[s:e]}\n"
        )
    document_text = "\n".join(contexts)
    label_list_str = "\n".join(f"  - {lbl['label']}" for lbl in labels)

    prompt_template = (PROMPTS_DIR / "3e_alternatives.txt").read_text(encoding="utf-8")
    prompt = (
        prompt_template
        .replace("{labels}", label_list_str)
        .replace("{document_text}", document_text)
    )

    try:
        result = llm.call_json(
            model=llm.models["sonnet"],
            system="You are a careful extractor. Return only the requested JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.2,
            label="3e_alternatives",
        )
    except Exception as exc:
        logger.warning("3e Sonnet describe LLM call failed: %s", exc)
        return None

    if not isinstance(result, dict):
        return None
    raw_alts = result.get("alternatives") or []
    if not isinstance(raw_alts, list):
        return None

    by_label: dict[str, dict] = {}
    for alt in raw_alts:
        if not isinstance(alt, dict):
            continue
        label = alt.get("label")
        if not isinstance(label, str):
            continue
        by_label[label.strip()] = {
            "description": (alt.get("description") or "").strip() or None,
            "found": bool(alt.get("found")),
        }
    return by_label
