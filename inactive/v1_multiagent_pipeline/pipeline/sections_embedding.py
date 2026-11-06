"""
Stage 2 fallback — embedding-similarity section detection.

Uses sentence-transformers/all-MiniLM-L6-v2 to find heading-like lines that
embed close to canonical descriptors of required sections. Only runs when the
regex pass misses a section that downstream stages need.

Imports sentence-transformers lazily; the module fails fast if it's not
installed and the caller (pipeline.sections.run) catches and degrades to
"not_found" without aborting the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from . import config
from .ingest import FakePage, page_range_for_span
from .sections import DetectedSection

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    """Load the embedding model once per process."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
    return SentenceTransformer(config.EMBEDDING_MODEL)


def _candidate_headings(raw: str, max_lines: int = 5000) -> list[tuple[int, str]]:
    """
    Pull lines that *could* be a section heading: short-ish, mostly uppercase,
    or look heading-shaped. Returns (char_offset, line_text).
    """
    out: list[tuple[int, str]] = []
    pos = 0
    n = len(raw)
    while pos < n and len(out) < max_lines:
        nl = raw.find("\n", pos)
        line_end = nl if nl != -1 else n
        line = raw[pos:line_end].strip()
        if 4 <= len(line) <= 80:
            # Heading-shaped: starts with uppercase or all-caps, no period at end
            if line[0].isupper() and not line.endswith("."):
                upper_ratio = sum(1 for c in line if c.isupper()) / max(1, sum(1 for c in line if c.isalpha()))
                if upper_ratio >= 0.5 or line.istitle():
                    out.append((pos, line))
        pos = line_end + 1
    return out


def embedding_detect(
    missing_section_names: list[str],
    raw: str,
    pages: list[FakePage],
) -> list[DetectedSection]:
    """
    For each missing section, score candidate headings against its descriptor
    and return the best match if it clears (threshold, margin).
    """
    descriptors = {
        name: config.SECTION_DESCRIPTORS[name]
        for name in missing_section_names
        if name in config.SECTION_DESCRIPTORS
    }
    if not descriptors:
        return []

    candidates = _candidate_headings(raw)
    if not candidates:
        logger.info("Embedding fallback: no candidate headings found")
        return []

    model = _model()
    desc_texts = list(descriptors.values())
    desc_names = list(descriptors.keys())

    desc_emb = model.encode(desc_texts, normalize_embeddings=True)
    cand_emb = model.encode([c[1] for c in candidates], normalize_embeddings=True)

    # cosine sim = dot product on normalized embeddings
    import numpy as np  # type: ignore
    sims = np.dot(cand_emb, desc_emb.T)  # shape (n_candidates, n_descriptors)

    detected: list[DetectedSection] = []
    for di, name in enumerate(desc_names):
        col = sims[:, di]
        order = col.argsort()[::-1]
        best_idx = int(order[0])
        best_score = float(col[best_idx])
        runner_idx = int(order[1]) if len(order) > 1 else best_idx
        runner_score = float(col[runner_idx]) if len(order) > 1 else 0.0

        if best_score < config.EMBEDDING_THRESHOLD:
            logger.info(
                "Embedding fallback: %s — best=%.3f below threshold %.2f",
                name, best_score, config.EMBEDDING_THRESHOLD,
            )
            continue
        if (best_score - runner_score) < config.EMBEDDING_MARGIN:
            logger.info(
                "Embedding fallback: %s — ambiguous (best=%.3f vs runner=%.3f, margin %.2f)",
                name, best_score, runner_score, config.EMBEDDING_MARGIN,
            )
            detected.append(DetectedSection(
                name=name,
                char_span=None,
                pages=None,
                confidence=best_score,
                status="ambiguous",
                detection_method="embedding_fallback",
            ))
            continue

        offset, line = candidates[best_idx]
        page_range = page_range_for_span(pages, (offset, offset + len(line)))
        detected.append(DetectedSection(
            name=name,
            char_span=(offset, offset + len(line)),
            pages=page_range,
            confidence=best_score,
            status="ok",
            detection_method="embedding_fallback",
        ))
        logger.info(
            "Embedding fallback hit: %s @ offset=%d page=%s | score=%.3f | line=%r",
            name, offset, page_range, best_score, line[:60],
        )

    return detected
