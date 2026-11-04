"""
Stage 2.4 — Alternatives extraction.
"""

from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING

from ..config import MODELS
from ..schema import AlternativeItem, EISRecord, EvidencePointer
from .retrieval import combine_chunk_context, get_chunks_by_keyword, get_chunks_by_tags

if TYPE_CHECKING:
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)

# Fallback keywords for when no chunk is tagged `alternatives`. The Apollo run
# missed alternatives entirely because of this tag-labeling gap.
_FALLBACK_KEYWORDS = [
    "alternative", "no action", "no-action", "preferred alternative",
    "alternatives considered", "alignment", "proposed action versus",
]

_SYSTEM = textwrap.dedent("""\
    You are extracting the alternatives considered in a U.S. Environmental Impact Statement.

    EIS documents are legally required to evaluate a range of alternatives including "No Action."
    However, individual documents use VARYING TERMINOLOGY for what counts as an alternative.
    All of the following are alternatives — extract every one of them you find:

      - "Alternative A", "Alternative 1", etc. — the most common naming
      - "No Action" / "No Build" / "Status Quo" — the legally required baseline
      - "Alignment 1", "Alignment 2", "Alignment A" — common in highway/transit docs
      - "Variation A", "Variation B", "Variation A-1" — sub-alternatives, especially common
                                                       for road projects (treat each Variation
                                                       as its own alternative)
      - "Option 1", "Option A", "Plan A" — used in older docs
      - "Route A", "Corridor 1" — used for transportation corridors
      - "Preferred Alternative" / "Recommended Alternative" — note which one is preferred
      - "Build Alternative" — used when contrasted with No Build

    Using ONLY the provided chunks, extract each named alternative with a 1–2 sentence
    description. Cite the chunk ID where each alternative is described.

    Respond with ONLY valid JSON:
    {
      "alternatives": [
        {
          "name": "No Action",
          "description": "...",
          "evidence": [{"chunk_id": "c03", "pages": [45, 46]}]
        },
        ...
      ]
    }

    Be exhaustive — it is better to include too many than miss one. If the document truly
    contains no alternatives discussion (extremely rare for a real EIS), return
    {"alternatives": []}.
""")

_USER_TMPL = textwrap.dedent("""\
    Document title: {title}

    Relevant chunks:
    {context}
""")


def run(record: EISRecord, client: "LLMClient") -> None:
    """Populate record.alternatives_proposed. Mutates record in place."""
    chunks = get_chunks_by_tags(record.chunks, ["alternatives"], max_chunks=6)

    if not chunks:
        logger.info(
            "Alternatives: no tagged chunks for %s — falling back to keyword search",
            record.doc_id,
        )
        chunks = get_chunks_by_keyword(record.chunks, _FALLBACK_KEYWORDS, max_chunks=6)

    if not chunks:
        logger.info("No alternatives content found for %s (tag + keyword empty)", record.doc_id)
        record.alternatives_proposed = []
        return

    # Bump context limit to 100k chars so we don't truncate the recommended-
    # alternative section at the end of large chunks (Harlem v2 regression).
    context = combine_chunk_context(chunks, max_chars=100_000)
    user_msg = _USER_TMPL.format(
        title=record.title or record.doc_id,
        context=context,
    )

    try:
        result = client.call_json(
            model=MODELS["heavy"],
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=1024,
            temperature=0.2,
            label=f"alternatives/{record.doc_id}",
        )
        raw_alts = result.get("alternatives") or []
        alternatives: list[AlternativeItem] = []
        for alt in raw_alts:
            if not isinstance(alt, dict) or not alt.get("name"):
                continue
            evidence = [
                EvidencePointer(chunk_id=e["chunk_id"], pages=e.get("pages", []))
                for e in (alt.get("evidence") or [])
                if isinstance(e, dict) and "chunk_id" in e
            ]
            alternatives.append(AlternativeItem(
                name=alt["name"],
                description=alt.get("description", ""),
                evidence=evidence,
            ))
        record.alternatives_proposed = alternatives
    except Exception as exc:
        logger.error("Alternatives extraction failed for %s: %s", record.doc_id, exc)
        record.alternatives_proposed = []
