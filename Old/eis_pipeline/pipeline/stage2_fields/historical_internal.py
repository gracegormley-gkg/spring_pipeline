"""
Stage 2.6 — Historical context (internal): what the document itself says about history.
"""

from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING

from ..config import MODELS
from ..schema import EISRecord, EvidencePointer, HistoricalContextClaim, HistoricalContextInternal
from .retrieval import (
    combine_chunk_context,
    get_chunks_by_keyword,
    get_chunks_by_tags,
)

if TYPE_CHECKING:
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)

_HISTORY_KEYWORDS = [
    "history", "background", "previously", "prior to", "established in",
    "originally", "historically", "in the past", "was built", "was constructed",
    "had been", "dating back", "since 19", "since 20",
]

_SYSTEM = textwrap.dedent("""\
    You are extracting historical context from a U.S. Environmental Impact Statement.

    Rules:
    - Use ONLY information stated in the provided document chunks.
    - Phrase every claim as "The document states that..." or "According to the document..."
    - Every claim needs a chunk ID and page number as evidence.
    - Do not add outside knowledge. Do not speculate.
    - If there is no meaningful historical context in the provided text, return status "insufficient_information".

    Respond with ONLY valid JSON:
    {
      "text": "Overall historical context paragraph...",
      "claims": [
        {
          "sentence": "The document states that the project site was previously used for...",
          "evidence": [{"chunk_id": "c02", "pages": [5, 6]}]
        },
        ...
      ],
      "status": "populated" | "insufficient_information"
    }
""")

_USER_TMPL = textwrap.dedent("""\
    Document title: {title}

    Relevant chunks:
    {context}
""")


def run(record: EISRecord, client: "LLMClient") -> None:
    """Populate record.historical_context_internal. Mutates record in place."""
    # Retrieve by tag + keyword
    tag_chunks = get_chunks_by_tags(
        record.chunks,
        ["purpose_and_need", "affected_environment", "proposed_action"],
        max_chunks=6,
    )
    keyword_chunks = get_chunks_by_keyword(
        record.chunks,
        _HISTORY_KEYWORDS,
        max_chunks=6,
    )

    # Deduplicate by chunk_id, preserve order
    seen: set[str] = set()
    combined: list = []
    for chunk in tag_chunks + keyword_chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            combined.append(chunk)

    # Rerank: prioritize chunks with keyword hits
    combined = combined[:8]

    if not combined:
        record.historical_context_internal = HistoricalContextInternal(
            text=None,
            claims=[],
            status="insufficient_information",
        )
        return

    context = combine_chunk_context(combined)
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
            label=f"historical_internal/{record.doc_id}",
        )
        text = result.get("text") or None
        raw_claims = result.get("claims") or []
        status = result.get("status", "insufficient_information")
        if status not in ("populated", "insufficient_information"):
            status = "insufficient_information"

        claims: list[HistoricalContextClaim] = []
        for claim in raw_claims:
            if not isinstance(claim, dict) or not claim.get("sentence"):
                continue
            evidence = [
                EvidencePointer(chunk_id=e["chunk_id"], pages=e.get("pages", []))
                for e in (claim.get("evidence") or [])
                if isinstance(e, dict) and "chunk_id" in e
            ]
            claims.append(HistoricalContextClaim(sentence=claim["sentence"], evidence=evidence))

        record.historical_context_internal = HistoricalContextInternal(
            text=text,
            claims=claims,
            status=status,  # type: ignore[arg-type]
        )
    except Exception as exc:
        logger.error("Historical context (internal) failed for %s: %s", record.doc_id, exc)
        record.historical_context_internal = HistoricalContextInternal(
            text=None,
            claims=[],
            status="insufficient_information",
        )
