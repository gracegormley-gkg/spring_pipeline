"""
Stage 2.1 — Summary extraction.
"""

from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING

from ..config import MODELS
from ..schema import EISRecord, EvidencePointer, SummaryField
from .retrieval import combine_chunk_context, get_chunks_by_keyword, get_chunks_by_tags

if TYPE_CHECKING:
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)

# Fallback keywords used when no chunks carry the target topic tags.
# Keeps the field from going silent if the Stage 1 labeler missed the relevant tags.
_FALLBACK_KEYWORDS = [
    "purpose", "proposed", "needed", "need for", "project description",
    "affected environment", "environmental impact", "would result",
    "alternatives considered",
]

_SYSTEM = textwrap.dedent("""\
    You are an analyst summarizing U.S. Environmental Impact Statements (EIS).

    Write a detailed, evidence-grounded summary (~180 words) that covers ALL of:
    (a) the community or population impacted
    (b) the final goal of the proposed project
    (c) why the project was needed (driving forces, stated justification)
    (d) the anticipated environmental impacts (positive and negative)

    Use only the provided document chunks. Do not introduce outside knowledge.
    Cite every factual claim with the chunk ID where it appears.
    You may use proper bureaucratic / technical terminology where it is accurate.

    Respond with ONLY valid JSON:
    {
      "summary": "...",
      "evidence": [
        {"chunk_id": "c01", "pages": [12, 13]},
        ...
      ]
    }
""")

_USER_TMPL = textwrap.dedent("""\
    Document title: {title}

    Relevant chunks:
    {context}
""")

_LAYMAN_SYSTEM = textwrap.dedent("""\
    You are rewriting a detailed Environmental Impact Statement summary for a
    general public audience (think: high school reading level, no policy or
    engineering background).

    Rules:
    - Use ONLY information present in the detailed summary I give you.
      Do NOT introduce new facts.
    - Replace bureaucratic or technical language with plain English.
      ("right-of-way" → "land taken for the road"; "mitigation" → "steps to
      reduce harm"; "alignment" → "the path the road takes")
    - Keep it short: 80–120 words. One paragraph. Conversational tone.
    - Do not include chunk IDs or evidence citations — this is the
      public-facing version.

    Respond with ONLY valid JSON:
    {
      "layman_summary": "..."
    }
""")


def run(record: EISRecord, client: "LLMClient") -> None:
    """Populate record.summary. Mutates record in place."""
    target_tags = ["purpose_and_need", "proposed_action", "affected_environment"]
    chunks = get_chunks_by_tags(record.chunks, target_tags, max_chunks=8)

    if not chunks:
        logger.info(
            "Summary: no chunks matched target tags for %s — falling back to keyword search",
            record.doc_id,
        )
        chunks = get_chunks_by_keyword(record.chunks, _FALLBACK_KEYWORDS, max_chunks=8)

    if not chunks:
        logger.info(
            "Summary: keyword fallback empty for %s — using first usable chunks",
            record.doc_id,
        )
        chunks = [c for c in record.chunks if c.used][:8]

    if not chunks:
        logger.warning("No usable chunks for summary in doc %s", record.doc_id)
        record.summary = SummaryField(
            text="",
            evidence=[],
            status="insufficient_information",
        )
        return

    context = combine_chunk_context(chunks)
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
            label=f"summary/{record.doc_id}",
        )
        text = result.get("summary", "").strip()
        raw_evidence = result.get("evidence", [])
        evidence = [
            EvidencePointer(chunk_id=e["chunk_id"], pages=e.get("pages", []))
            for e in raw_evidence
            if isinstance(e, dict) and "chunk_id" in e
        ]
        status = "populated" if text else "insufficient_information"

        layman = _generate_layman_summary(text, client, record.doc_id) if text else None

        record.summary = SummaryField(
            text=text,
            layman_text=layman,
            evidence=evidence,
            status=status,
        )
    except Exception as exc:
        logger.error("Summary extraction failed for %s: %s", record.doc_id, exc)
        record.summary = SummaryField(text="", evidence=[], status="insufficient_information")


def _generate_layman_summary(
    detailed_text: str,
    client: "LLMClient",
    doc_id: str,
) -> str | None:
    """
    Rewrite the detailed summary in plain language. Inherits factual grounding
    from `detailed_text` — does not re-read the document.
    """
    try:
        result = client.call_json(
            model=MODELS["heavy"],
            system=_LAYMAN_SYSTEM,
            messages=[{"role": "user", "content": f"Detailed summary:\n{detailed_text}"}],
            max_tokens=512,
            temperature=0.3,
            label=f"summary_layman/{doc_id}",
        )
        layman = result.get("layman_summary")
        if isinstance(layman, str):
            layman = layman.strip()
            return layman or None
        return None
    except Exception as exc:
        logger.warning("Layman summary failed for %s: %s — keeping detailed only", doc_id, exc)
        return None
