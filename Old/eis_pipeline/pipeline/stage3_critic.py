"""
Stage 3 — Critic: validates LLM-generated fields for claim/evidence fidelity.

Two layers:
1. Deterministic checks (no LLM): quotes (substring), year range, theme vocab, geocoding
2. LLM critic (Sonnet): per-claim evidence verification for summary, themes, location,
   alternatives, historical context
"""

from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING

from .config import ALL_SUBTHEMES, ALL_THEMES, MAX_YEAR, MODELS, NEPA_YEAR
from .schema import EISRecord

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def run(record: EISRecord, client: "LLMClient") -> list[str]:
    """Run all critic checks. Returns list of warning strings. Mutates record in place."""
    warnings: list[str] = []

    _check_quotes(record, warnings)
    _check_year(record, warnings)
    _check_themes(record, warnings)
    _check_geocoding(record, warnings)
    _llm_critic_summary(record, client, warnings)
    _llm_critic_historical_internal(record, client, warnings)

    return warnings


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

def _check_quotes(record: EISRecord, warnings: list[str]) -> None:
    """Hard pass/fail: every quote must be a verbatim substring of its cited chunk."""
    chunk_map = {c.chunk_id: c.text for c in record.chunks}

    for entity in record.key_people_and_groups:
        if entity.quote is None:
            continue
        q = entity.quote
        chunk_text = chunk_map.get(q.chunk_id, "")
        verified = q.text in chunk_text
        if not verified:
            logger.warning(
                "Quote substring check FAILED for entity '%s' in doc %s — nulling quote",
                entity.name, record.doc_id,
            )
            warnings.append(f"quote_verification_failed: entity={entity.name}")
            entity.quote = None
        else:
            entity.quote = q.model_copy(update={"substring_verified": True})


def _check_year(record: EISRecord, warnings: list[str]) -> None:
    if record.year is None:
        return
    if not (NEPA_YEAR <= record.year <= MAX_YEAR):
        msg = f"Year {record.year} out of valid range [{NEPA_YEAR}, {MAX_YEAR}]"
        logger.warning(msg)
        warnings.append(f"year_invalid: {msg}")
        record.year = None
        record.date = None


def _check_themes(record: EISRecord, warnings: list[str]) -> None:
    invalid_primary = [t for t in record.themes.primary if t not in ALL_THEMES]
    invalid_sub = [s for s in record.themes.subthemes if s not in ALL_SUBTHEMES]

    if invalid_primary:
        logger.warning("Invalid primary themes for %s: %s", record.doc_id, invalid_primary)
        warnings.append(f"invalid_themes: {invalid_primary}")
        record.themes.primary = [t for t in record.themes.primary if t in ALL_THEMES] or ["other"]

    if invalid_sub:
        logger.warning("Invalid subthemes for %s: %s", record.doc_id, invalid_sub)
        warnings.append(f"invalid_subthemes: {invalid_sub}")
        record.themes.subthemes = [s for s in record.themes.subthemes if s in ALL_SUBTHEMES]


def _check_geocoding(record: EISRecord, warnings: list[str]) -> None:
    loc = record.location
    if loc.name and (loc.latitude is None or loc.longitude is None):
        warnings.append(f"geocoding_failed: {loc.name!r}")


# ---------------------------------------------------------------------------
# LLM critic
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = textwrap.dedent("""\
    You are a fact-checking critic for Environmental Impact Statement metadata.

    For each (claim, chunk_id) pair below, answer whether the claim appears in the cited chunk.
    Answer "yes", "no", or "partial" with ONE sentence of justification.

    Respond with ONLY valid JSON:
    {
      "results": [
        {"claim_index": 0, "verdict": "yes", "justification": "..."},
        ...
      ]
    }
""")


def _llm_critic_summary(record: EISRecord, client: "LLMClient", warnings: list[str]) -> None:
    if record.summary is None or record.summary.status == "insufficient_information":
        return
    if not record.summary.evidence:
        return

    _run_claim_critic(
        field_name="summary",
        claims=[(record.summary.text, e) for e in record.summary.evidence],
        record=record,
        client=client,
        warnings=warnings,
        on_failure=lambda: _set_summary_insufficient(record),
    )


def _llm_critic_historical_internal(
    record: EISRecord, client: "LLMClient", warnings: list[str]
) -> None:
    ctx = record.historical_context_internal
    if ctx.status == "insufficient_information" or not ctx.claims:
        return

    claim_evidence_pairs = [
        (claim.sentence, e)
        for claim in ctx.claims
        for e in claim.evidence
    ]
    if not claim_evidence_pairs:
        return

    _run_claim_critic(
        field_name="historical_internal",
        claims=claim_evidence_pairs,
        record=record,
        client=client,
        warnings=warnings,
        on_failure=lambda: _set_historical_insufficient(record),
    )


def _run_claim_critic(
    field_name: str,
    claims: list[tuple[str, object]],  # (claim_text, EvidencePointer)
    record: EISRecord,
    client: "LLMClient",
    warnings: list[str],
    on_failure,
) -> None:
    chunk_map = {c.chunk_id: c.text for c in record.chunks}

    # Build user message listing each claim + its cited chunk text
    lines: list[str] = []
    for i, (claim_text, evidence) in enumerate(claims):
        chunk_text = chunk_map.get(evidence.chunk_id, "(chunk not found)")[:1000]  # type: ignore[union-attr]
        lines.append(
            f"Claim {i}: {claim_text}\n"
            f"Cited chunk ({evidence.chunk_id}):\n{chunk_text}"  # type: ignore[union-attr]
        )
    user_msg = "\n\n---\n\n".join(lines)

    try:
        result = client.call_json(
            model=MODELS["critic"],
            system=_CRITIC_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=1024,
            temperature=0.0,
            label=f"critic/{field_name}/{record.doc_id}",
        )
        results = result.get("results") or []
        any_failure = any(r.get("verdict") == "no" for r in results if isinstance(r, dict))
        if any_failure:
            logger.warning(
                "Critic found unsupported claims in %s/%s — downgrading to insufficient_information",
                field_name, record.doc_id,
            )
            warnings.append(f"critic_failed: {field_name}")
            on_failure()
    except Exception as exc:
        logger.warning("LLM critic failed for %s/%s: %s", field_name, record.doc_id, exc)
        warnings.append(f"critic_error: {field_name}: {exc}")


def _set_summary_insufficient(record: EISRecord) -> None:
    if record.summary:
        record.summary = record.summary.model_copy(update={"status": "insufficient_information"})


def _set_historical_insufficient(record: EISRecord) -> None:
    record.historical_context_internal = record.historical_context_internal.model_copy(
        update={"status": "insufficient_information"}
    )
