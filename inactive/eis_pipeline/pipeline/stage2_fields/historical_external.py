"""
Stage 2.7 — Historical context (external): v2 stub.

This field requires a Tier 1 allowlisted search API, domain filtering,
fetch + clean pipeline, and a stricter critic. Deferred to v2.
"""

from __future__ import annotations

from ..schema import EISRecord, HistoricalContextExternal


def run(record: EISRecord, client=None) -> None:  # noqa: ARG001
    """Return the v1 stub — no external context available."""
    record.historical_context_external = HistoricalContextExternal()
