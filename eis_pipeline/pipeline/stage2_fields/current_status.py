"""
Stage 2.8 — Project current status: v2 stub.

Requires a disambiguation-safe search pipeline to avoid false matches
(e.g., "Cedar Creek" appears in many unrelated projects). Deferred to v2.
"""

from __future__ import annotations

from ..schema import EISRecord, ProjectCurrentStatus


def run(record: EISRecord, client=None) -> None:  # noqa: ARG001
    """Return the v1 stub — status unknown, deferred to v2."""
    record.project_current_status = ProjectCurrentStatus()
