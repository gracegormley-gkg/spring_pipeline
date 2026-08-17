"""
M-Cal: calibration step between Segment A (human-graded 20-doc run) and
Segment B (production run over ~2000 docs).

Consumes Segment A human grades and emits stage-versioned calibration
artifacts that Segment B loads at runtime. See segment_b/MCAL_PLAN.md.

Entry point:  python -m mcal.build --stage v1
"""

from __future__ import annotations

from . import settings  # noqa: F401  -- installs the segment_a sys.path bridge

__all__ = ["settings"]
