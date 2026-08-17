"""
Segment B: production extraction over the ~2,000-doc corpus, gated by M-Cal
calibration artifacts.

Inherits M1/M2 extraction from Segment A and adds:
  - critic.py              evidence-first Critic, per-subfield, quote-verified
  - gate.py                conformal HUMAN_REVIEW gate + run_manifest.json
  - year_adjudicator.py    always-run year adjudication (M1 fix)
  - postproc/acronyms.py   deterministic glossary pre/post-pass
  - postproc/location_pipeline.py    scope-conditional geocoder cascade
  - postproc/key_people_pipeline.py  role-restricted entity extraction

Segment B pins whichever M-Cal stage is current; see mcal/settings.py
latest_stage() and MCAL_PLAN.md 7.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `mcal` importable, which in turn bridges segment_a's flat modules.
_MAY25_ROOT = Path(__file__).resolve().parent.parent
if str(_MAY25_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAY25_ROOT))

from mcal import settings  # noqa: F401,E402

__all__ = ["settings"]
