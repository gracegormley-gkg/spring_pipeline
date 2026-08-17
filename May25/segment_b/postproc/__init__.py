"""
Segment B post-processors.

Each module here runs AFTER the M2 extractors and BEFORE the Critic, replacing
or correcting a specific Segment A failure mode identified in MCAL_PLAN 1:

  acronyms.py            1(11) undefined acronyms, 8/8 docs
  location_pipeline.py   1(9)  location failures, 6/8 docs
  key_people_pipeline.py 1(10) commenters mislabeled as cooperators, 5/8 docs
"""

from __future__ import annotations
