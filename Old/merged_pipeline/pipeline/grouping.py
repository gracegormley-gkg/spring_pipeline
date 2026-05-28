"""
Stage 1.5 — Document Grouping (passthrough in v1).

Per synthesis_plan.md §Document grouping: one JSON per logical publication
(Draft / Final / Supplemental / ROD / NOI), with components[] covering main /
appendix / response_to_comments / errata. Cross-publication links deferred
to v2 (see schema's DeferredCrossLinks).

In v1 each doc_key is its own logical publication; multi-volume / appendix
detection is deferred until Phase 0a tells us how multi-record publications
actually surface in NUL accession data. This module is a passthrough wrapper
today — one component per doc_key, role="main", confidence=1.0 — but the
shape is forward-compatible: callers index `components[]` and `is_supplemental`
already, so plugging in real grouping logic later is additive.

Ported from v1_multiagent_pipeline/pipeline/grouping.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .ingest import IngestArtifact

logger = logging.getLogger(__name__)


@dataclass
class GroupedComponent:
    record_id: str
    role: str = "main"   # one of: main, appendix, response_to_comments, errata, combined_rod
    confidence: float = 1.0


@dataclass
class GroupingArtifact:
    publication_id: str
    components: list[GroupedComponent]
    is_supplemental: bool = False
    physical_record_ids: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "publication_id": self.publication_id,
            "is_supplemental": self.is_supplemental,
            "physical_record_ids": self.physical_record_ids,
            "components": [asdict(c) for c in self.components],
        }


def run(ingest: IngestArtifact) -> GroupingArtifact:
    """Trivial passthrough — one component per doc_key.

    Future versions will inspect NUL relations / accession patterns to merge
    multi-volume sets into a single publication_id. Until then, each doc is
    its own publication.
    """
    art = GroupingArtifact(
        publication_id=ingest.publication_id,
        physical_record_ids=ingest.physical_record_ids,
        components=[GroupedComponent(
            record_id=ingest.publication_id,
            role="main",
            confidence=1.0,
        )],
        is_supplemental=False,
    )
    logger.info("Stage 1.5: passthrough — %s as single 'main' component",
                art.publication_id)
    return art


def write_artifact(artifact: GroupingArtifact, output_dir: str | Path) -> Path:
    path = Path(output_dir) / f"{artifact.publication_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_json(), indent=2), encoding="utf-8")
    return path
