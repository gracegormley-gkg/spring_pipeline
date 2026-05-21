"""
Stage 1.5 — Grouping (passthrough in v1).

Per build brief §6 Stage 1.5: each doc_key is its own logical publication;
cross-volume grouping deferred to v2.
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
    role: str = "main"
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
    """Trivial passthrough — one component per doc_key."""
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
