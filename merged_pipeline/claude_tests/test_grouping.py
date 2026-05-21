"""
Stage 1.5 grouping tests — passthrough behavior.

In v1 grouping is a passthrough: one component per doc_key, role="main",
confidence=1.0, is_supplemental=False. These tests pin that behavior so a
later port to real multi-record grouping is forced to update the contract
explicitly rather than silently changing it.
"""

from __future__ import annotations

import json

from pipeline.grouping import (
    GroupedComponent,
    GroupingArtifact,
    run as grouping_run,
    write_artifact,
)
from pipeline.ingest import IngestArtifact


def _stub_ingest(doc_key: str = "p1074_test") -> IngestArtifact:
    return IngestArtifact(
        publication_id=doc_key,
        physical_record_ids=[doc_key],
        raw_text="hello world",
        text_normalized="hello world",
        alignment_map=[],
        pages=[],
        nul_metadata={},
        raw_text_hash="sha256:deadbeef",
    )


def test_passthrough_single_component() -> None:
    art = grouping_run(_stub_ingest("p1074_abc"))
    assert art.publication_id == "p1074_abc"
    assert art.physical_record_ids == ["p1074_abc"]
    assert art.is_supplemental is False
    assert len(art.components) == 1
    c = art.components[0]
    assert isinstance(c, GroupedComponent)
    assert c.record_id == "p1074_abc"
    assert c.role == "main"
    assert c.confidence == 1.0


def test_passthrough_to_json_shape() -> None:
    art = grouping_run(_stub_ingest("p1074_xyz"))
    j = art.to_json()
    # Forward-compat shape: callers index by these keys.
    assert set(j.keys()) == {"publication_id", "is_supplemental", "physical_record_ids", "components"}
    assert j["components"][0]["role"] == "main"
    assert j["is_supplemental"] is False


def test_write_artifact_roundtrip(tmp_path) -> None:
    art = grouping_run(_stub_ingest("p1074_round_trip"))
    out = write_artifact(art, tmp_path)
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == art.to_json()


def test_grouping_artifact_constructable_directly() -> None:
    """We expect future stages to construct GroupingArtifact themselves
    when reading cached output. Lock the constructor surface."""
    g = GroupingArtifact(
        publication_id="x",
        components=[GroupedComponent(record_id="x", role="main")],
        is_supplemental=False,
        physical_record_ids=["x"],
    )
    assert g.to_json()["components"][0]["confidence"] == 1.0
