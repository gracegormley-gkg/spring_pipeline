"""
End-to-end pipeline smoke test.

Runs the entire pipeline (Stage 1 -> 1.5 -> 2 -> 3a-g -> 4 -> 5) against the
fixture doc with --no-llm semantics. Asserts that:
  - The pipeline completes without raising
  - The final EISRecord round-trips through schema validation
  - All hard gates pass on the no-LLM path
  - Status downgrades happen as documented
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.schema import EISRecord
from pipeline.stage5_output import run_doc


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_doc.json"


def test_end_to_end_no_llm_on_fixture(tmp_path) -> None:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"fixture not present: {FIXTURE_PATH}")

    output_path = tmp_path / "out.json"
    nul_cache_dir = tmp_path / "nul_cache"

    record = run_doc(
        "p1074_35556035057348",
        docs_json_path=FIXTURE_PATH,
        nul_cache_dir=nul_cache_dir,
        llm=None,
        output_path=output_path,
        write_ledger=False,  # don't pollute output/token_ledger.json during tests
    )

    # Basic shape
    assert record.publication_id == "p1074_35556035057348"
    assert record.physical_record_ids == ["p1074_35556035057348"]

    # No LLM means: title from NUL (offline NUL stub returns nothing), so
    # title may be None — but year + agency + eis_type all extract via the
    # deterministic paths. We don't have NUL access in tests so title may
    # also be None unless a NUL cache hit is present.
    # eis_type: regex on cover catches "FINAL ENVIRONMENTAL STATEMENT"
    assert record.eis_type.value == "Final"
    assert record.eis_type.status == "extracted_from_cover"

    # year: regex on first 5 fake pages catches 1971
    assert record.year.value == 1971

    # Lead agency: cover scan finds USFS or USDA
    assert record.agency.lead_agency.value in {"USFS", "USDA"}

    # Without LLM, summary/themes/alternatives/location/stakeholders all
    # skip or fail gracefully (no exception)
    assert record.summary.value is None or record.summary.status in {
        "needs_review", "skipped_no_llm",
    }
    assert record.themes.status in {
        "skipped_summary_unavailable", "needs_review",
    }
    # No comment section detected -> stakeholders empty
    assert record.stakeholders == []
    assert record.stakeholder_status == "no_comment_section_found"

    # Hard gates all pass (deterministic paths only)
    gates = record.validation.hard_gates
    assert gates.verbatim_quotes == "pass"  # no quotes -> trivially passes
    assert gates.year_range == "pass"
    assert gates.theme_vocab == "pass"
    assert gates.geocoding_centroid == "pass"
    assert gates.schema_validation == "pass"
    assert record.validation.review_routing == "auto_approve"

    # Output file exists and round-trips through schema validation
    assert output_path.exists()
    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    EISRecord.model_validate(on_disk)


def test_end_to_end_unknown_doc_key_raises(tmp_path) -> None:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"fixture not present: {FIXTURE_PATH}")
    with pytest.raises(KeyError):
        run_doc(
            "p1074_does_not_exist",
            docs_json_path=FIXTURE_PATH,
            nul_cache_dir=tmp_path / "nul_cache",
            llm=None,
            output_path=tmp_path / "out.json",
            write_ledger=False,
        )
