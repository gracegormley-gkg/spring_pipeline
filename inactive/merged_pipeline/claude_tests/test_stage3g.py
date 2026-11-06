"""
Stage 3g themes tests.
"""

from __future__ import annotations

from pipeline import config
from pipeline.schema import EISRecord, FieldWithStatus
from pipeline.stage3g_themes import run as stage3g_run


def _record_with_summary(value: str | None, status: str = "ok") -> EISRecord:
    rec = EISRecord(publication_id="t")
    rec.title = FieldWithStatus[str](value="Test EIS", status="ok")
    rec.summary = FieldWithStatus[str](value=value, status=status)
    return rec


class FakeLLM:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.dry_run = False
        self.models = {"haiku": "haiku", "sonnet": "sonnet", "opus": "opus"}

    def call_json(self, model, system, messages, max_tokens=2048, temperature=0.1, label=""):
        self.calls.append({"model": model, "label": label})
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Abstention paths (no LLM call)
# ---------------------------------------------------------------------------

def test_themes_skipped_when_summary_missing() -> None:
    rec = _record_with_summary(None, status="needs_review")
    fake_llm = FakeLLM([])  # any LLM call would raise
    stage3g_run(rec, llm=fake_llm)
    assert rec.themes.status == "skipped_summary_unavailable"
    assert rec.themes.primary == []
    assert rec.themes.subthemes == []
    assert fake_llm.calls == []


def test_themes_skipped_when_summary_status_not_ok() -> None:
    rec = _record_with_summary("some summary text", status="needs_review")
    fake_llm = FakeLLM([])
    stage3g_run(rec, llm=fake_llm)
    assert rec.themes.status == "skipped_summary_unavailable"
    assert fake_llm.calls == []


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_themes_happy_path() -> None:
    rec = _record_with_summary("Power transmission line in national forest.")
    fake_llm = FakeLLM([{
        "primary": [
            {"value": "energy_infrastructure", "confidence": 0.9},
            {"value": "land_management", "confidence": 0.7},
        ],
        "subthemes": [
            {"value": "electric_transmission", "confidence": 0.95, "parent": "energy_infrastructure"},
            {"value": "public_lands_planning", "confidence": 0.6, "parent": "land_management"},
        ],
    }])

    stage3g_run(rec, llm=fake_llm)
    assert rec.themes.status == "ok"
    assert {p.value for p in rec.themes.primary} == {"energy_infrastructure", "land_management"}
    assert {(s.value, s.parent) for s in rec.themes.subthemes} == {
        ("electric_transmission", "energy_infrastructure"),
        ("public_lands_planning", "land_management"),
    }


# ---------------------------------------------------------------------------
# Vocab gate: out-of-vocab + orphan + wrong-parent drops
# ---------------------------------------------------------------------------

def test_themes_drops_oov_primary() -> None:
    rec = _record_with_summary("a project")
    fake_llm = FakeLLM([{
        "primary": [
            {"value": "energy_infrastructure", "confidence": 0.9},
            {"value": "made_up_theme", "confidence": 0.5},  # OOV
        ],
        "subthemes": [],
    }])
    warnings = stage3g_run(rec, llm=fake_llm)
    assert {p.value for p in rec.themes.primary} == {"energy_infrastructure"}
    assert rec.themes.status == "needs_review"
    assert any("oov" in w for w in warnings)


def test_themes_drops_orphan_subtheme() -> None:
    """Subtheme parent isn't in the chosen primaries -> drop."""
    rec = _record_with_summary("a project")
    fake_llm = FakeLLM([{
        "primary": [{"value": "energy_infrastructure", "confidence": 0.9}],
        "subthemes": [
            # Orphan: parent='transportation' wasn't chosen as primary
            {"value": "highway", "confidence": 0.8, "parent": "transportation"},
            # Valid: parent is chosen
            {"value": "wind", "confidence": 0.9, "parent": "energy_infrastructure"},
        ],
    }])
    stage3g_run(rec, llm=fake_llm)
    assert {s.value for s in rec.themes.subthemes} == {"wind"}
    assert rec.themes.status == "needs_review"


def test_themes_drops_subtheme_with_wrong_parent() -> None:
    """Subtheme value is in vocab but listed under a DIFFERENT primary."""
    rec = _record_with_summary("a project")
    fake_llm = FakeLLM([{
        "primary": [
            {"value": "energy_infrastructure", "confidence": 0.9},
            {"value": "transportation", "confidence": 0.6},
        ],
        "subthemes": [
            # 'highway' is a valid subtheme under 'transportation', not 'energy_infrastructure'.
            # The model returned it with the wrong parent — should drop.
            {"value": "highway", "confidence": 0.8, "parent": "energy_infrastructure"},
        ],
    }])
    warnings = stage3g_run(rec, llm=fake_llm)
    assert rec.themes.subthemes == []
    assert any("wrong-parent" in w for w in warnings)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_themes_no_llm_marks_needs_review() -> None:
    rec = _record_with_summary("a project")
    stage3g_run(rec, llm=None)
    assert rec.themes.status == "needs_review"
    assert rec.themes.primary == []


def test_themes_llm_failure_marks_needs_review() -> None:
    class RaisingLLM:
        dry_run = False
        def call_json(self, *a, **k):
            raise RuntimeError("outage")

    rec = _record_with_summary("a project")
    stage3g_run(rec, llm=RaisingLLM())
    assert rec.themes.status == "needs_review"


def test_themes_llm_returns_empty() -> None:
    rec = _record_with_summary("a project")
    fake_llm = FakeLLM([{"primary": [], "subthemes": []}])
    stage3g_run(rec, llm=fake_llm)
    # Empty primary -> empty subthemes -> ok status, just nothing matched
    # (drops list is empty, so status stays ok)
    assert rec.themes.status == "ok"
    assert rec.themes.primary == []
    assert rec.themes.subthemes == []


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------

def test_themes_schema_round_trip() -> None:
    rec = _record_with_summary("test")
    fake_llm = FakeLLM([{
        "primary": [{"value": "energy_infrastructure", "confidence": 0.9}],
        "subthemes": [{"value": "wind", "confidence": 0.9, "parent": "energy_infrastructure"}],
    }])
    stage3g_run(rec, llm=fake_llm)
    EISRecord.model_validate(rec.model_dump())
