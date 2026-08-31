"""
The M2 summary-map model switch, and the llm.py retry/budget changes
(PIPELINE_REVIEW 1.1, 1.5).

1.1 is the largest cost lever in the pipeline: the map step asks Opus to
summarize one 50-page chunk and copy quotes out of it, which is extraction rather
than synthesis, at ~40% of Opus input tokens and a 5x price difference. The switch
exists; the default does NOT move. Changing an extractor's model shifts the score
distribution the frozen thresholds were fitted to, so it has to land between
calibration stages with a rerun and a refit -- and the quality claim is untested
on 1970s OCR, where a cheaper model would degrade by producing quotes that fail
verification, which the pipeline would then blame on the reduce step.

1.5, retry classification: the loop caught bare `Exception` and slept 1s then 2s
on everything, so a 400 or an auth failure burned three identical calls per
document.

1.5, token budgets: every `max_tokens` here is defended in a comment as "a cap,
not a target". That was asserted, never measured -- and the 6,000-token reduce
budget it replaced was also believed ample until it truncated 4 of 8 documents.
"""

from __future__ import annotations

import pytest

import config
import llm
import m2 as m2_mod


# --- 1.1 the switch ----------------------------------------------------------


class TestSummaryMapModelSwitch:
    def test_the_default_is_still_opus(self):
        """
        A guard, not a preference. If this flips, `weight_validation` and every
        bucket's tau were fitted on a different extractor than the one running.
        """
        assert config.SUMMARY_MAP_MODEL == "opus"
        assert m2_mod._summary_map_model() is m2_mod.opus

    def test_sonnet_is_honoured(self, monkeypatch):
        monkeypatch.setattr(config, "SUMMARY_MAP_MODEL", "sonnet")
        assert m2_mod._summary_map_model() is m2_mod.sonnet

    def test_it_is_read_at_call_time_not_import_time(self, monkeypatch):
        """
        `from config import SUMMARY_MAP_MODEL` would bind the value at import, so
        flipping the switch would silently do nothing.
        """
        monkeypatch.setattr(config, "SUMMARY_MAP_MODEL", "sonnet")
        assert m2_mod._summary_map_model() is m2_mod.sonnet
        monkeypatch.setattr(config, "SUMMARY_MAP_MODEL", "opus")
        assert m2_mod._summary_map_model() is m2_mod.opus

    @pytest.mark.parametrize("value", ["haiku", "", "gpt-4", None])
    def test_an_unknown_value_fails_loudly(self, monkeypatch, value):
        monkeypatch.setattr(config, "SUMMARY_MAP_MODEL", value)
        if value in ("", None):
            # Falsy falls back to opus rather than crashing a 2,000-doc run.
            assert m2_mod._summary_map_model() is m2_mod.opus
        else:
            with pytest.raises(ValueError, match="must be 'opus' or 'sonnet'"):
                m2_mod._summary_map_model()

    def test_case_and_whitespace_are_tolerated(self, monkeypatch):
        monkeypatch.setattr(config, "SUMMARY_MAP_MODEL", "  Sonnet ")
        assert m2_mod._summary_map_model() is m2_mod.sonnet

    def test_the_map_step_uses_the_switch(self, monkeypatch):
        """End to end: `_summary_map_one` must route through the resolver."""
        from chunk import Chunk

        calls = []
        monkeypatch.setattr(config, "SUMMARY_MAP_MODEL", "sonnet")
        monkeypatch.setattr(
            m2_mod, "sonnet",
            lambda **kw: calls.append("sonnet") or {},
        )
        monkeypatch.setattr(
            m2_mod, "opus", lambda **kw: calls.append("opus") or {},
        )
        m2_mod._summary_map_one(
            Chunk(index=0, text="t", start_page=1, end_page=2,
                  ceq_chapter=None, label=None)
        )
        assert calls == ["sonnet"]


# --- 1.5 retry classification ------------------------------------------------


class _Status(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.status_code = code


class _Nested(Exception):
    """The shape where the status hangs off `.response`."""

    class _R:
        def __init__(self, code):
            self.status_code = code

    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.response = self._R(code)


class TestRetryClassification:
    @pytest.mark.parametrize("code", [429, 500, 502, 503, 529])
    def test_transient_statuses_retry(self, code):
        assert llm.is_retryable(_Status(code)) is True
        assert llm.is_retryable(_Nested(code)) is True

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 413, 422])
    def test_client_errors_do_not_retry(self, code):
        """
        These will not change within three attempts. Retrying burned two extra
        calls and three seconds of sleep per document to reach the same error.
        """
        assert llm.is_retryable(_Status(code)) is False
        assert llm.is_retryable(_Nested(code)) is False

    def test_transport_failures_retry(self):
        assert llm.is_retryable(TimeoutError("timed out")) is True
        assert llm.is_retryable(ConnectionError("reset by peer")) is True

    def test_a_programming_error_does_not_retry(self):
        assert llm.is_retryable(TypeError("bad kwarg")) is False
        assert llm.is_retryable(KeyError("nope")) is False

    def test_truncation_is_handled_separately(self):
        """`TruncatedResponse` short-circuits above the classifier: same budget,
        same truncation."""
        exc = llm.TruncatedResponse("m", 100, 100, "partial")
        assert llm.is_retryable(exc) is False


# --- 1.5 output-token headroom ----------------------------------------------


class TestOutputHeadroom:
    def test_headroom_is_recorded_per_call(self):
        usages = [
            {"model": "opus", "input_tokens": 10, "output_tokens": 900,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "max_tokens": 1000, "output_headroom": 0.1},
        ]
        agg = llm.aggregate_usages(usages)
        by_model = {m["model"]: m for m in agg["by_model"]}
        assert by_model["opus"]["max_output_tokens_seen"] == 900
        assert by_model["opus"]["min_output_headroom"] == 0.1

    def test_thin_headroom_calls_are_counted(self):
        usages = [
            {"model": "opus", "output_tokens": 990, "input_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "max_tokens": 1000, "output_headroom": 0.01},
            {"model": "opus", "output_tokens": 100, "input_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "max_tokens": 1000, "output_headroom": 0.9},
        ]
        agg = llm.aggregate_usages(usages)
        by_model = {m["model"]: m for m in agg["by_model"]}
        assert by_model["opus"]["n_calls_thin_headroom"] == 1
        assert by_model["opus"]["min_output_headroom"] == 0.01

    def test_usages_without_headroom_are_tolerated(self):
        """Every usage dict already on disk predates these keys."""
        agg = llm.aggregate_usages(
            [{"model": "opus", "input_tokens": 5, "output_tokens": 5,
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}]
        )
        by_model = {m["model"]: m for m in agg["by_model"]}
        assert by_model["opus"]["min_output_headroom"] is None
        assert by_model["opus"]["n_calls_thin_headroom"] == 0

    def test_a_thin_budget_warns(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="llm"):
            llm._warn_on_thin_headroom(
                {"output_tokens": 995, "output_headroom": 0.005}, "opus", 1000
            )
        assert "Raise max_tokens" in caplog.text

    def test_a_healthy_budget_is_silent(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="llm"):
            llm._warn_on_thin_headroom(
                {"output_tokens": 100, "output_headroom": 0.9}, "opus", 1000
            )
        assert caplog.text == ""
