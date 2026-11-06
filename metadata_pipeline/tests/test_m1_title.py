"""
Tests for M1 title extraction (segment_a/m1.py).

`title` is graded `ok` on 8/8 docs, so it has no dedicated fix in MCAL_PLAN 1 and
no regression suite existed. These tests were added when the LLM fallback was
removed, because changing a field with zero observed failures is exactly the
change most likely to go unnoticed.

The removal, and the measurements behind it:

  * Over all 54,105 inventory rows, ZERO lack a title and zero are under 5
    chars. Length distribution: min 23 / median 91 / max 2809.
  * 155 rows (0.29%) exceed 500 chars. Every one is a bound-volume aggregate --
    a single catalogue record listing many technical reports, `;`-separated.
  * So the fallback's only real trigger was a case where the correct title was
    already in hand and was being discarded to re-derive it from OCR.
"""

from __future__ import annotations

import pytest

from mcal import settings  # installs the segment_a bridge

import m1  # noqa: E402


# --- No LLM, ever -----------------------------------------------------------


class TestNoLLM:
    def test_extract_title_makes_no_model_call(self, monkeypatch):
        """
        The whole point of the change. If anything reintroduces a model call
        here, this fails loudly rather than showing up as a surprise line on a
        2000-doc invoice.
        """
        import llm

        def explode(*a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("extract_title must not call an LLM")

        monkeypatch.setattr(llm, "call_json", explode)
        monkeypatch.setattr(llm, "call_with_usage", explode)
        monkeypatch.setattr(m1, "sonnet", explode, raising=False)

        got = m1.extract_title({"title": "Some Environmental Impact Statement"}, None)
        assert got["value"] == "Some Environmental Impact Statement"

    def test_doc_is_not_read(self):
        """`doc` is retained for call-site stability and must go unused."""
        assert m1.extract_title({"title": "A perfectly good EIS title"}, None)["value"]

    def test_haiku_tier_has_no_call_sites(self):
        """
        The `haiku` tier is now unused. Kept as an assertion rather than a
        comment because the tier still exists in llm.py and could be picked up
        again by accident.

        Uses the AST rather than a substring search: `llm.py` defines the helper,
        and `config.py` mentions `llm.haiku()` in a comment explaining why the
        tier is dead. Neither is a call.
        """
        import ast
        from pathlib import Path

        src_dir = Path(m1.__file__).parent
        offenders: list[str] = []
        for path in sorted(src_dir.glob("*.py")):
            if path.name == "llm.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                name = (
                    fn.id
                    if isinstance(fn, ast.Name)
                    else fn.attr
                    if isinstance(fn, ast.Attribute)
                    else None
                )
                if name == "haiku":
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, f"haiku() called from {offenders}"


# --- Normalization ----------------------------------------------------------


class TestNormalizeIndexTitle:
    def test_normal_title_passes_through(self):
        n = m1._normalize_index_title("Operation Breakthrough : environmental impact statement.")
        assert n["truncated"] is False
        assert n["reason"] == "ok"
        assert n["value"] == "Operation Breakthrough : environmental impact statement."

    def test_whitespace_collapsed(self):
        n = m1._normalize_index_title("Some   title\n with\tbreaks")
        assert n["value"] == "Some title with breaks"

    def test_bound_volume_aggregate_truncates_at_first_semicolon(self):
        """
        The real 2809-char Alaska OCS record: informative head, then a 36-volume
        manifest. Keep the head, keep the whole thing under `full_title`.
        """
        head = (
            "Alaska OCS (Outer Continental Shelf) socioeconomic studies program: "
            "Prudhoe Bay case study, technical report B1#4"
        )
        tail = "; ".join(f"Beaufort Sea region report B1#{i}" for i in range(30))
        raw = f"{head}; {tail}" + " x" * 400
        n = m1._normalize_index_title(raw)
        assert n["truncated"] is True
        assert n["reason"] == "bound_volume_aggregate"
        assert n["value"] == head
        assert n["full_title"] == " ".join(raw.split())
        assert n["n_parts"] > 1

    def test_single_long_title_cuts_on_word_boundary(self):
        """A half-word would read as OCR damage to a downstream consumer."""
        raw = "Supercalifragilistic " * 60  # >500 chars, no semicolons
        n = m1._normalize_index_title(raw)
        assert n["truncated"] is True
        assert n["reason"] == "over_length"
        assert len(n["value"]) <= m1.TITLE_MAX_CHARS + 1  # +1 for the ellipsis
        assert n["value"].endswith("\u2026")
        # no partial word before the ellipsis
        assert n["value"][:-1].rstrip().endswith("Supercalifragilistic")

    def test_absent(self):
        for raw in ("", "   ", None):
            n = m1._normalize_index_title(raw)
            assert n["value"] == ""
            assert n["reason"] == "absent"


# --- extract_title contract -------------------------------------------------


class TestExtractTitle:
    def test_high_confidence_from_index(self):
        got = m1.extract_title({"title": "Bad Creek pumped storage project"}, None)
        assert got["confidence"] == "high"
        assert "inventory index" in got["sources"][0]
        assert "status" not in got

    def test_nul_metadata_nesting_supported(self):
        got = m1.extract_title({"nul_metadata": {"title": "A valid EIS title here"}}, None)
        assert got["value"] == "A valid EIS title here"

    def test_list_valued_title(self):
        got = m1.extract_title({"title": ["First valid title", "second"]}, None)
        assert got["value"] == "First valid title"

    def test_truncated_marks_medium_and_keeps_full(self):
        raw = "Head of the record; " + "; ".join(f"volume {i} report" for i in range(90))
        got = m1.extract_title({"title": raw}, None)
        assert got["confidence"] == "medium"
        assert got["truncated"] is True
        assert got["full_title"] == " ".join(raw.split())
        assert "bound volumes" in got["note"] or "truncated" in got["note"]

    def test_absent_title_returns_explicit_status_not_a_guess(self):
        """
        MCAL_PLAN 1(8)'s rule, applied to title: never return empty silently.
        A guessed title is worse than a blank one -- this field is what people
        search and cite on, so a plausible-but-wrong value is the hardest kind of
        error for a reader to catch.
        """
        got = m1.extract_title({"title": ""}, None)
        assert got["value"] == ""
        assert got["status"] == "title_not_in_index"
        assert got["confidence"] == "low"

    def test_too_short_title_is_treated_as_absent(self):
        got = m1.extract_title({"title": "EIS"}, None)
        assert got["status"] == "title_not_in_index"

    @pytest.mark.parametrize(
        "raw", ["A" * 5, "A" * 499, "A" * 500]
    )
    def test_boundary_lengths_accepted(self, raw):
        got = m1.extract_title({"title": raw}, None)
        assert got["value"]
        assert "status" not in got


# --- Against the real inventory --------------------------------------------


@pytest.fixture(scope="module")
def index():
    """The full 54,105-row inventory. Loaded once -- it is a 41MB CSV."""
    import inventory

    return inventory.load_inventory()


@pytest.mark.skipif(
    not settings.INVENTORY_CSV_PATH.exists(), reason="inventory CSV not present"
)
class TestAgainstRealInventory:
    def test_every_row_yields_a_usable_title(self, index):
        """
        The measurement the removal rests on. If a future inventory refresh
        introduces rows with no title, this fails and the decision gets revisited
        rather than silently sending documents to human review.
        """
        failures = [
            acc
            for acc, rec in index.items()
            if not m1.extract_title({"title": rec.get("title")}, None)["value"]
        ]
        assert not failures, f"{len(failures)} rows yield no title, e.g. {failures[:5]}"

    def test_no_row_exceeds_the_cap(self, index):
        over = [
            acc
            for acc, rec in index.items()
            if len(m1.extract_title({"title": rec.get("title")}, None)["value"])
            > m1.TITLE_MAX_CHARS + 1
        ]
        assert not over

    def test_over_long_rows_are_all_bound_volume_aggregates(self, index):
        """
        Documents *why* truncating is the right call rather than re-deriving. If
        a future refresh brings over-long titles that are not volume manifests,
        this fails and the truncation strategy needs rethinking.
        """
        reasons = {}
        for rec in index.values():
            raw = rec.get("title") or ""
            if len(raw) <= m1.TITLE_MAX_CHARS:
                continue
            n = m1._normalize_index_title(raw)
            reasons[n["reason"]] = reasons.get(n["reason"], 0) + 1
        assert reasons, "expected some over-long rows"
        assert "over_length" not in reasons, (
            f"over-long titles that are not bound-volume aggregates: {reasons}"
        )

    def test_graded_docs_unchanged(self, index):
        """No regression on the 8 docs whose title was graded ok."""
        import inventory

        for doc_id, expected_prefix in [
            ("p1074_35556036058550", "Operation Breakthrough"),
            ("p1074_35556039563135", "Lincoln Hwy upgrade"),
            ("p1074_35556036861797", "Average fuel economy standard"),
            ("p1074_35556036811230", "Buffalo light rail"),
        ]:
            work = inventory.lookup_work(doc_id)
            assert work, f"{doc_id} not in inventory"
            got = m1.extract_title(work, None)
            assert got["confidence"] == "high"
            assert got["value"].startswith(expected_prefix), got["value"][:60]
