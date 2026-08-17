"""
Tests for segment_b/postproc/location_pipeline.py (MCAL_PLAN 3.9 / 3.9a, item #8).

No network and no LLM calls: every geocoder hop is monkeypatched through the
`HOPS` registry and every Sonnet call through the module-level `sonnet` name.
Reduced mode (Census + Nominatim only) is exercised end to end, because that is
the only stack configured on this machine and therefore the only path that runs
in practice until PAD-US/GNIS/MAPBOX_TOKEN are downloaded.

Regression anchors, keyed to MCAL_PLAN 1(9):
  1(9b) wrong specificity -> TestSpecificityCascade
  1(9c) multi-site partial -> TestSitePairing, TestSiteScope
  1(9d) national rulemaking -> TestPlacelessScopes + TestFuelEconomyRegression
  the `_geocode_places` index-misalignment bug -> TestSitePairing
"""

from __future__ import annotations

import pytest

from segment_b.postproc import location_pipeline as lp

from conftest import BUFFALO, LINCOLN_HWY

# MCAL_PLAN 1(9d): the national CAFE rulemaking graded as "no location".
# Not in conftest's constant set, so it is named here.
FUEL_ECONOMY = "p1074_35556036861797"


# --- helpers ----------------------------------------------------------------


def make_doc(page_texts: list[str], doc_id: str = "synthetic"):
    """A Doc with correct page offsets, built the way pages.load_doc builds one."""
    from pages import PAGE_SEP, Doc, Page

    pages = [Page(page_num=i + 1, text=t) for i, t in enumerate(page_texts)]
    starts: list[int] = []
    cursor = 0
    for i, p in enumerate(pages):
        starts.append(cursor)
        cursor += len(p.text)
        if i < len(pages) - 1:
            cursor += len(PAGE_SEP)
    return Doc(
        doc_id=doc_id,
        pages=pages,
        full_text=PAGE_SEP.join(p.text for p in pages),
        _page_starts=starts,
    )


def result(lat, lon, bbox=None, *, source="test", confidence=0.9, level=None, admin=None):
    return lp._result(
        lat=lat,
        lon=lon,
        bbox=bbox,
        source=source,
        confidence=confidence,
        level=level,
        admin_hierarchy=admin,
    )


def boom(*a, **kw):  # a hop that must never be called
    raise AssertionError("this geocoder hop should not have been called")


def kill_all_hops(monkeypatch):
    for name in list(lp.HOPS):
        monkeypatch.setitem(lp.HOPS, name, boom)


@pytest.fixture(autouse=True)
def _reset_local_asset_caches():
    """
    The GNIS and PAD-US indexes are process-wide singletons loaded once and
    remembered (including their failures). Tests point them at temp files, so
    clear them around every test or the suite becomes order-dependent.
    """
    lp._GNIS.reset()
    lp._PADUS.reset()
    yield
    lp._GNIS.reset()
    lp._PADUS.reset()


# --- bbox geometry ----------------------------------------------------------


class TestBboxGeometry:
    def test_strict_containment(self):
        assert lp.bbox_contained((0, 0, 1, 1), (-1, -1, 2, 2))

    def test_disjoint_is_not_contained(self):
        assert not lp.bbox_contained((10, 10, 11, 11), (0, 0, 1, 1))

    def test_inner_larger_than_outer_rejected(self):
        """A 'city' bigger than its 'county' means the geocoder crossed levels."""
        assert not lp.bbox_contained((-5, -5, 5, 5), (0, 0, 1, 1))

    def test_near_containment_tolerates_vendor_slop(self):
        """90% inside: two vendors' boxes for the same city, one synthetic."""
        inner = (0.0, 0.0, 1.0, 1.0)
        outer = (0.05, -1.0, 3.0, 3.0)   # clips 5% off the left edge
        assert lp.bbox_contained(inner, outer)

    def test_mostly_outside_is_rejected(self):
        inner = (0.0, 0.0, 1.0, 1.0)
        outer = (0.6, -1.0, 3.0, 3.0)    # only 40% of inner is inside
        assert not lp.bbox_contained(inner, outer)

    def test_zero_area_inner_uses_point_in_polygon(self):
        assert lp.bbox_contained((5, 5, 5, 5), (0, 0, 10, 10))
        assert not lp.bbox_contained((50, 5, 50, 5), (0, 0, 10, 10))

    def test_intersection_for_corridors(self):
        assert lp.bbox_intersects((0, 0, 2, 2), (1, 1, 3, 3))
        assert not lp.bbox_intersects((0, 0, 1, 1), (5, 5, 6, 6))

    def test_point_in_bbox(self):
        assert lp.point_in_bbox(43.0, -78.8, (-79.0, 42.0, -78.0, 44.0))
        assert not lp.point_in_bbox(43.0, -70.0, (-79.0, 42.0, -78.0, 44.0))

    def test_synthetic_bboxes_are_ordered_by_level(self):
        """
        The whole specificity cascade rests on this ordering: a synthesized city
        box must be smaller than a county box, which must be smaller than a state
        box, or containment can never discriminate.
        """
        areas = []
        for level in ("poi", "neighborhood", "city", "county", "state"):
            x0, y0, x1, y1 = lp.synth_bbox(40.0, -80.0, level)
            areas.append((x1 - x0) * (y1 - y0))
        assert areas == sorted(areas)

    def test_bbox_centroid(self):
        assert lp.bbox_centroid((-2, -2, 2, 2)) == (0.0, 0.0)

    def test_points_centroid_of_two_points_is_the_midpoint(self):
        got = lp.points_centroid([(40.0, -80.0), (42.0, -78.0)])
        assert got == pytest.approx((41.0, -79.0))

    def test_points_centroid_empty(self):
        assert lp.points_centroid([]) is None

    def test_invalid_bbox_never_raises(self):
        assert lp.bbox_contained(None, (0, 0, 1, 1)) is False
        assert lp.bbox_contained(("x", 0, 1, 1), (0, 0, 1, 1)) is False


# --- admin hierarchy normalization ------------------------------------------


class TestAdminHierarchy:
    def test_positional_list_from_the_plan_is_accepted(self):
        got = lp.coerce_admin_hierarchy(
            ["Dam Site", None, "Vernal", "Uintah County", "Utah", "United States"]
        )
        assert got["poi"] == "Dam Site"
        assert got["city"] == "Vernal"
        assert got["state"] == "Utah"
        assert got["neighborhood"] is None

    def test_dict_is_accepted_and_unknown_keys_dropped(self):
        got = lp.coerce_admin_hierarchy({"city": "Buffalo", "planet": "Earth"})
        assert got["city"] == "Buffalo"
        assert set(got) == set(lp.ADMIN_LEVELS)

    def test_junk_yields_all_nulls(self):
        assert lp.coerce_admin_hierarchy("Buffalo") == {k: None for k in lp.ADMIN_LEVELS}

    @pytest.mark.parametrize("raw", ["null", "None", "n/a", "unknown", "  ", "-"])
    def test_model_spellings_of_null(self, raw):
        assert lp.clean_place_name(raw) is None

    def test_ocr_tolerant_name_match(self):
        assert lp.names_match("Modoc National Forest", "M0doc Nati0nal F0rest")
        assert not lp.names_match("Buffalo", "Amherst")


# --- specificity cascade (MCAL_PLAN 1(9b)) ----------------------------------


class TestSpecificityCascade:
    """
    Synthetic bboxes only. Each test wires a fake cascade keyed on `level` so the
    acceptance rule is exercised in isolation from any vendor behaviour.
    """

    @staticmethod
    def fake_cascade(by_level: dict[str, dict]):
        def _fake(query, *, level=None, state=None, stack=None, stats=None):
            return by_level.get(level)

        return _fake

    def test_accepts_finest_contained_level(self, monkeypatch):
        by_level = {
            "poi": result(43.0, -78.8, (-78.9, 42.9, -78.7, 43.1), level="poi"),
            "city": result(43.0, -78.85, (-79.0, 42.8, -78.6, 43.2), level="city"),
            "county": result(42.9, -78.8, (-79.2, 42.5, -78.4, 43.4), level="county"),
            "state": result(43.0, -75.0, (-80.0, 40.5, -71.8, 45.0), level="state"),
        }
        monkeypatch.setattr(lp, "geocode_cascade", self.fake_cascade(by_level))
        site = lp.Site(
            name="Erie Basin Marina",
            admin_hierarchy=lp.coerce_admin_hierarchy(
                {
                    "poi": "Erie Basin Marina",
                    "city": "Buffalo",
                    "county": "Erie County",
                    "state": "New York",
                }
            ),
        )
        accepted, tags, note = lp.resolve_site(site)
        assert accepted["accepted_level"] == "poi"
        assert accepted["acceptance_reason"] == "bbox_contained_in_city"
        assert tags == []

    def test_poi_wins_when_containing_city_matches(self, monkeypatch):
        """
        POI bbox deliberately NOT contained in the city bbox, but the POI's own
        containing city agrees with the document -- name agreement beats geometry
        here because vendor POI boxes are often a bare point.
        """
        by_level = {
            "poi": result(
                43.0,
                -78.8,
                (-90.0, 30.0, -70.0, 50.0),          # absurdly large
                level="poi",
                admin={"city": "Buffalo"},
            ),
            "city": result(43.0, -78.85, (-79.0, 42.8, -78.6, 43.2), level="city"),
        }
        monkeypatch.setattr(lp, "geocode_cascade", self.fake_cascade(by_level))
        site = lp.Site(
            name="Erie Basin Marina",
            admin_hierarchy=lp.coerce_admin_hierarchy(
                {"poi": "Erie Basin Marina", "city": "Buffalo"}
            ),
        )
        accepted, tags, _ = lp.resolve_site(site)
        assert accepted["accepted_level"] == "poi"
        assert accepted["acceptance_reason"] == "poi_city_matches_document_city"

    def test_uncontained_poi_falls_through_to_city(self, monkeypatch):
        """The Airport Spur shape: a bogus POI must not win, but the city can."""
        by_level = {
            "poi": result(10.0, 10.0, (9.9, 9.9, 10.1, 10.1), level="poi"),
            "city": result(43.0, -87.9, (-88.1, 42.9, -87.8, 43.2), level="city"),
            "county": result(43.0, -88.0, (-88.3, 42.7, -87.7, 43.3), level="county"),
        }
        monkeypatch.setattr(lp, "geocode_cascade", self.fake_cascade(by_level))
        site = lp.Site(
            name="Airport Spur",
            admin_hierarchy=lp.coerce_admin_hierarchy(
                {"poi": "Airport Spur", "city": "Milwaukee", "county": "Milwaukee County"}
            ),
        )
        accepted, tags, _ = lp.resolve_site(site)
        assert accepted["accepted_level"] == "city"
        assert tags == []

    def test_state_only_resolution_is_tagged_T07(self, monkeypatch):
        by_level = {"state": result(43.0, -75.0, (-80.0, 40.5, -71.8, 45.0), level="state")}
        monkeypatch.setattr(lp, "geocode_cascade", self.fake_cascade(by_level))
        site = lp.Site(
            name="Randolph",
            admin_hierarchy=lp.coerce_admin_hierarchy(
                {"city": "Randolph", "state": "New York"}
            ),
        )
        accepted, tags, note = lp.resolve_site(site)
        assert accepted["accepted_level"] == "state"
        assert lp.T_WRONG_SPECIFICITY in tags
        assert "Coarsest-level-only" in note

    def test_nothing_contained_keeps_coarsest_and_tags_T07(self, monkeypatch):
        by_level = {
            "city": result(43.0, -78.8, (-79.0, 42.8, -78.6, 43.2), level="city"),
            "county": result(10.0, 10.0, (9.0, 9.0, 11.0, 11.0), level="county"),
        }
        monkeypatch.setattr(lp, "geocode_cascade", self.fake_cascade(by_level))
        site = lp.Site(
            name="Randolph",
            admin_hierarchy=lp.coerce_admin_hierarchy(
                {"city": "Randolph", "county": "Cattaraugus County"}
            ),
        )
        accepted, tags, note = lp.resolve_site(site)
        assert accepted["accepted_level"] == "county"
        assert accepted["acceptance_reason"] == "no_level_passed_containment"
        assert lp.T_WRONG_SPECIFICITY in tags

    def test_single_level_accepted_without_containment_check(self, monkeypatch):
        by_level = {"city": result(43.0, -78.8, level="city")}
        monkeypatch.setattr(lp, "geocode_cascade", self.fake_cascade(by_level))
        site = lp.Site(
            name="Buffalo", admin_hierarchy=lp.coerce_admin_hierarchy({"city": "Buffalo"})
        )
        accepted, tags, _ = lp.resolve_site(site)
        assert accepted["accepted_level"] == "city"
        assert accepted["acceptance_reason"] == "no_coarser_level_available"
        assert tags == []

    def test_no_hop_resolves_tags_T06(self, monkeypatch):
        monkeypatch.setattr(lp, "geocode_cascade", self.fake_cascade({}))
        site = lp.Site(
            name="Randolph", admin_hierarchy=lp.coerce_admin_hierarchy({"city": "Randolph"})
        )
        accepted, tags, note = lp.resolve_site(site)
        assert accepted is None
        assert tags == [lp.T_GEOCODE_MISSING]
        assert "Randolph" in note

    def test_no_admin_hierarchy_at_all(self, monkeypatch):
        monkeypatch.setattr(lp, "geocode_cascade", self.fake_cascade({}))
        site = lp.Site(name="", admin_hierarchy=lp.coerce_admin_hierarchy({}))
        accepted, tags, _ = lp.resolve_site(site)
        assert accepted is None and tags == [lp.T_GEOCODE_MISSING]

    def test_level_queries_carry_coarser_context(self):
        site = lp.Site(
            name="Cottonwood Field Office",
            admin_hierarchy=lp.coerce_admin_hierarchy(
                {"city": "Vernal", "county": "Uintah County", "state": "Utah"}
            ),
        )
        queries = dict(lp._level_queries(site))
        assert queries["city"] == "Vernal, Uintah County, Utah"
        assert queries["state"] == "Utah"
        # The site's own name becomes the poi query when the model left poi empty.
        assert queries["poi"] == "Cottonwood Field Office, Vernal, Uintah County, Utah"


# --- cascade ordering and reduced mode --------------------------------------


class TestCascadeOrder:
    def test_reduced_stack_only_uses_census_and_nominatim(self, monkeypatch):
        called: list[str] = []

        def hop(name, conf=0.0):
            def _fn(query, *, level=None, state=None):
                called.append(name)
                return result(1.0, 2.0, source=name, confidence=conf) if conf else None

            return _fn

        monkeypatch.setitem(lp.HOPS, "census", hop("census"))
        monkeypatch.setitem(lp.HOPS, "nominatim", hop("nominatim", 0.7))
        for absent in ("gnis", "padus", "mapbox"):
            monkeypatch.setitem(lp.HOPS, absent, boom)

        got = lp.geocode_cascade("Randolph, New York", stack="reduced")
        assert called == ["census", "nominatim"]
        assert got["source"] == "nominatim"

    def test_confident_hop_short_circuits_the_rest(self, monkeypatch):
        called: list[str] = []

        def census(query, *, level=None, state=None):
            called.append("census")
            return result(1.0, 2.0, source="census", confidence=0.9)

        monkeypatch.setitem(lp.HOPS, "census", census)
        monkeypatch.setitem(lp.HOPS, "nominatim", boom)
        got = lp.geocode_cascade("Buffalo, New York", stack="reduced")
        assert got["confident"] is True
        assert called == ["census"]

    def test_low_confidence_hop_does_not_stop_the_cascade(self, monkeypatch):
        """
        A weak global-gazetteer hit must not pre-empt a stronger US-specific one.
        Both hops run; the better result wins and is flagged as non-confident
        only if nothing cleared the threshold.
        """
        monkeypatch.setitem(
            lp.HOPS,
            "census",
            lambda q, *, level=None, state=None: result(
                1.0, 2.0, source="census", confidence=0.55
            ),
        )
        monkeypatch.setitem(
            lp.HOPS,
            "nominatim",
            lambda q, *, level=None, state=None: result(
                3.0, 4.0, source="nominatim", confidence=0.58
            ),
        )
        got = lp.geocode_cascade("x", stack="reduced")
        assert got["source"] == "nominatim"
        assert got["confident"] is False

    def test_hop_exception_is_counted_not_raised(self, monkeypatch):
        stats = lp.HopStats()
        monkeypatch.setitem(lp.HOPS, "census", boom)
        monkeypatch.setitem(
            lp.HOPS,
            "nominatim",
            lambda q, *, level=None, state=None: result(
                1.0, 2.0, source="nominatim", confidence=0.8
            ),
        )
        got = lp.geocode_cascade("x", stack="reduced", stats=stats)
        assert got["source"] == "nominatim"
        assert stats.errors["census"] == 1
        assert stats.hits["nominatim"] == 1

    def test_hop_stats_hit_rates(self, monkeypatch):
        stats = lp.HopStats()
        monkeypatch.setitem(lp.HOPS, "census", lambda q, *, level=None, state=None: None)
        monkeypatch.setitem(
            lp.HOPS,
            "nominatim",
            lambda q, *, level=None, state=None: result(1.0, 2.0, confidence=0.9),
        )
        lp.geocode_cascade("a", stack="reduced", stats=stats)
        lp.geocode_cascade("b", stack="reduced", stats=stats)
        rates = stats.hit_rates()
        assert rates["census"] == 0.0
        assert rates["nominatim"] == 1.0
        assert stats.to_dict()["attempts"] == {"census": 2, "nominatim": 2}

    def test_empty_query_short_circuits(self, monkeypatch):
        kill_all_hops(monkeypatch)
        assert lp.geocode_cascade("   ", stack="reduced") is None

    def test_full_stack_order_matches_the_plan(self):
        assert lp.FULL_CASCADE == ("census", "gnis", "padus", "mapbox", "nominatim")
        assert lp.cascade_for_stack("reduced") == ("census", "nominatim")

    def test_resolve_stack_defaults_to_the_precheck(self):
        assert lp.resolve_stack() in ("full", "reduced")
        assert lp.resolve_stack("full") == "full"
        assert lp.resolve_stack("nonsense") in ("full", "reduced")


class TestGnisIndex:
    """The GNIS hop reads a local TSV; a tiny synthetic one exercises the parser."""

    def _write(self, tmp_path, text):
        p = tmp_path / "gnis.txt"
        p.write_text(text, encoding="utf-8")
        return p

    def test_legacy_schema_pipe_delimited(self, tmp_path, monkeypatch):
        path = self._write(
            tmp_path,
            "FEATURE_ID|FEATURE_NAME|FEATURE_CLASS|STATE_ALPHA|COUNTY_NAME|PRIM_LAT_DEC|PRIM_LONG_DEC\n"
            "1|Ashley National Forest|Forest|UT|Uintah|40.6|-109.9\n"
            "2|Randolph|Populated Place|NY|Cattaraugus|42.16|-78.97\n"
            "3|Randolph|Populated Place|UT|Rich|41.66|-111.18\n",
        )
        monkeypatch.setattr(lp.settings, "gnis_path", lambda: path)
        lp._GNIS.reset()

        got = lp.geocode_gnis("Ashley National Forest")
        assert got["source"] == "gnis"
        assert got["lat"] == pytest.approx(40.6)
        assert got["confidence"] >= lp.CONFIDENT_MIN
        assert got["level"] == "county"          # Forest class -> county-sized box

    def test_state_hint_disambiguates(self, tmp_path, monkeypatch):
        path = self._write(
            tmp_path,
            "feature_id|feature_name|feature_class|state_alpha|county_name|prim_lat_dec|prim_long_dec\n"
            "2|Randolph|Populated Place|NY|Cattaraugus|42.16|-78.97\n"
            "3|Randolph|Populated Place|UT|Rich|41.66|-111.18\n",
        )
        monkeypatch.setattr(lp.settings, "gnis_path", lambda: path)
        lp._GNIS.reset()

        ny = lp.geocode_gnis("Randolph", state="NY")
        assert ny["lat"] == pytest.approx(42.16)
        assert ny["confidence"] >= lp.CONFIDENT_MIN

        # Without the hint the name is ambiguous, so the hop stays below the
        # confidence floor and the cascade is allowed to keep looking.
        ambiguous = lp.geocode_gnis("Randolph")
        assert ambiguous["confidence"] < lp.CONFIDENT_MIN
        assert ambiguous["n_candidates"] == 2

    def test_ocr_damaged_query_still_matches(self, tmp_path, monkeypatch):
        path = self._write(
            tmp_path,
            "feature_name|feature_class|state_alpha|prim_lat_dec|prim_long_dec\n"
            "Modoc National Forest|Forest|CA|41.5|-120.5\n",
        )
        monkeypatch.setattr(lp.settings, "gnis_path", lambda: path)
        lp._GNIS.reset()
        assert lp.geocode_gnis("M0doc Nati0nal F0rest") is not None

    def test_missing_file_degrades(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lp.settings, "gnis_path", lambda: tmp_path / "nope.txt")
        lp._GNIS.reset()
        assert lp.geocode_gnis("Anything") is None
        assert "missing" in (lp._GNIS.error or "")

    def test_unconfigured_asset_degrades(self, monkeypatch):
        monkeypatch.setattr(lp.settings, "gnis_path", lambda: None)
        lp._GNIS.reset()
        assert lp.geocode_gnis("Anything") is None
        assert lp._GNIS.error == "GNIS_TSV_PATH not configured"

    def test_malformed_header_degrades(self, tmp_path, monkeypatch):
        path = self._write(tmp_path, "a|b|c\n1|2|3\n")
        monkeypatch.setattr(lp.settings, "gnis_path", lambda: path)
        lp._GNIS.reset()
        assert lp.geocode_gnis("Anything") is None
        assert "GNIS load failed" in (lp._GNIS.error or "")


class TestPadusHop:
    """PAD-US matching runs off plain records, so no geodatabase is needed."""

    UNITS = [
        lp.PadusUnit(
            name="Ashley National Forest",
            manager="USFS",
            lat=40.6,
            lon=-109.9,
            bbox=(-111.0, 40.0, -109.0, 41.2),
        ),
        lp.PadusUnit(
            name="Cottonwood Field Office",
            manager="BLM",
            lat=39.1,
            lon=-110.8,
            bbox=(-111.5, 38.5, -110.0, 39.7),
        ),
    ]

    def test_exact_name_is_confident_and_keeps_real_extent(self, monkeypatch):
        monkeypatch.setattr(lp, "padus_units", lambda: self.UNITS)
        got = lp.geocode_padus("Ashley National Forest")
        assert got["source"] == "padus"
        assert got["bbox_synthetic"] is False
        assert got["confidence"] >= lp.CONFIDENT_MIN
        assert got["manager"] == "USFS"

    def test_fuzzy_name_stays_below_the_confidence_floor(self, monkeypatch):
        monkeypatch.setattr(lp, "padus_units", lambda: self.UNITS)
        got = lp.geocode_padus("Cottonwood Field Ofice")   # OCR drop
        assert got is not None
        assert got["confidence"] < lp.CONFIDENT_MIN

    def test_unrelated_name_misses(self, monkeypatch):
        monkeypatch.setattr(lp, "padus_units", lambda: self.UNITS)
        assert lp.geocode_padus("Buffalo Light Rail") is None

    def test_no_geodatabase_degrades(self, monkeypatch):
        monkeypatch.setattr(lp, "padus_units", lambda: [])
        assert lp.geocode_padus("Ashley National Forest") is None

    @pytest.mark.parametrize(
        "manager,expected",
        [
            ("USFS", True),
            ("Bureau of Land Management", True),
            ("BLM", True),
            ("City of Buffalo", False),
            ("", False),
        ],
    )
    def test_federal_manager_filter(self, manager, expected):
        assert lp._is_federal_manager(manager) is expected


# --- corridor endpoint parsing ----------------------------------------------


class TestCorridorEndpointParser:
    @pytest.mark.parametrize(
        "text,a,b",
        [
            (
                "The proposed highway extends from Akron, Ohio to Cleveland, Ohio.",
                "Akron, Ohio",
                "Cleveland, Ohio",
            ),
            (
                "A rail line is proposed between Buffalo and Amherst in Erie County.",
                "Buffalo",
                "Amherst",
            ),
            (
                "Improvements to the Milwaukee-Racine segment are evaluated.",
                "Milwaukee",
                "Racine",
            ),
            (
                "The Lincoln Highway corridor runs from Aurora to Geneva.",
                "Aurora",
                "Geneva",
            ),
        ],
    )
    def test_patterns(self, text, a, b):
        got = lp.parse_corridor_endpoints(text)
        assert got is not None
        assert (got.endpoint_a, got.endpoint_b) == (a, b)

    def test_via_is_captured(self):
        got = lp.parse_corridor_endpoints(
            "from Akron, Ohio to Cleveland, Ohio via Hudson along the existing route"
        )
        assert got.via == "Hudson"

    def test_stronger_pattern_wins_over_incidental_mention(self):
        text = (
            "Chicago to Denver is discussed in the references. "
            "The action extends from Vernal, Utah to Rangely, Colorado."
        )
        got = lp.parse_corridor_endpoints(text)
        assert got.endpoint_a == "Vernal, Utah"
        assert got.pattern == "from_to"

    @pytest.mark.parametrize(
        "text",
        [
            "This Final Environmental Impact Statement to Chapter Two",
            "from the alternatives to the proposed action",
            "",
            "no corridor language at all here",
            "from Buffalo to Buffalo",     # degenerate: same endpoint twice
        ],
    )
    def test_rejects_non_places(self, text):
        assert lp.parse_corridor_endpoints(text) is None

    @pytest.mark.parametrize(
        "candidate,ok",
        [
            ("Akron, Ohio", True),
            ("Salt Lake City", True),
            ("the", False),
            ("Environmental Impact", False),
            ("Ch", False),
            ("A" * 80, False),
        ],
    )
    def test_place_plausibility(self, candidate, ok):
        assert lp.looks_like_place(candidate) is ok


class TestCorridorResolution:
    def _cascade(self, table):
        def _fake(query, *, level=None, state=None, stack=None, stats=None):
            return table.get(query)

        return _fake

    def test_three_points_with_interpolated_midpoint(self, monkeypatch):
        table = {
            "Akron, Ohio": result(41.08, -81.52, (-81.6, 41.0, -81.4, 41.2), level="city"),
            "Cleveland, Ohio": result(
                41.50, -81.69, (-81.8, 41.4, -81.6, 41.6), level="city"
            ),
            "Ohio": result(40.4, -82.9, (-84.9, 38.4, -80.5, 42.0), level="state"),
        }
        monkeypatch.setattr(lp, "geocode_cascade", self._cascade(table))
        endpoints = lp.CorridorEndpoints("Akron, Ohio", "Cleveland, Ohio")
        sites = [
            lp.Site(
                name="Akron",
                admin_hierarchy=lp.coerce_admin_hierarchy({"state": "Ohio"}),
            )
        ]
        points, tags, note = lp.resolve_corridor(endpoints, sites)
        assert [p["point_role"] for p in points] == [
            "endpoint_a",
            "midpoint",
            "endpoint_b",
        ]
        assert all(p["corridor"] for p in points)
        mid = points[1]
        assert mid["source"] == "interpolated"
        assert mid["lat"] == pytest.approx((41.08 + 41.50) / 2)
        assert all(p["container_ok"] for p in points)
        assert tags == []

    def test_named_via_is_geocoded_not_interpolated(self, monkeypatch):
        table = {
            "Akron, Ohio": result(41.08, -81.52, level="city"),
            "Cleveland, Ohio": result(41.50, -81.69, level="city"),
            "Hudson": result(41.24, -81.44, level="city"),
        }
        monkeypatch.setattr(lp, "geocode_cascade", self._cascade(table))
        endpoints = lp.CorridorEndpoints("Akron, Ohio", "Cleveland, Ohio", via="Hudson")
        points, _, _ = lp.resolve_corridor(endpoints, [])
        mid = [p for p in points if p["point_role"] == "midpoint"][0]
        assert mid["lat"] == pytest.approx(41.24)
        assert mid["source"] != "interpolated"

    def test_endpoint_outside_the_container_is_flagged(self, monkeypatch):
        table = {
            "Akron, Ohio": result(41.08, -81.52, (-81.6, 41.0, -81.4, 41.2), level="city"),
            "Cleveland, Ohio": result(10.0, 10.0, (9.9, 9.9, 10.1, 10.1), level="city"),
            "Ohio": result(40.4, -82.9, (-84.9, 38.4, -80.5, 42.0), level="state"),
        }
        monkeypatch.setattr(lp, "geocode_cascade", self._cascade(table))
        endpoints = lp.CorridorEndpoints("Akron, Ohio", "Cleveland, Ohio")
        sites = [
            lp.Site(name="Akron", admin_hierarchy=lp.coerce_admin_hierarchy({"state": "Ohio"}))
        ]
        points, tags, note = lp.resolve_corridor(endpoints, sites)
        bad = [p for p in points if p["point_role"] == "endpoint_b"][0]
        assert bad["container_ok"] is False
        assert "container check" in note

    def test_single_endpoint_resolved_tags_partial(self, monkeypatch):
        table = {"Akron, Ohio": result(41.08, -81.52, level="city")}
        monkeypatch.setattr(lp, "geocode_cascade", self._cascade(table))
        points, tags, note = lp.resolve_corridor(
            lp.CorridorEndpoints("Akron, Ohio", "Nowhere, Ohio"), []
        )
        assert lp.T_MULTI_SITE_PARTIAL in tags
        assert [p["point_role"] for p in points] == ["endpoint_a"]

    def test_no_endpoint_resolved_tags_missing(self, monkeypatch):
        monkeypatch.setattr(lp, "geocode_cascade", self._cascade({}))
        points, tags, _ = lp.resolve_corridor(lp.CorridorEndpoints("A Place", "B Place"), [])
        assert points == []
        assert lp.T_GEOCODE_MISSING in tags


# --- regional scope ---------------------------------------------------------


class TestRegionalScope:
    def test_stated_region_is_queried_first(self, monkeypatch):
        def fake(query, *, level=None, state=None, stack=None, stats=None):
            return result(34.0, -117.0, level="region") if query == "Southern California" else None

        monkeypatch.setattr(lp, "geocode_cascade", fake)
        points, tags, note, hr = lp.resolve_regional("Southern California", [])
        assert points[0]["point_role"] == "region"
        assert tags == [] and hr is False

    def test_centroid_fallback_over_primary_sites(self, monkeypatch):
        table = {
            "Seattle": result(47.6, -122.3, level="city"),
            "Tacoma": result(47.25, -122.44, level="city"),
        }

        def fake(query, *, level=None, state=None, stack=None, stats=None):
            return table.get(query)

        monkeypatch.setattr(lp, "geocode_cascade", fake)
        sites = [
            lp.Site(name="Seattle", admin_hierarchy=lp.coerce_admin_hierarchy({"city": "Seattle"})),
            lp.Site(name="Tacoma", admin_hierarchy=lp.coerce_admin_hierarchy({"city": "Tacoma"})),
        ]
        points, tags, note, hr = lp.resolve_regional("Puget Sound region", sites)
        assert points[0]["source"] == "site_polygon_centroid"
        assert points[0]["lat"] == pytest.approx((47.6 + 47.25) / 2)
        assert hr is False

    def test_fewer_than_two_primary_sites_tags_T14_and_human_review(self, monkeypatch):
        monkeypatch.setattr(
            lp, "geocode_cascade", lambda *a, **k: None
        )
        sites = [lp.Site(name="Somewhere", admin_hierarchy=lp.coerce_admin_hierarchy({}))]
        points, tags, note, hr = lp.resolve_regional(None, sites)
        assert points == []
        assert tags == [lp.T_REGIONAL_UNDERSPECIFIED]
        assert hr is True


# --- scope classification and short-circuit ---------------------------------


def scope_only_sonnet(scope: str, justification: str = "because", **extra):
    def _fake(system, user, **kw):
        if "GEOGRAPHIC SCOPE" in system:
            return {"scope": scope, "justification": justification, **extra}
        return {}

    return _fake


class TestScopeClassifier:
    def test_valid_scope(self, monkeypatch):
        monkeypatch.setattr(lp, "sonnet", scope_only_sonnet("corridor"))
        got = lp.classify_scope(make_doc(["page one"]), [])
        assert got.scope == "corridor"
        assert got.source == "sonnet"

    def test_off_vocabulary_scope_degrades_to_site(self, monkeypatch):
        monkeypatch.setattr(lp, "sonnet", scope_only_sonnet("statewide"))
        got = lp.classify_scope(make_doc(["page one"]), [])
        assert got.scope == "site"
        assert got.raw_scope == "statewide"
        assert got.source == "default_on_error"

    def test_llm_failure_degrades_to_site(self, monkeypatch):
        def explode(system, user, **kw):
            raise RuntimeError("no credentials")

        monkeypatch.setattr(lp, "sonnet", explode)
        got = lp.classify_scope(make_doc(["page one"]), [])
        assert (got.scope, got.source) == ("site", "default_on_error")

    def test_junk_response_degrades_to_site(self, monkeypatch):
        monkeypatch.setattr(lp, "sonnet", lambda system, user, **kw: ["not", "a", "dict"])
        got = lp.classify_scope(make_doc(["page one"]), [])
        assert got.source == "default_on_error"

    def test_stated_region_is_carried(self, monkeypatch):
        monkeypatch.setattr(
            lp, "sonnet", scope_only_sonnet("regional", stated_region="Puget Sound region")
        )
        got = lp.classify_scope(make_doc(["page one"]), [])
        assert got.stated_region == "Puget Sound region"


class TestPlacelessScopes:
    """MCAL_PLAN 3.9 step 2: national/international never touch a geocoder."""

    @pytest.mark.parametrize("scope", ["national", "international"])
    def test_early_return(self, monkeypatch, scope):
        monkeypatch.setattr(lp, "sonnet", scope_only_sonnet(scope))
        kill_all_hops(monkeypatch)
        res = lp.run_location_pipeline(make_doc(["nationwide rulemaking"]), [], stack="reduced")
        assert res["scope"] == scope
        assert res["sites"] == []
        assert res["geocoded"] == []
        assert res["textual_location"] == scope
        assert res["hop_stats"]["attempts"] == {}

    def test_no_site_extraction_call_is_made(self, monkeypatch):
        calls: list[str] = []

        def fake(system, user, **kw):
            calls.append("scope" if "GEOGRAPHIC SCOPE" in system else "sites")
            return {"scope": "national", "justification": "CAFE rulemaking"}

        monkeypatch.setattr(lp, "sonnet", fake)
        kill_all_hops(monkeypatch)
        lp.run_location_pipeline(make_doc(["text"]), [], stack="reduced")
        assert calls == ["scope"]


# --- end-to-end, reduced mode ----------------------------------------------

DOC_PAGES = [
    "The Buffalo Light Rail Rapid Transit project is located in Buffalo, Erie County, New York.",
    "A maintenance facility is proposed in Amherst, Erie County, New York.",
    "The Cheektowaga site was considered as an alternative location.",
]


def sonnet_for_sites(sites_payload, scope="site", **scope_extra):
    def _fake(system, user, **kw):
        if "GEOGRAPHIC SCOPE" in system:
            return {"scope": scope, "justification": "test", **scope_extra}
        if "PLACES" in system:
            return sites_payload
        return {}

    return _fake


class TestReducedModeEndToEnd:
    """
    The only stack configured on this machine (MCAL_PLAN 3.9a). PAD-US, GNIS and
    Mapbox must not even be attempted.
    """

    SITES = {
        "sites": [
            {
                "name": "Buffalo",
                "admin_hierarchy": {
                    "city": "Buffalo",
                    "county": "Erie County",
                    "state": "New York",
                },
                "role": "primary",
                "quote": "located in Buffalo, Erie County, New York",
            },
            {
                "name": "Amherst",
                "admin_hierarchy": {
                    "city": "Amherst",
                    "county": "Erie County",
                    "state": "New York",
                },
                "role": "primary",
                "quote": "A maintenance facility is proposed in Amherst",
            },
            {
                "name": "Cheektowaga",
                "admin_hierarchy": {"city": "Cheektowaga", "state": "New York"},
                "role": "alternative",
                "quote": "The Cheektowaga site was considered as an alternative location.",
            },
        ]
    }

    def _run(self, monkeypatch, census_table):
        monkeypatch.setattr(lp, "sonnet", sonnet_for_sites(self.SITES))
        for absent in ("gnis", "padus", "mapbox"):
            monkeypatch.setitem(lp.HOPS, absent, boom)
        monkeypatch.setitem(
            lp.HOPS,
            "census",
            lambda q, *, level=None, state=None: census_table.get(q),
        )
        monkeypatch.setitem(lp.HOPS, "nominatim", lambda q, *, level=None, state=None: None)
        return lp.run_location_pipeline(make_doc(DOC_PAGES), [], stack="reduced")

    def test_multi_site_all_resolved(self, monkeypatch):
        table = {
            "Buffalo, Erie County, New York": result(
                42.88, -78.87, (-79.0, 42.8, -78.7, 43.0), source="census", level="city"
            ),
            "Amherst, Erie County, New York": result(
                42.97, -78.79, (-78.9, 42.9, -78.7, 43.05), source="census", level="city"
            ),
            "Erie County, New York": result(
                42.75, -78.9, (-79.1, 42.4, -78.4, 43.1), source="census", level="county"
            ),
            "New York": result(
                43.0, -75.0, (-80.0, 40.4, -71.8, 45.1), source="census", level="state"
            ),
        }
        res = self._run(monkeypatch, table)

        assert res["reduced_mode"] is True
        assert res["geocoder_stack"] == "reduced"
        assert {g["site_name"] for g in res["geocoded"]} == {"Buffalo", "Amherst"}
        assert all(g["source"] == "census" for g in res["geocoded"])
        assert res["tags"] == []
        assert res["textual_location"] == "Buffalo; Amherst"
        assert any("Reduced geocoder stack" in n for n in res["notes"])
        # Non-primary sites are retained in the record but never geocoded.
        alt = [s for s in res["sites"] if s["role"] == "alternative"][0]
        assert alt["geocode"] is None
        assert res["hop_stats"]["hits"]["census"] >= 2

    def test_partial_multi_site_coverage_is_tagged(self, monkeypatch):
        table = {
            "Buffalo, Erie County, New York": result(
                42.88, -78.87, source="census", level="city"
            ),
        }
        res = self._run(monkeypatch, table)
        assert lp.T_MULTI_SITE_PARTIAL in res["tags"]
        assert [g["site_name"] for g in res["geocoded"]] == ["Buffalo"]
        # MCAL_PLAN 3.9 step 5: the unresolved place survives in text.
        assert "Amherst" in res["textual_location"]

    def test_total_geocode_failure_keeps_textual_location(self, monkeypatch):
        res = self._run(monkeypatch, {})
        assert res["geocoded"] == []
        assert lp.T_GEOCODE_MISSING in res["tags"]
        assert res["textual_location"] == "Buffalo; Amherst"
        assert any("textual location retained" in n for n in res["notes"])

    def test_site_extractor_failure_degrades(self, monkeypatch):
        def fake(system, user, **kw):
            if "GEOGRAPHIC SCOPE" in system:
                return {"scope": "site", "justification": "t"}
            raise RuntimeError("model down")

        monkeypatch.setattr(lp, "sonnet", fake)
        kill_all_hops(monkeypatch)
        res = lp.run_location_pipeline(make_doc(DOC_PAGES), [], stack="reduced")
        assert res["sites"] == []
        assert lp.T_GEOCODE_MISSING in res["tags"]
        assert res["site_extraction_meta"]["extractor_ok"] is False

    def test_evidence_is_verified_against_the_document(self, monkeypatch):
        res = self._run(monkeypatch, {})
        buffalo = [s for s in res["sites"] if s["name"] == "Buffalo"][0]
        ev = buffalo["evidence"][0]
        assert ev["quote_verified"] is True
        assert ev["source_pages"] == ["1"]

    def test_unverifiable_quote_is_kept_and_flagged(self, monkeypatch):
        payload = {
            "sites": [
                {
                    "name": "Buffalo",
                    "admin_hierarchy": {"city": "Buffalo"},
                    "role": "primary",
                    "quote": "a sentence that is not in this document at all",
                }
            ]
        }
        monkeypatch.setattr(lp, "sonnet", sonnet_for_sites(payload))
        kill_all_hops(monkeypatch)
        res = lp.run_location_pipeline(make_doc(DOC_PAGES), [], stack="reduced")
        ev = res["sites"][0]["evidence"][0]
        assert ev["quote_verified"] is False
        assert res["sites"][0]["name"] == "Buffalo"


class TestSitePairing:
    """
    Regression for the `_geocode_places` bug named in the build item: it did
    `if not name: continue` while appending, so geocoded[i] stopped describing
    places[i]. Here rows are name-keyed, and an unnamed entry cannot shift them.
    """

    PAYLOAD = {
        "sites": [
            {"name": "", "admin_hierarchy": {}, "role": "primary", "quote": ""},
            {
                "name": "Amherst",
                "admin_hierarchy": {"city": "Amherst", "state": "New York"},
                "role": "primary",
                "quote": "A maintenance facility is proposed in Amherst",
            },
        ]
    }

    def test_unnamed_site_cannot_shift_the_geocode_list(self, monkeypatch):
        monkeypatch.setattr(lp, "sonnet", sonnet_for_sites(self.PAYLOAD))
        monkeypatch.setitem(
            lp.HOPS,
            "census",
            lambda q, *, level=None, state=None: (
                result(42.97, -78.79, source="census", level="city")
                if q.startswith("Amherst")
                else None
            ),
        )
        monkeypatch.setitem(lp.HOPS, "nominatim", lambda q, *, level=None, state=None: None)
        res = lp.run_location_pipeline(make_doc(DOC_PAGES), [], stack="reduced")

        assert [g["site_name"] for g in res["geocoded"]] == ["Amherst"]
        # And the site object carries its own geocode, so no zip is ever needed.
        amherst = [s for s in res["sites"] if s["name"] == "Amherst"][0]
        assert amherst["geocode"]["lat"] == pytest.approx(42.97)

    def test_named_join_survives_reordering(self, monkeypatch):
        """geocoded rows are joinable by name in any order."""
        monkeypatch.setattr(lp, "sonnet", sonnet_for_sites(self.PAYLOAD))
        monkeypatch.setitem(
            lp.HOPS,
            "census",
            lambda q, *, level=None, state=None: result(1.0, 2.0, source="census"),
        )
        monkeypatch.setitem(lp.HOPS, "nominatim", lambda q, *, level=None, state=None: None)
        res = lp.run_location_pipeline(make_doc(DOC_PAGES), [], stack="reduced")
        by_name = {g["site_name"]: g for g in res["geocoded"]}
        for site in res["sites"]:
            if site["geocode"] is not None:
                assert by_name[site["name"]]["lat"] == site["geocode"]["lat"]


class TestCorridorEndToEnd:
    def test_corridor_scope_emits_three_points(self, monkeypatch):
        pages = [
            "The proposed freeway extends from Akron, Ohio to Cleveland, Ohio.",
            "Right-of-way requirements are described here.",
        ]
        payload = {
            "sites": [
                {
                    "name": "Akron",
                    "admin_hierarchy": {"city": "Akron", "state": "Ohio"},
                    "role": "primary",
                    "quote": "from Akron, Ohio to Cleveland, Ohio",
                }
            ],
            "corridor_endpoints": {"from": "Akron, Ohio", "to": "Cleveland, Ohio", "via": None},
        }
        monkeypatch.setattr(lp, "sonnet", sonnet_for_sites(payload, scope="corridor"))
        monkeypatch.setitem(
            lp.HOPS,
            "census",
            lambda q, *, level=None, state=None: result(
                41.0 + len(q) / 100.0, -81.5, source="census", level=level
            ),
        )
        monkeypatch.setitem(lp.HOPS, "nominatim", lambda q, *, level=None, state=None: None)
        res = lp.run_location_pipeline(make_doc(pages), [], stack="reduced")

        assert res["corridor"] is True
        assert res["corridor_endpoints"]["source"] == "regex"
        assert [g["point_role"] for g in res["geocoded"]] == [
            "endpoint_a",
            "midpoint",
            "endpoint_b",
        ]
        assert {g["site_name"] for g in res["geocoded"]} >= {"Akron, Ohio", "Cleveland, Ohio"}

    def test_corridor_without_sites_keeps_a_textual_location(self, monkeypatch):
        """Step 5 again: endpoints parsed but no sites extracted is still a place."""
        pages = ["The proposed freeway extends from Akron, Ohio to Cleveland, Ohio."]
        monkeypatch.setattr(
            lp, "sonnet", sonnet_for_sites({"sites": []}, scope="corridor")
        )
        kill_all_hops(monkeypatch)
        res = lp.run_location_pipeline(make_doc(pages), [], stack="reduced")
        assert res["geocoded"] == []
        assert res["textual_location"] == "Akron, Ohio to Cleveland, Ohio"

    def test_unparseable_corridor_falls_back_to_site_resolution(self, monkeypatch):
        pages = ["A transit improvement in the Buffalo area is proposed."]
        payload = {
            "sites": [
                {
                    "name": "Buffalo",
                    "admin_hierarchy": {"city": "Buffalo"},
                    "role": "primary",
                    "quote": "A transit improvement in the Buffalo area is proposed.",
                }
            ]
        }

        def fake(system, user, **kw):
            if "GEOGRAPHIC SCOPE" in system:
                return {"scope": "corridor", "justification": "t"}
            if "PLACES" in system:
                return payload
            if "ENDPOINTS" in system.upper():
                return {"endpoint_a": None, "endpoint_b": None}
            return {}

        monkeypatch.setattr(lp, "sonnet", fake)
        monkeypatch.setitem(
            lp.HOPS,
            "census",
            lambda q, *, level=None, state=None: result(42.88, -78.87, source="census"),
        )
        monkeypatch.setitem(lp.HOPS, "nominatim", lambda q, *, level=None, state=None: None)
        res = lp.run_location_pipeline(make_doc(pages), [], stack="reduced")
        assert res["corridor"] is False
        assert [g["site_name"] for g in res["geocoded"]] == ["Buffalo"]
        assert any("no endpoints could be parsed" in n for n in res["notes"])


# --- output adapter ---------------------------------------------------------


class TestM2Adapter:
    def test_shape_matches_the_m2_location_field(self, monkeypatch):
        monkeypatch.setattr(lp, "sonnet", scope_only_sonnet("national"))
        kill_all_hops(monkeypatch)
        res = lp.run_location_pipeline(make_doc(["x"]), [], stack="reduced")
        field = lp.as_m2_location_field(res)
        assert set(field) >= {"value", "confidence", "note"}
        assert field["value"]["places"] == []
        assert field["value"]["scope"] == "national"
        assert field["value"]["textual_location"] == "national"
        assert field["confidence"] in ("high", "medium", "low")

    def test_multi_site_flag(self, monkeypatch):
        monkeypatch.setattr(lp, "sonnet", sonnet_for_sites(TestReducedModeEndToEnd.SITES))
        kill_all_hops(monkeypatch)
        res = lp.run_location_pipeline(make_doc(DOC_PAGES), [], stack="reduced")
        field = lp.as_m2_location_field(res)
        assert field["value"]["is_multi_site"] is True
        assert len(field["value"]["places"]) == 3

    def test_reduced_mode_carries_the_gate_decision(self, monkeypatch):
        """MCAL_PLAN 3.9a: reduced mode forces the location bucket to HUMAN_REVIEW."""
        monkeypatch.setattr(lp, "sonnet", scope_only_sonnet("site"))
        kill_all_hops(monkeypatch)
        res = lp.run_location_pipeline(make_doc(["x"]), [], stack="reduced")
        assert res["geocoder_assets"]["gate_all_to_human"] is True
        full = lp.run_location_pipeline(make_doc(["x"]), [], stack="full")
        assert full["geocoder_assets"]["gate_all_to_human"] is False

    def test_no_sites_and_no_coordinates_is_stated_explicitly(self, monkeypatch):
        monkeypatch.setattr(lp, "sonnet", sonnet_for_sites({"sites": []}))
        kill_all_hops(monkeypatch)
        res = lp.run_location_pipeline(make_doc(["x"]), [], stack="reduced")
        assert res["textual_location"] == ""
        assert lp.T_GEOCODE_MISSING in res["tags"]
        assert any("Neither coordinates nor a place name" in n for n in res["notes"])

    def test_full_stack_reaches_the_local_assets(self, monkeypatch, tmp_path):
        """
        Full mode with a real (tiny) GNIS file: the federal-lands / named-feature
        hops are what reduced mode gives up, so this pins that they are wired in.
        """
        path = tmp_path / "gnis.txt"
        path.write_text(
            "feature_name|feature_class|state_alpha|prim_lat_dec|prim_long_dec\n"
            "Ashley National Forest|Forest|UT|40.6|-109.9\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(lp.settings, "gnis_path", lambda: path)
        lp._GNIS.reset()
        monkeypatch.setitem(lp.HOPS, "census", lambda q, *, level=None, state=None: None)
        monkeypatch.setitem(lp.HOPS, "padus", lambda q, *, level=None: None)
        monkeypatch.setitem(lp.HOPS, "mapbox", lambda q, *, level=None: None)
        monkeypatch.setitem(lp.HOPS, "nominatim", boom)   # must never be reached

        got = lp.geocode_cascade("Ashley National Forest", stack="full")
        assert got["source"] == "gnis"


# --- hop-stats accumulator --------------------------------------------------


class TestGlobalHopStats:
    def test_accumulates_across_runs(self, monkeypatch):
        lp.reset_global_hop_stats()
        monkeypatch.setattr(lp, "sonnet", sonnet_for_sites(TestReducedModeEndToEnd.SITES))
        monkeypatch.setitem(lp.HOPS, "census", lambda q, *, level=None, state=None: None)
        monkeypatch.setitem(lp.HOPS, "nominatim", lambda q, *, level=None, state=None: None)
        lp.run_location_pipeline(make_doc(DOC_PAGES), [], stack="reduced")
        stats = lp.global_hop_stats()
        assert stats["attempts"]["census"] > 0
        assert stats["hit_rates"]["census"] == 0.0
        lp.reset_global_hop_stats()
        assert lp.global_hop_stats()["attempts"] == {}


# --- regressions on real graded documents -----------------------------------


class TestFuelEconomyRegression:
    """
    MCAL_PLAN 1(9d): the Fuel Economy CAFE rulemaking was graded wrong because
    the pipeline had no way to say "national". The classifier stand-in below
    decides from the REAL document text, so this test fails if the document no
    longer carries the cues a scope classifier must key on.
    """

    def test_classifies_as_national_and_skips_geocoding(self, doc_loader, monkeypatch):
        doc = doc_loader(FUEL_ECONOMY)
        seen: dict[str, str] = {}

        def stand_in_classifier(system, user, **kw):
            assert "GEOGRAPHIC SCOPE" in system
            seen["user"] = user
            low = user.lower()
            is_national = any(
                cue in low
                for cue in ("fuel economy standard", "nonpassenger automobile")
            )
            return {
                "scope": "national" if is_national else "site",
                "justification": "nationwide fuel-economy rulemaking",
            }

        monkeypatch.setattr(lp, "sonnet", stand_in_classifier)
        kill_all_hops(monkeypatch)

        res = lp.run_location_pipeline(doc, [], stack="reduced")

        assert "fuel economy" in seen["user"].lower()
        assert res["scope"] == "national"
        assert res["geocoded"] == []
        assert res["sites"] == []
        assert res["textual_location"] == "national"
        # No hop was even attempted -- step 2 is a short-circuit, not a filter.
        assert res["hop_stats"]["attempts"] == {}

    def test_national_scope_is_not_an_absent_location(self, doc_loader, monkeypatch):
        """The old output was `places: []` with no scope, which grades as wrong."""
        monkeypatch.setattr(lp, "sonnet", scope_only_sonnet("national"))
        kill_all_hops(monkeypatch)
        res = lp.run_location_pipeline(doc_loader(FUEL_ECONOMY), [], stack="reduced")
        field = lp.as_m2_location_field(res)
        assert field["value"]["scope"] == "national"
        assert field["value"]["textual_location"] == "national"
        assert field["confidence"] != "low"


class TestCorridorParserOnRealDocs:
    """The corridor parser must find endpoints in real OCR, not just fixtures."""

    @pytest.mark.parametrize("doc_id", [LINCOLN_HWY, BUFFALO])
    def test_parser_runs_without_error_on_real_ocr(self, doc_loader, doc_id):
        doc = doc_loader(doc_id)
        got = lp.parse_corridor_endpoints(doc.full_text)
        # Either it finds a plausible pair or it declines; both are valid, but it
        # must never return a pair that fails its own plausibility filter.
        if got is not None:
            assert lp.looks_like_place(got.endpoint_a)
            assert lp.looks_like_place(got.endpoint_b)
            assert got.endpoint_a != got.endpoint_b
