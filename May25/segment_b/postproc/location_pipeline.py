"""
Scope-conditional location extraction + geocoder cascade
(MCAL_PLAN 3.9 / 3.9a / 4 Q4, build item #8).

Replaces segment_a/m2.py:extract_location + _geocode_places, which asked Sonnet
for a flat list of place names and handed each name to Nominatim exactly once.
That design failed on 6 of the 8 graded docs -- MCAL_PLAN 1(9) says 5/8, but the
Evaluation sheet carries a location defect on six (Randolph, LA Transit, Airport
Spur, Buffalo, Lincoln Hwy, Fuel Economy). The miscount is recorded rather than
silently corrected, because MCAL_PLAN 6 states its gating targets against the
plan's own baseline point estimates.

Four DISTINCT failure modes hid behind one "location is bad" grade, and each
needs its own mechanism -- which is why this is a pipeline redesign and not a
prompt tweak:

  1(9a) no geocode at all (Randolph, LA Transit). Nominatim is a global
        gazetteer; US federal-lands units and small place names are often simply
        absent from it, and the old code had no second opinion.
        -> five-hop US-first cascade (Census, GNIS, PAD-US, Mapbox, Nominatim),
           and the textual place name is RETAINED when every hop misses
           (MCAL_PLAN 3.9 step 5) instead of being replaced by nothing.
  1(9b) wrong specificity ("Milwaukee" for an airport-spur corridor). The doc
        says "Milwaukee metropolitan area" far more often than it names the
        corridor, so the LLM offered the coarsest containing city.
        -> bbox-containment cascade over poi/neighborhood/city/county/state; a
           state-only resolution is tagged T07 instead of reported as a win.
  1(9c) multi-site, 1 of 3 geocoded (Buffalo, Lincoln Hwy). The old code called
        the geocoder once with a concatenated string and kept the first parse.
        -> every primary site resolves independently, and partial coverage is
           tagged T09 rather than looking like success.
  1(9d) national rulemaking (Fuel Economy CAFE standards) reported as
        absent-location. There was no vocabulary for "correct answer = national".
        -> the scope classifier runs FIRST and short-circuits before geocoding.

This module also fixes a live alignment bug in the code it replaces.
`_geocode_places` did `if not name: continue` while appending to its output list,
so one unnamed place shifted every later entry and `geocoded[i]` stopped
describing `places[i]`. Nothing downstream could notice, because both lists
stayed plausible. Here every geocode is bound to its site by name and carried on
the site object itself; no consumer is ever asked to zip two lists by position
(MCAL_PLAN 3.9a "Implementation notes" + the explicit build-item requirement).

Reduced mode (MCAL_PLAN 3.9a) is the path that actually runs on most machines:
PAD-US (~1GB), GNIS (~2GB) and MAPBOX_TOKEN are user-supplied assets, so when
`settings.geocoder_precheck()` reports "reduced" the cascade collapses to
Census + Nominatim and the output is flagged. gate.py additionally forces the
whole location bucket to HUMAN_REVIEW in that mode, so reduced-mode coordinates
are never shipped unreviewed -- but they are still computed, because a partially
resolved location is what tells the calibration loop the rest of the pipeline
works.

Everything degrades instead of raising: a missing chapter, an absent
geodatabase, an HTTP timeout, or Sonnet returning junk each produce a tagged,
lower-confidence result. This is post-processing over a 2000-doc batch; an
exception costs more than a tagged gap.
"""

from __future__ import annotations

import csv
import logging
import re
import threading
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Optional, Sequence
from urllib.parse import quote as urlquote

from rapidfuzz import fuzz

from mcal import settings
from mcal.quote_check import normalize

# segment_a's flat modules; the sys.path bridge is installed by mcal.settings.
from chunk import detect_chapters, first_pages, text_for_ceq_chapter  # noqa: E402
from evidence import Evidence, verify_and_locate  # noqa: E402
from llm import sonnet  # noqa: E402
from pages import Doc  # noqa: E402

log = logging.getLogger(__name__)


# --- Vocabulary -------------------------------------------------------------

SCOPES = ("site", "corridor", "regional", "national", "international")
PLACELESS_SCOPES = ("national", "international")

# MCAL_PLAN 3.9 step 3 specifies admin_hierarchy as a POSITIONAL list
# [poi, neighborhood, city, county, state, country]. We accept that shape from
# the model but convert to a keyed mapping immediately: a positional list with
# null holes is precisely the class of bug that produced the `_geocode_places`
# misalignment this module exists to fix, and "index 2 is the city" is not
# something downstream code should have to know.
ADMIN_LEVELS = ("poi", "neighborhood", "city", "county", "state", "country")

# Finest -> coarsest: the order the specificity cascade walks.
SPECIFICITY_ORDER = ADMIN_LEVELS

# Failure tags. These codes are already emitted by mcal/grades.py's seed tagger,
# so a tag produced here lines up with the human grade it corresponds to.
T_GEOCODE_MISSING = "T06_geocode_missing"
T_WRONG_SPECIFICITY = "T07_geocode_wrong_specificity"
T_MULTI_SITE_PARTIAL = "T09_multi_site_partial_geocode"
T_REGIONAL_UNDERSPECIFIED = "T14_regional_scope_underspecified"

# A hop result at or above this confidence stops the cascade. Below it the
# result is kept as a fallback candidate and the next hop still fires -- a weak
# hit from a global gazetteer must not pre-empt a strong hit from a US-specific
# source further down the list.
CONFIDENT_MIN = 0.60

# Half-widths (degrees) used to synthesize a bbox for vendors that return a
# point only (Census, GNIS, PAD-US centroids, some Nominatim rows). The
# containment test is a specificity SANITY check, not real geometry: what
# matters is that a city-level box is smaller than a county-level box which is
# smaller than a state-level box, so "Milwaukee" cannot masquerade as a
# corridor. Nominal radii preserve that ordering when a vendor gives no extent.
_LEVEL_RADIUS_DEG = {
    "poi": 0.01,
    "neighborhood": 0.03,
    "city": 0.12,
    "county": 0.40,
    "state": 2.50,
    "country": 15.0,
    "region": 2.0,
}
_DEFAULT_RADIUS_DEG = 0.10

# Vendor bboxes for the same place disagree at the edges, and synthetic boxes
# are only nominal, so exact shapely containment is too strict. A finer level is
# accepted when this fraction of its area falls inside the coarser box.
CONTAINMENT_MIN_FRACTION = 0.80

# Chapter labels that plausibly describe project geography. Not CEQ chapters, so
# detect_chapters only surfaces them when the OCR carries such a heading --
# best-effort, exactly as in the code being replaced.
PROJECT_AREA_LABELS = ("project area", "study area", "project location", "affected area")

MAX_SECTION_CHARS = 60_000
MAX_PROMPT_CHARS = 120_000
TOC_SCAN_PAGES = 40
TOC_CHARS = 8_000


# --- Result types -----------------------------------------------------------


@dataclass
class ScopeDecision:
    """Output of the scope classifier (MCAL_PLAN 3.9 step 1)."""

    scope: str
    justification: str = ""
    source: str = "sonnet"           # sonnet | default_on_error
    raw_scope: Optional[str] = None  # what the model said, if off-vocabulary
    # Captured here so the regional resolver does not need a second LLM call to
    # recover the document's own words for its region ("Puget Sound region").
    stated_region: Optional[str] = None

    @property
    def is_placeless(self) -> bool:
        """national/international short-circuit the pipeline (step 2)."""
        return self.scope in PLACELESS_SCOPES

    def to_dict(self) -> dict:
        out = {
            "scope": self.scope,
            "justification": self.justification,
            "classifier_source": self.source,
        }
        if self.raw_scope is not None:
            out["raw_scope"] = self.raw_scope
        if self.stated_region:
            out["stated_region"] = self.stated_region
        return out


@dataclass
class Site:
    """One extracted place (MCAL_PLAN 3.9 step 3)."""

    name: str
    admin_hierarchy: dict[str, Optional[str]] = dc_field(default_factory=dict)
    role: str = "primary"            # primary | alternative | reference
    evidence: list[Evidence] = dc_field(default_factory=list)
    # Filled in by the resolver. Bound to the site object; never zipped by index.
    geocode: Optional[dict] = None
    geocode_note: str = ""
    levels_tried: list[str] = dc_field(default_factory=list)

    @property
    def is_primary(self) -> bool:
        return self.role == "primary"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "admin_hierarchy": dict(self.admin_hierarchy),
            "role": self.role,
            "evidence": list(self.evidence),
            "geocode": self.geocode,
            "geocode_note": self.geocode_note,
            "levels_tried": list(self.levels_tried),
        }


@dataclass
class HopStats:
    """
    Per-hop attempt/hit/error counters (MCAL_PLAN 3.9a "Implementation notes").

    MCAL_PLAN 3.9a says its coverage figures are a-priori guesses and asks for
    measured per-hop hit rates in calibration_report.v(N).md. These counters are
    what that report reads; the `source` key on each geocode carries the same
    information at item granularity.
    """

    attempts: dict[str, int] = dc_field(default_factory=dict)
    hits: dict[str, int] = dc_field(default_factory=dict)
    errors: dict[str, int] = dc_field(default_factory=dict)
    skipped: dict[str, int] = dc_field(default_factory=dict)

    def attempt(self, hop: str) -> None:
        self.attempts[hop] = self.attempts.get(hop, 0) + 1

    def hit(self, hop: str) -> None:
        self.hits[hop] = self.hits.get(hop, 0) + 1

    def error(self, hop: str) -> None:
        self.errors[hop] = self.errors.get(hop, 0) + 1

    def skip(self, hop: str) -> None:
        self.skipped[hop] = self.skipped.get(hop, 0) + 1

    def merge(self, other: "HopStats") -> None:
        for src, dst in (
            (other.attempts, self.attempts),
            (other.hits, self.hits),
            (other.errors, self.errors),
            (other.skipped, self.skipped),
        ):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v

    def hit_rates(self) -> dict[str, float]:
        return {
            hop: (self.hits.get(hop, 0) / n if n else 0.0)
            for hop, n in self.attempts.items()
        }

    def to_dict(self) -> dict:
        return {
            "attempts": dict(self.attempts),
            "hits": dict(self.hits),
            "errors": dict(self.errors),
            "skipped": dict(self.skipped),
            "hit_rates": {k: round(v, 3) for k, v in self.hit_rates().items()},
        }


# Process-wide accumulator, so a whole batch can be summarized without
# threading a stats object through every call site.
_GLOBAL_STATS = HopStats()
_GLOBAL_STATS_LOCK = threading.Lock()


def global_hop_stats() -> dict:
    """Cumulative per-hop counters for calibration_report (MCAL_PLAN 3.9a)."""
    with _GLOBAL_STATS_LOCK:
        return _GLOBAL_STATS.to_dict()


def reset_global_hop_stats() -> None:
    global _GLOBAL_STATS
    with _GLOBAL_STATS_LOCK:
        _GLOBAL_STATS = HopStats()


def _record_global(stats: HopStats) -> None:
    with _GLOBAL_STATS_LOCK:
        _GLOBAL_STATS.merge(stats)


# --- Rate limiting / retry --------------------------------------------------
# MCAL_PLAN 3.9a: Census needs no limiter, GNIS/PAD-US are local, Mapbox allows
# 600 req/min per token, Nominatim's public server allows 1 req/sec. Limiters are
# module-level and lock-protected because a batch run geocodes several docs
# concurrently and the published limits are per-token / per-IP, not per-thread.


class _RateLimiter:
    def __init__(self, min_interval_sec: float, name: str) -> None:
        self.min_interval = max(0.0, float(min_interval_sec))
        self.name = name
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            delta = time.monotonic() - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


_MAPBOX_LIMITER = _RateLimiter(60.0 / 600.0, "mapbox")
_NOMINATIM_LIMITER = _RateLimiter(settings.NOMINATIM_MIN_INTERVAL_SEC, "nominatim")

HTTP_TIMEOUT_SEC = 12.0
HTTP_MAX_ATTEMPTS = 3


def _http_get_json(
    url: str,
    params: Optional[dict] = None,
    *,
    limiter: Optional[_RateLimiter] = None,
    timeout: float = HTTP_TIMEOUT_SEC,
    attempts: int = HTTP_MAX_ATTEMPTS,
) -> Optional[dict]:
    """
    GET JSON with retry. Returns None on any unrecoverable failure.

    Never raises: a geocoder outage must degrade one field, not abort a
    2000-doc batch. Only 429/5xx are retried -- a 4xx means the query itself is
    wrong and retrying just burns quota.
    """
    try:
        import requests  # local import so this module imports without it
    except ImportError:  # pragma: no cover - requests is pinned
        log.warning("requests not installed; HTTP geocoder hops disabled")
        return None

    last_err: Optional[Exception] = None
    for attempt in range(max(1, attempts)):
        if limiter is not None:
            limiter.wait()
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_err = RuntimeError(f"HTTP {resp.status_code}")
            elif resp.status_code != 200:
                log.debug(f"geocoder {url} returned HTTP {resp.status_code}")
                return None
            else:
                return resp.json()
        except Exception as e:  # network error, JSON decode error, anything
            last_err = e
        time.sleep(min(0.5 * 2 ** attempt, 4.0))
    log.debug(f"geocoder GET failed after {attempts} attempts ({url}): {last_err}")
    return None


# --- bbox geometry ----------------------------------------------------------
# bbox convention everywhere below: (min_lon, min_lat, max_lon, max_lat), i.e.
# shapely's (minx, miny, maxx, maxy). Vendors disagree wildly on ordering
# (Nominatim ships south/north/west/east), so each hop normalizes to this one
# shape before returning.

Bbox = tuple[float, float, float, float]


def _shapely_box(bbox: Optional[Sequence[float]]):
    """shapely box for a bbox, or None if shapely is missing / bbox is invalid."""
    if bbox is None:
        return None
    try:
        from shapely.geometry import box  # type: ignore
    except ImportError:  # pragma: no cover - shapely is pinned
        return None
    try:
        minx, miny, maxx, maxy = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if minx > maxx:
        minx, maxx = maxx, minx
    if miny > maxy:
        miny, maxy = maxy, miny
    return box(minx, miny, maxx, maxy)


def synth_bbox(lat: float, lon: float, level: Optional[str] = None) -> Bbox:
    """Nominal bbox around a point, for vendors that return no extent."""
    r = _LEVEL_RADIUS_DEG.get(level or "", _DEFAULT_RADIUS_DEG)
    return (lon - r, lat - r, lon + r, lat + r)


def bbox_contained(
    inner: Optional[Sequence[float]],
    outer: Optional[Sequence[float]],
    *,
    min_fraction: float = CONTAINMENT_MIN_FRACTION,
) -> bool:
    """
    Is `inner` (near-)contained in `outer`? MCAL_PLAN 3.9 step 4, site scope.

    Exact `covers` first. Failing that, accept when at least `min_fraction` of
    inner's area lies inside outer: real vendor bboxes for the same city differ
    by hundredths of a degree and one side may be synthetic, so demanding strict
    containment would reject correct city-in-county pairs and push everything up
    to state level -- reintroducing the exact T07 failure this check exists to
    catch. A degenerate (zero-area) inner box degrades to point-in-polygon.
    """
    a, b = _shapely_box(inner), _shapely_box(outer)
    if a is None or b is None:
        return False
    if b.covers(a):
        return True
    if a.area <= 0:
        return bool(b.covers(a.centroid))
    try:
        return (a.intersection(b).area / a.area) >= min_fraction
    except Exception:  # pragma: no cover - shapely predicate failure
        return False


def bbox_intersects(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> bool:
    """Do two bboxes overlap at all? MCAL_PLAN 3.9 step 4, corridor scope."""
    ba, bb = _shapely_box(a), _shapely_box(b)
    if ba is None or bb is None:
        return False
    return bool(ba.intersects(bb))


def point_in_bbox(lat: float, lon: float, bbox: Optional[Sequence[float]]) -> bool:
    """Centroid-inside-coarser-bbox test (MCAL_PLAN 3.9 step 4, corridor)."""
    b = _shapely_box(bbox)
    if b is None:
        return False
    try:
        from shapely.geometry import Point  # type: ignore
    except ImportError:  # pragma: no cover - shapely is pinned
        return False
    return bool(b.covers(Point(float(lon), float(lat))))


def bbox_centroid(bbox: Optional[Sequence[float]]) -> Optional[tuple[float, float]]:
    """(lat, lon) centre of a bbox."""
    b = _shapely_box(bbox)
    if b is None:
        return None
    c = b.centroid
    return (float(c.y), float(c.x))


def points_centroid(
    points: Sequence[tuple[float, float]]
) -> Optional[tuple[float, float]]:
    """
    (lat, lon) centroid of the bounding polygon of several points.

    MCAL_PLAN 3.9 step 4's regional fallback asks for the "bounding-polygon
    centroid of enumerated primary sites". With two points the convex hull is a
    segment whose centroid is its midpoint -- the sensible answer -- so we take
    the hull rather than special-casing the count.
    """
    pts = [
        (float(lon), float(lat))
        for lat, lon in points
        if lat is not None and lon is not None
    ]
    if not pts:
        return None
    try:
        from shapely.geometry import MultiPoint  # type: ignore
    except ImportError:  # pragma: no cover - shapely is pinned
        return (
            sum(p[1] for p in pts) / len(pts),
            sum(p[0] for p in pts) / len(pts),
        )
    c = MultiPoint(pts).convex_hull.centroid
    return (float(c.y), float(c.x))


# --- Geocode payload construction ------------------------------------------


def _result(
    *,
    lat: float,
    lon: float,
    bbox: Optional[Sequence[float]],
    source: str,
    confidence: float,
    admin_hierarchy: Any = None,
    level: Optional[str] = None,
    query: str = "",
    matched_name: str = "",
    extra: Optional[dict] = None,
) -> dict:
    """
    Build a hop's return payload.

    MCAL_PLAN 3.9a fixes the key set as {lat, lon, bbox, source, confidence,
    admin_hierarchy}; `level`, `query`, `matched_name` and `bbox_synthetic` are
    additive and carry the provenance both the specificity cascade and a human
    auditor need. `source` is never dropped downstream (3.9a explicitly).
    """
    lat_f, lon_f = float(lat), float(lon)
    box_t = tuple(float(v) for v in bbox) if bbox else synth_bbox(lat_f, lon_f, level)
    out = {
        "lat": round(lat_f, 6),
        "lon": round(lon_f, 6),
        "bbox": [round(float(v), 6) for v in box_t],
        "source": source,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
        "admin_hierarchy": coerce_admin_hierarchy(admin_hierarchy),
        "level": level,
        "query": query,
        "matched_name": matched_name,
        "bbox_synthetic": not bool(bbox),
    }
    if extra:
        out.update(extra)
    return out


def coerce_admin_hierarchy(value: Any) -> dict[str, Optional[str]]:
    """
    Normalize an admin hierarchy into a keyed mapping over ADMIN_LEVELS.

    Accepts MCAL_PLAN's positional list, a dict, or junk. Unknown keys are
    dropped rather than passed through, so consumers can rely on the key set.
    """
    out: dict[str, Optional[str]] = {k: None for k in ADMIN_LEVELS}
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k).strip().lower()
            if key in out:
                out[key] = clean_place_name(v)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(list(value)[: len(ADMIN_LEVELS)]):
            out[ADMIN_LEVELS[i]] = clean_place_name(v)
    return out


_NULLISH = {"", "null", "none", "n/a", "na", "unknown", "unspecified", "-"}


def clean_place_name(value: Any) -> Optional[str]:
    """Trim a model-supplied place string; map the many spellings of null to None."""
    if value is None:
        return None
    s = str(value).strip().strip(",;")
    return None if s.lower() in _NULLISH else s


def names_match(a: Optional[str], b: Optional[str], *, threshold: float = 88.0) -> bool:
    """
    OCR-tolerant place-name comparison.

    Reuses mcal.quote_check.normalize so a name damaged by 1970s microfilm OCR
    ("M0doc Nati0nal F0rest") still matches the gazetteer spelling, which is the
    whole reason hops 2 and 3 can work off local files at all.
    """
    if not a or not b:
        return False
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    return fuzz.ratio(na, nb) >= threshold


# --- Hop 1: US Census Geocoder ---------------------------------------------


def geocode_census(query: str, *, level: Optional[str] = None) -> Optional[dict]:
    """
    Hop 1 (MCAL_PLAN 3.9a): US Census geocoder. Free, unlimited, US-only, no auth.

    Census returns a point and no extent, so the bbox is synthesized from the
    requested level. It also only resolves what it can place on a TIGER line --
    addresses and address-like strings -- so bare feature names ("Ashley
    National Forest") legitimately miss here and fall through to GNIS/PAD-US.
    That is the intended division of labour, not a defect.
    """
    q = (query or "").strip()
    if not q:
        return None
    data = _http_get_json(
        settings.CENSUS_GEOCODER_URL,
        {"address": q, "benchmark": "Public_AR_Current", "format": "json"},
    )
    if not isinstance(data, dict):
        return None
    matches = ((data.get("result") or {}).get("addressMatches") or [])
    if not isinstance(matches, list) or not matches:
        return None
    m = matches[0]
    if not isinstance(m, dict):
        return None
    coords = m.get("coordinates") or {}
    try:
        lon = float(coords["x"])
        lat = float(coords["y"])
    except (KeyError, TypeError, ValueError):
        return None
    comp = m.get("addressComponents") or {}
    admin = {
        "city": clean_place_name(comp.get("city")),
        "state": clean_place_name(comp.get("state")),
        "country": "United States",
    }
    # One unambiguous match is strong evidence. Several matches means Census
    # could not disambiguate -- exactly the T07 risk -- so we score that below
    # CONFIDENT_MIN and let a later hop try to do better.
    confidence = 0.90 if len(matches) == 1 else 0.55
    return _result(
        lat=lat,
        lon=lon,
        bbox=None,
        source="census",
        confidence=confidence,
        admin_hierarchy=admin,
        level=level,
        query=q,
        matched_name=str(m.get("matchedAddress") or "").strip(),
    )


# --- Hop 2: USGS GNIS (local TSV) ------------------------------------------
#
# GNIS ships under two schemas: the legacy NationalFile
# (FEATURE_NAME|STATE_ALPHA|PRIM_LAT_DEC...) and the current
# DomesticNames_National (feature_name|state_name|prim_lat_dec...). Columns are
# resolved by alias so either download works untouched, and the delimiter is
# sniffed because USGS has shipped both pipe- and tab-separated variants.

_GNIS_COLUMN_ALIASES = {
    "name": ("feature_name", "name"),
    "feature_class": ("feature_class", "class"),
    "state": ("state_alpha", "state_name", "state"),
    "county": ("county_name", "county"),
    "lat": ("prim_lat_dec", "primary_latitude_dec", "lat_dec", "latitude"),
    "lon": ("prim_long_dec", "primary_longitude_dec", "long_dec", "longitude"),
}

# GNIS feature classes mapped onto our specificity levels, so a GNIS hit gets a
# sensible synthetic bbox instead of the generic default.
_GNIS_CLASS_LEVEL = {
    "populated place": "city",
    "civil": "county",
    "county": "county",
    "forest": "county",
    "reserve": "county",
    "range": "county",
    "valley": "county",
    "stream": "county",
    "lake": "neighborhood",
    "reservoir": "neighborhood",
    "park": "poi",
    "military": "poi",
    "airport": "poi",
    "building": "poi",
    "school": "poi",
    "hospital": "poi",
    "dam": "poi",
    "summit": "poi",
    "mine": "poi",
}


@dataclass(frozen=True)
class GnisRecord:
    name: str
    feature_class: str
    state: Optional[str]
    county: Optional[str]
    lat: float
    lon: float


class _GnisIndex:
    """
    Lazily-built in-memory index over the GNIS domestic-names file.

    Keyed on `normalize(name)` so OCR damage in the document still finds the
    gazetteer row. We keep only the six fields we use, which is what makes
    holding ~2.3M rows in memory tolerable for a batch run; the file is read
    once per process and a failed load is remembered so we do not re-read a
    truncated 2GB download for every doc.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, list[GnisRecord]] = {}
        self._loaded = False
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def size(self) -> int:
        return len(self._by_name)

    def reset(self) -> None:
        """Drop the cache. Used by tests and after a re-download."""
        with self._lock:
            self._by_name = {}
            self._loaded = False
            self._error = None

    def load(self) -> bool:
        with self._lock:
            if self._loaded:
                return self._error is None
            self._loaded = True
            path = settings.gnis_path()
            if path is None:
                self._error = "GNIS_TSV_PATH not configured"
                return False
            if not path.exists():
                self._error = f"GNIS_TSV_PATH missing on disk: {path}"
                return False
            try:
                self._read(path)
            except Exception as e:  # truncated / malformed download
                self._error = f"GNIS load failed: {e}"
                log.warning(self._error)
                self._by_name = {}
                return False
            log.info(f"GNIS index loaded: {len(self._by_name)} distinct names")
            return True

    def _read(self, path) -> None:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            header_line = fh.readline()
            if not header_line.strip():
                raise ValueError("empty GNIS file")
            delimiter = "|" if header_line.count("|") >= header_line.count("\t") else "\t"
            header = [h.strip().lower() for h in header_line.rstrip("\r\n").split(delimiter)]
            cols = self._resolve_columns(header)
            index: dict[str, list[GnisRecord]] = {}
            for row in csv.reader(fh, delimiter=delimiter):
                if len(row) <= cols["lon"]:
                    continue
                name = (row[cols["name"]] or "").strip()
                if not name:
                    continue
                try:
                    lat = float(row[cols["lat"]])
                    lon = float(row[cols["lon"]])
                except (TypeError, ValueError):
                    continue  # unpopulated coordinates are common in GNIS
                rec = GnisRecord(
                    name=name,
                    feature_class=(
                        (row[cols["feature_class"]] or "").strip()
                        if cols.get("feature_class") is not None
                        and len(row) > cols["feature_class"]
                        else ""
                    ),
                    state=(
                        clean_place_name(row[cols["state"]])
                        if cols.get("state") is not None and len(row) > cols["state"]
                        else None
                    ),
                    county=(
                        clean_place_name(row[cols["county"]])
                        if cols.get("county") is not None and len(row) > cols["county"]
                        else None
                    ),
                    lat=lat,
                    lon=lon,
                )
                index.setdefault(normalize(name), []).append(rec)
            self._by_name = index

    @staticmethod
    def _resolve_columns(header: list[str]) -> dict[str, int]:
        cols: dict[str, int] = {}
        for logical, aliases in _GNIS_COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in header:
                    cols[logical] = header.index(alias)
                    break
        for required in ("name", "lat", "lon"):
            if required not in cols:
                raise ValueError(
                    f"GNIS header lacks a {required} column; got {header[:12]}"
                )
        return cols

    def lookup(self, name: str, *, state: Optional[str] = None) -> list[GnisRecord]:
        if not self.load():
            return []
        rows = self._by_name.get(normalize(name), [])
        if state and rows:
            narrowed = [r for r in rows if names_match(r.state, state)]
            if narrowed:
                return narrowed
        return rows


_GNIS = _GnisIndex()


def gnis_lookup(name: str, *, state: Optional[str] = None) -> list[GnisRecord]:
    """Exposed separately so tests (and a future SQLite backend) can swap it."""
    return _GNIS.lookup(name, state=state)


def gnis_status() -> dict:
    return {"available": _GNIS.error is None and _GNIS.load(), "error": _GNIS.error}


def geocode_gnis(
    query: str, *, level: Optional[str] = None, state: Optional[str] = None
) -> Optional[dict]:
    """
    Hop 2 (MCAL_PLAN 3.9a): named natural/cultural features from the local
    USGS GNIS file. No network.

    A state hint disambiguates the many identically-named features ("Springfield"
    exists in most states); without one, an ambiguous name is deliberately scored
    below CONFIDENT_MIN so the cascade keeps looking rather than committing to an
    arbitrary row -- the multi-site/wrong-place failure mode in miniature.
    """
    q = (query or "").strip()
    if not q:
        return None
    rows = gnis_lookup(q, state=state)
    if not rows:
        return None
    rec = rows[0]
    ambiguous = len(rows) > 1
    feature_level = _GNIS_CLASS_LEVEL.get(rec.feature_class.strip().lower(), level or "poi")
    return _result(
        lat=rec.lat,
        lon=rec.lon,
        bbox=None,
        source="gnis",
        confidence=0.55 if ambiguous else 0.78,
        admin_hierarchy={
            "county": rec.county,
            "state": rec.state,
            "country": "United States",
        },
        level=feature_level,
        query=q,
        matched_name=rec.name,
        extra={
            "feature_class": rec.feature_class,
            "n_candidates": len(rows),
        },
    )


# --- Hop 3: PAD-US spatial join (local geodatabase) ------------------------
#
# MCAL_PLAN 3.9a expects this to be the single biggest quality upgrade for this
# corpus: federal-agency EISs name their own managed units ("Cottonwood Field
# Office", "Ashley National Forest") and no general-purpose geocoder resolves
# those. PAD-US carries every federal parcel as a polygon, so the "spatial join"
# reduces to fuzzy-matching a unit name and taking that polygon's extent.
#
# Geodatabase access is confined to `_PadusIndex._read`, and the matcher works
# off plain `PadusUnit` records, so the fuzzy-match logic is testable without a
# 1GB download or a GDAL driver.

# Managing agencies MCAL_PLAN 3.9a names explicitly, plus the codes PAD-US uses
# for them. Matched case-insensitively against whichever manager column exists.
PADUS_FEDERAL_MANAGERS = (
    "BLM", "NPS", "FWS", "USFWS", "USFS", "FS", "DOD", "DOE", "BOR", "USBR",
    "Bureau of Land Management", "National Park Service",
    "Fish and Wildlife Service", "Forest Service", "Department of Defense",
)

_PADUS_NAME_FIELDS = ("Unit_Nm", "unit_nm", "Loc_Nm", "loc_nm", "d_Unit_Nm", "Name")
_PADUS_MANAGER_FIELDS = (
    "Mang_Name", "mang_name", "d_Mang_Nam", "Mang_Type", "d_Mang_Typ", "Own_Name",
)
# Preference order for the layer to read. PAD-US ships ~10 layers; the combined
# fee/designation/easement layer is the one that contains managed unit names.
_PADUS_LAYER_PREFERENCES = ("combined", "fee", "designation", "proclamation", "easement")


@dataclass(frozen=True)
class PadusUnit:
    name: str
    manager: str
    lat: float
    lon: float
    bbox: Bbox


class _PadusIndex:
    """Lazily-loaded, filtered view of the PAD-US geodatabase."""

    def __init__(self) -> None:
        self._units: list[PadusUnit] = []
        self._loaded = False
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def error(self) -> Optional[str]:
        return self._error

    def reset(self) -> None:
        with self._lock:
            self._units = []
            self._loaded = False
            self._error = None

    def units(self) -> list[PadusUnit]:
        with self._lock:
            if not self._loaded:
                self._loaded = True
                path = settings.padus_path()
                if path is None:
                    self._error = "PADUS_GEODATABASE_PATH not configured"
                elif not path.exists():
                    self._error = f"PADUS_GEODATABASE_PATH missing on disk: {path}"
                else:
                    try:
                        self._units = self._read(path)
                        log.info(f"PAD-US index loaded: {len(self._units)} federal units")
                    except Exception as e:
                        # A missing OpenFileGDB driver, a partial unzip, or an
                        # unexpected schema all land here. Reduced mode, not a
                        # crash (MCAL_PLAN 3.9a).
                        self._error = f"PAD-US load failed: {e}"
                        log.warning(self._error)
                        self._units = []
            return self._units

    @staticmethod
    def _read(path) -> list[PadusUnit]:
        import geopandas as gpd  # type: ignore

        layer = _PadusIndex._pick_layer(path)
        gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        if gdf is None or gdf.empty:
            raise ValueError(f"PAD-US layer {layer!r} is empty")

        name_col = next((c for c in _PADUS_NAME_FIELDS if c in gdf.columns), None)
        if name_col is None:
            raise ValueError(f"no unit-name column in PAD-US layer {layer!r}")
        mgr_col = next((c for c in _PADUS_MANAGER_FIELDS if c in gdf.columns), None)

        # PAD-US is distributed in USGS Albers; the cascade speaks WGS84 degrees.
        if getattr(gdf, "crs", None) is not None:
            try:
                gdf = gdf.to_crs("EPSG:4326")
            except Exception as e:  # pragma: no cover - pyproj misconfiguration
                log.warning(f"PAD-US reprojection failed ({e}); using native CRS")

        out: list[PadusUnit] = []
        for row in gdf.itertuples(index=False):
            name = str(getattr(row, name_col, "") or "").strip()
            if not name:
                continue
            manager = str(getattr(row, mgr_col, "") or "").strip() if mgr_col else ""
            if mgr_col and not _is_federal_manager(manager):
                continue
            geom = getattr(row, "geometry", None)
            if geom is None or geom.is_empty:
                continue
            minx, miny, maxx, maxy = (float(v) for v in geom.bounds)
            pt = geom.representative_point()
            out.append(
                PadusUnit(
                    name=name,
                    manager=manager,
                    lat=float(pt.y),
                    lon=float(pt.x),
                    bbox=(minx, miny, maxx, maxy),
                )
            )
        return out

    @staticmethod
    def _pick_layer(path) -> Optional[str]:
        try:
            from pyogrio import list_layers  # type: ignore
        except ImportError:  # pragma: no cover - pyogrio is pinned
            return None
        names = [str(row[0]) for row in list_layers(path)]
        for pref in _PADUS_LAYER_PREFERENCES:
            for n in names:
                if pref in n.lower():
                    return n
        return names[0] if names else None


_PADUS = _PadusIndex()


def _is_federal_manager(manager: str) -> bool:
    m = (manager or "").strip().lower()
    if not m:
        return False
    return any(code.lower() in m or m == code.lower() for code in PADUS_FEDERAL_MANAGERS)


def padus_units() -> list[PadusUnit]:
    """Exposed separately so tests can substitute synthetic units."""
    return _PADUS.units()


def padus_status() -> dict:
    return {"available": bool(padus_units()), "error": _PADUS.error}


# Federal-lands names are long and OCR-damaged, so the acceptance bar here is
# looser than the generic `names_match` threshold -- but only a normalized-exact
# (or one-character-off) match earns confident status. A fuzzy hit is scored
# below CONFIDENT_MIN so the cascade continues and a better source can win: PAD-US
# contains thousands of similarly-named field offices and ranger districts, and
# picking the wrong one is worse than falling through to Mapbox.
PADUS_EXACT_MIN = 99.0
PADUS_FUZZY_MIN = 86.0


def geocode_padus(query: str, *, level: Optional[str] = None) -> Optional[dict]:
    """
    Hop 3 (MCAL_PLAN 3.9a): fuzzy-name join against federal land units. No network.

    Returns the matched polygon's real extent (not a synthetic box) plus a
    representative interior point -- `representative_point` rather than the
    centroid, because a national forest's centroid can fall in a hole or outside
    a horseshoe-shaped boundary, which would then fail the containment test for
    a reason that has nothing to do with specificity.
    """
    q = (query or "").strip()
    if not q:
        return None
    units = padus_units()
    if not units:
        return None
    nq = normalize(q)
    if not nq:
        return None

    best: Optional[PadusUnit] = None
    best_score = 0.0
    for u in units:
        nu = normalize(u.name)
        if not nu:
            continue
        score = 100.0 if nu == nq else float(fuzz.token_sort_ratio(nq, nu))
        if score > best_score:
            best, best_score = u, score
    if best is None or best_score < PADUS_FUZZY_MIN:
        return None

    return _result(
        lat=best.lat,
        lon=best.lon,
        bbox=best.bbox,
        source="padus",
        confidence=0.85 if best_score >= PADUS_EXACT_MIN else 0.55,
        admin_hierarchy={"poi": best.name, "country": "United States"},
        level=level or "poi",
        query=q,
        matched_name=best.name,
        extra={"manager": best.manager, "name_score": round(best_score, 1)},
    )


# --- Hop 4: Mapbox ----------------------------------------------------------

# Mapbox context ids -> our admin levels.
_MAPBOX_CONTEXT_LEVELS = {
    "poi": "poi",
    "neighborhood": "neighborhood",
    "locality": "neighborhood",
    "place": "city",
    "district": "county",
    "region": "state",
    "country": "country",
}


def geocode_mapbox(query: str, *, level: Optional[str] = None) -> Optional[dict]:
    """
    Hop 4 (MCAL_PLAN 3.9a): POIs and named highways. 600 req/min per token.

    Skipped silently when MAPBOX_TOKEN is unset -- that is the documented
    reduced-mode state, not an error worth logging per query.
    """
    q = (query or "").strip()
    token = settings.mapbox_token()
    if not q or not token:
        return None
    url = f"{settings.MAPBOX_GEOCODER_URL}/{urlquote(q)}.json"
    data = _http_get_json(
        url,
        {"access_token": token, "limit": 1, "country": "us", "language": "en"},
        limiter=_MAPBOX_LIMITER,
    )
    if not isinstance(data, dict):
        return None
    feats = data.get("features") or []
    if not isinstance(feats, list) or not feats:
        return None
    f = feats[0]
    if not isinstance(f, dict):
        return None
    center = f.get("center") or []
    try:
        lon, lat = float(center[0]), float(center[1])
    except (IndexError, TypeError, ValueError):
        return None

    admin: dict[str, Optional[str]] = {}
    place_types = [str(t) for t in (f.get("place_type") or [])]
    own_level = next(
        (_MAPBOX_CONTEXT_LEVELS[t] for t in place_types if t in _MAPBOX_CONTEXT_LEVELS),
        level or "poi",
    )
    admin[own_level] = clean_place_name(f.get("text"))
    for ctx in f.get("context") or []:
        if not isinstance(ctx, dict):
            continue
        kind = str(ctx.get("id") or "").split(".")[0]
        mapped = _MAPBOX_CONTEXT_LEVELS.get(kind)
        if mapped and not admin.get(mapped):
            admin[mapped] = clean_place_name(ctx.get("text"))

    bbox = f.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        norm_bbox: Optional[Sequence[float]] = bbox
    else:
        norm_bbox = None
    try:
        relevance = float(f.get("relevance", 0.5))
    except (TypeError, ValueError):
        relevance = 0.5
    return _result(
        lat=lat,
        lon=lon,
        bbox=norm_bbox,
        source="mapbox",
        confidence=max(0.4, min(0.92, 0.55 + 0.4 * relevance)),
        admin_hierarchy=admin,
        level=own_level,
        query=q,
        matched_name=str(f.get("place_name") or f.get("text") or "").strip(),
        extra={"relevance": round(relevance, 3), "place_type": place_types},
    )


# --- Hop 5: Nominatim (last resort) ----------------------------------------


def geocode_nominatim(query: str, *, level: Optional[str] = None) -> Optional[dict]:
    """
    Hop 5 (MCAL_PLAN 3.9a): public Nominatim via geopy, 1 req/sec.

    Kept last, and kept at all, for international mentions (border projects) --
    it is the hop that produced the original single-shot failures, so it must
    never outrank a US-specific source. Confidence is derived from OSM
    `importance`, which is low for the obscure US places that broke hop-5-only
    geocoding, so a weak Nominatim hit stays visibly weak.
    """
    q = (query or "").strip()
    if not q:
        return None
    try:
        from geopy.geocoders import Nominatim  # type: ignore
    except ImportError:
        log.debug("geopy not installed; Nominatim hop disabled")
        return None
    _NOMINATIM_LIMITER.wait()
    try:
        geo = Nominatim(user_agent=settings.NOMINATIM_USER_AGENT)
        r = geo.geocode(q, timeout=15, addressdetails=True, exactly_one=True)
    except Exception as e:
        log.debug(f"Nominatim failed for {q!r}: {e}")
        return None
    if r is None:
        return None

    raw = getattr(r, "raw", None) or {}
    bbox: Optional[tuple[float, float, float, float]] = None
    bb = raw.get("boundingbox")
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        try:
            south, north, west, east = (float(v) for v in bb)
            bbox = (west, south, east, north)  # -> (minlon, minlat, maxlon, maxlat)
        except (TypeError, ValueError):
            bbox = None
    addr = raw.get("address") if isinstance(raw.get("address"), dict) else {}
    admin = {
        "city": clean_place_name(
            addr.get("city") or addr.get("town") or addr.get("village")
        ),
        "county": clean_place_name(addr.get("county")),
        "state": clean_place_name(addr.get("state")),
        "country": clean_place_name(addr.get("country")),
    }
    try:
        importance = float(raw.get("importance", 0.3))
    except (TypeError, ValueError):
        importance = 0.3
    return _result(
        lat=float(r.latitude),
        lon=float(r.longitude),
        bbox=bbox,
        source="nominatim",
        confidence=max(0.40, min(0.85, 0.45 + 0.5 * importance)),
        admin_hierarchy=admin,
        level=level,
        query=q,
        matched_name=str(getattr(r, "address", "") or "").strip(),
        extra={"importance": round(importance, 3)},
    )


# --- Cascade ----------------------------------------------------------------

HopFn = Callable[..., Optional[dict]]

# Hops are registered by NAME and looked up at call time. The name doubles as the
# HopStats key and as the `source` each hop stamps on its result, so a hit rate
# in calibration_report and a `source` in a manifest can never drift apart.
HOPS: dict[str, HopFn] = {
    "census": geocode_census,
    "gnis": geocode_gnis,
    "padus": geocode_padus,
    "mapbox": geocode_mapbox,
    "nominatim": geocode_nominatim,
}

# MCAL_PLAN 3.9a cascade order.
FULL_CASCADE: tuple[str, ...] = ("census", "gnis", "padus", "mapbox", "nominatim")

# Reduced mode: the two hops that need no user-supplied asset (MCAL_PLAN 3.9a
# "Reduced-pipeline fallback").
REDUCED_CASCADE: tuple[str, ...] = ("census", "nominatim")


def resolve_stack(stack: Optional[str] = None) -> str:
    """"full" or "reduced" -- from the caller, else from the asset precheck."""
    if stack in ("full", "reduced"):
        return stack
    try:
        return settings.geocoder_precheck().get("stack", "reduced")
    except Exception:  # pragma: no cover - precheck never raises by contract
        return "reduced"


def cascade_for_stack(stack: str) -> tuple[str, ...]:
    return FULL_CASCADE if stack == "full" else REDUCED_CASCADE


def geocode_cascade(
    query: str,
    *,
    level: Optional[str] = None,
    state: Optional[str] = None,
    stack: Optional[str] = None,
    stats: Optional[HopStats] = None,
) -> Optional[dict]:
    """
    Run the vendor cascade for one query string (MCAL_PLAN 3.9a).

    Each hop fires only if no earlier hop produced a CONFIDENT result. A
    below-threshold result is retained as a fallback and returned if nothing
    better appears, because a low-confidence coordinate plus its `source` is
    strictly more useful to a reviewer than no coordinate at all -- and step 5
    forbids discarding a located place merely because we are unsure of it.
    """
    q = (query or "").strip()
    if not q:
        return None
    stack = resolve_stack(stack)
    stats = stats if stats is not None else HopStats()
    best: Optional[dict] = None

    for name in cascade_for_stack(stack):
        fn = HOPS.get(name)
        if fn is None:  # pragma: no cover - registry is static
            stats.skip(name)
            continue
        stats.attempt(name)
        try:
            res = fn(q, level=level, state=state) if name == "gnis" else fn(q, level=level)
        except Exception as e:
            # A hop must never take the pipeline down with it.
            stats.error(name)
            log.debug(f"geocoder hop {name} raised on {q!r}: {e}")
            continue
        if not isinstance(res, dict) or res.get("lat") is None:
            continue
        stats.hit(name)
        if best is None or res.get("confidence", 0) > best.get("confidence", 0):
            best = res
        if res.get("confidence", 0.0) >= CONFIDENT_MIN:
            res["confident"] = True
            res["stack"] = stack
            return res

    if best is not None:
        best["confident"] = False
        best["stack"] = stack
    return best


# --- Document text selection ------------------------------------------------


def _toc_text(doc: Doc, chapters: Optional[list[dict]]) -> str:
    """
    Best-effort table of contents for the scope classifier (MCAL_PLAN 3.9 step 1).

    Prefers a real TOC found in the front matter; falls back to the labels
    `detect_chapters` recovered, which for a badly-OCR'd scan is often all the
    structure that exists. Returning "" is acceptable -- the classifier also
    receives the first 30 pages.
    """
    head = first_pages(doc, min(TOC_SCAN_PAGES, doc.n_pages or TOC_SCAN_PAGES))
    m = re.search(r"\b(table\s+of\s+contents|contents)\b", head, re.IGNORECASE)
    if m:
        return head[m.start() : m.start() + TOC_CHARS]
    if chapters:
        return "\n".join(
            f"{ch.get('label', '')} (pp. {ch.get('start_page')}-{ch.get('end_page')})"
            for ch in chapters
        )
    return ""


def _project_area_text(doc: Doc, chapters: Optional[list[dict]]) -> tuple[str, str]:
    """
    Text of any Project/Study Area section, plus a short provenance label.

    Same best-effort chapter-label scan as the code being replaced (m2.py:536),
    kept because these are not CEQ chapters and only appear when the OCR carried
    the heading. Falls back to the Affected Environment chapter, which is where
    the study area is described when no dedicated heading exists.
    """
    pieces: list[str] = []
    labels: list[str] = []
    for ch in chapters or []:
        label = (ch.get("label") or "").lower()
        if any(k in label for k in PROJECT_AREA_LABELS):
            seg = doc.full_text[ch.get("start_char", 0) : ch.get("end_char", 0)]
            if seg.strip():
                pieces.append(seg[:MAX_SECTION_CHARS])
                labels.append(ch.get("label") or "")
    if not pieces:
        fallback = text_for_ceq_chapter(doc, chapters or [], "Affected Environment")
        if fallback and fallback[0].strip():
            pieces.append(fallback[0][:MAX_SECTION_CHARS])
            labels.append("Affected Environment")
    return ("\n\n---\n\n".join(pieces), "; ".join(labels))


def _sonnet_json(system: str, user: str, *, max_tokens: int = 1200) -> Optional[dict]:
    """
    One Sonnet call that must return a JSON object, or None.

    Wraps every failure mode the extractors care about: no credentials, a
    transport error, unparseable output, or a list where an object was promised.
    Callers degrade and tag; nothing here propagates.
    """
    try:
        out = sonnet(system=system, user=user, max_tokens=max_tokens)
    except Exception as e:
        log.warning(f"Sonnet call failed: {e}")
        return None
    if not isinstance(out, dict):
        log.warning(f"Sonnet returned {type(out).__name__}, expected object")
        return None
    return out


# --- Step 1: scope classifier -----------------------------------------------

SCOPE_SYSTEM = (
    "You classify the GEOGRAPHIC SCOPE of an Environmental Impact Statement.\n"
    "Respond ONLY with JSON:\n"
    "{\n"
    '  "scope": "site|corridor|regional|national|international",\n'
    '  "justification": "<ONE sentence citing what in the text decided it>",\n'
    '  "stated_region": "<the region name if scope is regional, else null>"\n'
    "}\n"
    "Definitions:\n"
    "- site: one or more discrete places (a dam, a plant, a housing project, a\n"
    "  timber sale, a set of candidate sites).\n"
    "- corridor: a linear facility between two endpoints (highway, transit line,\n"
    "  transmission line, pipeline, railway).\n"
    "- regional: a named multi-county or multi-state region analyzed as a whole\n"
    "  (\"Southern California\", \"the Puget Sound region\"), with no single site.\n"
    "- national: a rulemaking, standard, program or policy applying across the\n"
    "  United States with no project location. Fuel-economy standards, emission\n"
    "  standards, nationwide program EISs are national.\n"
    "- international: the action spans a national border or occurs abroad.\n"
    "Choose national over site when the action is a rule rather than a place. A\n"
    "document that merely mentions many states is NOT regional unless it analyzes\n"
    "one named region."
)


def classify_scope(doc: Doc, chapters: Optional[list[dict]] = None) -> ScopeDecision:
    """
    MCAL_PLAN 3.9 step 1: one Sonnet call over the first 30 pages + the TOC.

    Runs FIRST and unconditionally. That ordering is the fix for 1(9d): the old
    pipeline had no way to express "national" and so reported the Fuel Economy
    CAFE rulemaking as having no location, which is a different (and wrong)
    claim. An off-vocabulary or failed classification degrades to `site`, the
    only choice that keeps the rest of the pipeline running; `classifier_source`
    records that it was a default so gate.py can treat it as weak.
    """
    head = first_pages(doc, min(30, doc.n_pages or 30))
    toc = _toc_text(doc, chapters)
    user = (
        f"FIRST 30 PAGES:\n{head[:MAX_PROMPT_CHARS]}\n\n"
        f"TABLE OF CONTENTS / CHAPTER HEADINGS:\n{toc or '(none detected)'}"
    )
    out = _sonnet_json(SCOPE_SYSTEM, user, max_tokens=500)
    if out is None:
        return ScopeDecision(
            scope="site",
            justification="Scope classifier unavailable; defaulted to site.",
            source="default_on_error",
        )
    raw = str(out.get("scope") or "").strip().lower()
    if raw not in SCOPES:
        return ScopeDecision(
            scope="site",
            justification=str(out.get("justification") or "").strip(),
            source="default_on_error",
            raw_scope=raw or None,
        )
    decision = ScopeDecision(
        scope=raw, justification=str(out.get("justification") or "").strip()
    )
    decision.stated_region = clean_place_name(out.get("stated_region"))
    return decision


# --- Step 3: site extraction ------------------------------------------------

SITES_SYSTEM = (
    "You extract the PLACES an Environmental Impact Statement is about.\n"
    "Respond ONLY with JSON:\n"
    "{\n"
    '  "sites": [\n'
    "    {\n"
    '      "name": "<the place as the document names it>",\n'
    '      "admin_hierarchy": {\n'
    '        "poi": "<named facility/feature or null>",\n'
    '        "neighborhood": "<or null>",\n'
    '        "city": "<or null>",\n'
    '        "county": "<or null>",\n'
    '        "state": "<or null>",\n'
    '        "country": "<or null>"\n'
    "      },\n"
    '      "role": "primary|alternative|reference",\n'
    '      "quote": "<verbatim phrase from the excerpts establishing this place>"\n'
    "    }\n"
    "  ],\n"
    '  "stated_region": "<named region if the document analyzes one, else null>",\n'
    '  "corridor_endpoints": {"from": "<or null>", "to": "<or null>", "via": "<or null>"}\n'
    "}\n"
    "Rules:\n"
    "- role=primary: a place where the proposed action would actually occur.\n"
    "  role=alternative: a candidate site NOT selected as the proposal.\n"
    "  role=reference: mentioned only for comparison, precedent or context.\n"
    "- List EVERY primary place separately. Do not merge two towns into one entry\n"
    "  and do not substitute a containing city for a smaller named place.\n"
    "- Fill admin_hierarchy only from what the document states; null otherwise.\n"
    "  Never guess a county or state.\n"
    "- The quote MUST be copied character-for-character from the excerpts; a\n"
    "  verifier substring-matches it against the document."
)


def extract_sites(
    doc: Doc, chapters: Optional[list[dict]] = None
) -> tuple[list[Site], dict]:
    """
    MCAL_PLAN 3.9 step 3: Sonnet over first 30pp + the Project/Study Area chapter.

    Returns `(sites, meta)`. Every site carries `evidence` from
    `verify_and_locate`, and sites with `role != primary` are kept in the output
    but excluded from geocoding by the resolvers -- the plan rejects them from
    geocoding, not from the record, since an alternative site the agency
    considered is real information about the document.
    """
    head = first_pages(doc, min(30, doc.n_pages or 30))
    area_text, area_label = _project_area_text(doc, chapters)
    user = (
        f"FIRST 30 PAGES:\n{head[:MAX_PROMPT_CHARS]}\n\n"
        f"PROJECT / STUDY AREA SECTION"
        f"{f' ({area_label})' if area_label else ' (not detected)'}:\n"
        f"{area_text[:MAX_SECTION_CHARS] or '(none)'}"
    )
    out = _sonnet_json(SITES_SYSTEM, user, max_tokens=2500)
    meta: dict = {
        "project_area_section": area_label or None,
        "extractor_ok": out is not None,
    }
    if out is None:
        return [], meta

    meta["stated_region"] = clean_place_name(out.get("stated_region"))
    endpoints = out.get("corridor_endpoints")
    if isinstance(endpoints, dict):
        meta["llm_corridor_endpoints"] = {
            "from": clean_place_name(endpoints.get("from")),
            "to": clean_place_name(endpoints.get("to")),
            "via": clean_place_name(endpoints.get("via")),
        }

    raw_sites = out.get("sites")
    if not isinstance(raw_sites, list):
        meta["extractor_ok"] = False
        return [], meta

    sites: list[Site] = []
    seen: set[str] = set()
    for item in raw_sites:
        if not isinstance(item, dict):
            continue
        name = clean_place_name(item.get("name"))
        admin = coerce_admin_hierarchy(item.get("admin_hierarchy"))
        if not name:
            # Fall back to the finest populated admin level rather than dropping
            # the entry: an unnamed-but-located site is still a location, and
            # dropping entries mid-list is how the old code desynchronized its
            # two parallel lists.
            name = next((admin[k] for k in SPECIFICITY_ORDER if admin.get(k)), None)
        if not name:
            continue
        key = normalize(name)
        if key in seen:
            continue
        seen.add(key)
        role = str(item.get("role") or "primary").strip().lower()
        if role not in ("primary", "alternative", "reference"):
            role = "primary"
        quote = str(item.get("quote") or "").strip()
        ev: Evidence = (
            verify_and_locate(quote, doc)
            if quote
            else {
                "quote": "",
                "source_pages": [],
                "quote_verified": False,
                "note": "No quote returned by extractor.",
            }
        )
        sites.append(Site(name=name, admin_hierarchy=admin, role=role, evidence=[ev]))

    meta["n_sites"] = len(sites)
    meta["n_primary"] = sum(1 for s in sites if s.is_primary)
    return sites, meta


# --- Step 4a: site-scope specificity cascade --------------------------------


def _level_queries(site: Site) -> list[tuple[str, str]]:
    """
    (level, query) pairs finest -> coarsest for one site.

    Each query carries its coarser context ("Cottonwood Field Office, Uintah
    County, Utah") because every vendor in the cascade disambiguates far better
    with it, and the site's own `name` is used as the poi query when the model
    left the poi slot empty -- the name is what the document actually calls the
    place.
    """
    admin = site.admin_hierarchy
    out: list[tuple[str, str]] = []
    for i, level in enumerate(SPECIFICITY_ORDER):
        own = admin.get(level)
        if level == "poi" and not own:
            coarser_names = {
                normalize(admin[k]) for k in SPECIFICITY_ORDER[1:] if admin.get(k)
            }
            if site.name and normalize(site.name) not in coarser_names:
                own = site.name
        if not own:
            continue
        context = [admin[k] for k in SPECIFICITY_ORDER[i + 1 :] if admin.get(k)]
        out.append((level, ", ".join([own] + context)))
    return out


def resolve_site(
    site: Site,
    *,
    stack: Optional[str] = None,
    stats: Optional[HopStats] = None,
) -> tuple[Optional[dict], list[str], str]:
    """
    MCAL_PLAN 3.9 step 4, `site` scope. Returns (accepted, tags, note).

    The rule: geocode each populated admin level, then walk finest -> coarsest
    and accept the finest level whose bbox is contained in the next-coarser
    level's bbox. Containment is what distinguishes "the POI really is inside
    that city" from "the geocoder handed back some other Springfield", and it is
    the mechanism that stops the 1(9b) failure where the coarsest containing
    city was returned for a corridor inside it.

    Two documented deviations from a literal reading of the plan:
      * POI-wins short-circuit (also in the plan): if a POI result's own
        containing city matches the document's stated city, accept the POI
        immediately, without requiring the bbox test. Vendor POI bboxes are
        frequently a single point, and city bboxes are sometimes tight polygons,
        so a name agreement is stronger evidence here than the geometry.
      * If a level has no coarser sibling to be checked against (e.g. the doc
        gave only a city), we accept it with a note instead of rejecting it. The
        alternative is discarding a correct city geocode for lack of a county,
        which would manufacture 1(9a) misses.
    """
    stats = stats if stats is not None else HopStats()
    tags: list[str] = []
    queries = _level_queries(site)
    site.levels_tried = [lvl for lvl, _ in queries]
    if not queries:
        return None, [T_GEOCODE_MISSING], "No admin hierarchy or name to geocode."

    state_hint = site.admin_hierarchy.get("state")
    results: dict[str, dict] = {}
    for level, query in queries:
        res = geocode_cascade(
            query, level=level, state=state_hint, stack=stack, stats=stats
        )
        if res is not None:
            results[level] = res

    if not results:
        return (
            None,
            [T_GEOCODE_MISSING],
            f"No cascade hop resolved any level of {site.name!r}.",
        )

    # POI wins if its containing city agrees with the document's city.
    poi = results.get("poi")
    doc_city = site.admin_hierarchy.get("city")
    if poi is not None and doc_city:
        poi_city = (poi.get("admin_hierarchy") or {}).get("city")
        if names_match(poi_city, doc_city):
            accepted = dict(poi)
            accepted["accepted_level"] = "poi"
            accepted["acceptance_reason"] = "poi_city_matches_document_city"
            return accepted, tags, "POI accepted on city agreement."

    ordered = [lvl for lvl in SPECIFICITY_ORDER if lvl in results]
    failed_checks: list[str] = []
    for i, level in enumerate(ordered):
        coarser = ordered[i + 1] if i + 1 < len(ordered) else None
        if coarser is None:
            # Coarsest available level: there is nothing left to validate
            # against, so it is accepted -- but if finer levels were REJECTED on
            # the way here, the hierarchy is internally inconsistent and the
            # result is only as good as the coarsest box, which is the T07
            # condition.
            accepted = dict(results[level])
            accepted["accepted_level"] = level
            if failed_checks:
                accepted["acceptance_reason"] = "no_level_passed_containment"
                tags.append(T_WRONG_SPECIFICITY)
                note = (
                    f"No finer level passed bbox containment "
                    f"({', '.join(failed_checks)}); accepted {level}."
                )
            else:
                accepted["acceptance_reason"] = "no_coarser_level_available"
                note = f"Accepted {level} without a containment check (no coarser level)."
            if level in ("state", "country") and T_WRONG_SPECIFICITY not in tags:
                tags.append(T_WRONG_SPECIFICITY)
                note += " Coarsest-level-only resolution."
            accepted["failed_containment_checks"] = failed_checks
            return accepted, tags, note
        if bbox_contained(results[level]["bbox"], results[coarser]["bbox"]):
            accepted = dict(results[level])
            accepted["accepted_level"] = level
            accepted["acceptance_reason"] = f"bbox_contained_in_{coarser}"
            accepted["failed_containment_checks"] = failed_checks
            note = f"Accepted {level}; bbox contained in {coarser}."
            if level in ("state", "country"):
                tags.append(T_WRONG_SPECIFICITY)
                note += " Coarsest-level-only resolution."
            return accepted, tags, note
        failed_checks.append(f"{level}_not_in_{coarser}")

    # Unreachable: the loop always accepts at the coarsest level. Kept as a
    # belt-and-braces guard so a future edit cannot silently return None here.
    return None, [T_GEOCODE_MISSING], "No level could be accepted."


# --- Step 4b: corridor endpoints --------------------------------------------
#
# MCAL_PLAN 3.9a: "Corridor endpoint parsing runs before geocoding (regex +
# Sonnet for hard cases: 'from X to Y', 'between X and Y', 'the X-Y segment')."
# Regex first because it is free, deterministic and auditable; the LLM only sees
# the cases it fails on.

_WORD = r"[A-Z][A-Za-z'\u2019.\-]*"
# Up to four capitalized words, optionally followed by a ", State" tail. Inter-word
# whitespace is HORIZONTAL only: a place name never spans a line break, and
# allowing \s+ let "Cleveland, Ohio.\n\nRight-of-way" parse as one endpoint.
_SP = r"[ \t]{1,3}"
_SP0 = r"[ \t]{0,3}"
_PLACE = (
    rf"{_WORD}(?:{_SP}{_WORD}){{0,3}}"
    rf"(?:{_SP0},{_SP0}{_WORD}(?:{_SP}{_WORD}){{0,2}})?"
)

# Capitalized words that are never a place on their own. Without this,
# "Environmental Impact Statement to Final" style OCR noise parses as a corridor.
_NOT_A_PLACE = frozenset(
    """
    the a an this that these those environmental impact statement draft final
    supplemental record decision chapter section appendix table figure page
    project proposed action alternative alternatives no build volume summary
    purpose need affected environment consequences mitigation consultation
    coordination federal state county city agency department bureau administration
    highway interstate route corridor segment north south east west northern
    southern eastern western january february march april may june july august
    september october november december
    """.split()
)

_CORRIDOR_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("from_to", rf"\bfrom\s+(?P<a>{_PLACE})\s+to\s+(?P<b>{_PLACE})", 3.0),
    ("between_and", rf"\bbetween\s+(?P<a>{_PLACE})\s+and\s+(?P<b>{_PLACE})", 3.0),
    (
        "segment",
        rf"\b(?P<a>{_PLACE})\s*[\-\u2013\u2014]\s*(?P<b>{_PLACE})\s+"
        r"(?:segment|corridor|section|route|line|freeway|expressway|highway)\b",
        2.5,
    ),
    ("plain_to", rf"\b(?P<a>{_PLACE})\s+to\s+(?P<b>{_PLACE})\b", 1.0),
)

_VIA_RE = re.compile(rf"\bvia\s+(?P<via>{_PLACE})")

CORRIDOR_SCAN_CHARS = 120_000

# Trailing sentence punctuation must come off a captured endpoint ("... to
# Cleveland, Ohio." -> "Cleveland, Ohio"), or the geocoder query and the
# site_name join key both carry a stray period. A final single-letter
# abbreviation keeps its period, so "Washington, D.C." survives intact.
_TRAILING_PUNCT_RE = re.compile(r"(?<!\b[A-Z])[\s.,;:]+$")


def _trim_endpoint(s: str) -> str:
    return _TRAILING_PUNCT_RE.sub("", (s or "").strip()).strip()


@dataclass
class CorridorEndpoints:
    endpoint_a: str
    endpoint_b: str
    via: Optional[str] = None
    pattern: str = ""
    quote: str = ""
    source: str = "regex"     # regex | site_extractor | sonnet
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "endpoint_a": self.endpoint_a,
            "endpoint_b": self.endpoint_b,
            "via": self.via,
            "pattern": self.pattern,
            "quote": self.quote,
            "source": self.source,
            "score": round(self.score, 2),
        }


def looks_like_place(candidate: Optional[str]) -> bool:
    """Cheap plausibility filter for a regex-captured endpoint."""
    s = (candidate or "").strip().strip(",;:")
    if not (3 <= len(s) <= 60):
        return False
    tokens = [t for t in re.split(r"[\s,]+", s) if t]
    if not tokens:
        return False
    meaningful = [t for t in tokens if t.strip(".").lower() not in _NOT_A_PLACE]
    if not meaningful:
        return False
    # At least one token must still look like a proper noun.
    return any(t[:1].isupper() for t in meaningful)


def parse_corridor_endpoints(text: str) -> Optional[CorridorEndpoints]:
    """
    Deterministic endpoint extraction (MCAL_PLAN 3.9a implementation notes).

    All patterns are scanned and the best-scoring plausible match wins, rather
    than the first match: "from X to Y" appearing in the purpose-and-need
    statement is far better evidence than an incidental "Chicago to Denver" in a
    references list, and pattern strength encodes that.
    """
    if not text:
        return None
    window = text[:CORRIDOR_SCAN_CHARS]
    best: Optional[CorridorEndpoints] = None
    for name, pattern, weight in _CORRIDOR_PATTERNS:
        for m in re.finditer(pattern, window):
            a = _trim_endpoint(m.group("a"))
            b = _trim_endpoint(m.group("b"))
            if not looks_like_place(a) or not looks_like_place(b):
                continue
            if normalize(a) == normalize(b):
                continue
            score = weight
            # A ", State" tail on either side is strong evidence these are real
            # places and not a sentence fragment.
            if "," in a or "," in b:
                score += 0.5
            tail = window[m.end() : m.end() + 80]
            via_m = _VIA_RE.search(tail)
            via = _trim_endpoint(via_m.group("via")) if via_m else None
            if via and not looks_like_place(via):
                via = None
            cand = CorridorEndpoints(
                endpoint_a=a,
                endpoint_b=b,
                via=via,
                pattern=name,
                quote=m.group(0).strip(),
                source="regex",
                score=score,
            )
            if best is None or cand.score > best.score:
                best = cand
    return best


CORRIDOR_SYSTEM = (
    "You identify the two ENDPOINTS of a linear project (highway, transit line,\n"
    "transmission line, pipeline, railway) described in an EIS excerpt.\n"
    "Respond ONLY with JSON:\n"
    '{"endpoint_a": "<place or null>", "endpoint_b": "<place or null>",\n'
    ' "via": "<named intermediate place or null>",\n'
    ' "quote": "<verbatim phrase from the excerpt stating the endpoints>"}\n'
    "Use the place names as the document writes them, with state if given. If the\n"
    "excerpt does not state two endpoints, return nulls -- do not guess."
)


def sonnet_corridor_endpoints(text: str) -> Optional[CorridorEndpoints]:
    """Sonnet fallback for the hard cases the regexes cannot parse."""
    out = _sonnet_json(
        CORRIDOR_SYSTEM, f"EXCERPT:\n{text[:MAX_PROMPT_CHARS]}", max_tokens=400
    )
    if out is None:
        return None
    a = clean_place_name(out.get("endpoint_a"))
    b = clean_place_name(out.get("endpoint_b"))
    if not a or not b or normalize(a) == normalize(b):
        return None
    return CorridorEndpoints(
        endpoint_a=a,
        endpoint_b=b,
        via=clean_place_name(out.get("via")),
        pattern="llm",
        quote=str(out.get("quote") or "").strip(),
        source="sonnet",
        score=2.0,
    )


def _state_hint(sites: list[Site]) -> Optional[str]:
    for s in sites:
        if s.is_primary and s.admin_hierarchy.get("state"):
            return s.admin_hierarchy["state"]
    for s in sites:
        if s.admin_hierarchy.get("state"):
            return s.admin_hierarchy["state"]
    return None


def resolve_corridor(
    endpoints: CorridorEndpoints,
    sites: list[Site],
    *,
    stack: Optional[str] = None,
    stats: Optional[HopStats] = None,
) -> tuple[list[dict], list[str], str]:
    """
    MCAL_PLAN 3.9 step 4, `corridor` scope. Returns (points, tags, note).

    Each endpoint runs the FULL cascade independently (3.9a), then both are
    validated against the coarser admin container by bbox intersection AND
    centroid containment -- intersection rather than containment because a
    corridor endpoint's own bbox legitimately straddles a state line.

    Deviation worth stating: the plan asks for the midpoint to run the cascade
    too, but a geometric midpoint has no name to geocode. We geocode a NAMED
    intermediate ("via Toledo") through the full cascade when the document
    supplies one, and otherwise interpolate between the two endpoints and label
    the point `source="interpolated"` so it is never mistaken for a vendor hit.
    """
    stats = stats if stats is not None else HopStats()
    tags: list[str] = []
    notes: list[str] = []
    state = _state_hint(sites)

    a = geocode_cascade(
        endpoints.endpoint_a, level="city", state=state, stack=stack, stats=stats
    )
    b = geocode_cascade(
        endpoints.endpoint_b, level="city", state=state, stack=stack, stats=stats
    )

    # Coarser container: prefer the document's stated state, else whatever the
    # endpoints themselves resolved to.
    container_name = state
    if not container_name:
        for res in (a, b):
            if res and (res.get("admin_hierarchy") or {}).get("state"):
                container_name = res["admin_hierarchy"]["state"]
                break
    container = (
        geocode_cascade(container_name, level="state", stack=stack, stats=stats)
        if container_name
        else None
    )

    def _validate(res: Optional[dict], role: str) -> Optional[dict]:
        if res is None:
            return None
        point = dict(res)
        point["point_role"] = role
        if container is None:
            point["container_checked"] = False
            return point
        point["container_checked"] = True
        point["container"] = container_name
        intersects = bbox_intersects(point["bbox"], container["bbox"])
        inside = point_in_bbox(point["lat"], point["lon"], container["bbox"])
        point["container_ok"] = bool(intersects and inside)
        if not point["container_ok"]:
            notes.append(
                f"{role} {point.get('query')!r} failed the {container_name} "
                f"container check (intersects={intersects}, centroid_inside={inside})."
            )
        return point

    points: list[dict] = []
    pa = _validate(a, "endpoint_a")
    pb = _validate(b, "endpoint_b")

    mid: Optional[dict] = None
    if endpoints.via:
        mid = _validate(
            geocode_cascade(
                endpoints.via, level="city", state=state, stack=stack, stats=stats
            ),
            "midpoint",
        )
    if mid is None and pa is not None and pb is not None:
        lat = (pa["lat"] + pb["lat"]) / 2.0
        lon = (pa["lon"] + pb["lon"]) / 2.0
        mid = _result(
            lat=lat,
            lon=lon,
            bbox=None,
            source="interpolated",
            confidence=round(
                0.8 * min(pa.get("confidence", 0.0), pb.get("confidence", 0.0)), 3
            ),
            admin_hierarchy={"state": container_name},
            level="city",
            query=f"midpoint({endpoints.endpoint_a} .. {endpoints.endpoint_b})",
        )
        mid["point_role"] = "midpoint"
        mid["container_checked"] = container is not None
        if container is not None:
            mid["container_ok"] = point_in_bbox(lat, lon, container["bbox"])

    for p in (pa, mid, pb):
        if p is not None:
            p["corridor"] = True
            points.append(p)

    resolved_endpoints = sum(1 for p in (pa, pb) if p is not None)
    if resolved_endpoints == 0:
        tags.append(T_GEOCODE_MISSING)
        notes.append("Neither corridor endpoint resolved.")
    elif resolved_endpoints == 1:
        tags.append(T_MULTI_SITE_PARTIAL)
        notes.append("Only one corridor endpoint resolved.")
    return points, tags, " ".join(notes)


# --- Step 4c: regional scope ------------------------------------------------


def resolve_regional(
    region_name: Optional[str],
    sites: list[Site],
    *,
    stack: Optional[str] = None,
    stats: Optional[HopStats] = None,
) -> tuple[list[dict], list[str], str, bool]:
    """
    MCAL_PLAN 3.9 step 4, `regional` scope.

    Returns (points, tags, note, human_review). Query the stated region first
    (that is the coarsest admin level whose name matches the doc's own words);
    fall back to the bounding-polygon centroid of the enumerated primary sites;
    and with fewer than two primary sites and no usable region name there is
    nothing defensible to emit, so tag T14 and route to HUMAN_REVIEW.
    """
    stats = stats if stats is not None else HopStats()
    tags: list[str] = []
    notes: list[str] = []
    primary = [s for s in sites if s.is_primary]

    region_point: Optional[dict] = None
    if region_name:
        region_point = geocode_cascade(
            region_name, level="region", stack=stack, stats=stats
        )
        if region_point is not None:
            region_point = dict(region_point)
            region_point["point_role"] = "region"
            region_point["accepted_level"] = "region"
            notes.append(f"Region {region_name!r} resolved via {region_point['source']}.")
        else:
            notes.append(f"Region {region_name!r} did not resolve in any hop.")

    site_points: list[tuple[float, float]] = []
    for s in primary:
        res, site_tags, site_note = resolve_site(s, stack=stack, stats=stats)
        s.geocode = res
        s.geocode_note = site_note
        if res is not None:
            site_points.append((res["lat"], res["lon"]))

    points: list[dict] = []
    if region_point is not None:
        points.append(region_point)
    elif len(site_points) >= 2:
        c = points_centroid(site_points)
        if c is not None:
            lat, lon = c
            centroid = _result(
                lat=lat,
                lon=lon,
                bbox=None,
                source="site_polygon_centroid",
                confidence=0.5,
                admin_hierarchy={"state": _state_hint(sites)},
                level="region",
                query=f"centroid of {len(site_points)} primary sites",
            )
            centroid["point_role"] = "region"
            centroid["accepted_level"] = "region"
            points.append(centroid)
            notes.append(
                f"Region derived as the bounding-polygon centroid of "
                f"{len(site_points)} primary sites."
            )

    human_review = False
    if not points and len(primary) < 2:
        tags.append(T_REGIONAL_UNDERSPECIFIED)
        human_review = True
        notes.append(
            f"Regional scope with {len(primary)} primary site(s) and no resolvable "
            f"region name -- underspecified."
        )
    elif not points:
        tags.append(T_GEOCODE_MISSING)
        notes.append("Regional scope: neither the region name nor any site resolved.")
    return points, tags, " ".join(notes), human_review


# --- Textual location (MCAL_PLAN 3.9 step 5) --------------------------------


def textual_location(scope: str, sites: list[Site], region_name: Optional[str]) -> str:
    """
    The human-readable place string, independent of any geocode.

    MCAL_PLAN 3.9 step 5: "a named place without coordinates is still valid
    output". This is computed from the extracted sites alone and is never
    conditioned on geocoding success, which is the structural guarantee that a
    1(9a) geocode miss can no longer erase the location. Primary sites first;
    non-primary sites are used only if the document named nothing primary.
    """
    if scope in PLACELESS_SCOPES:
        return scope
    if region_name:
        return region_name
    primary = [s.name for s in sites if s.is_primary and s.name]
    if primary:
        return "; ".join(primary)
    other = [s.name for s in sites if s.name]
    return "; ".join(other)


# --- Top level --------------------------------------------------------------


def run_location_pipeline(
    doc: Doc,
    chapters: Optional[list[dict]] = None,
    *,
    stack: Optional[str] = None,
    scope: Optional[ScopeDecision] = None,
) -> dict:
    """
    Full scope-conditional location pipeline (MCAL_PLAN 3.9, build item #8).

    `stack` overrides the asset precheck (mostly for tests); `scope` lets a
    caller supply an already-computed classification instead of paying for the
    call twice.

    Output contract:
        {
          scope, scope_justification, classifier_source,
          sites: [{name, admin_hierarchy, role, evidence, geocode, ...}],
          geocoded: [{site_name, lat, lon, bbox, source, confidence, ...}],
          textual_location, corridor, corridor_endpoints,
          tags, human_review, notes,
          geocoder_stack, reduced_mode, hop_stats, geocoder_assets
        }

    `geocoded[i]` carries `site_name`, and each site carries its own `geocode`
    payload, so the two collections are joinable by name and never by index --
    the `_geocode_places` bug this module was written to remove.
    """
    stack = resolve_stack(stack)
    stats = HopStats()
    tags: list[str] = []
    notes: list[str] = []

    # `chapters` is normally handed in by the M2 driver (chunk.chunks_for_doc).
    # Recovering it here keeps the module usable standalone -- and a silently
    # empty chapter list would mean the Project/Study Area section is never
    # found, which is a quiet recall loss rather than a visible failure.
    if chapters is None:
        try:
            chapters = detect_chapters(doc)
        except Exception as e:  # pragma: no cover - detector is pure regex
            log.warning(f"chapter detection failed: {e}")
            chapters = []

    decision = scope or classify_scope(doc, chapters)
    if decision.source == "default_on_error":
        notes.append("Scope classifier degraded to the `site` default.")

    result: dict = {
        **decision.to_dict(),
        "sites": [],
        "geocoded": [],
        "textual_location": "",
        "corridor": False,
        "corridor_endpoints": None,
        "tags": [],
        "human_review": False,
        "notes": [],
        "geocoder_stack": stack,
        "reduced_mode": stack == "reduced",
        "hop_stats": stats.to_dict(),
        "geocoder_assets": {},
    }

    # --- Step 2: national / international short-circuit ---------------------
    # No geocoding at all. This is the 1(9d) fix: "national" is now an ANSWER,
    # not the absence of one, and it costs exactly one Sonnet call.
    if decision.is_placeless:
        result["textual_location"] = decision.scope
        result["notes"] = notes + [
            f"Scope={decision.scope}: no geocoding attempted (MCAL_PLAN 3.9 step 2)."
        ]
        return result

    # --- Step 3: site extraction -------------------------------------------
    sites, meta = extract_sites(doc, chapters)
    result["site_extraction_meta"] = meta
    if not meta.get("extractor_ok"):
        notes.append("Site extractor returned no usable JSON.")
    region_name = meta.get("stated_region") or decision.stated_region

    primary = [s for s in sites if s.is_primary]
    rejected = [s for s in sites if not s.is_primary]
    if rejected:
        notes.append(
            f"{len(rejected)} non-primary site(s) retained but excluded from "
            f"geocoding per MCAL_PLAN 3.9 step 3."
        )

    geocoded: list[dict] = []

    # --- Step 4: scope-conditional geocoding -------------------------------
    if decision.scope == "corridor":
        endpoints = parse_corridor_endpoints(doc.full_text)
        llm_pair = meta.get("llm_corridor_endpoints") or {}
        if endpoints is None and llm_pair.get("from") and llm_pair.get("to"):
            # Free: the site extractor was already asked for endpoints.
            endpoints = CorridorEndpoints(
                endpoint_a=llm_pair["from"],
                endpoint_b=llm_pair["to"],
                via=llm_pair.get("via"),
                pattern="site_extractor",
                source="site_extractor",
                score=2.0,
            )
        if endpoints is None:
            endpoints = sonnet_corridor_endpoints(
                first_pages(doc, min(30, doc.n_pages or 30))
            )
        if endpoints is None:
            notes.append(
                "Corridor scope but no endpoints could be parsed; falling back to "
                "per-site resolution."
            )
            geocoded.extend(_resolve_sites_individually(primary, stack, stats, tags))
        else:
            result["corridor_endpoints"] = endpoints.to_dict()
            points, corridor_tags, corridor_note = resolve_corridor(
                endpoints, sites, stack=stack, stats=stats
            )
            tags.extend(corridor_tags)
            if corridor_note:
                notes.append(corridor_note)
            result["corridor"] = True
            for p in points:
                p["site_name"] = (
                    endpoints.endpoint_a
                    if p.get("point_role") == "endpoint_a"
                    else endpoints.endpoint_b
                    if p.get("point_role") == "endpoint_b"
                    else endpoints.via or "midpoint"
                )
            geocoded.extend(points)

    elif decision.scope == "regional":
        points, region_tags, region_note, human_review = resolve_regional(
            region_name, sites, stack=stack, stats=stats
        )
        tags.extend(region_tags)
        if region_note:
            notes.append(region_note)
        result["human_review"] = result["human_review"] or human_review
        for p in points:
            p["site_name"] = region_name or "region"
        geocoded.extend(points)
        # resolve_regional already resolved each primary site; surface those too
        # so a regional doc still reports its constituent coordinates.
        for s in primary:
            if s.geocode is not None:
                geocoded.append({**s.geocode, "site_name": s.name, "role": s.role})

    else:  # site scope (and the default when classification degraded)
        geocoded.extend(_resolve_sites_individually(primary, stack, stats, tags))
        n_resolved = sum(1 for s in primary if s.geocode is not None)
        if primary and n_resolved == 0:
            if T_GEOCODE_MISSING not in tags:
                tags.append(T_GEOCODE_MISSING)
            notes.append("No primary site resolved to coordinates.")
        elif len(primary) > 1 and n_resolved < len(primary):
            # 1(9c): partial multi-site coverage must not look like success.
            tags.append(T_MULTI_SITE_PARTIAL)
            notes.append(
                f"{n_resolved}/{len(primary)} primary sites geocoded."
            )
        if not primary:
            tags.append(T_GEOCODE_MISSING)
            notes.append("Site scope but no primary site was extracted.")

    # --- Step 5: textual location is retained regardless -------------------
    result["textual_location"] = textual_location(decision.scope, sites, region_name)
    if not result["textual_location"] and result.get("corridor_endpoints"):
        # A corridor whose endpoints parsed but whose site extraction came back
        # empty still has a perfectly good textual location.
        ep = result["corridor_endpoints"]
        result["textual_location"] = f"{ep['endpoint_a']} to {ep['endpoint_b']}"
    if not geocoded and result["textual_location"]:
        notes.append(
            "Geocoding produced nothing; textual location retained "
            "(MCAL_PLAN 3.9 step 5)."
        )
    elif not geocoded:
        notes.append("Neither coordinates nor a place name could be produced.")

    if stack == "reduced":
        notes.append(
            "Reduced geocoder stack (Census + Nominatim only): PAD-US, GNIS or "
            "MAPBOX_TOKEN unavailable. The location bucket is gated to "
            "HUMAN_REVIEW in this mode (MCAL_PLAN 3.9a)."
        )

    result["sites"] = [s.to_dict() for s in sites]
    result["geocoded"] = geocoded
    result["tags"] = _dedupe(tags)
    result["notes"] = notes
    result["hop_stats"] = stats.to_dict()
    result["geocoder_assets"] = {
        "stack": stack,
        "precheck_missing": _precheck_missing(),
        # What gate.py acts on: MCAL_PLAN 3.9a forces the location bucket to
        # HUMAN_REVIEW whenever the stack is reduced. Surfaced here so the
        # decision travels with the extraction instead of having to be
        # re-derived from thresholds.v(N).json.
        "gate_all_to_human": stack == "reduced",
    }
    _record_global(stats)
    return result


def _resolve_sites_individually(
    sites: list[Site],
    stack: Optional[str],
    stats: HopStats,
    tags: list[str],
) -> list[dict]:
    """
    Resolve each site independently and pair the result WITH the site.

    This is the shape of the `_geocode_places` fix: the geocode payload is stored
    on the Site object and every emitted row carries `site_name`, so no consumer
    can be misled by an index. Sites that fail to resolve are recorded on the
    site (geocode=None + note) and simply contribute no row -- which is safe
    precisely because rows are name-keyed.
    """
    out: list[dict] = []
    for site in sites:
        res, site_tags, note = resolve_site(site, stack=stack, stats=stats)
        site.geocode = res
        site.geocode_note = note
        for t in site_tags:
            if t not in tags:
                tags.append(t)
        if res is not None:
            out.append({**res, "site_name": site.name, "role": site.role})
    return out


def _precheck_missing() -> list[str]:
    try:
        return list(settings.geocoder_precheck().get("missing", []))
    except Exception:  # pragma: no cover - precheck never raises by contract
        return []


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for i in items:
        seen.setdefault(i, None)
    return list(seen.keys())


def as_m2_location_field(result: dict) -> dict:
    """
    Adapt the pipeline output to the `location` field shape M2 emits, so the
    Critic, grading sheets and gate.py keep working unchanged.

    `value.places` mirrors segment_a's key name (each entry keeps its evidence),
    and `value.geocoded` is the name-keyed list. The scope-conditional fields are
    additive; nothing that read the old shape breaks.
    """
    return {
        "value": {
            "places": [
                {
                    "name": s["name"],
                    "admin_hierarchy": s["admin_hierarchy"],
                    "role": s["role"],
                    "evidence": s["evidence"],
                }
                for s in result.get("sites", [])
            ],
            "is_multi_site": sum(
                1 for s in result.get("sites", []) if s.get("role") == "primary"
            ) > 1,
            "geocoded": result.get("geocoded", []),
            "scope": result.get("scope"),
            "textual_location": result.get("textual_location"),
            "corridor": result.get("corridor", False),
        },
        "confidence": _field_confidence(result),
        "note": " ".join(result.get("notes", []))[:1000],
        "tags": result.get("tags", []),
        "human_review": result.get("human_review", False),
        "geocoder_stack": result.get("geocoder_stack"),
        "hop_stats": result.get("hop_stats", {}),
    }


def _field_confidence(result: dict) -> str:
    """Coarse high/medium/low, matching the vocabulary the grading sheet uses."""
    if result.get("human_review"):
        return "low"
    if result.get("scope") in PLACELESS_SCOPES:
        return "high" if result.get("classifier_source") == "sonnet" else "medium"
    geocoded = result.get("geocoded") or []
    if not geocoded:
        return "low"
    if result.get("tags"):
        return "medium"
    confident = [g for g in geocoded if g.get("confidence", 0) >= CONFIDENT_MIN]
    return "high" if confident and result.get("geocoder_stack") == "full" else "medium"
