"""
Stage 2.3 — Location extraction + geocoding.
"""

from __future__ import annotations

import logging
import textwrap
import time
from typing import TYPE_CHECKING

from geopy.geocoders import Nominatim  # type: ignore
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable  # type: ignore

from ..config import MODELS
from ..schema import EISRecord, LocationInfo
from .retrieval import combine_chunk_context, get_chunks_by_tags

if TYPE_CHECKING:
    from ..llm_client import LLMClient

logger = logging.getLogger(__name__)

_GEOCODER = Nominatim(user_agent="eis_pipeline_v1/1.0")
_GEOCODE_RATE_LIMIT_S = 1.0  # Nominatim ToS: max 1 req/sec

_SYSTEM = textwrap.dedent("""\
    You are extracting the primary geographic location from a U.S. Environmental Impact Statement.

    Return ONLY valid JSON:
    {
      "place_name": "Cedar Creek, Wyoming",
      "state": "WY"
    }

    Rules:
    - place_name: the most specific named place (city/county/region/corridor). Do not invent or guess.
    - state: 2-letter U.S. state abbreviation, or null if federal/international/unknown.
    - If there are multiple locations, pick the one most central to the project.
    - If you cannot determine the location, return {"place_name": null, "state": null}.
""")

_USER_TMPL = textwrap.dedent("""\
    Document title: {title}

    Relevant chunks:
    {context}
""")


def run(record: EISRecord, client: "LLMClient") -> None:
    """Populate record.location. Mutates record in place."""
    target_tags = ["affected_environment", "purpose_and_need", "proposed_action"]
    chunks = get_chunks_by_tags(record.chunks, target_tags, max_chunks=4)

    if not chunks:
        # Location is usually stated in the first pages — use them as a fallback
        # when no chunks carry the relevant tags.
        logger.info(
            "Location: no tagged chunks for %s — using first usable chunks",
            record.doc_id,
        )
        chunks = [c for c in record.chunks if c.used][:4]

    context = combine_chunk_context(chunks, max_chars=16_000) if chunks else ""

    user_msg = _USER_TMPL.format(
        title=record.title or record.doc_id,
        context=context or "(no chunks available)",
    )

    try:
        result = client.call_json(
            model=MODELS["light"],
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=128,
            temperature=0.1,
            label=f"location/{record.doc_id}",
        )
        place_name = result.get("place_name") or None
        state = result.get("state") or None
    except Exception as exc:
        logger.error("Location extraction failed for %s: %s", record.doc_id, exc)
        record.location = LocationInfo()
        return

    lat, lon, geocode_source = None, None, None
    if place_name:
        lat, lon, geocode_source = _geocode(place_name)

    record.location = LocationInfo(
        name=place_name,
        state=state,
        latitude=lat,
        longitude=lon,
        geocode_source=geocode_source,
    )


def _geocode(place_name: str) -> tuple[float | None, float | None, str | None]:
    try:
        time.sleep(_GEOCODE_RATE_LIMIT_S)
        geo = _GEOCODER.geocode(place_name, timeout=10)
        if geo:
            return round(geo.latitude, 6), round(geo.longitude, 6), "geopy_nominatim"
        logger.info("Geocode returned no results for: %r", place_name)
        return None, None, None
    except (GeocoderTimedOut, GeocoderUnavailable) as exc:
        logger.warning("Geocoding failed for %r: %s", place_name, exc)
        return None, None, None
    except Exception as exc:
        logger.warning("Unexpected geocoding error for %r: %s", place_name, exc)
        return None, None, None
