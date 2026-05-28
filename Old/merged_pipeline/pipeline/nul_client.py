"""
NUL Digital Collections API client with disk cache.

Source of authoritative title/agency/date metadata for v1 (per
synthesis_plan.md §3a: METS-first with extractive cross-check fallback).

Cache lives at output/nul_cache/{doc_id}.json — keyed on doc_id (the part
after the `p1074_` prefix), not the full doc_key, because some accession
lookups historically fell back to the bare barcode.

Note: in the current NUL collection, the bare-barcode fallback returns 0
hits — every record is indexed with the full `p1074_<barcode>` accession
number. The fallback is retained as defensive cheap insurance for any
future corpus where bare-barcode indexing exists.

Ported from v1_multiagent_pipeline/pipeline/nul_client.py without behavior
changes; only docstring revisions.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)


@dataclass
class NULRecord:
    """Parsed NUL response. Empty fields = NUL had nothing for them."""
    title: str | None = None
    creator: list[str] = field(default_factory=list)   # raw NUL creators (agencies)
    contributor: list[str] = field(default_factory=list)
    date_created: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    found: bool = False


class NULClient:
    """NUL API + on-disk cache. Cache is permanent unless manually deleted."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else config.DEFAULT_NUL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get(self, doc_key: str) -> NULRecord:
        """
        Fetch metadata for a doc_key like 'p1074_35556036099737'.
        Reads cache if present; otherwise calls NUL and writes the cache.
        """
        doc_id = self._doc_id_from_key(doc_key)
        cache_path = self.cache_dir / f"{doc_id}.json"

        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                logger.debug("NUL cache hit: %s", cache_path)
                return self._record_from_cached(cached)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("NUL cache read failed %s (%s) — refetching", cache_path, exc)

        record = self._fetch(doc_key, doc_id)

        # Write cache regardless of found/not-found so we don't keep retrying misses
        cache_payload = {
            "doc_key": doc_key,
            "doc_id": doc_id,
            "fetched_at": time.time(),
            "found": record.found,
            "title": record.title,
            "creator": record.creator,
            "contributor": record.contributor,
            "date_created": record.date_created,
            "raw": record.raw,
        }
        cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
        logger.debug("NUL cache wrote: %s", cache_path)
        return record

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_id_from_key(doc_key: str) -> str:
        parts = doc_key.split("_", 1)
        return parts[1] if len(parts) == 2 else doc_key

    @staticmethod
    def _record_from_cached(cached: dict) -> NULRecord:
        rec = NULRecord(
            title=cached.get("title"),
            creator=cached.get("creator") or [],
            contributor=cached.get("contributor") or [],
            date_created=cached.get("date_created"),
            raw=cached.get("raw") or {},
            cache_hit=True,
            found=cached.get("found", False),
        )
        return rec

    def _fetch(self, doc_key: str, doc_id: str) -> NULRecord:
        try:
            work = self._search(f'accession_number:"{doc_key}"')
            if work is None:
                # Defensive fallback: bare barcode (returns 0 hits in current
                # NUL collection but kept for robustness against re-indexing).
                work = self._search(f'accession_number:"{doc_id}"')
            if work is None:
                logger.info("NUL API: no match for %s", doc_key)
                return NULRecord(found=False)

            rec = NULRecord(
                title=_extract_title(work),
                creator=_extract_string_list(work.get("creator")),
                contributor=_extract_string_list(work.get("contributor")),
                date_created=_extract_date(work),
                raw=work,
                found=True,
            )
            logger.info(
                "NUL API match for %s: title=%r creator=%r date=%r",
                doc_key, rec.title, rec.creator[:1], rec.date_created,
            )
            time.sleep(config.NUL_POLITENESS_DELAY_S)
            return rec

        except Exception as exc:
            logger.warning("NUL API fetch failed for %s: %s", doc_key, exc)
            return NULRecord(found=False)

    def _search(self, query: str) -> dict[str, Any] | None:
        params = urllib.parse.urlencode({"query": query, "size": 1})
        url = f"{config.NUL_API_BASE}/search?{params}"
        logger.debug("NUL API request: %s", url)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=config.NUL_REQUEST_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
        hits = data.get("data", []) or []
        return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def _extract_title(work: dict) -> str | None:
    title = work.get("title")
    if isinstance(title, list):
        return title[0] if title else None
    return title or None


def _extract_string_list(value: Any) -> list[str]:
    """NUL fields like creator/contributor are lists of {label, id} dicts."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                label = item.get("label") or item.get("id")
                if label:
                    out.append(str(label))
            elif item:
                out.append(str(item))
        return out
    return []


def _extract_date(work: dict) -> str | None:
    dates = work.get("date_created")
    if isinstance(dates, list) and dates:
        return str(dates[0])
    if isinstance(dates, str):
        return dates
    return None
