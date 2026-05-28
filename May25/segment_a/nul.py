"""
NUL Digital Collections fetch + accession matching.

Caches the full collection response so we only hit the API once per run.
Match strategy preserved from V1 pipeline (3 fallback strategies).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import requests

from config import COLLECTION_ID, DOCS_WITH_DIGITS_PATH, NUL_API_BASE, NUL_CACHE_PATH

log = logging.getLogger(__name__)

NUL_FIELDS_TO_KEEP = [
    "id", "title", "alternate_title", "description", "abstract",
    "date_created", "date_created_edtf", "create_date", "modified_date",
    "work_type", "visibility", "published",
    "rights_statement", "terms_of_use", "license",
    "collection", "contributor", "creator", "subject", "genre", "language",
    "publisher", "source", "related_url", "identifier",
    "accession_number", "catalog_key", "library_unit",
    "physical_description_size", "physical_description_material",
    "box_number", "box_name", "folder_number", "folder_name", "series",
    "style_period", "technique", "scope_and_contents", "table_of_contents",
    "notes", "caption", "provenance", "keywords", "nav_place",
    "iiif_manifest", "thumbnail", "representative_file_set", "file_sets",
    "batch_ids", "project", "ingest_project", "ingest_sheet",
]


def fetch_collection_works(force: bool = False) -> list[dict]:
    """Fetch all works in the collection. Cached to NUL_CACHE_PATH on disk."""
    if NUL_CACHE_PATH.exists() and not force:
        log.info(f"Loading NUL works from cache: {NUL_CACHE_PATH}")
        with open(NUL_CACHE_PATH) as f:
            return json.load(f)

    NUL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    works: list[dict] = []
    page = 1
    while True:
        log.info(f"Fetching NUL works (page={page})...")
        last_err: Optional[Exception] = None
        data = None
        for attempt in range(5):
            try:
                resp = requests.get(
                    f"{NUL_API_BASE}/search",
                    params={
                        "query": f"collection.id:{COLLECTION_ID}",
                        "size": 25,
                        "page": page,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                last_err = e
                wait = 2 ** attempt
                log.warning(f"  page={page} attempt {attempt+1}/5 failed: {e}; sleeping {wait}s")
                time.sleep(wait)
        if data is None:
            raise RuntimeError(f"NUL search failed after retries: {last_err}")
        hits = data.get("data", [])
        if not hits:
            break
        works.extend(hits)
        if not data.get("pagination", {}).get("next_url"):
            break
        page += 1
        time.sleep(0.5)

    log.info(f"Fetched {len(works)} NUL works. Caching to {NUL_CACHE_PATH}")
    with open(NUL_CACHE_PATH, "w") as f:
        json.dump(works, f, indent=2, ensure_ascii=False)
    return works


def load_docs() -> dict[str, str]:
    """Load the OCR text mapping keyed by accession-like ID."""
    with open(DOCS_WITH_DIGITS_PATH) as f:
        return json.load(f)


def _identifier_strings(work: dict) -> list[str]:
    """Flatten the `identifier` field into a list of strings (handles list/dict/str shapes)."""
    raw = work.get("identifier")
    if not raw:
        return []
    out: list[str] = []

    def harvest(obj: Any) -> None:
        if isinstance(obj, str):
            out.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                harvest(v)
        elif isinstance(obj, list):
            for item in obj:
                harvest(item)

    harvest(raw)
    return out


def find_ocr(work: dict, docs: dict[str, str]) -> Optional[tuple[str, str]]:
    """Match a NUL work to a docs_with_digits entry. Returns (doc_id, text) or None."""
    accession = work.get("accession_number", "")
    if accession:
        if accession in docs:
            return accession, docs[accession]
        a_lower = accession.lower()
        for doc_id in docs:
            if doc_id.lower() == a_lower:
                return doc_id, docs[doc_id]

    id_strings = _identifier_strings(work)
    for s in id_strings:
        if s in docs:
            return s, docs[s]
    for s in id_strings:
        if not s:
            continue
        for doc_id in docs:
            if s in doc_id:
                return doc_id, docs[doc_id]
    return None


def extract_nul_metadata(work: dict) -> dict:
    """Keep only the fields we care about from a NUL work."""
    out = {f: work[f] for f in NUL_FIELDS_TO_KEEP if f in work}
    work_id = work.get("id", "")
    out["source_url"] = f"https://dc.library.northwestern.edu/items/{work_id}"
    out["api_link"] = f"{NUL_API_BASE}/works/{work_id}"
    return out


def get_year(work: dict) -> Optional[int]:
    """Pull a 4-digit year from create_date / date_created / date_created_edtf."""
    import re
    for key in ("create_date", "date_created_edtf", "date_created"):
        val = work.get(key)
        if not val:
            continue
        if isinstance(val, list):
            val = " ".join(str(x) for x in val)
        m = re.search(r"\b(19\d{2}|20\d{2})\b", str(val))
        if m:
            return int(m.group(1))
    return None


def get_contributors(work: dict) -> list[str]:
    """Return a list of contributor/creator labels."""
    out: list[str] = []
    for key in ("contributor", "creator"):
        val = work.get(key)
        if not val:
            continue
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    label = item.get("label_with_role") or item.get("label") or item.get("name")
                    if label:
                        out.append(label)
                elif isinstance(item, str):
                    out.append(item)
        elif isinstance(val, str):
            out.append(val)
    return out
