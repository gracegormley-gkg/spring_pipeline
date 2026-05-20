"""
Stage 0 — Deterministic triage (no LLM).
Populates: ocr, eis_type, length_category, word_count, has_headings, has_toc,
           sections, lead_agency, date, year, ner.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from typing import TYPE_CHECKING

from rapidfuzz import process as rfuzz_process  # type: ignore

from .config import (
    AGENCY_FLAT,
    AGENCY_VOCAB,
    DATE_PATTERNS,
    EIS_TYPE_PATTERNS,
    HEADING_ADDRESS_KEYWORDS,
    HEADING_PATTERNS,
    HEADING_ZIP_PATTERN,
    LONG_THRESHOLD,
    MAX_YEAR,
    NEPA_YEAR,
    MIDDLE_INITIAL_PATTERN,
    NAME_PARTICLES,
    NAME_TOKEN_PATTERN,
    OCR_UNCLEAR_THRESHOLD,
    SHORT_THRESHOLD,
    SPACY_MODELS_PREFERENCE,
    TOC_MARKER_PATTERN,
    lookup_agency,
)
from .io_layer import Document
from .ner_dicts import find_ngos, find_tribes
from .schema import (
    EISRecord,
    LeadAgency,
    NERResult,
    OCRInfo,
    SectionInfo,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def run(doc: Document, record: EISRecord) -> list[str]:
    """
    Run all Stage 0 triage steps. Returns a list of warning strings.
    Mutates `record` in place.
    """
    warnings: list[str] = []

    _ocr_confidence(doc, record, warnings)
    _eis_type(doc, record)
    _word_count_and_length(doc, record)
    _headings_and_toc(doc, record)
    _date_and_year(doc, record, warnings)
    _lead_agency(doc, record, warnings)
    _ner(doc, record)
    _title(doc, record)

    return warnings


# ---------------------------------------------------------------------------
# 0.1  OCR confidence
# ---------------------------------------------------------------------------

def _ocr_confidence(doc: Document, record: EISRecord, warnings: list[str]) -> None:
    page_confs = [p.median_confidence for p in doc.pages if p.median_confidence is not None]

    if not page_confs:
        logger.warning("No OCR confidence data found for doc %s", doc.doc_id)
        record.ocr = OCRInfo(
            median_confidence=None,
            page_count=len(doc.pages),
            unclear_document_flag=False,
        )
        warnings.append("ocr_confidence_unavailable")
        return

    median_conf = statistics.median(page_confs)
    flag = median_conf < OCR_UNCLEAR_THRESHOLD

    record.ocr = OCRInfo(
        median_confidence=round(median_conf, 4),
        page_count=len(doc.pages),
        unclear_document_flag=flag,
    )

    if flag:
        msg = f"Low OCR confidence: median={median_conf:.3f} (threshold={OCR_UNCLEAR_THRESHOLD})"
        logger.warning(msg)
        warnings.append(f"ocr_unclear: {msg}")


# ---------------------------------------------------------------------------
# 0.2  EIS type
# ---------------------------------------------------------------------------

def _eis_type(doc: Document, record: EISRecord) -> None:
    # Check only the first 250 words
    first_text = " ".join(doc.full_text.split()[:250])

    matches: set[str] = set()
    for eis_type, pattern in EIS_TYPE_PATTERNS.items():
        if pattern.search(first_text):
            matches.add(eis_type)
            logger.debug("EIS type match: %s (pattern: %s)", eis_type, pattern.pattern)

    if len(matches) == 1:
        record.eis_type = matches.pop()  # type: ignore[assignment]
    else:
        record.eis_type = "Unlabelled"
        if matches:
            logger.info("Multiple EIS type matches for %s: %s — marking Unlabelled", doc.doc_id, matches)


# ---------------------------------------------------------------------------
# 0.3  Word count and length category
# ---------------------------------------------------------------------------

def _word_count_and_length(doc: Document, record: EISRecord) -> None:
    count = len(doc.full_text.split())
    record.word_count = count

    if count < SHORT_THRESHOLD:
        record.length_category = "short"
    elif count > LONG_THRESHOLD:
        record.length_category = "long"
    else:
        record.length_category = "medium"

    logger.info("Word count: %d → %s", count, record.length_category)


# ---------------------------------------------------------------------------
# 0.4  Headings and TOC
# ---------------------------------------------------------------------------

def _headings_and_toc(doc: Document, record: EISRecord) -> None:
    text = doc.full_text

    # TOC: look for marker within first ~10% of the document
    first_10pct = text[: max(1000, len(text) // 10)]
    record.has_toc = bool(TOC_MARKER_PATTERN.search(first_10pct))

    # Headings: look across full doc
    all_heading_matches: list[tuple[str, int]] = []
    for pattern in HEADING_PATTERNS:
        for m in pattern.finditer(text):
            heading_text = m.group(0).strip()
            if _is_real_heading(heading_text):
                all_heading_matches.append((heading_text, m.start()))

    # Deduplicate by position (keep earliest per 100-char window)
    seen_positions: set[int] = set()
    unique_headings: list[tuple[str, int]] = []
    for heading, pos in sorted(all_heading_matches, key=lambda x: x[1]):
        bucket = pos // 100
        if bucket not in seen_positions:
            seen_positions.add(bucket)
            unique_headings.append((heading, pos))

    record.has_headings = len(unique_headings) >= 5
    logger.info("Headings found: %d (has_headings=%s, has_toc=%s)",
                len(unique_headings), record.has_headings, record.has_toc)

    if not record.has_headings:
        return

    # Build sections list with page numbers
    sections: list[SectionInfo] = []
    for i, (heading_text, char_pos) in enumerate(unique_headings):
        start_page = _char_to_page(doc, char_pos) or 1

        # end_page = page before next heading (or last page)
        if i + 1 < len(unique_headings):
            next_pos = unique_headings[i + 1][1]
            end_page = max(start_page, (_char_to_page(doc, next_pos) or start_page) - 1)
        else:
            end_page = len(doc.pages)

        sections.append(SectionInfo(
            title=heading_text[:120],
            start_page=start_page,
            end_page=end_page,
        ))

    record.sections = sections


def _char_to_page(doc: Document, offset: int) -> int | None:
    for page in doc.pages:
        if page.char_start <= offset <= page.char_end:
            return page.page_num
    return None


def _is_real_heading(text: str) -> bool:
    """
    Reject heading-pattern matches that are actually legal citations
    ("Section 4(f)"), street addresses ("143 SOUTH THIRD STREET"), or
    OCR letterhead fragments. Without this filter, consultation-appendix
    correspondence floods the section list (see Harlem Ave test run).
    """
    text = text.strip()
    if not text:
        return False

    if len(text.split()) < 3:
        return False

    if HEADING_ADDRESS_KEYWORDS.search(text):
        return False

    if HEADING_ZIP_PATTERN.search(text):
        return False

    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / len(text) < 0.4:
        return False

    return True


# ---------------------------------------------------------------------------
# 0.5  Date and year
# ---------------------------------------------------------------------------

def _date_and_year(doc: Document, record: EISRecord, warnings: list[str]) -> None:
    # Scan first 5 pages, then fall back to full doc
    first_5_text = " ".join(p.text for p in doc.pages[:5])
    year_from_first5 = _extract_dominant_year(first_5_text)
    year_from_full = _extract_dominant_year(doc.full_text) if year_from_first5 is None else None

    extracted_year = year_from_first5 or year_from_full

    # Cross-check with METS
    mets_year: int | None = None
    if doc.mets.date:
        m = re.search(r"\b(\d{4})\b", doc.mets.date)
        if m:
            y = int(m.group(1))
            if NEPA_YEAR <= y <= MAX_YEAR:
                mets_year = y

    if mets_year is not None and extracted_year is not None:
        if abs(mets_year - extracted_year) > 1:
            warnings.append(
                f"year_mismatch: METS={mets_year} vs extracted={extracted_year} — using METS"
            )
            logger.warning(
                "Year mismatch for %s: METS=%d extracted=%d — using METS",
                doc.doc_id, mets_year, extracted_year,
            )
        record.year = mets_year
        record.date = doc.mets.date
    elif mets_year is not None:
        record.year = mets_year
        record.date = doc.mets.date
    elif extracted_year is not None:
        record.year = extracted_year
        record.date = str(extracted_year)
    else:
        warnings.append("year_not_found")
        logger.warning("Could not determine year for %s", doc.doc_id)


def _extract_dominant_year(text: str) -> int | None:
    """Find the most frequent valid 4-digit year in the text."""
    year_counts: Counter = Counter()
    # Use only the specific date patterns first (more reliable than bare year)
    for pattern in DATE_PATTERNS[:-1]:
        for m in pattern.finditer(text):
            try:
                y = int(m.group(1))
                if NEPA_YEAR <= y <= MAX_YEAR:
                    year_counts[y] += 3  # weight specific date patterns higher
            except (IndexError, ValueError):
                pass

    # Bare year pattern as fallback weight
    for m in DATE_PATTERNS[-1].finditer(text):
        y = int(m.group(1))
        if NEPA_YEAR <= y <= MAX_YEAR:
            year_counts[y] += 1

    if not year_counts:
        return None
    return year_counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# 0.6  Lead agency
# ---------------------------------------------------------------------------

def _lead_agency(doc: Document, record: EISRecord, warnings: list[str]) -> None:
    # 1. METS is authoritative
    if doc.mets.agency:
        canonical = lookup_agency(doc.mets.agency)
        if canonical:
            abbr = _get_abbr(canonical)
            record.lead_agency = LeadAgency(name=canonical, abbreviation=abbr, source="mets")
            return
        else:
            # Use METS value as-is if it doesn't match our vocab
            record.lead_agency = LeadAgency(name=doc.mets.agency, abbreviation=None, source="mets")
            return

    # 2. Search first 3 pages
    first_3_text = " ".join(p.text for p in doc.pages[:3])

    # Exact match against all known variants
    for token in re.findall(r"[A-Z][A-Za-z\s&,.'-]{2,60}", first_3_text):
        canonical = lookup_agency(token.strip())
        if canonical:
            abbr = _get_abbr(canonical)
            record.lead_agency = LeadAgency(name=canonical, abbreviation=abbr, source="regex")
            return

    # Abbreviation match
    for token in re.findall(r"\b[A-Z]{2,6}\b", first_3_text):
        canonical = lookup_agency(token)
        if canonical:
            abbr = token
            record.lead_agency = LeadAgency(name=canonical, abbreviation=abbr, source="regex")
            return

    # Fuzzy match against all variant strings
    all_variants = list(AGENCY_FLAT.keys())
    # Try each sentence-like chunk from first 3 pages
    candidates = re.findall(r"[A-Z][A-Za-z\s&,.'-]{5,80}", first_3_text)
    for candidate in candidates[:50]:  # cap to avoid O(n²) on huge docs
        result = rfuzz_process.extractOne(
            candidate.lower(), all_variants, score_cutoff=85
        )
        if result:
            matched_variant, score, _ = result
            canonical = AGENCY_FLAT[matched_variant]
            abbr = _get_abbr(canonical)
            record.lead_agency = LeadAgency(name=canonical, abbreviation=abbr, source="fuzzy_match")
            logger.debug("Fuzzy agency match: %r → %r (score=%d)", candidate, canonical, score)
            return

    warnings.append("lead_agency_not_found")
    logger.warning("Could not identify lead agency for %s", doc.doc_id)
    record.lead_agency = LeadAgency(name=None, abbreviation=None, source="unknown")


def _get_abbr(canonical: str) -> str | None:
    data = AGENCY_VOCAB.get(canonical, {})
    abbrs = data.get("abbreviations", [])
    return abbrs[0] if abbrs else None


# ---------------------------------------------------------------------------
# 0.7  NER (spaCy)
# ---------------------------------------------------------------------------

_NLP = None  # lazy-loaded
_NLP_MODEL_NAME: str | None = None


def _get_nlp():
    """
    Load the preferred spaCy model. Prefers en_core_web_trf (transformer-based,
    higher recall on person names) and falls back to en_core_web_lg if the
    transformer model isn't installed.
    """
    global _NLP, _NLP_MODEL_NAME
    if _NLP is not None:
        return _NLP

    import spacy  # type: ignore
    last_error: Exception | None = None
    for model_name in SPACY_MODELS_PREFERENCE:
        try:
            _NLP = spacy.load(model_name)
            _NLP_MODEL_NAME = model_name
            logger.info("NER: loaded spaCy model %s", model_name)
            return _NLP
        except OSError as exc:
            last_error = exc
            logger.debug("spaCy model %s not available: %s", model_name, exc)

    raise RuntimeError(
        f"No spaCy model available. Tried: {SPACY_MODELS_PREFERENCE}. "
        f"Run: python -m spacy download en_core_web_trf  (preferred) "
        f"or:  python -m spacy download en_core_web_lg   (fallback)"
    )


def _ner(doc: Document, record: EISRecord) -> None:
    """
    Layered NER:
      1. spaCy PERSON  → filtered through full-name regex (rejects bare surnames)
      2. spaCy ORG     → kept as messy catch-all (Haiku triage cleans up later)
      3. Dict lookup   → agencies (existing AGENCY_VOCAB) + tribes + NGOs.
                          Dict matches bypass downstream triage.

    A Haiku gap-fill pass runs later in Stage 2 (key_people.py) for chunks
    where this initial pass produced suspiciously few entities.
    """
    text = doc.full_text
    sources: dict[str, str] = {}
    raw_count = 0

    # ---- Layer 1+2: spaCy ----
    spacy_people: list[str] = []
    spacy_orgs: list[str] = []
    try:
        nlp = _get_nlp()
        chunk_size = 100_000
        raw_people: list[str] = []
        raw_orgs: list[str] = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            spacy_doc = nlp(chunk)
            for ent in spacy_doc.ents:
                if ent.label_ == "PERSON":
                    raw_people.append(ent.text.strip())
                elif ent.label_ == "ORG":
                    raw_orgs.append(ent.text.strip())

        raw_count = len(raw_people) + len(raw_orgs)

        spacy_people = _dedup_names([p for p in raw_people if _is_full_name(p)])
        spacy_orgs = _dedup_names(raw_orgs)

        for p in spacy_people:
            sources[p] = "spacy_person"
        for o in spacy_orgs:
            sources[o] = "spacy_org"
    except RuntimeError as exc:
        logger.warning("NER spaCy layer skipped: %s — falling back to dictionary only", exc)

    # ---- Layer 3: dictionary lookups (full-text scan) ----
    # Agencies: scan AGENCY_FLAT keys. Done at the document level so we don't
    # depend on spaCy catching them.
    dict_agencies = _find_agencies(text)
    dict_tribes = [m.name for m in find_tribes(text)]
    dict_ngos = [m.name for m in find_ngos(text)]

    # Combine ORG-like entities: spaCy orgs + dict agencies + dict tribes + dict NGOs.
    # Dict entries take precedence (their canonical name overrides any spaCy variant).
    organizations: list[str] = []
    seen_orgs: set[str] = set()

    def _add_org(name: str, source: str) -> None:
        key = name.lower()
        if key in seen_orgs:
            # If we have a dict source for an org spaCy also caught, upgrade the source.
            if source.startswith("dict_") and not sources.get(name, "").startswith("dict_"):
                sources[name] = source
            return
        seen_orgs.add(key)
        organizations.append(name)
        sources[name] = source

    for ag in dict_agencies:
        _add_org(ag, "dict_agency")
    for tr in dict_tribes:
        _add_org(tr, "dict_tribe")
    for ng in dict_ngos:
        _add_org(ng, "dict_ngo")
    for org in spacy_orgs:
        # If spaCy returned a variant of an agency, collapse to canonical.
        canonical = lookup_agency(org)
        if canonical:
            _add_org(canonical, "dict_agency")
        else:
            _add_org(org, "spacy_org")

    deduped_count = len(spacy_people) + len(organizations)
    logger.info(
        "NER: model=%s raw=%d deduped=%d (people=%d orgs=%d; dict_tribes=%d dict_ngos=%d dict_agencies=%d)",
        _NLP_MODEL_NAME or "none",
        raw_count, deduped_count,
        len(spacy_people), len(organizations),
        len(dict_tribes), len(dict_ngos), len(dict_agencies),
    )

    record.ner = NERResult(
        people=spacy_people,
        organizations=organizations,
        sources=sources,
        raw_count_before_dedupe=raw_count,
        deduped_count=deduped_count,
    )


def _is_full_name(name: str) -> bool:
    """
    True if `name` looks like a real multi-token person name.

    A real name token must start uppercase AND contain at least one lowercase
    letter (so "UNITED STATES" is rejected even though both tokens are
    capitalized). Allows middle initials, name particles ("van", "de la"),
    and hyphenated names.
    """
    name = name.strip()
    if not name:
        return False
    tokens = name.split()
    if len(tokens) < 2 or len(tokens) > 5:
        return False

    real_name_tokens = []
    for t in tokens:
        if MIDDLE_INITIAL_PATTERN.match(t):
            continue
        if t.lower() in NAME_PARTICLES:
            continue
        if not NAME_TOKEN_PATTERN.match(t):
            return False
        real_name_tokens.append(t)

    return len(real_name_tokens) >= 2


def _dedup_names(names: list[str]) -> list[str]:
    """Case-insensitive dedup; preserves first occurrence's casing."""
    seen: dict[str, str] = {}
    for n in names:
        key = n.lower().strip()
        if key and key not in seen:
            seen[key] = n
    return list(seen.values())


def _find_agencies(text: str) -> list[str]:
    """
    Scan text for known agencies by variant/abbreviation. Returns canonical
    names in first-appearance order, deduplicated.
    """
    # Build a single alternation regex over all variants (case-insensitive).
    # AGENCY_FLAT maps variant->canonical; we look for any variant.
    variants = sorted(AGENCY_FLAT.keys(), key=len, reverse=True)
    if not variants:
        return []
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(v) for v in variants) + r")\b",
        re.IGNORECASE,
    )
    seen: dict[str, int] = {}
    for m in pattern.finditer(text):
        variant = m.group(0).lower()
        canonical = AGENCY_FLAT.get(variant)
        if canonical and canonical not in seen:
            seen[canonical] = m.start()
    # Return in first-appearance order
    return [name for name, _ in sorted(seen.items(), key=lambda kv: kv[1])]


# ---------------------------------------------------------------------------
# 0.8  Title
# ---------------------------------------------------------------------------

def _title(doc: Document, record: EISRecord) -> None:
    if doc.mets.title:
        record.title = doc.mets.title
    else:
        # Grab the first non-empty line as a fallback
        for line in doc.full_text.splitlines():
            stripped = line.strip()
            if len(stripped) > 10:
                record.title = stripped[:200]
                logger.info(
                    "Title not in METS for %s — using first text line: %r",
                    doc.doc_id, record.title,
                )
                break
