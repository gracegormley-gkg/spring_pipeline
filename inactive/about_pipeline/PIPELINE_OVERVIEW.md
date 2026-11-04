# EIS Metadata Extraction Pipeline — Overview

**Version:** v1.0
**Location:** `eis_pipeline/`
**Purpose:** Extract structured metadata from U.S. Environmental Impact Statement (EIS) documents using a combination of deterministic text analysis and Claude LLM calls. Produces one validated JSON record per document.

---

## What It Does

Each EIS document — a dense government PDF from the 1970s–90s — gets processed into a structured record covering:

| Field | What it captures |
|-------|-----------------|
| Summary (detailed) | ~180-word evidence-cited description of the project, community, need, and environmental impact |
| Summary (layman) | ~80–120-word plain-English rewrite of the detailed summary, for public audiences |
| Location | Primary geographic area + geocoded lat/lon |
| Themes | 1–2 primary themes + 2–5 subthemes from a controlled taxonomy |
| Alternatives | Each named alternative the document considered (incl. "Alignment"/"Variation"/"Option" terminology), with descriptions |
| Key People & Groups | Named stakeholders with role, opinion summary, stance, and verbatim quote — sorted by document appearance order |
| Historical Context | What the document itself says about the project's history and background |
| EIS Type | Draft / Final / Supplemental / Unlabelled |
| Lead Agency | The federal agency that produced the document |
| Date & Year | Publication date, cross-checked against catalog metadata |

Two fields — **external historical context** and **current project status** — are stubbed out and deferred to v2. They require a search API, domain allowlisting, and a stricter critic to avoid confidently wrong outputs.

---

## Input Formats

The pipeline accepts two input formats:

**1. S3 bucket format** (full fidelity — preferred)
```
P0491_<DOC_ID>/
├── TXT/<DOC_ID>_00000001.txt    # one file per page
├── CONFIDENCES/<DOC_ID>_00000001.json
├── mets.xml
└── mets.yaml
```
Provides per-page OCR confidence scores and authoritative METS catalog metadata.

**2. `docs_with_digits.json`** (flat text — current working format)
A single JSON file mapping accession keys (`P0491_35556036056489`) to full OCR text strings. No page boundaries or confidence scores. The pipeline splits the text into ~2500-character fake pages on line boundaries and fetches title/agency/date from the NUL Digital Collections API automatically.

---

## Architecture: Four Stages

```
Input (S3 folder or docs_with_digits.json)
        │
        ▼
  Stage 0 — Triage         (deterministic, no LLM)
        │
        ▼
  Stage 1 — Chunking       (Haiku: label each chunk)
        │
        ▼
  Stage 2 — Field Extraction  (Opus + Haiku: one module per field)
        │
        ▼
  Stage 3 — Critic         (Sonnet: verify evidence; deterministic checks)
        │
        ▼
  EISRecord (validated Pydantic JSON)
```

---

## Stage 0 — Triage (No LLM)

Deterministic extraction from raw text and METS metadata. No API calls.

- **OCR confidence** — computes median confidence across all pages; sets `unclear_document_flag` if below 0.8. Pages below 0.7 confidence are excluded from all downstream retrieval.
- **EIS type** — regex scan of the first 250 words for Draft/Final/Supplemental signals. If two families match (e.g. a doc that says "Final" but also references "the Draft"), marks as Unlabelled.
- **Word count & length category** — short (<10k words), medium, or long (>60k words). Short docs skip chunking and become one chunk.
- **Headings & TOC** — regex sweep for chapter/section patterns and a "Table of Contents" marker. Regex patterns use `[ \t]+` (not `\s+`) so they can't cross line boundaries, and a post-match `_is_real_heading` filter rejects bare legal citations ("Section 4(f)"), street addresses, ZIP codes, and OCR letterhead fragments. Regex-detected sections are a *starting point* — Stage 1's AI-TOC pass can replace them.
- **Date & year** — regex extraction of dates from the first 5 pages; falls back to full doc. Cross-checked against METS: if they disagree by more than 1 year, METS wins and a warning is logged.
- **Lead agency** — checked against a controlled vocabulary of 20 federal agencies with all known abbreviations and variants. Tries METS first (authoritative), then exact match in first 3 pages, then fuzzy match (score ≥ 85).
- **NER — layered**:
    1. **spaCy** (`en_core_web_trf`, falls back to `_lg`) extracts PERSON and ORG entities. PERSON output is filtered through a full-name regex that rejects single tokens ("Smith"), all-caps strings ("UNITED STATES"), and lowercase fragments. ORG output is kept as a permissive catch-all (Haiku triage cleans it up downstream).
    2. **Dictionary lookups** scan the full text for known entities: ~20 federal agencies (`AGENCY_VOCAB`), ~190 federally recognized tribes (`ner_dicts.TRIBES`), and ~50 environmental NGOs (`ner_dicts.ENV_NGOS`). Dictionary matches bypass downstream Haiku triage because their identity as stakeholders is established by list membership.
    3. **Provenance tracking**: `NERResult.sources` records where each entity came from (`spacy_person`, `spacy_org`, `dict_agency`, `dict_tribe`, `dict_ngo`, `haiku_gapfill`).
  Skips spaCy layer gracefully if no spaCy model is installed.
- **Title** — from METS if available; falls back to first non-empty text line.

---

## Stage 1 — Chunking

Splits the document into chunks and labels each one with a title, description, and topic tags. Chunk quality matters — every Stage 2 field uses tag-based retrieval to find relevant chunks, so a mislabeled chunk can cause a field to miss its content (mitigated by retrieval fallbacks; see Stage 2).

**Section discovery, in order of preference:**
1. **AI-TOC pass (Haiku)** — Reads three samples of the document (beginning, middle, end — ~15k chars total) and returns major structural sections as `{title, anchor_phrase}`. Each anchor phrase is searched in the full document (exact → case-insensitive → whitespace-normalized regex that handles OCR line breaks) to map sections to char offsets and page numbers. If ≥3 locatable sections are found, these replace the Stage 0 regex sections. This is what salvages section structure in typewritten 1970s docs where formal headings are absent.
2. **Stage 0 regex headings** — Used if AI-TOC finds <3 sections. Good for typeset, well-formatted modern docs.
3. **Fixed 30-page chunks** — Fallback for medium/long docs with no detectable sections via either method.
4. **Single chunk** — Short docs (<10k words) skip chunking entirely.

**Labeling (Haiku):** For each chunk, sends the text (up to ~32k characters) and asks for:
- A 1-line title
- A 2–3 sentence factual description
- 1–4 topic tags from a fixed 11-item vocabulary: `purpose_and_need`, `proposed_action`, `affected_environment`, `alternatives`, `mitigation`, `consultation`, `cumulative_impacts`, `comments_and_responses`, `appendix`, `references`, `other`

Haiku is used here (not Opus) because this is a classification task with a fixed vocabulary — cheap and fast, run once per chunk.

---

## Stage 2 — Field Extraction

Each field is its own module: retrieval → prompt → parse. All LLM outputs include evidence pointers (chunk ID + page numbers) for the critic to verify.

### Summary (Opus, 2 calls)
**Pass 1 — Detailed:** Retrieves top 5–8 chunks tagged `purpose_and_need`, `proposed_action`, or `affected_environment`. Asks for a ~180-word evidence-grounded summary covering: (a) community impacted, (b) project goal, (c) why it was needed, (d) environmental impacts. Every claim must be cited to a chunk ID. Stored as `summary.text`.

**Pass 2 — Layman:** Takes the detailed summary text as input (does NOT re-read the document) and rewrites in plain English at high-school reading level: 80–120 words, no chunk citations, replaces bureaucratic terms ("right-of-way" → "land taken for the road"). Stored as `summary.layman_text`. Inherits the factual grounding of the detailed pass — no new facts.

**Retrieval fallback:** If no chunks match the target tags (e.g. the Stage 1 labeler missed them), falls back to keyword search (`purpose`, `proposed`, `affected environment`, etc.), then to first N usable chunks. The summary field cannot go silent due to a chunk-tagging error.

### Themes (Opus)
Uses only the summary text + title (no raw chunks). Classifies into 1–2 primary themes and 2–5 subthemes from a controlled taxonomy of 12 primary themes. The prompt explicitly warns against overusing `other` — a >30% `other` rate signals a vocabulary gap in the taxonomy itself.

**Current theme taxonomy:**
`energy_infrastructure`, `transportation`, `land_management`, `water_resources`, `defense_and_military`, `urban_development`, `mining_and_extraction`, `agriculture_and_forestry`, `wildlife_and_habitat`, `cultural_heritage`, `waste_and_remediation`, `other`

### Location (Haiku)
Sends title + up to 4 chunks tagged `affected_environment` or `purpose_and_need`. Extracts a place name and 2-letter state. The place name is then geocoded via **Nominatim (OpenStreetMap)** — no LLM involved in geocoding. Returns null lat/lon if geocoding fails.

### Alternatives (Opus)
Retrieves chunks tagged `alternatives`. Extracts each named alternative with a 1–2 sentence description and chunk citation. The prompt explicitly recognizes the varied terminology EIS docs use — "Alternative", "No Action", "Alignment", "Variation", "Option", "Route", "Corridor", "Plan", "Preferred Alternative", "Build / No Build" — because individual documents pick one of these conventions and stick with it.

**Retrieval fallback:** If no `alternatives`-tagged chunks exist, falls back to keyword search (`alternative`, `no action`, `preferred`, etc.). Context limit raised to 100k chars so the end of large multi-tag chunks isn't truncated (the recommended-alternative summary often lives there).

### Key People & Groups (Opus + Haiku)

This is the most complex field. Runs in substeps:

1. **Gap-fill (Haiku, up to 6 calls per doc)** — Before filtering, runs Haiku on stakeholder-dense chunks to catch entities spaCy + dictionaries missed. Triggers: chunks tagged `comments_and_responses` (unconditional, these are stakeholder-dense by definition) + chunks with >2,000 words and <3 already-known entities (conditional). New entities are added with `source: haiku_gapfill` and bypass triage.
2. **Filter (rule-based)** — Drops short / lowercase / boilerplate names. **Trusted sources (dict_*, haiku_gapfill) skip this step** — they're already vetted.
3. **Score & cap** — Sort by `frequency × chunk_spread`, take top `MAX_KEY_PEOPLE` (50).
4. **Triage (Haiku, one batch call)** — Scrubs *only spaCy-sourced* entities (dict + gap-fill bypass). Asks which are genuine stakeholders vs. noise.
5. **Entity pack (Opus, ONE call per entity)** — Returns stance + role + opinion_summary + evidence + quote candidate in a single call. The quote is verified by deterministic substring check; if it fails, the quote is nulled (no retry call). Person and organization prompts differ — `role` means "title/position" for a person, "role in the project" for a group.

Output is sorted by document appearance order (first chunk where the entity is mentioned), with `appearance_order` numbered 1..N in that order.

### Historical Context — Internal (Opus)
Retrieves chunks tagged `purpose_and_need`/`affected_environment` plus any chunks containing keywords like "history", "background", "previously", "prior to". Every claim must be phrased as "The document states that..." or "According to the document..." — this forces a reporting register and makes the critic's job tractable. Returns claim-by-claim source mapping, not a single paragraph.

### Historical Context — External (v2 stub)
Returns `no_external_context_available`. Requires a search API, Tier 1 domain allowlist, and a stricter critic to avoid producing confidently wrong historical claims about real-world project outcomes. Deferred to v2.

### Project Current Status (v2 stub)
Returns `unknown`. Requires disambiguation logic to avoid false matches (many EIS projects share common place names). Deferred to v2.

---

## Stage 3 — Critic

Runs after Stage 2. Two layers of verification:

**Deterministic checks (no LLM):**
- **Quotes** — hard substring check against cited chunk. Fails → quote is nulled out
- **Year** — must be between 1969 (NEPA enacted) and current year
- **Themes** — all values must be in the controlled vocabulary
- **Geocoding** — logs a warning if place name was extracted but geocoding returned no coordinates

**LLM critic (Sonnet):**
For summary and historical context: sends each (claim, cited chunk) pair and asks for a `yes`/`no`/`partial` verdict with one sentence of justification. If any verdict is `no`, the field's status is downgraded to `insufficient_information`. Sonnet is used here (not Opus) — the task is narrow and well-defined, and Sonnet is cheaper.

---

## Output Schema

One JSON file per document. All fields always present; missing data uses `null`, `"unknown"`, or `"insufficient_information"`.

```json
{
  "doc_id": "35556036056489",
  "project_id": "P0491",
  "title": "...",
  "ocr": { "median_confidence": 0.91, "page_count": 412, "unclear_document_flag": false },
  "eis_type": "Draft | Final | Supplemental | Unlabelled",
  "length_category": "short | medium | long",
  "word_count": 184213,
  "has_headings": true,
  "has_toc": true,
  "sections": [{ "title": "Purpose and Need", "start_page": 12, "end_page": 34 }],
  "lead_agency": { "name": "Bureau of Land Management", "abbreviation": "BLM", "source": "mets" },
  "date": "1972-03-16",
  "year": 1972,
  "location": { "name": "Kennedy Space Center, Florida", "state": "FL", "latitude": 28.52, "longitude": -80.68, "geocode_source": "geopy_nominatim" },
  "themes": { "primary": ["energy_infrastructure"], "subthemes": ["nuclear_power"] },
  "summary": { "text": "...", "layman_text": "...", "evidence": [{ "chunk_id": "c01", "pages": [12, 13] }], "status": "populated" },
  "alternatives_proposed": [{ "name": "No Action", "description": "...", "evidence": [...] }],
  "key_people_and_groups": [{
    "name": "Cook County Forest Preserve District",
    "type": "organization",
    "role": "consulted agency",
    "opinion_summary": "Objected to characterizing the 3 acres of land required for the highway improvement as being of 'little or no value'...",
    "stance": "mixed",
    "first_appearance_chunk": "c03",
    "appearance_order": 6,
    "quote": {...}
  }],
  "historical_context_internal": { "text": "...", "claims": [...], "status": "populated" },
  "historical_context_external": { "deferred_to_v2": true },
  "project_current_status": { "value": "unknown", "deferred_to_v2": true },
  "ner": {
    "people": [...],
    "organizations": [...],
    "sources": { "Cook County Forest Preserve District": "haiku_gapfill", "Sierra Club": "dict_ngo", ... },
    "raw_count_before_dedupe": 412,
    "deduped_count": 287
  },
  "chunks": [{ "chunk_id": "c01", "title": "...", "description": "...", "topic_tags": [...], "pages": [...], "median_confidence": 0.93, "used": true }],
  "pipeline_metadata": { "version": "v1.0", "run_timestamp": "...", "models_used": {...}, "total_tokens": 32733, "total_cost_usd": 0.39, "warnings": [] }
}
```

A persistent **token ledger** (`output/token_ledger.json`) accumulates per-run token usage across all runs — input/output tokens, per-model breakdown, cost, duration. Intended for environmental-impact accounting and cost regression tracking across the full collection.

---

## How to Run

```bash
# Install dependencies
cd eis_pipeline/
pip install -r requirements.txt
python -m spacy download en_core_web_trf   # preferred (transformer, higher recall)
python -m spacy download en_core_web_lg    # fallback (auto-used if trf missing)

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run on a single document from docs_with_digits.json
python run.py \
  --json-file "/path/to/docs_with_digits.json" \
  --doc-key "P0491_35556036056489" \
  --output output/result.json \
  --budget-usd 4.00

# Run on a full S3-format document folder
python run.py \
  --doc-dir ./data/P0491_35556036056489 \
  --output output/result.json

# Useful flags
--dry-run                              # log prompts, skip API calls
--skip-stages 2,3                      # run only deterministic stages
--only-fields summary,themes           # run a subset of Stage 2 fields
--budget-usd 5.00                      # hard cost cap
--token-ledger output/ledger.json      # override ledger path (default: output/token_ledger.json)
```

**Before running on S3 data for the first time:**
```bash
python inspect_layout.py --doc-dir /path/to/P0491_<ID>
```
This prints the actual METS and confidence JSON schemas so you can verify the parser handles them correctly before running the full pipeline.

---

## Models Used

| Role | Model | Used for |
|------|-------|----------|
| Heavy | `claude-opus-4-6` | Summary, themes, alternatives, key people stance/quotes, historical context |
| Light | `claude-haiku-4-5-20251001` | Chunk labeling, location extraction, entity triage |
| Critic | `claude-sonnet-4-6` | Per-claim evidence verification |

---

## Known Limitations (v1)

- **No spaCy = degraded NER.** Without `en_core_web_trf` or `_lg`, the dict-lookup layer (agencies/tribes/NGOs) still works, but spaCy-derived PERSON/ORG output is empty — most key-people will be missed.
- **Chunk tagging is still a soft dependency.** Retrieval fallbacks (keyword search → first N chunks) prevent fields from going completely silent when chunks are mislabeled, but a well-tagged chunk gives much better context. The AI-TOC pass also reduces dependency on chunk tags by producing more semantically meaningful section boundaries.
- **Theme vocabulary has gaps.** Documents that don't fit the 12 themes (e.g. space programs, scientific research) will land on `other`. Audit `other` outputs after the first batch and expand the vocab.
- **Fake pages from `docs_with_digits.json`** mean page citations in evidence pointers are approximations, not ground truth. Page numbers become reliable once running against the full S3 data.
- **External historical context and current status are stubs.** These are the fields most likely to produce confidently wrong outputs if implemented naively — deferred to v2 by design.
- **Geocoding match rate ~60%.** Nominatim struggles with corridor descriptions ("Highway 99 between Eugene and Roseburg") and historical place names. Null lat/lon is the correct fallback.
- **NER raw output is noisy.** spaCy ORG catches a long tail of OCR fragments, line-break-split variants ("Cook County Forest\nPreserve District"), and truncations. The Haiku triage cleans this up before the final key-people output, but the raw `ner.organizations` list will contain garbage. A dedupe-normalization pass (strip leading "the", collapse newlines, fuzzy-merge variants) would help — currently not implemented.
- **Costs scale with stakeholder count.** Each entity that survives triage costs one Opus call. Docs with many real stakeholders (e.g. busy consultation appendices) are more expensive. Budget $3–5 per medium doc.
