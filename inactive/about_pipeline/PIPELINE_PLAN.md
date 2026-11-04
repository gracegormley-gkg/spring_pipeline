# EIS Metadata Pipeline — Implementation Plan (v1 prototype)

A working prototype of the metadata-extraction pipeline described in `Pipeline.pdf`. Input is a single EIS document directory laid out as in the Impulse S3 bucket; output is a single JSON metadata record covering every field the PDF lists.

This document is the spec to hand to Claude Code. It is opinionated about scope and model choice — see the **Scope and pushback** section for the decisions worth challenging before implementation starts.

---

## 1. Input layout (per document)

Each document lives in a folder under the bucket root. From the screenshot, the layout is:

```
P0491_<DOC_ID>/
├── TXT/
│   ├── <DOC_ID>_00000001.txt        # plaintext, one file per page
│   ├── <DOC_ID>_00000001.json       # per-page token data (likely word boxes/confidence)
│   ├── <DOC_ID>_00000002.txt
│   ├── <DOC_ID>_00000002.json
│   └── ...
├── CONFIDENCES/
│   ├── <DOC_ID>_00000001.json       # per-page confidence scores
│   └── ...
├── mets.xml                          # METS metadata (authoritative for title/agency/date)
└── mets.yaml                         # same data in YAML
```

**Open question for first run:** the exact schemas of the per-page JSON files and the METS files are not in the project context. The implementation should:
- Inspect a real example with a small `inspect_layout.py` utility before hardcoding parsing rules
- Parse defensively (try common field names, log what's actually present, fail loud on schema surprises rather than silently producing wrong data)

`DOC_ID` is the Mongo serial used to cross-reference NUL records. `P0491` is the project/collection ID.

---

## 2. Output schema

One JSON record per document. All fields are always present; missing data is represented as `null`, `"unknown"`, or `"insufficient_information"` depending on the field (see notes below). Every LLM-generated claim carries evidence pointers (chunk IDs + page numbers).

```json
{
  "doc_id": "35556036063543",
  "project_id": "P0491",
  "title": "...",                                          // from METS, fallback to LLM
  "ocr": {
    "median_confidence": 0.91,
    "page_count": 412,
    "unclear_document_flag": false                          // true if median < 0.8
  },
  "eis_type": "Draft" | "Final" | "Supplemental" | "Unlabelled",
  "length_category": "short" | "medium" | "long",
  "word_count": 184213,
  "has_headings": true,
  "has_toc": true,
  "sections": [                                             // empty list if has_headings=false
    {"title": "Purpose and Need", "start_page": 12, "end_page": 34}
  ],
  "lead_agency": {
    "name": "Bureau of Land Management",
    "abbreviation": "BLM",
    "source": "mets" | "regex" | "fuzzy_match" | "nul"
  },
  "date": "1998-03-15",                                     // ISO; null if not found
  "year": 1998,
  "location": {
    "name": "Cedar Creek, Wyoming",
    "state": "WY",
    "latitude": 42.1234,
    "longitude": -106.789,
    "geocode_source": "geopy_nominatim"
  },
  "themes": {
    "primary": ["energy_infrastructure"],
    "subthemes": ["pipeline", "public_lands", "tribal_consultation"]
  },
  "summary": {
    "text": "...",
    "evidence": [{"chunk_id": "c03", "pages": [12,13]}, ...]
  },
  "alternatives_proposed": [
    {"name": "No Action", "description": "...", "evidence": [...]},
    {"name": "Alternative B — Northern Route", "description": "...", "evidence": [...]}
  ],
  "key_people_and_groups": [
    {
      "name": "Sierra Club",
      "type": "organization",
      "first_appearance_chunk": "c02",
      "appearance_order": 1,
      "stance": "opposed",
      "stance_evidence": [{"chunk_id": "c05", "pages": [88]}],
      "quote": {
        "text": "...",
        "chunk_id": "c05",
        "page": 88,
        "substring_verified": true
      }
    }
  ],
  "historical_context_internal": {
    "text": "The document says the project is needed because...",
    "claims": [
      {"sentence": "...", "evidence": [{"chunk_id": "c01", "pages": [3]}]}
    ],
    "status": "populated" | "insufficient_information"
  },
  "historical_context_external": {
    "text": null,
    "sources": [],
    "tier": null,
    "status": "no_external_context_available",             // v2 — see below
    "deferred_to_v2": true
  },
  "project_current_status": {
    "value": "unknown",                                     // v2 — see below
    "source_url": null,
    "source_passage": null,
    "source_tier": null,
    "evidence_date": null,
    "as_of_date": null,
    "disambiguation_checks_passed": [],
    "deferred_to_v2": true
  },
  "ner": {
    "people": [...],
    "organizations": [...],
    "raw_count_before_dedupe": 412,
    "deduped_count": 287
  },
  "chunks": [
    {
      "chunk_id": "c01",
      "title": "Purpose and Need",
      "description": "Three-sentence chunk description from Haiku",
      "pages": [12, 13, 14],
      "median_confidence": 0.93,
      "used": true                                          // false if confidence below 0.7
    }
  ],
  "pipeline_metadata": {
    "version": "v1.0",
    "run_timestamp": "...",
    "models_used": {"summary": "claude-opus-4-7", "critic": "claude-haiku-4-5", ...},
    "total_tokens": 412334,
    "total_cost_usd": 1.42,
    "warnings": []
  }
}
```

Use Pydantic models for this schema. Validate before writing.

---

## 3. Project layout

```
eis_pipeline/
├── README.md
├── requirements.txt
├── pyproject.toml
├── run.py                          # CLI entrypoint: one doc → one JSON
├── inspect_layout.py               # utility: dump structure of a real doc folder
├── fetch_data.py                   # S3 downloader (run separately)
├── pipeline/
│   ├── __init__.py
│   ├── config.py                   # agency vocab, theme list, regex patterns, model IDs
│   ├── schema.py                   # Pydantic models for the metadata record
│   ├── io_layer.py                 # read METS, TXT, CONFIDENCES; concatenate; produce Document
│   ├── stage0_triage.py            # deterministic extraction
│   ├── stage1_chunking.py          # chunking + LLM chunk labeling
│   ├── stage2_fields/              # one file per field, all share retrieval helpers
│   │   ├── __init__.py
│   │   ├── retrieval.py            # chunk retrieval / reranking helpers
│   │   ├── summary.py
│   │   ├── themes.py
│   │   ├── location.py             # also calls geopy
│   │   ├── alternatives.py
│   │   ├── key_people.py           # NER filter + stance + ordering + quotes
│   │   ├── historical_internal.py
│   │   ├── historical_external.py  # v1 stub, v2 implementation
│   │   └── current_status.py       # v1 stub, v2 implementation
│   ├── stage3_critic.py            # per-field critics + deterministic checks
│   └── llm_client.py               # Anthropic API wrapper + dry-run mode + retry/backoff
├── tests/
│   ├── test_stage0.py
│   ├── test_io_layer.py
│   └── fixtures/                   # tiny synthetic docs for unit tests
└── output/
```

---

## 4. Stage 0 — Triage (deterministic, no LLM)

### 4.1 Load and concatenate
- Read all `TXT/*.txt` files in page order (sort by the numeric suffix, not lexicographically — `_00000010` must sort after `_00000009`)
- Concatenate with `\f` (form feed) page separators so downstream code can recover page boundaries
- Build a parallel page-number ↔ char-offset index so any later character position can be mapped back to a page

### 4.2 OCR confidence
- For each page, read the corresponding `CONFIDENCES/<DOC_ID>_<PAGE>.json`
- The exact schema isn't known yet — `inspect_layout.py` should print one out before hardcoding. Common shapes: per-word `{word, confidence}` lists, or a single `median_confidence` per page
- Compute:
  - `median_confidence` = median across all word confidences in the doc (or median of per-page medians if that's all that's available)
  - Per-chunk medians (computed later in Stage 1)
- If `median_confidence < 0.8` → set `ocr.unclear_document_flag = true` and add a warning. Do not abort — still run the pipeline, but the flag propagates to consumers
- Chunks with `median_confidence < 0.7` get `used: false` and are excluded from retrieval

### 4.3 EIS type
- Look at the first 250 words of concatenated text (case-insensitive)
- Patterns: `r"\bdraft\b"`, `r"\bfinal\b"`, `r"\bsupplemental\b"`, `r"\bDEIS\b"`, `r"\bFEIS\b"`, `r"\bSEIS\b"`
- Rule: if exactly one family matches → that's the type. If none or multiple conflicting families → `"Unlabelled"`. Log which pattern matched

### 4.4 Headings and TOC
- Regex sweep for typical TOC patterns: `r"^\s*(?:CHAPTER|SECTION|APPENDIX)\s+[IVXLC\d]+"` (multiline), numbered headings like `r"^\s*\d+(?:\.\d+)*\s+[A-Z][A-Z\s]{4,}"`, and explicit TOC markers (`r"^\s*Table of Contents\s*$"`)
- `has_toc = true` if a TOC marker is found within the first ~10% of the document
- `has_headings = true` if ≥ 5 distinct heading-like lines are found across the full doc
- If headings found: extract them with their start positions, then map to page numbers using the page index from 4.1. This populates `sections`

### 4.5 Date and year
- Regex: `r"\b(?:Jan|Feb|Mar|...|January|February|...)\s+\d{1,2},?\s+\d{4}\b"`, `r"\b\d{1,2}/\d{1,2}/\d{4}\b"`, `r"\b\d{4}-\d{2}-\d{2}\b"`
- Filter: 1969 ≤ year ≤ current year (NEPA was passed in 1969 — anything earlier is OCR noise or a citation, not the EIS date)
- Strategy: take the most frequent valid year in the first 5 pages as the document year. If nothing in first 5 pages, fall back to most frequent across the whole doc
- Cross-check against METS date if present — if METS and regex disagree by >1 year, log a warning and prefer METS

### 4.6 Lead agency
- Maintain a controlled vocabulary in `config.py`: dict mapping canonical name → list of variants and abbreviations (BLM, USFS, USACE, EPA, NPS, FWS, NASA, DOE, DOT, FERC, BOEM, BIA, BOR, etc.)
- First check METS (if it has the agency, use it — that's authoritative)
- Else search the first 3 pages with both exact matching and `rapidfuzz.process.extractOne` for fuzzy hits with score ≥ 85
- Record `source` field showing which path produced it

### 4.7 NER
- Use spaCy `en_core_web_lg` (not `_sm` — too many misses on agency/group names)
- Extract `PERSON` and `ORG` entities across the full document
- Dedupe with case-insensitive normalization + the agency vocab (so "Bureau of Land Mgmt" and "BLM" collapse to one)
- Store raw counts and deduped counts for diagnostics
- Don't filter to relevance here — that happens in Stage 2's key-people extractor

### 4.8 Length
- `word_count = len(text.split())`
- Categories: short < ~10k words (roughly ≤30 pages), medium 10k–60k, long > 60k. Tune after seeing real distributions
- Short docs skip chunking in Stage 1

### 4.9 NUL cross-check
- If a NUL/Mongo lookup endpoint is wired in (presumably by `DOC_ID`), pull title, bureau, year from there and use as the authoritative source, overriding regex extraction
- If not available in v1, log it as a TODO and continue with METS + regex. Don't fake this

---

## 5. Stage 1 — Chunking

### Short docs (≤ ~30 pages)
- One chunk = whole doc
- Still run the chunk labeler so the chunk has a `title` and `description` for downstream uniformity

### Medium/long docs
- **If `has_headings`:** split along detected headings. Each chunk = one section. The section title is the chunk title
- **If not:** fixed 30-page chunks with no overlap initially. Add overlap later only if retrieval shows boundary-loss problems

### Chunk labeling (LLM)
- For every chunk, call **Haiku 4.5** with the chunk text (truncated to ~8k tokens if huge) and ask for:
  - A 1-line title (use the heading if available, else generate)
  - A 2–3 sentence factual description ("This chunk covers X, Y, and Z")
  - A list of likely topic tags from a fixed vocab: `["purpose_and_need", "affected_environment", "proposed_action", "alternatives", "mitigation", "consultation", "cumulative_impacts", "comments_and_responses", "appendix", "references", "other"]`
- Compute median OCR confidence per chunk
- Chunks with median confidence < 0.7 are marked `used: false` and excluded from all retrieval

---

## 6. Stage 2 — Per-field extraction

Each field is its own pipeline: retrieval → prompt → critic. All outputs must include evidence pointers.

### Model choice (revised from PDF, which predates Opus 4.7)
- **Heavy synthesis fields** (summary, historical context, themes, key people analysis, alternatives): **Opus 4.7** — `claude-opus-4-7`. The PDF says "Sonnet" but Opus 4.7 is now the strongest model and well worth the cost for these high-stakes fields
- **Light fields** (chunk labeling, location-from-title, simple critic checks): **Haiku 4.5** — `claude-haiku-4-5-20251001`
- **Critic for complex fields**: Sonnet 4.6 — `claude-sonnet-4-6`. Cheaper than running Opus twice, strong enough to catch claim/evidence mismatches

### 6.1 Summary
- **Retrieval:** rerank chunks by topic tags (`purpose_and_need`, `proposed_action`, `affected_environment`); take top 5–8
- **Prompt:** require coverage of (a) community impacted, (b) final goal of the project, (c) why the project is needed, (d) anticipated environmental impact. Cite chunk IDs for every claim. Use only the provided chunk text
- **Output:** prose summary + evidence list

### 6.2 Themes
- Controlled vocabulary in `config.py`. Suggested top-level themes: `energy_infrastructure`, `transportation`, `land_management`, `water_resources`, `defense_and_military`, `urban_development`, `mining_and_extraction`, `agriculture_and_forestry`, `wildlife_and_habitat`, `cultural_heritage`, `waste_and_remediation`, `other`. Each with 4–8 subthemes
- Assign 1–2 primary themes, 2–5 subthemes
- **Important:** include `"other"` as a valid theme. If the model is forced to pick from a closed list when nothing fits, it will pick the closest match and the data will be wrong. Audit `"other"` outputs after the first 20 docs and expand the vocab

### 6.3 Location
- **Haiku** with title + first chunk + any chunks tagged `affected_environment`
- Output: place name + state. Then run through `geopy.geocoders.Nominatim` (with a polite user-agent and 1s rate limit) to get lat/lon
- If geocoding fails or returns multiple plausible matches, set `latitude`/`longitude` to null and log

### 6.4 Alternatives
- **Retrieval:** chunks tagged `alternatives`
- **Prompt:** extract each named alternative (almost always includes "No Action") with a 1–2 sentence description. Cite chunk IDs
- Output: list of `{name, description, evidence}`

### 6.5 Key people/groups + opinions + quotes + order
This is the most complex field. Break it into substeps:

1. **Start from NER list** from Stage 0
2. **Filter:** drop entries that are clearly not relevant — single-name people (just "Smith"), entries appearing only once, document-author boilerplate signatures, citation authors. Use a rule-based first pass (frequency ≥ 2, full name for people) then a Haiku call to triage borderline cases
3. **Cap at top 30** by frequency × cross-chunk spread
4. **Stance analysis:** for each entry, retrieve all chunks mentioning it. Ask Opus: based on the language attributed to or about this entity, is their stance toward the project `supportive`, `opposed`, `mixed`, `neutral`, or `insufficient_information`? Require evidence chunk IDs
5. **Ordering:** record the chunk_id of first appearance and rank by document order
6. **Quotes:** for each entity, ask Opus for a single quote that either (a) encapsulates their stance or (b) uses emotionally charged language. **Hard requirement: the quote must be a verbatim substring of the chunk text.** After generation, run a deterministic substring check; if it fails, retry once with stricter prompting, then null the quote and flag

The PDF specifically calls out that earlier models extracted hollow boilerplate like "we sincerely hope you consider our letter" because "sincerely" is emotive. Counter this with negative examples in the prompt and a length floor (e.g. ≥ 8 words, ≤ 40 words).

### 6.6 Historical context — internal
- **Retrieval:** chunks tagged `purpose_and_need`, `affected_environment`, intro chunks of `proposed_action`; keyword search across all chunks for "history", "background", "previously", "prior to", "established in", "originally"; date references to past events
- Pull 15–20 chunks, rerank to top 8
- **Prompt:** extract historical context from the document only. Every claim needs a chunk ID + page number. Phrase the output as "The document states that..." or "According to the document..."
- Output: claim-by-claim source mapping (not one big paragraph)
- Default to `insufficient_information` if retrieval is empty

### 6.7 Historical context — external (v2 — stub in v1)
The PDF specifies an extensive plan for this (Tier 1 allowlist, strict critic, default to no-context, oversample human review). In v1, **scaffold the interface and return `no_external_context_available`** unless a search API key is wired in.

Reasoning for deferring: this field needs a Tier 1 allowlisted search (Bing / Brave / Google CSE), domain filtering, fetch + clean pipeline, and a stricter critic with auto-fail. That's a meaningful build by itself. Ship internal extraction first, prove it works, then add external in a second pass with proper calibration. **Push back if you disagree** — but the PDF's own roadmap puts this in v2.

### 6.8 Project current status (v2 — stub in v1)
Same reasoning as 6.7. The PDF explicitly calls out the disambiguation problem ("Cedar Creek" is a common name) as the biggest failure mode. Building this without the disambiguation check is worse than not building it at all. Stub returns `"unknown"` with `deferred_to_v2: true`.

---

## 7. Stage 3 — Critic

Run after Stage 2. Each field gets a critic appropriate to its risk profile.

### Standard critic (summary, themes, location, alternatives, historical internal)
- For each evidence pointer, fetch the cited chunk
- Ask Sonnet 4.6: "Does this claim appear in the cited chunk? Answer yes/no/partial with one sentence of justification"
- Aggregate: if any claim returns `no`, downgrade `status` to `insufficient_information` and log

### Deterministic checks (no LLM)
- **Quotes:** verbatim substring check against the cited chunk. Hard pass/fail
- **Year:** must be 1969 ≤ year ≤ current year; must agree with METS within 1 year if METS has it
- **Location:** geocoding must succeed for lat/lon to be populated
- **Themes:** all themes/subthemes must be in the controlled vocabulary

### Strict critic (historical external, current status — for v2)
- Per-sentence source URL required
- Cited URL must be in the retrieval log (no inventing sources)
- Citation domain must be on the Tier 1 allowlist
- Substring + semantic match between claim and source passage
- Any failure → auto-fail the whole field, retry once with stricter prompt, then null out and flag for human review

---

## 8. Scope and pushback (decisions worth challenging)

These are the choices I'm making that you should sanity-check before implementation:

1. **Defer external historical context and current status to v2.** The PDF's own roadmap does this. They need infrastructure (search API, allowlist, disambiguation, stricter critic) that doubles the build size. Ship internal first.

2. **Use Opus 4.7 for heavy fields, not Sonnet as the PDF says.** The PDF predates Opus 4.7. For high-stakes fields where evidence-pointer fidelity matters, Opus is worth the cost. If budget is tight, Sonnet 4.6 is a fine fallback — make it a config flag.

3. **Read METS first, fall back to extraction.** The PDF treats NUL/METS as a cross-check. I think it should be authoritative for title, agency, and date when present, with regex extraction as the fallback. METS exists for a reason.

4. **`inspect_layout.py` before hardcoding.** Don't assume the per-page JSON and METS schemas — inspect a real doc first. The screenshot shows the directory structure but not the file contents.

5. **No NUL Mongo lookup in v1 unless an endpoint is provided.** The PDF mentions it; if there's no API to call, METS is the practical substitute.

6. **No human-review queue in v1.** Out of scope for a prototype. Outputs include a `warnings` list that surfaces what would have been flagged.

7. **Single-document scope per `run.py` invocation.** Parallelism is a job-queue concern that sits around this script. Don't bake it in.

8. **Push back especially on point 1** — if you'd rather ship a fragile end-to-end v1 with all fields than a solid v1 with two fields stubbed, that's a defensible choice. I'd just want to be clear that the external-data fields are the field most likely to produce confidently-wrong outputs.

---

## 9. How to run

```bash
# Install
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Look at a real doc folder before assuming schemas
python inspect_layout.py --doc-dir /path/to/P0491_35556036063543

# Fetch from S3 (run separately; needs AWS creds)
python fetch_data.py --bucket nu-impulse-production --doc-id 35556036063543 --output-dir ./data

# Deterministic stages only — no API key needed; good for debugging Stage 0
export ANTHROPIC_API_KEY=""
python run.py --doc-dir ./data/P0491_35556036063543 --output ./output/result.json --dry-run

# Full run
export ANTHROPIC_API_KEY=sk-ant-...
python run.py --doc-dir ./data/P0491_35556036063543 --output ./output/result.json
```

Flags worth supporting on `run.py`:
- `--dry-run` — skip LLM calls, log prompts that would have been sent
- `--skip-stages 2,3` — run only deterministic stages
- `--only-fields summary,themes` — extract a subset (useful when iterating on one prompt)
- `--budget-usd 5.00` — abort if total cost projection exceeds this

---

## 10. What to verify after the first real run

In order of importance:

1. **Page index correctness.** Does a known passage on page 50 actually resolve back to page 50? If not, the form-feed concatenation is wrong and every downstream page citation is junk
2. **OCR confidence pipeline.** Does `unclear_document_flag` trip on the docs you expect it to? Are chunks with bad OCR actually excluded?
3. **Chunk descriptions.** Spot-check 10 chunks: do the descriptions match the content? If not, fix Stage 1 before doing anything else — every downstream retrieval depends on these
4. **Evidence pointers.** Pick 5 summary claims at random and verify they appear in the cited chunks. If they don't, the critic isn't catching hallucination
5. **Quote substring check.** Should be 100% pass after the deterministic verifier. If it isn't, there's a bug in the verifier
6. **NER over-extraction.** Expect spaCy to surface 100+ "people" most of which are noise. The Stage 2 filter has to be aggressive
7. **Theme `"other"` rate.** If >30% of docs end up with `"other"` as primary, the vocab is wrong
8. **Cost per document.** Track this from run 1. If a single doc costs >$5 something is wrong (probably feeding whole chunks where reranked retrieval should be working)

---

## 11. Dependencies

```
anthropic>=0.40.0
pydantic>=2.0
spacy>=3.7
rapidfuzz>=3.6
geopy>=2.4
lxml>=5.0          # METS parsing
pyyaml>=6.0        # mets.yaml fallback
boto3>=1.34        # S3 fetch
tenacity>=8.2      # retry/backoff for API calls
tiktoken           # token counting for budget tracking
```

Plus: `python -m spacy download en_core_web_lg`

---

## 12. Out of scope for this prototype

- Batch / parallel processing across many docs (use a queue around `run.py`)
- Web UI / dashboard
- Human review interface
- Refresh / re-check infrastructure for current-status
- Cross-document linking (using newer EIS as a status source for older one)
- Specialist fine-tuned models (comes after enough calibration data exists)
- Tier 2 / 3 source integration (Wikipedia, news, academic)

These are real follow-on work but they don't belong in v1.
