# EIS Metadata Pipeline — v1 Build Brief

> **You are a Claude instance asked to build the EIS v1 pipeline. Read this brief first; it's what you implement against.**
>
> This brief is a build-and-run version of `inter_agent_plan.md`. The canonical plan deliberately leaves several decisions open for measurement (Phase 0a). This brief closes those decisions by picking the simpler option in every case, so you can build without a corpus audit.
>
> Where this brief and `inter_agent_plan.md` disagree, **this brief wins for v1.**

---

## 0. Read order

1. This file (whole thing) — the build target.
2. `inter_agent_plan.md` §0 (schema) and §1.2 (stage specs) — the canonical detail.
3. `about_pipeline/PIPELINE_OVERVIEW.md` — describes the existing implementation at `eis_pipeline/`. Read for reference patterns (NUL client, token ledger, prompt style). **Do not modify `eis_pipeline/`.**
4. `GKG_Tests/TEST_RUN_*.md` — the writeup format you'll mirror for v1 eval runs.

---

## 1. Decisions this brief locks (vs. `inter_agent_plan.md`)

| Decision | Canonical plan | v1 build brief |
| --- | --- | --- |
| Phase 0a corpus audit | Required before build | **Skipped.** NUL Digital Collections API is declared authoritative for title, year, lead_agency. METS reliability not measured. |
| 10-doc adjudicated gold set | Required for thresholds | **Skipped.** Eval is narrative test-run writeups (see §7). |
| Input source | S3 Inventory + Mongo + METS XML | **`docs_with_digits.json` only**, supplemented by NUL API calls. |
| Storage | `s3://nu-impulse-production/...` | **Local filesystem** at `spring_pipeline/v1_build/output/`. |
| Spatial DB | DuckDB Spatial **or** PostGIS | **DuckDB Spatial** (in-process, no separate server). |
| Section-detection embedding model | "embedding similarity" | **`sentence-transformers/all-MiniLM-L6-v2`.** |
| GLiNER model for location | "GLiNER with EIS-typed entities" | **`urchade/gliner_multi-v2.1`** with labels `national_forest, watershed, highway_corridor, place, tribe`. |
| Stance escalation threshold | "confidence low or mixed" | **Sonnet runs if Haiku confidence < 0.7 OR stance == "mixed".** |
| Layout-aware section detection (N1) | Conditional on Phase 0a | **Dropped.** Input has no layout metadata. |
| Image-based review UI | Conditional on Phase 0a | **Dropped.** Input has no images. CLI review tool is text-only. |
| Corpus filter | `eis_type=Final AND lead_agency=USFS AND year≥2000` | **Same filter, applied via NUL lookup.** If lead_agency is missing from NUL, doc is excluded with `excluded_reason: "missing_agency_metadata"`. |
| Manifest authority | S3 Inventory ∩ Mongo | **Pre-filtered doc-key list** at `v1_build/manifest/usfs_final_2000_plus.txt`, generated once by `tools/build_manifest.py`. |
| Cost cap | $1/doc | **$1/doc — same.** Hard cap with partial-output semantics. |
| Cross-document grouping (Stage 1.5) | Detect appendix/RtC/errata grouping | **Trivial passthrough.** Each `doc_key` is its own logical publication. Cross-volume grouping deferred to v2. |
| Pydantic models | Not transcribed in canonical plan | **You build them.** See §4. |
| Runtime verification | Hard gates only | **Three layers**: (a) Pydantic model validators fire on every write, (b) deterministic self-consistency checks (Stage 4a), (c) Sonnet critic scoped to summary, alternative descriptions, stakeholder stance, themes (Stage 4b). See §6 Stage 4. |

---

## 2. Practice-mode I/O

### Input

- **`docs_with_digits.json`** — flat JSON at `/Users/gracegormley/Desktop/Y2/Q2/Knight Lab/docs_with_digits.json`. Shape: `{doc_key: full_text_string}`. No page boundaries, no OCR confidence. Pages are computed by splitting on line boundaries into ~2500-char fake pages.
- **NUL Digital Collections API** — fetches title, creator/agency, date for a given accession ID. The existing implementation in `eis_pipeline/pipeline/io_layer.py` (`load_from_digits_json` and the NUL fetch helper) is your reference; port it, don't rewrite from scratch.
- **Manifest** — `v1_build/manifest/usfs_final_2000_plus.txt`, one doc_key per line. Generate once via `tools/build_manifest.py` which walks `docs_with_digits.json`, hits NUL per doc, and writes the filtered list.

### Output

```
v1_build/output/
├── manifest/
│   └── usfs_final_2000_plus.txt
├── nul_cache/{doc_id}.json                 # cached NUL responses
├── stage1_assembled/{publication_id}.json
├── stage1_5_grouped/{publication_id}.json
├── stage2_sections/{publication_id}.json
├── stage3_fields/{publication_id}.json     # accumulated; one file per doc, updated by each 3* stage
├── stage4_validated/{publication_id}.json
├── final/{publication_id}.json             # the deliverable
├── review_log/{publication_id}_{field}_{ts}.json
├── runs/{timestamp}.log
└── token_ledger.json
```

### Provenance rules

- `provenance.source = "nul_api"` for title, year, lead_agency, agency_office.
- `provenance.source = "docs_with_digits_json"` for body-text-derived fields (sections, summary, alternatives, location mentions, stakeholders).
- `provenance.page` is the **fake-page index** (1-based). Record it; do not pretend it's authoritative. Downstream consumers know v1 page numbers are approximate.

---

## 3. Repo layout

Build under `spring_pipeline/v1_build/`. Do not touch `spring_pipeline/eis_pipeline/`.

```
spring_pipeline/v1_build/
├── README.md
├── requirements.txt
├── run.py                                  # CLI entrypoint
├── tools/
│   └── build_manifest.py                   # generates usfs_final_2000_plus.txt
├── pipeline/
│   ├── __init__.py
│   ├── config.py                           # model IDs, taxonomies, thresholds, regex patterns
│   ├── schema.py                           # Pydantic models — full v2 schema, deferred fields enforced
│   ├── llm_client.py                       # Anthropic wrapper: dry-run, budget cap, per-model usage
│   ├── token_ledger.py                     # port from eis_pipeline/pipeline/token_ledger.py
│   ├── nul_client.py                       # NUL API + disk cache
│   ├── ingest.py                           # Stage 1
│   ├── grouping.py                         # Stage 1.5 (passthrough in v1)
│   ├── sections.py                         # Stage 2
│   ├── stage3a_mets_fields.py
│   ├── stage3b_eis_type.py
│   ├── stage3c_summary.py
│   ├── stage3d_location.py
│   ├── stage3e_alternatives.py
│   ├── stage3f_stakeholders.py
│   ├── stage3g_themes.py
│   ├── validation.py                       # Stage 4 — hard gates, self-consistency, rule thresholds, routing
│   ├── critic.py                           # Stage 4b — Sonnet per-field critic
│   ├── output_writer.py                    # Stage 5
│   ├── review_cli.py                       # Stage 6
│   └── gazetteer/
│       ├── __init__.py
│       ├── tiger_load.py                   # one-time loader for Census TIGER state/county shapefiles
│       ├── usfs_load.py                    # one-time loader for USFS Admin Forests
│       └── lookup.py                       # query API used by stage3d
├── prompts/
│   ├── 3b_eis_type_fallback.txt
│   ├── 3c_summary.txt
│   ├── 3d_location_mentions.txt
│   ├── 3e_alternatives_description.txt
│   ├── 3f_stance_haiku.txt
│   ├── 3f_stance_sonnet.txt
│   ├── 3f_quote_select.txt
│   ├── 3g_themes.txt
│   └── 4_critic.txt                        # parameterized per-field critic prompt
├── tests/
│   ├── fixtures/                           # short text snippets, hand-curated
│   ├── test_schema.py
│   ├── test_ingest.py
│   ├── test_sections.py
│   ├── test_stakeholder_parser.py
│   └── test_verbatim_check.py
└── output/                                 # gitignored
```

---

## 4. Schema → Pydantic

Transcribe `inter_agent_plan.md` §0 into `pipeline/schema.py`. Key requirements:

- **One top-level model: `EISRecord`.** Pydantic v2 (`from pydantic import BaseModel, Field`).
- **Field wrapper: `FieldWithStatus[T]`** — generic with `value: T | None`, `status: StatusLiteral`, `provenance: Provenance | None`.
- **Provenance**: `{page: int | None, char_offset_raw: tuple[int, int] | None, section: str | None, source: Literal["nul_api", "docs_with_digits_json"], source_text_hash: str}`.
- **Quote model**: `text_raw`, `text_display`, `char_offset_raw`, `page`, `section`, `source_text_hash`, `normalization_rules_applied: list[str]`.
- **Status enums** per field. Deferred-v1 fields use `Literal["deferred_v1"]` to make accidental population a schema error.
- **`stakeholder_extraction_scope: Literal["organizations_only"]`** locked in v1.
- **`excluded_stakeholder_types: list[str]`** with `["person"]` as the v1 default.
- **`comment_blocks: list[CommentBlock]`** at the top level (schema addition beyond canonical plan §0). Each `CommentBlock = {block_id, char_span_raw: tuple[int, int], pages: list[int]}`. Required so Stage 4 self-consistency can verify quote-inside-block.
- **Model validators** (`@model_validator(mode="after")`) that raise on bad structure. These catch *producer* bugs; content checks live in Stage 4.
  - Every `subtheme.parent` ∈ chosen `themes.primary[].value`.
  - Every `quote.char_offset_raw[0] < char_offset_raw[1]`, both ≥ 0.
  - Every `provenance.section` matches a name in `sections[].name`.
  - Every `stakeholder.stance_records[].quote.section ∈ {public_comments, response_to_comments}`.
  - `location.project_area_polygon` non-null only if some `named_places[]` entry has `role: project_area` and `polygon: not null`.
  - `cross_publication_links.status == "deferred_v1"` (no auto-linking in v1).
  - Every `stakeholder.comment_block_id` matches a `comment_blocks[].block_id`.

Validate every output JSON against `EISRecord`. A schema violation is a bug — fix the producer, don't relax the model.

---

## 5. Build sequence

Build in groups, in order. Each group has a clear acceptance gate. Recommended handoff shape: **stop after Group B, run on one doc, review with Grace before continuing.**

| Group | Stages | Days | Acceptance gate |
| --- | --- | --- | --- |
| **A — Scaffold + ingest** | schema, NUL client, Stage 1, Stage 1.5 | 1 | `run.py --stage 1 --doc-key <k>` produces a valid Stage 1 JSON. |
| **B — Deterministic Stage 2 + trusted fields** | Stage 2 (sections), Stage 3a (NUL fields), Stage 3b (EIS type) | 1–2 | One doc has title, year, agency, EIS type, sections. **Checkpoint: review with Grace.** |
| **C — Extractive Stage 3 (cheap)** | Stage 3c (summary), Stage 3e (alternatives), Stage 3g (themes) | 2 | Doc has summary, alternatives, themes. |
| **D — Location** | Gazetteer loaders, Stage 3d | 2 | Doc has location with `named_places`, `project_area_polygon` (or null), `spatial_summary`. |
| **E — Stakeholders** | Stage 3f (comment-block parser, stance, span-ID quote selection) | 3 | Doc has stakeholders[] with verified quotes; verbatim hard gate firing correctly. |
| **F — Validation + critic + output + review** | Stage 4 (hard gates + self-consistency + Sonnet critic), Stage 5, Stage 6 | 2 | `run.py` produces complete schema-valid JSON; critic verdicts attached to every complex field; CLI review surfaces below-threshold fields. |
| **G — Test runs** | Run on 5 docs, write writeups | parallel with E/F | One `TEST_RUN_v2_{N}.md` per doc; bugs filed; cost measured. |

---

## 6. Per-stage build briefs

### Stage 1 — Ingest (`pipeline/ingest.py`)

**Build:**
- Load `docs_with_digits.json` once per process (lazy module-level cache).
- For a given `doc_key`, fetch NUL metadata via `nul_client.get(doc_id)`.
- Compute `text_normalized` from raw text: whitespace normalization + soft-hyphen dehyphenation across line breaks (`-\n` → `` empty `` with offset map) + Unicode quote normalization. Keep raw text untouched. Maintain an alignment map: `normalized_char → raw_char`.
- Compute fake pages by splitting raw text on line boundaries into ~2500-char buckets. Record `{page_num, char_start_raw, char_end_raw}`.
- **No OCR confidence filtering. Confidence is metadata, never an extraction filter.**

**Output:**
```jsonc
{
  "publication_id": "...",
  "physical_record_ids": ["..."],
  "raw_text": "...",
  "text_normalized": "...",
  "alignment_map": [{"norm_start": 0, "norm_end": 1200, "raw_start": 0, "raw_end": 1210}, ...],
  "pages": [{"page_num": 1, "char_start_raw": 0, "char_end_raw": 2487}, ...],
  "nul_metadata": { /* raw NUL response */ }
}
```

**Acceptance:** schema-valid Stage 1 JSON for `p1074_35556036099737`.

---

### Stage 1.5 — Grouping (`pipeline/grouping.py`)

**Build:** trivial passthrough.
- `publication_id = doc_key`
- `components = [{record_id: doc_key, role: "main", confidence: 1.0}]`
- `is_supplemental = false` (corpus filter excludes supplementals)
- Cross-volume grouping (appendix/RtC/errata detection) is deferred to v2.

---

### Stage 2 — Section detection (`pipeline/sections.py`)

**Build, in priority order:**

1. **Regex pass.** For each section in the CEQ taxonomy (`cover, summary, purpose_and_need, alternatives, affected_environment, environmental_consequences, public_comments, response_to_comments, rod`), try a regex set. Starter patterns:
   - `cover`: pages 1–3 by default (no detection).
   - `summary`: `^\s*(SUMMARY|EXECUTIVE\s+SUMMARY|ABSTRACT)\s*$` (multiline, case-insensitive).
   - `purpose_and_need`: `^\s*(PURPOSE\s+(AND|&)\s+NEED|NEED\s+FOR\s+(THE\s+)?(ACTION|PROJECT))\s*$`.
   - `alternatives`: `^\s*(ALTERNATIVES?(\s+(CONSIDERED|ANALYZED|TO\s+THE\s+PROPOSED\s+ACTION))?|DESCRIPTION\s+OF\s+ALTERNATIVES)\s*$`.
   - `affected_environment`: `^\s*(AFFECTED\s+ENVIRONMENT|EXISTING\s+CONDITIONS)\s*$`.
   - `environmental_consequences`: `^\s*(ENVIRONMENTAL\s+(CONSEQUENCES|EFFECTS|IMPACTS))\s*$`.
   - `public_comments`: `^\s*(PUBLIC\s+(COMMENTS?|INVOLVEMENT|PARTICIPATION)|COMMENTS\s+ON\s+THE\s+DRAFT)\s*$`.
   - `response_to_comments`: `^\s*(RESPONSE\s+TO\s+COMMENTS|COMMENTS?\s+AND\s+RESPONSES)\s*$`.
   - `rod`: `^\s*(RECORD\s+OF\s+DECISION|ROD)\s*$`.
   - Reject matches inside ZIP-code lines, legal citations (`Section\s+\d+\(`), and addresses.

2. **Embedding-similarity fallback** if a *required-for-downstream* section is missing. Canonical descriptors (embed once, cache):
   - `summary`: "executive summary of the project, what it does, why, what the impacts are"
   - `purpose_and_need`: "the reason this project is needed, the goals it serves, what problem it addresses"
   - `alternatives`: "alternatives considered including the no-action alternative and other options the agency evaluated"
   - `public_comments`: "letters and comments from members of the public, agencies, and tribes submitted during the public review"
   - `response_to_comments`: "the agency's written responses to public comments"

   Model: `sentence-transformers/all-MiniLM-L6-v2` loaded once at process start.

   Compare each candidate heading-like line against descriptors; assign section if cosine ≥ 0.55 and best candidate ≥ 0.10 above next-best.

3. **Abstention is a hard gate.** If a section needed by a downstream stage is `not_found` after both passes, set `sections[section_name].status = "not_found"` and let the downstream stage abstain.

**Output:** `{sections: [{name, char_span: [s, e], pages: [start, end], confidence, status}]}`.

---

### Stage 3a — METS-equivalent fields (`pipeline/stage3a_mets_fields.py`)

**Build:** read from the cached NUL response.
- `title` ← NUL `title[0]` (or first non-empty). `status: "ok"`. `provenance.source: "nul_api"`.
- `year` ← parse from NUL `date_created[0]`. If parse fails, fall back to regex on first 5 pages of raw text; `status` becomes `"needs_review"`. Year range gate: 1969 ≤ year ≤ current.
- `lead_agency` ← NUL `creator[*]` filtered against the controlled vocab in `config.AGENCY_VOCAB` (copy from `eis_pipeline/pipeline/config.py`). For v1 corpus, this should be `"USFS"`; otherwise the doc was wrongly filtered into the manifest — log a warning, keep the value.
- `agency.office_or_region`: deferred_v1.
- `agency.cooperating_agencies`: deferred_v1.
- `date.publication`: from NUL with `precision: "year"` unless NUL provides a month/day (rare).
- `date.filing` / `date.comment_deadline` / `date.rod_date`: deferred_v1.

---

### Stage 3b — EIS Type (`pipeline/stage3b_eis_type.py`)

**Build:**
1. Regex on cover section: `(Draft|Final|Supplemental(?:\s+(?:Draft|Final))?)\s+Environmental\s+Impact\s+Statement`. Take the first match.
2. If regex finds nothing, call Haiku 4.5 with the cover-section text and the prompt at `prompts/3b_eis_type_fallback.txt`:

   ```
   You are reading the cover page of a U.S. Environmental Impact Statement.
   Identify the EIS type. Return exactly one of: Draft, Final, Supplemental, ROD, NOI.
   If the text does not contain enough information to decide, return: Unlabelled.
   Return only the single word, no explanation.

   COVER TEXT:
   {cover_text}
   ```

3. Validate output is in `{Draft, Final, Supplemental, ROD, NOI, Unlabelled}`. For v1, corpus filter restricts to Final; non-Final docs are bugs in manifest filtering.

---

### Stage 3c — Summary (`pipeline/stage3c_summary.py`)

**Build:**
- If `sections.summary.status == "not_found"`: emit `{value: null, status: "skipped_section_not_found"}` and exit.
- Otherwise: extract text of the Summary section, cap at 8K input tokens (truncate from the end). Call Haiku 4.5 with `prompts/3c_summary.txt`:

  ```
  You are summarizing a U.S. Environmental Impact Statement based ONLY on its Summary section.
  Do not introduce facts that are not present in the provided text.

  Produce a single paragraph, 80–300 words, that covers:
  - The community or area affected.
  - The proposed action (what the agency plans to do).
  - The reason the action is needed.
  - The major environmental impacts the document identifies.

  Cite nothing. Do not hedge with "the document states." Just summarize the content factually.

  SUMMARY SECTION TEXT:
  {summary_section_text}
  ```

- Auto-check: 80 ≤ word_count ≤ 300; must reference the lead_agency abbreviation OR a location string from the title. Fail → `status: "needs_review"`.

---

### Stage 3d — Location (`pipeline/stage3d_location.py`)

**Build:**

1. **Mention extraction.** From Cover/Summary/Purpose-and-Need/Alternatives sections **only** (never public comments). Two parallel passes, merged:
   - **GLiNER** with `urchade/gliner_multi-v2.1`, labels `[national_forest, watershed, highway_corridor, place, tribe]`. Threshold 0.5.
   - **Sonnet structured extraction** with `prompts/3d_location_mentions.txt`:

     ```
     Extract every named geographic place mentioned in the following EIS text excerpt.
     For each place, return:
       - name (the canonical place name as it appears)
       - feature_type (one of: state, county, city, national_forest, national_park, watershed, river, highway, tribal_land, other)
       - role (one of: project_area, commenter_address, context_reference, unknown)
       - spatial_qualifier (one of: exact, partial, near, between, within, downstream_of, none)

     Return a JSON array. Do not invent places. Do not include addresses or ZIP codes.
     If the text contains no geographic places, return an empty array.

     TEXT:
     {section_text}
     ```

2. **Spatial-qualifier parsing.** Mentions with `spatial_qualifier ∈ {partial, near, between, within, downstream_of}` are kept in `context_polygons` only — they do NOT promote to `project_area_polygon`.

3. **Gazetteer lookup** via `pipeline/gazetteer/lookup.py`:
   - State / county: TIGER/Line lookup (returns polygon).
   - National Forest: USFS Admin Forests lookup (returns polygon).
   - All other feature types: name-only, `polygon: null`.

4. **Polygon resolution:** if a `role: "project_area"` mention with `spatial_qualifier: "exact"` has a gazetteer match, set `project_area_polygon` from that boundary. Otherwise `null`. **Never manufacture geometry.**

5. **Geometry classification:** `geometry_role` defaults to `unknown` in v1 (full classification is v2).

6. **GeoJSON validation:** WGS84 (EPSG:4326), normalize Polygon → MultiPolygon, validate with `shapely.is_valid` and `is_simple`. Semantic check: extracted state must contain the polygon's centroid per TIGER.

**Gazetteer data sources (one-time download):**
- TIGER: `https://www2.census.gov/geo/tiger/TIGER2024/STATE/` and `/COUNTY/`.
- USFS Admin Forests: `https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.AdministrativeForest.zip`.

`tiger_load.py` and `usfs_load.py` are one-shot scripts that download, load into DuckDB Spatial, build R-tree indices. Run once during Group D setup.

---

### Stage 3e — Alternatives (`pipeline/stage3e_alternatives.py`)

**Build:**

1. **Stage A — labels (free, no LLM).** Inside the Alternatives section, regex for label patterns:
   - `^(Alternative\s+[A-Z0-9]+|Alternative\s+\d+|No\s+Action(\s+Alternative)?|Preferred\s+Alternative|Alignment\s+[A-Z0-9]+|Variation\s+[A-Z0-9]+|Option\s+[A-Z0-9]+|Route\s+[A-Z0-9]+|Corridor\s+[A-Z0-9]+|Plan\s+[A-Z0-9]+|Build|No\s+Build)\b`
   - Multiline, case-insensitive, must be at line start or after a heading marker.
   - Dedupe label strings.

2. **Stage B — Sonnet description.** For each label, extract the first 500 tokens of text after the label. Call Sonnet with `prompts/3e_alternatives_description.txt`:

   ```
   You are summarizing a single alternative considered in a U.S. Environmental Impact Statement.

   The alternative is labeled: {label}

   Below is the text that immediately follows this label in the document. Based on this text only,
   write a one-to-two-sentence factual description of what this alternative entails. Do not include
   the label name in the description. Do not invent details not present in the text.

   TEXT:
   {first_500_tokens_after_label}
   ```

3. **Output:** `alternatives: [{label, description, provenance: {char_offset_raw, page}}]`.

---

### Stage 3f — Stakeholders (`pipeline/stage3f_stakeholders.py`)

**Most complex stage. Build in this order:**

1. **Comment-block detection.** Operate on the `public_comments` and `response_to_comments` sections only. Starter regex patterns (refine after you see real docs):
   - **Numbered comment headers**: `^(COMMENT|Comment)\s+(?:No\.\s*)?(\d+)[\s:.\-]`
   - **Letterhead detection**: a block starting with an ALL-CAPS organization name (4–80 chars) followed within 5 lines by a street-address-shaped line.
   - **Signature blocks**: `^\s*(Sincerely|Respectfully(\s+submitted)?|Regards|Thank\s+you)[,.]?\s*$` followed within 5 lines by a name + title.
   - **Hearing transcripts**: speaker labels like `^([A-Z][A-Z\s.]+):` followed by the speech.
   - **Form-letter groups**: `^(Form\s+Letter|Letter\s+Type)\s+\d+:|^Group\s+\d+:\s+\d+\s+(respondents|signatories|members)`.

   Each detected block gets a `block_id` (sequential), `char_span_raw`, `pages`.

2. **Comment/response split.** Inside each block, find a `^\s*(Response|Agency\s+Response):\s*` marker. Split into `comment_text` (before) and `agency_response_text` (after). If no marker, the whole block is `comment_text` (likely a public-comment-only section, not RtC). If split fails on what looks like an RtC block (block has high keyword density for agency replies), mark `needs_review` and skip quote extraction.

3. **Author canonicalization.** Extract `{name, type, affiliation, authorship_role}` from the block header. **Reject persons** — if the header looks like a person name (regex: `^[A-Z][a-z]+\s+[A-Z][a-z]+,?\s*(Ph\.?D\.?|Esq\.?|Jr\.?|Sr\.?|III)?$` without an org indicator), skip the block entirely. Record `excluded_stakeholder_types: ["person"]` at the doc level.

4. **Stance (two-pass).** For each block's `comment_text`:
   - **Haiku pass** with `prompts/3f_stance_haiku.txt`:

     ```
     You are reading a single public comment on a U.S. Environmental Impact Statement.
     Identify the commenter's stance on each clearly-stated target.

     Return a JSON array of stance records. Each record:
     {
       "stance": "supportive" | "opposed" | "mixed" | "neutral",
       "stance_target": {
         "type": "proposed_action" | "no_action" | "specific_alternative" | "mitigation" | "process" | "unknown",
         "label_hint": "<the alternative label if mentioned, else null>"
       },
       "confidence": 0.0-1.0
     }

     Do not include records where you are not at least 0.5 confident.

     COMMENT TEXT:
     {comment_text}
     ```

   - **Sonnet escalation** if any Haiku stance has `confidence < 0.7` OR `stance == "mixed"`. Use `prompts/3f_stance_sonnet.txt` (same shape, with few-shot examples of `mixed` vs. `opposed` vs. `supportive`).

5. **Span-ID quote selection.**
   - Split `comment_text` into spans (sentence or paragraph), each gets a stable `span_id`.
   - Cue-based ranking: score each span by presence of stance cues (`oppose, support, do not support, object to, endorse, recommend against, urge, request that, concerned that, strongly disagree`) and absence of emotive-only cues (`sincerely hope, appreciate the opportunity, thank you for`).
   - Take top-3 spans + the stance_target, send to Sonnet with `prompts/3f_quote_select.txt`:

     ```
     You are picking the best quote to represent a commenter's stance.

     Commenter: {author_name}
     Stance: {stance}
     Stance target: {stance_target}

     Below are 3 candidate spans from the comment text, each with an ID. Choose the ONE that best
     represents this commenter's stance on this target. Do not modify or copy the text. Return only
     the span_id of your choice.

     SPAN A (span_id={a_id}): {a_text}
     SPAN B (span_id={b_id}): {b_text}
     SPAN C (span_id={c_id}): {c_text}

     Return JSON: {"span_id": "<chosen_id>"}
     ```

   - **The model never types the quote.** Code looks up the span by `span_id` and copies `text_raw`, `char_offset_raw`, `page` from the Stage 1 alignment map. `text_display` is the normalized version.

6. **Verbatim verification.** Hard gate. `text_raw` must appear as an exact substring of `raw_text[char_offset_raw[0]:char_offset_raw[1]]`. Failure → `quote_status: "rejected_nonverbatim"`, drop the quote (keep the stance), log to review.

---

### Stage 3g — Themes (`pipeline/stage3g_themes.py`)

**Build:**
- If `summary.status != "ok"`: emit `{primary: [], subthemes: [], status: "skipped_summary_unavailable"}`.
- Otherwise: Haiku 4.5 with `prompts/3g_themes.txt` and input = `summary.value + "\n\nTitle: " + title`:

  ```
  You are classifying a U.S. Environmental Impact Statement into a controlled theme taxonomy
  based on the project's title and summary.

  PRIMARY THEMES (pick 1–2):
  energy_infrastructure, transportation, land_management, water_resources, defense_and_military,
  urban_development, mining_and_extraction, agriculture_and_forestry, wildlife_and_habitat,
  cultural_heritage, waste_and_remediation, aerospace_and_space_exploration, other

  SUBTHEMES (pick 0–3, must be scoped to a primary you chose):
  {subthemes_per_primary_list}

  Do NOT default to "other". Pick the closest specific theme and use the confidence score to
  express uncertainty. Use "other" only if the project genuinely does not fit any specific theme.

  TITLE: {title}
  SUMMARY: {summary}

  Return JSON:
  {
    "primary":   [{"value": "...", "confidence": 0.0-1.0}, ...],
    "subthemes": [{"value": "...", "parent": "...", "confidence": 0.0-1.0}, ...]
  }
  ```

- Hard gate (deterministic): every `value` in controlled vocab; every subtheme's `parent` in the doc's chosen primaries. Out-of-vocab → drop, `status: "needs_review"`.
- The subthemes-per-primary list is locked in `config.SUBTHEMES`. Pull from `inter_agent_plan.md` §1.2 Stage 3g.

---

### Stage 4 — Validation (`pipeline/validation.py` + `pipeline/critic.py`)

Three-layer verification. Run in this order; each layer can downgrade `status` or null bad values.

#### Layer 1 — Pydantic structural validators

Already fire at model construction (see §4). If they raise, the bug is in the producer — do not relax the schema. Cover: status enums, source enums, char-span ordering, subtheme `parent ∈ primary`, provenance section names, polygon presence preconditions.

#### Layer 2a — Hard gates (deterministic, binary)

- **Quotes:** verbatim substring check on raw text. Failure → drop the quote, keep the stance, log to review.
- **Year:** 1969 ≤ year ≤ current. Failure → `year.status = needs_review`.
- **Themes:** every value in controlled vocab; every subtheme `parent` in chosen primaries. Out-of-vocab → drop, `themes.status = needs_review`.
- **Geocoding:** extracted state's TIGER polygon must contain the project-area polygon's centroid. Mismatch → null the polygon, `location.status = needs_review`.
- **Schema:** Pydantic validation re-runs before write (belt-and-suspenders).

#### Layer 2b — Self-consistency checks (deterministic, no LLM)

Catch a class of single-pass errors no LLM is needed to find. Pure functions over the assembled `EISRecord`.

- **Summary mentions location.** If `location.named_places[0].name` (case-insensitive substring) is absent from `summary.value`, flag `summary.status = needs_review`. Skipped if `location.status != ok`.
- **Summary contains an action verb.** Regex over `summary.value` for `(construct|build|expand|decommission|designate|lease|reroute|widen|abandon|reconstruct|restore|harvest|treat|prescribe|authorize|approve|withdraw|implement|adopt)`. Zero matches → `summary.status = needs_review`.
- **Themes coherent with summary.** For each chosen `primary` theme, look up its keyword set in `config.THEME_KEYWORDS` (one small list per primary — `transportation: [highway, road, transit, rail, bridge, corridor]`, etc.). If `summary.value` contains zero keywords for any chosen primary, flag `themes.status = needs_review`.
- **Alternatives count sanity.** Every EIS has ≥ 2 alternatives (no-action + at least one build). `len(alternatives) < 2` → `alternatives.status = needs_review`.
- **Stakeholder quote inside its block.** Every `stakeholder.stance_records[].quote.char_offset_raw` must fall inside its `comment_block_id`'s `char_span_raw` (looked up via the doc-level `comment_blocks[]` array). Out-of-bounds → drop the quote, log "span-ID lookup error" to review.

#### Layer 3 — Sonnet critic (`pipeline/critic.py`)

Scoped to fields the deterministic layers can't verify: claim-grounding for **summary**, **alternative descriptions**, **stakeholder stance**, and **themes**. One Sonnet call per (field, item) pair. Single parameterized prompt at `prompts/4_critic.txt`:

```
You are verifying whether a claim is supported by a source text.

CLAIM TYPE: {field_type}    # summary | alternative_description | stance | theme
CLAIM: {claim}
SOURCE TEXT: {source}

Decide whether the claim is supported:
- "yes": every factual element of the claim is directly supported by the source text.
- "partial": mostly supported, but contains at least one element that goes beyond the source (overreach, added detail, slight misstatement).
- "no": the claim contradicts the source, or makes a substantive assertion not present in the source.

Return JSON only: {"verdict": "yes" | "partial" | "no", "justification": "<one sentence>"}
```

Call sites:

- **Summary critic.** `claim = summary.value`, `source = summary section text`. `no` → `summary.status = needs_review`; `partial` → `summary.status = partial_grounding`. Verdict attached to `summary.critic`.
- **Alternative description critic.** Per alternative: `claim = description`, `source = first 500 tokens after label`. Per-alternative verdict; any `no` → that alternative's `status = needs_review`.
- **Stance critic.** Per `stance_record`: `claim = "<author> is <stance> on <stance_target>"`, `source = quote.text_raw`. `no` → drop the stance_record (keep the stakeholder), log to review. `partial` → keep but downgrade `stance_confidence` by 0.3.
- **Themes critic.** Per primary theme: `claim = "This document is about <theme>"`, `source = summary.value`. `no` → drop that theme. If all primaries dropped → `themes.status = needs_review`.

**Budget discipline:** critic runs at the end of the pipeline, after all extractors. If the per-doc budget is exhausted, critic is skipped and every critic-eligible field gets `critic: {status: "skipped_budget_cap"}`. Hard gates and self-consistency still run regardless.

#### Layer 4 — Rule-threshold field confidence + routing

Per field, simple rules (not calibrated; calibrated logistic deferred to v2):

- `title`: NUL-sourced → 0.99. Extractive fallback → 0.75.
- `year`: NUL-sourced → 0.98. Extractive → 0.80.
- `summary`: critic `yes` + self-consistency passed → 0.90. `partial` → 0.65. `no` → 0.30.
- `alternatives`: per-alternative critic verdicts averaged.
- `themes`: critic-survived primary confidences averaged; if all primaries dropped → 0.0.
- `location`: polygon present and TIGER-semantic-check passed → 0.90. Polygon null → 0.40.
- `stakeholders`: per-stakeholder, average of stance_confidence (post-critic adjustment); doc-level is the mean.

Field-level review routing: any field with confidence < 0.70 OR `status ∈ {needs_review, partial_grounding, partial_budget_cap}` is queued in `validation.review_routing`. Doc-level `auto_approve` requires every field ≥ 0.70 AND every field's `status == ok`.

---

### Stage 5 — Output (`pipeline/output_writer.py`)

**Build:**
- Compose final `EISRecord` from Stage 1.5 + Stage 2 + Stage 3a–g + Stage 4.
- Pydantic-validate. Schema violation = bug, do not write.
- Write to `output/final/{publication_id}.json`.
- Append to `output/index/extracted.parquet` (DuckDB-readable). Columns: `publication_id, title, year, lead_agency, eis_type, primary_themes (list), n_stakeholders, n_alternatives, has_polygon, validation_routing, run_timestamp, cost_usd`.

---

### Stage 6 — CLI Review tool (`pipeline/review_cli.py`)

**Build:** lightweight CLI. For one doc:
- For each field in `validation.review_routing`:
  - Print field name, extracted value, status, confidence.
  - Print OCR text window: `raw_text[char_offset_raw[0] - 200 : char_offset_raw[1] + 200]`.
  - Print page number.
  - Prompt: `[a]ccept / [r]eject / [c]orrect / [s]kip`.
- Save annotator decisions to `output/review_log/{publication_id}_{field}_{timestamp}.json`.

---

## 7. Eval mode — test runs, no gold set

**v1 does NOT use a 10-doc adjudicated gold set.** Reasons (Grace's call): small adjudicated sets don't generalize across the corpus, and human-labeling time is better spent fixing prompts than producing precision/recall numbers.

**Instead:** narrative test-run writeups, same pattern as the existing `GKG_Tests/TEST_RUN_*.md` files.

For every meaningful pipeline change (new prompt, new stage, threshold tweak), run on 3–5 docs spanning short/medium/long and write a `GKG_Tests/TEST_RUN_v2_{N}.md` per doc.

### What enforces v1 quality without labels

- **Pydantic model validators** fire on every output construction — structural invariants (subtheme-parent, status enums, provenance enums, char-span ordering, polygon preconditions) raise before the JSON is written. A schema-violating producer is a bug, not a JSON to ship.
- **Hard gates** (verbatim quote, vocab membership, abstention on missing section, GeoJSON validity, year range) fire automatically and either drop bad output or downgrade status.
- **Deterministic self-consistency checks** (Stage 4a) catch single-pass errors no LLM is needed to find — summary missing a location reference, summary missing an action verb, themes incoherent with summary keywords, alternatives count < 2, stakeholder quote outside its block's char span.
- **Sonnet critic** (Stage 4b) verifies that summary claims, alternative descriptions, stakeholder stance, and theme assignments are grounded in their cited source spans. Any `no` verdict downgrades field status to `needs_review`; `partial` downgrades to `partial_grounding`.
- **`other`-rate batch warning** (themes) catches taxonomy gaps without per-doc labels.
- **Cost regression** via `token_ledger.json` catches accidental Opus-on-everything mistakes.
- **Schema validation** catches structural drift.

### Test run writeup template (`GKG_Tests/TEST_RUN_v2_{N}.md`)

```markdown
# Test Run v2.{N} — {doc_label}

**Doc:** `{doc_key}`
**Date:** {YYYY-MM-DD}
**Word count:** {n}
**Run cost:** ${X.XX}
**Run duration:** {N} min

## Per-field outcome

| Field | Status | Confidence | Notes |
| --- | --- | --- | --- |
| title | ok | 0.99 | NUL |
| year | ok | 0.98 | NUL |
| lead_agency | ok | 0.99 | USFS, NUL |
| eis_type | ok | — | Final, regex |
| summary | ok | 0.85 | clean |
| alternatives | ok | — | 4 labels found |
| location | needs_review | 0.40 | no polygon — Idaho Panhandle NF didn't gazetteer-match |
| stakeholders | ok | 0.78 | 12 orgs, 11 verified quotes |
| themes | ok | 0.91 | land_management + agriculture_and_forestry |

## Spot-check

- 3 quotes manually verified against raw OCR text: ✅ all match.
- Summary read against the doc's actual Summary section: ✅ accurate.
- Location: project area is Idaho Panhandle National Forest. Gazetteer missed it.
- Critic verdicts: summary `yes`; alternative descriptions 4/4 `yes`; stance verdicts 10/12 `yes`, 2 `partial` (see Bugs); themes `yes`.
- Self-consistency: all 5 checks passed.

## Bugs found

- Location gazetteer doesn't include national forests by abbreviation ("IPNF").
- Stance for one stakeholder ("Friends of the Clearwater") labeled "neutral"; reading the comment, it's clearly "opposed."

## Next-run recommendation

Fix gazetteer abbreviation lookup. Add a few-shot example to `prompts/3f_stance_haiku.txt` for environmental NGOs (they're rarely neutral).

## One-line summary

Pipeline works end-to-end. Location and stance need iteration.
```

---

## 8. Open items — bring back to Grace, do not decide alone

- **Subtheme list.** Use the starter list in `inter_agent_plan.md` §1.2 Stage 3g exactly. Grace said "gonna change down the line" — do not iterate during the build.
- **Which 5 docs for first test runs.** Recommendation: `p1074_35556036099737` (already-used Harlem doc, for regression sanity-check) plus 4 USFS Final EIS ≥ 2000 from the generated manifest, spanning short / medium / long / has-comment-section.
- **Whether to port `eis_pipeline/PROMPTS.md` prompts directly or use the starter prompts in §6.** Recommendation: use §6 starters; treat `eis_pipeline/PROMPTS.md` as a *style reference* for tone, not a copy-source (different schema, different stages).
- **CLI flag surface.** Recommendation: mirror `eis_pipeline/run.py` (`--doc-key`, `--budget-usd`, `--only-fields`, `--skip-stages`, `--dry-run`, `--token-ledger`). Add `--manifest` to filter by the pre-built doc-key list.

---

## 9. Reference pointers

- **Canonical spec:** `inter_agent_plan.md`
- **Existing implementation (reference, do not modify):** `eis_pipeline/`
  - `pipeline/io_layer.py` — NUL fetch + flat-JSON loader
  - `pipeline/token_ledger.py` — port directly
  - `pipeline/config.py` — `AGENCY_VOCAB`, model IDs, regex patterns (selectively port)
  - `PROMPTS.md` — prompt style reference
- **Test writeups:** `GKG_Tests/TEST_RUN_*.md` — follow this format for v2 runs.
- **Model IDs** (locked):
  - Haiku: `claude-haiku-4-5-20251001`
  - Sonnet: `claude-sonnet-4-6`
  - Opus (retries only): `claude-opus-4-7`
- **Anthropic Batch API**: used for non-realtime stages (3c, 3e, 3g) to get the 50% discount. Realtime stages (3b regex fallback, 3f span-pick) call the standard API.
- **Cost target**: p50 < $0.15/doc, p90 < $0.35/doc, hard cap $1.00/doc with partial-output semantics. Critic adds ~$0.05–0.10/doc depending on stakeholder count; revised up from canonical plan's pre-critic numbers.

---

## 10. Recommended handoff shape (read this before starting)

The full build is ~10–11 working days. Do not attempt it in one session. The clean checkpoint shape:

1. **Handoff #1** — Build Group A + B. Run on `p1074_35556036099737`. Show Grace the Stage 2 JSON and the NUL-sourced fields. Stop and review before continuing.
2. **Handoff #2** — Build Group C + G (first test run). Show Grace the summary, alternatives, themes on the test doc. Stop and review.
3. **Handoff #3** — Build Group D. Run gazetteer setup, test location on the test doc. Stop and review.
4. **Handoff #4** — Build Group E + F. Final integration: stakeholders, validation (hard gates + self-consistency + Sonnet critic), output writer, CLI review. Multi-doc test run. Hand to Grace for production review.

If you start a fresh session, read this brief, identify the last completed group from `v1_build/output/`, and resume at the next group's acceptance gate.
