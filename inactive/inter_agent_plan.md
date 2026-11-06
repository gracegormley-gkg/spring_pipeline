# EIS Metadata Extraction Pipeline — Final Plan

> Source requirements: `eis_pipeline_goals_for__jake_plan.md`
> Converged through 4 rounds of adversarial review with GPT (Codex CLI). See `critiques/CHATGPT_HANDOFF_01_eis-metadata-pipeline/` for the full ledger.
> Verdict: APPROVED at round 4 with 5 non-blocking clarifications, all incorporated.

---

## Plan structure

This plan stages the build:

- **v1 — Thin Vertical Slice** (~3 weeks, 15 working days): the concrete pipeline we build first. USFS Final EISes (≥ 2000) only. Lower target accuracy; proves the pipeline end-to-end; produces v2-schema-compatible JSON with `deferred_v1` markers on un-tackled fields.
- **v2 — Target Architecture** (~6–8 weeks after v1): everything else the requirements ask for. Multi-agency corpus, full date/place/stance models, cross-doc entity registry, calibrated confidence, web review UI, layered gazetteer with feature-type dispatch.
- **v3 — Research Mode** (post-v2): active learning, reviewer-assist polygon proposal, automated drift detection.

The output JSON schema is **the v2 schema from day 1**. v1 simply populates fewer fields and marks the rest `status: "deferred_v1"`. This guarantees that v1-produced JSONs remain valid and queryable when v2 ships.

---

## 0. Output Schema (locked before any extraction starts)

One JSON per **logical EIS publication**. A logical publication is one of {Draft EIS, Final EIS, Supplemental EIS, ROD, NOI}, grouping its main volume + appendices + response_to_comments + errata + combined_rod components. Draft/Final/Supplemental/ROD are **separate** publications cross-linked via `cross_publication_links` (v2). Volumes within one publication are merged.

```jsonc
{
  "publication_id": "usfs_pals_12345_feis_2017",
  "publication_type": "Final EIS",                 // Draft|Final|Supplemental|ROD|NOI
  "is_supplemental": false,
  "physical_record_ids": ["P0491_35556036063543", "..."],
  "components": [
    { "record_id": "...", "role": "main|appendix|response_to_comments|errata|combined_rod", "confidence": 0.94 }
  ],
  "title":            { "value": "...", "status": "ok|needs_review|deferred_v1", "provenance": {...} },
  "summary":          { "value": "...", "status": "...", "provenance": {...} },
  "date": {
    "publication":    { "value": "2017-06-15", "precision": "day|month|year", "evidence": {...} },
    "filing":         { "value": null, "status": "deferred_v1" },
    "comment_deadline":{ "value": null, "status": "deferred_v1" },
    "rod_date":       { "value": null, "status": "deferred_v1" }
  },
  "agency": {
    "lead_agency":           { "value": "USFS", "status": "asserted_by_corpus_filter|extracted_from_mets|extracted_from_cover", "provenance": {...} },
    "cooperating_agencies":  [],
    "office_or_region":      { "value": null, "status": "..." }
  },
  "eis_type":         { "value": "Final EIS", "status": "..." },
  "alternatives":     [
    { "label": "No Action Alternative", "description": "...", "provenance": {...} }
  ],
  "themes": {
    "primary":    [{ "value": "transportation", "confidence": 0.92 }],
    "subthemes":  [{ "value": "highway", "confidence": 0.88, "parent": "transportation" }],
    "status":     "ok|skipped_summary_unavailable|needs_review|deferred_v1",
    "provenance": { "derived_from": "summary", "summary_status": "ok" }
  },
  "location": {
    "named_places": [
      {
        "name": "Tongass National Forest",
        "role": "project_area|agency_office|commenter_address|context_reference|alternative_site|comparison_site|mitigation_location",
        "source_dataset": "USFS Admin Forests",
        "source_feature_id": "...",
        "polygon": { /* GeoJSON Feature */ }
      }
    ],
    "project_area_polygon": null,                  // GeoJSON Feature or null; NEVER manufactured from hulls/buffers
    "context_polygons":     [],
    "source_geometry":      null,                  // Point|LineString|Polygon|MultiPolygon as originally found
    "geometry_role":        "site_specific|corridor|regional|programmatic|unknown",
    "geometry_status":      "named_feature|unknown",
    "polygon_uncertainty":  "low|medium|high|unknown",
    "spatial_summary": {
      "representative_point": [-134.5, 56.2],      // lon, lat
      "centroid_method":     "polygon_centroid|weighted_block_centroid|null",
      "centroid_status":     "valid_for_pin|misleading_for_pin|not_applicable",
      "bbox":                [[-135.0, 55.5], [-134.0, 56.8]]
    }
  },
  "stakeholders": [
    {
      "comment_block_id": "block_42",
      "comment_author": {
        "name": "Sierra Club",
        "type": "person|organization|tribal_government|coalition|agency|anonymous",
        "canonical_id": null,                       // cross-doc registry: v2
        "aliases": []
      },
      "represented_entity": null,
      "affiliation": null,
      "signatories": [],
      "authorship_role": "primary_author|co_signatory|spokesperson|hearing_speaker|form_letter_member",
      "stance_records": [
        {
          "stance":           "supportive|opposed|mixed|neutral",
          "stance_target":    { "type": "proposed_action|no_action|specific_alternative|mitigation|process|unknown", "reference_id": null },
          "stance_confidence": 0.86,
          "quote": {
            "text_raw":               "...",        // exact substring of OCR raw text
            "text_display":           "...",        // dehyphenated/normalized for reading
            "char_offset_raw":        [4821, 4912],
            "page":                   213,
            "section":                "public_comments",
            "source_text_hash":       "sha256:...",
            "normalization_rules_applied": ["soft_hyphen_dehyphenation", "unicode_quote_normalization"]
          },
          "sequence_order":   1
        }
      ]
    }
  ],
  "stakeholder_extraction_scope": "organizations_only|all",     // v1 = organizations_only
  "excluded_stakeholder_types":   ["person"],
  "stakeholder_status":           "ok|no_comment_section_found|partial_budget_cap|comment_response_split_failed",
  "historical_context":           { "value": null, "status": "deferred_v1" },
  "project_status":               { "value": null, "status": "deferred_v1" },
  "cross_publication_links":      { "value": [],   "status": "deferred_v1" },
  "extraction_budget_status":     "complete|partial_budget_cap|error",
  "coverage_estimate":            0.94,
  "unprocessed_sections":         [],
  "per_field_status":             { "stakeholders": "ok", "location": "ok", "..." : "..." },
  "validation": {
    "approach":                   "rule_threshold_v1|calibrated_v2",
    "field_level_confidence":     { "title": 0.99, "year": 0.98, "summary": 0.82, "...": "..." },
    "hard_gates":                 { "verbatim_quotes": "pass", "...": "pass" },
    "review_routing":             "auto_approve|partial_review|full_review",
    "routing_reasons":            []
  },
  "pipeline": {
    "pipeline_version":           "v1.0.0",
    "stage_versions":             { "stage1": "1.0.0", "stage2": "1.0.0", "...": "..." },
    "model_ids":                  { "haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-4-6", "...": "..." },
    "gazetteer_versions":         { "tiger": "2024", "usfs_admin_forests": "2024-07" },
    "calibration_model_id":       null,
    "extracted_at":               "2026-06-15T12:00:00Z"
  }
}
```

This schema is **stable** across v1/v2/v3. v1 simply populates fewer fields; deferred fields carry `status: "deferred_v1"` instead of being absent.

---

## 1. v1 — Thin Vertical Slice (3-week build)

### 1.1 Corpus filter (v1)

`eis_type == "Final EIS" AND is_supplemental == false AND lead_agency == "USFS" AND publication_year >= 2000`

Supplemental Final, Draft, standalone ROD, NOI, and ambiguous types are excluded from the v1 corpus; logged with `excluded_reason`. The `v1_corpus_manifest` is **frozen on Day 3** of the build — "100% of v1 corpus" is measured against the frozen manifest, not an open-ended live query.

### 1.2 v1 pipeline stages

**Stage 1 — Ingest & Assemble.**
- Manifest authority: S3 Inventory + Mongo (intersected). `docs_with_digits.json` is a *backup text source field*, not a manifest source. Records sourced only from backup get `provenance.source = "backup"` and `provenance.page = null`.
- Per-record output: raw `full_text` (single canonical, no high-conf variant; OCR confidence is per-page metadata, never an extraction filter), `text_normalized` with explicit char-offset alignment map back to raw, page index, METS fields, OCR confidence per page.
- OCR cleanup is conservative only: whitespace normalization, soft-hyphen dehyphenation across line breaks, Unicode quote normalization (only for the normalized variant; raw is preserved). No `0↔O`, no `1↔l` — those corrupt coordinates, dates, agency codes.

**Stage 1.5 — Document Grouping.**
- Within the v1 (Final EIS) corpus, group `physical_record_id`s into logical publications with component roles: `main`, `appendix`, `response_to_comments`, `errata`, `combined_rod`. Detection uses titles, METS series fields, filename patterns.
- Output: `{ publication_id, component_role, grouping_confidence, needs_grouping_review, evidence }`. Below-threshold groupings block downstream extraction for that publication; the publication enters a review queue.
- Gold labeling includes (record_id → publication_id, component_role) mappings.

**Stage 2 — Structural Section Detection.**
- Method: regex on header patterns first; embedding-similarity fallback against canonical section descriptors on docs where regex finds < N expected sections. Layout-aware detection (N1) is **conditional on Phase 0a OCR JSON audit** — if layout metadata exists, fold in; if not, drop.
- Taxonomy (full): `cover, summary, purpose_and_need, alternatives, affected_environment, environmental_consequences, public_comments, response_to_comments, rod`.
- Output: each section gets `{ pages: [start, end], char_span: [start, end], confidence }`. char_span is the primary unit; page ranges are derived.
- **Abstention is a hard gate**: if a downstream-required section is not found above threshold, dependent field is `status: "skipped_section_not_found"`. No fallback to broad-doc extraction.

**Stage 3a — Title / Year / Agency (METS-anchored).**
- Phase 0a measures METS exact-match on the 10-doc pilot. If exact-match ≥ 0.90 per field, METS is authoritative; otherwise that field gets `status: "needs_review"` and a regex/Haiku extractive cross-check runs.
- METS may be overridden ONLY by extractive evidence from cover/METS/TOC/title pages with explicit provenance. LLM may *classify* evidence, never invent a value. If METS and extractive evidence both exist and disagree → review.
- `lead_agency = "USFS"` in v1 carries `status: "asserted_by_corpus_filter"` unless independently extracted, to keep the shortcut honest.
- Agency-name canonicalization through a controlled vocabulary before any comparison.

**Stage 3b — EIS Type.**
- Regex on cover (`(Draft|Final|Supplemental)\s+Environmental Impact Statement`) → Haiku fallback on cover-page text if regex misses. Single-token output checked against {Draft, Final, Supplemental, ROD, NOI}.

**Stage 3c — Summary.**
- Haiku 4.5 on the detected Summary section (capped at 8K input tokens).
- Abstain if Summary section is `not_found`.
- Auto-check: 80–300 words; references the location and the action.

**Stage 3d — Location.**
- **Mention extraction**: GLiNER with EIS-typed entities (`national_forest`, `watershed`, `highway_corridor`, `place`, `tribe`) + Sonnet structured extraction on Cover/Summary/Purpose-and-Need/Alternatives sections only (NEVER public comments — that contaminates with commenter addresses).
- **Place-role tagging in v1**: only `project_area` and `commenter_address` (commenter_address mentions are excluded from spatial summary). Full taxonomy (alternative_site, comparison_site, mitigation_location, context_reference) is v2.
- **Spatial-qualifier parsing**: detect "parts of", "near", "within", "corridor", "between", "north of", "downstream from". When a qualifier is partial-or-relative, the named feature stays in `context_polygons` only — it does NOT promote to `project_area_polygon`.
- **Gazetteer (v1)**: feature-type-routed lookup.

  | Feature class | v1 source | Polygon? |
  |---|---|---|
  | County / state | Census TIGER/Line | Yes |
  | National Forest | USFS Admin Forests | Yes |
  | All other feature types | name-only (no polygon in v1) | No → `polygon: null` |

  Loaded into PostGIS or DuckDB Spatial for in-process lookup. No self-hosted Nominatim, no paid geocoder. Other gazetteers (PAD-US, NHD, NPS, BLM, USFWS, EPA FRS, GeoNames, OSM, Wikidata) ship in v2 with feature-type dispatch.

- **Polygon resolution (v1)**: named-feature lookup only. If a matched feature exists in TIGER or USFS Admin Forests, `project_area_polygon` is set from that boundary. **Otherwise `null`.** No hulls, no buffers, no LLM-reasoned polygons. v1 never manufactures geometry.
- **Doc classification**: `geometry_role` defaults to `unknown` in v1; full classification (`site_specific|corridor|regional|programmatic`) is v2.
- **GeoJSON schema (strict)**: WGS84 (EPSG:4326), MultiPolygon-normalized output, shapely `is_valid` + `is_simple` validated, plus a v1 semantic check (extracted state matches polygon's TIGER state membership).
- **External geometry search (v2 only)**: USFS PALS, BLM NEPA Register, EPA EIS database, state portals, METS-attached `.shp/.kml/.kmz` — all v2.

**Stage 3e — Alternatives.**
- Stage A: regex/heading detection inside the Alternatives section → labels (free).
- Stage B: Sonnet description for each label, input capped to the **first 500 tokens after the label** (≈ first paragraph). Hard token cap to control cost.

**Stage 3f — Stakeholders (organizations only in v1).**
1. **Comment-block detection** (structural, primary): parse comment headers (`COMMENT 42:`, etc.), letterheads, signature blocks, hearing transcripts with speaker labels, form-letter group headers (`Group 1: Form Letter from N respondents`). NER is **not** the primary entry point — it's a v2 fallback for blocks lacking structural headers.
2. **Comment/response split**: each block split into `comment_text` and `agency_response_text` using "Response:" / "Agency Response:" markers + indentation/formatting cues. If the split fails, block marked `needs_review` and excluded from quote extraction. **Quote selection only ever reads from `comment_text`** — quoting an agency response as stakeholder stance is a serious attribution error.
3. **Author canonicalization**: extract `{name, type, affiliation, authorship_role}` from the block header. **Persons excluded in v1** (`stakeholder_extraction_scope: "organizations_only"`); their comment blocks are skipped, not anonymized, to sidestep PII policy until v2. Output explicitly lists `excluded_stakeholder_types: ["person"]` so downstream consumers don't misread silence as absence.
4. **Stance with minimal target**: `stance_target ∈ {proposed_action, no_action, specific_alternative, mitigation, process, unknown}`. v1 doesn't resolve `specific_alternative` to an `alternative_id` (v2). Each commenter gets one stance record per target.
5. **Stance classification**: two-pass — Haiku classifies; if confidence low or "mixed", Sonnet re-classifies. Few-shot prompt with stance-target examples.
6. **Quote selection via span IDs**:
   - Sentence/block-split the `comment_text` using OCR JSON layout if available, otherwise paragraph-block split. Assign each span a stable `span_id`.
   - **Cue-based ranking (v1)**: score spans by presence of stance cues (`"oppose"`, `"support"`, `"do not support"`, `"object to"`, `"endorse"`, `"recommend against"`, etc.), absence of emotive-only cues (`"sincerely hope"`, `"appreciate the opportunity"` alone).
   - Send top-3 spans + the stance_target to Sonnet: "Pick the `span_id` whose text best represents this entity's stance on `<target>`. Do NOT copy the text." Sonnet returns a `span_id` only.
   - Code looks up the span by `span_id` and copies the exact raw text + char_offset_raw + page from Stage 1's alignment map. **The model never types the quote.**
   - NLI-based ranking (N2) is v2 — adds an NLI model that scores stance entailment per span window, with target-conditioned hypotheses. Defers because v2 also lifts the stance_target circular dependency by extracting candidate targets first.
7. **Verbatim verification**: substring check on raw OCR text using `char_offset_raw`. Hard gate. Failure (e.g. span_id from a different block) → mark `quote_status: "rejected_nonverbatim"`, drop the quote, log for review.

**Stage 3g — Themes.**
- Haiku 4.5 on `summary.value` + `title`. No raw sections; themes is fully downstream of Stage 3c and never re-reads the document.
- Returns **1–2 primary themes** from the controlled taxonomy, each with a `confidence` score, plus **0–3 subthemes** scoped to the chosen primaries.
- Controlled taxonomy (locked before extraction; expected to iterate after the first batch):
  - **13 primary themes**: `energy_infrastructure`, `transportation`, `land_management`, `water_resources`, `defense_and_military`, `urban_development`, `mining_and_extraction`, `agriculture_and_forestry`, `wildlife_and_habitat`, `cultural_heritage`, `waste_and_remediation`, `aerospace_and_space_exploration`, `other`.
  - **Subthemes** (starter list — per-primary, scoped; a subtheme is only valid under its declared `parent`):
    - `energy_infrastructure`: `nuclear_power`, `hydroelectric`, `wind`, `solar`, `oil_and_gas`, `electric_transmission`, `pipelines`
    - `transportation`: `highway`, `rail`, `transit`, `aviation`, `port_and_harbor`, `bridge`
    - `land_management`: `public_lands_planning`, `grazing`, `recreation`, `timber`, `fire_management`, `designation`
    - `water_resources`: `dam_and_reservoir`, `water_supply`, `irrigation`, `flood_control`, `wetlands`, `watershed_restoration`
    - `defense_and_military`: `base_realignment`, `training_range`, `weapons_testing`, `munitions`
    - `urban_development`: `housing`, `redevelopment`, `commercial`, `mixed_use`
    - `mining_and_extraction`: `coal`, `hardrock_mining`, `oil_and_gas_leasing`, `sand_and_gravel`
    - `agriculture_and_forestry`: `forest_plan`, `timber_harvest`, `rangeland`, `pest_management`
    - `wildlife_and_habitat`: `endangered_species`, `habitat_restoration`, `fisheries`, `predator_management`
    - `cultural_heritage`: `historic_preservation`, `archaeological`, `tribal_sacred_sites`
    - `waste_and_remediation`: `superfund`, `landfill`, `hazardous_waste`, `cleanup`
    - `aerospace_and_space_exploration`: `launch_facility`, `satellite`, `crewed_mission`, `research_program`, `planetary_science`
    - `other`: no subthemes; selecting `other` flags the doc for taxonomy-gap review at the batch level.
- **Hard gate** (deterministic, no LLM): every emitted `value` must be in the controlled vocabulary; every subtheme's `parent` must match a primary in the same record. Out-of-vocab values or orphaned subthemes are dropped and the field's `status` is set to `needs_review`.
- **Abstention**: if `summary.status != ok` (Stage 3c skipped or downgraded), themes is set to `status: skipped_summary_unavailable` and no extraction is attempted. Themes never invents content the summary couldn't ground.
- **Batch-level signal**: if `other` exceeds 30% of primary selections across a run's frozen-manifest batch, the run report flags a taxonomy-gap warning. Per-doc `other` is allowed and not a per-field gate.
- **Prompt instruction**: the prompt explicitly tells the model not to default to `other` — it must pick the closest specific theme and use `confidence` to express uncertainty, reserving `other` only for documents that genuinely do not fit any specific theme.

**Stage 4 — Validation (rule-threshold field-level in v1).**
- **Hard gates** (binary, per field):
  - Verbatim substring check on every quote: must pass.
  - METS-extractive disagreement on title/year/agency: must agree (after canonicalization) or field = `needs_review`.
- **Field-level confidence**: rule thresholds per field, NOT calibrated logistic regression (deferred to v2 where label counts support it). Per-field threshold determined **before** evaluating the gold set — no tuning on the eval set.
- **Field-level review routing**: each below-threshold field routed individually. Doc-level "auto-approve" = all fields above their thresholds. Below-threshold field doesn't tank the rest.

**Stage 5 — Output.**
- One JSON per **logical publication** to `s3://nu-impulse-production/pipeline/v1/final/{publication_id}.json`.
- Append to `s3://.../pipeline/v1/index/extracted.parquet` (DuckDB-readable) for queries. PostGIS deferred to v2 (only required when polygon queries dominate).
- Stage outputs cached at `s3://.../pipeline/v1/stage{N}/{publication_id}.json` with versioned cache key: `sha256(input_text_hash, prompt_version, model_id, model_params_hash, gazetteer_version, stage_version, pipeline_version)`.

**Stage 6 — CLI Review Tool (v1).**
- Lightweight Python CLI:
  - For each below-threshold field, surfaces: field name + extracted value + OCR text window (char_offset_raw ± 200 chars) + page number + status + suggested correction prompt.
  - Annotator picks accept/reject/correct; correction recorded to `s3://.../pipeline/v1/review_log/{publication_id}_{field}_{timestamp}.json`.
- Review corrections go to a **production review log**, NOT to the eval gold set. Three label pools (locked: training, production_corrections, eval) are maintained from day one.
- Image-based review UI deferred to v2 — pending Phase 0a confirmation of whether page images are actually present in S3 (data bundle only lists TXT/JSON/CONFIDENCES/METS; images are not guaranteed).

### 1.3 v1 cost model

Cost is reported as **measurement, not target.** Days 14–15 deliverable includes measured `p50` and `p90` cost across the 10-doc pilot, broken out by stage. The README ships with the actual measured number.

Cost formula (calibrated on 5 real Final EISes in Phase 0):
```
cost(doc) = base + α·n_blocks + β·n_candidate_stakeholders + γ·n_stance_records + δ·n_quote_spans + retry_overhead
```
Two parallel budgets:
- **Undiscounted worst-case** (no Batch API) — drives capacity planning and per-doc hard caps.
- **Batch-discounted expected case** — what we actually spend; uses Anthropic Batch API for non-realtime stages (50% discount).

Per-doc hard cap: **$1.00**. On cap, partial output with:
```json
"extraction_budget_status": "partial_budget_cap",
"coverage_estimate": 0.73,
"unprocessed_sections": ["public_comments_appendix_B"],
"per_field_status": { "stakeholders": "partial_budget_cap" }
```

Pre-build cost expectation (subject to measurement):
- p50 < $0.10/doc
- p90 < $0.25/doc
- p99 < $0.50/doc
- hard cap $1.00/doc with partial-output semantics

### 1.4 v1 gold set & success criteria

- 10-doc adjudicated USFS-Final-2000+ gold set, hand-labeled across Phase 0a.
- All metrics include CI lower bound; success requires CI lower bound to clear threshold, not point estimate.
- **Field-level v1 thresholds** (deliberately moderate; v2 ratchets):
  - Title (METS): exact match ≥ 0.90 (CI LB).
  - Year (METS or extractive): exact match ≥ 0.85.
  - EIS Type: ≥ 0.95.
  - Summary: human eval ≥ 0.70 "captures issue + scope + decision".
  - Themes (primary): precision ≥ 0.85, recall ≥ 0.80; abstention correctness ≥ 0.90. Subthemes not formally eval-gated in v1 (used to validate taxonomy coverage; precision/recall added in v2).
  - Location centroid (when polygon exists): within 25 km of labeled centroid ≥ 0.80.
  - Alternatives labels: precision ≥ 0.85, recall ≥ 0.75.
  - Stakeholders (organizations): commenter recall ≥ 0.75; quote verbatim 1.00 (hard gate); stance agreement ≥ 0.75; abstention correctness ≥ 0.90.
- Three label pools maintained: training/calibration, production_corrections, locked stratified eval (refreshed only via scheduled labeling rounds).

### 1.5 v1 acceptance criteria

- 100% of records in the **frozen v1_corpus_manifest** produce a valid JSON with all required fields populated (value or status).
- Verbatim quote check: 100% pass.
- METS field accuracy meets Phase-0a-measured threshold.
- Section-abstention rates documented per doc.
- Field-by-field 10-doc gold report committed alongside v1 shipping; thresholds locked before evaluation.

### 1.6 v1 build order (15 working days)

| Days | Phase 0 / Stage | Output |
|---|---|---|
| D1–D3 | Phase 0a: corpus audit (image availability, OCR JSON structure, METS reliability spot-check on 20 USFS Final EISes from 2000+); **freeze v1_corpus_manifest on Day 3** | Audit report; manifest |
| D4–D5 | Schemas locked. 10-doc gold labeling begins (in parallel with build) | Locked schema; labeling kickoff |
| D6–D8 | Stages 1, 1.5, 2 (with abstention) | Ingest + grouping + sections working end-to-end |
| D9–D10 | Stages 3a, 3b, 3c, 3e, 3g | Trusted fields, type, summary, alternatives, themes |
| D11–D13 | Stage 3d (TIGER+USFS gazetteer, named-feature polygon only) and Stage 3f (structural comment-block parser, comment/response split, cue-ranked span-ID quote selection) | Location + stakeholders |
| D14–D15 | Stage 4 (rule thresholds), Stage 5 (output), Stage 6 (CLI review tool); measure p50/p90 cost; field-by-field eval against 10-doc gold; commit results | v1 ship |

### 1.7 Explicitly deferred from v1 to v2

- PAD-US, NHD, NPS, BLM, USFWS, EPA FRS, BIA, state-portal gazetteers
- Self-hosted Nominatim; paid geocoders
- Derived polygons (hulls/buffers); LLM-reasoned polygons (N3) — never in production, becomes reviewer-assist tool only in v3
- Full place-role taxonomy beyond `project_area`/`commenter_address`
- NLI quote ranking with target-conditioned hypotheses (N2)
- Stance_target `reference_id` resolution to `alternative_id`
- Cross-doc entity canonical registry; entity resolution with blocking keys and type-specific non-merge rules
- Calibrated field confidence (logistic / isotonic)
- Full date_type model (filing, comment_deadline, ROD date)
- Person stakeholders (with PII redaction policy)
- Draft/Supplemental/ROD as their own publications; cross-publication links
- Web review UI; image-based review (gated on Phase 0a image-availability)
- Layout-aware section detection (N1) — conditional inclusion in v1 if Phase 0a finds layout metadata, else v2

---

## 2. v2 — Target Architecture (post-v1, ~6–8 weeks)

v2 expands corpus and lifts every v1 deferral. Major workstreams:

### 2.1 Corpus expansion
- All EIS types (Draft, Final, Supplemental, ROD, NOI) as separate publications.
- Multi-agency: USFS → BLM, NPS, USFWS, FHWA, Army Corps, BIA, etc.
- Pre-2000 corpus (lower OCR quality; Phase 0 measures unprocessable fraction).
- Cross-publication linking via `cross_publication_links[]` (Draft → Final → ROD).

### 2.2 Location stack (full)
- Layered gazetteer with **feature-type dispatch**:

  | Feature class | Primary source | Polygon? |
  |---|---|---|
  | County / municipality / CDP | Census TIGER/Line | Yes |
  | State | Census TIGER/Line | Yes |
  | National Forest (district detail) | USFS Admin Forests | Yes |
  | National Park / Monument / Wilderness | NPS / PAD-US | Yes |
  | BLM lands / districts / field offices | BLM | Yes |
  | NWR / refuge | USFWS | Yes |
  | Watershed / HUC | USGS WBD | Yes |
  | River / stream | USGS NHD | LineString (buffer if needed) |
  | EPA facility / Superfund site | EPA FRS / FACTS | Point |
  | Highway / road corridor | OSM + state DOT | LineString (buffer if needed) |
  | Tribal lands | BIA / Census AIANNH | Yes |
  | International / generic | GeoNames / OSM | Mixed |
  | Generic place name | USGS GNIS | Point only (name + disambiguation, NOT polygon) |

- Self-hosted Nominatim (Docker) + Pelias as backup; paid geocoder (Google/Mapbox) for hard ambiguous cases (<5% of mentions).
- Full place-role taxonomy: `project_area, agency_office, commenter_address, context_reference, alternative_site, comparison_site, mitigation_location`.
- Doc classification: `geometry_role ∈ {site_specific, corridor, regional, programmatic, unknown}` BEFORE polygon resolution.
- Spatial-qualifier parsing fully wired into polygon decision (partial qualifiers → context_polygons only).
- External geometry search: USFS PALS, BLM NEPA Register, EPA EIS database, state portals, METS-attached `.shp/.kml/.kmz`.
- Allow Point/LineString source_geometry with optional derived buffer polygon (`derivation_method` recorded).
- Semantic polygon validation (extracted state matches polygon's TIGER membership; county containment when applicable; centroid within `R` km of geocoded mentions, `R` tuned by geometry_role).
- `spatial_summary` with `representative_point`, `centroid_method`, `centroid_status`, `bbox`.

### 2.3 Stakeholder stack (full)
- NLI-ranked quote selection (N2) with target-first ordering:
  1. Detect comment block, get author.
  2. Extract candidate stance targets (cue-based or small LLM) — lifts NLI's circular dependency.
  3. For each (block, target), run NLI over rolling span windows (3-span context window).
  4. Top-NLI spans → Sonnet picks `span_id` from top-3.
  5. Code copies raw text.
- Cue-based fallback runs in parallel; LLM adjudicates disagreement.
- Cross-doc entity registry with:
  - Authorship data model: `comment_author`, `represented_entity`, `affiliation`, `signatories[]`, `authorship_role`.
  - Blocking keys (org_type + state + first 5 chars of canonical name).
  - Type-specific thresholds (agencies strict, NGOs medium, individuals very high — false-merge cost > false-split cost).
  - Geography-aware (regional "Friends of X" not merged across states without explicit cross-reference).
  - `possible_duplicate` review queue (never auto-merge borderline cases).
- Stance_target resolution: `specific_alternative` resolved to `alternative_id` extracted from the same publication.
- Person stakeholders with PII policy:
  - Persons stored separately from organizations.
  - For persons appearing only as comment authors: name retained; email/phone/street address never stored (PII redaction pass).
  - `person.privacy_status: "public_official | identified_in_comment | redacted_anonymous"`.
  - Default index access excludes persons; explicit role flag required.

### 2.4 Validation stack (full)
- Calibrated field-level confidence (logistic or isotonic) for fields with abundant signal (title, year, agency, type).
- Rule thresholds + abstention curves for sparse-signal fields (polygon role, stance target, attribution) until n ≥ 500 labels.
- Versioned calibration models (`calibration_model_id`, `calibrated_on_dataset_id`, `calibrated_at`). Mandatory recalibration trigger after any change to prompts/model IDs/OCR normalization/gazetteer/grouping. Stale → spot audit blocks deploy.

### 2.5 Review UI (full)
- Web app: failing fields + evidence snippets + page images (if available, per Phase 0a finding) or OCR text + suggested-correction form.
- SLA: each reviewed doc cycles back to validation within 24h.
- Corrections feed into `production_corrections` pool; locked eval set never receives production-review additions.

### 2.6 v2 schema additions
- Full `date` model populated.
- `cross_publication_links[]` populated.
- `historical_context`, `project_status` evaluated (require external knowledge sources; out-of-band feasibility study before commitment).

---

## 3. v3 — Research Mode (post-v2)

- **Reviewer-assist polygon proposal (N3-as-tool)**: when no human-drawn polygon exists, an LLM (with named features + optional satellite tiles) proposes geometry for human approval. Never enters production index without human approval. Marked `polygon_source: "llm_reasoned_human_approved"`, `polygon_uncertainty: "high"`.
- **Active learning loop**: low-confidence predictions surfaced to reviewers preferentially; feedback retrains stance classifier on ≥ 200 labeled examples (lifts stance from few-shot to fine-tuned DistilBERT).
- **Automated drift detection**: monitor per-field accuracy week-over-week; alert on regression > 2 pp.
- **Cross-agency coverage targets**: stratified eval set expands to ≥ 500 docs.

---

## 4. Cross-cutting concerns (apply to all versions)

### 4.1 Orchestration
Plain Python + idempotent stage artifacts (each stage writes a versioned JSON keyed by publication_id). No Prefect/Airflow until scale demands. The idempotent artifact pattern is the actual win; orchestrators are bookkeeping.

### 4.2 Caching
Cache key: `sha256(record_id, input_text_hash, prompt_version, model_id, model_params_hash, gazetteer_version, stage_version, pipeline_version)`. All components recorded in the output for auditability. Bump any component to force re-run.

### 4.3 Model routing
- Local models (GLiNER, embeddings, NLI in v2) for high-volume cheap tasks.
- Haiku 4.5 (~$1/M in, ~$5/M out) for cheap LLM tasks (type, summary, candidate-sentence ranking).
- Sonnet 4.6 (~$3/M in, ~$15/M out) for harder tasks (stance, location disambiguation, attribution).
- Opus 4.7 (~$5/M in, ~$25/M out) reserved for failed-validation retries on known-hard docs.
- Anthropic Batch API for non-realtime stages (50% discount on supported calls).

### 4.4 Provenance
Every extracted value carries `{page, char_offset_raw, section, source: "mets|ocr|backup", source_text_hash}`. Quotes additionally carry `text_raw`, `text_display`, `normalization_rules_applied`. Lets reviewers jump to evidence and re-verify against arbitrary future OCR re-runs.

### 4.5 Storage layout
```
s3://nu-impulse-production/
├── raw/                                       (existing)
│   └── P{id}/{TXT,JSON,CONFIDENCES,mets.xml}
├── pipeline/v1/
│   ├── manifest/v1_corpus_manifest.parquet    (frozen Day 3)
│   ├── stage1_assembled/{publication_id}.json
│   ├── stage1_5_grouped/{publication_id}.json
│   ├── stage2_sections/{publication_id}.json
│   ├── stage3_fields/{publication_id}.json
│   ├── stage4_validated/{publication_id}.json
│   ├── final/{publication_id}.json
│   ├── review_log/{publication_id}_{field}_{ts}.json
│   └── index/extracted.parquet
└── gold/
    ├── pilot_10/                              (v1 eval set, locked)
    ├── pool_training/
    └── pool_production_corrections/
```

### 4.6 Three label pools (locked structurally)
- `training_calibration` — annotated labels + adjudicated production corrections; used for model fits and rule-threshold setting.
- `production_corrections` — raw correction log; used for drift monitoring; biased toward failures.
- `eval` — locked, stratified, NEVER receives production additions. Refreshed only via scheduled labeling rounds with adjudication. v1 = 10-doc pilot; v2 expands to 200; v3 expands further.

### 4.7 PHASE 0 (the work that gates v1 starting)

Cannot be one day. The build days quoted above assume Phase 0 is in progress; the corpus audit gates start-of-build.

| Step | Output |
|---|---|
| Corpus audit (image availability, OCR JSON structure, METS reliability spot-check on 20 USFS Final EISes ≥ 2000) | Audit report |
| Labeling schema draft (definitions, edge cases, adjudication rules; covers grouping + every required field) | Schema doc |
| 10-doc pilot with 2 annotators + adjudication | Adjudicated v1 gold set |
| Schema refinement | Locked schema |
| (Parallel, ongoing) 200-doc stratified labeling | v2 gold set (post-v1) |

---

## 5. Concrete Decisions Table (converged)

| Decision area | v1 pick | v2 pick |
|---|---|---|
| Manifest authority | S3 Inventory + Mongo (intersected) | same |
| Backup text role | populated text field with `provenance.source=backup`, page=null | same |
| Page assembly | single `full_text` raw + alignment-mapped normalized; OCR confidence is metadata, never a filter | same |
| OCR cleanup | whitespace + soft-hyphen dehyphenation + Unicode quote norm only (normalized variant only) | same; layout-aware extensions |
| Document grouping | within-Final-EIS components (main/appendix/RtC/errata/combined_rod) | + cross-publication links across Draft→Final→ROD |
| Section detection | regex → embedding-similarity fallback; abstention hard gate; layout-aware (N1) conditional on Phase 0a audit | + TOC reconciliation with printed↔physical page mapping |
| Section taxonomy | Full CEQ-aligned (cover, summary, P&N, alternatives, AE, EC, comments, RtC, ROD) | same |
| Title/Year/Agency | METS authoritative iff Phase-0a-measured exact-match ≥ 0.90; otherwise needs_review + regex/Haiku cross-check; override only by extractive evidence (never by LLM judgment) | same + calibrated confidence |
| Date model | publication date + month/day when extractable; others `deferred_v1` | full {publication, filing, comment_deadline, rod_date, precision, date_type, evidence} |
| Agency model | lead_agency (USFS = asserted_by_corpus_filter); cooperating_agencies from cover; office_or_region deferred | full {lead, cooperating[], office/region} with controlled vocab and aliases |
| EIS type | regex → Haiku fallback on cover | same |
| Summary | Haiku on Summary section, capped 8K tokens; abstain on section_not_found | same + calibrated confidence |
| Alternatives | regex labels + Sonnet description capped to first 500 tokens after label | same |
| Themes | Haiku on `summary.value` + `title`; 13-item primary vocab + per-primary subthemes; controlled-vocab hard gate; abstain on `summary_unavailable`; batch-level `other`-rate warning | + calibrated confidence; subtheme precision/recall in gold; scheduled taxonomy-expansion review |
| Location — mention extraction | GLiNER (EIS-typed) + Sonnet, on Cover/Summary/P&N/Alternatives only (NEVER public comments) | + full place-role taxonomy |
| Location — gazetteer | feature-type-routed: TIGER (state/county), USFS Admin Forests. Other features: name-only, no polygon. | full layered gazetteer with feature-type dispatch (TIGER, USFS, PAD-US, NPS, BLM, USFWS, NHD/WBD, EPA FRS, BIA, GeoNames, OSM, Wikidata) + self-hosted Nominatim + paid escalation |
| Polygon resolution | named-feature lookup only; null if not found; NO hulls, NO buffers, NO LLM-reasoned | named-feature lookup + spatial-qualifier-aware promotion + Point/LineString source geometry with optional derived buffer + external geometry search (PALS/NEPA Register/state portals/METS-attached) |
| Place roles | project_area + commenter_address only (commenter_address excluded from spatial summary) | full taxonomy |
| Doc geometry classification | `unknown` default | `site_specific|corridor|regional|programmatic|unknown` BEFORE polygon resolution |
| GeoJSON schema | strict WGS84 / MultiPolygon-normalized / shapely-validated / semantic state-containment check | same + full semantic checks (county containment, distance-to-mentions by geometry_role) |
| Stakeholder NER | structural comment-block parser PRIMARY; NER not used in v1 | + NER fallback for blocks lacking headers |
| Comment/response split | required; quote selection only from `comment_text` | same |
| Stakeholder scope | organizations only (`stakeholder_extraction_scope: organizations_only`, `excluded_stakeholder_types: ["person"]`) | + persons with PII redaction policy (`person.privacy_status`) |
| Entity resolution | per-doc only | cross-doc registry with blocking keys, type-specific thresholds, geography-aware, possible_duplicate queue |
| Stance | two-pass Haiku → Sonnet escalation | + NLI feature in ranking |
| Stance target | minimal taxonomy {proposed_action, no_action, specific_alternative, mitigation, process, unknown}; no reference_id resolution | + `reference_id` resolved to `alternative_id` |
| Quote selection | layout-aware span split → cue-based ranking → Sonnet picks `span_id` → code copies raw text → verbatim hard gate | + NLI ranking with target-conditioned hypotheses on rolling block windows |
| Quote schema | `{text_raw, text_display, char_offset_raw, page, section, source_text_hash, normalization_rules_applied}` | same |
| Validation | rule-threshold field-level (thresholds locked BEFORE eval); per-field hard gates; field-level routing | calibrated field-level confidence (versioned, recalibration on any pipeline change) |
| Review UI | CLI tool with OCR text + char-window highlight; corrections to `production_corrections` log | web app with images-if-available; SLA 24h |
| Cost reporting | measured p50/p90 on 10-doc pilot; pre-build expectation p50 < $0.10, p90 < $0.25, hard cap $1.00 | measured continuously; per-stage formula calibrated on real data |
| Budget cap behavior | `extraction_budget_status`, `coverage_estimate`, `unprocessed_sections[]`, per-field statuses | same |
| Orchestration | Plain Python + idempotent stage artifacts | same until scale demands |
| Caching key | `sha256(record_id, input_text_hash, prompt_version, model_id, model_params_hash, gazetteer_version, stage_version, pipeline_version)` | same |
| Index | DuckDB over Parquet | + PostGIS view if polygon queries become primary |
| Label pools | training_calibration / production_corrections / locked eval (10-doc) | + scheduled eval expansion to 200+ |
| Phase 0 | corpus audit + schema + 10-doc adjudicated pilot before build | + 200-doc stratified expansion ongoing |

---

## 6. Open Risks (after convergence)

These remain risks that monitoring should track; they're not blockers but they shape what we measure:

1. **Phase 0a finding may force v1 scope changes.** If METS reliability < 0.90 on the 10-doc pilot for title/year/agency, the "METS-authoritative" shortcut breaks and we add cover-page extractive cross-check earlier than planned. If OCR JSON lacks layout metadata, N1 is dropped from v1 entirely. If page images don't exist in S3, review UI in v1 is text-only.
2. **Worst-case cost on pathological comment volumes is unknown until measured.** 1000-page comment appendices on Final EISes could push beyond p99=$0.50; hard cap behavior must work correctly when triggered.
3. **Polygon coverage in v1 is intentionally narrow** (TIGER + USFS only). For Final EISes whose project area isn't a national forest or county, `project_area_polygon` will be null. Downstream consumers must handle that as "polygon coverage is v2," not "no project area."
4. **Stance target inference quality** is the main open question for v2. Cue-based target extraction works for simple comments; multi-target commenters need the NLI-with-targets approach.
5. **PII boundary on persons is ducked in v1.** v2 implementation requires explicit policy review before persons are extracted at all.

---

## 7. Pointer to the critique record

Full adversarial review log: `critiques/CHATGPT_HANDOFF_01_eis-metadata-pipeline/`
- R1: 35 issues raised, all accepted.
- R2: 28 issues raised, all accepted.
- R3: 14 issues raised, all accepted; v1/v2/v3 staging introduced.
- R4: VERDICT: APPROVED with 5 non-blocking clarifications, all incorporated above (frozen manifest, project location source restriction, USFS agency status field, threshold-locking discipline, review-log pool separation).
