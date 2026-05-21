# Merged EIS Pipeline — Output Description

This document describes what the `merged_pipeline` produces when you run it on a single
document key (e.g. `python run.py --doc-key p1074_35556035057348 --output out.json`).

## Output artifacts

A run writes up to three things to disk:

1. **EISRecord JSON** — one structured JSON file per `--doc-key`, written to the
   `--output` path. This is the primary product of the pipeline.
2. **`output/token_ledger.json`** — appended after each LLM-enabled run with
   per-run and lifetime token/cost totals (skipped when `--no-ledger` or `--no-llm`).
3. **`output/nul_cache/`** — cached responses from the Northwestern University Library
   (NUL) catalog API, populated transparently as documents are ingested.

The pipeline always writes a *valid* `EISRecord` JSON, even on degraded runs
(missing LLM, budget cap, missing sections). Fields that couldn't be populated
carry an explicit status code rather than being omitted.

## EISRecord JSON — top-level shape

The schema is locked and defined in `pipeline/schema.py:571` (`EISRecord`).
A populated record contains the following top-level groups:

### Identity & components
- `publication_id` — stable doc key
- `publication_type` — one of `Draft EIS | Final EIS | Supplemental EIS | ROD | NOI`
- `is_supplemental` — bool
- `physical_record_ids` — list of underlying record IDs
- `components[]` — each with `record_id`, `role` (`main | appendix |
  response_to_comments | errata | combined_rod`), and `confidence`

### Bibliographic METS fields (Stage 3a)
- `title` — `FieldWithStatus[str]`
- `summary` — `FieldWithStatus[str]` (LLM-generated narrative summary)
- `layman_summary` — `FieldWithStatus[str]` (plain-language version)
- `date.publication` — `{value, precision: day|month|year, status, provenance, evidence}`
- `date.filing | comment_deadline | rod_date` — deferred to v2
- `year` — `FieldWithStatus[int]`
- `agency.lead_agency` — `FieldWithStatus[str]`
- `agency.cooperating_agencies[]` — list of `{value, status, provenance}`
- `agency.office_or_region` — `FieldWithStatus[str]`

### EIS classification (Stage 3b)
- `eis_type` — `FieldWithStatus[Draft|Final|Supplemental|ROD|NOI|Unlabelled]`

### Sections (Stage 2)
- `sections[]` — one `SectionRecord` per canonical section name
  (`cover | summary | purpose_and_need | alternatives | affected_environment |
  environmental_consequences | public_comments | response_to_comments | rod`)
  with `char_span`, `pages`, `confidence`, `status`, and
  `detection_method` (`regex | ai_toc | embedding_fallback | default_pages | manual`).

### Comment structure (Stage 3f)
- `comment_blocks[]` — each with `block_id`, `char_span_raw`, `pages`,
  `comment_text_span`, `agency_response_span`, `split_status`, and a list of
  `spans[]` (sentence-level `SpanRecord` entries used for span-ID quote selection).

### Substantive content
- `alternatives[]` — `{label, description, provenance, status}` (Stage 3e)
- `themes.primary[]` and `themes.subthemes[]` — controlled-vocab themes with
  confidence scores; subthemes carry a `parent` reference (Stage 3g)
- `location` (Stage 3d):
  - `named_places[]` — `{name, role, feature_type, source_dataset, polygon}`
  - `project_area_polygon`, `context_polygons[]` — GeoJSON Features
  - `source_geometry` (Point/LineString/Polygon/MultiPolygon),
    `geometry_role`, `geometry_status`, `polygon_uncertainty`
  - `spatial_summary` — representative point, centroid method/status, bbox

### Stakeholders (Stage 3f)
- `stakeholders[]` — each with:
  - `comment_block_id` (must reference a block in `comment_blocks`)
  - `comment_author` — `{name, type, canonical_id, aliases}`, where `type` is
    `person | organization | tribal_government | coalition | agency | anonymous`
  - `represented_entity`, `affiliation`, `signatories[]`, `authorship_role`
  - `stance_records[]` — each with `stance` (`supportive | opposed | mixed |
    neutral`), `stance_target`, `stance_confidence`, optional verbatim `quote`
    (with `text_raw`, `text_display`, `char_offset_raw`, `page`, `section`,
    `source_text_hash`, `span_id`)
  - `appearance_order`
- `stakeholder_extraction_scope` — defaults to `organizations_only` in v1
- `excluded_stakeholder_types`, `stakeholder_status`

### Validation (Stage 4 — tiered critic)
- `validation.approach` — `rule_threshold_v1` in v1
- `validation.verdicts[]` — `CriticVerdict` entries with
  `field_path`, `tier` (`haiku | sonnet | opus | skipped_budget_cap`),
  `verdict` (`pass | partial | no | skipped`), `evidence_quote`, `reasoning`
- `validation.critic_pass_rate`, `field_level_confidence`
- `validation.hard_gates` — `verbatim_quotes`, `year_range`, `theme_vocab`,
  `geocoding_centroid`, `schema_validation` (each `pass | fail | not_run`)
- `validation.review_routing` — `auto_approve | partial_review | full_review`
- `validation.routing_reasons[]`

### Budget & coverage
- `extraction_budget_status` — `complete | partial_budget_cap | error`
- `coverage_estimate` — fraction of expected fields populated
- `unprocessed_sections[]`, `per_field_status{}`

### Pipeline metadata
- `pipeline.pipeline_version`, `stage_versions{}`, `model_ids{}`,
  `gazetteer_versions{}`, `calibration_model_id`
- `pipeline.extracted_at` — ISO-8601 UTC
- `pipeline.total_tokens`, `total_cost_usd`, `duration_seconds`
- `pipeline.warnings[]` — human-readable warnings collected across stages

### Deferred to v2 (always present, always `deferred_v1`)
- `historical_context`, `project_status`, `cross_publication_links`

## Provenance & status conventions

Every populated field carries a `Provenance` object indicating the source:

- External / catalog: `nul_api`, `mets_xml`, `docs_with_digits_json`
- Deterministic: `regex`, `fuzzy_match`, `controlled_vocab_match`
- ML: `embedding_fallback`, `gliner`
- NER: `spacy_person`, `spacy_org`, `dict_agency`, `dict_tribe`, `dict_ngo`
- LLM tiers: `haiku_classifier`, `haiku_gapfill`, `sonnet_extractor`, `opus_extractor`
- Retrieval fallback: `fallback_keyword_search`, `fallback_first_n_chunks`

When a field can't be filled, its `status` reflects why
(see `pipeline/schema.py:11`):

- `ok` — populated and verified
- `needs_review` — populated but failed a check
- `deferred_v1` — explicitly out-of-scope for v1
- `skipped_section_not_found` — source section missing
- `skipped_summary_unavailable` — themes-style abstention (downstream of summary)
- `skipped_no_llm` — producer required the LLM and got `--no-llm`
- `partial_grounding` — critic returned `partial`
- `partial_budget_cap` / `skipped_budget_cap` — budget cap effects
- `extracted_from_mets` / `extracted_from_cover` — populated from external/cover
- `fallback_keyword_search` / `fallback_first_n_chunks` — degraded retrieval path
- `rejected_nonverbatim` — quote failed the substring gate
- `comment_response_split_failed`, `no_comment_section_found` — Stage 3f abstentions

## Token ledger (`output/token_ledger.json`)

Appended after every LLM-enabled run by `pipeline/token_ledger.py:25`. Shape:

```jsonc
{
  "runs": [
    {
      "timestamp": "...",
      "doc_id": "...",
      "pipeline_version": "v1.0.0",
      "duration_seconds": 13.67,
      "total_input_tokens": 0,
      "total_output_tokens": 0,
      "total_tokens": 0,
      "total_cost_usd": 0.0,
      "calls": 0,
      "by_model": { "<model_id>": { "input_tokens", "output_tokens", "calls", "cost_usd" } }
    }
  ],
  "lifetime_totals": { /* recomputed on every write */ }
}
```

## Run-mode effects on output

| Mode | Effect on the JSON |
| --- | --- |
| `--no-llm` | LLM-dependent fields degrade to `needs_review` or stage-specific abstention statuses; record is still valid and complete. |
| `--dry-run` | Same shape as a normal LLM run, but no API calls are made (prompts are logged). |
| `--budget-usd N` mid-run trip | Pipeline catches `BudgetExceededError`, flips `extraction_budget_status` to `partial_budget_cap`, and writes whatever has been populated; remaining fields stay at schema defaults. |
| `--no-ledger` | Skips appending to `output/token_ledger.json`. |

## Example one-line summary

After a successful run, `run.py` prints:

```
OK doc=<key> routing=<auto_approve|partial_review|full_review> warnings=<n> cost=$X.XXXX duration=Y.Ys
```

A real example output is in `output/smoke_test.json` (a `--no-llm` run on
`p1074_35556035057348`, which produced `title`, `eis_type`, `year`, and
`lead_agency` deterministically while degrading `summary`, `themes`,
`location`, and `stakeholders`).
