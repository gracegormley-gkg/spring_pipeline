# segment_a — output layout

This directory contains the artifacts from running the **segment_a** extraction pipeline
over a sampled batch of EIS documents. Each doc flows through two extraction stages
(**M1** → **M2**), then a **Critic** review, and finally a **grading sheet** for human QA.

```
output/
├── run.log              # Chronological log for the whole batch run
├── run_summary.json     # Per-doc paths + LLM usage/cost roll-up
├── selection.json       # How the batch was sampled (seed, buckets, targets)
├── m1/<doc_id>.json     # Stage 1 outputs (one file per doc)
├── m2/<doc_id>.json     # Stage 2 outputs (one file per doc)
├── critic/<doc_id>.json # Rubric-based review of M1+M2 fields
└── grading_sheets/<doc_id>.csv  # Flat CSV for human grading
```

## M1 — fast metadata

Cheap, high-precision extraction of the four "always-known" bibliographic fields.
Populated from a combination of NUL catalog metadata, first-page regex, and a short
Sonnet pass over the first ~2 pages.

Fields per file:

| field         | notes                                                          |
|---------------|----------------------------------------------------------------|
| `title`       | Document title                                                 |
| `year`        | Publication year (validated within EIS range)                  |
| `eis_type`    | `Draft` / `Final` / `Supplemental` / `Unknown`                 |
| `lead_agency` | List of preparing agencies                                     |

Each value carries `confidence` (`high`/`medium`/`low`) and a `sources` list
identifying where it came from (`NUL`, `regex (first page)`, `Sonnet (first 2 pages)`,
etc.). Small file, ~30 lines.

## M2 — deep content extraction

Heavier structured extraction, run mostly against Opus with page-cited evidence for
every claim. Chunk-aware: the pipeline maps document sections onto CEQ chapters and
routes each chunk to the appropriate prompt.

Top-level sections per file:

| section         | what it contains                                                                                     |
|-----------------|------------------------------------------------------------------------------------------------------|
| `summary`       | Six paragraphs: `overview`, `project_description`, `affected_community`, `alternatives_overview`, `environmental_impact`, `public_response` — each with quoted evidence and exact source pages |
| `alternatives`  | Structured list of `{name, description, evidence}` for each alternative considered                   |
| `themes`        | Themes + subthemes drawn from a controlled taxonomy, plus supporting quotes                          |
| `location`      | Places (`region`/`state`/etc.), multi-site flag, and geocoded results                                |
| `key_people`    | Agency preparers, cooperating agencies, and public commenters (with verified quotes where possible)  |
| `chunking_meta` | Diagnostics: chunk count, CEQ chapters detected, per-chunk page ranges                               |

Every quoted piece of evidence includes `source_pages` and a `quote_verified` flag
indicating whether the quote was found verbatim in the per-page source JSON.

## Critic

For every M1 and M2 field the Critic evaluates a rubric and emits:

- `rubric_results` — per-check `yes`/`no`/`n/a` + note
- `verdict` — `PASS`, `PASS_WITH_NOTE`, `RE_EXTRACT`, or `HUMAN_REVIEW`
- `model_confidence` and free-text `notes`

Quotes that fail verbatim verification are forced to `HUMAN_REVIEW`.

## Grading sheets

`grading_sheets/<doc_id>.csv` is a flat, human-friendly view of every extracted
field alongside its evidence, critic verdict, and empty `your_grade` / `your_notes`
columns for review. Header comment lines document grade options and page-number
provenance.

## Support files

- **`selection.json`** — the sampling plan for this batch: RNG seed, per-length
  bucket targets (`short`/`medium`/`long`), realized bucket counts, and type/bureau
  distributions.
- **`run_summary.json`** — per-doc paths for M1/M2/critic/grading sheet and a
  by-model token + cost breakdown.
- **`run.log`** — the pipeline's stdout log for the batch run.
