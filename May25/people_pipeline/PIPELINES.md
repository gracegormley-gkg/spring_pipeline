# May25 Pipelines — Outputs and Methods

This file is a one-stop summary of the two pipelines in this directory:

- `segment_a/` — the Segment A *calibration* pipeline (per-doc structured fields + critic + grading sheet)
- `people_pipeline/` — an exhaustive `(entity, stance)` extractor that reuses Segment A's chunking and quote-checking machinery

Both pipelines run against the same 20-document calibration sample
(`segment_a/output/selection.json`), pulled from `docs_with_digits.json` (flat
OCR strings — no PDFs). Page numbers throughout both pipelines are
**estimated** at `char_offset / 2500`, since the source is unpaginated text.

---

## 1. `segment_a/` — Calibration Run

Implements the Segment A flow from `Pipeline.pdf`. For each of 20 stratified
docs (5 short / 10 medium / 5 long), it runs two extraction modules (M1, M2),
a per-field critic, and writes a per-doc grading CSV.

### Pipeline stages

1. **NUL fetch** (`nul.py`) — pulls metadata for each doc from the NUL DC API
   using three accession-matching strategies. Cached under `cache/`.
2. **Chunk** (`chunk.py`) — regex-based chapter detection mapped onto CEQ §1502
   chapters, then 50-page chunks with 2-page overlap (in practice
   125,000-char chunks with 5,000-char overlap, since the source is flat OCR).
3. **Selection** (`selection.py`) — stratified sample of 20 docs by length.
   Output: `output/selection.json`.
4. **M1 — header fields** (`m1.py`):
   - `title` — NUL first, Haiku on first page as fallback
   - `year` — NUL first, regex on first 3 pages as fallback (disagreements
     are flagged with low confidence)
   - `eis_type` — regex on first page → Sonnet on first 2 pages
   - `lead_agency` — NUL first, Sonnet on first 4 pages as fallback
5. **M2 — body fields** (`m2.py`):
   - `summary` — Opus map-reduce over chunks, producing five sub-fields:
     `project_description`, `affected_community`, `alternatives_overview`,
     `environmental_impact`, `public_response` (with `based_on_main_doc_only`)
   - `alternatives` — Sonnet on the detected Alternatives chapter; deliberately
     skipped (no word-regex fallback) when the chapter isn't structurally found
   - `themes` — Sonnet over the summary, restricted to a fixed taxonomy
     (1–3 themes, 2–5 subthemes)
   - `location` — Sonnet on the first ~30 pages and any Project/Study Area
     section, then geocoded with Nominatim (geopy)
   - `key_people` — Sonnet on the Consultation chapter (preparers + cooperating
     agencies), plus a comment-response sweep for public commenters; all
     quotes are verified verbatim against the doc text
6. **Critic** (`critic.py`) — Sonnet runs a per-field rubric and returns one
   of `PASS / PASS_WITH_NOTE / RE_EXTRACT / HUMAN_REVIEW`. Hard override:
   private-individual stance attributions are forced to `HUMAN_REVIEW`.
7. **Grading sheet** (`grading.py`) — flattens M1 + M2 + critic into a per-doc
   CSV with one row per field, ready for human grading
   (`correct | minor_issue | wrong | cant_tell`).

Stages are checkpointed per-doc, so reruns resume cheaply. `run.py select` /
`run.py process` / `run.py status` are the entry points.

### Models used

| Field          | Primary                                    | Fallback / verifier              | Model     |
|----------------|--------------------------------------------|----------------------------------|-----------|
| `title`        | NUL                                        | Haiku on first page              | Haiku     |
| `year`         | NUL                                        | Regex on first 3 pages           | (none)    |
| `eis_type`     | Regex on first page                        | Sonnet on first 2 pages          | Sonnet    |
| `lead_agency`  | NUL                                        | Sonnet on first 4 pages          | Sonnet    |
| `summary`      | Opus map-reduce over chunks                | —                                | Opus      |
| `alternatives` | Sonnet on detected Alternatives chapter    | — (skipped if chapter not found) | Sonnet    |
| `themes`       | Sonnet from summary                        | —                                | Sonnet    |
| `location`     | Sonnet on first 30pp + Project/Study Area  | Nominatim geocoder               | Sonnet    |
| `key_people`   | Sonnet on Consultation ch + comment sweep  | Verbatim quote check             | Sonnet    |
| `critic`       | Sonnet, per-field rubric                   | Auto `HUMAN_REVIEW` on parse fail | Sonnet   |

### Output layout

```
segment_a/output/
├── selection.json                  # the 20 sampled docs
├── run_summary.json                # per-doc paths + elapsed time + errors
├── run.log
├── m1/<doc_id>.json                # title, year, eis_type, lead_agency
├── m2/<doc_id>.json                # summary, alternatives, themes, location, key_people, chunking_meta
├── critic/<doc_id>.json            # per-field rubric_results, verdict, model_confidence, notes
└── grading_sheets/<doc_id>.csv     # one row per field, ready for human grading
```

### What each output file contains

- **`m1/<doc_id>.json`** — for each header field: `value`, `confidence`
  (`high|medium|low`), `sources` (e.g. `["NUL", "regex (first 3 pages)"]`),
  and an optional `evidence` / `note` (e.g. flagging NUL vs. regex disagreements).
- **`m2/<doc_id>.json`** — `summary` (5 narrative sub-fields, each with
  `source_pages`), `alternatives` (with `note` if skipped), `themes`
  (`themes` + `subthemes` + `justification`), `location` (`places`,
  `is_multi_site`, `geocoded` with lat/lon/address), `key_people`
  (`agency_preparers`, `cooperating_agencies`, `public_commenters`,
  `comment_response_present`), and `chunking_meta` (`n_chunks`,
  `n_chapters_detected`, detected chapters with CEQ mapping, chunk size,
  page-estimation note).
- **`critic/<doc_id>.json`** — for every M1/M2 field: a list of
  `rubric_results` (`check`, `result` ∈ `yes|no|n/a`, `note`), a `verdict`,
  a `model_confidence`, free-text `notes`, and the `source_pages` that were
  pulled in as text for the critic.
- **`grading_sheets/<doc_id>.csv`** — flat CSV with header comments
  (doc id, work id, title, grade options, page-estimation note) and one
  row per field with `extracted_value`, `source_pages`, `critic_verdict`,
  `model_confidence`, an empty `your_grade` column, and a truncated
  `your_notes` column from the critic.

---

## 2. `people_pipeline/` — Exhaustive `(entity, stance)` Extraction

A second pipeline, layered on Segment A's chunking + LLM client + verbatim
quote-checking, that produces an exhaustive list of every stance-bearing
entity in each of the same 20 docs. Each merged `(entity, stance)` row is
designed to become one grading row downstream.

### Pipeline stages (per doc)

1. **Chunk** — reused as-is from `segment_a/chunk.py` (50-page chunks,
   2-page overlap, CEQ-chapter labels where detected).
2. **Extract** (`extract.py`) — Sonnet on each chunk in parallel, returning
   every stance-bearing entity it finds. Entries without a recognized
   closed-set stance are dropped immediately.
3. **Verify** (`verify.py`) — every quote is checked against the full doc
   text (whitespace-normalized). Quotes not found verbatim keep
   `quote_verified=false` and force `HUMAN_REVIEW` later.
4. **Merge** (`merge.py`) — group by `(normalized_entity, stance)`. Pick
   the longest verified quote as `summary_quote`, dedupe evidence pages,
   keep all per-chunk mentions, assign `sequence` by first appearance
   (lowest `chunk_index`).
5. **Critic** (`critic.py`) — Sonnet rubric per merged row, with the cited
   pages **pulled in as actual text** (per the v2 M2 Check spec). Returns
   `PASS / PASS_WITH_NOTE / RE_EXTRACT / HUMAN_REVIEW`. Hard overrides:
   - `summary_quote_verified == false` → forced `HUMAN_REVIEW`
   - `kind == "individual"` → forced `HUMAN_REVIEW` (matches v2 policy
     that private-individual stance attributions always go to a human)

### Design choices

| decision                | value |
|-------------------------|-------|
| who counts as "person"  | anyone or anything with an attributed stance — individuals, officials, orgs, agencies, tribes, governments |
| stance vocabulary       | closed set: `in_favor`, `opposed`, `conditional`, `neutral` (no-stance entries dropped at extract time) |
| dedup                   | one row per `(entity, stance)`. Same entity with two distinct stances → two rows. Conservative: under-merging is preferred over over-merging |
| sequence                | order of first appearance (lowest `chunk_index` in the merged group) |
| docs                    | the same 20 from `segment_a/output/selection.json` |
| chunking                | reused from `segment_a/chunk.py` (50-page chunks, 2-page overlap) |
| LLM                     | Sonnet for extract + critic (no Opus — exhaustive enumeration over many chunks would be too costly) |

### Output layout

```
people_pipeline/output/
├── run_summary.json                  # per-doc n_entries, verdict_counts, stance_counts, paths
├── raw_extract/<doc_id>.json         # checkpoint: per-chunk extractor output (skipped on rerun unless --force)
└── entries/<doc_id>.json             # final per-doc output (always rewritten)
```

### `entries/<doc_id>.json` schema

```json
{
  "doc_id": "...",
  "work_id": "...",
  "title": "...",
  "n_entries": 17,
  "verdict_counts": { "PASS": 9, "PASS_WITH_NOTE": 3, "RE_EXTRACT": 1, "HUMAN_REVIEW": 4 },
  "stance_counts":  { "in_favor": 6, "opposed": 7, "conditional": 3, "neutral": 1 },
  "entries": [
    {
      "sequence": 1,
      "entity": "Sierra Club",
      "kind": "organization",
      "role": "national environmental advocacy org",
      "stance": "opposed",
      "summary_quote": "...",
      "summary_quote_verified": true,
      "evidence_pages": ["142-143", "151"],
      "n_mentions": 3,
      "mentions": [
        {
          "chunk_index": 6,
          "evidence_pages": ["142-143"],
          "quote": "...",
          "quote_verified": true,
          "stance_basis": "calls the proposal 'unacceptable'",
          "entity_as_written": "Sierra Club",
          "role_as_written": ""
        }
      ],
      "critic": {
        "verdict": "PASS",
        "model_confidence": "high",
        "notes": "...",
        "rubric_results": [{ "check": "quote verbatim", "result": "yes", "note": "" }]
      }
    }
  ]
}
```

### Checkpoints

- `output/raw_extract/<doc_id>.json` — per-chunk extractor output. Reruns
  skip the extract step unless `--force` is passed.
- `output/entries/<doc_id>.json` — final per-doc output. Always rewritten on
  each run (verify / merge / critic re-run from the cached extract).

Entry points: `python run.py process [--limit N] [--doc <id>] [--force]` and
`python run.py status`.

---

## Shared caveats

- **Page numbers are estimated** in both pipelines (char_offset / 2500), and
  every grading sheet header notes this.
- **Quote verification** is "verbatim somewhere in the doc," not "verbatim on
  the cited page" — a consequence of unpaginated OCR input.
- **Sonnet stands in for Haiku** on this Bedrock account (see
  `segment_a/config.py`). When Haiku 4-5 becomes accessible, the
  `people_pipeline` extract step is the natural place to use it — extraction
  over many chunks is the most expensive call there.
- **Private-individual stances** are always routed to `HUMAN_REVIEW` in both
  the Segment A critic (for `key_people.public_commenters`) and the
  people-pipeline critic (for `kind == "individual"`).
