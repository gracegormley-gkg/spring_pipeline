# `statements_pipeline/` — Per-Person Statement Extraction

A third pipeline alongside `segment_a/` and `people_pipeline/`. It reuses
some of their machinery (per-page text loading, chunking, the verbatim quote
verifier) and replaces the rest with stage-split modules: discovery-only
extract, entity-only merge, and a find_statement step that *also* judges
stance, captures any nearby agency response, and assigns a confidence label.

The output unit is a per-doc folder of per-person JSONs. Each file describes
one entity, what they said (full text where the model can locate it,
paraphrase otherwise), the stance it took, how confident we are in that
stance, any **agency / preparer response** to their concern, and a short
summary of the position.

Rows whose `statement.form` is `narrator_paraphrase` are pulled out of the
main numbered sequence into a `paraphrases/` subfolder — they're preserved
for the data, just kept out of the way of entities with real statements.

## Per-doc flow

```
chunk (segment_a)
  → extract: discovery only, no stance (local)
  → verify: verbatim quote check (people_pipeline, reused)
  → merge by ENTITY (local; no stance key)
  → find_statement + stance + confidence (local)
  → write per-person folder + index.json
```

1. **Doc text load** (`pages.load_doc` from `segment_a/`). Reads
   `Documents/output/<doc_id>/page_NNNN.json` and joins pages into a single
   `full_text` with a real page-offset table.
2. **Chunk** (`segment_a/chunk.py`). 50-page page-aligned chunks with 2-page
   overlap, with CEQ-chapter labels stamped on chunks whose midpoint falls
   inside a detected chapter.
3. **Extract — discovery only** (`extract.py`, local). Sonnet on each chunk
   in parallel, returning every named entity whose POSITION is attributable
   from the text (individual, official, organization, agency, tribe,
   government). Returns `entity`, `kind`, `role`, `attribution_mode`,
   verbatim `quote`, `evidence_pages`. **Stance is NOT classified here** — the
   model is only enumerating candidates.
4. **Verify** (`people_pipeline/verify.py`, reused). Every quote is checked
   verbatim (whitespace-normalized) against the doc text. Verified rows get
   `evidence_pages` replaced with the single exact page; unverified rows
   keep `quote_verified=false`.
5. **Merge by entity** (`merge.py`, local). Group by `normalized_entity`.
   Pick the longest verified quote as `summary_quote`, dedupe evidence
   pages, keep all per-chunk mentions, assign `sequence` by first
   appearance. Stance is **not** part of the merge key, so the same entity
   produces at most one row regardless of attribution-mode mix.
6. **Find statement + stance + response** (`find_statement.py`, local). For
   each merged row, build a doc-text *window* around the entity's evidence
   pages (`±10` pages by default, capped at `WINDOW_CHAR_CAP=60_000` chars),
   then ask Sonnet to:
   - classify `statement_form ∈ {letter, testimony, written_comment,
     narrator_paraphrase, sectional, none}`
   - return verbatim `opening_anchor` + `closing_anchor` if a contiguous
     statement exists
   - **judge `stance ∈ {in_favor, opposed, conditional, neutral}`** off the
     full statement (or paraphrase / sectional heading) — whatever's
     available
   - report `stance_confidence ∈ {high, medium, low}` and a short
     `stance_basis` phrase
   - **look for a `response`** — an agency reply, preparer note, or
     discussion block addressing this entity's concern, often appearing
     immediately after their letter/paraphrase. Returns
     `response_form ∈ {agency_response, preparer_reply, discussion, none}`
     plus verbatim opening/closing anchors and a 1-2 sentence summary.
   - always return a 2–3 sentence `summary` of the entity's opinion
7. **Anchor verification + slice**. Anchors are matched whitespace-tolerantly
   (tokens joined by `\s+`) and **biased toward the entity's evidence pages**
   (statement) or **the end of the verified statement** (response) so generic
   anchors lock onto the right occurrence. The closing anchor is preferred
   near the opening. If both verify and `closing > opening`, the full text
   is sliced from the doc. If either fails, `text` is `null` but the
   `summary` is always preserved.
8. **Confidence cap**. After the model returns:
   - `statement_form == "none"` → confidence forced to `low`
   - `narrator_paraphrase / sectional / unverified anchors` → confidence
     capped at `medium`
   - else → take the model's value
9. **Write** (`writer.py`). Rows are partitioned:
   - `statement.form == "narrator_paraphrase"` → written to
     `paraphrases/<slug>.json` (no `NNN_` prefix), and listed in the
     `paraphrases` array of `index.json` with their own count blocks.
   - everything else → written to `NNN_slug.json` in a re-numbered main
     sequence (1..N by first appearance), and listed in the `people` array.
   The doc's `index.json` carries metadata, both listings, and rolled-up
   counts.

There is **no LLM critic in this pipeline.** Review flagging is rules-based
(see *Human-review flag* below). The split between discovery and stance
evaluation already does most of the work a critic would do — find_statement
sees the whole letter, not just a 50-page chunk window.

## Why split discovery from stance evaluation?

The earlier shape had the per-chunk extractor classify both "is this entity
making a position statement?" and "what is their stance?" in one call. That
forced the model to commit to a stance label off whatever sentences it saw
in a 50-page chunk — sometimes just one ambiguous narrator mention. We saw
two failure modes:

1. **Stance flipping across chunks** for the same entity. Different chunks
   would see different parts of the same letter and disagree on whether the
   entity was `opposed` or `conditional`. The merge step would then split
   the entity into two rows — same letter, two stances, two output files.
2. **Premature commitment** on weak evidence. A row's stance was
   irrevocable once the chunk-level extractor labeled it. There was no
   "I'm not sure" — just a closed-vocabulary stance label.

Now extract only discovers candidates, merge collapses them to one row per
entity, and find_statement makes the stance call with the full letter
in view. When evidence is weak, `stance_confidence` records that — the
stance is still committed, but the row is flagged for human review.

## Doc source and metadata

- **Doc text:** every doc found under `PAGES_DATA_DIR`
  (`Documents/output/<doc_id>/page_NNNN.json`). `pages.list_doc_ids()` is
  the source of truth for what's in scope. No selection JSON.
- **Work metadata:** `inventory.lookup_work(doc_id)` from segment_a. When the
  doc isn't in the local MARC-shaped CSV (`inventory.py`), the run still
  proceeds — `work_id` and `title` are left empty in the per-doc output.
- **Selection list (`segment_a/output/selection.json`) is not consulted.**
  This pipeline runs against whatever per-page JSONs are present.

## Window sizing for `find_statement`

Why a window instead of the whole doc:

- **Cost.** Sending a 600k-char doc with 200+ entities and asking for a
  per-entity statement would be 200+ full-doc passes.
- **Anchor disambiguation.** A generic anchor like `Sincerely yours,`
  exists in many letters. Searching for it inside a small page-window keeps
  it from matching a different person's signature on the other side of the
  doc. We further bias the search toward the entity's evidence pages
  (`ANCHOR_PROXIMITY_CHARS`) so even within the window, generic anchors
  lock onto the right occurrence.

Window construction (`find_statement._build_window`):

- Parse `evidence_pages` (e.g. `["34", "67-69"]`) into integer page numbers.
- `start = max(min(pages) - 10, first_doc_page)`,
  `end = min(max(pages) + 10, last_doc_page)`.
- `text = doc.text_for_pages(start, end)`, hard-capped at
  `WINDOW_CHAR_CAP=60_000` chars; `end_page` is recomputed so it reflects
  what actually fits.
- If evidence pages can't be parsed, fall back to the head of the doc with
  `end_page` similarly recomputed.

## Stance, confidence, and review flagging

Each row carries a top-level `stance ∈ {in_favor, opposed, conditional,
neutral}` plus a `stance_confidence ∈ {high, medium, low}`. Confidence is
the model's call, capped by structural evidence:

| situation                                    | max confidence |
|----------------------------------------------|----------------|
| `statement_form == "none"`                   | forced `low`   |
| `narrator_paraphrase` / `sectional`          | `medium`       |
| anchors didn't verify even though form claims a statement | `medium` |
| verified contiguous letter / testimony / written_comment | up to `high` |

`stance_basis` is a short phrase pointing at the evidence ("opens letter
calling project unacceptable", "listed under PRO REGULATIONS heading",
etc.).

`writer._needs_human_review` flags rows with any of:

| reason | trigger |
|---|---|
| `private_individual` | `kind == "individual"` (matches the v2 policy used by people_pipeline's critic) |
| `quote_not_verbatim` | `summary_quote_verified == False` |
| `low_stance_confidence` | `stance_confidence == "low"` |

`needs_human_review` is the OR of all triggered reasons. All triggered
reasons are recorded in `human_review_reasons` so reviewers can triage.
Note that `quote_not_verbatim` currently fires for narrator-paraphrase
entries too (their `summary_quote` is the narrator's paraphrase, not a
verbatim line); if the noise gets bothersome, add a separate
`paraphrase_only` reason.

## Design choices

| decision                | value |
|-------------------------|-------|
| who counts as a person  | anyone with an attributable position — individuals, officials, orgs, agencies, tribes, governments |
| stance vocabulary       | closed: `in_favor`, `opposed`, `conditional`, `neutral` |
| stance confidence       | closed: `high`, `medium`, `low` |
| statement forms         | closed: `letter`, `testimony`, `written_comment`, `narrator_paraphrase`, `sectional`, `none` |
| dedup                   | one file per entity (no stance in merge key) |
| sequence                | order of first appearance (lowest `chunk_index` in the merged group) |
| docs in scope           | every per-page JSON dir under `PAGES_DATA_DIR` |
| window margin           | `±10` pages around evidence pages, capped at 60,000 chars |
| anchor proximity bias   | `25,000` chars (`ANCHOR_PROXIMITY_CHARS`) |
| statement text cap      | `40,000` chars (`STATEMENT_CHAR_CAP`) |
| LLM                     | Sonnet for both extract and find_statement; no Opus |
| parallelism             | 4 for extract, 4 for find_statement |

## Output layout

```
statements_pipeline/output/
├── run_summary.json                     # per-doc paths, counts, usage
├── raw_extract/<doc_id>.json            # checkpoint: per-chunk extractor output
└── people/<doc_id>/
    ├── index.json
    ├── 001_amax_exploration_inc.json    # main sequence
    ├── 002_american_mining_congress.json
    ├── ...
    └── paraphrases/                     # narrator_paraphrase rows
        ├── city_of_albuquerque.json     #   no NNN_ prefix; not in main count
        ├── ...
```

### Per-person file (`<NNN>_<slug>.json` in main sequence, or `paraphrases/<slug>.json`)

```json
{
  "sequence": 1,
  "merge_sequence": 4,
  "doc_id": "p0491_35556036091957",
  "work_id": "csv:35556036091957",
  "entity": "Sierra Club",
  "kind": "organization",
  "role": "national environmental advocacy org",
  "stance": "opposed",
  "stance_confidence": "high",
  "stance_basis": "opens letter calling proposal 'unacceptable'",
  "summary": "Sierra Club opposes the proposed regulations because ...",
  "statement": {
    "present": true,
    "form": "letter",
    "text": "Dear Sir,\n\n[full letter body]\n\nSincerely,\nJohn Smith, Sierra Club",
    "opening_anchor": "Dear Sir, We the undersigned at Sierra Club ...",
    "closing_anchor": "... Sincerely, John Smith, Sierra Club",
    "opening_anchor_verified": true,
    "closing_anchor_verified": true,
    "window_pages": [140, 165]
  },
  "response": {
    "present": true,
    "form": "agency_response",
    "text": "Response: The Department recognizes the concerns raised by Sierra Club ...",
    "summary": "BLM acknowledges the concerns and commits to additional mitigation in the final EIS.",
    "opening_anchor": "Response: The Department recognizes",
    "closing_anchor": "... addressed in the final EIS.",
    "opening_anchor_verified": true,
    "closing_anchor_verified": true
  },
  "needs_human_review": false,
  "human_review_reasons": [],
  "evidence_pages": ["142-143", "151"],
  "summary_quote": "...",
  "summary_quote_verified": true,
  "attribution_mode": "direct_quote",
  "attribution_modes_seen": ["direct_quote"],
  "n_mentions": 3,
  "mentions": [
    {
      "chunk_index": 6,
      "evidence_pages": ["142-143"],
      "attribution_mode": "direct_quote",
      "quote": "...",
      "quote_verified": true,
      "entity_as_written": "Sierra Club",
      "role_as_written": ""
    }
  ]
}
```

Field notes:

- `sequence` is the output position in the main numbered listing (`null`
  for paraphrase rows). `merge_sequence` is the original first-appearance
  order from merge — preserved on every record for traceability.
- `stance` is decided by find_statement, not by the per-chunk extractor.
  Extract only does discovery.
- `stance_confidence` is the model's confidence, capped by structural
  evidence (see table above). `low` always trips `needs_human_review`.
- `statement.text` is the exact slice from the doc between the verified
  opening and closing anchors. Null when either anchor fails to verify.
- `response.text` is the analogous slice for the nearby agency / preparer
  reply. Null when no response was found or anchors didn't verify.
  `response.summary` is always populated when `response.present` is true.
- `evidence_pages` are real page numbers (per-page JSONs give us exact
  pages) carried over from the merge step.

### `index.json`

Counts apply to the **main rows** at the top level. Paraphrase counts live
under `paraphrase_counts` and the listing is a separate `paraphrases` array.

```json
{
  "doc_id": "...",
  "work_id": "csv:...",
  "title": "...",
  "n_pages": 356,
  "n_chunks": 8,
  "n_raw_rows": 227,
  "n_people": 189,
  "n_paraphrases": 30,
  "stance_counts":            { "opposed": 49, "neutral": 24, "in_favor": 37, "conditional": 79 },
  "stance_confidence_counts": { "high": 142, "medium": 28, "low": 19, "unknown": 0 },
  "review_counts":  {
    "needs_review": 86,
    "auto_ok": 103,
    "reasons": { "quote_not_verbatim": 34, "private_individual": 46, "low_stance_confidence": 19 }
  },
  "statement_counts": {
    "letter": 166, "testimony": 0, "written_comment": 22,
    "narrator_paraphrase": 0, "sectional": 0, "none": 1,
    "present": 174, "absent": 15
  },
  "response_counts": {
    "agency_response": 142, "preparer_reply": 4, "discussion": 11, "none": 32,
    "present": 157, "absent": 32
  },
  "paraphrase_counts": {
    "stance":             { "neutral": 14, "opposed": 8, "in_favor": 5, "conditional": 3 },
    "stance_confidence":  { "high": 0, "medium": 22, "low": 8, "unknown": 0 },
    "review":             { "needs_review": 12, "auto_ok": 18, "reasons": {"quote_not_verbatim": 12} },
    "response":           { "agency_response": 18, "preparer_reply": 0, "discussion": 4, "none": 8, "present": 22, "absent": 8 }
  },
  "elapsed_sec": 510.7,
  "usage": { "extract": {...}, "find_statement": {...}, "total": {...} },
  "schema": { "stance_vocabulary": [...], "response_forms": [...], ... },
  "people": [
    {
      "sequence": 1, "merge_sequence": 4,
      "file": "001_sierra_club.json",
      "entity": "Sierra Club", "kind": "organization", "role": "...",
      "stance": "opposed", "stance_confidence": "high",
      "statement_present": true, "statement_form": "letter",
      "response_present": true, "response_form": "agency_response",
      "needs_human_review": false, "human_review_reasons": []
    }
  ],
  "paraphrases": [
    {
      "merge_sequence": 17,
      "file": "paraphrases/city_of_albuquerque.json",
      "entity": "City of Albuquerque", "kind": "government", "role": "commenter",
      "stance": "conditional", "stance_confidence": "medium",
      "response_present": true, "response_form": "agency_response",
      "needs_human_review": false, "human_review_reasons": []
    }
  ]
}
```

The `people` and `paraphrases` arrays make `index.json` enough to triage
a doc without opening individual files.

## Module shadowing (how local extract/merge replace the upstream ones)

`settings.py` appends `segment_a/` and `people_pipeline/` to `sys.path`
(`sys.path.append`, not `insert`). Local-module-wins on name collisions
means:

| name | resolves to |
|---|---|
| `chunk`, `pages`, `inventory`, `config`, `llm` | `segment_a/` |
| `verify` | `people_pipeline/` (reused) |
| `extract`, `merge` | **local** (this directory) |
| `find_statement`, `writer`, `settings`, `run` | local |

The upstream `people_pipeline/extract.py` and `people_pipeline/merge.py`
remain untouched; they're shadowed for this pipeline only. `EXTRACT_CHAR_CAP`
in `settings.py` is the value `extract.py` reads via `import settings`, so it
must stay set even though no other code in this pipeline reads it directly.

## Checkpoints and reruns

- `output/raw_extract/<doc_id>.json` — per-chunk extractor output. Reruns
  reuse this and skip the (expensive) extract step unless `--force` is
  passed.
- `output/people/<doc_id>/` — final per-person output. Always rewritten on
  each run (verify / merge / find_statement re-run from the cached extract).

## Entry points

```
python run.py process              # all docs in PAGES_DATA_DIR
python run.py process --doc <id>   # one doc; need not be in inventory
python run.py process --limit N    # first N docs in PAGES_DATA_DIR order
python run.py process --force      # ignore raw_extract checkpoint
python run.py status               # how many docs have raw_extract / people/
```

Per-doc log line shape:

```
Wrote .../people/<doc_id> (219 people; statement_present=174 conf=H142/M58/L19 needs_review=86) in 510.7s — est. cost $8.0123
```

`H/M/L` = high/medium/low stance confidence buckets.

## Cost shape

The split adds a small amount of work at find_statement (an extra ~150
output tokens per row for stance + confidence + basis), and removes a
similar amount from extract (no more stance / stance_basis per entity).
Net effect is roughly cost-neutral. The dominant cost is still
find_statement, which scales with merged-entity count and the doc-window
size.

Two natural levers if cost matters:

1. **Tighten the window.** Drop `WINDOW_MARGIN_PAGES` from 10 to 4 for
   docs whose `evidence_pages` are dense. Anchor-distinctness goes down a
   bit, but token count drops a lot.
2. **Skip narrator-paraphrase / sectional rows.** A cheap pre-filter
   ("if all mentions are `attribution_mode == 'paraphrased'`, skip
   find_statement and synthesize summary + low-confidence stance from the
   existing quote") would cut the long tail.

## Known issues / TODO

- **Paraphrase-only entries get `quote_not_verbatim`.** They're correctly
  flagged as needing review, but the *reason* is misleading — there's no
  verbatim quote to verify because the source was a narrator paraphrase.
  Consider a separate `paraphrase_only` reason.
- **No critic.** All review flags are rules-based. Adding a Sonnet critic
  pass over the (statement, stance, summary) tuple is a natural extension.
- **Same entity, genuinely two stances.** Now collapses to one row with
  whatever stance find_statement picks (possibly with capped confidence).
  The previous schema produced two rows. If you need to preserve both,
  reintroduce a stance key in merge but also keep the find_statement
  override.

## How this differs from `people_pipeline/`

| aspect | `people_pipeline/` | `statements_pipeline/` |
|---|---|---|
| doc text source | `docs_with_digits.json` (flat OCR) | per-page JSONs (`Documents/output/<doc_id>/`) |
| page numbers | estimated (`char_offset / 2500`) | real (from per-page JSONs) |
| docs in scope | 20-doc segment_a selection | every per-page JSON dir |
| stance assigned | per-chunk at extract time | downstream of merge, off the full statement |
| stance confidence | none | `high` / `medium` / `low`, capped by form/anchor outcome |
| critic | Sonnet rubric, per merged row | none (rules-based review flag) |
| output unit | one entry per merged row, all packed in one JSON | one file per merged row, in a per-doc folder + `index.json` |
| key new field | `summary_quote` | `statement.text` (full sliced statement) + `summary` + `stance_confidence` |

The two pipelines can coexist — they read from different doc sources and
write to different output dirs.
