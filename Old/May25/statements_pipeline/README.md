# `statements_pipeline/` — Per-Person Statement Extraction

A third pipeline alongside `segment_a/` and `people_pipeline/`. It reuses
their machinery (per-page text loading, chunking, the `(entity, stance)`
extractor + verifier + merger) and adds one new step on top: for each merged
entity, ask the model to **find the entity's full statement** in the doc and
write a per-person JSON.

The output unit is a per-doc folder of per-person JSONs. Each file describes
one stance-bearing entity, what they said (full text where the model can
locate it, paraphrase otherwise), and a short summary of their position.

## Per-doc flow

```
chunk (segment_a) → extract (people_pipeline) → verify (people_pipeline)
                  → merge by (entity, stance) (people_pipeline)
                  → find_statement (local) → write per-person folder + index.json
```

1. **Doc text load** (`pages.load_doc` from `segment_a/`). Reads
   `Documents/output/<doc_id>/page_NNNN.json` and joins pages into a single
   `full_text` with a real page-offset table.
2. **Chunk** (`segment_a/chunk.py`). 50-page page-aligned chunks with 2-page
   overlap, with CEQ-chapter labels stamped on chunks whose midpoint falls
   inside a detected chapter.
3. **Extract** (`people_pipeline/extract.py`). Sonnet on each chunk in
   parallel, returning every stance-bearing entity it finds. Entries without
   a recognized closed-set stance are dropped.
4. **Verify** (`people_pipeline/verify.py`). Every quote is checked verbatim
   (whitespace-normalized) against the doc text. Quotes not found verbatim
   keep `quote_verified=false`.
5. **Merge** (`people_pipeline/merge.py`). Group by
   `(normalized_entity, stance)`. Pick the longest verified quote as
   `summary_quote`, dedupe evidence pages, keep all per-chunk mentions, and
   assign `sequence` by first appearance.
6. **Find statement** (`find_statement.py`, local). For each merged row, build
   a doc-text *window* around the entity's evidence pages (`±10` pages by
   default, capped at `WINDOW_CHAR_CAP=60_000` chars), then ask Sonnet to:
   - classify `statement_form ∈ {letter, testimony, written_comment,
     narrator_paraphrase, sectional, none}`
   - return verbatim `opening_anchor` + `closing_anchor` if a contiguous
     statement exists
   - always return a 2–3 sentence `summary` of the entity's opinion
7. **Anchor verification + slice**. We search for `opening_anchor` /
   `closing_anchor` in the window with a whitespace-tolerant regex
   (tokens joined by `\s+`, so anchors copied with collapsed whitespace
   still match OCR text with line breaks). If both verify and `closing > opening`,
   we slice the full statement out of the doc. If either fails,
   `statement.text` is `null` but `summary` is always preserved.
8. **Write** (`writer.py`). One file per entity: `NNN_slug.json` in sequence
   order, plus `index.json` with metadata + counts + a `people` table that
   summarises every file in the folder.

There is **no LLM critic in this pipeline.** Review flagging is rules-based
(see *Human-review flag* below). A real critic could be added later as a
separate stage; for now, the per-anchor verification + the rules-based flag
are what we lean on.

## Doc source and metadata

- **Doc text:** every doc found under `PAGES_DATA_DIR`
  (`Documents/output/<doc_id>/page_NNNN.json`). `pages.list_doc_ids()` is
  the source of truth for what's in scope. No selection JSON.
- **Work metadata:** `inventory.lookup_work(doc_id)` from segment_a. When the
  doc isn't in the local MARC-shaped CSV (`inventory.py`), the run still
  proceeds — `work_id` and `title` are left empty in the per-doc output.
- **Selection list (`segment_a/output/selection.json`) is not consulted.** This
  pipeline runs against whatever per-page JSONs are present.

## Window sizing for `find_statement`

Why a window instead of the whole doc:

- **Cost.** Sending a 600k-char doc with 200+ entities and asking for a
  per-entity statement would be 200+ full-doc passes.
- **Anchor disambiguation.** A generic anchor like `Sincerely yours,` exists
  in many letters. Searching for it inside a small page-window keeps it from
  matching a different person's signature on the other side of the doc.

Window construction (`find_statement._build_window`):

- Parse `evidence_pages` (e.g. `["34", "67-69"]`) into integer page numbers.
- `start = max(min(pages) - 10, first_doc_page)`,
  `end = min(max(pages) + 10, last_doc_page)`.
- `text = doc.text_for_pages(start, end)`, hard-capped at
  `WINDOW_CHAR_CAP=60_000` chars.
- If evidence pages can't be parsed, fall back to the first 60k chars of the
  doc.

## Human-review flag

There's no LLM critic. `writer._needs_human_review` applies two rules:

| reason | trigger |
|---|---|
| `private_individual` | `kind == "individual"` (matches the v2 policy used by people_pipeline's critic) |
| `quote_not_verbatim` | `summary_quote_verified == False` |

`needs_human_review` is the OR of the two. Both reasons (or just one) are
recorded in `human_review_reasons` so reviewers can triage.

Note that `quote_not_verbatim` currently fires for narrator-paraphrase
entries too (their `summary_quote` is the narrator's paraphrase, not a
verbatim line). This is a known false-positive bucket; if it gets noisy, add
a separate `paraphrase_only` reason.

## Design choices

| decision                | value |
|-------------------------|-------|
| who counts as a person  | anyone with an attributed stance — individuals, officials, orgs, agencies, tribes, governments |
| stance vocabulary       | closed: `in_favor`, `opposed`, `conditional`, `neutral` |
| statement forms         | closed: `letter`, `testimony`, `written_comment`, `narrator_paraphrase`, `sectional`, `none` |
| dedup                   | one file per `(entity, stance)`. Same entity with two distinct stances → two files |
| sequence                | order of first appearance (lowest `chunk_index` in the merged group) |
| docs in scope           | every per-page JSON dir under `PAGES_DATA_DIR` |
| window margin           | `±10` pages around evidence pages, capped at 60,000 chars |
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
    ├── 001_amax_exploration_inc.json
    ├── 002_american_mining_congress.json
    └── ...
```

### Per-person file (`<NNN>_<slug>.json`)

```json
{
  "sequence": 1,
  "doc_id": "p0491_35556036091957",
  "work_id": "csv:35556036091957",
  "entity": "Amax Exploration, Inc.",
  "kind": "organization",
  "role": "mining industry commenter",
  "stance": "opposed",
  "summary": "2-3 sentence model summary of the entity's opinion.",
  "statement": {
    "present": false,
    "form": "narrator_paraphrase",
    "text": null,
    "opening_anchor": "",
    "closing_anchor": "",
    "opening_anchor_verified": false,
    "closing_anchor_verified": false,
    "window_pages": [24, 44]
  },
  "needs_human_review": true,
  "human_review_reasons": ["quote_not_verbatim"],
  "evidence_pages": ["34"],
  "summary_quote": "...",
  "summary_quote_verified": false,
  "attribution_mode": "paraphrased",
  "attribution_modes_seen": ["paraphrased"],
  "n_mentions": 1,
  "mentions": [
    {
      "chunk_index": 0,
      "evidence_pages": ["34"],
      "attribution_mode": "paraphrased",
      "quote": "...",
      "quote_verified": false,
      "stance_basis": "...",
      "entity_as_written": "...",
      "role_as_written": "..."
    }
  ]
}
```

Field notes:

- `statement.text` is the exact slice from the doc between the verified
  opening and closing anchors. **It is `null` whenever either anchor fails to
  verify** (or when the closer precedes the opener — likely a wrong-occurrence
  match). The `summary` is always populated even when text is null.
- `statement.window_pages` is `[start_page, end_page]` of the page-window the
  model was given. Useful for debugging anchor-verification failures.
- `evidence_pages` are real page numbers (per-page JSONs gives us exact
  pages, not estimates) carried over from the merge step.

### `index.json`

```json
{
  "doc_id": "...",
  "work_id": "csv:...",
  "title": "...",
  "n_pages": 356,
  "n_chunks": 8,
  "n_raw_rows": 227,
  "n_people": 219,
  "stance_counts":  { "opposed": 49, "neutral": 24, "in_favor": 37, "conditional": 109 },
  "review_counts":  {
    "needs_review": 74,
    "auto_ok": 145,
    "reasons": { "quote_not_verbatim": 34, "private_individual": 46 }
  },
  "statement_counts": {
    "letter": 166, "testimony": 0, "written_comment": 22,
    "narrator_paraphrase": 30, "sectional": 0, "none": 1,
    "present": 174, "absent": 45
  },
  "elapsed_sec": 510.7,
  "usage": { "extract": {...}, "find_statement": {...}, "total": {...} },
  "schema": { "stance_vocabulary": [...], "kind_vocabulary": [...], "statement_forms": [...] },
  "people": [
    {
      "sequence": 1,
      "file": "001_amax_exploration_inc.json",
      "entity": "Amax Exploration, Inc.",
      "kind": "organization",
      "role": "mining industry commenter",
      "stance": "opposed",
      "statement_present": false,
      "statement_form": "narrator_paraphrase",
      "needs_human_review": true,
      "human_review_reasons": ["quote_not_verbatim"]
    }
  ]
}
```

The `people` array makes `index.json` enough to triage a doc without opening
each per-person file.

## Checkpoints and reruns

- `output/raw_extract/<doc_id>.json` — per-chunk extractor output. Reruns
  reuse this and skip the (expensive) extract step unless `--force` is passed.
- `output/people/<doc_id>/` — final per-person output. Always rewritten on
  each run (verify / merge / find_statement re-run from the cached extract).

## Entry points

```
python run.py process              # all docs in PAGES_DATA_DIR
python run.py process --doc <id>   # one doc; need not be in inventory CSV
python run.py process --limit N    # first N docs in PAGES_DATA_DIR order
python run.py process --force      # ignore raw_extract checkpoint
python run.py status               # how many docs have raw_extract / people/
```

## Cost shape (one 356-page doc, 219 entities)

From the first end-to-end run on `p0491_35556036091957`:

| stage          | calls | input tokens | output tokens | est. cost |
|----------------|-------|--------------|---------------|-----------|
| extract        | 8     | 140,606      | 32,401        | $0.91     |
| find_statement | 219   | 2,092,875    | 54,535        | $7.10     |
| **total**      | 227   | 2,233,481    | 86,936        | **$8.00** |

`find_statement` dominates: it scales with the number of merged entities, and
each call carries up to 60k chars of doc-window context. Two natural levers
if cost matters:

1. **Tighten the window.** Drop `WINDOW_MARGIN_PAGES` from 10 to 4 for docs
   whose `evidence_pages` are dense. Anchor-distinctness goes down a bit, but
   token count drops a lot.
2. **Skip narrator-paraphrase / sectional rows.** Roughly 30/219 rows in the
   sample produced no contiguous statement. A cheap pre-filter (e.g. "if all
   mentions are `attribution_mode == "paraphrased"`, skip find_statement and
   synthesize the summary from the existing quote") would cut that long tail.

## Known issues / TODO

- **`run_summary.json` grand-total bug.** The per-doc `usage.total.cost_usd`
  is correct, but `cmd_process` rolls up the grand total via
  `(r.get("usage") or {}).get("total", {}).get("cost_usd", 0)` (`run.py:242`).
  The actual nesting is `usage.total.total.cost_usd`, so the printed grand
  total currently reads `$0.0000` when there is in fact real cost. Fix:
  add the `.total` hop, mirroring `cmd_status` (`run.py:279`).
- **Paraphrase-only entries get `quote_not_verbatim`.** They're correctly
  flagged as needing review, but the *reason* is misleading — there's no
  verbatim quote to verify because the source was a narrator paraphrase.
  Consider a separate `paraphrase_only` reason.
- **No critic.** All review flags are rules-based. Adding a Sonnet critic
  pass over the (statement, summary) pair is a natural extension.

## How this differs from `people_pipeline/`

| aspect | `people_pipeline/` | `statements_pipeline/` |
|---|---|---|
| doc text source | `docs_with_digits.json` (flat OCR) | per-page JSONs (`Documents/output/<doc_id>/`) |
| page numbers | estimated (`char_offset / 2500`) | real (from per-page JSONs) |
| docs in scope | 20-doc segment_a selection | every per-page JSON dir |
| critic | Sonnet rubric, per merged row | none (rules-based review flag) |
| output unit | one entry per merged row, all packed in one JSON | one file per merged row, in a per-doc folder + `index.json` |
| key new field | `summary_quote` | `statement.text` (full sliced statement) + `summary` |
| typical cost | ~$3–5 per 350pp doc | ~$8 per 350pp doc (find_statement adds 219 calls) |

The two pipelines can coexist — they read from different doc sources and
write to different output dirs.
