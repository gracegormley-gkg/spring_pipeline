# statements_pipeline — Per-Person Statement Extraction

Layered on top of `people_pipeline/`'s extract / verify / merge machinery, this
pipeline tries to locate **each entity's whole statement** in a doc — their
comment letter, their town-hall testimony, their written position paper — and
writes one JSON per person per doc, plus an `index.json` per doc.

The two halves it adds on top of people_pipeline:

1. A **find_statement** step that asks Sonnet to identify the statement's
   verbatim opening and closing anchors, then slices the doc text between
   them. Always produces a 2–3 sentence `summary` even when no contiguous
   statement exists.
2. A **per-person writer** that emits one JSON file per `(entity, stance)`
   row into a per-doc folder, plus an `index.json` listing them.

There is **no LLM critic** in this pipeline. A rules-based
`needs_human_review` flag substitutes for the verdict.

---

## Doc source

Processes every doc found in segment_a's `PAGES_DATA_DIR`:

```
Documents/output/<doc_id>/page_NNNN.json
```

Each per-page JSON carries `{page_number, text, ...}`. Page numbers are
**real**, taken straight from the JSON. Per-page JSONs are joined into a
single `full_text` by `pages.load_doc` — chunks and the find_statement window
slice contiguous spans from that joined text, so the LLM never sees raw
per-page boundaries.

Title and `work_id` come from `inventory.lookup_work(doc_id)` when the doc is
in the local MARC-shaped inventory CSV (`segment_a/inventory.py`). When a doc
isn't in the inventory, the entry is still processed; title/work_id are
just left empty.

---

## Pipeline stages (per doc)

1. **Chunk** (`segment_a/chunk.py:chunks_for_doc`) — regex chapter detection
   mapped onto CEQ §1502 chapters, then 50-page chunks with 2-page overlap.
   Reused as-is from segment_a.
2. **Extract** (`people_pipeline/extract.py:extract_doc`) — per-chunk Sonnet
   call that returns every stance-bearing entity it sees (individual,
   official, organization, agency, tribe, government, other). Stances are
   the closed set `in_favor | opposed | conditional | neutral`; entries
   without a recognized stance are dropped at parse time. Checkpointed at
   `output/raw_extract/<doc_id>.json`.
3. **Verify** (`people_pipeline/verify.py:verify_rows`) — verbatim quote check
   per page via `Doc.find_quote`. Verified rows get their `evidence_pages`
   replaced with the single exact page; unverified rows keep the LLM's
   reported pages and a `quote_verified=false` flag.
4. **Merge** (`people_pipeline/merge.py:merge_rows`) — one row per
   `(normalized_entity, stance)`. Same entity with two distinct stances
   produces two rows. Sequence is assigned by first appearance (lowest
   `chunk_index` in the merged group).
5. **Find statement** (`find_statement.py`) — for each merged row, Sonnet is
   shown the entity info + an exemplar quote + a doc-text **window** around
   the entity's evidence pages, and returns:
   - `statement_form` ∈ `letter | testimony | written_comment |
     narrator_paraphrase | sectional | none`
   - `opening_anchor` and `closing_anchor` — verbatim short strings
     (~50–150 chars each) that bound the statement, or `""`
   - `summary` — 2–3 sentence summary of the entity's opinion, always present
   Anchors are verified via **whitespace-tolerant** substring search in the
   same window we showed the model (`find_statement._find_anchor`). When both
   verify, the full statement text is sliced from the doc between them and
   stored as `statement.text`. Otherwise `statement.text` is `null` but
   `summary` is still emitted.
6. **Write** (`writer.py:write_doc`) — one JSON per `(entity, stance)` at
   `output/people/<doc_id>/NNN_slug.json`, plus an `index.json` listing the
   files and aggregate counts.

### Why a window, not the whole doc?

For each merged row, find_statement only looks within
`evidence_pages ± WINDOW_MARGIN_PAGES` (default ±10) up to `WINDOW_CHAR_CAP`
characters (default 60 000). Two reasons:

- **Cost.** A 356-page doc with ~20 merged rows would otherwise cost ~20
  full-doc Sonnet calls. The window keeps each call to roughly the size of
  one chunk.
- **Anchor disambiguation.** A generic anchor like `"Sincerely yours,"` could
  match many letters across a long doc. Scoping the verbatim search to the
  same window the model saw means anchors lock to the right occurrence.

When a statement extends past the window (e.g. evidence page is 100 but the
letter ends on page 113 with a margin of ±10), the closing anchor won't
verify and the statement is recorded with `text=null` and an unverified
anchor flag. Bump `WINDOW_MARGIN_PAGES` in `settings.py` if this happens a
lot in your corpus.

---

## `needs_human_review` flag

No LLM critic runs in this pipeline. Instead, `writer._needs_human_review`
applies two cheap rules:

| reason | trigger |
|--------|---------|
| `private_individual` | `kind == "individual"` — mirrors the v2 policy used by people_pipeline's critic: private-individual stance attributions always go to a human |
| `quote_not_verbatim` | `summary_quote_verified == false` — the exemplar quote wasn't found verbatim in the doc, so the row's source line can't be auto-trusted |

Each person record carries `needs_human_review: bool` and
`human_review_reasons: list[str]`. `index.json` aggregates these into
`review_counts`.

If you need a real LLM critic later, the natural place to add it is between
find_statement and write — taking the statement.text plus the row as input.

---

## Output layout

```
statements_pipeline/output/
├── run_summary.json                        # per-doc counts, paths, cost
├── raw_extract/<doc_id>.json               # per-chunk extractions (checkpointed)
└── people/<doc_id>/
    ├── index.json                          # doc metadata + counts + file list
    ├── 001_sierra_club.json                # one file per (entity, stance)
    ├── 002_john_smith.json
    └── ...
```

`output/raw_extract/<doc_id>.json` is the only checkpoint. Reruns skip the
extract step unless `--force` is passed; verify / merge / find_statement /
write always re-run from the cached extract.

### Per-person JSON schema

```json
{
  "sequence": 1,
  "doc_id": "p0491_35556036091957",
  "work_id": "csv:35556036091957",
  "entity": "Sierra Club",
  "kind": "organization",
  "role": "national environmental advocacy org",
  "stance": "opposed",
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
      "stance_basis": "calls the proposal 'unacceptable'",
      "entity_as_written": "Sierra Club",
      "role_as_written": ""
    }
  ]
}
```

When no contiguous statement is present (`narrator_paraphrase`, `sectional`,
or `none`), `statement.present` is `false`, `statement.text` is `null`, both
anchors are `""` with their `_verified` flags `false`. `summary` is still
populated from the available mentions.

### `index.json` schema

```json
{
  "doc_id": "p0491_35556036091957",
  "work_id": "csv:35556036091957",
  "title": "Off-road vehicle regulations, proposed : environmental impact statement.",
  "n_pages": 356,
  "n_chunks": 7,
  "n_raw_rows": 42,
  "n_people": 17,
  "stance_counts":    { "in_favor": 6, "opposed": 7, "conditional": 3, "neutral": 1 },
  "review_counts":    { "needs_review": 4, "auto_ok": 13, "reasons": { "private_individual": 3, "quote_not_verbatim": 1 } },
  "statement_counts": { "letter": 5, "testimony": 2, "written_comment": 1, "narrator_paraphrase": 7, "sectional": 1, "none": 1, "present": 8, "absent": 9 },
  "elapsed_sec": 142.8,
  "usage": {
    "extract":        { "by_model": [ ... ], "total": { ... } },
    "find_statement": { "by_model": [ ... ], "total": { ... } },
    "total":          { "by_model": [ ... ], "total": { ... } }
  },
  "schema": { ... vocabulary + caveats ... },
  "people": [
    { "sequence": 1, "file": "001_sierra_club.json", "entity": "Sierra Club", "stance": "opposed",
      "statement_present": true, "statement_form": "letter",
      "needs_human_review": false, "human_review_reasons": [] }
  ]
}
```

---

## How per-page JSONs flow through `statements_pipeline/`

Same `pages.Doc` abstraction as `segment_a/`, applied to a different set of
downstream stages. No downstream stage knows the input came from many small
files.

1. **Load + index** (`pages.load_doc`). Reads
   `<PAGES_DATA_DIR>/<doc_id>/page_NNNN.json`, sorts by `page_number`, builds
   `pages`, `full_text` (joined with `PAGE_SEP = "\n\n"`), and `_page_starts`
   (char offset where each page begins in `full_text`).
2. **Chunking** (`chunk.chunks_for_doc`). 50-page chunks with 2-page overlap;
   each `Chunk.text` is `doc.text_for_pages(start, end)`.
3. **Extract** sees one chunk at a time. The LLM gets contiguous prose with
   page-separator newlines — no per-page JSON shape leaks in.
4. **Verify** (`Doc.find_quote`) — whitespace-normalized search **per page**,
   never across the seam between two pages. Returns the exact `page_num`. This
   is what lets the merged row cite an exact page in `evidence_pages` rather
   than the whole chunk's span.
5. **Find statement** builds a window via
   `doc.text_for_pages(start_page, end_page)` (`find_statement._build_window`)
   sized to `evidence_pages ± WINDOW_MARGIN_PAGES`, capped at
   `WINDOW_CHAR_CAP` chars. The model sees contiguous text; the anchor search
   runs over the same window, so anchors and the doc text agree.
6. **Statement slicing.** Anchors are matched whitespace-tolerantly:
   `find_statement._find_anchor` builds a regex from the anchor's normalized
   tokens joined by `\s+`, so an anchor copied with collapsed whitespace
   matches doc text that has mid-sentence line breaks from OCR. The slice is
   `window[open_start : close_end]` — preserves original whitespace.

---

## Models, parallelism, caps

| setting | default | where |
|---------|---------|-------|
| extract model | `MODEL_SONNET` | `segment_a/config.py:65` |
| extract parallel | 4 | `settings.EXTRACT_PARALLEL` |
| extract chunk char cap | 80 000 | `settings.EXTRACT_CHAR_CAP` |
| find_statement model | `MODEL_SONNET` | `segment_a/config.py:65` |
| find_statement parallel | 4 | `settings.STATEMENT_PARALLEL` |
| window margin (pages) | ±10 | `settings.WINDOW_MARGIN_PAGES` |
| window char cap | 60 000 | `settings.WINDOW_CHAR_CAP` |
| statement char cap | 40 000 | `settings.STATEMENT_CHAR_CAP` |

No Opus in this pipeline — exhaustive per-entity enumeration over many
chunks plus a per-entity find_statement call would be too costly. Sonnet
covers both stages.

Costs are aggregated by stage in `index.json` and across docs in
`run_summary.json`. USD numbers are estimates from
`settings.PRICES_USD_PER_M`; verify against the AWS Bedrock invoice.

---

## Entry points

```bash
cd May25/statements_pipeline

python run.py process                  # process every doc in PAGES_DATA_DIR
python run.py process --doc <doc_id>   # process one doc (need not be in inventory)
python run.py process --limit N        # cap to first N docs
python run.py process --force          # ignore raw_extract checkpoint
python run.py status                   # progress + cost summary
```

---

## Caveats specific to this pipeline

- **Anchor verification is window-scoped, not doc-wide.** A statement that
  extends past `WINDOW_MARGIN_PAGES` will produce an unverified closing
  anchor and a `null` statement.text. Bump the margin in `settings.py` if
  your docs have long letters that span more than ~20 pages.
- **No critic.** `needs_human_review` is rules-based only. Rows that pass
  both rules are not auto-validated — they're just not flagged. If you need
  per-row PASS / PASS_WITH_NOTE / RE_EXTRACT / HUMAN_REVIEW verdicts, add a
  critic stage after find_statement and copy the people_pipeline shape.
- **Private-individual policy.** `kind == "individual"` always trips
  `needs_human_review`, matching the v2 policy used by segment_a's and
  people_pipeline's critics. Statement extraction and summarization still
  happen for these rows; the flag just routes the row to a human grader.
- **Sonnet stands in for Haiku** on this Bedrock account (see
  `segment_a/config.py:64–65`). When Haiku 4-5 becomes accessible, the
  extract step is the natural place to switch — extraction over many chunks
  is the most expensive call here.
