# `statements_pipeline/` — Per-Complaint Statement & Response Extraction

A pipeline that walks an Environmental Impact Statement, finds every named
stakeholder taking a position, locates each of their **complaint instances**
in the doc text, captures any **agency response** nearby, and writes a
normalized per-doc folder of JSON files.

The output is normalized in three layers:

1. **`index.json`** — sequential, link-only. Lists complaints in document
   order with their response IDs.
2. **`complaints/`** — parents. One file per complaint instance. Carries the
   complainer's identity, stance, summary, and the full statement text.
3. **`responses/`** — children. One file per agency reply. Carries the
   responding agency, the response form (`agency_response` /
   `preparer_reply` / `discussion`), the summary, and the full response text.

A single complainer (e.g. Sierra Club) may produce **multiple complaint files**
in the same doc — typically one for the paraphrased summary in a
Comments-and-Responses section AND one for the full letter in an appendix.
They're linked by a shared `complainer_id`.

## ID scheme

Doc-prefixed, globally unique:

| ID | shape | one per |
|---|---|---|
| `complainer_id` | `<doc_id>_K<NNN>` | merged entity |
| `complaint_id`  | `<doc_id>_C<NNN>` | complaint instance |
| `child_id`      | `<doc_id>_R<NNN>` | agency response |

`complainer_id` is shared across all of an entity's complaint files;
`complaint_id` and `child_id` are unique per file.

## Per-doc flow

```
chunk (segment_a)
  → extract: discovery only, no stance        (local, shadows people_pipeline/extract.py)
  → verify: verbatim quote check              (people_pipeline/verify.py, reused)
  → merge by ENTITY                           (local; no stance key)
  → find_statement + stance + responses       (local; one LLM call per entity,
                                              returns multiple complaints + responses)
  → write complaints/, responses/, index.json (local)
```

1. **Doc text load** (`pages.load_doc` from `segment_a/`).
2. **Chunk** (`segment_a/chunk.py`). 50-page page-aligned chunks, 2-page overlap.
3. **Extract — discovery only** (`extract.py`, local). Sonnet on each chunk
   returns every named entity whose POSITION is attributable. No stance.
4. **Verify** (`people_pipeline/verify.py`). Quote-verbatim check.
5. **Merge by entity** (`merge.py`, local). One row per normalized entity.
6. **Find statement + stance + response** (`find_statement.py`). For each
   merged row, build a doc-text window around the entity's evidence pages
   (`±10` pages by default, capped at `WINDOW_CHAR_CAP=60_000` chars), then
   ask Sonnet to:
   - return a **`complaints` array** — one entry per distinct mention, each
     with `form`, verbatim opening/closing anchors, `evidence_pages`, and a
     1-sentence `complaint_summary`
   - attach an optional **`response`** to each complaint, with its own form,
     anchors, responding agency, and 1-2 sentence summary
   - judge entity-level **`stance ∈ {in_favor, opposed, conditional, neutral}`**
     unified across all complaints
   - report **`stance_confidence ∈ {high, medium, low}`** and a short
     `stance_basis`
   - return a 2-3 sentence entity-level `summary`
7. **Anchor verification + slice**. Anchors (statement and response) are
   matched whitespace-tolerantly and biased toward the entity's evidence
   pages. The closing anchor is preferred near the opening; the response
   anchor is preferred just after the verified complaint. If both anchors
   verify and `closing > opening`, the text is sliced; otherwise `text` is
   `null` but the summary is preserved.
8. **Confidence cap**:
   - all complaints have `form == "none"` (or no complaints) → forced `low`
   - no contiguous statement verified in any complaint → capped at `medium`
   - else → take the model's value
9. **Write** (`writer.py`).
   - The doc dir (`output/people/<doc_id>/`) is **wiped** first to prevent
     ghost files from prior runs.
   - Each complaint becomes a parent file at
     `complaints/<doc_id>_C<NNN>_<slug>.json` with `complainer_id` linking
     back to siblings.
   - Each response becomes a child file at
     `responses/<doc_id>_R<NNN>_<agency_slug>.json` referencing its
     `parent_id`.
   - `index.json` lists the sequential order plus full per-record
     summaries.

There is **no LLM critic in this pipeline.** Review flagging is rules-based.

## Why split into complainer / complaint / response?

The earlier denormalized shape (one file per entity, with `statement` and
`response` embedded) failed in two real cases:

1. **Same complainer, multiple appearances.** Sierra Club paraphrased on
   page 34 (with an inline agency response right there) and reproduced as a
   full letter on page 142. The denormalized shape can only show one — the
   model picks one form, so either the response gets lost or the full text
   does.
2. **Stable references.** A reviewer linking to a specific complaint or
   response wants a stable ID. With one file per entity, you'd reference
   `output/people/<doc>/<NNN>_<slug>.json#statement.text` — fragile and
   awkward. Now: `complaints/<doc_id>_C014_sierra_club.json` and
   `responses/<doc_id>_R007_blm.json`.

Splitting also makes filtering trivial — load `index.json` and walk the
sequence; load `complaints/` to build a complainer-keyed view.

## Doc source and metadata

- **Doc text:** every doc found under `PAGES_DATA_DIR`
  (`Documents/output/<doc_id>/page_NNNN.json`).
- **Work metadata:** `inventory.lookup_work(doc_id)` from segment_a. When
  the doc isn't in the local CSV, the run still proceeds; `work_id` and
  `title` are left empty.

## Window sizing for `find_statement`

- Page window: `evidence_pages ± WINDOW_MARGIN_PAGES` (default ±10),
  capped at `WINDOW_CHAR_CAP=60_000` chars.
- Anchor matching is biased toward the evidence pages
  (`ANCHOR_PROXIMITY_CHARS=25_000`) so generic anchors lock onto the right
  occurrence within the window.
- Response anchors are biased toward the END of the verified complaint —
  responses follow comments inline.

## Stance and review flagging

Stance is at the **entity level** (same across all of that entity's
complaints) but `stance_confidence` is capped by the structural evidence
across all complaints:

| situation | max confidence |
|---|---|
| no complaints at all | forced `low` |
| all complaints have `form == "none"` | forced `low` |
| no verified contiguous statement (only paraphrase / sectional / unverified) | `medium` |
| at least one verified contiguous statement | up to `high` |

`writer._needs_human_review` flags rows with any of:

| reason | trigger |
|---|---|
| `private_individual` | `kind == "individual"` |
| `quote_not_verbatim` | merged `summary_quote_verified == False` |
| `low_stance_confidence` | `stance_confidence == "low"` |
| `no_complaints` | find_statement returned an empty complaints list |

## Output layout

```
statements_pipeline/output/
├── run_summary.json                     # per-doc paths, counts, usage
├── raw_extract/<doc_id>.json            # checkpoint: per-chunk extractor
└── people/<doc_id>/
    ├── index.json
    ├── complaints/
    │   ├── p1074_058550_C001_sierra_club.json
    │   ├── p1074_058550_C002_john_smith.json
    │   └── ...
    └── responses/
        ├── p1074_058550_R001_blm.json
        └── ...
```

### `index.json`

```json
{
  "doc_id": "p1074_058550",
  "title": "Operation Breakthrough : environmental impact statement.",
  "n_pages": 72,
  "n_chunks": 2,
  "n_complainers": 25,
  "n_complaints": 31,
  "n_responses": 18,
  "stance_counts": { "opposed": 4, "neutral": 11, "in_favor": 8, "conditional": 2 },
  "stance_confidence_counts": { "high": 17, "medium": 6, "low": 2, "unknown": 0 },
  "review_counts": {
    "needs_review": 6,
    "auto_ok": 19,
    "reasons": { "low_stance_confidence": 2, "private_individual": 4 }
  },
  "statement_form_counts": {
    "letter": 19, "narrator_paraphrase": 7, "sectional": 4, "none": 1,
    "present": 30, "absent": 1
  },
  "response_form_counts": {
    "agency_response": 14, "discussion": 4, "preparer_reply": 0, "none": 13,
    "present": 18, "absent": 13
  },
  "elapsed_sec": 47.2,
  "usage": { "extract": {...}, "find_statement": {...}, "total": {...} },
  "schema": { "ids": {...}, ... },
  "sequence": [
    {
      "order": 1,
      "complaint_id": "p1074_058550_C001",
      "complainer_id": "p1074_058550_K003",
      "complaint_file": "complaints/p1074_058550_C001_sierra_club.json",
      "response_ids": ["p1074_058550_R001"]
    }
  ],
  "complaints": [{...full per-complaint summary...}],
  "responses": [{...full per-response summary...}]
}
```

### Complaint file (`complaints/<doc_id>_C<NNN>_<slug>.json`)

```json
{
  "complaint_id": "p1074_058550_C001",
  "complainer_id": "p1074_058550_K003",
  "doc_id": "p1074_058550",
  "work_id": "csv:35556036058550",
  "order": 1,
  "entity": "Sierra Club",
  "kind": "organization",
  "role": "national environmental advocacy org",
  "stance": "opposed",
  "stance_confidence": "high",
  "stance_basis": "opens letter calling proposal 'unacceptable'",
  "summary": "Sierra Club opposes the proposed regulations because ...",
  "form": "narrator_paraphrase",
  "complaint_summary": "Paraphrase: Sierra Club expressed concern that ORV use should be halted in the affected areas.",
  "evidence_pages": ["34"],
  "statement": {
    "text": null,
    "opening_anchor": "",
    "closing_anchor": "",
    "opening_anchor_verified": false,
    "closing_anchor_verified": false
  },
  "response_ids": ["p1074_058550_R001"],
  "needs_human_review": false,
  "human_review_reasons": [],
  "summary_quote": "...",
  "summary_quote_verified": true,
  "attribution_mode": "paraphrased",
  "attribution_modes_seen": ["paraphrased", "direct_quote"],
  "n_mentions": 2,
  "mentions": [...],
  "window_pages": [24, 44],
  "merge_sequence": 4
}
```

A second complaint file `p1074_058550_C014_sierra_club.json` with the same
`complainer_id` would carry the full letter (`form: "letter"`, populated
`statement.text`, etc.). The two are linked by the shared `complainer_id`.

### Response file (`responses/<doc_id>_R<NNN>_<agency_slug>.json`)

```json
{
  "child_id": "p1074_058550_R001",
  "parent_id": "p1074_058550_C001",
  "complainer_id": "p1074_058550_K003",
  "doc_id": "p1074_058550",
  "work_id": "csv:35556036058550",
  "agency": "Bureau of Land Management",
  "agency_kind": "agency",
  "form": "agency_response",
  "summary": "BLM acknowledges the concern and commits to additional mitigation in the final EIS.",
  "text": "Response: The Department recognizes the concerns raised by Sierra Club ... addressed in the final EIS.",
  "opening_anchor": "Response: The Department recognizes",
  "closing_anchor": "addressed in the final EIS.",
  "opening_anchor_verified": true,
  "closing_anchor_verified": true,
  "complaint_evidence_pages": ["34"]
}
```

## Module shadowing (how local extract/merge replace the upstream ones)

`settings.py` appends `segment_a/` and `Old/people_pipeline/` to `sys.path`
(`sys.path.append`, not `insert`). Local-module-wins on name collisions:

| name | resolves to |
|---|---|
| `chunk`, `pages`, `inventory`, `config`, `llm` | `segment_a/` |
| `verify` | `people_pipeline/` (reused) |
| `extract`, `merge` | **local** (this directory) |
| `find_statement`, `writer`, `settings`, `run` | local |

## Checkpoints and reruns

- `output/raw_extract/<doc_id>.json` — per-chunk extractor output. Reused
  on rerun unless `--force` is passed.
- `output/people/<doc_id>/` — final output. **Wiped at the start of each
  write_doc** so reruns produce a clean slate; no ghost files from prior
  runs.

## Entry points

```
python run.py process              # all docs in PAGES_DATA_DIR
python run.py process --doc <id>   # one doc
python run.py process --limit N    # first N docs
python run.py process --force      # ignore raw_extract checkpoint
python run.py status               # progress + cost summary
```

Per-doc log line:

```
Wrote .../people/<doc_id> (25 complainers, 31 complaints, 18 responses; conf=H17/M6/L2 needs_review=6) in 47.2s — est. cost $0.7421
```

## Cost shape

Multi-complaint output adds output tokens but only when entities actually
have multiple mentions (rare). Most rows still produce 1 complaint, so
typical cost is similar to the previous schema. With ~2 complaints/entity
average for densely-discussed entities, +50-150 output tokens per call.

## Helpers

- `python .check_cost.py` — per-doc cost table from `run_summary.json`.
- `python .recover_failed.py [--apply]` — identifies docs whose complaint
  files contain `find_statement_error` and prints (or runs) the
  `rm -rf output/people/<doc>/` + re-run commands.

## Known issues / TODO

- **No LLM critic.** Review flagging is rules-based only.
- **One response per complaint.** The schema supports a list (`response_ids`)
  but find_statement currently returns at most one response per complaint.
  Easy to extend if you need it.
- **Paraphrase-only complaints get `quote_not_verbatim`** because their
  exemplar quote was the narrator's paraphrase, not a verbatim line. Known
  false-positive bucket.
- **No agency normalization.** Two responses by "BLM" and "Bureau of Land
  Management" are written to two distinct response files. A future pass
  could canonicalize agencies and group responses.
