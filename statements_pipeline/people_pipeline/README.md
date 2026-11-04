# people_pipeline — exhaustive (entity, stance) extraction

A second pipeline that reuses Segment A's chunking, LLM client, and
verbatim-quote-checking machinery to produce an exhaustive list of every
stance-bearing entity in each document.

Runs against the same 20 docs as `segment_a/output/selection.json` so the
results align with the calibration sample.

## Output schema

One JSON file per doc at `output/entries/<doc_id>.json`:

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
        { "chunk_index": 6, "evidence_pages": ["142-143"], "quote": "...",
          "quote_verified": true, "stance_basis": "calls the proposal 'unacceptable'",
          "entity_as_written": "Sierra Club", "role_as_written": "" }
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

## Design choices

| decision               | value |
|------------------------|-------|
| who counts as "person" | anyone or anything with an attributed stance — individuals, officials, orgs, agencies, tribes, governments |
| stance vocabulary      | closed set: `in_favor`, `opposed`, `conditional`, `neutral`. Entries without a clearly attributed stance are dropped at extract time |
| dedup                  | one row per `(entity, stance)` pair. If the same entity holds two different stances in the doc, two rows |
| sequence               | order of first appearance (lowest chunk_index in the merged group) |
| docs                   | the same 20 from `segment_a/output/selection.json` |
| chunking               | reused as-is from `segment_a/chunk.py` (50-page chunks, 2-page overlap) |
| LLM                    | Sonnet for extract + critic (no Opus — exhaustive enumeration over many chunks) |

## Pipeline stages (per doc)

1. **Chunk** — `segment_a/chunk.py` (50-page chunks, CEQ-chapter labels where detected)
2. **Extract** (`extract.py`) — Sonnet on each chunk in parallel, returns every stance-bearing entity it finds. Entries without a recognized closed-set stance are dropped immediately.
3. **Verify** (`verify.py`) — every quote is checked against the full doc text (whitespace-normalized). Quotes that aren't found verbatim keep `quote_verified=false` and force `HUMAN_REVIEW` later.
4. **Merge** (`merge.py`) — group by `(normalized_entity, stance)`. Pick the longest verified quote as `summary_quote`, dedupe evidence pages, keep all per-chunk mentions, assign `sequence` by first appearance.
5. **Critic** (`critic.py`) — Sonnet rubric per merged row, with the cited pages **pulled in as actual text** (per the v2 spec for M2 Check). Returns `PASS / PASS_WITH_NOTE / RE_EXTRACT / HUMAN_REVIEW`.
   Hard overrides:
   - `summary_quote_verified == false` → forced `HUMAN_REVIEW`
   - `kind == "individual"` → forced `HUMAN_REVIEW` (matches v2 policy that private-individual stance attributions always go to a human)

## How to run

```bash
cd "May25/people_pipeline"

# Smoke test: one doc end to end
python run.py process --limit 1

# A specific doc
python run.py process --doc P0491_35556036107910

# All 20
python run.py process

# Re-run a doc, ignoring checkpoints
python run.py process --doc P0491_35556036107910 --force

# Progress
python run.py status
```

## Checkpoints

- `output/raw_extract/<doc_id>.json` — per-chunk extractor output. Reruns skip the extractor unless `--force` is passed.
- `output/entries/<doc_id>.json` — final per-doc output. Always rewritten on each run (verify/merge/critic re-run from the cached extract).

## Caveats

- **Page numbers are estimated** from char offsets at 2500 chars/page. Same as Segment A.
- **Sonnet stands in for Haiku** on this Bedrock account (see `segment_a/config.py`). When Haiku 4-5 becomes accessible, the extract step is a natural place to use it — extraction over many chunks is the most expensive call in this pipeline.
- The merge step is intentionally conservative about deduping entity names. Over-merging (collapsing two distinct people because of similar names) is worse than under-merging, since each merged row becomes one grading row.
