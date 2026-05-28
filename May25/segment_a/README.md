# Segment A — Calibration Run

Implements the Segment A pipeline from `Pipeline.pdf` against
`docs_with_digits.json` (flat OCR strings — no PDFs).

## Layout

```
segment_a/
├── config.py        # paths, models, taxonomy, page-estimation constant
├── nul.py           # NUL DC API fetch + accession matching (3 strategies)
├── chunk.py         # chapter detection + 50-page/2-page overlap chunking
├── select.py        # stratified sample of 20 (5 short / 10 medium / 5 long)
├── llm.py           # Anthropic wrapper (Haiku / Sonnet / Opus)
├── m1.py            # title, year, eis_type, lead_agency (NUL-first)
├── m2.py            # summary, alternatives, themes, location, key_people
├── critic.py        # Sonnet per-field critic w/ private-stance override
├── grading.py       # per-doc CSV grading sheet
├── run.py           # orchestrator (subcommands: select | process | status)
├── cache/           # NUL works fetch is cached here
└── output/
    ├── selection.json
    ├── m1/<doc_id>.json
    ├── m2/<doc_id>.json
    ├── critic/<doc_id>.json
    └── grading_sheets/<doc_id>.csv
```

## How to run (in opencode where ANTHROPIC_API_KEY is set)

```bash
cd "May25/segment_a"
pip install requests anthropic geopy  # if not already in your env

# 1. Pick 20 docs (stratified). Writes output/selection.json.
python run.py select

# 2. Smoke test on 1 doc end-to-end (M1 + M2 + Critic + grading sheet)
python run.py process --limit 1

# 3. If the smoke test looks reasonable, do the full 20.
python run.py process

# Inspect progress at any time
python run.py status

# Re-run a single doc, ignoring checkpoints
python run.py process --doc P0491_35556036806768 --force
```

Outputs are checkpointed per-stage per-doc, so rerunning resumes cheaply.

## Key adaptations from the v2 plan

The source is flat OCR text, not paginated PDFs. So:

| Plan says                              | What we do here                                      |
|----------------------------------------|------------------------------------------------------|
| PDF outline/bookmarks                  | Not available — skipped                              |
| Regex TOC + LLM section labels         | Regex chapter detection mapped to CEQ §1502 chapters |
| 50-page chunks with 2-page overlap     | 125_000-char chunks w/ 5_000-char overlap            |
| `source_pages` exact page numbers      | **Estimated** via `char_offset / 2500`               |
| "Quote verbatim against exact page"    | Quote must exist verbatim somewhere in the doc (we record `quote_verified=false` and force HUMAN_REVIEW otherwise) |

All grading-sheet headers note that page numbers are estimated.

## Field-by-field tier choices

| Field        | Primary                           | Fallback / verifier             | Model      |
|--------------|-----------------------------------|---------------------------------|------------|
| title        | NUL                               | Haiku on first page             | Haiku      |
| year         | NUL                               | Regex on first 3 pages          | (none)     |
| eis_type     | Regex on first page               | Sonnet on first 2 pages         | Sonnet     |
| lead_agency  | NUL                               | Sonnet on first 4 pages         | Sonnet     |
| summary      | Opus map-reduce over chunks       | (none)                          | Opus       |
| alternatives | Sonnet on detected Alternatives ch| (none — skipped if no chapter)  | Sonnet     |
| themes       | Sonnet from summary               | (none)                          | Sonnet     |
| location     | Sonnet on first 30 pp + Project/Study Area | Nominatim geocoder      | Sonnet     |
| key_people   | Sonnet on Consultation ch (preparers + cooperating), comment-response sweep for commenters | Verbatim quote check | Sonnet |
| **critic**   | Sonnet, per-field rubric          | Auto HUMAN_REVIEW on parse fail | Sonnet     |

## Grade options for `your_grade`

`correct | minor_issue | wrong | cant_tell`

Fill these in the per-doc CSVs. Segment B (M-Cal) uses them to calibrate
the Critic and lock per-field accuracy targets.
