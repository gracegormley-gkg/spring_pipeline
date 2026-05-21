# CLAUDE_CONTEXT.md — Catch-up for Future Sessions

**Read this first.** Then `PIPELINE_OVERVIEW.md` (architecture), `PROMPTS.md` (every LLM prompt), `GKG_Tests/TEST_RUN_*.md` (most recent results — Test Run 3 = current state).

---

## What this project is

A metadata-extraction pipeline (`eis_pipeline/`) that turns 1970s–90s U.S. Environmental Impact Statement PDFs into validated structured JSON records. v1.0, single-doc-per-invocation, written by Grace, working with Claude as a coding collaborator.

Inputs: either S3 folder format (`P0491_<DOC_ID>/` with TXT/CONFIDENCES/mets.xml) or a flat `docs_with_digits.json` (current working format — splits text into ~2500-char fake pages, fetches METS-equivalent metadata from the NUL Digital Collections API).

Output: one Pydantic-validated `EISRecord` JSON per doc.

---

## Where things live

```
EIS Job/
├── PIPELINE_OVERVIEW.md       <- architecture, stages, schema (read first)
├── PIPELINE_PLAN.md           <- original implementation spec / rationale
├── PROMPTS.md                 <- every LLM prompt with model-choice reasoning
├── eis_pipeline/              <- the actual code
│   ├── run.py                 <- CLI entrypoint
│   ├── inspect_layout.py      <- run before first S3-format doc
│   ├── fetch_data.py          <- S3 downloader
│   ├── pipeline/
│   │   ├── config.py          <- agency vocab, theme taxonomy, regex, model IDs, thresholds
│   │   ├── schema.py          <- Pydantic EISRecord + sub-models
│   │   ├── io_layer.py        <- load_document (S3) + load_from_digits_json
│   │   ├── llm_client.py      <- Anthropic wrapper, dry-run, budget cap, per-model usage
│   │   ├── ner_dicts.py       <- federally recognized tribes + environmental NGOs lists
│   │   ├── token_ledger.py    <- persistent per-run token usage tally
│   │   ├── stage0_triage.py   <- deterministic: OCR, EIS type, headings, date, agency, layered NER
│   │   ├── stage1_chunking.py <- AI-TOC pass + splits + Haiku chunk labeling
│   │   ├── stage2_fields/     <- one file per field; retrieval.py shared helper
│   │   └── stage3_critic.py   <- Sonnet per-claim verification + deterministic checks
│   ├── tests/                 <- test_stage0 (incl. heading + full-name regressions), test_io_layer
│   └── output/                <- pipeline outputs + token_ledger.json
├── GKG_Tests/                 <- real run outputs + writeups (Apollo, Harlem v1, Harlem v2)
└── Old Work/                  <- previous iteration of project + website (EIS-Final, PipelineV1_Storage). Reference for what we're improving on. Don't modify.
```

---

## Models in use (config.py MODELS dict)

| Role | Model | What it does |
|------|-------|---|
| `heavy` | `claude-opus-4-6` | summary, themes, alternatives, key-people stance/quotes, historical context |
| `light` | `claude-haiku-4-5-20251001` | chunk labeling, location extraction, entity triage |
| `critic` | `claude-sonnet-4-6` | per-claim evidence verification |

Note: `PIPELINE_PLAN.md` references `claude-opus-4-7`; the actual code uses `4-6`. The system-prompt-reported model name (e.g. `anthropic.claude-opus-4-7`) is the model running this conversation, not the pipeline.

---

## Project state — post 5/14-meeting iteration

Three real test runs done. Pipeline works end-to-end. Most issues from the original meeting fix-list have been addressed; the remaining items below were deferred or are new findings from Test Run 3.

### What was done in the 5/14-meeting iteration (2026-05-15)

| Change | Where |
|---|---|
| Heading detector tightened (regex won't cross newlines; post-filter rejects addresses/ZIPs/legal citations) | `config.HEADING_PATTERNS`, `stage0_triage._is_real_heading` |
| Retrieval fallback (tag miss → keyword search → first N chunks) | `summary.py`, `alternatives.py`, `location.py` |
| Layered NER: `en_core_web_trf` for PERSON with full-name regex filter; dict lookups for ~190 tribes + ~50 NGOs + agencies; provenance tracked in `NERResult.sources` | `stage0_triage._ner`, new `pipeline/ner_dicts.py` |
| Haiku gap-filler on stakeholder-dense chunks (capped at 6/doc) | `key_people._gap_fill_ner` |
| Simplified triage — only scrubs spaCy-sourced entities (dict + gap-fill bypass) | `key_people._triage_entities` |
| Schema additions: `KeyPersonOrGroup.role` + `.opinion_summary`, `SummaryField.layman_text`, `NERResult.sources` | `schema.py` |
| Consolidated key-people Opus calls — one call now returns stance + role + opinion + evidence + quote (was 2 calls) | `key_people._extract_entity_pack` |
| Key people sorted by document appearance order (was by frequency × spread) | `key_people.run` |
| Two-pass summary: detailed (Opus, evidence-cited) + layman (Opus, plain-English rewrite of the detailed) | `summary.py` |
| Persistent token ledger (per-model breakdown, lifetime totals) | new `pipeline/token_ledger.py`, wired into `run.py` |
| Alternatives prompt expanded to recognize "Alignment / Variation / Option / Route / Plan / Build" terminology; context limit raised to 100k chars | `alternatives.py` |
| **AI-TOC pass** — Haiku reads beginning/middle/end samples (~15k chars), identifies real section structure via title + anchor phrase; we map anchors to page numbers. Replaces Stage 0 regex sections when AI finds ≥3 locatable sections. | `stage1_chunking._ai_toc` |
| `MAX_KEY_PEOPLE` bumped 30 → 50 ("cut it down less") | `config.py` |

### Open issues (priority order, from TEST_RUN_3_HARLEM_V2.md)

| Priority | Issue | Where to fix |
|---|---|---|
| **High** | NER dedupe doesn't normalize variants — "Cook County Forest Preserve District" / "the Cook County Forest Preserve District" / "Cook County Forest\nPreserve District" all kept as separate entities | `stage0_triage._dedup_names`: strip leading "the ", collapse newlines/hyphen-breaks, drop short fragments, optionally fuzzy-merge variants |
| **High** | Generic entities ("United States", "STATE OF ILLINOIS") survive triage and appear in key-people output | `key_people._triage_entities` prompt — explicitly reject country/state names as stakeholders |
| **High** | EIS type returns "Unlabelled" when title clearly says "FINAL" but consultation appendix references "the DRAFT" | `stage0_triage._eis_type()`: title-check tiebreaker |
| **High** | Theme taxonomy lacks `aerospace_and_space_exploration` (Apollo landed on `other`) | `pipeline/config.py` `THEMES` dict |
| **Medium** | Gap-fill re-adds entities already in dict layer (substring overlap, e.g. "Cook County Forest Preserve Commission" vs spaCy's "the Cook County Forest Preserve Commission") | `key_people._gap_fill_ner` — substring-aware dedup against existing entities |
| **Medium** | Date extractor pulls draft submission date from consultation appendix (Harlem returns 1971 for a 1972 doc) | `stage0_triage` date logic — prefer dates near signature blocks |
| **Medium** | Quotes consistently failing substring check (0/16 verified in Harlem v2) — possible OCR text vs LLM output mismatch, or Opus is paraphrasing despite the prompt | Investigate substring check logic; consider char-normalization (smart quotes, hyphens, whitespace) before matching |
| **Low** | Lead agency on Harlem still returns vague "Department of Transportation" instead of FHWA | Add FHWA-specific second-pass regex in agency matcher |

### Known architectural limitations (not bugs, design decisions)

- **External historical context** + **project current status** are v2 stubs. Need search API, Tier 1 allowlist, stricter critic, disambiguation. Schema enforces `deferred_to_v2: True` via `Literal` types — don't try to populate them in v1.
- **Tag-based retrieval has fallbacks but isn't bulletproof.** Each Stage 2 field that uses tag retrieval now has a keyword fallback. Still — fields with idiosyncratic content (like alternatives with "Alignment 1" naming) need explicit prompt support.
- **Fake pages from `docs_with_digits.json`** mean page citations are approximate. Real page numbers only when running against full S3 data.
- **Geocoding match rate ~60%.** Nominatim struggles with corridors and historical place names. Null lat/lon is the correct fallback.

---

## How to run

```bash
cd "/Users/gracegormley/Desktop/EIS Job/eis_pipeline"
export ANTHROPIC_API_KEY=sk-ant-...

# From the flat JSON (current working format) — three short pastes to dodge
# terminal line-wrap issues with the long paths:
DOCS="/Users/gracegormley/Desktop/Y2/Q2/Knight Lab/docs_with_digits.json"
python run.py --json-file "$DOCS" --doc-key p1074_35556036099737 --output "../GKG_Tests/harlem_v3.json" --budget-usd 4

# Useful flags
--dry-run                              # no API calls, log prompts only
--skip-stages 2,3                      # deterministic only
--only-fields summary,themes           # iterate on one prompt
--budget-usd 5.00                      # hard cap
--token-ledger output/ledger.json      # override default ledger path
```

Cost reference (latest runs):
- Apollo (5.5k words, 1 chunk) = $0.39
- Harlem v1 (27k words, 39 chunks) = $2.13
- Harlem v2 (27k words, 13 chunks, full NER + layman summary) = $4.42 (hit $4 budget cap)
- Harlem v3 (with consolidated Opus calls) — **projected ~$3.20–$3.50, not yet run**

Cost is dominated by per-entity stance/role/opinion/quote Opus calls in `key_people`. With consolidation (one call instead of two) the per-doc cost should drop ~$1. Full 181-doc collection projects to roughly $600 at the post-consolidation rate.

---

## Where to find what was tried / what worked

- **Apollo run:** `GKG_Tests/apollo_result.json` + `TEST_RUN_1_APOLLO.md` — short doc, single chunk, summary worked, alternatives empty (tag miss), themes hit `other` (vocab gap)
- **Harlem v1:** `GKG_Tests/harlem_ave_result.json` + `TEST_RUN_2_HARLEM_AVE.md` — medium doc, 39 noise chunks, summary + themes + alternatives strong, key_people weak (5 entities, mostly empty)
- **Harlem v2 (post-meeting fixes):** `GKG_Tests/harlem_v2.json` + `TEST_RUN_3_HARLEM_V2.md` — 13 chunks (heading detector improved), 16 key_people with rich role/opinion/stance, layman summary working, alternatives regressed to empty (now fixed in code, untested), cost $4.42 (now consolidated, projected lower)

Test Run 3 is the current ground truth — read it first.

---

## Next-test recommendation

**Re-run Harlem** (`p1074_35556036099737`) to validate the three post-Run-3 fixes:
1. Alternatives prompt fix (should produce 9 alternatives again — Variations A–E, Alignments 1/2, etc.)
2. Consolidated Opus calls in key_people (should cut cost ~$1)
3. AI-TOC pass (should produce semantically named sections instead of all "Section 4(f) ...")

After Harlem v3 validates, try a longer doc (60–100k words) with a clean section structure. Candidates: `p1074_35556036806552`, `p1074_35556036093169`, or a 50k-range p1074.

---

## Working norms with Grace

- She wants pushback on bad calls, not validation. Be direct.
- When picking up the project, briefly confirm the plan before doing large refactors.
- Test reports are written narrative-style ("✅ Strong / ⚠️ Issue / ❌ Failed" with root-cause analysis). Match that tone if writing new ones.
- Don't proactively expand scope. v1 stubs are stubs on purpose.
