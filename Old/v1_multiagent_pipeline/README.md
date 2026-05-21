# v1_multiagent_pipeline

EIS metadata extraction pipeline, v1 build per `../v1_multiagent_plan.md`.

**Status:** Group A + B (scaffold + ingest + Stage 2 + Stage 3a/3b). Other groups deferred to future sessions.

## Layout

```
v1_multiagent_pipeline/
├── run.py                       # CLI
├── pipeline/
│   ├── config.py                # model IDs, taxonomies, regex, agency vocab
│   ├── schema.py                # Pydantic v2 EISRecord (per inter_agent_plan §0)
│   ├── llm_client.py            # Anthropic wrapper, dry-run, budget cap
│   ├── token_ledger.py          # persistent per-run usage tally
│   ├── nul_client.py            # NUL Digital Collections API + disk cache
│   ├── ingest.py                # Stage 1 — load doc, normalize, fake pages, alignment map
│   ├── grouping.py              # Stage 1.5 — passthrough in v1
│   ├── sections.py              # Stage 2 — regex + embedding-similarity fallback
│   ├── stage3a_mets_fields.py   # title, year, lead_agency, publication_date (NUL-sourced)
│   └── stage3b_eis_type.py      # regex on cover; Haiku fallback
├── prompts/
│   └── 3b_eis_type_fallback.txt
├── tests/
│   ├── test_schema.py
│   └── test_ingest.py
├── tools/                       # build_manifest.py etc. (Group A+ deferred)
└── output/                      # gitignored
    ├── manifest/
    ├── nul_cache/
    ├── stage1_assembled/
    ├── stage1_5_grouped/
    ├── stage2_sections/
    ├── stage3_fields/           # accumulated per-doc, updated stage-by-stage
    ├── stage4_validated/
    ├── final/
    ├── review_log/
    ├── runs/
    └── token_ledger.json
```

## Acceptance gate (Group A + B)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
DOCS="/Users/gracegormley/Desktop/Y2/Q2/Knight Lab/docs_with_digits.json"

# Run through Stage 3b (NUL fields + EIS type)
python run.py \
  --json-file "$DOCS" \
  --doc-key p1074_35556036099737 \
  --through-stage 3b \
  --budget-usd 0.50

# Output: output/stage3_fields/p1074_35556036099737.json
# Should contain: title, year, lead_agency, publication date, eis_type, sections
```

## Reference

- Build brief: `../v1_multiagent_plan.md`
- Canonical schema: `../inter_agent_plan.md` §0
- Existing v1 (do not modify): `../eis_pipeline/`
