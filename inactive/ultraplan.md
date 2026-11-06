# Tiered Stage 3 Critic for `eis_pipeline/` — Refined Plan

## Context

The repo at `/home/user/repo/eis_pipeline/` is the v1 prototype described in `about_pipeline/PIPELINE_OVERVIEW.md`. The current Stage 3 critic (`pipeline/stage3_critic.py`) runs four deterministic checks (`_check_quotes`, `_check_year`, `_check_themes`, `_check_geocoding`) and two LLM critics on **summary** and **historical_internal** only, both at `MODELS["critic"] = claude-sonnet-4-6`. Failures emit a stringified warning and downgrade the field's `status` to `insufficient_information`. Existing test runs (`GKG_Tests/TEST_RUN_3_HARLEM_V2.md`) show: (a) the budget guard already trips on critic during stakeholder-heavy docs, (b) no structured per-field verdicts exist — just unparseable warning strings — and (c) most extracted fields get no LLM check at all.

This plan extends that critic to be **tiered by field difficulty**, replaces stringified warnings with a structured `validation` block on the record, and treats Grace's review of ~10 hand-picked docs (not an adjudicated gold set) as the quality bar. The intended outcome: running `python run.py` on each of Grace's 10 docs produces one final JSON file with per-field critic verdicts that she can compare against her own reading of the source EIS.

The draft plan referenced `spring_pipeline/`, `v1_multiagent_pipeline/`, `inter_agent_plan.md`, and `v1_multiagent_plan.md` — **none of these exist in this repo**. All edits in this plan target the actual `eis_pipeline/` layout.

---

## Approach — Tiered Critic

Three layers from the existing critic stay as-is:
- **Layer 1**: Pydantic validation (already runs at end of `run.py` lines 228–233).
- **Layer 2a**: Deterministic hard gates — `_check_quotes`, `_check_year`, `_check_themes`, `_check_geocoding`. Keep in `stage3_critic.py`.
- **Layer 2b**: Self-consistency / cross-field checks (new but small — e.g. lead_agency ↔ first chunk text).

**Layer 3 — LLM critic — becomes tiered.** Each critic call returns a `CriticVerdict`:

```python
class CriticVerdict(BaseModel):
    field_path: str                # "summary.text", "alternatives_proposed[2].description"
    tier: Literal["haiku", "sonnet", "opus", "skipped_budget_cap"]
    verdict: Literal["pass", "partial", "no", "skipped"]
    evidence_quote: str | None
    reasoning: str
```

Verdict effects:
- `pass` → no change.
- `partial` → field keeps its value; verdict logged; field downgraded for review queue (no change to existing `status` Literal — the side-channel `validation` block carries the nuance).
- `no` → for summary / historical_internal: downgrade `status` to `insufficient_information` (preserving current behavior); for other fields: keep value, mark field for review.

### Field → tier → check table

Field paths use the **actual schema names** from `pipeline/schema.py`:

| Field path | Tier | Source span | What it confirms |
|---|---|---|---|
| `title` | Haiku | first 3 pages + any METS-derived cover text | Title appears verbatim (case-insensitive). Replaces no current check — title is currently trusted from METS without verification. |
| `year` | Haiku | first 5 pages | Four-digit year appears within 200 chars of "Environmental Impact" / publication / signature markers. Complements existing `_check_year` range gate. |
| `eis_type` | Haiku | first 250 words (same span Stage 0 reads) | The literal phrase ("Final Environmental Impact Statement" etc.) is present verbatim. Currently set by regex with no LLM cross-check. |
| `lead_agency.name` | Haiku | first 3 pages | Canonical agency string or known alias from `AGENCY_VOCAB` (in `config.py`) appears verbatim. |
| `alternatives_proposed[].name` | Haiku | first 500 tokens after the label in any `alternatives`-tagged chunk | Each label appears as a heading-like line in the cited chunk. |
| `summary.text` | Sonnet | summary's cited chunks (cap 8k tokens), reusing `summary.evidence` | Issue + scope + decision grounded in source; no overreach. Replaces existing `_llm_critic_summary` but with structured verdict output. |
| `alternatives_proposed[].description` | Sonnet | first 500 tokens after the alt label in the cited chunk | Description grounded in that span; no invented features. |
| `key_people_and_groups[].stance` + `quote.text` | Sonnet | the `stance_evidence` chunks + ±150 chars context around the quote | Quote supports the labeled stance and belongs to the labeled author. Builds on the deterministic substring check (`_check_quotes`) which only confirms the quote exists, not that it supports the stance. |
| `themes.primary` | Sonnet | `summary.text` | The "this doc is about <theme>" claim is grounded in the summary. |
| `location.name` | Sonnet | first chunk + any `affected_environment` chunks | The named place is the project area, not a context reference. |
| `historical_context_internal.claims[]` | Sonnet | each claim's cited chunk | Per-claim evidence verification. Replaces existing `_llm_critic_historical_internal`. |
| any field where Sonnet returned `no` AND a non-Sonnet retry might rescue it | Opus retry | same span Sonnet saw | Binding final adjudication. Capped at **3 retries / doc** (`OPUS_RETRY_BUDGET_PER_DOC`). First thing shed under budget pressure. |

**Why this split.** Haiku checks are presence/format only — they cost cents and catch the high-volume garbage. Sonnet checks need to read meaning; Haiku would wave through paraphrases. Opus retries rescue Sonnet false-negatives on fields where a single bad verdict would block a doc from shipping.

### Single parameterized prompt

Create `eis_pipeline/prompts/4_critic.txt` (new directory). One template with slots `{tier}`, `{field_path}`, `{claim}`, `{source}`, `{output_schema}`, plus per-tier few-shot blocks bundled in the template. The expected output schema is the `CriticVerdict` JSON shape above. Note: existing `_CRITIC_SYSTEM` in `stage3_critic.py` uses `"yes" | "no" | "partial"` — switch to `"pass" | "no" | "partial"` for consistency with the verdict-effects semantics.

### Budget guard

Existing `LLMClient` already raises `BudgetExceededError` when `--budget-usd` is set. Reuse it. Add to `pipeline/config.py`:

```python
CRITIC_BUDGET_PER_DOC_USD = 3.00   # default critic-stage cap; can be tightened later
OPUS_RETRY_BUDGET_PER_DOC = 3
CRITIC_TIERS: dict[str, str] = {   # field_path → "haiku" | "sonnet"
    "title": "haiku",
    "year": "haiku",
    "eis_type": "haiku",
    "lead_agency.name": "haiku",
    "alternatives_proposed[].name": "haiku",
    "summary.text": "sonnet",
    "alternatives_proposed[].description": "sonnet",
    "key_people_and_groups[].stance": "sonnet",
    "themes.primary": "sonnet",
    "location.name": "sonnet",
    "historical_context_internal.claims[]": "sonnet",
}
```

Under budget pressure, tiers degrade **in this order**: drop Opus retries → drop Sonnet → drop Haiku. Each skipped field gets `verdict: "skipped"`, `tier: "skipped_budget_cap"`. Pydantic + Layer 2a + Layer 2b always run regardless of budget.

---

## Quality Regime — No Gold Set

Per-doc auto-approve gate (computed in the critic aggregator and stored on the record):
- ≥ 90% of critic-eligible fields return `pass` AND
- 0 hard-gate (Layer 2a) failures AND
- 0 Pydantic violations
- → `review_status: "auto_approved"`.

1 field `no` and the rest `pass`/`partial` → `review_status: "partial_review"` (single-field queue, doc still ships).
2+ `no` → `review_status: "full_review"`.

**Review set (replaces the canonical 10-doc adjudicated gold).** Grace picks ~10 docs spanning short / medium / long / has-comment-section / regression-doc. Cadence: mandatory before any change to `prompts/4_critic.txt`. Disagreement rule: if Grace and the critic disagree on the same `field_path` in ≥ 3 of 10 docs → roll back the prompt change OR add a targeted few-shot example to `prompts/4_critic.txt` and re-run.

Batch-level monitors (cheap, no labels needed) — surface in the existing `output/token_ledger.json` output and a sibling `output/critic_summary.json`:
- Critic pass-rate per `field_path` per run.
- p50/p90 critic cost per doc.
- `other` theme rate per run (already a known monitor target).

---

## Build Sequence

```
                          existing
                   ┌─────────────────────────┐
                   │ Stage 0  Stage 1  Stage 2│
                   └─────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────────────┐
        │ Stage 3 — Critic (rewired)                       │
        │                                                  │
        │  Layer 1: Pydantic validation (unchanged)        │
        │                                                  │
        │  Layer 2a: hard gates (existing _check_*)        │
        │    _check_quotes, _check_year,                   │
        │    _check_themes, _check_geocoding               │
        │                                                  │
        │  Layer 2b: self-consistency (new, small)         │
        │                                                  │
        │  Layer 3: tiered LLM critic (NEW)                │
        │  ┌─────────────────────────────────────────────┐ │
        │  │  critic_router.dispatch(record, client)     │ │
        │  │                                             │ │
        │  │   per field_path → CRITIC_TIERS lookup     │ │
        │  │      │                                      │ │
        │  │      ├── Haiku    (presence checks)         │ │
        │  │      │     title, year, eis_type,           │ │
        │  │      │     agency.name, alt names           │ │
        │  │      │                                      │ │
        │  │      └── Sonnet   (grounded reasoning)      │ │
        │  │            summary, alt descs, stance,      │ │
        │  │            themes, location, hist_internal  │ │
        │  │                │                            │ │
        │  │                └── on "no" → Opus retry     │ │
        │  │                     (cap 3 / doc)           │ │
        │  │                                             │ │
        │  │  Budget pressure: shed Opus → Sonnet → Haiku│ │
        │  └─────────────────────────────────────────────┘ │
        │                                                  │
        │  Aggregate → EISRecord.validation                │
        │     { verdicts: [CriticVerdict, ...],            │
        │       critic_pass_rate: float,                   │
        │       review_status: auto_approved |             │
        │                      partial_review |            │
        │                      full_review }               │
        └──────────────────────────────────────────────────┘
                              │
                              ▼
                   Pydantic write → output/<doc>.json
```

Implementation order — each step independently testable:

1. **Schema** (`pipeline/schema.py`)
   - Add `CriticVerdict` and `CriticValidation` models.
   - Add `EISRecord.validation: CriticValidation = Field(default_factory=CriticValidation)`.
   - **Do not** modify existing field models (`SummaryField.status` etc.) — side-channel only.

2. **Config** (`pipeline/config.py`)
   - Add `CRITIC_TIERS`, `OPUS_RETRY_BUDGET_PER_DOC`, `CRITIC_BUDGET_PER_DOC_USD`.

3. **Prompt** (`pipeline/prompts/4_critic.txt`, new directory)
   - One parameterized template with per-tier few-shot blocks.
   - Single output schema: `CriticVerdict`-shaped JSON.

4. **Critic core** — refactor `stage3_critic.py` in place:
   - Keep `_check_quotes`, `_check_year`, `_check_themes`, `_check_geocoding` unchanged.
   - Replace `_llm_critic_summary` and `_llm_critic_historical_internal` and `_run_claim_critic` with a single new `_run_critic_call(tier, field_path, claim, source, client) -> CriticVerdict`.
   - Add `_dispatch_critics(record, client) -> CriticValidation` that walks the `CRITIC_TIERS` table, calls `_run_critic_call` per field, applies the Opus retry rule on Sonnet `no`s up to `OPUS_RETRY_BUDGET_PER_DOC`, and assembles the `CriticValidation` block.
   - Add a small budget-pressure helper that consults `client.usage.total_cost_usd` against `CRITIC_BUDGET_PER_DOC_USD` before each tier and degrades accordingly. Mark skipped fields with `tier="skipped_budget_cap"`.
   - `run(record, client)` now: Layer 2a checks → Layer 2b → `_dispatch_critics` → attach to `record.validation`. Returns the same warning-list shape (so `run.py` is unchanged) but additionally populates `record.validation`.

5. **Critic aggregator** — inside `stage3_critic.py`:
   - Compute `critic_pass_rate = pass / (pass + partial + no)` over fields with non-`skipped` verdicts.
   - Compute `review_status` per the rules in Quality Regime above.

6. **Wiring** — no edit to `run.py` needed; `stage3_critic.run` already runs at line 208. Behavior changes are isolated to that function.

7. **Lightweight batch monitor** — extend `pipeline/token_ledger.py` (or add `pipeline/critic_summary.py` reading from a sibling ledger file) to write `output/critic_summary.json` with: per-`field_path` pass-rate across the runs in the ledger, p50/p90 critic cost. Don't over-build — a small append-on-each-run JSON is enough.

8. **Test writeup** — append `GKG_Tests/TEST_RUN_4_*.md` (the next number after `TEST_RUN_3_HARLEM_V2.md`) for the first sanity-set doc Grace picks; one writeup per sanity-set doc in the same series.

### Files to create

- `eis_pipeline/prompts/4_critic.txt` — parameterized critic prompt.

### Files to edit

- `eis_pipeline/pipeline/schema.py` — add `CriticVerdict`, `CriticValidation`, `EISRecord.validation`.
- `eis_pipeline/pipeline/config.py` — add `CRITIC_TIERS`, `OPUS_RETRY_BUDGET_PER_DOC`, `CRITIC_BUDGET_PER_DOC_USD`.
- `eis_pipeline/pipeline/stage3_critic.py` — refactor Layer 3 into `_dispatch_critics` + per-field `_run_critic_call`; keep `_check_*` helpers; preserve the existing `run(record, client) -> list[str]` signature so `run.py` is untouched.
- `eis_pipeline/pipeline/token_ledger.py` — add a small `write_critic_summary` helper writing to `output/critic_summary.json` (or a sibling file in the same dir as the ledger).

### Files to read but not modify

- `eis_pipeline/pipeline/llm_client.py` — already has `BudgetExceededError` and per-call usage tracking; reuse via `client.usage.total_cost_usd`.
- `eis_pipeline/pipeline/stage2_fields/summary.py` and siblings — for the field-path mapping; understand which evidence pointers exist per field.
- `eis_pipeline/run.py` — no edits needed.

---

## Risks

1. **Recursive critic.** A miscalibrated critic produces bad ship gates with no external truth.
   - Mitigation: sanity-set review runs before every prompt change; `partial` never fail-closes; Opus retry rescues Sonnet false-`no`s on critical fields; the 3-of-10 disagreement rule gates further prompt edits.

2. **Per-field critic cost.** Going from 2 Sonnet calls (current) to ~5 Haiku + ~6 Sonnet calls per doc raises cost. Current run cost on Harlem v2 was $4.42 with the $4 cap tripping mid-critic. Mitigation: tier ordering means Opus retries are shed first under pressure; raise `--budget-usd` to $5 for the sanity-set run, re-baseline, then tighten.

3. **Existing `summary.status` downgrade behavior.** The current critic downgrades `summary` and `historical_internal` to `insufficient_information` on any `no`. New side-channel does not change this — preserve the existing downgrade behavior for those two fields specifically, so the schema-level `status` field stays consistent with what consumers already read.

4. **Schema additivity.** Adding `EISRecord.validation` with a default factory means all existing test fixtures and `harlem_v2.json` etc. will still validate; no breaking change to consumers.

---

## Verification

End-to-end test of the modified Stage 3:

1. **Unit test** — add `eis_pipeline/tests/test_critic_router.py`. Build an `EISRecord` fixture with deliberately-wrong values (title not in cover text, summary citing a chunk that doesn't contain the claim, stance quote pulled from an agency response paragraph). Expectations:
   - title → Haiku `no` → `record.validation.verdicts` contains a `pass="no"` entry; `review_status` reflects it.
   - summary → Sonnet `no` → `summary.status` downgraded to `insufficient_information` (preserving current behavior) AND a verdict entry exists.
   - stance → Sonnet `no` → verdict entry exists; field value preserved.

2. **Integration on a known doc** — `python run.py --json-file <path> --doc-key p1074_35556036099737 --output output/harlem_v3.json --budget-usd 5.00`. Inspect the output: every critic-eligible field listed in `CRITIC_TIERS` has a corresponding entry in `output.validation.verdicts` with `{field_path, tier, verdict, evidence_quote, reasoning}`. `validation.critic_pass_rate` and `validation.review_status` are populated.

3. **Budget-cap behavior** — re-run with `--budget-usd 0.10` (small but > Stage 0+1+2 cost is unlikely so the cap will hit inside Layer 3). Confirm: Pydantic + Layer 2a + Layer 2b still completed; verdicts present for fields whose tier ran before exhaustion; remaining fields show `verdict: "skipped"`, `tier: "skipped_budget_cap"`; `total_cost_usd` is at or just over the cap; the doc still writes successfully (this matches the existing graceful-degrade behavior — see `run.py:192–198`).

4. **Sanity-set ship deliverable** — Grace picks ~10 docs from the corpus, mixing short / medium / long / has-comments / known-regression docs. Run the pipeline on each, producing 10 JSON files. For each doc, append a `GKG_Tests/TEST_RUN_{N}_{SHORTNAME}.md` writeup following the existing template (the `TEST_RUN_3_HARLEM_V2.md` structure). Grace reads each JSON's `validation.verdicts` block and compares against her reading of the source EIS. If ≥ 3 of 10 disagree on the same `field_path`, iterate on `prompts/4_critic.txt` and re-run.

5. **Cost re-baseline** — read `output/token_ledger.json` after the 10-doc run; compute p50 and p90 per-doc cost. If p90 > $3 with the Opus retry tier engaged, the most likely culprit is Sonnet-on-stakeholders with large comment sections (the Harlem v2 finding); tune by tightening the chunk-count cap on the stance critic before raising the budget further.

6. **Schema regression** — `pytest eis_pipeline/tests/` should pass without modification of existing tests (the new `validation` field has a default factory; no fixtures break).

No code may ship Stage 3 changes until steps 1–4 pass.