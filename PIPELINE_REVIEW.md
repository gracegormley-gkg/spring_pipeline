# Segment A / Segment B Pipeline Review

**Scope:** `May25/segment_a`, `May25/segment_b`, `May25/mcal`, `mcal/artifacts/v1`, `segment_b/output` (11-doc batch), `Evaluation - Sheet1.csv`
**Baseline reviewed:** v1 artifacts, stage `v1`, 11 documents processed
**Method:** two independent passes (cost/efficiency, correctness/airtightness), then a verification pass over every claim. Claims that failed verification are marked and corrected in place.

---

## 0. Read this first

`mcal/artifacts/v1/weight_validation.v1.json`:

```
production_weighting: quote50_critic50
auroc:      0.5718
auroc_ci95: [0.4683, 0.6545]
```

**The 95% CI on the composite confidence score includes 0.5.** On the 8 graded documents, `0.5·s_quote + 0.5·s_critic` is not demonstrably better than chance at separating a correct extraction from a wrong one. Every downstream mechanism — conformal thresholds, the gate, the human-review queue, the recalibration loop — is fitted on that score.

This is not about self-reported model confidence. That was correctly phased out of Segment B: `gate.signals_for_field` reads `quote_verdict`, `critic_verdict`, `citation_rate`, `shard_rate`, `acronym_rate`, `source_agreement` and never touches `entry["confidence"]`. The problem is in the successor signal.

It shows up mechanically in `thresholds.v1.json`:

| bucket | tau_deployed | saturated | gate_all | consequence |
|---|---|---|---|---|
| M1 | 0.5 | false | false | functioning |
| summary_narrative | 1.0 | **true** | false | nothing can pass |
| summary_numeric | 1.0 | **true** | false | nothing can pass |
| summary_of_interest | inf | false | true | no calibration data (n=0) |
| alternatives+themes | inf | false | true | starved (n=1) |
| location | inf | true | true | force-gated (geocoder) |
| key_people | 1.0 | **true** | false | nothing can pass |

`gate_simulation.v1.json` reports `false_defer_rate: 1.0` on five of seven buckets. Observed batch gate rate: **77%**.

### Will more graded documents fix it?

Partially. `k = ceil((n+1)(1-α))`, so as *n* grows `k/n → 1-α = 0.85` and `tau` becomes the 85th percentile of `R_doc`, not the max. **`tau` drops below 1.0 only once the fraction of wrong-docs scoring at ceiling falls below α = 15%.** Actual distributions:

```
summary_narrative   R_docs=[0.25, 0.5, 0.85, 1.0, 1.0]        40% at ceiling  -> will NOT converge
summary_numeric     R_docs=[0.5, 0.85, 1.0]                   33% at ceiling  -> will NOT converge
key_people          R_docs=[0.25, 0.85, 0.85, 0.85, 1.0, 1.0] 33% at ceiling  -> will NOT converge
location            R_docs=[0.0, 0.6, 0.85, 0.85, 0.85, 1.0]  17% at ceiling  -> BORDERLINE
alternatives+themes R_docs=[0.15]                              0% at ceiling  -> WILL converge (starved on n, not signal)
M1                  R_docs=[0.5, 0.5, 0.5]                     0% at ceiling  -> functioning
```

Three buckets are blocked on **signal count, not sample size**. One (`alternatives+themes`) is blocked purely on `n=1 < DEGENERATE_SEVERE_MIN_WRONG_DOCS=3` and will unlock on grading volume alone — **grade this bucket first.**

The at-ceiling items are the diagnosis. From the Evaluation sheet: `wrong: "Magnitude 7.5" correct: "Magnitude 7.0"`. The quote is verbatim on the page, the Critic passes it, the claim is wrong. `s_quote` and `s_critic` are both blind to it by construction. You need signals that pull *wrong* items off 1.0.

Of the six declared signals, four carry weight 0 and two of those are never computed at all (see §2.2).

---

## 1. Efficiency and cost

Baseline from `segment_b/output/batch_summary.json`: 11 docs, $74.33, **$6.76/doc**, projected **$13,515 / 2,000 docs**. Opus is 87–93% of spend.

Measured on `p1074_35556036525772` (303pp, ~115k tokens OCR, 15 Opus calls, 333,415 Opus input tokens):

```
map calls (7)        ~40% of Opus input
reduce + SOI (2)      ~9%
Opus Critic (6)      ~51%
```

### 1.1 Move the summary map step to Sonnet  — *largest lever*

`m2._summary_map_one` asks Opus to summarize one 50-page chunk into five short fields plus verbatim quotes. That is extraction and copying; the synthesis happens in `_summary_reduce`. Map is ~40% of Opus input and a large share of Opus output, at a 5× price differential ($3/$15 vs $15/$75 per M).

**Caveat, from your own `config.py` comment:** changing an extractor's model mid-stage shifts the score distribution the frozen thresholds were fitted to. This must land **between** calibration stages with a refit. Given that three buckets are saturated and passing nothing today, there is no working calibration to protect — this is the cheapest moment you will ever have.

Also worth testing: the six Opus Critic calls are ~51% of Opus input. Same constraint applies.

**Est. 40–55% total cost reduction.**

### 1.2 Use the Batch API

No latency requirement on a 2,000-doc corpus. Flat 50% discount, composes with everything else. Current wall clock is 250–500 hours sequential (252–898 s/doc observed). Batch should be the default execution mode.

**Est. 50% on whatever remains.**

### 1.3 Prompt caching — smaller than it looks

`cache_read_input_tokens: 0` on every call in every document; no `cache_control` anywhere in the codebase.

But caching requires a **shared prefix**, and this pipeline is built so the expensive calls deliberately don't have one: each map call gets a different chunk, each Critic call gets a different cited-page window. Genuine wins:

- `_summary_of_interest` reuses `_summary_reduce`'s per-chunk findings payload → ~4% of Opus input
- per-field Critic system prompts are constant across documents → ~5% under batch TTL

**Est. 8–12% of total.** Not the 40%+ it first appears to be.

A large win is available by making the whole document a cached prefix for all 15 Critic calls — but that undoes the cited-pages restriction, which is load-bearing (see §2.7). Genuine design conflict, not an oversight.

### 1.4 Chunk selection wastes the budget it protects

`m2._summary_pipeline`: `selected = (tagged + untagged)[:max_chunks]`. With zero detected CEQ chapters this is *the first 12 chunks in document order* — front matter and Purpose & Need — and Environmental Consequences is never read.

Binds on 1 of 12 docs in this batch (the 775pp doc, 17 chunks). Will bind more at 2,000-doc scale. Compounded by weak chapter detection (§2.9).

### 1.5 Minor

- `m1.extract_eis_type` calls Sonnet on every document even when the page-1 regex is unambiguous.
- `m1.extract_lead_agency`'s Sonnet fallback fires on **4.4%** of inventory rows (2,392 of 54,220 lack `710$b`) — rarely reached, not dead. ~88 docs at full scale.
- `llm.call_with_usage` catches bare `Exception` and retries 3× with backoff on everything except `TruncatedResponse`. A 400 or auth failure burns two sleeps and two calls. Classify before retrying.
- `process_doc` checkpoints at `m1.json` / `m2.json` / `critic.json`. A failure in the reduce step discards all completed Opus map calls — the most expensive artifact in the pipeline. **Persist `partials` before reducing.**
- `SUMMARY_REDUCE_MAX_TOKENS = 24_000` is defended as "a cap, not a target." Observed output is 16–32k tokens/doc across 11–15 Opus calls, so it isn't binding — but that's asserted, not measured. Log the output-token distribution per call type.

**Combined projection:** $6.76/doc → **~$1.60–2.20/doc**, or **$3,200–4,400** for the full corpus.

---

## 2. Correctness and airtightness

### 2.1 The gate misreports its own failure mode  — `segment_b/gate.py`

`gate_field` checks `threshold.gate_all_to_human` and `threshold.degenerate_severe`, but **never reads `threshold.saturated`**. `saturated` appears only in `thresholds_from_payload` (rehydration) and `bucket_flags` (reporting). So a saturated bucket falls through to `composite_below_tau` — which `gate.py`'s own module docstring defines as *"the genuine 'gate too conservative' signal."*

Live example, `p1074_35556036525772`:

```
summary.overview          critic_verdict=PASS  composite=1.0  s_quote=1.0  s_critic=1.0
                          -> gate_reason: composite_below_tau
summary.affected_community  identical
```

A perfect field, gated, with a diagnostic telling you to tune a threshold when the real message is that the score has no discriminative power. `confidence.py` already knows — its own note reads *"functionally equivalent to gate_all_to_human for this bucket."*

**Fix:** add `bucket_saturated` to `GATE_REASONS` above `composite_below_tau` in `GATE_REASON_PRIORITY`, and set it from `threshold.saturated`.

### 2.2 `atomic_verify.py` (2,354 lines) never runs — and `s_shard` has never been computed anywhere

`atomic_verify` is called only from `mcal/build.py`. `segment_b/run.py` doesn't touch it, which is why `gate.signals_for_field` hard-codes `shard_rate=None`.

Worse than that: **`build.py`'s own `compute_signals` call passes no `shard_rate` either.** It supplies `quote_verdict`, `critic_verdict`, `citation_rate`, `acronym_rate`. So `atomic_verify` feeds only `atomic_citation_rates` (into `s_citation`, weight 0) and the false-negative audit. **The shard signal does not exist in calibration or in production.**

Consequence: claim-level verification never happens on shipped output. The reduce step verifies *quotes*, not *claims*; nothing binds claim *i* to quote *j*. A summary sentence can be wholly unsupported so long as other quotes in the field verify, because `s_quote_for` averages and `aggregate_verdict` returns `"mixed"` for a partial set.

This is exactly the Lincoln Hwy failure in the Evaluation sheet — `hallucination: "or important wildlife habitats are affected"` — and the mechanism that would catch it is built, tested, and disconnected.

**This is the same fix as the AUROC problem.** Wiring `s_shard` is the most plausible route to a composite that separates.

### 2.3 Overview and themes launder failed verification  — `segment_a/m2.py`

In `_summary_reduce`, if none of the overview's own quotes verify, up to three *verified quotes from the five subfields* are substituted into `final["overview"]["evidence"]`. `extract_themes` does the same via `carried_evidence`.

The grading sheet then renders `quote_verified: yes` with real page numbers for a field whose own claims were never checked. All 8 docs graded `summary.overview` "ok" — that grade may be measuring nothing.

**Fix:** keep the union if the reviewer needs pages to look at, but tag it (`evidence_source: "carried_from_subfields"`) and compute `s_quote` from the overview's own quotes only.

### 2.4 The blind grading pass is defeated arithmetically  — `mcal/grading_sheet.py`

Emitted blind sheet header:

```
field,extracted_value,quote,source_pages,quote_verified,model_confidence,your_grade,...
```

Emitted values:

```
summary.overview             model_confidence='composite=1.000'
summary.project_description  model_confidence='composite=0.750'
```

The sheet preamble states *"critic_verdict is deliberately ABSENT ... the human grade and the Critic verdict must be independent because their agreement is what fits the thresholds."*

But `composite = 0.5·s_quote + 0.5·s_critic`, and `quote_verified` is the adjacent column. So `s_critic = 2·composite − s_quote`, and the four verdict scores (PASS 1.0 / PASS_WITH_NOTE 0.7 / RE_EXTRACT 0.3 / HUMAN_REVIEW 0.0) are distinct. **The Critic verdict is exactly recoverable from two columns on the blind sheet.**

`rows_from_manifest` carefully prefixes the value with `composite=` "so it is never mistaken for the extractor's self-reported confidence" — the author considered the wrong confound.

These grades are the labels the AUROC is measured against. **Fix before grading another batch:** move `model_confidence` into `CRITIC_COLUMNS` (reveal-only), or round it to a coarse band that isn't invertible.

### 2.5 `RE_EXTRACT` is dead code in production

`segment_b/run.py` calls `gate_mod.run_gate(doc_id, m1, m2, critic_results, stage=stage, doc=doc, out_dir=base.parent)` — no `reextract` callback. The entire MCAL_PLAN 7 Q8 retry path in `gate.py` never fires.

In the sampled manifest, 3 of 15 fields returned `RE_EXTRACT`; all three became `HUMAN_REVIEW` via `gate_field`'s `critic_verdict` reason. You are hand-reviewing mechanically fixable items.

### 2.6 EIS-type regex is order-dependent, not position-dependent  — `segment_a/m1.py`

```python
for label, pat in EIS_TYPE_PATTERNS.items():
    if re.search(pat, head, re.IGNORECASE):
        regex_hits.append(label)
        break   # first hit wins by config ordering
```

`break` on first match means **dict order** decides, not where the phrase appears in the text. `config.EIS_TYPE_PATTERNS` order is `Supplemental, ROD, Draft, Final`. A Final EIS whose cover page mentions "record of decision" anywhere is labeled ROD.

Consistent with the Lincoln Hwy grade (`wrong: "ROD", correct: "Final"`), though that document isn't in `Documents/output/` so the attribution can't be confirmed directly.

Compounding: on regex/Sonnet disagreement the code returns the **regex** value with `confidence: low`. In the one observed failure the regex is the losing source.

**Fix:** score by match position in the text; on disagreement route to review rather than preferring one source by default.

### 2.7 Mis-citation is invisible to the score

`quote_check.check_quote` searches `cited ± 2` and records `matched_page`. But `gate.extraction_quote_verdict` returns `quote_check.aggregate_verdict(checks)` — a bare `yes|mixed|no`. A quote cited to p.36 and found on p.34 scores identically to an exact hit.

The Evaluation sheet's citation complaints are page-specific (`missing citation - pg 34, 36, 146`). Mis-citation is live and unmeasured.

**Fix:** emit `cited_page == matched_page` as a distinct signal.

### 2.8 No omission check anywhere

Correction to an earlier draft of this review: `_base.md` **Q2** *does* ask "does every claim correspond to a substring in the cited pages" → `RE_EXTRACT` + `T03_outside_text_fabrication`. That is a real claim-level precision check.

The narrower true gaps:

1. **Q1** only requires "at least one page" **for the whole value**, so a correct claim with no citation of its own passes. This is precisely the Evaluation sheet's 9× `"ok, missing citation - pg 35"`.
2. There is no question anywhere about content the document contains but the extraction **omitted**. Every rubric question is a precision check.

**Fix:** add a claims-without-own-citation ratio and wire it to `s_citation` (already computed, weight 0).

### 2.9 CEQ chapter detection is weak enough to break downstream extractors

Across the 12 processed docs:

```
Alternatives 9,  Environmental Consequences 4,  Consultation 2,
Affected Environment 1,  Mitigation 1,  Purpose and Need 0
```

`Purpose and Need` was never detected once. `extract_alternatives` has no fallback by design ("no word-regex fallback per v2 plan"), so **3 of 12 docs produce `alternatives: []`** — and `[]` is explicitly *not* in `EMPTY_IS_VALID_FIELDS`, so it registers as a failure rather than a result.

`chunk.detect_chapters` (uppercase-dominance + marker prefix, 3000-char TOC guard) is under-performing on 1970s OCR, and it is upstream of `alternatives`, `location`, and `key_people`.

### 2.10 A summary map call failed silently

```
p1074_35556036854040: n_chunks=17, n_chunks_summarized=11, chunks_truncated=True
```

`max_chunks=12`, so 11 partials means one map call raised. `_summary_pipeline` catches it, emits `log.error`, and reduces on partial input. It does **not** reach `DocResult.warnings` (that doc's only warning is the geocoder), doesn't reach the manifest, and `n_chunks_summarized` conflates "capped at 12" with "one call died."

A document summarized from 11/17 chunks is indistinguishable in the artifacts from one summarized from 12/17 by design.

### 2.11 The 12-chunk cap is an unread recall ceiling

`max_chunks=12` × 50pp = 600 pages summarized, maximum. `chunking_meta.chunks_truncated` is recorded and **nothing reads it** — not the Critic, not the gate, not the manifest. A claim absent because it was on page 700 is indistinguishable from a claim the document doesn't make.

Bit on 1 of 12 docs in this batch. Will bite more at scale.

**Fix:** propagate `chunks_truncated` into the manifest and force those docs' summary buckets to review.

### 2.12 Reduced geocoder mode is the permanent state

All 11 docs ran reduced; the location bucket is 100% force-gated. That's 1/15 fields of every document blocked for an installation reason, and location has produced zero automated calibration data.

Two downloads (PAD-US, GNIS) and a Mapbox token unblock a whole bucket. Cheapest accuracy win in the repo — and note from §0 that `location` is at 17% at-ceiling, i.e. borderline for convergence.

### 2.13 Caveat, not a finding: quote-check gate independence

`quote_check.normalize` folds `o→0, l/i→1, s→5, b→8, g→9, z→2`, plus `rn→m` and full punctuation stripping. `content_tokens` then runs on that **same folded string**, so the ratio gate and the coverage gate are not independent in the way the docstring implies.

Against that: the measured separation is strong (`settings.py` comment: coverage ≥0.70 gave 0% FP / 0% FN on 444 quotes), and the known Lincoln Hwy fabrication was spot-checked **against its own cited pages** (ratio 62.8, coverage 0.00). The systematic negative control is cross-document; the harder within-document case was checked only on that one example.

**Suggested:** re-run the negative control against *adjacent pages of the same document* to close the gap. Low priority.

### 2.14 Note on code-to-evidence ratio

~38,000 lines calibrated on 8 documents and 111 graded items. `atomic_verify.py` (2,354), `location_pipeline.py` (2,408), `active_select.py` (1,026) are elaborate relative to n=8, and several are justified by MCAL_PLAN citations rather than by measurement.

The docstrings are unusually well-reasoned. But a plan reference is not evidence. `weight_validation.v1.json` is the one artifact that honestly reports "this doesn't work yet," and it is the one that should be driving the roadmap.

---

## 3. Where efficiency and correctness conflict

1. **Opus map → Sonnet (§1.1) invalidates fitted thresholds.** Resolution: do it now. The thresholds it would invalidate are saturated and passing nothing.
2. **Wiring `atomic_verify` (§2.2) adds calls.** Resolution: it is the only credible path to an AUROC above chance, and §1.1 + §1.2 more than pay for it. Spend the savings here.
3. **Full-document prompt caching (§1.3) undoes the cited-pages restriction (§2.7).** Resolution: don't. Take the smaller caching win.
4. **Efficiency wants fewer Critic calls; correctness wants more checks.** Less opposed than it looks — §2.7, §2.8 and §2.11 are all *deterministic* signals derived from data already computed. Add them at zero LLM cost before adding any judge calls.

---

## 4. Recommended sequence

| # | Action | Type | Ref |
|---|---|---|---|
| 1 | Move summary map step Opus → Sonnet | cost | §1.1 |
| 2 | Switch to Batch API | cost | §1.2 |
| 3 | Fix the blind-sheet composite leak **before grading another batch** | correctness | §2.4 |
| 4 | Wire `s_shard` / `atomic_verify` into `segment_b/run.py` **and** `build.py` | correctness | §2.2 |
| 5 | Add free deterministic signals: cited-page match, per-claim citation, truncation flag | correctness | §2.7, §2.8, §2.11 |
| 6 | Add `bucket_saturated` gate reason | diagnostics | §2.1 |
| 7 | Install PAD-US / GNIS / Mapbox token | correctness | §2.12 |
| 8 | Improve `chunk.detect_chapters` | correctness | §2.9 |
| 9 | Surface silent map failures in warnings + manifest | diagnostics | §2.10 |
| 10 | Wire the `reextract` callback | cost + queue volume | §2.5 |
| 11 | Fix EIS-type match ordering | correctness | §2.6 |
| 12 | Refit, then **re-check AUROC before grading another batch** | — | §0 |

**Grading priority:** `alternatives+themes` first — it is the one bucket blocked purely on volume and will converge.

**Stop condition:** if AUROC is still ≤0.6 after step 5, the problem is the grading rubric or the bucket definitions, not the pipeline.

---

## Appendix A — Verification log

Every claim above was re-checked against source. Status of the original review:

| Claim | Status |
|---|---|
| AUROC CI includes 0.5 | confirmed |
| 4 buckets saturated at tau=1.0 | confirmed |
| A wrong item scored composite 1.0 | confirmed |
| "More graded docs will not fix this" | **corrected** — holds for 3 buckets, not `alternatives+themes`; mechanism restated (§0) |
| No prompt caching anywhere | confirmed |
| "Caching = 40–55% savings" | **corrected** — ~8–12% (§1.3) |
| Batch API unused | confirmed |
| Opus map doing Sonnet work | confirmed; promoted to top lever |
| Chunk selection `(tagged+untagged)[:12]` | confirmed; binds 1/12 docs |
| `extract_eis_type` always calls Sonnet | confirmed |
| lead_agency fallback "unreachable" | **corrected** — fires on 4.4% of rows |
| Retry / checkpoint / token budget | confirmed |
| `gate_field` never reads `saturated` | confirmed |
| `atomic_verify` never runs in production | confirmed, **and worse** — absent in calibration too |
| Overview/themes launder verification | confirmed |
| "No rubric checks claim-level support" | **corrected** — Q2 does; real gap is per-claim citation (§2.8) |
| EIS-type dict-order bug | code confirmed; Lincoln Hwy attribution unverifiable |
| `matched_page` discarded by gate | confirmed |
| `RE_EXTRACT` dead in production | confirmed |
| 12-chunk cap unread | confirmed; severity lower than stated |
| Reduced geocoder / LOC counts | confirmed |
| `_model_confidence` leaks self-report to blind sheet | **wrong mechanism** — that path is Segment A only; real leak is composite invertibility (§2.4) |

**Surfaced during verification, not in the original review:** §2.2 (`s_shard` absent at calibration), §2.9 (chapter detection breaks `alternatives` on 25% of docs), §2.10 (silent map-call failure).

---

## Appendix B — Reproducing the key numbers

```bash
# AUROC and candidate weightings
python3 -c "import json;print(json.load(open('mcal/artifacts/v1/weight_validation.v1.json'))['candidates'])"

# Per-bucket R_doc distributions and at-ceiling rates
python3 -c "
import json
t=json.load(open('mcal/artifacts/v1/thresholds.v1.json'))
for b,v in t['buckets'].items():
    rd=v.get('r_docs') or {}
    if not rd: continue
    vals=sorted(rd.values()); c=sum(1 for x in vals if x>=1.0)
    print(f'{b:22s} n={len(vals)} R={vals} ceiling={c}/{len(vals)}')
"

# Gate reasons and per-field composites for one doc
python3 -c "
import json
m=json.load(open('segment_b/output/p1074_35556036525772/run_manifest.json'))
for f,e in sorted(m.items()):
    if f.startswith('_'): continue
    print(f'{f:32s} {e[\"critic_verdict\"]:15s} comp={e[\"composite\"]} tau={e[\"applied_tau\"]} {e[\"gate_reason\"]}')
"

# Chunk truncation and silent map failures across the batch
python3 -c "
import json,glob
for f in sorted(glob.glob('segment_b/output/*/m2.json')):
    cm=json.load(open(f))['chunking_meta']
    print(f.split('/')[2][:26], cm['n_chunks'], cm.get('n_chunks_summarized'), cm.get('chunks_truncated'))
"

# Cost breakdown by model
python3 -c "
import json
for d in json.load(open('segment_b/output/batch_summary.json'))['docs']:
    for m in d['usage']['by_model']:
        print(d['doc_id'][:26], m['model'][-12:], m['calls'], m['input_tokens'], m['output_tokens'], m['cost_usd'])
"
```

---

## Appendix C — Implementation status

Added when the review was applied. The body above is left as written; this
section records what landed, what did not, and the four places where measuring a
claim changed the conclusion. Every number here is reproducible from the test
suite (1333 tests, no network).

### C.1 Applied

| Ref | Item | Landed as |
|---|---|---|
| §2.4 | Blind-sheet composite leak | `model_confidence` moved to `REVEAL_ONLY_COLUMNS`. `tests/test_grading_sheet.py` executes the inversion (`s_critic = 2·composite − s_quote`) so the reason stays demonstrated. Banding was rejected: with `s_quote` on the adjacent column, any informative band still pins `s_critic` to one or two of four levels. |
| §2.1 | `bucket_saturated` gate reason | Added above `composite_below_tau` in `GATE_REASON_PRIORITY`, set from `threshold.saturated`, reported per bucket in the rollup, and excluded from the null-tag numerator. Three causes now report distinctly because they have three remedies: grade more / refit / find a signal that separates. |
| §2.3 | Overview and themes launder verification | Borrowed quotes are now *appended* rather than replacing the field's own (the old code discarded the unverified entries exactly when they were most informative), tagged `evidence_source="carried_from_subfields"` with the source field, and excluded from `s_quote` and `s_citation` everywhere. `themes` gets `s_quote = None` — excluded, not floored at 1.0. Untagged evidence still counts as own, so nothing already calibrated on moves. |
| §2.5 | `RE_EXTRACT` dead in production | `reextract` callback wired, on by default, `--no-reextract` to disable. Needed a prerequisite: see C.3. |
| §2.6 | EIS-type match ordering | Position-scored. On seven realistic cover-page phrasings dict order gets 3 wrong and position gets 0; on the 21 materialized documents they disagree on one. Disagreement with Sonnet no longer prefers the regex — `review_required` plus a new `m1_source_disagreement` gate reason, with both candidates carried into the manifest. |
| §2.10 | Silent map failures | `m2` reports `n_chunks_failed` and `failed_chunk_indices` separately from the cap; `run.py` raises both as operator warnings. |
| §2.11 | 12-chunk cap unread | `chunking_meta` is read for the first time. `summary_source_incomplete` routes the summary-derived buckets to HUMAN_REVIEW, and `summary_coverage` lands in the rollup. |
| §2.13 | Within-document quote control | 602 decoy checks across the graded corpus, re-citing each verified quote to a page of its own document more than ±2 away. Result in C.3. |
| §1.1 | Opus map → Sonnet | `config.SUMMARY_MAP_MODEL`, **default unchanged (Opus)**. It shifts the distribution the frozen thresholds were fitted to, so it lands between calibration stages with a rerun and a refit. |
| §1.5 | Retry classification | `llm.is_retryable`; 4xx re-raises immediately instead of burning two more identical calls and three seconds of sleep per document. |
| §1.5 | Persist `partials` | `m2.save_partials`, written before the reduce. |
| §1.5 | Token-budget distribution | `max_tokens`, `output_headroom` recorded per call; `min_output_headroom` and `n_calls_thin_headroom` aggregated per model, plus a warning under 10% headroom. "A cap, not a target" is now measurable. |
| §2.9 | Chapter detection | Root cause was not the heuristic. See C.3. |

### C.2 Not applied

| Ref | Item | Why |
|---|---|---|
| §1.2 | Batch API | Deferred by choice — it is a new execution mode (async submit/poll) rather than a fix, and changes how every run is driven. Still the second-largest cost lever after §1.1. |
| §1.3 | Prompt caching | Not attempted. The review's own estimate is 8–12%, and the two genuine wins both depend on prompt structure that §1.1 would change anyway. |
| §2.12 | PAD-US / GNIS / Mapbox | Cannot be done from here: two downloads and a token. Still the cheapest accuracy win in the repo — `location` is 100% force-gated and has produced zero automated calibration data. |
| §2.2 | Wire `s_shard` | **Not implementable.** See C.3. |
| §4 step 12 | Refit and re-check AUROC | Requires a rerun; deliberately left to you. Note that several changes here only take effect on newly generated artifacts, so the refit must follow a rerun, not precede it. |

### C.3 Where measurement changed the conclusion

**§2.2 — `s_shard` cannot be wired, and the review's `build.py` claim was half right.**
MCAL_PLAN 3.3 defines `s_shard` via a per-atom `source_chunk_id`. Nothing in the
repo emits that field: one grep hit, the `gate.py` comment asserting
`atomic_verify` supplies it. It does not — it decomposes the *final* prose and has
no view of which map chunk a claim came from, and `m2` discarded the `partials`
that do carry `chunk_index`. Making it real needs the checkpoint (now added) plus
atom→chunk attribution that nobody has built. The comment is corrected in both
files and a test asserts the signal is absent everywhere.

What `build.py` actually had wrong was the *fallback*: when atomic verification
had not run, `s_citation` fell back to an evidence-level citation rate. That rate
is `s_quote` under another name — numerically identical on 40/40 graded summary
subfields under both label pairings, with matching AUROC (0.598 pre-amendment,
0.476 current). It is identical by construction: an evidence entry with pages
verifies, one without pages cannot be located and scores 0. Weighting it at n≥60
would have double-counted `s_quote` while the weight table claimed the weight
belonged to citation coverage. Removed.

**§2.7 — mis-citation is structurally impossible on the extraction path.**
0 divergences over 1985 located quotes, all fields, 20 documents.
`evidence.verify_and_locate` writes the page `doc.find_quote` *found* the quote
on, so the cite is derived from the match and cannot disagree with it. The
Evaluation sheet notes cited as evidence — `missing citation - pg 34, 36, 146` —
are a reviewer naming pages that *should* have been cited: T01 omission, which is
what atom-level `s_citation` is for, not a wrong page. Instrumented anyway, but as
a regression guard rather than a signal: the invariant holds only while the cite
comes from an exact substring match, and would break the day `find_quote` is
swapped for the fuzzy matcher. Deliberately *not* applied to
`CriticResult.source_pages`, which is a search window (M1 defaults to `[1,2,3]`)
and reports a spurious 5.4% divergence.

**§2.9 — the alias table, not the heading heuristic.**
The heuristic finds the headings; `_map_to_ceq` then discarded them, because
`CHAPTER_ALIASES` held only modern CEQ vocabulary and no document in this corpus
uses it. These are NEPA 102(2)(C) documents organized as a numbered five-point
response: `1. DESCRIPTION OF PROPOSED ACTION`, `2. DESCRIPTION OF THE EXISTING
ENVIRONMENT`, `PROBABLE ENVIRONMENTAL IMPACTS`, `9. STAFF DISCUSSION OF
SIGNIFICANT ENVIRONMENTAL MATTERS`, `EVALUATION AND DISPOSITION OF COMMENTS
RECEIVED...`. `Purpose and Need` was detected zero times because these documents
have no chapter by that name. Aliases were taken from the headings the corpus
actually contains, collected by dumping every line that passed the heuristic and
failed the mapper. Over 21 documents:

| chapter | before | after |
|---|---|---|
| Purpose and Need | 0 | 9 |
| Alternatives | 17 | 18 |
| Affected Environment | 1 | 7 |
| Environmental Consequences | 10 | 16 |
| Mitigation | 2 | 4 |
| Consultation | 2 | 10 |
| documents with no chapter at all | 1 | 0 |
| spans covering >80% of the document | 1 | 1 |

Alias precedence is now longest-first rather than dict order, which the table's
new length makes necessary (`proposed action` vs `proposed action and
alternatives`). Matching stays START-anchored: containment would map `ECONOMIC
IMPACTS` and `UNITED STATES ENVIRONMENTAL PROTECTION AGENCY` to chapters.

**§2.13 — the within-document control is harder than cross-document, and the
gate still holds.** 602 decoys, 1 acceptance, and that acceptance is a sentence
that genuinely appears on both the cited page 11 and the decoy page 36 at
coverage 1.000.

| | median | p95 | max |
|---|---|---|---|
| `partial_ratio` | 51.1 | **60.3** | 96.6 |
| content coverage | 0.214 | 0.571 | 1.000 |

The ratio p95 of 60.3 is *above* MCAL_PLAN 3.2's original 60 floor, and above the
58.9 measured cross-document — so a ratio-only gate false-accepts within a
document more readily than across documents, the direction that matters. Coverage
p95 of 0.571 sits below the 0.70 threshold and is what carries the decision.

### C.4 Two defects found while applying the review

**`s_acronym` was never populated.** `gate.acronym_rate` read the rate off the
field entry; `run.py:_run_acronyms` writes it to
`m2["acronym_meta"]["per_field"][field]["s_acronym"]`. So the signal was `None` on
every field of every document in the v1 batch while being computed and written on
each one — which is indistinguishable, in the artifacts, from a signal nobody
implemented. Both shapes are now accepted.

**`batch_id` was never passed.** The null-tag monitor records a batch per
document and defaulted every one to `"unbatched"`, though `process_batch` knew the
batch all along.

### C.5 Consequences for the roadmap

1. **The v1 blind grading sheets on disk still contain the leak.** They were
   written before §2.4 and carry `model_confidence` with the composite. Grading
   from them reproduces the problem the fix was for. Regenerating them is
   deterministic and LLM-free (`--no-reextract` reuses the m1/m2/critic
   checkpoints), but it rewrites the manifests, so it is left for you to trigger.
2. **§0's AUROC and the `R_doc` distributions are fitted on a stale pairing.**
   `tests/test_atomic_verify.py::TestT01Invisibility` establishes that the human
   grades describe the *pre-amendment* prose, which the build-#4/#5 rerun
   rewrote — every one of the 40 labelled subfields was rewritten. So the review's
   §0 numbers, and every threshold derived from them, pair current artifacts with
   labels that no longer describe them. Re-grading is the only thing that fixes
   this, and it is upstream of the "stop condition" in §4.
3. **Several fixes only bite on new artifacts.** Carried-evidence tagging,
   position-scored EIS type and the chunk-coverage flags are all written at
   extraction time. The refit in §4 step 12 must therefore follow a rerun.
