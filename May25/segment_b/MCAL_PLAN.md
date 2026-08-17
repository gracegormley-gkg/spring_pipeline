# M-Cal Plan (agreed)

Calibration step consuming Segment A human grades and emitting stage-versioned artifacts that Segment B loads at runtime. Produced via a two-agent adversarial design loop (Planner ↔ Critic, 3 rounds), then an independent review pass, then a consistency pass. Verdict: AGREED — stakeable.

Artifacts are **stage-versioned** (`.v1`, `.v2`, ...) per the multi-round calibration protocol in §7.5. Paths in this document are written `.v(N)` where the stage varies.

Salience is implemented in v1 as `summary_of_interest` (§3.15) — a second, separate summary emitted alongside the standard one. Remaining deferrals are in §8.

---

## §0. Context

Academic NLP pipeline extracting structured metadata from ~2,000 US Environmental Impact Statements (long OCR'd PDFs, 200–1500+ pp).

- **Segment A** — 20-doc calibration run. Human grades every field per doc in a CSV. 9/20 done; 8 in the Evaluation CSV summary.
- **Segment B** — production run over ~2,000 docs with M-Cal-calibrated Critic + confidence gate. M-Cal is the new step that consumes Segment A grades and produces the calibration artifacts.

**Multi-round calibration protocol.** M-Cal is not a one-shot step. The user's workflow is:
- Build **M-Cal seed v1** on the current 9 grades. Most confidence buckets will be `degenerate_severe` under seed v1 (N_wrong_docs < 3); those buckets set `gate_all_to_human=true`, meaning Segment B routes almost every field to HUMAN_REVIEW at seed v1. **This is expected and intentional** — Segment B seed v1 exists to produce extractions on a targeted next-batch of docs, not to run at production scale.
- Use `active_select.py` (elevated in build order) to pick the next batch of ~10 docs. Run Segment B seed v1 over them; user grades all outputs (including HUMAN_REVIEW routes, which is where most of the fresh calibration signal comes from).
- Rebuild as **M-Cal v2** on the augmented ~19-doc grade set. Fewer buckets are degenerate; τ_deployed thresholds meaningfully constrain gating.
- Continue at ~10-doc cadence until enough buckets have `N_wrong_docs ≥ 15` for the plan to run at scale.

Chunking currently operates on flat OCR text: 125k-char chunks with 5k overlap; page numbers are **estimated** via `char_offset / 2500`. This estimation is the reason for the ±2 page tolerance throughout.

---

## §1. Failure-mode analysis (from the Evaluation CSV)

### (1) `year` — 3/8 wrong, all pre-1980
- **Cause:** OCR noise on 1970s scans + M1 regex on first-3pp only. Old EISs often carry the actual date on a signature/approval page or transmittal letter, not the cover.
- **Layer:** M1 extractor.
- **Fix:** Year adjudicator **always runs**. Inputs: first 5pp + last 3pp of front matter + signature/approval page (detected via keywords `Approved | Signed | Date: | Transmittal`). Priority ordering in the adjudicator prompt: **signature > transmittal > cover > body**. OCR-normalize digits before regex.

### (2) `eis_type` — 1/8 wrong (Lincoln Hwy: Final flagged as ROD)
- **Cause:** First-page regex hit "Record of Decision" in body/citation text; Sonnet verifier on first 2pp also failed (old cover pages are lightly-textual).
- **Fix:** Combine filename slug + document-provided title + first 5pp + last 2pp of front matter + presence of "Draft/Final Environmental Impact Statement" as a **section heading** (not body text). Critic rubric adds: "Is 'Record of Decision' a heading, or a citation to a future document?"

### (3) `summary.project_description` — numeric hallucination ($659M Alt V vs $369M Alt XI)
- **Cause:** Map-reduce decouples `(alternative_label ↔ cost_figure)`. Each chunk emits its own numeric claim; reduce step may synthesize a mismatched pair.
- **Layer:** M2 reducer, then Critic misses it.
- **Fix:** Handled by `atomic_verify.py` at verify time. Opus decomposes the prose into atomic claims (`{subject: "Alt XI", predicate: "cost", object: "$369M", page: 87}`); each atom is substring-verified against its cited page. Mismatched pairs fail verification.

### (4) `summary.environmental_impact` — numeric hallucination (Mag 7.5 vs 7.0) + outside-text fabrication ("or important wildlife habitats are affected")
- **Cause:** Same decoupling as (3) for the magnitude. The wildlife clause is pure prior-injection — Opus completing a plausible NEPA sentence.
- **Fix:** Atomic decomposition + coordination-splitting. Any coordinating conjunction (`or`, `and`, `as well as`, `along with`) introducing a new noun-phrase subject **must yield a separate atom**. The wildlife clause becomes its own atom and fails substring verification. Regression tests: `tests/test_lincoln_hwy_wildlife_clause.py`, `tests/test_env_impact_magnitude_75_vs_70.py`.

### (5) `summary.public_response` — 4/8 missing citations (MOST COMMON)
- **Cause:** Comment-response tables are structurally different from body chapters; no per-subfield citation enforcement.
- **Fix:** Hard schema — `public_response` is a JSON list of `{claim, page, evidence_quote}`; empty list is valid; free-text is not. If comment-response chapter not identified, emit `{status: "no_comment_response_chapter_identified", based_on_main_doc_only: true}` with no prose.

### (6)–(7) `summary.affected_community` (2/8 missing cites) and `summary.alternatives_overview` (1/8 missing cite)
- **Fix:** Blanket schema enforcement across all `summary.*` subfields — no claim without a page cite.

### (8) `alternatives[0]` empty (Buffalo Light Rail)
- **Cause:** Structural identification of the Alternatives chapter failed on an atypical title.
- **Fix:** If structural identification fails, fallback = Sonnet classifies each TOC chapter heading with "does this chapter enumerate action alternatives?" Never return empty silently — return contents or `{status: "alternatives_chapter_not_found"}` for Critic → HUMAN_REVIEW.

### (9) `location` — 5/8 issues
Multiple distinct failure modes:
- (9a) No geocode (Randolph, LA Transit): geocoder returned nothing, no fallback.
- (9b) Wrong specificity (Milwaukee for Airport Spur): LLM picked the coarsest containing city.
- (9c) Multi-site, 1/3 geocoded (Buffalo, Lincoln Hwy).
- (9d) "No location" for Fuel Economy: national CAFE rulemaking treated as absent-location.
- **Fix:** Scope-conditional pipeline redesign — see §3.9.

### (10) `key_people` — 5/8 "all commenters = cooperators"
- **Cause:** The Consultation chapter (titles vary: "Consultation and Coordination", "List of Persons Consulted") often bundles cooperating agencies + consulted agencies + draft-EIS recipients + commenters. Deterministic extractor labels all "cooperator." NEPA §1501.8 defines cooperating agency narrowly; extractor has no such filter.
- **Fix:** Restrict cooperating-agency extraction to subsections whose heading fuzzy-matches `{"cooperating agencies", "joint lead agencies", "assisting agencies"}` (OCR-normalized). Else empty + HUMAN_REVIEW. See §3.10.

### (11) Undefined acronyms — 8/8
- **Cause:** No document-level acronym glossary; no post-processor.
- **Fix:** Deterministic pre-pass + post-pass in `postproc/acronyms.py`. Prompt-only enforcement is known-insufficient (failed 8/8).

### Fields with no observed failures: `title`, `themes`, `lead_agency`, `summary.overview`
All four were graded `ok` in 8/8 docs. **No dedicated fix is specified for them.** They still pass through the generic machinery — evidence-first Critic, quote-verify override, composite confidence, and the CP gate for their respective buckets (`M1` for title/lead_agency, `alternatives+themes` for themes, `summary_narrative` for overview). They are called out here so an implementer doesn't assume they were overlooked. Note that "no observed failures in 8 docs" is weak evidence — the Wilson interval on 8/8 still admits a true error rate around 30%, so these fields are candidates for closer attention as the graded corpus grows.

---

## §2. Artifacts

All under `May25/mcal/artifacts/`, suffixed with the current stage (`.v1`, `.v2`, ...). Frozen within a stage; Segment B pins whichever stage is current. Paths below are written `.v(N)` to indicate stage-versioning.

| Path | Schema | Consumed by |
|---|---|---|
| `taxonomy.v(N).json` | `{tags: [{id, name, description, exemplars: [{doc_id, field, note}]}], version, frozen_at}` | `critic_prompt.py`; reviewer UI |
| `thresholds.v(N).json` | per-bucket: `{alpha, alpha_effective, N_wrong_docs, tau_raw, curation_slack, tau_deployed, saturated, guarantee_conditioning, degenerate, degenerate_severe, gate_all_to_human}` — 7 buckets | `segment_b/gate.py` |
| `confidence_config.v(N).json` | `{signals: [name, weight], per_field_overrides: {}, dependent_fields: {year: [key_people]}, geocoder_stack: "full"\|"reduced", judge_model_by_field: {}}` | `segment_b/gate.py`, `critic.py` |
| `critic_prompts/{field}.v(N).md` | Header + anti-hallucination clause + rubric (ordered yes/no) + few-shot exemplars + JSON output schema | `segment_b/critic.py` at load time |
| `acronym_commons.v(N).json` | `{acronyms: [{token, expansion, sources: []}]}` — ~40-entry NEPA seed | `segment_b/postproc/acronyms.py` |
| `atomic_schema.v(N).json` | JSON schema for atomic claim tuples | `atomic_verify.py`, Critic |
| `weight_validation.v(N).json` | Diagnostic AUROC comparison of candidate weightings on graded set | Reporting only |
| `gate_simulation.v(N).json` | Per-bucket {gate rate, caught-error rate, false-defer rate} simulated against the current graded set | Reporting |
| `atomic_verify_failure_log.v(N).json` | Per-atom rejection log: `{atom_id, atom_text, evidence_quote, page, failure_reason, subfield_human_grade}` | Pre-freeze audit; false-negative diagnostic |
| `null_tag_monitor.json` (Segment B, rolling) | Per-bucket rolling rate of HUMAN_REVIEW routes with `failure_tag = null`; batch-level | Signal for next M-Cal recalibration (§6) |
| `next_batch.csv` | `doc_id, uncertainty_score, dominant_predicted_failure_tags` — **~10 rows** (one grading batch per §7.5 cadence) | Reviewer for next grading round |
| `calibration_report.v(N).md` | Human-readable roll-up. **Must include a `Cost Summary` section**: total input/output tokens per model tier (Sonnet, Opus) on the current-stage calibration rerun, per-doc average, and projected Segment B cost at 2000 docs (with clear labels for the Opus atomic-verify slice, Sonnet Critic slice, and M2 rerun slice). Used for the go/no-go call before full-scale Segment B. | User |
| `tests/regressions/*.json` | Lincoln Hwy wildlife-clause + env_impact magnitude fixtures | CI |

**Taxonomy versioning rule:** v(N+1) may only *add* codes (`T19+`, since the seed taxonomy occupies T01–T18) or mark old codes `deprecated` with `superseded_by`. Never renames or renumbers T01–T18.

**Seed failure taxonomy (to be confirmed by TnT-LLM induction on ratified grades):**
`T01_missing_citation`, `T02_numeric_hallucination`, `T03_outside_text_fabrication`, `T04_undefined_acronym`, `T05_commenter_mislabeled_as_cooperator`, `T06_geocode_missing`, `T07_geocode_wrong_specificity`, `T08_scope_misclassified_national`, `T09_multi_site_partial_geocode`, `T10_alternatives_chapter_missed`, `T11_year_ocr_error`, `T12_eis_type_confused_with_rod`, `T13_pre_1978_nepa_format`, `T14_regional_scope_underspecified`, `T15_jargon_without_gloss`, `T16_abstract_when_concrete_available`, `T17_manufactured_salience`, `T18_salience_duplicates_summary`.

Note: T17 and T18 apply only to `summary_of_interest` (§3.15) and have no exemplars in the current graded set, since the field is new. Their induction entries will be empty at seed v1; the Critic still checks for them via rubric Q7.

---

## §3. Implementation (module-by-module)

Directory layout:
```
May25/mcal/
  __init__.py
  taxonomy.py
  confidence.py
  critic_prompt.py
  atomic_verify.py
  quote_check.py
  active_select.py
  build.py                    # orchestrator: python -m mcal.build
  artifacts/                  # emitted files
  templates/
    critic_header.md          # anti-hallucination clause etc.
    rubrics/{field}.md        # per-field rubric fragments
May25/segment_b/
  critic.py                   # modified
  gate.py                     # new
  postproc/
    acronyms.py
    location_pipeline.py
    key_people_pipeline.py
  year_adjudicator.py         # new (M1)
```

### 3.1 `mcal/taxonomy.py`
- **Input:** all graded rows in `May25/segment_a/output/grading_sheets/*.csv` (plus any grading sheets produced from prior Segment B calibration batches).
- **Output:** `artifacts/taxonomy.v(N).json`.
- **Logic:** load rows where `your_grade != "ok"` OR `your_notes` non-empty; TnT-LLM-style induction prompt to Sonnet: "cluster these notes into 6–12 named failure modes"; emit draft `.v(N)-draft.json`; human ratification → promote to `.v(N).json` + `frozen_at`. On recalibration stages, the prior taxonomy is loaded first and extended add-only (§2 versioning rule).

### 3.2 `mcal/quote_check.py`
Upgrade of the existing verifier.
- **Input:** `quote` string, `source_pages: [int]`, full OCR text with page-offset index.
- **Output:** `{verified: yes|mixed|no, normalized_match_span: (page, char_start, char_end) | None}`.
- **Logic:** OCR-normalize both sides (`rn↔m`, `l↔1↔I`, `O↔0`, `S↔5`, whitespace collapse, lowercase, punctuation strip); rapidfuzz `partial_ratio` over each page in `source_pages ± 2`; ≥90 → yes; 60–90 → mixed; <60 → no.

### 3.3 `mcal/confidence.py`
- **Composite:** `composite = 0.5·s_quote + 0.5·s_critic` for all buckets. Other signals (`s_source`, `s_citation`, `s_shard`, `s_acronym`) are **computed and logged** with weight 0. Per the §7.5 add-only guarantees, weights stay frozen at 0.5/0.5 through at least stage v3, and weight-validation remains advisory until the graded set reaches roughly n ≥ 60 (where per-field AUROC confidence intervals become interpretable).
- **Signal definitions** (all in [0,1]):
  - `s_quote` = quote_check verdict → {yes: 1.0, mixed: 0.5, no: 0.0}. **M1 fields (`year`, `eis_type`, `lead_agency`, `title`) have no verbatim quote in their extracted values by design; for these fields `s_quote` defaults to 1.0, so the M1 composite collapses to `0.5·s_critic + 0.5` — effectively a 0.5 floor plus half the Critic verdict.** This is intentional: M1 correctness is checked via `s_source` in future weight iterations, not via quotes.
  - `s_critic` = critic_verdict → {PASS: 1.0, PASS_WITH_NOTE: 0.7, RE_EXTRACT: 0.3, HUMAN_REVIEW: 0.0}
  - `s_shard` = frac of chunks whose atomic claim survived into reduce output (only defined for `summary.*` and `alternatives_overview` given per-atom `source_chunk_id`)
  - `s_source` = NUL/regex/Sonnet agreement (M1 fields): {all: 1.0, 2/3: 0.5, disagree: 0.0}
  - `s_citation` = frac of atomic claims with page cite
  - `s_acronym` = defined-first-use rate

- **Weight-validation (diagnostic only):** Kendall-τ / AUROC of candidate weightings vs correctness on graded set, doc-stratified, paired bootstrap 1000 resamples. **Not a decision gate at current calibration scale** — at n well under 60 the confidence intervals are too wide for any candidate to be meaningfully dominated. Written to `weight_validation.v(N).json` for reporting. Tiebreaker in `confidence.py` docstring: fewest signals with non-zero weight; ties broken lexicographically.

- **Split-conformal recipe (single formula):**
  - Buckets: **M1, summary_narrative, summary_numeric, summary_of_interest, alternatives+themes, location, key_people** (7 buckets).
    - `summary_narrative` = `project_description, affected_community, alternatives_overview, public_response`
    - `summary_numeric` = `environmental_impact`
    - `summary_of_interest` = the new salience-weighted summary (§3.15). **Its own bucket, not pooled with `summary_narrative`** — it has a different error profile (manufactured salience, duplication) and, critically, **zero graded examples at seed v1**. It will therefore be `degenerate_severe` → `gate_all_to_human=true` until it accumulates graded data, which is the correct behavior for a brand-new field.
  - Per-bucket calibration set = **only docs with ≥1 wrong item in B**. Size = `N_wrong_docs(B)`.
  - Per-doc nonconformity in bucket B: `R_doc = max{s_i : i ∈ doc, i ∈ B, y_i = 0}` (well-defined by construction).
  - `τ_raw = ⌈(N_wrong_docs+1)(1−α)⌉ / N_wrong_docs` empirical quantile of `{R_doc}`.
  - **Degeneracy gates:**
    - `N_wrong_docs < 6` at α=0.15 → `degenerate=true`, `α_effective=0.25`.
    - `N_wrong_docs < 3` at α_effective=0.25 → `degenerate_severe=true`, `gate_all_to_human=true`, τ effectively `+∞`.
  - **Curation slack (leave-one-doc-out, restricted).** LOO is computed **only over the wrong-item-containing docs** in each bucket (the same subset used for τ_raw). For each of those `N_wrong_docs` folds, refit τ on the remaining `N_wrong_docs − 1` and measure `Δτ_i = |τ_full − τ_leave_i|`. Because `N_wrong_docs` is typically small at early stages (often 2–8 per bucket at seed v1, growing with each round), use **`curation_slack = max(Δτ_i)`** rather than the 95th percentile — the percentile is unreliable at these sample sizes and dominated by the discreteness of the empirical quantile. Report the full `{Δτ_i}` distribution in `calibration_report.v(N).md` for auditability.
  - **τ_deployed = min(1.0, τ_full + curation_slack)**; if clamped, record `saturated=true` in `thresholds.v(N).json`.
  - Accept in Segment B iff `composite(x_new) > τ_deployed`.
  - **Guarantee (stored in `thresholds.v(N).json.guarantee_conditioning`, per-doc form matching what doc-clustered CP actually delivers):**
    > `P(∃ i in doc : s_i > τ_B | y_i = 0, doc has ≥1 wrong item in bucket B, doc exchangeable with Segment A) ≤ α.`
    > `Distribution shift to full-corpus Segment B untested at this stage.`

### 3.4 `mcal/atomic_verify.py` (Opus post-hoc decomposition)
- **Scope:** the 5 `summary.*` subfields (`project_description`, `affected_community`, `alternatives_overview`, `environmental_impact`, `public_response`) **plus `summary_of_interest`** (§3.15). Note: `alternatives_overview` here refers to the summary subfield; the standalone `alternatives` list has its own structured verifier and is not decomposed.
- **`summary_of_interest` handling:** each list entry is already claim-shaped (`{claim, salience_criterion, page, evidence_quote, why_notable}`), so decomposition is a no-op for the `claim` field — it goes straight to per-atom verification. The `why_notable` sentence is decomposed and verified separately, because that's where unsupported editorializing would appear. An entry whose `claim` verifies but whose `why_notable` does not is tagged `T17_manufactured_salience`.
- **Input:** subfield prose value + its citations + OCR text of cited pages ±2.
- **Output:** list of atomic claims `{id, text, subject, predicate, object, page, evidence_quote, claim_type, polarity, coreference_resolved}`.
- **`claim_type` enum:** `{prose, numeric, comparative, categorical, temporal, geospatial}`.
- **`polarity` enum:** `{affirmative, negative}` — required field, defaults to `affirmative`. Any claim with negation cue words (`not`, `no`, `neither`, `never`, `without`, `except`, `unless`, `fails to`, `does not`, `would not`) must be tagged `negative` and its `evidence_quote` must contain the negation cue verbatim.
- **`coreference_resolved`:** boolean; must be `true` before verification runs. Atom `text` cannot contain unresolved pronouns (`it`, `they`, `this`, `that`, `these`, `those`) or generic anaphora (`the agency`, `the project`, `the alternative`) — the decomposer must substitute the antecedent from the source passage.
- **Logic:**
  1. **Opus** decomposition prompt (mandatory rules, verbatim in `templates/atomic_decomposition.md`):
     - "Split this passage into minimal factual claims. One subject-predicate-object per claim."
     - "**Coreference resolution:** replace all pronouns and generic anaphora (`it`, `they`, `this`, `the agency`, `the project`) with their explicit antecedents from the source passage. If the antecedent is ambiguous, emit `coreference_resolved=false` and let verification handle it as a failure."
     - "**Negation preservation:** if the claim contains a negation cue, tag `polarity=negative` AND ensure the `evidence_quote` field contains the negation cue verbatim. Do NOT emit an affirmative-polarity atom for a negated claim."
     - "**Coordination splitting:** any sentence containing a coordinating conjunction (`or`, `and`, `as well as`, `along with`) introducing a new noun-phrase subject MUST yield separate atoms."
     - "Numeric values are separate claims. Preserve which alternative/entity each number attaches to."
  2. **Per-atom verification:** OCR-normalized substring match of `evidence_quote` against `page ± 2` via `quote_check.py`. Type-specific checks: `numeric` → value+unit match; `temporal` → date within ±1yr tolerance; `negative` polarity → negation cue token must be present in the substring match.
  3. **Aggregation:** `subfield_score = mean(atomic_scores)` with **2× penalty** on `numeric` and `geospatial` claim failures.
- **False-negative audit.** On the currently-graded calibration set (post-M2-rerun), for every atom that fails verification, `atomic_verify.py` writes `{atom_id, atom_text, evidence_quote, page, failure_reason, human_grade_of_subfield}` to `artifacts/atomic_verify_failure_log.v(N).json`. **At seed v1 (n≈9): advisory only** — the sample of atoms drawn from correctly-graded subfields is too small (≈15–25 atoms) for a meaningful false-negative rate; log is reviewed for qualitative patterns (missing coreference cases, missing negation cases) that inform prompt refinement. **From v2 onward (n≥19): gating** — if the false-negative rate on correctly-graded subfields exceeds 10%, tune the decomposition prompt before freezing.
- **Why Opus-on-Opus:** JudgeBench's Sonnet-ceiling warning applies specifically to weak-judge-strong-writer pairings. Opus decomposing Opus has no ceiling gap. Preserves Segment A extractions once they've been re-run under the amended M2 prompts (build item #4).
- **Prompt is hand-written; not tuned on the calibration set** (tuning on the same docs used to fit τ would leak).

### 3.5 `mcal/critic_prompt.py`
Builds per-field prompt files.
- **Structure per prompt:**
  1. **Role header** (shared): "You are a strict quote-anchored verifier. You do not reason about plausibility. You verify text against evidence."
  2. **Anti-hallucination clause** (shared, from `templates/critic_header.md`):
     > "Any claim in the extracted value that cannot be located as a substring (with OCR-normalization tolerance) in the cited pages is unsupported. Do NOT use world knowledge, do NOT infer, do NOT assume that plausible-sounding NEPA boilerplate is present in this document. Before emitting your verdict, you MUST emit an `evidence_quote` field containing a ≥20-character substring copied verbatim from the cited pages that supports the extracted value. If no such substring exists, `evidence_quote = null` and `verdict = RE_EXTRACT`."
  3. **EVIDENCE section** (mandatory input, per prompt): plain-text OCR blob of `[min(cited_pages)−2 .. max(cited_pages)+2]` with `[[PAGE n]]` markers.
  4. **Rubric — ordered binary yes/no checks** per field. Decision table maps answers to verdict.
     Example for `summary.public_response`:
     - Q1: Does the extracted value cite at least one page?
     - Q2: Do all claims correspond to a substring in the cited pages (OCR-normalized)?
     - Q3: Are all acronyms defined on first use within this value?
     - Q4: Are stances attributed to named commenters (or "private commenter")?
     - Q5: For any stance about a **private individual** (see definition below) — is it flagged for human review?
     - Q6: Is the value readable and concrete for a non-specialist? Specifically: (a) are NEPA-specific terms and regulatory citations glossed in-line on first mention, using support from the cited pages? (b) does the value describe the project, affected community, alternatives, impacts, or public response in concrete terms (named entities, specified quantities, plain nouns) rather than abstract nominalizations, where the document supports concreteness?

     Decision: any Q1/Q2/Q4 = no → RE_EXTRACT. Q3 = no → PASS_WITH_NOTE + tag T04. Q5 = no → HUMAN_REVIEW. Q6(a) = no → PASS_WITH_NOTE + tag T15. **Q6(b) at v1: logged only** — Sonnet-judged concreteness is subjective and not calibrated at current scale; Q6(b) verdicts are recorded in `run_manifest.json` under `rubric_answers.Q6b` for offline audit and revisited at a later stage (spot-check against the graded corpus before promoting to PASS_WITH_NOTE + tag T16).

     **Additional rubric for `summary_of_interest` only (Q7):**
     - Q7(a): For each entry, does the cited page actually support the assigned `salience_criterion`? (E.g. an entry tagged `contested` must have a cited page showing actual disagreement, not merely a topic that *could* be contested.) Also: is `why_notable` grounded in the cited page rather than in general knowledge about NEPA practice?
     - Q7(b): Does any entry merely restate content already present in the standard `summary.*` fields without independently meeting a salience criterion?
     - Q7(c): If the list is **empty** — is that plausibly correct for this document? (An empty list on a genuinely routine EIS is a PASS, not a failure. The Critic must not penalize emptiness.) Conversely, if the list is non-empty, is each entry genuinely atypical rather than standard EIS content?

     Decision for `summary_of_interest`: Q7(a) = no → RE_EXTRACT + tag `T17_manufactured_salience`. Q7(b) = no → PASS_WITH_NOTE + tag `T18_salience_duplicates_summary`. Q7(c) empty-and-plausible → PASS. Q7(c) non-empty-but-routine → RE_EXTRACT + tag `T17_manufactured_salience`.
  5. **Few-shot examples** — greedy set-cover across the top-K tags observed for that field, `K = min(3, #distinct tags with ≥1 exemplar)`. If fewer than 3 slots filled after cover, fill remainder with **positive controls** (correctly-graded examples). Never below 3 total slots.
  6. **Output schema (strict JSON, `evidence_quote` before `verdict`):**
     ```json
     {
       "evidence_quote": "string|null",
       "rubric_answers": {"Q1": "yes|no", ...},
       "verdict": "PASS|PASS_WITH_NOTE|RE_EXTRACT|HUMAN_REVIEW",
       "failure_tag": "T01|T02|...|null",
       "note": "string|null"
     }
     ```

**Operational definition of "private individual"** (lives in `templates/critic_header.md`; referenced from rubric Q5 and from `segment_b/critic.py`):

> A named person is a **private individual** iff the cited passage does not identify them with a government agency, elected office, tribal/nation role, incorporated organization, or a professional/expert role relevant to their stance. Titles such as "Dr.", "Prof.", "Chair", "Director of X", "Mayor", "Council Member", or "Secretary" indicate a **non-private** role.
>
> **Dual-capacity handling:** a stance's capacity is bound to what the cited passage states at the point of stance attribution. If the same person appears elsewhere in the document in a different capacity (e.g., a mayor commenting officially in one chapter and as a resident in another), treat only the current stance according to its own cited passage. If the passage is ambiguous about which capacity is being expressed, route to HUMAN_REVIEW regardless of Critic verdict.

### 3.6 `mcal/active_select.py`
Uncertainty sampling for the next grading batch (**~10 docs**, matching the §7.5 recalibration cadence). Rank candidate docs (remaining Segment A pool + any Segment B pilot docs already extracted) by predicted composite variance across fields; prefer docs whose predicted dominant failure tags are underrepresented in the current graded set. Emit `artifacts/next_batch.csv`.

Batch size is a `--n` flag defaulting to 10, so the cadence can be adjusted without editing code.

### 3.7 `mcal/build.py`
Orchestrator. Invocations:
- **Seed build:** `python -m mcal.build --stage v1 --grades May25/segment_a/output/grading_sheets/ --out May25/mcal/artifacts/`
- **Recalibration:** `python -m mcal.build --stage v2 --grades May25/segment_a/output/grading_sheets/ --prior v1 --out May25/mcal/artifacts/` (repeat with `--stage v3 --prior v2`, etc.)

**Staged behavior.** On a seed build, all artifacts are produced from scratch. On a recalibration build, the prior stage's `taxonomy.v(N-1).json` is loaded and carried forward add-only (new `T19+` codes may be added from new failure notes; T01–T18 are never renamed or dropped, per §2 versioning rule). Thresholds, prompts, and confidence config are **rebuilt** from the augmented grade set — not incrementally patched — so they always reflect the full accumulated calibration data.

**Step-0 M2 rerun requirement.** Because item #4 in the build order amends the M2 summary prompts, `build.py` must verify that all currently-graded Segment A calibration docs have M2 output produced *under the amended prompts* before any calibration work runs. On startup:
- Read a marker `segment_a/output/m2/_prompt_version.txt`. If missing or `!= "v1_plain_language"`, halt with a message directing the user to run `segment_a/run.py process --force` on the graded docs.
- If marker present, proceed to taxonomy induction (or add-only update), atomic verification, confidence + CP calibration, prompt build, and artifact emission in order.
- All emitted artifacts are written to `.v(N)-draft/` first; user ratification of `taxonomy.v(N)-draft.json` promotes the whole draft directory to `.v(N)/`.

**Startup precheck for geocoder assets** (per §3.9a): verify `PADUS_GEODATABASE_PATH`, `GNIS_TSV_PATH`, and `MAPBOX_TOKEN` are set and readable. If any is missing, fail loud with a checklist AND mark `confidence_config.v(N).json.geocoder_stack = "reduced"` (Census + Nominatim only); the location bucket in `thresholds.v(N).json` is then flagged `gate_all_to_human=true` and calibration continues. This lets users run the full M-Cal loop before geocoder setup is complete, at the cost of routing all location fields to HUMAN_REVIEW in Segment B until the full stack is available.

### 3.8 `segment_b/postproc/acronyms.py`
- **Pre-pass** (once per doc): build glossary.
  - Regex `\b([A-Z][A-Z0-9]{1,}[A-Z0-9])\b` co-located with parenthetical expansions: `"Full Name (FN)"` and reverse `"FN (Full Name)"`.
  - Scan front matter (pp. i–xxx) and last 30pp for section headings fuzzy-matching `{"abbreviations", "acronyms", "glossary", "list of acronyms"}`; parse table-formatted entries preferentially.
  - **Denylist:** Roman numerals (I–XX), section markers, 2-letter state postal codes when co-located with place names.
- **Post-pass** (per output field): first occurrence of each acronym rewritten to `"Full Name (FN)"`. Subsequent occurrences left as-is.
- **Fallback:** `acronym_commons.v(N).json` (seed ~40 entries: EIS, NEPA, CEQ, EPA, USACE, NOAA, USFWS, USFS, BLM, DOT, FHWA, FAA, ROD, FONSI, DEIS, FEIS, SEIS, EA, LEDPA, NHPA, ESA, CWA, CAA, NAAQS, PM2.5, VOC, NOx, SO2, MSAT, GHG, CO2, VMT, HOV, LOS, ADT, ROW, DBE, MBE, SHPO, THPO). Doc glossary takes priority.
- **Unknown acronym (not in doc glossary and not in commons):** tag `T04_undefined_acronym`, field → PASS_WITH_NOTE. **Do not rewrite** — never fabricate an expansion.

### 3.9 `segment_b/postproc/location_pipeline.py`
Replaces the current single-shot location extractor.
1. **Scope classifier** (Sonnet, one call, first 30pp + tables of contents): output ∈ `{site, corridor, regional, national, international}` + one-sentence justification.
2. **National/international** → emit `{scope, sites: [], geocoded: [], textual_location: "national" | "international"}`. Done.
3. **Site extraction** (Sonnet on first 30pp + Project/Study Area chapter): list of `{name, admin_hierarchy: [poi, neighborhood, city, county, state, country], role: primary|alternative|reference}`. Reject `role != primary` from geocoding.
4. **Geocoding — scope-conditional cascade over a US-optimized vendor stack** (see §3.9a for stack details):
   - **`site` scope:** query at successively finer levels `poi → neighborhood → city → county → state`. **Accept the finest level whose returned bbox is contained in the returned bbox of the next-coarser level.** POI wins if containing-city matches `admin_hierarchy.city`. State-only pass → tag `T07_geocode_wrong_specificity`.
   - **`corridor` scope:** geocode as linear feature — extract endpoints from doc ("from X to Y") + implied midpoint. For each of {endpoint_A, midpoint, endpoint_B}, bbox-**intersection** with the coarser admin container AND centroid inside coarser bbox. Return list of 3 points + `corridor: true` flag.
   - **`regional` scope:** query at coarsest admin level whose name matches the doc's stated region ("Southern California", "Puget Sound region"). Fallback to bounding-polygon centroid of enumerated primary sites. If <2 primary sites → tag `T14_regional_scope_underspecified`, HUMAN_REVIEW.
5. **Retain textual location on geocode failure** — a named place without coordinates is still valid output.

### 3.9a Geocoder vendor stack

Every EIS is a US federal document. General-purpose global geocoders (Nominatim) are the wrong default. The stack below is optimized for the four hardest failure modes observed in the Evaluation CSV: federal-lands references, highway corridors, named natural features, and admin-level disambiguation.

**Cascade order** (each hop only fires if the previous produced no confident result). **Coverage figures below are rough a-priori expectations, not measured values** — actual hit rates should be recorded per hop via the `source` field (see Implementation notes) and reported in `calibration_report.v(N).md` once real data exists:

1. **US Census Geocoder** — free, unlimited, US-only. Primary hop for anything the scope classifier tagged `site` with a city/county/state/tract/address name. REST API (`geocoding.geo.census.gov`). *Expected to resolve a plurality of locations.* No auth required.
2. **USGS GNIS** — free, bulk-downloadable (~2GB). Second hop for named natural and cultural features (rivers, forests, mountains, historic sites, populated places). Loaded into a local SQLite/GeoPandas index at build time.
3. **PAD-US spatial join** — free, bulk-downloadable (~1GB). Third hop for federal-lands references ("Cottonwood Field Office", "Ashley National Forest", "Modoc National Forest"). Every federal-agency parcel as a polygon; fuzzy-name spatial join over BLM/NPS/USFWS/USFS/DoD units. **Expected to be the biggest single quality upgrade for this corpus** — federal-agency EISs routinely reference their own managed units by name, and no general-purpose geocoder resolves those. Requires GeoPandas + local geodatabase.
4. **Mapbox Geocoding API** — 100k/mo free tier (comfortably covers expected call volume at ~10 calls/doc × 2000 docs). Fourth hop for POIs, named highways, and disambiguation with document context. Requires API key in `.env` as `MAPBOX_TOKEN`. Research-friendly ToS (attribution required in published outputs).
5. **Nominatim (public server)** — free, rate-limited (1 req/sec). Last-resort fallback for international mentions (rare in US federal EISs but not zero — e.g., border projects). Existing dependency via `geopy`; kept in the cascade rather than removed.

**Setup responsibility (user):**
- Download PAD-US current version from USGS (Protected Areas Database) and unzip to a path configured in `settings.py` under `PADUS_GEODATABASE_PATH`.
- Download GNIS domestic-names file from USGS (National Map) and place at `GNIS_TSV_PATH`.
- Register a free Mapbox account and put the token in `.env` as `MAPBOX_TOKEN`.

**Reduced-pipeline fallback.** `build.py` performs a startup precheck (see §3.7). If any of PAD-US, GNIS, or `MAPBOX_TOKEN` is missing, M-Cal proceeds in **reduced mode**: the cascade collapses to Census + Nominatim only, `confidence_config.v(N).json.geocoder_stack` is set to `"reduced"`, and the location bucket in `thresholds.v(N).json` is forced `gate_all_to_human=true`. Segment B runs in this mode route every location field to HUMAN_REVIEW. This is by design — it lets you exercise the full M-Cal pipeline end-to-end before completing geocoder setup, and prevents Segment B from silently producing degraded locations. To move to full mode, complete the downloads and re-run `python -m mcal.build`.

**Implementation notes:**
- Cascade lives in `segment_b/postproc/location_pipeline.py`; each hop is a separate function returning `Optional[{lat, lon, bbox, source, confidence, admin_hierarchy}]`.
- `source` field is preserved in output so downstream can filter/audit by geocoder provenance.
- Corridor endpoint parsing runs before geocoding (regex + Sonnet for hard cases: "from X to Y", "between X and Y", "the X–Y segment"). Each endpoint runs the full cascade independently.
- Rate-limit and retry wrappers per vendor: Census (none needed), GNIS/PAD-US (local, none needed), Mapbox (600 req/min per token — well under free tier), Nominatim (1 req/sec, added to existing wrapper).

**Cost:** $0 within free tiers at expected call volume. If Mapbox usage exceeds 100k/mo (unlikely at 2000 docs × ~10 calls each), overage is $0.75/1k.

### 3.10 `segment_b/postproc/key_people_pipeline.py`
Replaces the current 3-bucket extractor with stricter role-tagging.
1. **Agency preparers:** unchanged — deterministic from Preparers chapter.
2. **Era gate:** if `year.critic_verdict ∈ {RE_EXTRACT, HUMAN_REVIEW}` → key_people is HUMAN_REVIEW unconditionally (dependent-field cascade). If `year < 1978` AND `year.critic_verdict ∈ {PASS, PASS_WITH_NOTE}` → route `cooperating_agencies` to HUMAN_REVIEW with tag `T13_pre_1978_nepa_format`.
3. **Cooperating agencies (post-1978):** extract ONLY from chapters/subsections whose heading OCR-normalized-fuzzy-matches `{"cooperating agencies", "joint lead agencies", "assisting agencies"}`. If no such heading → run Sonnet fallback ("Is any entity described in this doc as a formally designated cooperating agency under NEPA §1501.8 or its predecessor CEQ guidance?"). If uncertain → empty list + HUMAN_REVIEW. **Do not** default to pulling the whole Consultation chapter.
4. **Consulted entities** (new bucket): entities from Consultation chapter that are not cooperating agencies. Sonnet role-tags each: `{consulted_agency, tribe, recipient_of_draft, other}`. NOT labeled "cooperator."
5. **Public commenters with stance:** ONLY from Comment/Response chapter (headings matching "comments received", "response to comments", "public hearing transcripts"). If absent → empty list. Stances for private individuals → mandatory HUMAN_REVIEW (policy, not calibrated).
6. **Critic role-check:** per-entity — "Is this entity described in the cited passage as (a) formally-designated cooperating agency, (b) public commenter, or (c) neither?" Mismatch → RE_EXTRACT with tag `T05_commenter_mislabeled_as_cooperator`.

### 3.11 `segment_b/critic.py` (modified)
- Loads `artifacts/critic_prompts/{field}.v(N).md`, `atomic_schema.v(N).json`, `confidence_config.v(N).json`.
- **Evidence-first schema:** requires `evidence_quote` before `verdict` in the JSON output (schema field order enforced).
- **Deterministic quote-verify override:** after receiving Critic output, run `quote_check.py` on `evidence_quote` against cited pages. If not verified → override `verdict = HUMAN_REVIEW`, add note `critic_evidence_unverifiable`.
- **Judge model routing:** Sonnet by default. All five `summary.*` subfields (which includes `alternatives_overview`) **plus `summary_of_interest`** route to Opus per JudgeBench — `summary_of_interest` especially, since Q7(a)/(c) require judging whether a salience claim is *genuinely* atypical, which is exactly the kind of reasoning task where Sonnet-tier judges are unreliable. Configurable via `confidence_config.v(N).json` field `judge_model_by_field`.
- **Private-individual stance** → HUMAN_REVIEW unconditionally (policy; definition in §3.5).

### 3.12 `segment_b/gate.py` (new)
- Loads `thresholds.v(N).json`, `confidence_config.v(N).json`.
- For each field: compute composite → look up bucket τ_deployed → if `composite ≤ τ_deployed`, emit `HUMAN_REVIEW` regardless of Critic verdict. If bucket has `gate_all_to_human=true`, HUMAN_REVIEW unconditionally.
- **Emits `run_manifest.json` per doc.** The manifest must carry enough information for a human to grade the field *without* opening any other file — this is load-bearing for the multi-round protocol (§7.5), where most fields at seed v1 and v2 are gated and the reviewer's grades are the next round's calibration data. Per-field schema:
  ```json
  {
    "<field_name>": {
      "extracted_value": "<the actual extraction — string or structured object>",
      "evidence_quote": "string|null",
      "source_pages": [12, 47],
      "verdict": "PASS|PASS_WITH_NOTE|RE_EXTRACT|HUMAN_REVIEW",
      "rubric_answers": {"Q1": "yes", "Q2": "yes", "Q6b": "no"},
      "composite": 0.62,
      "applied_tau": 0.71,
      "gated_to_human": true,
      "gate_reason": "composite_below_tau|bucket_degenerate_severe|policy_private_individual|critic_verdict|null",
      "failure_tag": "T01|...|null",
      "bucket": "summary_narrative",
      "artifact_stage": "v1",
      "judge_model": "opus|sonnet"
    }
  }
  ```
- `rubric_answers` includes `Q6b` even though Q6(b) is logged-only at v1 (§3.5) — that's where the offline audit reads it from. For `summary_of_interest` it also includes `Q7a` / `Q7b` / `Q7c`.
- `gate_reason` distinguishes *why* a field was routed to human, which matters for diagnosing whether the gate is too conservative vs. the Critic being the binding constraint.
- `artifact_stage` records which M-Cal stage produced the thresholds, so grades collected under v1 can be distinguished from grades collected under v2 during later recalibration.
- **`summary_of_interest` is always emitted, including when empty.** An empty list is a substantive result (the document is routine), not a missing value, and must be distinguishable from "the field failed to generate." Use `"extracted_value": []` for a legitimate empty result and `"extracted_value": null` plus a `gate_reason` for a generation failure.

### 3.13 `segment_b/year_adjudicator.py` (new, M1)
- Adjudicator **always runs**. Inputs: all year mentions in first 5pp + last 3pp of front matter + signature/approval page (keyword-detected: `Approved | Signed | Date: | Transmittal`).
- Sonnet prompt with priority rule: **transmittal/signature dates outrank cover-page dates; cover-page dates outrank in-body mentions.**
- Output: `{year, source_type ∈ {signature, transmittal, cover, body, adjudicated}, confidence}`.

### 3.14 M2 summary prompt amendment — plain-language clause
Segment B inherits M2 (Opus map-reduce) from Segment A. This amendment is an **in-place edit** to the existing M2 summary prompts in `segment_a/m2.py`, not a new module.

**Rationale.** Segment A summaries are technically correct but presume NEPA fluency and often describe projects in abstract or evasive terms. A reader with no NEPA background should be able to answer, from the summary alone: *what is being proposed? where? who does it affect? what are the environmental stakes? what is actually being decided?* Both jargon and vague nominalizations obstruct that. The five `summary.*` subfields already map onto these reader questions; the fix is a clarity constraint on how each is written, not a schema change.

**Clause (append to each `summary.*` subfield's map and reduce prompts):**

> "Write for a reader with no background in NEPA or federal environmental review. Two constraints:
>
> **1. Plain language.** When you use a domain-specific term (e.g., 'cumulative effects', 'tiered review', 'Section 106 consultation', 'de minimis finding', 'Preferred Alternative', 'scoping', 'programmatic EIS'), briefly gloss it in-line on first mention within this subfield, using text drawn from or directly supported by the cited pages. Same for regulatory citations ('40 CFR §1502', 'NEPA §101'). Prefer plain nouns and active voice over nominalizations and passive constructions. Acronym expansion is handled by a separate post-processor; you do NOT need to expand acronyms yourself, but you MUST NOT use undefined domain jargon.
>
> **2. Concreteness.** Describe the project and its impacts in terms a non-specialist reader can visualize and act on:
> - Name the thing being built or decided, not its regulatory category. ('A 47-mile 500-kV transmission line from Substation X to Substation Y' — not 'a linear energy infrastructure element'.)
> - Specify quantities, locations, and durations when the document provides them. ('The project would affect approximately 1,200 acres of sagebrush habitat over 30 years' — not 'the project has land-use implications'.)
> - Name affected communities concretely. ('Residents of three census tracts in south Milwaukee; the Ho-Chunk Nation; two elementary schools within 500 feet of the alignment' — not 'nearby stakeholders'.)
> - State the decision under review plainly. ('The Bureau of Land Management is deciding whether to approve, approve with modifications, or deny the proposed right-of-way' — not 'the agency is engaged in a decisional process'.)
>
> **Both constraints are bounded by the cited pages.** Every plain-language rephrasing or concrete detail must still be supported by the cited pages — do not invent glosses or specifics from world knowledge. If the document does not support a plain-language rephrasing or a concrete specification, quote the document's own language verbatim and leave it unglossed rather than fabricating."

**Failure modes caught by Critic Q6 (see §3.5):**
- Undefined domain jargon → PASS_WITH_NOTE + tag `T15_jargon_without_gloss`.
- Abstract nominalizations where the document supports a concrete description → PASS_WITH_NOTE + tag `T16_abstract_when_concrete_available`.

**Interaction with anti-hallucination:** the "bounded by the cited pages" constraint is load-bearing. Readability improvements are a plausible new hallucination vector (the model could invent a satisfying-sounding plain-language rephrasing not supported by the source). `atomic_verify.py` and the Critic's substring check remain the enforcement layer — a gloss that isn't substring-supported on the cited page fails verification exactly like any other unsupported claim.

**Coverage:** applies to all five `summary.*` subfields (`project_description`, `affected_community`, `alternatives_overview`, `environmental_impact`, `public_response`). Not applied to `themes` (closed taxonomy — no prose), `key_people` (structured), `location` (structured).

**Rollback path.** Before re-running M2 under the amended prompts (build item #4), archive the existing output: `cp -r segment_a/output/m2/ segment_a/output/m2_pre_amendment/`. Write the version marker `segment_a/output/m2/_prompt_version.txt` containing `v1_plain_language` after the rerun completes. This gives you (a) a side-by-side comparison to confirm the plain-language clause improved rather than degraded the summaries, and (b) a clean revert if it degraded them. Include a short before/after comparison on 2–3 subfields in `calibration_report.v(N).md`. If the amendment turns out to hurt — e.g., the gloss constraint causes Opus to drop substantive content to stay within cited-page support — revert the prompt, restore from `m2_pre_amendment/`, and reconsider the clause wording before re-attempting.

### 3.15 `summary_of_interest` — second, salience-weighted summary (new field)

**Design decision.** Salience is implemented as **Option A (rubric embedded in the M2 prompt)** but emitted as a **separate, additional field** rather than as a modification to the existing summary. The five existing `summary.*` subfields are unchanged in purpose and content: they remain the faithful, proportional condensation of the document. `summary_of_interest` sits **alongside** them.

**Why alongside rather than instead:**
- Zero regression risk to the existing `summary.*` fields, which are already graded in the Evaluation CSV and whose calibration buckets are already defined.
- The two summaries answer different questions. The standard summary answers *"what does this document say?"* — the appropriate output for a corpus index. `summary_of_interest` answers *"what is notable about this document relative to a typical EIS?"* — the appropriate output for research triage. Collapsing them would degrade both.
- Direct comparability: with both emitted, you can evaluate whether salience-weighting is actually surfacing useful signal, and revert to standard-summary-only at no cost if it isn't.

**Audience default (tunable).** The rubric is written for *a researcher studying environmental review and its policy/community consequences*. This is a default, not a permanent commitment — the audience framing is the single knob most worth revisiting once you've read a batch of outputs. A policy researcher, a community-impact scholar, and an environmental engineer would each weight the criteria differently, and the rubric's opening line is the place to change that.

**Schema.** `summary_of_interest` is a JSON list (possibly empty) of:
```json
{
  "claim": "string",
  "salience_criterion": "contested|unusual_impact|large_magnitude|novel_alternative|community_pushback|precedent|cross_jurisdictional",
  "page": 147,
  "evidence_quote": "string (verbatim from cited page)",
  "why_notable": "one sentence, grounded in the cited page"
}
```

**Prompt clause (new M2 call, run after the standard summary reduce step):**

> "Produce a SECOND, separate summary called `summary_of_interest`. This is NOT a replacement for the standard summary — both are emitted and both are kept.
>
> **Purpose.** Surface what a researcher studying environmental review and its policy and community consequences would find *notable* about this specific document, relative to a typical Environmental Impact Statement. Routine content belongs in the standard summary, not here.
>
> **Salience criteria.** Include a claim only if it matches one of the following, and tag it with the criterion it matches:
> - `contested` — the document records substantive disagreement: between agencies, between the agency and commenters, or among its own technical findings.
> - `unusual_impact` — an impact category, affected population, or resource that is atypical for this project type.
> - `large_magnitude` — the largest quantified impacts in the document (acreage, cost, displacement, emissions, duration, population affected).
> - `novel_alternative` — an alternative beyond the standard no-action / preferred / minor-variant pattern.
> - `community_pushback` — public comment that visibly changed the analysis, the scope, or the preferred alternative.
> - `precedent` — the document explicitly frames itself as precedent-setting, first-of-kind, or programmatic for future actions.
> - `cross_jurisdictional` — friction or coordination burden across agencies, states, or tribal nations.
>
> **Rules.**
> 1. Every claim requires a page cite and a verbatim `evidence_quote` from that page — identical evidentiary standard to the standard summary. The `why_notable` sentence must also be grounded in the cited page, not in world knowledge about NEPA practice generally.
> 2. **If the document is routine and nothing meets the criteria above, return an empty list.** An empty `summary_of_interest` is a CORRECT and expected output for an unremarkable document. **Do NOT manufacture interest.** Most EISs are routine; a pipeline that finds something 'notable' in every document is producing noise.
> 3. Do not restate the standard summary. If a claim already appears in `summary.*` and does not independently meet a salience criterion above, leave it out.
> 4. Cap at 6 claims. If more than 6 qualify, keep the most contested and the largest-magnitude.
> 5. Apply the same plain-language and concreteness constraints as §3.14 — write for a reader with no NEPA background, gloss domain terms on first mention, name concrete entities and quantities."

**Failure modes and tags (new):**
- `T17_manufactured_salience` — a claim is tagged with a salience criterion the cited page does not support (e.g. labeled `contested` when the page records no disagreement). Caught by Critic Q7(a) and by `atomic_verify.py`.
- `T18_salience_duplicates_summary` — `summary_of_interest` restates standard-summary content without independent salience justification. Caught by Critic Q7(b).

**The empty-list safeguard is the load-bearing anti-hallucination provision for this field.** A model asked "what's interesting here?" will find something, whether or not anything is. Rule 2 is therefore stated twice in the prompt (once affirmatively, once as a prohibition), the Critic checks for it explicitly (Q7c), and the diagnostic in §6 tracks the non-empty rate across the corpus. **If more than ~60% of documents produce a non-empty `summary_of_interest`, treat that as evidence the field is manufacturing salience rather than detecting it**, and tighten the rubric.

**Cost.** One additional Opus reduce call per document, operating on the already-computed chunk summaries rather than raw text — so its input is small relative to the standard summary map-reduce. Estimated marginal cost is well under the standard summary's, and is broken out separately in the Cost Summary (§2, `calibration_report`).

---

## §4. Direct answers to the four asks

### Q1. All acronyms defined on first mention
Two-stage. Pre-pass builds a per-doc glossary from parenthetical expansions in the OCR (both `"Full Name (FN)"` and `"FN (Full Name)"`) plus any front/back-matter Acronyms/Glossary section. Post-pass rewrites first occurrence per output field. Fallback: `acronym_commons.v(N).json` seed. Unknown acronyms are **tagged (T04), not rewritten** — the pipeline will not fabricate an expansion. Location: `segment_b/postproc/acronyms.py`, run after M2 extractors, before Critic. **This is a post-processor, not a prompt instruction** — prompt-only enforcement failed 8/8 in the Evaluation CSV. Determinism beats persuasion.

### Q2. Anti-hallucination safeguard (nothing from outside the text)
Four layers:
1. **Schema-level:** all `summary.*` subfields require `{claim, page, evidence_quote}` triples; free-text without triples rejected at parse.
2. **Prompt-level:** shared anti-hallucination clause in every Critic prompt (verbatim in §3.5) forcing `evidence_quote` emission before `verdict`.
3. **Deterministic override:** Critic's own `evidence_quote` substring-checked against cited pages via `quote_check.py`. Failure → HUMAN_REVIEW.
4. **Atomic decomposition + coordination-splitting** in `atomic_verify.py` across all five `summary.*` subfields: each atomic claim independently quote-verified. The Lincoln Hwy "or important wildlife habitats are affected" clause becomes its own atom and fails verification.

Prompt words alone are known-insufficient (the wildlife-habitat hallucination happened despite prompt hygiene). Layers 3 and 4 are the load-bearing ones.

### Q3. Why `key_people` is vulnerable — and the fix
**Mechanism.** The current pipeline treats "Consultation and Coordination" as source-of-truth for cooperating agencies. Real NEPA docs use that chapter as a **catch-all**: cooperating agencies (narrow legal category under 40 CFR §1501.8), consulted agencies (broader), tribal governments, and often the entire distribution list of draft-EIS recipients (libraries, NGOs, elected officials, commenters). The deterministic extractor pulls every entity from that chapter and labels them all "cooperator." Failure appears in 5/8 docs because 5/8 bundle the commenter list into the Consultation chapter. Compounding factor: pre-1978 docs predate the modern §1501.8 schema entirely.

**Fix.** Cooperating agencies extracted ONLY from subsections whose heading OCR-normalized-fuzzy-matches the whitelist. If no such subsection → Sonnet fallback + HUMAN_REVIEW. Everyone else in the Consultation chapter goes to a new `consulted_entities` bucket with role sub-tagging. Public commenters extracted ONLY from Comment/Response chapters. Pre-1978 docs bypass the modern schema (T13). Per-entity Critic role-check catches remaining mislabels. See `segment_b/postproc/key_people_pipeline.py`.

### Q4. Why `location` is vulnerable — and the fix
**Mechanism.** Three independent design flaws compound:
1. **No scope classification.** The pipeline assumes every doc has a geographic point. Fuel Economy is a national CAFE rulemaking with no place; the extractor returned nothing and there was no way to say "correct answer = national."
2. **Single-shot geocode.** Multi-site projects call the geocoder once with a concatenated string; Nominatim returns the first parse. Remaining sites are dropped.
3. **Specificity blindness.** For Airport Spur, the LLM picked "Milwaukee" because the doc says "in the Milwaukee metropolitan area" more often than it names the specific corridor. Nominatim returned Milwaukee-the-city; nothing checked whether that resolution matches the doc's actual scope.

**Fix (pipeline redesign, not prompt tweak).** Scope-conditional cascade in `segment_b/postproc/location_pipeline.py`:
1. Scope classifier: `{site, corridor, regional, national, international}`.
2. National/international → early return with `scope=national`, empty geocode list, textual "national".
3. Site extraction with admin hierarchy per site; per-site geocoding with scope-specific rules.
4. Site scope: bbox-containment cascade. Corridor: endpoints + midpoint, bbox-intersection. Regional: coarsest admin match with polygon-centroid fallback.
5. Retain textual location on geocode failure — a named place without coordinates is still valid output. Vendor stack is the US-optimized cascade specified in §3.9a (Census → GNIS → PAD-US → Mapbox → Nominatim), with a reduced-mode fallback if local assets aren't yet downloaded.

---

## §5. Prioritized build order

Ranked by ROI × inverse effort. **Item #4 (M2 prompt amendment + re-extraction of the currently-graded calibration docs) is a hard prerequisite for #5, #6 and #10** — τ must be calibrated against the same M2 prose Segment B will ship, or the frozen thresholds encode an untested distribution shift.

1. `mcal/quote_check.py` — OCR-normalized fuzzy match, ±2 page tolerance. (½ day. Unlocks meaningful `s_quote` signal for everything downstream. Highest ROI.)
2. `segment_b/critic.py` — evidence-first schema + deterministic quote-verify override + EVIDENCE input. (1 day. Requires #1.)
3. `segment_b/postproc/acronyms.py` — pre-pass glossary + post-pass rewrite. (1 day.)
4. **M2 summary prompt amendment** — plain-language + concreteness clause appended to `segment_a/m2.py` summary prompts. **Then re-run M2 on all currently-graded Segment A calibration docs** (9 at seed v1) so downstream calibration operates on the same prose Segment B will produce. Archive the pre-amendment output to `segment_a/output/m2_pre_amendment/` first (see §3.14 rollback note). (½ day prompt edit + ~½ day compute for a 9-doc rerun, scaling with corpus size at later stages. HARD PREREQUISITE for #5, #6 and #10.)
5. **`summary_of_interest` — new salience-weighted second summary** (§3.15). New M2 reduce call operating on existing chunk summaries; emitted alongside the standard summary, never replacing it. Ships in the same M2 rerun as #4. (1 day: prompt authoring + schema + wiring. Depends on #4.)
6. `mcal/atomic_verify.py` — Opus post-hoc decomposition + per-atom verification, covering the 5 `summary.*` subfields plus `summary_of_interest`. (2 days. Requires #1, #4, #5.)
7. `segment_b/postproc/key_people_pipeline.py` — role-restricted extraction + era gate + dependent-field cascade. (2 days.)
8. `segment_b/postproc/location_pipeline.py` — scope classifier + scope-conditional cascade over Census / GNIS / PAD-US / Mapbox / Nominatim. (2 days pipeline logic + 1 day for user to download PAD-US and GNIS locally.)
9. `mcal/taxonomy.py` — induction + human ratification. (1 day incl. review turnaround.)
10. `mcal/confidence.py` — composite + LOO curation slack + degeneracy gates + gate simulation, across all 7 buckets. (1 day. Requires #1, #4, #9.)
11. `mcal/critic_prompt.py` — per-field prompt files w/ failure-coverage few-shots, including the Q7 salience rubric for `summary_of_interest`. (1 day. Requires #9.)
12. `segment_b/gate.py` — HUMAN_REVIEW gate wired into Segment B. Preserves and emits raw extraction alongside gate decision (per §7 Q8). (½ day. Requires #10.)
13. `mcal/active_select.py` — next-batch selector for the multi-round protocol. **Elevated in this build order because it directly feeds the recalibration cadence**: after seed v1 freezes, `active_select` picks the next ~10 docs by predicted composite-variance × underrepresented-tag priority. Run between M-Cal rounds. (½ day. Requires #10.)
14. Opus routing for non-summary Critic calls (½ day config change).
15. `segment_b/year_adjudicator.py` — always-run adjudicator (½ day, M1 improvement, could run parallel with #1–#3).

**Explicitly deferred / skipped at current calibration scale (n < ~60):**
- DSPy/MIPRO prompt optimization (needs n≫20).
- Per-field Platt/isotonic calibration (insufficient per-field data; bucketed CP instead).
- ECE as headline metric (misleading at this scale).
- Growing the failure taxonomy mid-round (taxonomy is frozen within a stage; new codes only at the next `mcal.build --stage`).
- Fitting confidence signal weights from data (hand-set 0.5/0.5; revisit per §7.5 add-only guarantees).
- Removing private-individual → HUMAN_REVIEW (policy call, permanent).
- Salience Option B (two-pass claim tagging) — Option A ships in v1 as `summary_of_interest`; see §8.

---

## §6. Metrics + acceptance criteria

All gating targets stated as: **post-amendment rate `p̂` measured on the current stage's graded corpus (n_stage) has Clopper-Pearson 95% lower confidence bound ≥ baseline point estimate `p_0` from the original 8-doc CSV.**

Required N per target (for 80% power to detect the specified effect size) is computed and published in `calibration_report.v(N).md`. **Targets requiring N > n_stage are demoted to diagnostic for that stage** and re-evaluated as the corpus grows.

### Gating targets (all must pass — from stage v2 onward; see Overall acceptance)

| Metric | Baseline (8-doc) | Gating rule |
|---|---|---|
| Acronyms defined on first use per field | 0/8 | CP_LCB_95(p̂, n_stage) ≥ 0.70 |
| `key_people` — post-1978 docs' commenters not labeled cooperators | 3/8 | CP_LCB_95(p̂, n_stage) ≥ 0.375 |
| Missing-citation rate, `summary.public_response` | 4/8 (50%) | CP_UCB_95 of missing-rate < 0.5 |
| Numeric-hallucination rate, `summary_numeric` | 2/8 (25%) | CP_UCB_95 of hallucination-rate < 0.25 |
| Outside-text fabrication rate, `summary.*` | 1/8 | Any observed fabrication must be caught by `atomic_verify.py` (subfield → RE_EXTRACT or HUMAN_REVIEW). A fabrication that slips through the atomic verifier fails the gate regardless of count. |

### Diagnostic targets (reported, non-gating)

- `year` overall correct rate (baseline 5/8 — too noisy for gating)
- `eis_type` correct (baseline 7/8 — ceiling effect)
- `location` scope-classification correct (no baseline)
- Per-primary-site geocode success excluding national (no clean baseline)
- Critic `evidence_quote` verifiable rate (new signal)
- Gate rate at published τ (from `gate_simulation.v(N).json`)
- CP empirical coverage on calibration (should be ≥ 1−α by construction; report anyway)
- **`atomic_verify.py` false-negative rate** on correctly-graded subfields (from `atomic_verify_failure_log.v(N).json`, per §3.4). Target ≤ 10%. **Advisory at seed v1** (atom sample too small); **gating from v2 onward** — exceeding it triggers a decomposition-prompt review before freezing.
- **Null-tag rate on HUMAN_REVIEW routes.** In Segment B, whenever a field routes to HUMAN_REVIEW with `failure_tag = null`, it means the taxonomy did not have a matching category. Aggregate this per bucket in `run_manifest.json`; if the null-tag rate exceeds **15%** in any bucket during a Segment B batch, this is a signal that the taxonomy needs a `v(N+1)` refresh with new `T19+` codes. Reported in a rolling `null_tag_monitor.json` at the batch level; not a Segment B halt condition, but a mandatory input to the next M-Cal recalibration.

**`summary_of_interest` diagnostics (all non-gating — the field is new and has no baseline in the Evaluation CSV):**
- **Non-empty rate across the corpus.** Fraction of documents producing a non-empty `summary_of_interest`. **Expected to be well under 60%** — most EISs are routine. A rate above ~60% is evidence the field is manufacturing salience rather than detecting it, and calls for tightening the rubric (§3.15). A rate near 0% suggests the rubric is too strict.
- **Salience-criterion distribution.** Counts per criterion (`contested`, `unusual_impact`, `large_magnitude`, `novel_alternative`, `community_pushback`, `precedent`, `cross_jurisdictional`). A distribution collapsed onto one or two criteria suggests either a genuine corpus property or a rubric that isn't discriminating; worth a look either way.
- **`T17_manufactured_salience` and `T18_salience_duplicates_summary` rates** per batch.
- **Overlap with standard summary.** Rough textual overlap (e.g. token-level Jaccard) between `summary_of_interest` claims and the standard `summary.*` fields. High overlap means the field is duplicating rather than complementing.
- **Human usefulness spot-check.** On the first graded batch, the reviewer records a simple yes/no per document: *"did `summary_of_interest` tell me something the standard summary didn't?"* This is the field's real acceptance test and no automated metric substitutes for it. Recorded in the grading sheet as a new column `soi_useful`.

### Overall acceptance

**Seed v1 (n≈9):** the goal is not "meet gating targets" — it's "produce a usable seed calibration that lets the user grade the next batch efficiently." Seed v1 acceptance:
1. All 15 build items ship and pass their own unit tests.
2. `atomic_verify_failure_log.v1.json` is reviewed by user; obvious prompt gaps (coreference, negation) are patched before moving to v2.
3. `taxonomy.v1.json` is human-ratified.
4. Segment B runs end-to-end on ≥3 pilot docs and emits `run_manifest.json` with the full per-field schema in §3.12 — including `extracted_value`, `evidence_quote`, and `source_pages` for HUMAN_REVIEW / gated fields, so they can actually be graded.
5. M2 before/after comparison (per §3.14 rollback note) shows the plain-language amendment did not degrade summary substance.
6. Cost Summary in `calibration_report.v1.md` is within expected order-of-magnitude of the §7 Q2 estimate.

Seed v1 will have most buckets `degenerate_severe=true`; that's the point. **Do not gate seed v1 on the CP-interval criteria above.**

**v2 and beyond (n≥19):** all of the following must hold:
1. All gating targets in the table above pass under CP-interval discipline at that stage's n.
2. No regression on any field graded `ok` in ≥7/8 of the original seed corpus.
3. At least 4 of the 6 original buckets have `degenerate = false` in `thresholds.v(N).json`. `summary_of_interest` is excluded from this count until it has at least one fully graded batch — it starts with zero graded examples by construction, so counting it would make the criterion unreachable at v2.

If (3) fails, recalibrate at higher α or defer freezing until the next grading batch. **Full-scale Segment B** (production runs beyond calibration-targeted batches) begins only when a stage satisfies all three conditions AND the smallest non-empty bucket has `N_wrong_docs ≥ 15`.

---

## §7. Closed decisions

| # | Question | Decision |
|---|---|---|
| Q1 | Grading completeness / freeze cadence | **Multi-round protocol.** M-Cal freezes at three named stages: **seed v1** at n=9 (current), **v2** after the next ~10 graded docs (n≈19), **v3** after another ~10 (n≈29), and so on. Each stage produces a fully-frozen artifact set; Segment B pins the current stage. At seed v1 and v2, most buckets will have `N_wrong_docs < 15` and will be flagged `degenerate` or `degenerate_severe`; this is expected. Buckets with `degenerate_severe=true` route all fields to HUMAN_REVIEW. Recalibration to v(N+1) happens whenever an additional ~10 docs are graded; `mcal.build --stage v2 --prior v1` (etc.) reads the v1 taxonomy add-only and refits thresholds on the augmented calibration set. Full-scale Segment B (i.e., production runs beyond calibration-targeted batches) begins only when the smallest non-empty bucket has `N_wrong_docs ≥ 15`. |
| Q2 | Opus judge budget | Opus for the five `summary.*` subfields' atomic verification (`project_description`, `affected_community`, `alternatives_overview`, `environmental_impact`, `public_response`). Sonnet for everything else. Estimated ~1.4× total Segment B judge cost, ~$3–4/doc marginal on those fields, ~$6–8k marginal at 2000 docs. Fallback if too expensive: 2-Sonnet ensemble on the same five. |
| Q3 | Critic cited-pages input | Critic receives the full OCR text of `[min(cited_pages)−2 .. max(cited_pages)+2]` interleaved with `[[PAGE n]]` markers, in a dedicated `EVIDENCE` prompt section. Baked into all prompts in §3.5. |
| Q4 | α value | α = 0.15 across all buckets. Degenerate buckets get `α_effective = 0.25`. Not renegotiated per-field. |
| Q5 | Blinded reviewer UI | Two-column CSV workflow. Reviewer fills `your_grade` and `your_failure_tag` in one pass **without seeing `critic_verdict`**; a second pass reveals the Critic column for meta-analysis only. No new tool. **Grading sheets gain one doc-level column, `soi_useful` (yes/no)**: did `summary_of_interest` tell the reviewer something the standard summary didn't? This is the real acceptance test for the new field (§3.15, §6) and no automated metric substitutes for it. |
| Q6 | Acronym commons seeding | Seed with ~40-entry NEPA commons list AND induce doc-level glossary per document. Doc glossary takes priority; commons is fallback. |
| Q7 | Artifact location | `May25/mcal/artifacts/`. M-Cal is its own step; not nested under `segment_a/output/`. |
| Q8 | Segment B failure handling | `RE_EXTRACT` → one automated re-extraction attempt with temperature +0.2, re-run through Critic. `HUMAN_REVIEW` → the raw extraction and Critic output are still **fully emitted** to `run_manifest.json`; the field is just flagged for human review, not skipped. Gated fields (composite ≤ τ_deployed) also emit their raw extraction alongside the HUMAN_REVIEW flag. **This is critical for the multi-round protocol**: at seed v1 and v2, most fields will be gated, and the human reviewer needs the raw extractions in order to grade them and produce the next round's calibration data. |

---

## §7.5 Multi-round calibration protocol

This section formalizes the user's iterative calibration workflow. It supersedes any earlier language that assumes a one-shot M-Cal build.

**Rationale.** Segment A grading is expensive. Rather than block M-Cal on completing all 20 grades, the user runs calibration in rounds: build a seed calibration on what's graded so far, use it (via `active_select.py`) to pick the next batch, grade that batch under the seed pipeline, recalibrate, repeat. This matches the "re-calibration cadence" concept in Pipeline.pdf but at a tighter interval (~10 docs, not ~50).

**Round schedule (indicative):**

| Round | Grades in | Named artifact stage | Segment B mode |
|---|---|---|---|
| 0 | 9 (current) | *pre-M-Cal* | — |
| 1 | 9 | **seed v1** | Mostly HUMAN_REVIEW; used to extract on next-batch docs so the user can grade them |
| 2 | ~19 | **v2** | Fewer degenerate buckets; still calibration-oriented, not full production |
| 3 | ~29 | **v3** | Non-degenerate buckets start meeting CP-interval acceptance |
| 4+ | ~40+ | v4, v5, ... | Full-scale Segment B begins when smallest bucket has `N_wrong_docs ≥ 15` |

**Between rounds:** the user runs `active_select.py` to pick the next ~10 docs from the pool; runs Segment B under the current artifact stage over those docs; grades the outputs (including reviewing raw extractions on HUMAN_REVIEW / gated fields, which is where most fresh calibration signal comes from); then triggers `mcal.build --stage v(N+1) --prior v(N)`.

**Add-only guarantees carried across rounds:**
- `taxonomy` may add new codes (`T19+` for now, since v1 seeds with T01–T18), never rename or drop.
- Confidence-signal weights are frozen at 0.5·s_quote + 0.5·s_critic through at least v3; weight validation graduates from diagnostic to advisory-only until n ≥ 60 (rough rule-of-thumb where per-field AUROC CIs become interpretable).
- Bucket definitions (M1, summary_narrative, summary_numeric, summary_of_interest, alternatives+themes, location, key_people) are frozen.
- Anti-hallucination architecture (schema, evidence-first Critic, quote-verify override, atomic_verify) is frozen.

**What *does* change round-to-round:** τ_deployed per bucket, degeneracy flags, LOO curation slack, few-shot exemplars in Critic prompts (as more graded exemplars become available), false-negative audit gating status, and the roster of docs in `next_batch.csv`.

**When Segment B goes full-scale.** Not before the six original buckets satisfy the v2+ acceptance criteria in §6 simultaneously (`summary_of_interest` is evaluated on its own diagnostics, not as a full-scale blocker). Until then, Segment B runs are calibration-targeted batches, not production.

---

## §8. Deferred (noted for follow-up)

- **Salience Option B (two-pass claim tagging).** Salience is implemented in v1 as **Option A** — rubric embedded in the M2 prompt, emitted as the separate `summary_of_interest` field (§3.15). The heavier Option B design (map step emits `salience_reason` tags on every candidate claim; reduce step prioritizes over the tagged pool) remains deferred. It would give better recall on salient content buried deep in long documents, at the cost of touching the M2 map step and increasing per-chunk output size. **Revisit if the `summary_of_interest` diagnostics in §6 show low recall** — specifically, if the human usefulness spot-check (`soi_useful`) comes back negative on documents you know to be interesting.
- **Audience re-tuning for the salience rubric.** §3.15 defaults to "a researcher studying environmental review and its policy and community consequences." After reading a batch of outputs, this is the knob most worth adjusting. Alternative framings to consider: community-impact scholar (weight `community_pushback`, `unusual_impact` higher), policy/legal researcher (weight `precedent`, `contested`, `cross_jurisdictional`), environmental engineer (weight `large_magnitude`, `novel_alternative`).

---

## §9. Provenance

- Research report: Anthropic Claude general-purpose research agent, targeted literature review on hallucination reduction, LLM-as-judge reliability, confidence calibration from small human-labeled sets, learning from small graded sets, HITL queue design, EIS-specific gotchas.
- Plan authored via adversarial Planner↔Critic loop across 3 rounds (v1 → critique → v2 → critique → v3 → AGREED).
- Independent fresh-eyes review round after v3 lockdown surfaced 8 substantive fixes (S1–S8) plus 4 minor cleanup items; all folded into this document (build ordering, atomic_verify coreference/negation rules, geocoder graceful degradation, "private individual" operational definition, CP guarantee per-doc notation, LOO curation-slack scope, acronyms gate tightening to LCB ≥ 0.70, null-tag monitoring, s_quote M1 default, Q6(b) demotion, cost summary in calibration report).
- Multi-round calibration protocol added post-review (§0, §3.7, §6, §7 Q1, §7 Q8, §7.5) to support the user's chosen workflow of building seed v1 on 9 grades, iterating via `active_select.py` at ~10-doc cadence, and reaching full-scale Segment B only after buckets clear CP-interval acceptance.
- Consistency pass (final) resolved 15 items: duplicated §8 block; `run_manifest.json` schema expanded to carry `extracted_value` / `evidence_quote` / `source_pages` / `rubric_answers` / `gate_reason` / `artifact_stage` (required for grading gated fields); stale "vendor stack TBD" in §4 Q4; build-order cross-references; taxonomy rule corrected to T17+/T01–T16; `next_batch.csv` sized to ~10 docs to match cadence; weight-freeze horizon aligned between §3.3 and §7.5; private-individual definition moved out of the middle of the rubric list; `alternatives_overview` double-counting removed in §3.11/§4/§7 Q2; `.v1` → `.v(N)` stage-versioning throughout; `n=20` references generalized to `n_stage`; dangling "(3)" reference in §6 numbered; M2 amendment rollback path added (§3.14); no-observed-failure note added for `title`/`themes`/`lead_agency`/`summary.overview` (§1); §3.9a coverage percentages relabeled as a-priori expectations rather than measured values.
- Salience decision (final): Option A rubric, implemented as a separate additive field `summary_of_interest` (§3.15) rather than as a modification to the existing summary. Adds a 7th CP bucket, taxonomy tags T17/T18, Critic rubric Q7, and a set of §6 diagnostics centered on the non-empty rate. The existing five `summary.*` subfields are unchanged.
- Human evaluation input: `May25/Evaluation - Sheet1.csv` (8 docs of 9 graded at time of writing).
- Source pipeline spec: `May25/Pipeline.pdf`.
- Existing implementation: `May25/segment_a/`.

---

## §10. Implementation addendum — empirical findings

Added during implementation. The plan above is left as ratified; this section
records where the code deviates from it and why, so the original's provenance
stays intact. Every claim here is reproducible from the test suite.

### §10.1 Corrections to stated facts

| Plan says | Actually | Consequence |
|---|---|---|
| §0, §6: `n=9` graded at seed v1 | **n=8.** `p0491_35556036091957` has a grading sheet and OCR but no Evaluation-sheet column, so it is ungraded. | `mcal/grades.py` warns loudly. All acceptance arithmetic uses 8. |
| §3.1: grades live in `grading_sheets/*.csv` filtered on `your_grade != "ok"` | **0 of 333 rows are graded**, and `"ok"` is not in that file's vocabulary (`grading.py:21` = `correct\|minor_issue\|wrong\|cant_tell`). Grades are in the transposed, free-text `Evaluation - Sheet1.csv`. | `mcal/grades.py` reads both shapes; the Evaluation sheet is the seed-v1 source, per-doc sheets take over from v2. |
| §1(9): `location` "5/8 issues" | **6/8.** Randolph + LA Transit (no geocode), Airport Spur (specificity), Buffalo + Lincoln Hwy (partial multi-site), Fuel Economy (national). Only Operation Breakthrough and Bad Creek are clean. | `location` is the worst field in the corpus, not the second-worst. |
| §1(10): `key_people` 5/8 | 5/8 for the cooperator mode specifically, but **6/8 defective** (Fuel Economy is "nearly empty" — under-inclusion, a different failure) and Lincoln Hwy is ungraded, so the denominator is 7. | New code `T20_role_bucket_underpopulated`. |
| §0, §3.2: chunking is 125k-char with pages estimated `char_offset / 2500` | **Does not exist.** `config.py` chunks in real pages (`CHUNK_PAGES = 50`) and page numbers are exact from per-page JSON. `pages.py:52` already has a real char→page index. | ±2 page tolerance is retained on different grounds (OCR noise, page-seam straddling), documented in `settings.QUOTE_PAGE_TOLERANCE`. |
| §3.2: `partial_ratio ≥ 90 → yes, 60–90 → mixed` | The 60 floor is **below the chance ceiling**. Measured over 444 verified quotes scored against foreign documents: median 49.5, p95 58.9, and up to **67.7 for quotes under 40 chars**. | A second orthogonal gate (content-token coverage) was added. See §10.2. |
| §1(3), §1(4): both numeric errors are hallucinations from map-reduce "decoupling", fixable by atomic decomposition + substring verification | **Both are substring-true and correctly coupled.** See §10.3. | New code `T19_scope_qualifier_dropped`; generation-side mitigation in the §3.14 clause. |
| §3.5: rubrics reference tags per field | Four rubrics referenced tags absent from their own field's vocabulary (`T02` on `affected_community`, `T03` on `alternatives`/`themes`, `T05` on `lead_agency`); `T04` was unavailable on 6 fields whose shared Q3 mandates it. | `taxonomy.applies_to` widened; `critic_prompt.check_tag_references` now fails the build on any dangling reference. An untaggable defect becomes `failure_tag: null` and pollutes the §6 null-tag monitor — the very signal that decides when the taxonomy needs new codes. |

### §10.2 quote_check needs two gates, not one

`rapidfuzz.partial_ratio` slides a window over a long page and finds a decent
alignment by luck, and the shorter the needle the worse it gets. Atomic claims
are short by construction (§3.4 asks for one subject-predicate-object each), so
a ratio-only gate would have been a systematic false-accept in atomic
verification.

The Lincoln Hwy fabricated clause "or important wildlife habitats are affected"
scores **62.8** against its own cited pages — clearing the plan's 60 floor —
despite being absent from the document.

Content-token coverage (fraction of a quote's content words, ≥4 chars,
non-stopword, NEPA boilerplate excluded, present on the page) separates cleanly
because it is insensitive to window position:

```
                     true positives    foreign quotes
partial_ratio        median 100.0      median 49.5, p95 58.9, max 100.0
content coverage     median   1.00     median  0.10, p95  0.33, max   0.67
```

At `coverage ≥ 0.70`: **0.0% false-negative, 0.0% false-positive** on 444
quotes. The wildlife clause scores 0.00. Both gates must now pass; coverage is
the binding one. Reproduced by
`tests/test_quote_check.py::TestAgainstCorpus`.

### §10.3 Both "numeric hallucinations" are scope-qualifier loss, and one is uncatchable

**`summary.project_description`, LA Transit — "$659 million (Alt. V)".** The
figure is on pp.31/214/215 and *correctly* paired with Alternative V. p.283
scopes it: *"For the Rail/Bus Alternatives I–V, the capital costs in 1977
dollars range from 659 million and 1.120 billion dollars."* The summary dropped
that restriction and presented $659M as the overall minimum; the true overall
minimum is $369M (Alt. XI, an all-bus option). Not a fabrication — a dropped
qualifier. **This one is catchable** by the mandatory-qualifier rule: a range
endpoint requires a qualifier, and p.283's qualifier does not verify against the
cited p.214 window.

**`summary.environmental_impact`, LA Transit — "Magnitude 7.5".** p.146 states
verbatim: *"a Magnitude 7.5 earthquake occurring on the Newport-Inglewood
Fault"*, and p.144's Figure IV.4 "Maximum Credible Earthquake Richter Magnitude"
also lists Newport-Inglewood at **7.5**. The human's 7.0 comes from p.145:
*"a Magnitude 7.0 is reasonable for the maximum credible event"* — same fault,
same quantity. **The document contradicts itself, table versus narrative.**

Consequences, all pinned in `tests/test_env_impact_magnitude_75_vs_70.py` so
they stay visible rather than looking like a passing gate:

- `check_quote(..., require_numeric=True)` returns `verified="yes"`, `numeric_ok=True`. Correct, and unavoidable.
- §1(4)'s prescribed fix — coordination splitting — **does not fire** on this sentence; the coordination sits inside a prepositional object.
- A summary that keeps its qualifier and still picks the table value over the narrative value passes every automated check in the module. **A human reader comparing pages is the only detector.**
- §6's gating target "numeric-hallucination rate `CP_UCB_95 < 0.25`" therefore contains one case the specified mechanism cannot catch. The target should be read as covering the *fabrication* class only.

The `atomic_verify` atom schema gained a `scope_qualifier` field and a
comparative/superlative/range check, and the §3.14 clause gained a
"preserve scope qualifiers exactly" paragraph — a generation-side mitigation,
since verification cannot reach this class.

### §10.4 T01 (missing citation) is invisible to the frozen composite

The single most common failure — 10 of 30 wrong items, 4/8 docs on
`summary.public_response` — cannot be separated by `0.5·s_quote + 0.5·s_critic`.
Measured over the 40 graded summary subfields:

```
graded "ok, missing citation - pg ..."   n=10   mean s_quote 0.960   evidence/sentence 2.50
graded clean "ok"                        n=27   mean s_quote 0.988   evidence/sentence 2.10
```

A 0.028 gap, AUROC near chance. The mechanism: every citation the extractor
*did* make verifies fine — the defect is a claim made *without* one, and
`s_quote` averages only over evidence that exists. The obvious field-level proxy
points the **wrong way** (defective subfields carry *more* evidence per
sentence), which rules out a cheap fix.

Only atom-level citation coverage separates them, which makes build item #6
(`atomic_verify.py`) a prerequisite for calibrating T01 rather than an
enhancement. Exposed as `DocumentVerification.citation_rates()` in the exact
shape `confidence.compute_signals(citation_rate=...)` accepts. Pinned in
`tests/test_atomic_verify.py::TestT01Invisibility`.

### §10.5 The degeneracy gates are the conformal feasibility boundaries

§3.3's `N_wrong_docs < 6` and `< 3` gates are not heuristics. With
`k = ceil((n+1)(1-α))` as the order statistic:

```
α = 0.15:  n=6 → k=6  feasible (exactly)   n=5 → k=6  INFEASIBLE
α = 0.25:  n=3 → k=3  feasible (exactly)   n=2 → k=3  INFEASIBLE
```

`confidence.py` derives degeneracy from feasibility directly and cross-checks
against the plan's constants, so a renegotiated α moves the gates automatically
instead of silently invalidating them. A consequence worth stating: at n=6 and
n=3, `k = n`, so τ is the **maximum** observed wrong-item score. Early-stage
thresholds mean "beat every wrong item we have ever seen".

### §10.6 The composite is too coarse to gate on at seed v1

`s_quote ∈ {0, 0.5, 1}` and `s_critic ∈ {0, 0.3, 0.7, 1}` yield only ~12
distinct composite values. With n=8 docs, τ almost always lands exactly on a tie
cluster, and since acceptance is strict `>`, everything in that cluster is
gated. Observed on a dry run: **5 of 7 buckets gate 100% of items, including
`false_defer_rate = 1.0`** — every *correct* item gated too.

§7.5 predicted "mostly HUMAN_REVIEW" at seed v1 and treats it as intended. It is
worth being precise that the cause is *score granularity* as much as sample
size, and that it is fixed by the same thing that fixes §10.4: continuous
atom-level signals. Until then, seed-v1 and v2 Segment B runs are grading
instruments, not extraction runs.

### §10.7 Other deviations

- **`summary.overview` bucket.** §3.3 omits it from `summary_narrative`; §1 assigns it there. Followed §1 — otherwise the field has no bucket and can never be gated.
- **`page` is derived, not model-supplied.** §3.15's schema asks the model for `page`, but the M2 reduce payload only shows per-chunk page *ranges*, so a model-supplied page would fail verification for the wrong reason. `verify_and_locate` resolves it, as every other M2 field does.
- **M2 prompts extracted to `mcal/templates/`.** §3.14 calls for an in-place edit, but the prompts were two inline literals covering all subfields at once, and the §3.7 version marker is meaningless without a file to version. The clause now has one source of truth shared by the generator and by Critic Q6, which grades it.
- **§7 Q3's evidence window is unusable literally.** `[min−2 .. max+2]` over LA Transit's real citation set spans 187 pages. Implemented as literal-contiguous while ≤30 pages, else per-citation windows merged.
- **§3.6's "predicted composite variance" is not computable** for the candidate pool by definition — those docs have no extraction. Replaced with a documented cold-start heuristic over cheap document features, explicitly labelled `calibrated: false`.
- **`gate_reason` enum extended** with `dependent_field_cascade`, `reduced_geocoder_stack`, `extraction_missing`, `critic_missing`. The §3.12 enum cannot express them, and folding them into existing values would destroy the diagnostic that distinguishes a too-conservative gate from a binding Critic.
- **Acronym issues do not set `y_i = 0`.** §3.5 routes T04 to `PASS_WITH_NOTE`, so folding the doc-level note into correctness would mark every summary field of all 8 docs wrong and erase the distinction between an unglossed acronym and a fabricated magnitude. Recorded as an `acronym_issue` flag feeding `s_acronym` and the §6 target.
- **No graded document has an Abbreviations/Acronyms/Glossary section** — they predate the convention — so §3.8's section parser is exercised only by synthetic tests. Doc-derived expansions also faithfully carry OCR damage (`NAAQS → "National Annient Air Quality Standard"`) because §3.8 gives the doc glossary priority over the commons.
- **`year` has two unreachable targets.** Operation Breakthrough's graded 1976 appears nowhere in its OCR (max year present: 1973); Lincoln Hwy's 1971 appears only in body prose outside every window §3.13 specifies, while the document's own approval block reads "DATE: DEC 3 1976". The deterministic tier reaches 6/8 versus a 3/8 baseline.
