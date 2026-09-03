# How the pipeline works

A step-by-step account of the EIS extraction pipeline as it stands, what each
stage is responsible for, how the last round of review feedback was applied, and
the working rules for applying the next round.

Companion documents:

- `May25/segment_b/MCAL_PLAN.md` — the ratified design and its §10 implementation
  addendum (where the code deviates from the plan, and why).
- `PIPELINE_REVIEW.md` — the review itself, plus Appendix C recording what was
  applied, what was declined, and the four claims that changed once measured.

Everything below is verifiable from the test suite: `May25/tests`, 1333 tests, no
network required.

---

## 1. The shape of it

```
  MongoDB (praxis.colt)
        |
        |  Documents/extract_page_text.py          Stage 0: materialize OCR
        v
  Documents/output/<doc_id>/page_0001.json ...     exact page numbers live here
        |
        |  segment_b/run.py process_doc            one document, stages 1-10
        v
  +-----------------------------------------------------------------+
  | 1. chunk        chunks_for_doc: 50-page chunks + CEQ chapters    |
  | 2. M1           title, year, eis_type, lead_agency              |
  | 3. year adj.    signature-page year vs inventory year           |
  | 4. M2           summary.* (map -> reduce), summary_of_interest,  |
  |                 alternatives, themes, location, key_people       |
  | 5. location     postproc geocoder cascade (replaces M2's)        |
  | 6. key_people   postproc role/era pipeline (replaces M2's)       |
  | 7. acronyms     first-use glossing over summary prose            |
  | 8. Critic       one judged verdict per field, evidence-first     |
  | 9. gate         composite vs tau -> PASS or HUMAN_REVIEW         |
  | 10. sheets      blind/ and reveal/ grading CSVs                  |
  +-----------------------------------------------------------------+
        |
        |  run_manifest.json  (gradable without opening another file)
        v
  human grades  ->  mcal/build.py  ->  artifacts/v(N)-draft/  -> --ratify -> v(N)
        |                                                                     |
        +---------------------- next round pins the new stage ----------------+
```

The loop at the bottom is the point. The pipeline is not a one-shot extractor; it
is a calibration loop that tightens as documents get graded.

---

## 2. Vocabulary you need first

**15 canonical fields** (`mcal/settings.py:ALL_FIELDS`). These strings are the
keys used in thresholds, prompts, manifests and grading sheets. Nothing renames
them anywhere.

| group | fields |
|---|---|
| M1 (cheap metadata) | `title`, `year`, `eis_type`, `lead_agency` |
| summary | `summary.overview`, `summary.project_description`, `summary.affected_community`, `summary.alternatives_overview`, `summary.environmental_impact`, `summary.public_response` |
| salience | `summary_of_interest` |
| structured | `alternatives`, `themes`, `location`, `key_people` |

**7 buckets** (`settings.BUCKET_ORDER`). Fields are grouped into buckets because
a conformal threshold needs more than one field's worth of graded examples to
mean anything: `M1`, `summary_narrative`, `summary_numeric`,
`summary_of_interest`, `alternatives+themes`, `location`, `key_people`.

**Stages** are frozen artifact sets: `v1`, `v2`, ... Segment B pins one. A stage
is a directory of artifacts (`mcal/artifacts/v1/`) — taxonomy, thresholds,
confidence config, per-field Critic prompts (one per field), acronym commons,
atomic schema, two diagnostics (`weight_validation`, `gate_simulation`) and the
human-readable calibration report. Every filename is stage-versioned on disk
(`thresholds.v1.json`, `calibration_report.v1.md`); this document uses the
unversioned logical names, as the code does when it calls
`settings.artifact_path`. That is nine entries for v1; a tenth,
`atomic_verify_failure_log`, appears whenever `--atomic` runs, so treat the count
as descriptive rather than structural. Promoted stages are never edited; a new
round writes a new stage.

**Signals** are the six inputs to the confidence composite
(`mcal/confidence.py:Signals`). Only two carry weight today:

```
composite = 0.5*s_quote + 0.5*s_critic
s_source, s_citation, s_shard, s_acronym  ->  weight 0, recorded not used
```

Weights are frozen at 0.5/0.5 through at least v3 (MCAL_PLAN §7.5), and weight
validation stays advisory until roughly n ≥ 60 graded documents. That constraint
governs a lot of what follows: **a new signal cannot improve gating right now.
It can only be recorded so that it is available when the weights unfreeze.**

---

## 3. Stage 0 — materializing documents

`Documents/extract_page_text.py` reassembles per-page text out of MongoDB and
writes one JSON file per page.

```bash
# next 10 documents not yet extracted, restricted to inventory barcodes.
# Run from Documents/: --from-csv is opened relative to the CWD.
# Needs MONGO_URI in the environment.
cd Documents
python extract_page_text.py --new --limit 10 \
    --from-csv '../eis-inventory-2nd-pass(eis inventory 2nd pass csv).csv'
```

Why per-page files matter: **every page number in this pipeline is exact.** There
is no `char_offset / 2500` estimation anywhere (MCAL_PLAN §10.1 corrects the plan
on this point). `pages.Doc` keeps a real char→page index, which is what makes
quote verification against "the cited page ±2" a meaningful check rather than a
guess.

---

## 4. Stage 1 — chunking and chapter detection

`segment_a/chunk.py:chunks_for_doc` produces two things:

1. **Chunks** — 50 real pages each (`config.CHUNK_PAGES`), not character windows.
2. **CEQ chapters** — headings mapped to six canonical chapters.

Chapter detection is load-bearing well beyond the summary: `alternatives`,
`location` and `key_people` all read chapter slices, and `extract_alternatives`
has no fallback by design, so a missed Alternatives chapter yields
`alternatives: []` — which is not in `EMPTY_IS_VALID_FIELDS` and registers as a
failure rather than a result.

Detection is a three-part filter (`detect_chapters`):

- a candidate line: ≤100 chars, terminated by a newline, past a 3000-char TOC guard;
- shape: a marker prefix (`CHAPTER IV`, `2.`, `A.`, roman numerals) **or** ≥70% uppercase letters;
- the line must then start with a known alias from `config.CHAPTER_ALIASES`, longest alias winning.

Each chapter's span runs to the next *detected* heading, so the last one runs to
EOF. That is now reported as `ends_at_eof`, because a lone detection swallows the
rest of the document and `text_for_ceq_chapter` was silently returning most of a
document with no way for the caller to tell.

> **Feedback applied here (§2.9).** The review blamed the heading heuristic. It
> was the alias table. These are 1970s NEPA 102(2)(C) documents organized as a
> numbered five-point response — `1. DESCRIPTION OF PROPOSED ACTION`,
> `PROBABLE ENVIRONMENTAL IMPACTS`, `9. STAFF DISCUSSION OF SIGNIFICANT
> ENVIRONMENTAL MATTERS` — and the table held only modern CEQ vocabulary, so
> `Purpose and Need` was detected **zero** times in 21 documents. Aliases were
> taken from the headings the corpus actually contains. Result: Purpose and Need
> 0→9, Consultation 2→10, Affected Environment 1→7, documents with no chapter at
> all 1→0, runaway spans unchanged at 1. `tests/test_detect_chapters.py`.

---

## 5. Stage 2 — M1, the cheap metadata

`segment_a/m1.py`. Four fields, NUL/inventory-first rather than LLM-first.

| field | how |
|---|---|
| `title` | Inventory index (MARC 245$ab) only. Deterministic, no LLM. |
| `year` | Inventory value, cross-checked against a modal-year regex over the first 3 pages, then overridden by the year adjudicator. |
| `eis_type` | Position-scored cover-page regex, verified by one Sonnet call. |
| `lead_agency` | Inventory `710$b`, then a Sonnet fallback (fires on 4.4% of rows). |

`year` is worth stating precisely: the regex is **not** a fallback. It always
runs, as a cross-check — agreement gives `confidence: high`, disagreement keeps
the inventory value at `confidence: low` with a flag note, and the regex becomes
the value only when the inventory has none. The adjudicator (next stage) outranks
all of it.

`title` has no LLM fallback because measuring all 54,105 inventory rows showed
none lack a title; the fallback's only live trigger was an over-*long* one, i.e.
a case where the correct title was in hand and was being discarded to re-derive
it from OCR. Over-long titles are bound-volume aggregates and are truncated at
the first `;`, with the original kept under `full_title`.

> **Feedback applied here (§2.6).** `extract_eis_type` looped over
> `config.EIS_TYPE_PATTERNS` and `break`-ed on the first match, so dict order
> (`Supplemental, ROD, Draft, Final`) decided the label rather than where the
> phrase appeared — a Final EIS mentioning "record of decision" anywhere became a
> ROD. Now position-scored: on seven realistic cover-page phrasings dict order
> gets 3 wrong and position gets 0, and on the 21 materialized documents they
> disagree on one. On regex/Sonnet disagreement the field no longer prefers the
> regex (it was the wrong source in the one observed failure): it sets
> `review_required`, carries both candidates into the manifest, and the gate emits
> `m1_source_disagreement`. `tests/test_eis_type.py`.
>
> The Sonnet verifier is deliberately **not** skipped when the regex is
> unambiguous, though the review suggested it: 6 of 21 documents produce no regex
> hit at all, and the call is ~$0.008/doc — about $17 across the corpus, 0.13% of
> projected cost. Not a lever worth the only cross-check this field has.

---

## 6. Stage 3 — the year adjudicator

`segment_b/year_adjudicator.py` runs on every document. The inventory year and
the document's own year disagree often enough to matter, and the signature page is
not where you would guess: MCAL_PLAN §1(1) records the transmittal-page cause, and
`year_adjudicator.py` records the case that produced it — Lincoln Hwy, where the
date is on p.70 of 347. The adjudicator finds
the signature/transmittal date and reconciles it, and the outcome cascades: a
`year` the pipeline cannot settle produces a `dependent_field_cascade` gate reason
on `key_people`, because the era determines which roles are plausible.

---

## 7. Stage 4 — M2, the expensive extraction

`segment_a/m2.py`. This is where nearly all the money goes.

### 7.1 The summary: map then reduce

```
selected = (CEQ-tagged chunks + untagged)[:max_chunks]      max_chunks = 12
    |
    |  _summary_map_one, one call per chunk, 4 in parallel
    v
partials: per-chunk findings + verbatim candidate quotes
    |
    |  save_partials  <-- checkpoint to disk BEFORE the reduce
    v
    |  _summary_reduce   one call: synthesize 6 subfields, cite claim by claim
    |  _summary_of_interest  one call: salience-weighted second summary
    v
every quote verified against the document; verified page numbers written
into `evidence`
```

Three things worth knowing:

**The 12-chunk cap is a recall ceiling, not just a cost knob.** 12 × 50pp = 600
pages maximum summarized. A 775-page document cannot have its later content
appear in the summary at all.

**The map stage is checkpointed.** `partials` used to exist only as a local
variable and as text inside the reduce prompt, so a truncated or throttled reduce
discarded up to 12 Opus calls — not hypothetical, that is recorded as having hit 4
of 8 documents on one run. `m2.save_partials` writes them before the reduce.

**Which model runs the map step is a switch.** `config.SUMMARY_MAP_MODEL`,
default `"opus"`.

> **Feedback applied here.**
>
> **§1.1 (largest cost lever).** The map step is extraction and transcription —
> summarize one chunk, copy quotes out of it — while synthesis happens in the
> reduce. It is ~40% of Opus input at a 5× price difference, so moving it to
> Sonnet is a 40–55% cost cut. The switch exists; **the default does not move.**
> Changing an extractor's model shifts the score distribution the frozen
> thresholds were fitted to, so it has to land between stages with a rerun and a
> refit. `tests/test_llm_and_model_switch.py` asserts the default has not drifted.
>
> **§2.10 + §2.11 (silent gaps).** A failed map call reached `log.error` and
> nothing else — one document in the v1 batch reduced from 11 partials against a
> cap of 12, and its only recorded warning is the geocoder. Meanwhile
> `chunks_truncated` was written into `chunking_meta` and read by nobody. Both now
> report: failed chunk indices separately from the cap, operator warnings from
> `run.py`, and a `summary_source_incomplete` gate reason. The two causes are kept
> distinct because they have different remedies — raise the cap vs retry the
> document.
>
> **§2.3 (laundered verification).** When none of `summary.overview`'s own quotes
> verified, the code **replaced** its evidence with up to three verified quotes
> borrowed from the subfields — discarding the unverified entries exactly when they
> were the most informative thing on the row, and turning a failure into a
> clean-looking success. All 8 graded documents grade `summary.overview` "ok", a
> grade that may have been measuring the subfields. Borrowed quotes are now
> appended rather than substituted, tagged
> `evidence_source="carried_from_subfields"` with the source field, and excluded
> from `s_quote` and `s_citation` by every scoring path. `themes`, whose evidence
> is 100% borrowed by construction, gets `s_quote = None` — excluded from the
> composite, not floored at 1.0. Untagged evidence still counts as own, so nothing
> already calibrated on moves. `tests/test_carried_evidence.py`.

### 7.2 The other M2 fields

`alternatives` (Alternatives chapter), `themes` (derived from the summary text, a
taxonomy label), `location` and `key_people` (chapter slices) — the last two are
then replaced by the postproc pipelines below.

---

## 8. Stage 5 — post-processing

| step | module | what it does |
|---|---|---|
| location | `segment_b/postproc/location_pipeline.py` | Five-hop geocode cascade: Census → GNIS → PAD-US spatial join → Mapbox → Nominatim, with bbox-containment specificity checks. |
| key_people | `segment_b/postproc/key_people_pipeline.py` | Role and era tagging; consumes the adjudicated year. |
| acronyms | `segment_b/postproc/acronyms.py` | First-use glossing over summary prose only — rewriting an agency name inside `key_people` would corrupt the string the Critic must match. |

**`location` is currently in reduced mode on every document**, because PAD-US,
GNIS and a Mapbox token are not installed. That force-gates the whole bucket, so
1 of 15 fields on every document is blocked for an installation reason and
`location` has produced zero automated calibration data. This is review §2.12 and
it is the cheapest accuracy win available: two downloads and a token.

> **Defect found while applying the review (not in it).** `run.py` writes the
> per-field acronym rate to `m2["acronym_meta"]["per_field"][field]["s_acronym"]`;
> `gate.acronym_rate` read it off the field entry. So `s_acronym` was `None` on
> every field of every document in the v1 batch **while being computed and written
> on each one** — indistinguishable, in the artifacts, from a signal nobody
> implemented. Both shapes are now accepted.

---

## 9. Stage 6 — the Critic

`segment_b/critic.py`. One judged verdict per field, from a per-field rubric
frozen in the pinned stage (`artifacts/v1/critic_prompts/<field>.v1.md`).

- **Evidence-first.** The Critic is shown the cited pages, not the whole
  document. That restriction is load-bearing: it is what stops the judge from
  finding support the extractor never cited. It is also why full-document prompt
  caching is declined (§1.3) — the saving would undo the restriction.
- **Four verdicts** → `s_critic`: `PASS` 1.0, `PASS_WITH_NOTE` 0.7,
  `RE_EXTRACT` 0.3, `HUMAN_REVIEW` 0.0.
- **A quote-verify override.** If the Critic's own supporting quote does not
  verify — `no` *or* `mixed` — the verdict is forced to `HUMAN_REVIEW`. Two
  deliberate exemptions: a legitimately empty `summary_of_interest`, and
  `RE_EXTRACT` with no quote at all, so the override cannot suppress the retry
  path.
- **Failure tags** come from the stage's taxonomy, restricted per field. A
  dangling tag reference fails the build.

Judge routing is per field and independent of what segment_a used — the harder
fields go to Opus.

---

## 10. Stage 7 — confidence and the gate

`mcal/confidence.py` computes the signals; `segment_b/gate.py` decides.

```
per field:  signals -> composite -> accept iff composite > tau_deployed
                                    (tau from the pinned stage)
```

`s_quote` comes from live quote verification of the extraction's **own**
evidence — not from the Critic's quote, which would double-count one measurement
as two signals.

Quote verification (`mcal/quote_check.py`) is a **two-gate** test:

| gate | why |
|---|---|
| `partial_ratio ≥ 90` | fuzzy match tolerant of OCR damage |
| content-token coverage `≥ 0.70` | insensitive to window position |

Both must pass, with one documented exception: a quote with fewer than
`QUOTE_COVERAGE_MIN_TOKENS = 3` content words cannot be scored for coverage, so
the ratio gate decides alone. That is a genuine soft spot — short strings are
exactly where `partial_ratio` is least trustworthy — and it is the narrow case a
future round should look at.

One gate is not enough, and the numbers say so. Against foreign documents,
ratio p95 is 58.9 — but the Lincoln Hwy fabricated clause "or important wildlife
habitats are affected" scores **62.8 against its own cited pages**, clearing
MCAL_PLAN's original 60 floor, while its coverage is **0.00**.

> **Feedback applied here (§2.13).** The systematic negative control was
> cross-document only; the harder within-document case had been spot-checked on
> one example. Now measured properly: 602 decoys across the graded corpus,
> re-citing each verified quote to a page of its own document more than ±2 away.
>
> | | median | p95 | max |
> |---|---|---|---|
> | `partial_ratio` | 51.1 | **60.3** | 96.6 |
> | coverage | 0.214 | 0.571 | 1.000 |
>
> The ratio p95 of 60.3 is *above* the original 60 floor and above the 58.9
> measured cross-document — a ratio-only gate false-accepts within a document more
> readily than across documents, which is the direction that matters. Coverage
> carries the decision. 1 acceptance in 602, and it is a sentence that genuinely
> appears on both the cited page 11 and the decoy page 36 at coverage 1.000; a
> test requires any acceptance to have coverage ≥ 0.95 so a real false accept
> cannot be waved through as "the known one".

### 10.1 Gate reasons

A gated field reports **one** primary reason, chosen by priority, with every
applicable reason kept in `gate_reasons`. There are 11, most specific first:

| # | reason | remedy |
|---|---|---|
| 1 | `policy_private_individual` | none — policy exclusion |
| 2 | `dependent_field_cascade` | fix the upstream field (usually `year`) |
| 3 | `reduced_geocoder_stack` | install PAD-US / GNIS / Mapbox token |
| 4 | `bucket_degenerate_severe` | grade more documents |
| 5 | `extraction_missing` | the field produced no value |
| 6 | `critic_missing` | no Critic result supplied |
| 7 | `m1_source_disagreement` | **new** — two M1 sources disagree; a human picks |
| 8 | `summary_source_incomplete` | **new** — the summary did not read the whole document |
| 9 | `critic_verdict` | the judge objected |
| 10 | `bucket_saturated` | **new** — tau = 1.0; nothing can pass at any quality |
| 11 | `composite_below_tau` | refit the threshold |

> **Feedback applied here (§2.1).** `gate_field` never read
> `threshold.saturated`, so a bucket whose tau saturated at 1.0 — where
> `accepts()` requires `score > 1.0` and nothing can ever pass — fell through to
> `composite_below_tau`, which `gate.py`'s own docstring defines as the genuine
> "gate too conservative" signal. The live case: `summary.overview` with critic
> PASS, `s_quote` 1.0, `s_critic` 1.0, composite 1.0 — a perfect field, gated,
> with a diagnostic telling you to tune a threshold when the real message is that
> the score has no discriminative power left. Three causes now report distinctly
> because they have three different remedies: **grade more** (degenerate),
> **refit** (below tau), **find a signal that separates** (saturated).

All three new reasons are also excluded from the null-tag monitor's numerator: a
field gated for a mechanical reason has no failure tag through no fault of the
taxonomy, and counting it would inflate the very rate that decides when the
taxonomy needs new codes.

### 10.2 What the manifest carries

`run_manifest.json`, one entry per field, must be gradable **without opening
another file** — that is load-bearing for the multi-round protocol, where most
fields are gated and the reviewer's grades are the next round's calibration data.
The raw extraction is always carried through, gated or not.

Newly added audit fields: `quote_page_match` (cited vs matched page),
`extraction_review` (both candidates when M1 sources disagree), and a
`summary_coverage` / `quote_page_match` rollup at document level.

> **Feedback applied here (§2.7), with the conclusion reversed.** The review
> asked for `cited_page == matched_page` as a signal. Measured first: **0
> divergences over 1985 located quotes**, all fields, 20 documents. Mis-citation
> is structurally impossible on the extraction path, because
> `evidence.verify_and_locate` writes the page `doc.find_quote` *found* the quote
> on — the cite is derived from the match and cannot disagree with it. The
> Evaluation-sheet notes offered as evidence (`missing citation - pg 34, 36, 146`)
> are a reviewer naming pages that *should* have been cited: T01 omission, a
> different quantity. So it is instrumented as a **regression guard**, not a
> signal: the invariant holds only while the cite comes from an exact substring
> match and would break the day `find_quote` becomes the fuzzy matcher. It is
> deliberately not applied to the Critic's `source_pages`, which are a search
> window (M1 defaults to `[1,2,3]`) and report a spurious 5.4% divergence.

### 10.3 The RE_EXTRACT retry

A field the Critic marks `RE_EXTRACT` gets exactly one automated retry at
temperature +0.2 before a human sees it.

> **Feedback applied here (§2.5).** This path was **dead**. `run.py` called
> `run_gate` without a `reextract` callback and gate.py guards the whole retry on
> that argument, so all 59 `RE_EXTRACT` verdicts across the v1 batch became
> hand-reviewed `HUMAN_REVIEW` without a second attempt.
>
> Wiring it needed a prerequisite, which is why it had not been done: 41 of those
> 59 are summary fields, and re-extracting one meant redoing up to 12 Opus map
> calls. With the map checkpoint (§7.1) a summary retry re-runs only the reduce —
> one call instead of thirteen.
>
> Coverage is partial and the gaps are **recorded, not hidden**: `location` and
> `key_people` decline explicitly, because their values come from deterministic
> postproc pipelines and resampling cannot change them. A callback that silently
> returned nothing would be indistinguishable from one that tried and failed.
>
> One honesty note in the audit: no extractor accepts a temperature, so the +0.2
> goes through a scoped `llm.temperature()` override — and **Opus ignores
> temperature entirely**, so on Opus-routed fields a retry is a plain resample.
> The audit records the temperature *requested*, not applied.

---

## 11. Stage 8 — grading sheets

`mcal/grading_sheet.py` writes two CSVs per document, into two directories.

| pass | columns | purpose |
|---|---|---|
| `blind/` | field, extracted_value, quote, source_pages, quote_verified, + the reviewer's four cells | the grade that fits tau |
| `reveal/` | the same, plus critic_verdict, critic_failure_tag, critic_notes and model_confidence, inserted before the reviewer's cells | meta-analysis, after the blind pass |

Two directories rather than two filenames because `grades.load_grading_sheets`
does a non-recursive `glob("*.csv")` and would otherwise count every grade twice.

Blinding matters mechanically: the human grade and the Critic verdict must be
independent, because **their agreement is what fits the thresholds.** Anchoring
the human to the Critic inflates that agreement and biases every threshold
downstream.

> **Feedback applied here (§2.4) — this was the pre-grading blocker.** Removing
> `critic_verdict` from the blind sheet was not enough, because
> `model_confidence` stayed on it and on the manifest path carried the gate
> composite. Since
>
> ```
> composite = 0.5*s_quote + 0.5*s_critic     =>     s_critic = 2*composite - s_quote
> ```
>
> and `quote_verified` — which fixes `s_quote` — is the *adjacent blind column*,
> and the four verdict scores are distinct, the Critic verdict was **exactly
> recoverable from two columns of the "blind" sheet.** The column is now
> reveal-only. Banding the value was considered and rejected: with `s_quote`
> known, any band narrow enough to be informative still pins `s_critic` to one or
> two of four levels. A test executes the inversion so the reason stays
> demonstrated rather than asserted. `tests/test_grading_sheet.py`.
>
> Carried evidence is also now labelled `[CARRIED FROM <field>]` in the quote
> cell, and a row with nothing but borrowed quotes reads `n/a (carried)` rather
> than `yes`.

---

## 12. Stage 9 — the M-Cal calibration build

`python -m mcal.build --stage vN [--prior vN-1]`. Reads grades and artifacts on
disk, writes `artifacts/vN-draft/`. Order (`build.py:run_build`):

1. **Precheck: M2 prompt version** (halting). Refuses to calibrate unless every
   `m2.json` was generated under the current prompt version — tau must be fitted
   to the prose Segment B will actually ship.
2. **Precheck: geocoder assets** (non-halting). Failure forces `location` to
   `gate_all_to_human`.
3. **Load grades.** Both shapes: the transposed `Evaluation - Sheet1.csv` and the
   per-document sheets.
4. **Taxonomy** — carried forward add-only from `--prior`, extended by induction
   over new failure notes. T01–T18 are never renamed or dropped.
5. **Atomic verification** (`--atomic`, opt-in, expensive). One Opus
   decomposition per field in `settings.ATOMIC_VERIFY_FIELDS` — the five summary
   subfields plus `summary_of_interest`, with `summary.overview` deliberately
   excluded because its evidence is carried forward and decomposing it would
   double-count — then per-atom quote checks. Yields atom-level `s_citation`.
6. **Score the graded corpus** → signals and composites per (doc, field).
7. **Calibrate** conformal thresholds per bucket: `k = ceil((n+1)(1-α))`,
   leave-one-doc-out curation slack, degeneracy flags.
8. **Diagnostics** — `weight_validation`, `gate_simulation`.
9. **Build Critic prompts** per field, with few-shot exemplars from the grades.
10. **Acronym commons, atomic schema, `next_batch.csv`** (uncertainty sampling
    for the next ~10 documents to grade).
11. **`calibration_report.md`** — human-readable, and the thing to read before
    ratifying.

Then a human reviews the draft and promotes it:

```bash
python -m mcal.build --stage v2 --prior v1     # writes v2-draft/
python -m mcal.build --stage v2 --ratify       # promotes to v2/
```

> **Feedback applied here (§2.2), with the request found to be impossible.** The
> review asked for `s_shard` to be wired into both `run.py` and `build.py`.
> `s_shard` is **not implementable**: MCAL_PLAN §3.3 defines it via a per-atom
> `source_chunk_id`, and nothing in the repo emits that field — one grep hit, the
> `gate.py` comment *asserting* `atomic_verify` supplies it. It does not:
> `atomic_verify` decomposes the *final* prose and has no view of which map chunk
> a claim came from, and `m2` discarded the partials that carry `chunk_index`.
> Making it real needs the checkpoint (now added) plus atom→chunk attribution
> nobody has built. The stale comment is corrected in both files and a test
> asserts the signal is absent everywhere.
>
> What `build.py` did have wrong was its **fallback**. When atomic verification
> had not run, `s_citation` fell back to an evidence-level citation rate — which
> is `s_quote` under another name: numerically identical on 40/40 graded summary
> subfields under both label pairings, with matching AUROC (0.598 / 0.476). It is
> identical *by construction*: an evidence entry with pages verifies, one without
> pages cannot be located and scores 0. Weighting it at n ≥ 60 would have
> double-counted `s_quote` while the weight table claimed the weight belonged to
> citation coverage. Removed; `s_citation` is now atom-level or absent, and
> `diagnostics.n_items_with_s_citation` makes the absence visible instead of
> letting it read as a computed zero.

---

## 13. The loop across rounds

| round | grades in | stage | Segment B mode |
|---|---|---|---|
| 1 | 8 | **v1** (current) | mostly HUMAN_REVIEW; used to extract the next batch so it can be graded |
| 2 | ~19 | v2 | fewer degenerate buckets; still calibration-oriented |
| 3 | ~29 | v3 | non-degenerate buckets start meeting acceptance |
| 4+ | ~40+ | v4… | full-scale begins when the smallest bucket has `N_wrong_docs ≥ 15` |

Between rounds: `active_select.py` picks the next ~10 documents → Segment B runs
them under the current stage → you grade the output (including the raw
extractions on gated fields, which is where most fresh signal comes from) →
`mcal.build --stage v(N+1) --prior v(N)`.

**Frozen across rounds:** bucket definitions, signal weights (through v3),
taxonomy codes (add-only), the anti-hallucination architecture.
**Changes each round:** τ per bucket, degeneracy flags, curation slack, Critic
few-shot exemplars, `next_batch.csv`.

---

## 14. Running it

The `python -m ...` commands run **from `May25/`** — that is the package root for
`segment_b` and `mcal`, and importing `mcal.settings` installs the `sys.path`
bridge to segment_a's flat modules. The test command runs from the repo root.
Every corpus path resolves through `mcal.settings`, so nothing else depends on the
working directory.

```bash
cd May25

# run this first: pinned stage, blockers, geocoder stack, how many docs are ready
python -m segment_b.run status

# cost upper bound; exits without calling any model
python -m segment_b.run process --next-batch --limit 10 --estimate

# a batch, under the pinned stage
python -m segment_b.run process --next-batch --limit 10

# one document, reproducibly (no retry resampling)
python -m segment_b.run process --doc p1074_35556036525772 --no-reextract

# calibration
python -m mcal.build --status
python -m mcal.build --stage v2 --prior v1     # writes artifacts/v2-draft/
python -m mcal.build --stage v2 --ratify       # promotes to artifacts/v2/

# the tests: from the repo root, no network
cd ..
.venv/bin/python -m pytest May25/tests -q
```

Checkpointing means a re-run is cheap: `process` reuses `m1.json`, `m2.json` and
`critic.json` unless `--force` is passed, so re-running a batch to regenerate
manifests and grading sheets costs nothing but the gate. Add `--no-reextract` to
keep it free of model calls entirely.

---

## 15. How the last round of feedback was applied

Full record in `PIPELINE_REVIEW.md` Appendix C. Digest:

**Applied** — §2.1, §2.3, §2.4, §2.5, §2.6, §2.7, §2.9, §2.10, §2.11, §2.13,
§1.1 (switch only), and all of §1.5 (retry classification, partials checkpoint,
token-budget instrumentation).

**Declined, with reasons** — §1.2 Batch API (a new execution mode, not a fix, and
it changes how every run is driven); §1.3 caching (8–12%, and §1.1 would change
the prompt structure it depends on); §2.12 (needs two downloads and a token);
§2.2 `s_shard` (not implementable); the refit (needs a rerun).

**Four claims that changed once measured** — §2.2 (`s_shard` impossible; the real
defect was a duplicate signal), §2.7 (mis-citation structurally impossible on the
extraction path), §2.9 (the alias table, not the heuristic), §2.13 (the
within-document control is harder than cross-document, and the gate still holds).

**Two defects found while applying it, not in the review** — `s_acronym` never
populated; `batch_id` never passed.

---

## 16. How the next round of feedback gets applied

The rules below are what the last round produced, and they are worth following
because four of that round's headline items changed under them.

### Rule 1 — Measure the claim before writing the fix

Every substantive item in the last review was checked against the corpus before
any code moved. Four conclusions inverted. Two proposed signals would have been
actively harmful (one a duplicate of `s_quote`, one a constant). One requested
change was impossible. One diagnosis pointed at the wrong module.

The measurement is usually ten lines against `settings.M2_DIR` and
`settings.PAGES_DATA_DIR`. It costs minutes and it is the difference between
fixing the problem and fixing the description of the problem.

### Rule 2 — Classify the change before scheduling it

| class | examples | when it lands |
|---|---|---|
| **Zero-cost** | gate reasons, manifest fields, diagnostics, bug fixes, tests | immediately |
| **Changes new artifacts only** | evidence tagging, position scoring, coverage flags | immediately, but the refit must follow a **rerun** |
| **Changes a model, prompt, or threshold** | §1.1 Sonnet map, rubric edits | only at a **stage boundary**, with a rerun and a refit |
| **Costs money per document** | atomic verification in production, extra judge calls | behind an opt-in flag, default off, with the per-doc cost stated |
| **Needs a human or an install** | grading, PAD-US/GNIS/Mapbox | hand back with a checklist |

Mixing classes is how a "safe" fix silently invalidates a threshold. The
`_prompt_version` precheck exists because that already happened once.

### Rule 3 — A new signal lands at weight 0, or not at all

Weights are frozen through v3 and validation is advisory until n ≈ 60. So a new
signal cannot improve gating now; it can only be *recorded* for when they
unfreeze. Before adding one, check two things:

1. **Is it independent of `s_quote` and `s_critic`?** The removed `s_citation`
   fallback was numerically identical to `s_quote` on 40/40 items.
2. **Does it vary on this corpus?** `cited == matched` is constant at 1.0 by
   construction.

If either fails, record it as an audit quantity in the manifest — not as a signal.

### Rule 4 — Record declines where the decision is visible

A silent no-op is indistinguishable from a bug. `location` and `key_people`
decline re-extraction *in the manifest audit*, with the reason. The summary
retry declines *by name* when there is no checkpoint. `s_shard` is documented as
unimplementable in both files that used to imply otherwise.

### Rule 5 — Pin every measurement in a test

Not the conclusion, the number. `tests/test_detect_chapters.py` holds both the
old and the new alias tables and asserts the delta. `tests/test_eis_type.py`
asserts the *old* rule got 3 of 7 wrong. `test_the_control_is_actually_hard`
fails if the negative control stops exercising the gate. This is what stops a
future change from quietly undoing the fix, and it is why claims in this
repository can be checked rather than trusted.

### Rule 6 — Promoted artifacts are immutable

Never edit `artifacts/vN/`. Recalibration writes `v(N+1)-draft/` and a human
ratifies. Taxonomy is add-only. Bucket definitions are frozen.

### Rule 7 — Write the outcome down next to the critique

`MCAL_PLAN.md` §10 records where the code deviates from the ratified plan.
`PIPELINE_REVIEW.md` Appendix C records what was applied, declined, and
corrected. A future reader needs to know not just what the code does but which
recommendations it declined and on what evidence.

### The checklist

For each incoming item:

1. Restate it as a checkable claim.
2. Measure it on the corpus. Record the number.
3. If the measurement contradicts the claim, say so and stop — then decide what
   the *real* problem is.
4. Classify per Rule 2. Say what it costs and when it can land.
5. Implement, with the measurement in a comment at the code that depends on it.
6. Add tests that pin the numbers, including the pre-fix behaviour.
7. Run the full suite.
8. Update Appendix C: applied / declined / corrected, and any consequence for the
   roadmap.

---

## 17. Known-bad state you should fix first

1. **The v1 blind grading sheets on disk still contain the composite leak.** They
   were written before §2.4. Grading from them reproduces the problem the fix was
   for. Regenerating is deterministic and LLM-free — `--no-reextract` reuses the
   m1/m2/critic checkpoints — but it rewrites the manifests, so it is left for you
   to trigger.

2. **The headline AUROC is fitted on a stale artifact/label pairing.**
   `tests/test_atomic_verify.py::TestT01Invisibility` establishes that the human
   grades describe the **pre-amendment** prose, and the build-#4/#5 rerun rewrote
   all 40 labelled subfields. So the review's §0 finding — that the 95% CI on the
   composite includes 0.5 — and every threshold derived from it, pair current
   artifacts with labels that no longer describe them. **Re-grading is the only
   thing that fixes this,** and it sits upstream of the review's own stop
   condition.

3. **`location` is 100% force-gated** for want of two downloads and a token
   (§2.12), and has produced no automated calibration data at all.

4. **Three buckets are saturated**, i.e. no score separates right from wrong in
   them. More graded documents will not fix that on their own — saturation needs
   a signal that pulls wrong items off 1.0, which is what atom-level `s_citation`
   is for, and it only becomes usable when the weights unfreeze.

5. **Several fixes only bite on new artifacts.** Carried-evidence tagging,
   position-scored EIS type and the chunk-coverage flags are all written at
   extraction time. Any refit must therefore follow a rerun, not precede it.
