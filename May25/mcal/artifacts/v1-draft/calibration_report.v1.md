# M-Cal calibration report — stage `v1`

Generated 2026-08-10T18:07:17+00:00. Draft: `True`.

## Prechecks

| check | status | detail |
|---|---|---|
| `m2_prompt_version` | PASS | v1_plain_language |
| `geocoder_assets` | ATTENTION | reduced |
| `grades` | PASS | 8 docs, 111 items, 30 wrong |

**`geocoder_assets` notes**

> PADUS_GEODATABASE_PATH unset. Download PAD-US (Protected Areas Database of the US) from USGS, unzip, and point this at the .gdb directory. ~1GB. Enables federal-lands geocoding (hop 3), the biggest expected quality win for this corpus.
> GNIS_TSV_PATH unset. Download the USGS GNIS domestic-names file from The National Map and point this at the .txt/.tsv. ~2GB. Enables named natural/cultural feature geocoding (hop 2).
> MAPBOX_TOKEN unset. Register a free Mapbox account and put the token in .env as MAPBOX_TOKEN. 100k requests/month free tier comfortably covers ~10 calls/doc x 2000 docs. Enables POI and named-highway geocoding (hop 4).

**`grades` notes**

> 8 of 21 docs with OCR on disk are graded. Ungraded: ['p0491_35556036091957', 'p0491_35556036854362', 'p1074_35556036108041', 'p1074_35556036522308', 'p1074_35556036525772', 'p1074_35556036525913', 'p1074_35556036535615', 'p1074_35556036543080', 'p1074_35556036796613', 'p1074_35556036854040', 'p1074_35556036855435', 'p1074_35556036861656', 'p1074_35556038062691']. MCAL_PLAN 0 assumes n=9 at seed v1; the actual seed n is 8.
> STALE LABELS: 8/8 graded docs have M2 output NEWER than the grade source (Evaluation - Sheet1.csv). Those grades describe prose that has since been regenerated, so the (composite, y) pairs used to fit tau are only approximately valid. This is expected immediately after the MCAL_PLAN 5 item #4 M2 rerun and is not fixable by code: either re-grade these docs against the current output, or accept that this stage's thresholds are indicative and treat the next stage -- graded under the shipped pipeline -- as the first internally consistent one. Affected: ['p1074_35556036058550', 'p1074_35556036105336', 'p1074_35556036546182', 'p1074_35556036806586', 'p1074_35556036811230', 'p1074_35556036861797', 'p1074_35556038322269', 'p1074_35556039563135']

## Calibration set

- Documents graded: **8**
- Graded items: **111**
- Wrong items: **30**
- Grade granularity: ['coarse']
- Grade sources: ['evaluation_sheet']
- Critic granularity: coarse (Segment A): summary.* subfields share one verdict

**Warnings**

- 8 of 21 docs with OCR on disk are graded. Ungraded: ['p0491_35556036091957', 'p0491_35556036854362', 'p1074_35556036108041', 'p1074_35556036522308', 'p1074_35556036525772', 'p1074_35556036525913', 'p1074_35556036535615', 'p1074_35556036543080', 'p1074_35556036796613', 'p1074_35556036854040', 'p1074_35556036855435', 'p1074_35556036861656', 'p1074_35556038062691']. MCAL_PLAN 0 assumes n=9 at seed v1; the actual seed n is 8.
- STALE LABELS: 8/8 graded docs have M2 output NEWER than the grade source (Evaluation - Sheet1.csv). Those grades describe prose that has since been regenerated, so the (composite, y) pairs used to fit tau are only approximately valid. This is expected immediately after the MCAL_PLAN 5 item #4 M2 rerun and is not fixable by code: either re-grade these docs against the current output, or accept that this stage's thresholds are indicative and treat the next stage -- graded under the shipped pipeline -- as the first internally consistent one. Affected: ['p1074_35556036058550', 'p1074_35556036105336', 'p1074_35556036546182', 'p1074_35556036806586', 'p1074_35556036811230', 'p1074_35556036861797', 'p1074_35556038322269', 'p1074_35556039563135']

## Thresholds

Accept in Segment B iff `composite > tau_deployed`.

| bucket | N_wrong_docs | tau_raw | curation_slack | tau_deployed | flags |
|---|---|---|---|---|---|
| `M1` | 3 | 0.5000 | 0.0000 | 0.5000 | degenerate |
| `summary_narrative` | 5 | 1.0000 | 0.0000 | 1.0000 | degenerate, saturated |
| `summary_numeric` | 3 | 1.0000 | 0.1500 | 1.0000 | degenerate, saturated |
| `summary_of_interest` | 0 | — | 0.0000 | ∞ | **gate_all_to_human**, degenerate_severe |
| `alternatives+themes` | 1 | — | 0.0000 | ∞ | **gate_all_to_human**, degenerate_severe |
| `location` | 6 | 1.0000 | 0.1500 | ∞ | **gate_all_to_human**, saturated |
| `key_people` | 6 | 1.0000 | 0.0000 | 1.0000 | saturated |

### Leave-one-doc-out curation slack

MCAL_PLAN 3.3 uses `max(delta)` rather than the 95th percentile, because at these sample sizes the percentile is dominated by the discreteness of the empirical quantile. Full distributions:

- `M1`: [0.0000, 0.0000, 0.0000] → max = 0.0000
- `summary_narrative`: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000] → max = 0.0000
- `summary_numeric`: [0.0000, 0.1500, 0.0000] → max = 0.1500
- `summary_of_interest`: not computable (n < 2)
- `alternatives+themes`: not computable (n < 2)
- `location`: [0.1500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000] → max = 0.1500
- `key_people`: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000] → max = 0.0000

### Per-bucket notes

**`M1`**

- n=3 wrong docs is infeasible at alpha=0.15 (needs the 4-th of 3 order statistics); relaxed to alpha_effective=0.25.
- k == n == 3: tau is the MAXIMUM observed wrong-item score, so acceptance requires beating every wrong item in the calibration set. Expected and correct at this sample size.

**`summary_narrative`**

- n=5 wrong docs is infeasible at alpha=0.15 (needs the 6-th of 5 order statistics); relaxed to alpha_effective=0.25.
- k == n == 5: tau is the MAXIMUM observed wrong-item score, so acceptance requires beating every wrong item in the calibration set. Expected and correct at this sample size.
- tau_raw + curation_slack = 1.0000 >= 1.0, clamped to 1.0. Since acceptance requires composite > tau, a clamped threshold rejects everything -- functionally equivalent to gate_all_to_human for this bucket.

**`summary_numeric`**

- n=3 wrong docs is infeasible at alpha=0.15 (needs the 4-th of 3 order statistics); relaxed to alpha_effective=0.25.
- k == n == 3: tau is the MAXIMUM observed wrong-item score, so acceptance requires beating every wrong item in the calibration set. Expected and correct at this sample size.
- tau_raw + curation_slack = 1.1500 >= 1.0, clamped to 1.0. Since acceptance requires composite > tau, a clamped threshold rejects everything -- functionally equivalent to gate_all_to_human for this bucket.

**`summary_of_interest`**

- No graded wrong items in this bucket, so there is nothing to calibrate against. Gating to HUMAN_REVIEW is the only sound action -- an empty calibration set is not evidence of quality.

**`location`**

- k == n == 6: tau is the MAXIMUM observed wrong-item score, so acceptance requires beating every wrong item in the calibration set. Expected and correct at this sample size.
- tau_raw + curation_slack = 1.1500 >= 1.0, clamped to 1.0. Since acceptance requires composite > tau, a clamped threshold rejects everything -- functionally equivalent to gate_all_to_human for this bucket.
- Forced to gate_all_to_human by the build (reduced geocoder stack). The threshold statistics above are retained for reference but are not in force.

**`key_people`**

- k == n == 6: tau is the MAXIMUM observed wrong-item score, so acceptance requires beating every wrong item in the calibration set. Expected and correct at this sample size.
- tau_raw + curation_slack = 1.0000 >= 1.0, clamped to 1.0. Since acceptance requires composite > tau, a clamped threshold rejects everything -- functionally equivalent to gate_all_to_human for this bucket.

## Acceptance status

- Buckets gated entirely: **3/7**
- Degenerate buckets: **5/7**
- Original (pre-SOI) buckets non-degenerate: **2/6** (['location', 'key_people'])
- MCAL_PLAN 6 v2+ criterion 3 (>=4 of 6 non-degenerate): **NOT MET**
- Smallest non-empty bucket N_wrong_docs: **1** (full-scale needs >= 15)
- Full-scale Segment B unlocked: **NO**

> M1 tau_deployed=0.500 >= 0.5, but the M1 composite has a hard 0.5 floor (s_quote defaults to 1.0 for fields with no verbatim quote). Every M1 field with a PASS verdict scores exactly 1.0 and everything else scores <= 0.85, so this threshold is effectively 'PASS only'. Not wrong, but worth knowing.

## Gate simulation

> In-sample: these items fitted the thresholds, so caught_error_rate is optimistic and false_defer_rate is pessimistic. Not a held-out estimate. Use as a sanity check on gate volume only.

Overall gate rate: **0.7928**

| bucket | graded items | gate rate | caught-error rate | false-defer rate |
|---|---|---|---|---|
| `M1` | 32 | 0.2812 | 1.0 | 0.1786 |
| `summary_narrative` | 40 | 1.0 | 1.0 | 1.0 |
| `summary_numeric` | 8 | 1.0 | 1.0 | 1.0 |
| `summary_of_interest` | 0 | 1.0 | None | None |
| `alternatives+themes` | 16 | 1.0 | 1.0 | 1.0 |
| `location` | 8 | 1.0 | 1.0 | 1.0 |
| `key_people` | 7 | 1.0 | 1.0 | 1.0 |

## Empirical CP coverage

Should meet target by construction; reported per MCAL_PLAN 6.

| bucket | coverage | target | meets |
|---|---|---|---|
| `M1` | 1.0 | 0.75 | True |
| `summary_narrative` | 1.0 | 0.75 | True |
| `summary_numeric` | 1.0 | 0.75 | True |
| `summary_of_interest` | 1.0 | None | None |
| `alternatives+themes` | 1.0 | None | None |
| `location` | 1.0 | None | None |
| `key_people` | 1.0 | 0.85 | True |

## Taxonomy

- Tags: **20** (seed 18, empirical 2, induced 0)
- Induction run: **True**
- Tags with exemplars: **12/20**
- Without exemplars: ['T13_pre_1978_nepa_format', 'T14_regional_scope_underspecified', 'T15_jargon_without_gloss', 'T16_abstract_when_concrete_available', 'T17_manufactured_salience', 'T18_salience_duplicates_summary', 'T19_scope_qualifier_dropped', 'T20_role_bucket_underpopulated']

Observed tag counts in the graded set:

- `T01_missing_citation`: 10
- `T05_commenter_mislabeled_as_cooperator`: 5
- `T11_year_ocr_error`: 3
- `T02_numeric_hallucination`: 2
- `T06_geocode_missing`: 2
- `T09_multi_site_partial_geocode`: 2
- `T03_outside_text_fabrication`: 1
- `T07_geocode_wrong_specificity`: 1
- `T08_scope_misclassified_national`: 1
- `T10_alternatives_chapter_missed`: 1
- `T12_eis_type_confused_with_rod`: 1

**Induction notes for the reviewer**

- Notes 5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101 all share a doc_note suggesting that for national-scope documents the location field should return 'national' rather than empty. Note 101 (grade_text: 'no location') is the clearest instance of T08_scope_misclassified_national; the others are graded 'ok' and the observation appears only in the doc_note. No new code is warranted since T08 covers the failure, but a human may want to confirm whether the 'ok' grades on those national-doc rows mean the location field was already handled correctly or was simply not penalised.
- Note 46 (summary.project_description, wrong cost figure and wrong alternative label) could be T02_numeric_hallucination or T19_scope_qualifier_dropped. The figure is attached to the wrong alternative entirely, which looks more like T02; assigned T02 but flagging for human review.
- Note 84 (alternatives field returned 'empty') is assigned T10_alternatives_chapter_missed. If the document genuinely had no alternatives chapter this would be a false positive; human should verify.

## Critic prompts

- Prompts written: **15** in `mcal/artifacts/v1-draft/critic_prompts`

| field | slots | failure examples | positive controls | tags covered |
|---|---|---|---|---|
| `title` | 3 | 0 | 3 | — |
| `year` | 3 | 1 | 2 | T11_year_ocr_error |
| `eis_type` | 3 | 1 | 2 | T12_eis_type_confused_with_rod |
| `lead_agency` | 3 | 0 | 3 | — |
| `summary.overview` | 3 | 0 | 3 | — |
| `summary.project_description` | 3 | 2 | 1 | T01_missing_citation, T02_numeric_hallucination |
| `summary.affected_community` | 3 | 1 | 2 | T01_missing_citation |
| `summary.alternatives_overview` | 3 | 1 | 2 | T01_missing_citation |
| `summary.environmental_impact` | 3 | 3 | 0 | T01_missing_citation, T02_numeric_hallucination, T03_outside_text_fabrication |
| `summary.public_response` | 3 | 1 | 2 | T01_missing_citation |
| `summary_of_interest` | 3 | 0 | 3 | — |
| `alternatives` | 3 | 1 | 2 | T10_alternatives_chapter_missed |
| `themes` | 3 | 0 | 3 | — |
| `location` | 3 | 3 | 0 | T06_geocode_missing, T09_multi_site_partial_geocode, T07_geocode_wrong_specificity |
| `key_people` | 3 | 1 | 2 | T05_commenter_mislabeled_as_cooperator |

## Weight validation (advisory, non-gating)

> n_docs=8 is below WEIGHT_VALIDATION_MIN_N=60; intervals are too wide to dominate any candidate. Weights stay frozen at 0.5/0.5 through at least v3 per MCAL_PLAN 7.5.

Resampling unit: document. 
Production Kendall tau: 0.081265

| candidate | signals | AUROC | 95% CI |
|---|---|---|---|
| `quote70_critic30` | 2 | 0.572222 | [0.46494, 0.654107] |
| `quote50_critic50` | 2 | 0.571811 | [0.468325, 0.654525] |
| `quote30_critic70` | 2 | 0.567284 | [0.464923, 0.650365] |
| `quote40_critic40_citation20` | 3 | 0.561111 | [0.442299, 0.65905] |
| `critic_only` | 1 | 0.551235 | [0.457071, 0.626078] |
| `quote_only` | 1 | 0.547531 | [0.464479, 0.628833] |
| `all_equal` | 6 | 0.506584 | [0.380141, 0.632917] |

> Tiebreaker (from the module docstring): prefer the candidate with the fewest signals at non-zero weight; break remaining ties lexicographically by candidate name.

## Cost Summary

- Documents in the most recent run: **8**
- Total: **$41.4436**
- Per-doc average: **$5.1804**
- Linear projection to 2000 docs: **$10360.9**

| model tier | calls | input tokens | output tokens | cost |
|---|---|---|---|---|
| `us.anthropic.claude-sonnet-4-6` | 98 | 538,013 | 40,759 | $2.2255 |
| `us.anthropic.claude-opus-4-7` | 57 | 1,506,996 | 190,468 | $36.89 |

> Linear projection from the most recent run.py invocation. Excludes the M-Cal Opus atomic-verify pass and Segment B's per-field Critic split, both of which add Opus calls not present in this run.

## Environment

- Geocoder stack: **reduced**
- M2 prompt version: `v1_plain_language`
- alpha = 0.15, alpha_effective(degenerate) = 0.25
- Signal weights: {'s_quote': 0.5, 's_critic': 0.5, 's_source': 0.0, 's_citation': 0.0, 's_shard': 0.0, 's_acronym': 0.0}

