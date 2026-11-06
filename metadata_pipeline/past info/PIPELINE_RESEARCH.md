# Pipeline change report — informed by two literature reviews

Companion to `PIPELINE.md` and `PIPELINE_REVIEW.md`. Produced by a two-researcher
literature pass (methodology + EIS/NEPA domain) plus a repo-expert brief.
Everything below is proposed **inside** the pipeline's existing working rules
(measure before fix; weight-0 or not at all; promoted artifacts immutable;
record declines; pin measurements in tests). No existing design decision is
negated — four are strengthened by external evidence, five are extended, and
three previously unforeseen upgrades are proposed.

Full reports:
- `/tmp/eis-methodology-lit-review.md` (conformal, grounding, long-doc, judge)
- `/tmp/eis-domain-lit-review.md` (NEPABench, NEPATEC, OCR, geocoding, 1970s)
- `/tmp/eis-pipeline-brief.md` (brief given to the researchers)

---

## 0. What the literature confirms (keep as-is)

Four decisions already in the pipeline turned out to be literature-aligned once
checked externally. These are **not changes**; they are evidence that the
existing design is doing the right thing.

- **Evidence-first Critic.** The Critic sees cited pages only. The 2024–2025
  LLM-as-judge literature confirms full-document judges inflate PASS rates via
  position/length/authority bias. Keep.
- **Blind sheet / composite-column removal.** Thakur et al. 2025 show raw
  blind-grade/Critic agreement overstates κ by 33–41 points; the pipeline's
  §2.4 fix was exactly the right instrument. Keep.
- **Chapter alias retrofit to 1970s vocabulary.** The 1973 CEQ guidelines (38 FR
  20550) codified the numbered-section template the pipeline now detects. The
  alias retrofit is pinned to the correct ground truth. Keep.
- **Removal of `s_citation` evidence-level fallback.** Multi-dim conformal
  literature (Zhao 2024, Blot 2025) is explicit that numerically redundant
  signals in composed scores are strictly harmful. Removal was correct.

---

## 1. Drop-in changes that need neither a refit nor a stage boundary

These do not touch frozen weights, thresholds, or promoted artifacts. They are
schema additions, data-resource swaps, or audit upgrades.

### 1.1 Ship `location` on a free reduced stack — defuses the 100% force-gate

Current state: `location` is force-gated on every document because PAD-US / GNIS
/ Mapbox are not installed. This is the single cheapest accuracy win the
pipeline flags, and the domain review found that **PAD-US is not actually
required to be schema-compliant with the CEQ data standard** — which defines
`location` at city/county/geographic-area resolution. GNIS covers that.

Minimum viable free stack:

- **GNIS** (USGS Board on Geographic Names) — ~2M US features, free, no key.
  REST endpoint `carto.nationalmap.gov/arcgis/rest/services/geonames/MapServer`.
  Carries Census codes natively, so GNIS→Census join is free.
- **US Census Geocoder API** — free, no key, high rate ceiling.
- **Mordecai 3** (Halterman) — open-source neural geoparser designed for
  "place name in prose," which matches 1970s EIS narrative text better than any
  address-oriented tool.

What this unlocks: `location` starts producing automated calibration data for
the first time. The force-gate becomes a *reduced-stack* gate (one reason, not
all reasons), and PAD-US / Mapbox become quality upgrades rather than
blockers.

Why this does not break the calibration loop: `location` has produced no
calibration data today, so adding data to an empty bucket is strictly
additive — no stale thresholds get re-fit against a changed distribution.

### 1.2 Add two canonical fields the schema is missing

NEPATEC v2.0 / CEQ data standard includes two fields the pipeline conflates
into `lead_agency`:

- **`project_sponsor`** — often a non-federal entity (state agency, port
  authority, utility). On 1970s cover pages this is distinct from the federal
  lead agency and frequently named explicitly.
- **`prepared_by`** — the contractor or consultant that drafted the EIS (e.g.,
  "Prepared for the Corps of Engineers by Dames & Moore, Inc."). A 1970s role
  that `lead_agency` cannot represent.

Add-only: new fields, new buckets (or add to M1 if cheap deterministic). Does
not touch existing field definitions. Follows the "add-only" convention the
pipeline already applies to taxonomy codes.

### 1.3 Benchmark against EISBench + MAPLEv2 on shared fields

NEPABench's `EISBench` (1,365 entries, 14 fields) and the MAPLEv2 evaluator
(`github.com/pnnl/PermitAI`) overlap with 7 of this pipeline's 15 fields:
`title`, `year`, `eis_type`, `lead_agency`, `location`, `themes`,
`alternatives`. The pipeline currently has no externally comparable numbers.
Running MAPLEv2 on the graded corpus produces per-field numbers that let
reviewers — and future funders — compare without having to accept the
pipeline's internal grading scheme at face value.

Zero-cost: evaluation only, no change to extraction or gating.

### 1.4 Replace hand-coded role/era resolution with cached Wikidata

Wikidata exposes `P39` (position held) with `P580/P582` (start/end time)
qualifiers. One SPARQL query, cacheable overnight, resolves (agency, year) →
plausible officials. The domain review provides a drop-in query shape for US
Secretary of the Interior; the pattern generalizes to cabinet secretaries,
assistant secretaries, agency administrators, and regional directors.

Fallbacks: Federal Register API, American Presidency Project (UCSB).

Why this is not weight-territory: this replaces a *table*, not a signal. The
`key_people` postproc pipeline already consumes the adjudicated year — this
upgrades the source of truth it joins against.

### 1.5 Pin the chapter alias table to the 1973 CEQ guidelines (38 FR 20550)

The current alias table was fitted inductively on 21 observed documents and the
fix delivered large detection gains. The 1973 CEQ guidelines are the *de jure*
source for 1970s 102(2)(C) section structure, and the review identified two
sections the pipeline is not currently slicing:

- **§7 — Objections raised by other Federal/State/local agencies**
- **§8 — Discussion of problems and objections**

Both carry named officials with dated agency affiliations — a direct,
structured source for `key_people` that bypasses role/era inference entirely.
Add §7 and §8 to the alias set and to the `key_people` chapter slices.

### 1.6 Record LOO threshold stability per bucket per round (audit)

Methodology finding with the hardest edge: **per-bucket conformal thresholds
with ~5–6 samples per bucket are below the α=0.05 split-conformal validity
floor of 19**. The current refit ritual is doing real-looking work but
produces high-variance estimates. The literature-aligned mitigation that
respects the pipeline's "don't touch what's frozen" discipline is to
**measure** the variance and surface it:

- Per round, per bucket: `max |τ_k(fit) − τ_k(LOO)|` across leave-one-doc-out
  folds (Liang et al., ICLR 2025 LOO-StabCP). Flag buckets where the LOO span
  exceeds the current threshold interval as "under-powered." No auto-adjust.
- Per round: Spearman ρ matrix across the six recorded signals, with a
  pre-registered redundancy veto at ρ > 0.9. Avoids the "`s_citation ≡ s_quote`
  on 40/40" surprise next time by inspection rather than retrospectively.
- Headline metric is **κ**, not raw agreement, per bucket per round.

Zero-cost, pure diagnostic. The honest message is that current thin-bucket
thresholds are not finite-sample valid in the usual sense; the diagnostic
makes that visible without pretending otherwise.

---

## 2. Weight-0 signals worth recording now

Each is independent of `s_quote` and `s_critic` by construction, has a
plausible path to being discriminative on wrong items currently stuck at 1.0,
and would land as recorded-only until weights unfreeze at v4+.

| Signal | Stage | What it addresses |
|---|---|---|
| `s_coverage_margin` (continuous value on [0.70, 1.0]) | quote-gate | Saturation tie-break — a quote passing at 0.71 vs 1.00 is distinguishable |
| `s_citation_breadth` (cited-page count − 1) | post-extract | Saturation tie-break — single-page citations rank below multi-page corroborated |
| `s_adjacent_concordance` (short spans only; p±1, p±2 match) | quote-gate | Within-doc decoy signature for <3-token spans |
| `s_span_rarity` (bigram-IDF within document) | quote-gate | Common-bigram false matches ("the project") get down-weighted |
| `s_short_span_requires_corroboration` (bool, audit) | quote-gate | Routes <3-token spans to HUMAN_REVIEW regardless of ratio — see §4 |
| `s_recall_ceiling_exposed` (bool: `pages_covered < total_pages`) | M2 input | Lets 20–40 more graded docs answer "does the ceiling actually hurt a visible field?" |
| `s_critic_swap` (verdict stability under paraphrased extraction quote) | critic | LLM-judge reliability (Thakur 2025); orthogonal to `s_critic` value |
| `s_critic_adjacent_contrast` (verdict with ±1 pages added) | critic | Evidence-Lock mitigation (arXiv 2508.05353) — see §3 |
| `s_resample_agreement` (N=3, cheap stages only — M1, year, Critic; **not** M2 Opus) | multiple | Non-temperature consistency signal for Opus-ignored-temp case |
| `s_prompt_paraphrase_delta` (RE_EXTRACT only) | retry | Evidence for whether paraphrase retry outperforms temperature retry, before replacing the primary path |

Everything above lands as a column in the manifest / grading sheets' *reveal*
side (not blind — some are derivable from the composite and must not leak).
No weight change, no threshold change.

---

## 3. The highest-value contradictions to pipeline assumptions

The pipeline's working rule is "measure the claim before writing the fix."
These four are the places where the literature flatly contradicts an implicit
or explicit pipeline assumption. Each is phrased as a checkable claim.

1. **"Evidence-first Critic is strictly conservative" is only half-true.**
   It is conservative *against false-refutal* (judge invents support that
   isn't cited) but *permissive against incompleteness-driven false-passing*
   (judge can only verify what the extractor already chose to cite). Source:
   *Evidence Lock Before Commitment*, arXiv 2508.05353, 2025. **Cheap
   mitigation:** `s_critic_adjacent_contrast` — re-run the Critic with cited
   pages ±1, record whether the verdict flips. Weight 0.

2. **Per-bucket thresholds with n ≈ 5–6 are below the split-conformal
   validity floor.** The α=0.05 floor is 19 per stratum. Current refits are
   high-variance estimates not finite-sample valid in the usual sense.
   Mitigation: (a) LOO stability diagnostic (§1.6), (b) at v4 consider
   pooling buckets with a Mondrian weakening or marginal thresholds with
   bucket as a recorded covariate. Not a v3 change; a v4 question.

3. **One-shot temperature retry has no literature support on Opus-class
   models.** Opus ignores temperature; the self-consistency literature
   (Taubenfeld 2025, Qiang EACL 2024) prefers *paraphrase retry* and
   *N≥3 resample with weighted vote*. **Measurement path first**
   (`s_prompt_paraphrase_delta`); replacement path at stage boundary.

4. **Ratio-only verification on <3-content-token spans in OCR-noisy
   historical documents is a known high-false-accept regime.** Nguyen et al.
   2025 shows OCR-noise QA degrades sharply below ~15 characters. The
   defensible policy is to route those cases to HUMAN_REVIEW rather than let
   `partial_ratio` decide alone — this is a verdict-routing rule, not a
   signal, and is implementable as an audit-flag now and a routing rule at
   v4.

---

## 4. Between-stage changes (land at the next stage boundary)

These change extraction behavior and therefore must be bundled with a rerun
and refit, per the pipeline's own classification rules.

- **Static prompt-paraphrase retry replacing temperature +0.2 retry.** The
  paraphrase is version-controlled, deterministic, independent of the first
  attempt. Does not require an adaptive retry policy. Costs the same as the
  current retry.
- **Hierarchical / two-level reduce for documents past the 12-chunk cap.**
  ToM (arXiv 2511.00489, EMNLP 2025) is the direct precedent: reduce chunks
  1–6 → A, 7–12 → B, 13–16 → C, then reduce {A, B, C}. Keeps every reduce
  call under caching reach. Only lands when (a) `s_recall_ceiling_exposed`
  calibration data shows the ceiling hurts a visible field and (b) the
  threshold refit is budgeted.
- **Hard routing rule: <3-content-token quotes → HUMAN_REVIEW regardless of
  verdict.** Contingent on the audit flag (§3 item 4) producing a measurable
  false-accept signal.

---

## 5. Experiments to run before any of §4

Each is cheap and answers a question that gates one of the §4 changes.

- **Re-OCR a 50-doc sample through olmOCR 2 or Churro** and compare Gate A/B
  pass-rate uplift. If HathiTrust's Northwestern 33,000-volume EIS collection
  already carries OCR for in-corpus documents, use that first. If OCR quality
  is the actual bottleneck, the recall-ceiling and short-span problems may
  shrink before they are "solved."
- **Read Raab et al. 2025 (arXiv:2507.08109)** on auditing/aligning LM-powered
  subroutines before v3 weights unfreeze. Substantively the same calibration
  problem as M-Cal; worth a close read for signal choices.
- **Run MAPLEv2 on the current graded corpus** (§1.3). Numbers before any
  changes land are the baseline for every subsequent claim.

---

## 6. Explicitly do NOT do

Each is a tempting change the literature does not support in this context, or
that the pipeline has already evaluated and declined for good reason.

- **Do not add atom→chunk attribution yet to resurrect `s_shard`.** The
  pipeline shows it is not implementable without infrastructure nobody has
  built, and the methodology review did not surface any technique that would
  make it both cheap and discriminative.
- **Do not replace the two-gate quote check.** Already stress-tested on 602
  decoys. Augment with `s_adjacent_concordance` and `s_span_rarity` — do not
  substitute.
- **Do not enable full-document prompt caching for the Critic.** Would undo
  the Evidence Lock mitigation and inflate PASS rates on well-cited but
  incomplete extractions.
- **Do not unfreeze weights before n ≈ 60.** The literature confirms
  small-sample conformal is where threshold variance is worst; adding weighted
  signals mid-stream compounds the problem rather than solving it.
- **Do not expand the alias table beyond 1973 CEQ guidelines without a
  measurement.** Inductive expansion is how the 1970s/modern vocabulary
  mismatch happened in the first place.

---

## 7. Priority order

Grouped by cost and risk, highest leverage first.

1. §1.1 reduced-stack `location` — defuses a 100% force-gate.
2. §1.2 `project_sponsor` + `prepared_by` schema add — closes a canonical gap.
3. §1.3 MAPLEv2 on shared fields — gets externally comparable numbers on the
   record.
4. §1.6 LOO threshold stability + Spearman ρ matrix + κ-as-headline — makes
   the current calibration honesty visible.
5. §2 weight-0 signals (all 10) — logged in parallel; most of the work is one
   pass through the manifest writer.
6. §1.4 Wikidata role/era + §1.5 1973 CEQ alias pin + §7/§8 slices — domain
   wins that don't touch calibration.
7. §5 experiments — gate §4 changes.
8. §4 between-stage changes — only after a rerun is budgeted.

---

## Sources

Full citation lists in the two companion reports. Load-bearing citations:

- Thurin et al. 2025, Optimal Transport-based Conformal Prediction (arXiv:2501.18991).
- Blot et al. 2025, Benchmarking non-conformity score functions (arXiv:2505.24983).
- Liang et al. 2025, Leave-One-Out Stable Conformal Prediction (ICLR).
- Yang et al. 2025, Selective Conformal Uncertainty in LLMs (ACL).
- Farquhar et al. 2024, Semantic entropy (Nature 630:625).
- Guo et al. 2025, ToM: Tree-oriented MapReduce (EMNLP 2025, arXiv:2511.00489).
- Thakur et al. 2025, Reliability without Validity: LLM-as-a-Judge (arXiv:2506.19544).
- Evidence Lock Before Commitment, 2025 (arXiv:2508.05353).
- Nguyen et al. 2025, LLM Robustness on Noisy OCR QA (arXiv:2502.16781).
- Taubenfeld et al. 2025, CISC: Confidence Improves Self-Consistency (arXiv:2502.06233).
- Meyur et al. 2024/2025, Benchmarking LLMs for Environmental Review (arXiv:2407.07321, SIGKDD 2025 workshop).
- Raab et al. 2025, Audit, Alignment, and Optimization of LM-Powered Subroutines (arXiv:2507.08109).
- PNNL NEPATEC v2.0 technical report, Aug 2025.
- 1973 CEQ Guidelines, 38 FR 20550.
- olmOCR 2 (arXiv:2502.18443); Churro (arXiv:2509.19768); PreP-OCR (ACL 2025).
