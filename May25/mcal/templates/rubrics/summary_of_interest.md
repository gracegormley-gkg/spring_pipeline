# Rubric overlay — `summary_of_interest`

New field (MCAL_PLAN §3.15) with **zero graded examples at seed v1**, its
own CP bucket, and its own tags T17/T18. Judged by Opus because Q7(a)/(c)
require deciding whether a salience claim is *genuinely* atypical — exactly the
reasoning task where Sonnet-tier judges are unreliable.

The load-bearing risk is manufactured salience. A model asked \"what is
interesting here?\" will find something whether or not anything is.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

A list (possibly empty) of `{claim, salience_criterion, page,
evidence_quote, why_notable}` recording what is notable about THIS document
relative to a typical EIS.

## QUESTIONS

- **Q7a.** For each entry, does the cited page actually support the assigned `salience_criterion`? An entry tagged `contested` must have a cited page showing actual disagreement, not merely a topic that *could* be contested. Is `why_notable` grounded in the cited page rather than in general knowledge about NEPA practice?
- **Q7b.** Does any entry merely restate content already present in the standard `summary.*` fields without independently meeting a salience criterion?
- **Q7c.** If the list is **empty** — is that plausibly correct for this document? An empty list on a genuinely routine EIS is a `PASS`, not a failure; you must not penalize emptiness. If the list is non-empty, is each entry genuinely atypical rather than standard EIS content?
- **Q7d.** Is each `salience_criterion` drawn from the closed vocabulary (`contested`, `unusual_impact`, `large_magnitude`, `novel_alternative`, `community_pushback`, `precedent`, `cross_jurisdictional`)?

## DECISION

1. Q7a = **no** → `RE_EXTRACT`, `failure_tag = T17_manufactured_salience`.
2. Q7c = non-empty-but-routine → `RE_EXTRACT`, `failure_tag = T17_manufactured_salience`.
3. Q7c = empty-and-plausible → `PASS`. Stop here; do not look for defects in an empty list.
4. Q7d = **no** → `PASS_WITH_NOTE`, `note` naming the off-vocabulary criterion.
5. Q7b = **yes** (duplicates the standard summary) → `PASS_WITH_NOTE`, `failure_tag = T18_salience_duplicates_summary`.
6. Otherwise fall through to the base decision table.

Most EISs are routine. If you find yourself justifying six notable claims about
an ordinary highway widening, the correct answer is that the list should have
been shorter or empty.

