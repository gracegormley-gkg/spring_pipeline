# Rubric overlay — `location`

Defective on **6 of 8** docs — the worst field in the corpus. (MCAL_PLAN
§1(9) says 5/8; it miscounts. Only Operation Breakthrough and Bad Creek are
clean.) Four distinct modes: no geocode at all, wrong specificity, partial
multi-site, and a national rulemaking treated as absent-location.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

`{scope, sites, geocoded, textual_location}` where scope is one of `site`,
`corridor`, `regional`, `national`, `international`.

## QUESTIONS

- **Q7.** Is the assigned `scope` correct for this document? A nationwide rulemaking with no project site is `national` — **not** a missing location.
- **Q8.** Is every primary site the document names present in `sites`? A multi-site project with only one site listed is incomplete even if that one geocoded.
- **Q9.** Is the resolved specificity appropriate to the scope? Returning the containing city for a specific corridor or facility is too coarse.
- **Q10.** Where geocoding failed, is the textual place name still retained? A named place without coordinates is valid output and must not be dropped.
- **Q11.** Does each site's `admin_hierarchy` match the document, and does the resolved point actually fall inside the stated county/state?

## DECISION

1. Q7 = **no** → `RE_EXTRACT`, `failure_tag = T08_scope_misclassified_national` when a national/international document was treated as sited; otherwise `null`.
2. Q8 = **no** → `RE_EXTRACT`, `failure_tag = T09_multi_site_partial_geocode`.
3. Q11 = **no** → `RE_EXTRACT`, `failure_tag = T07_geocode_wrong_specificity`.
4. Q9 = **no** → `PASS_WITH_NOTE`, `failure_tag = T07_geocode_wrong_specificity`.
5. Q10 = **no** → `RE_EXTRACT`, `failure_tag = T06_geocode_missing`.
6. Otherwise fall through to the base decision table.

`scope = national` with an empty `geocoded` list is a **`PASS`**, not a missing
location. This is the Fuel Economy case.

