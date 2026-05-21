# Test Run 1 — Apollo Space Program EIS

**Date:** 2026-05-13
**Document:** Apollo Space Program Environmental Impact Statement (Final, 1972)
**Accession key:** `P0491_35556036056489`
**Input source:** `docs_with_digits.json`
**Output:** `GKG_Test 1/apollo_result.json`

---

## Document Profile

| | |
|---|---|
| Title | Apollo Space Program : environmental impact statement. |
| Lead agency | National Aeronautics and Space Administration |
| Year | 1972 |
| Word count | 5,464 (short) |
| Pages (fake) | 14 |
| EIS type | Unlabelled (see below) |
| Chunks | 1 (whole doc — short doc path) |

---

## Cost & Performance

| | |
|---|---|
| Total tokens | 32,733 |
| Total cost | $0.39 |
| Warnings | 1 (`ocr_confidence_unavailable` — expected, using JSON source) |
| Run time | ~30 seconds |

---

## Field-by-Field Results

### Summary ✅ Strong
The summary accurately covers all four required elements: community (Titusville, Cocoa Beach, Atlantic/Pacific ocean areas), project goal (manned lunar landings and exploration), justification (scientific knowledge, space operations, U.S. international leadership), and environmental impacts (atmospheric pollution from exhaust, noise up to 131 dB, water pollution from RP-1 fuel, plutonium-238 radiation risk). Specific figures from the document are cited correctly. Reads clearly for a public audience.

### Location ✅ Correct
Extracted "Kennedy Space Center, Florida" and geocoded it accurately to lat 28.52, lon -80.68.

### Themes ⚠️ Vocabulary gap
Returned `other / unclassified`. The Apollo program is a space exploration project — it doesn't fit cleanly into any of the 12 current themes. The closest available option would be `defense_and_military` but that's genuinely a poor match. This is a known limitation: the theme taxonomy needs a `aerospace_and_space_exploration` entry, or this type of doc will always land on `other`.

### Alternatives ⚠️ Retrieval miss
Returned empty. The document does reference an alternatives section ("Alternatives were considered with respect to different technical ways of accomplishing the lunar landing objective"), but the single chunk wasn't labeled with the `alternatives` topic tag by the chunk labeler. Since Stage 2 retrieves by tag, it got no context and returned nothing. **Root cause:** the chunk labeler gave the doc 6 correct tags but missed `alternatives`. This is the core risk of tag-based retrieval on short single-chunk docs.

### Key People & Groups ⚠️ NER not running
Returned empty — spaCy is not installed, so Stage 0 NER was skipped. The document has real extractable entities: George Marienthal (EPA Acting Director, wrote the comment letter), the Interagency Safety Evaluation Panel, the Council on Environmental Quality, and the Atomic Energy Commission. These would be captured once spaCy is installed.

### Historical Context — Internal ✅ Correctly empty
Returned `insufficient_information`. This is the right answer for this document: the Apollo program was ongoing at the time of writing, so there is no historical backstory to extract in the way there would be for a proposed dam or highway project. The model correctly recognized there was nothing meaningful to return rather than fabricating context.

### EIS Type ⚠️ Correctly flagged as ambiguous
Returned `Unlabelled`. The cover says "FINAL Environmental Statement" but the document also repeatedly references "the draft statement" (meaning the prior draft version it supersedes). Both patterns matched, so the pipeline correctly declined to guess. The document is genuinely Final — a post-processing rule could resolve this (if the doc explicitly says "Final" in the title, prefer that).

### Lead Agency ✅ Correct
"United States. National Aeronautics and Space Administration" pulled directly from the NUL API (METS source). No regex or fuzzy matching needed.

---

## What the NUL API Fetch Got Us

Without any manual configuration, the pipeline automatically fetched title, agency, and date from the Northwestern University Libraries Digital Collections API using the accession number. This means even running from the flat `docs_with_digits.json` format, the record gets accurate catalog metadata.

---

## Issues to Fix Before Next Run

| Priority | Issue | Fix |
|----------|-------|-----|
| High | spaCy not installed — no NER, no key people | `pip install spacy && python -m spacy download en_core_web_lg` |
| Medium | Theme taxonomy missing `aerospace_and_space_exploration` | Add to `pipeline/config.py` THEMES dict |
| Low | EIS type logic: prefer "Final" when title explicitly says so | Add title-check tiebreaker in `stage0_triage._eis_type()` |

---

## Overall Assessment

For a first run on a real document, the results are solid. The hardest field (summary) worked well. The empty fields are mostly explained by missing infrastructure (spaCy) or a genuine vocabulary gap (themes), not prompt failures. The cost per document is low enough to run the full 181-doc collection once infrastructure gaps are closed.

**Next recommended test:** a medium-length highway or dam EIS from the 1980s — a more typical document where headings, alternatives, and named stakeholders are all present and the theme taxonomy is a better fit.
