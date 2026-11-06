# Rubric overlay — `key_people`

Defective on 6/8 docs; 5/8 with the same mode — every entity in the
Consultation chapter labelled a cooperating agency (MCAL_PLAN §1(10)). Real NEPA
documents use that chapter as a catch-all: cooperating agencies, consulted
agencies, tribes, and the entire draft-EIS distribution list. 40 CFR §1501.8
defines cooperating agency narrowly.

Adds to `_base.md`. Questions are numbered from Q7 so they never collide with
the shared Q1–Q6.

---

## FIELD_DESCRIPTION

`{agency_preparers, cooperating_agencies, consulted_entities,
public_commenters}`. `consulted_entities` is a distinct bucket and is never
labelled \"cooperator\".

## QUESTIONS

- **Q7.** For **each** entity in `cooperating_agencies`: does the cited passage show the document **formally designating** it a cooperating agency (or joint lead / assisting agency), under 40 CFR §1501.8 or its predecessor CEQ guidance? Mere appearance in a consultation, distribution or comment list is NOT designation.
- **Q8.** Is any entity in `cooperating_agencies` actually a commenter, a draft-EIS recipient, a consulted agency, a library, or an NGO?
- **Q9.** For each `public_commenters` entry: does it come from a comment/response chapter or hearing transcript, rather than from a distribution list?
- **Q10.** Is the document pre-1978? If so, the modern cooperating-agency schema does not apply and that bucket cannot be populated by designation.
- **Q11.** For each commenter with a stance: is the person a **private individual** per PRIVATE_INDIVIDUAL, and is their capacity unambiguous in the cited passage?

## DECISION

1. Q11 = private individual with a stance, **or** capacity ambiguous → `HUMAN_REVIEW`, `failure_tag = null`. Policy; overrides everything below.
2. Q8 = **yes** → `RE_EXTRACT`, `failure_tag = T05_commenter_mislabeled_as_cooperator`; `note` must name the misfiled entity.
3. Q7 = **no** for any entity → `RE_EXTRACT`, `failure_tag = T05_commenter_mislabeled_as_cooperator`.
4. Q10 = **yes** and `cooperating_agencies` is non-empty → `HUMAN_REVIEW`, `failure_tag = T13_pre_1978_nepa_format`.
5. Q9 = **no** → `RE_EXTRACT`, `failure_tag = T05_commenter_mislabeled_as_cooperator`.
6. Otherwise fall through to the base decision table.

An **empty** `cooperating_agencies` list is the correct answer when the document
designates none. Prefer empty over speculative. Do not treat emptiness as a
defect.

