# EIS Metadata Extraction Pipeline — Project Goals

## Overview

Build a pipeline that extracts structured metadata from poorly-structured U.S. government Environmental Impact Statement (EIS) documents. Output is **one JSON object per source document**, designed to be queryable for research and to minimize the volume of documents requiring human verification. The data is from Northwestern Libraries (NUL) and those records will have the bureau, title and year available, usually. 

## Data Source & Format

Source documents live in a MongoDB-backed corpus, mirrored to S3 (`nu-impulse-production.s3.us-east-1.amazonaws.com`). Each EIS record is a directory keyed by record ID (e.g. `P0491_35556036063543/`) containing:

- `TXT/` — per-page OCR output: `{record_id}_{page}.txt` (raw text) and `{record_id}_{page}.json` (structured OCR data)
- `CONFIDENCES/` — per-page OCR confidence scores as JSON
- `mets.xml` / `mets.yaml` — document-level METS metadata

OCR quality varies. Documents span ~1969 to present and range from a few pages to several hundred. The METS/NUL records provide reliable values for some fields (title, lead agency, date) that should be preferred over LLM extraction where available.

There is also a docs_with_digits.json which is a large dict with record as keys and the values are the full txt of each document. Use this as a backup for data.

## Target Metadata (per document)

document title; summary; historical context; location (written location name, polygonal area and longitude/latitude); year (and month and date if possible); EIS type (draft, supplemental, final, other); alternatives; Named stakeholders with stance (supportive/opposed/mixed/neutral) and verbatim quotes, labelled in sequential order; The federal agency that produced the document; completed/incomplete (project status)

*for historical context and completed/incomplete, we'll likely need external information beyonf just the doc, so maybe skip those for now. 


## Optimization Targets

- **Low cost and high accuracy of data per document.**


## Known Hard Problems (call these out in the plan)

- **Polygonal location areas**
- **Verbatim stakeholder quotes** are a known LLM failure mode — models tend to grab emotive-sounding fragments ("We sincerely hope...") rather than stance-bearing ones. 
- **Stakeholder stance detection** depends on correctly attributing statements to entities. make sure NER, however you do it, is solid.

## Your Task
- create a plan for a pipeline to accomplish these goals. For each segment or objective, suggest a few methods of accomplishing it as well as a critereon (or a few) to ensure that the finihsed pipeline fulfills the goal/requirement. 