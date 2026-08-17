"""
Segment B driver: end-to-end extraction over a batch of documents.

    python -m segment_b.run process --next-batch
    python -m segment_b.run process --doc p1074_35556036535615
    python -m segment_b.run process --docs a,b,c --stage v1
    python -m segment_b.run status

MCAL_PLAN's build order (5) enumerates fifteen components but never a driver to
wire them, so this module exists to close that gap. It is the thing 7.5 means by
"runs Segment B under the current artifact stage over those docs".

Per document, in order:

    1. load OCR + chunk                  (segment_a: pages, chunk)
    2. M1 metadata                       (segment_a.m1)
    3. year adjudication                 (segment_b.year_adjudicator)   -- always runs
    4. M2 summary + summary_of_interest   (segment_a.m2)
    5. location pipeline                 (postproc.location_pipeline)   -- replaces M2's
    6. key_people pipeline               (postproc.key_people_pipeline) -- replaces M2's
    7. acronym post-pass                 (postproc.acronyms)            -- after M2, before Critic
    8. per-field Critic                  (segment_b.critic)
    9. conformal gate + run_manifest     (segment_b.gate)
   10. blind + reveal grading sheets     (mcal.grading_sheet)

Steps 5 and 6 overwrite the M2 fields rather than running alongside them: the
old extractors are the failure modes 1(9) and 1(10) describe, and leaving both
in the output would make it ambiguous which one the Critic judged. Step 7 runs
after all extraction and before the Critic because the acronym rewrite changes
the text the Critic reads, and 4 Q1 requires the rewrite to be deterministic
rather than something the Critic negotiates.

Ordering note on the era gate: key_people depends on the year VERDICT, which
only exists after the Critic has run. Rather than run the Critic twice, the
pipeline seeds the gate with the adjudicator's own confidence, then
`critic.run_critic` applies the authoritative dependent-field cascade over the
finished verdicts (`settings.DEPENDENT_FIELDS`). Both layers are recorded.

Cost: every LLM call is wrapped in a usage session, so each doc's manifest and
the batch summary carry real token counts and dollar figures. At observed Segment
A rates (~$5.62/doc) plus the per-field Critic split and the salience call,
budget noticeably more per doc than Segment A -- check `--estimate` first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Callable, Optional, Sequence

from mcal import grading_sheet, settings, taxonomy

from . import critic as critic_mod
from . import gate as gate_mod
from . import year_adjudicator
from .postproc import acronyms as acronyms_mod
from .postproc import key_people_pipeline, location_pipeline

log = logging.getLogger("segment_b.run")

# segment_a flat modules (bridged by mcal.settings on import).
import m1 as m1_mod  # noqa: E402
import m2 as m2_mod  # noqa: E402
from chunk import chunks_for_doc  # noqa: E402
from inventory import lookup_work  # noqa: E402
from llm import aggregate_usages, end_usage_session, start_usage_session  # noqa: E402
from pages import load_doc  # noqa: E402


OUTPUT_DIR = settings.SEGMENT_B_DIR / "output"

# Fields whose text the acronym post-pass rewrites. Structured fields are left
# alone: rewriting an agency name inside `key_people` would corrupt the string
# the Critic must match against the cited passage.
ACRONYM_REWRITE_FIELDS = settings.SUMMARY_FIELDS + (settings.SUMMARY_OF_INTEREST,)


@dataclass
class DocResult:
    doc_id: str
    title: str = ""
    n_pages: int = 0
    elapsed_sec: float = 0.0
    stage: str = ""
    paths: dict[str, str] = dc_field(default_factory=dict)
    rollup: dict = dc_field(default_factory=dict)
    usage: dict = dc_field(default_factory=dict)
    warnings: list[str] = dc_field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "doc_id": self.doc_id,
            "title": self.title,
            "n_pages": self.n_pages,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "stage": self.stage,
            "paths": self.paths,
            "rollup": self.rollup,
            "usage": self.usage,
            "warnings": self.warnings,
        }
        if self.error:
            d["error"] = self.error
        return d


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return path


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.warning(f"  corrupt checkpoint {path.name}: {e}; recomputing")
        return None


# --- Pipeline ---------------------------------------------------------------


def process_doc(
    doc_id: str,
    *,
    stage: str,
    out_dir: Optional[Path] = None,
    force: bool = False,
    write_sheets: bool = True,
    critic_call: Optional[Callable] = None,
) -> DocResult:
    """
    Run the whole Segment B pipeline over one document.

    Stage outputs are checkpointed per doc under
    `<out_dir>/<doc_id>/{m1,m2,critic,run_manifest}.json`, and reused unless
    `force` is set -- the same cheap-rerun property segment_a/run.py has, which
    matters because the Critic split makes a full doc materially more expensive
    than it was in Segment A.
    """
    base = (out_dir or OUTPUT_DIR) / settings.normalize_doc_id(doc_id)
    result = DocResult(doc_id=doc_id, stage=stage)
    t0 = time.time()

    doc = load_doc(doc_id)
    result.n_pages = doc.n_pages
    work = lookup_work(doc_id)
    title = ""
    if work:
        title = work.get("title") or (work.get("nul_metadata") or {}).get("title") or ""
    result.title = title

    log.info(f"=== {doc_id} | {title[:60]!r} ({doc.n_pages} pages) ===")

    start_usage_session()
    try:
        chunked = chunks_for_doc(doc)
        chunks, chapters = chunked["chunks"], chunked["chapters"]
        log.info(
            f"  {len(chunks)} chunks, {len(chapters)} CEQ chapters detected"
        )

        # --- 2. M1 ---
        m1_path = base / "m1.json"
        m1 = None if force else _read_json(m1_path)
        if m1 is None:
            m1 = m1_mod.run_m1(work or {}, doc)
            _write_json(m1_path, m1)
        log.info(f"  M1: year={_val(m1, 'year')} type={_val(m1, 'eis_type')}")

        # --- 3. year adjudicator (always runs) ---
        adj_path = base / "year_adjudication.json"
        adj = None if force else _read_json(adj_path)
        if adj is None:
            try:
                adj = year_adjudicator.adjudicate(doc, m1_year=_val(m1, "year"))
            except Exception as e:
                log.warning(f"  year adjudicator failed: {e}")
                adj = {"year": None, "source_type": None, "note": str(e)}
            _write_json(adj_path, adj)
        m1 = _apply_year_adjudication(m1, adj, result)
        _write_json(m1_path, m1)

        # --- 4. M2 (+ summary_of_interest) ---
        m2_path = base / "m2.json"
        m2 = None if force else _read_json(m2_path)
        if m2 is None:
            m2 = m2_mod.run_m2(doc, chunked=chunked)
            _write_json(m2_path, m2)
        soi = m2.get("summary_of_interest")
        log.info(
            f"  M2: summary_of_interest = "
            f"{len(soi) if isinstance(soi, list) else 'MISSING'} claim(s)"
        )

        # --- 5. location pipeline (replaces M2's) ---
        try:
            loc = location_pipeline.run_location_pipeline(doc, chapters)
            m2["location"] = location_pipeline.as_m2_location_field(loc)
            m2["location_pipeline"] = loc
            log.info(
                f"  location: scope={loc.get('scope')} "
                f"sites={len(loc.get('sites') or [])} "
                f"geocoded={len(loc.get('geocoded') or [])} "
                f"stack={loc.get('geocoder_stack')}"
            )
            if loc.get("reduced_mode"):
                result.warnings.append(
                    "location resolved in reduced geocoder mode; the bucket is "
                    "force-gated to HUMAN_REVIEW"
                )
        except Exception as e:
            log.warning(f"  location pipeline failed, keeping M2's: {e}")
            result.warnings.append(f"location_pipeline: {e}")

        # --- 6. key_people pipeline (replaces M2's) ---
        try:
            kp = key_people_pipeline.run_key_people_pipeline(
                doc,
                chapters,
                year=_val(m1, "year"),
                m1=m1,
            )
            m2["key_people"] = key_people_pipeline.as_m2_key_people_field(kp)
            m2["key_people_pipeline"] = kp
            log.info(
                f"  key_people: preparers={len(kp.get('agency_preparers') or [])} "
                f"cooperating={len(kp.get('cooperating_agencies') or [])} "
                f"consulted={len(kp.get('consulted_entities') or [])} "
                f"commenters={len(kp.get('public_commenters') or [])}"
            )
        except Exception as e:
            log.warning(f"  key_people pipeline failed, keeping M2's: {e}")
            result.warnings.append(f"key_people_pipeline: {e}")

        # --- 7. acronym post-pass ---
        try:
            acr_stats = _apply_acronyms(m2, doc, stage)
            log.info(
                f"  acronyms: {acr_stats['n_glossary']} in glossary, "
                f"{acr_stats['n_rewritten']} first-use rewrites, "
                f"{len(acr_stats['undefined'])} undefined"
            )
            m2["acronym_meta"] = acr_stats
        except Exception as e:
            log.warning(f"  acronym post-pass failed: {e}")
            result.warnings.append(f"acronyms: {e}")

        _write_json(m2_path, m2)

        # --- 8. Critic (per field) ---
        critic_path = base / "critic.json"
        cached = None if force else _read_json(critic_path)
        if cached is not None:
            critic_results = critic_mod.results_from_payload(cached)
        else:
            critic_results = critic_mod.run_critic(
                doc, m1, m2, stage=stage, call=critic_call
            )
            _write_json(
                critic_path,
                {f: r.to_dict() for f, r in critic_results.items()},
            )
        _log_verdicts(critic_results)

        # --- 9. gate + manifest ---
        gated = gate_mod.run_gate(
            doc_id,
            m1,
            m2,
            critic_results,
            stage=stage,
            doc=doc,
            out_dir=base.parent,
        )
        result.rollup = gated.rollup if hasattr(gated, "rollup") else {}
        manifest = gated.manifest if hasattr(gated, "manifest") else {}
        result.paths["run_manifest"] = str(base / gate_mod.MANIFEST_FILENAME)
        _log_gate(result.rollup)

        # --- 10. grading sheets ---
        if write_sheets:
            try:
                tax = taxonomy.load_current(stage)
                sheets = grading_sheet.build_and_write(
                    base,
                    doc_id,
                    m1=m1,
                    m2=m2,
                    manifest=manifest,
                    doc=doc,
                    title=title,
                    artifact_stage=stage,
                    tax=tax,
                )
                for name, p in sheets.items():
                    result.paths[f"sheet_{name}"] = str(p)
                log.info(f"  sheets: {', '.join(sorted(sheets))}")
            except Exception as e:
                log.warning(f"  grading sheet write failed: {e}")
                result.warnings.append(f"grading_sheet: {e}")

        result.paths.update(
            {"m1": str(m1_path), "m2": str(m2_path), "critic": str(critic_path)}
        )

    except Exception as e:
        log.exception(f"  FAILED: {e}")
        result.error = str(e)
    finally:
        usages = end_usage_session()
        result.usage = aggregate_usages(usages)
        result.elapsed_sec = time.time() - t0

    cost = ((result.usage or {}).get("total") or {}).get("cost_usd", 0)
    log.info(f"  done in {result.elapsed_sec:.0f}s, ${cost:.2f}")
    return result


def _val(m1: Optional[dict], field: str):
    entry = (m1 or {}).get(field)
    return entry.get("value") if isinstance(entry, dict) else entry


def _apply_year_adjudication(m1: dict, adj: dict, result: DocResult) -> dict:
    """
    Fold the adjudicator's verdict into M1's `year`.

    The adjudicator outranks M1 whenever it produced a year, because M1's
    first-3-pages regex is the documented cause of 3/8 wrong years (1(1)) and
    the adjudicator reads the signature/transmittal pages M1 never sees. M1's
    original value is preserved under `year.m1_value` so the disagreement stays
    auditable rather than being silently overwritten.
    """
    m1 = dict(m1 or {})
    adj_year = adj.get("year")
    entry = dict(m1.get("year") or {})
    entry["adjudication"] = adj
    if adj_year:
        old = entry.get("value")
        if old != adj_year:
            result.warnings.append(
                f"year: adjudicator returned {adj_year} "
                f"(source={adj.get('source_type')}), M1 had {old}"
            )
            log.info(
                f"  year adjudicated {old} -> {adj_year} "
                f"({adj.get('source_type')}, {adj.get('confidence')})"
            )
        entry["m1_value"] = old
        entry["value"] = adj_year
        entry["source_type"] = adj.get("source_type")
        entry["confidence"] = adj.get("confidence") or entry.get("confidence")
    m1["year"] = entry
    return m1


def _apply_acronyms(m2: dict, doc, stage: str) -> dict:
    """Build the glossary once, then rewrite first uses in every prose field."""
    glossary = acronyms_mod.build_glossary(doc, stage=stage)
    n_rewritten = 0
    undefined: dict[str, list[str]] = {}
    per_field: dict[str, dict] = {}

    for field in ACRONYM_REWRITE_FIELDS:
        if field == settings.SUMMARY_OF_INTEREST:
            entries = m2.get("summary_of_interest")
            if not isinstance(entries, list):
                continue
            for i, item in enumerate(entries):
                if not isinstance(item, dict):
                    continue
                for key in ("claim", "why_notable"):
                    text = item.get(key) or ""
                    if not text:
                        continue
                    new, tags, stats = acronyms_mod.annotate_field(text, glossary)
                    item[key] = new
                    n_rewritten += stats.get("n_rewritten", 0)
                    if stats.get("undefined"):
                        undefined[f"{field}[{i}].{key}"] = list(stats["undefined"])
            continue

        sub = field.split(".", 1)[1]
        entry = (m2.get("summary") or {}).get(sub)
        if not isinstance(entry, dict):
            continue
        text = entry.get("text") or ""
        if not text:
            continue
        new, tags, stats = acronyms_mod.annotate_field(text, glossary)
        entry["text"] = new
        entry["acronym_tags"] = tags
        entry["acronym_stats"] = stats
        n_rewritten += stats.get("n_rewritten", 0)
        if stats.get("undefined"):
            undefined[field] = list(stats["undefined"])
        per_field[field] = {
            "s_acronym": acronyms_mod.defined_first_use_rate(stats),
            "n_rewritten": stats.get("n_rewritten", 0),
            "undefined": list(stats.get("undefined") or []),
        }

    return {
        "n_glossary": len(getattr(glossary, "entries", {}) or {}),
        "n_rewritten": n_rewritten,
        "undefined": undefined,
        "per_field": per_field,
    }


def _log_verdicts(critic_results: dict) -> None:
    counts: dict[str, int] = {}
    for r in critic_results.values():
        v = getattr(r, "verdict", None) or "?"
        counts[v] = counts.get(v, 0) + 1
    log.info(
        "  critic: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        + f" over {len(critic_results)} fields"
    )


def _log_gate(rollup: dict) -> None:
    if not rollup:
        return
    log.info(
        f"  gate: {rollup.get('n_gated')} gated / "
        f"{rollup.get('n_fields')} fields"
    )
    reasons = rollup.get("gate_reasons") or {}
    if reasons:
        log.info(
            "  reasons: "
            + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        )


# --- Batch ------------------------------------------------------------------


def resolve_batch(
    *,
    doc_ids: Optional[Sequence[str]] = None,
    next_batch: bool = False,
    limit: Optional[int] = None,
) -> list[str]:
    """
    Work out which documents to process.

    `--next-batch` reads `artifacts/next_batch.csv` -- the roster
    `mcal.active_select` produced -- so the multi-round protocol in 7.5 does not
    depend on the user retyping doc ids.
    """
    if doc_ids:
        ids = [settings.normalize_doc_id(d) for d in doc_ids if d.strip()]
    elif next_batch:
        path = settings.NEXT_BATCH_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Generate it with:\n"
                f"    python -m mcal.active_select --n {settings.NEXT_BATCH_SIZE}\n"
                f"or run a build, which writes it as a side effect."
            )
        import csv

        with open(path, newline="", encoding="utf-8") as fh:
            ids = [
                settings.normalize_doc_id(row["doc_id"])
                for row in csv.DictReader(fh)
                if row.get("doc_id")
            ]
    else:
        raise ValueError("give --doc, --docs, or --next-batch")

    # Drop anything without materialized OCR rather than failing mid-batch.
    resolved: list[str] = []
    for d in ids:
        if settings.resolve_doc_dir(d) is None:
            log.warning(f"skipping {d}: no per-page OCR under {settings.PAGES_DATA_DIR}")
            continue
        resolved.append(d)

    return resolved[:limit] if limit else resolved


def process_batch(
    doc_ids: Sequence[str],
    *,
    stage: Optional[str] = None,
    out_dir: Optional[Path] = None,
    force: bool = False,
    write_sheets: bool = True,
) -> dict:
    """Process a batch and write `<out_dir>/batch_summary.json`."""
    stage = critic_mod.resolve_stage(stage)
    out = out_dir or OUTPUT_DIR
    log.info(f"Segment B: {len(doc_ids)} doc(s) at artifact stage {stage}")

    results: list[DocResult] = []
    for i, doc_id in enumerate(doc_ids, 1):
        log.info(f"\n[{i}/{len(doc_ids)}]")
        try:
            results.append(
                process_doc(
                    doc_id,
                    stage=stage,
                    out_dir=out,
                    force=force,
                    write_sheets=write_sheets,
                )
            )
        except Exception as e:
            log.exception(f"{doc_id} failed outside the usage session: {e}")
            results.append(DocResult(doc_id=doc_id, stage=stage, error=str(e)))

    ok = [r for r in results if not r.error]
    total_cost = sum(
        ((r.usage or {}).get("total") or {}).get("cost_usd", 0) or 0 for r in ok
    )
    n_fields = sum((r.rollup or {}).get("n_fields", 0) for r in ok)
    n_gated = sum((r.rollup or {}).get("n_gated", 0) for r in ok)

    summary = {
        "stage": stage,
        "n_docs": len(results),
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "total_cost_usd": round(total_cost, 4),
        "per_doc_avg_usd": round(total_cost / len(ok), 4) if ok else None,
        "projected_2000_docs_usd": (
            round(total_cost / len(ok) * 2000, 2) if ok else None
        ),
        "n_fields": n_fields,
        "n_gated": n_gated,
        "overall_gate_rate": round(n_gated / n_fields, 4) if n_fields else None,
        "docs": [r.to_dict() for r in results],
    }
    _write_json(out / "batch_summary.json", summary)

    log.info("")
    log.info(
        f"Batch done: {len(ok)}/{len(results)} ok, ${total_cost:.2f} total, "
        f"gate rate {summary['overall_gate_rate']}"
    )
    log.info(f"Grade the blind sheets under {out}/<doc_id>/, then:")
    log.info(f"    python -m mcal.build --stage v{settings.stage_number(stage)+1} "
             f"--prior {stage}")
    return summary


def estimate(doc_ids: Sequence[str]) -> dict:
    """
    Rough cost projection before spending anything.

    Derived from Segment A's observed $5.62/doc, adjusted for the two changes
    that add calls: the per-field Critic split (9 -> 15 judgements, six of them
    on Opus) and the salience reduce call. Deliberately crude -- it exists to
    catch an order-of-magnitude surprise, not to be accurate.
    """
    SEGMENT_A_PER_DOC = 5.62
    CRITIC_SPLIT_MULTIPLIER = 1.4   # 7 Q2's own estimate
    SALIENCE_ADD = 0.60
    per_doc = SEGMENT_A_PER_DOC * CRITIC_SPLIT_MULTIPLIER + SALIENCE_ADD

    pages = []
    for d in doc_ids:
        dd = settings.resolve_doc_dir(d)
        pages.append(len(list(dd.glob("*.json"))) if dd else 0)

    return {
        "n_docs": len(doc_ids),
        "total_pages": sum(pages),
        "assumed_per_doc_usd": round(per_doc, 2),
        "estimated_total_usd": round(per_doc * len(doc_ids), 2),
        "basis": (
            f"segment_a observed ${SEGMENT_A_PER_DOC}/doc (91% Opus) x "
            f"{CRITIC_SPLIT_MULTIPLIER} for the per-field Critic split "
            f"(MCAL_PLAN 7 Q2) + ${SALIENCE_ADD} for the salience reduce call"
        ),
        "caveat": (
            "Page counts vary 30-775 in this corpus and M2 caps at 12 chunks, "
            "so long docs cost less than linear. Treat as an upper bound."
        ),
        "docs": [{"doc_id": d, "pages": p} for d, p in zip(doc_ids, pages)],
    }


# --- CLI --------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m segment_b.run",
        description="Run Segment B extraction over a batch of documents.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="extract, judge and gate a batch")
    p.add_argument("--doc", help="single doc_id")
    p.add_argument("--docs", help="comma-separated doc_ids")
    p.add_argument(
        "--next-batch",
        action="store_true",
        help="use artifacts/next_batch.csv from mcal.active_select",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--stage", help="artifact stage to pin (default: latest promoted)")
    p.add_argument("--out", help="output dir (default: segment_b/output)")
    p.add_argument("--force", action="store_true", help="ignore checkpoints")
    p.add_argument("--no-sheets", action="store_true", help="skip grading sheets")
    p.add_argument(
        "--estimate",
        action="store_true",
        help="print a cost estimate and exit without calling any model",
    )

    sub.add_parser("status", help="show what is ready and what is missing")

    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.cmd == "status":
        print(json.dumps(status(), indent=2))
        return 0

    try:
        doc_ids = resolve_batch(
            doc_ids=(
                [args.doc]
                if args.doc
                else (args.docs.split(",") if args.docs else None)
            ),
            next_batch=args.next_batch,
            limit=args.limit,
        )
    except (ValueError, FileNotFoundError) as e:
        log.error(str(e))
        return 2

    if not doc_ids:
        log.error("no processable documents resolved")
        return 2

    if args.estimate:
        print(json.dumps(estimate(doc_ids), indent=2))
        return 0

    try:
        critic_mod.resolve_stage(args.stage)
    except critic_mod.MissingArtifactError as e:
        log.error(f"\n{e}")
        return 3

    summary = process_batch(
        doc_ids,
        stage=args.stage,
        out_dir=Path(args.out) if args.out else None,
        force=args.force,
        write_sheets=not args.no_sheets,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "docs"}, indent=2))
    return 0 if summary["n_failed"] == 0 else 1


def status() -> dict:
    """Readiness check: can Segment B run right now, and if not, why not."""
    stage = settings.latest_stage()
    blockers: list[str] = []
    if stage is None:
        blockers.append(
            "No promoted M-Cal stage. Run `python -m mcal.build --stage v1` "
            "then `--stage v1 --ratify`."
        )
    geo = settings.geocoder_precheck()
    if geo["stack"] != "full":
        blockers.append(
            f"Geocoder in reduced mode (missing {geo['missing']}); the location "
            f"bucket will be force-gated to HUMAN_REVIEW."
        )
    if not settings.NEXT_BATCH_PATH.exists():
        blockers.append(
            f"{settings.NEXT_BATCH_PATH.name} not generated; use --doc/--docs "
            f"or run `python -m mcal.active_select`."
        )

    return {
        "ready": stage is not None,
        "pinned_stage": stage,
        "blockers": blockers,
        "materialized_docs": len(settings.available_doc_ids()),
        "output_dir": str(OUTPUT_DIR),
        "next_batch_exists": settings.NEXT_BATCH_PATH.exists(),
    }


if __name__ == "__main__":
    sys.exit(main())
