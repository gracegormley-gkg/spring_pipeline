"""
Per-row Critic for the people pipeline (Sonnet).

For each merged row, we resend the cited pages (pulled in as actual text, per
the v2 spec for M2 Check) and ask Sonnet to verify:

  1. Quote is verbatim on the cited pages.
  2. Stance attribution belongs to the named entity (not the document narrator
     and not a different speaker on the same page).
  3. Stance label matches the closed vocabulary and the quote supports it.
  4. Entity name is well-formed for the entity kind (e.g. private individuals
     should be last-name-only or 'private commenter' per v2 plan).

Verdicts: PASS | PASS_WITH_NOTE | RE_EXTRACT | HUMAN_REVIEW.

Hard overrides (applied AFTER the model's verdict):
  - `summary_quote_verified == False` → HUMAN_REVIEW (we never auto-pass an
    unverified quote).
  - `kind == 'individual'` → HUMAN_REVIEW. Per the v2 plan: stance attribution
    for private individuals always goes to a human, regardless of Critic.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import settings  # registers segment_a/ on sys.path

from chunk import page_range_chars         # from segment_a/
from config import MODEL_SONNET             # from segment_a/
from llm import call_json_with_usage        # from segment_a/

log = logging.getLogger(__name__)

VERDICTS = ("PASS", "PASS_WITH_NOTE", "RE_EXTRACT", "HUMAN_REVIEW")


_RUBRIC = (
    "- Is the SUMMARY QUOTE present verbatim somewhere in the cited pages?\n"
    "- Does the attribution actually link the named ENTITY to the stated stance?\n"
    "  * For attribution_mode=direct_quote: the entity is the speaker of the quote.\n"
    "  * For attribution_mode=paraphrased: the cited narrator sentence names the entity\n"
    "    and states their position.\n"
    "  * For attribution_mode=sectional: the entity name appears in a list/table that\n"
    "    sits under the cited stance heading, and the heading establishes the stance\n"
    "    for everyone in that list.\n"
    "- Does the stance label match the evidence? "
    "(in_favor / opposed / conditional / neutral)\n"
    "- Is the ENTITY name well-formed? Private individuals should use last-name "
    "only or 'private commenter'; orgs/agencies/tribes should use their "
    "official name.\n"
    "- Is the ROLE description supported by the cited text?"
)

_SYSTEM = (
    "You are a Critic verifying ONE extracted (entity, stance) row from an "
    "Environmental Impact Statement.\n\n"
    f"RUBRIC — answer each check yes/no/n/a, then give a verdict:\n{_RUBRIC}\n\n"
    "Respond ONLY with JSON:\n"
    "{\n"
    '  "rubric_results": [{"check": "<short>", "result": "yes|no|n/a", "note": "<short>"}],\n'
    '  "verdict": "PASS|PASS_WITH_NOTE|RE_EXTRACT|HUMAN_REVIEW",\n'
    '  "model_confidence": "low|medium|high",\n'
    '  "notes": "<2-3 sentences>"\n'
    "}\n\n"
    "Verdict guide:\n"
    "  PASS: all checks yes, quote verbatim, stance and entity unambiguous.\n"
    "  PASS_WITH_NOTE: minor issues (e.g. stance is correct but quote includes "
    "extra surrounding text); still trustworthy.\n"
    "  RE_EXTRACT: extraction is wrong in a fixable way (e.g. wrong stance "
    "label for the quote, wrong speaker attribution).\n"
    "  HUMAN_REVIEW: ambiguous, anonymous, or sensitive — needs a human."
)


def _resolve_cited_text(text: str, source_pages: list[str]) -> str:
    """Pull the actual cited pages' text from the doc — same approach as segment_a's critic."""
    pieces: list[str] = []
    for span in source_pages or []:
        if not isinstance(span, str):
            continue
        try:
            if "-" in span:
                a, b = span.split("-", 1)
                sp = int(a.strip())
                ep = int(b.strip())
            else:
                sp = ep = int(span.strip())
        except ValueError:
            continue
        s, e = page_range_chars(sp, ep, len(text))
        pieces.append(text[s:e])
    return "\n\n[...]\n\n".join(pieces)[:60_000]


def _ask_critic(row: dict, cited_text: str) -> tuple[dict, dict | None]:
    extracted = {
        "entity": row.get("entity"),
        "kind": row.get("kind"),
        "role": row.get("role"),
        "stance": row.get("stance"),
        "attribution_mode": row.get("attribution_mode"),
        "summary_quote": row.get("summary_quote"),
        "summary_quote_verified": row.get("summary_quote_verified"),
        "evidence_pages": row.get("evidence_pages"),
    }
    user = (
        f"EXTRACTED ROW:\n{json.dumps(extracted, ensure_ascii=False, indent=2)}\n\n"
        f"CITED PAGES (text pulled in):\n{cited_text or '(no cited-page text available)'}"
    )
    try:
        out, usage = call_json_with_usage(
            MODEL_SONNET, _SYSTEM, user, max_tokens=1200,
        )
    except Exception as e:
        log.warning(f"critic call failed for entity {row.get('entity')!r}: {e}")
        return {
            "verdict": "HUMAN_REVIEW",
            "model_confidence": "low",
            "notes": f"Critic call failed: {e}",
            "rubric_results": [],
        }, None
    verdict = out.get("verdict")
    if verdict not in VERDICTS:
        out["verdict"] = "HUMAN_REVIEW"
        out["notes"] = (out.get("notes") or "") + " [Unknown verdict — forcing HUMAN_REVIEW.]"
    return out, usage


def _apply_overrides(row: dict, critic_result: dict) -> dict:
    """Force HUMAN_REVIEW for unverified quotes and for private-individual rows."""
    notes = critic_result.get("notes") or ""
    if not row.get("summary_quote_verified"):
        if critic_result["verdict"] != "HUMAN_REVIEW":
            critic_result["verdict"] = "HUMAN_REVIEW"
            critic_result["notes"] = notes + " [Forced HUMAN_REVIEW: quote not verbatim.]"
            notes = critic_result["notes"]
    if row.get("kind") == "individual":
        if critic_result["verdict"] != "HUMAN_REVIEW":
            critic_result["verdict"] = "HUMAN_REVIEW"
            critic_result["notes"] = notes + (
                " [Forced HUMAN_REVIEW: stance attribution for a private individual "
                "always goes to a human (v2 policy override).]"
            )
    return critic_result


def critique_row(row: dict, full_text: str) -> dict:
    """Returns the row with a `critic` block added (and `_critic_usage` for aggregation)."""
    cited_text = _resolve_cited_text(full_text, row.get("evidence_pages") or [])
    crit, usage = _ask_critic(row, cited_text)
    crit = _apply_overrides(row, crit)
    out = dict(row)
    out["critic"] = crit
    if usage is not None:
        out["_critic_usage"] = usage  # stripped before final write; aggregated by run.py
    return out


def critique_all(rows: list[dict], full_text: str, parallel: int = settings.CRITIC_PARALLEL) -> list[dict]:
    """Run the per-row critic in parallel; preserves input order via sequence."""
    if not rows:
        return []
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(critique_row, r, full_text): r for r in rows}
        for fut in as_completed(futures):
            try:
                out.append(fut.result())
            except Exception as e:
                r = futures[fut]
                log.exception(f"critic crashed on entity {r.get('entity')!r}: {e}")
                fallback = dict(r)
                fallback["critic"] = {
                    "verdict": "HUMAN_REVIEW",
                    "model_confidence": "low",
                    "notes": f"Critic crashed: {e}",
                    "rubric_results": [],
                }
                out.append(fallback)
    out.sort(key=lambda r: r.get("sequence", 10**9))
    return out
