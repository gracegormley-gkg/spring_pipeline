"""
Shared prompt fragments for M2 (Segment A / Segment B).

Loads the prompt clauses that M-Cal and M2 must agree on from
`May25/mcal/templates/`. Read from disk by relative path rather than imported,
so segment_a keeps no import dependency on the mcal package -- the dependency
runs the other way (mcal/settings.py appends segment_a to sys.path).

Why the clauses live under mcal/templates/ rather than here: they are consumed
by BOTH sides. `m2.py` uses them to generate, and `mcal/critic_prompt.py` uses
them to build Critic rubric Q6, which asks whether the generator obeyed them.
One copy means the rubric can never drift from the instruction it grades.

PROMPT_VERSION is the contract with `mcal/build.py`. MCAL_PLAN 3.7's step-0
precheck refuses to calibrate unless `output/m2/_prompt_version.txt` matches,
because tau must be fitted to the same prose Segment B will ship. Bump it
whenever a template changes, then re-run:

    python run.py process --force --doc <doc_id>
"""

from __future__ import annotations

from pathlib import Path

# MCAL_PLAN 3.7 / 3.14: the marker value that build.py requires.
PROMPT_VERSION = "v1_plain_language"

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "mcal" / "templates"

PLAIN_LANGUAGE_TEMPLATE = _TEMPLATES_DIR / "m2_plain_language.md"
SUMMARY_OF_INTEREST_TEMPLATE = _TEMPLATES_DIR / "m2_summary_of_interest.md"

# Templates carry a human-facing header explaining provenance, separated from
# the model-facing body by a horizontal rule. Only the body is sent to the LLM.
_BODY_SEPARATOR = "\n---\n"


def _load_body(path: Path) -> str:
    """Read a template and return only the model-facing body."""
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template missing: {path}\n"
            "M2 prompts are assembled from mcal/templates/. If you moved or "
            "renamed the templates directory, update segment_a/prompts.py."
        )
    text = path.read_text(encoding="utf-8")
    _, sep, body = text.partition(_BODY_SEPARATOR)
    return (body if sep else text).strip()


def plain_language_clause() -> str:
    """
    The MCAL_PLAN 3.14 plain-language + concreteness clause.

    Appended to every `summary.*` map and reduce prompt.
    """
    return _load_body(PLAIN_LANGUAGE_TEMPLATE)


def summary_of_interest_prompt() -> str:
    """The MCAL_PLAN 3.15 salience prompt for the second summary."""
    return _load_body(SUMMARY_OF_INTEREST_TEMPLATE)


def write_version_marker(m2_dir: Path) -> Path:
    """
    Stamp `_prompt_version.txt` after a completed M2 rerun.

    Called by run.py once every targeted doc has been reprocessed -- NOT per
    doc, since a partial rerun must not look complete to build.py.
    """
    m2_dir.mkdir(parents=True, exist_ok=True)
    marker = m2_dir / "_prompt_version.txt"
    marker.write_text(PROMPT_VERSION + "\n", encoding="utf-8")
    return marker


def read_version_marker(m2_dir: Path) -> str | None:
    marker = m2_dir / "_prompt_version.txt"
    if not marker.exists():
        return None
    return marker.read_text(encoding="utf-8").strip() or None
