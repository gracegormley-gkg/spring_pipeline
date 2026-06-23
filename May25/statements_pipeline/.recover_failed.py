"""Recovery helper for the new schema.

Identifies docs whose complaint files contain `find_statement_error` (i.e. the
find_statement LLM call failed for that complaint), deletes those doc dirs
(keeps raw_extract caches), and prints the re-run commands.

Usage:
    python .recover_failed.py            # dry-run: just report
    python .recover_failed.py --apply    # actually delete output dirs
"""
import json
import glob
import shutil
import sys
from pathlib import Path
from collections import Counter

apply = "--apply" in sys.argv

per_doc_total = Counter()
per_doc_fail = Counter()
for path in glob.glob("output/people/*/complaints/*.json"):
    with open(path) as f:
        rec = json.load(f)
    d = rec.get("doc_id", "?")
    per_doc_total[d] += 1
    if rec.get("find_statement_error"):
        per_doc_fail[d] += 1

bad = sorted(d for d, n in per_doc_fail.items() if n > 0)

print(f"{'doc_id':<32} {'total':>6} {'failed':>6}  status")
for d in sorted(per_doc_total):
    t, f = per_doc_total[d], per_doc_fail[d]
    status = "CLEAN" if f == 0 else f"{f}/{t} failed"
    print(f"  {d:<32} {t:>6} {f:>6}  {status}")
print()

if not bad:
    print("No failed docs — nothing to do.")
    sys.exit(0)

print(f"Docs with failures ({len(bad)}):")
for d in bad:
    p = Path(f"output/people/{d}")
    if apply and p.exists():
        shutil.rmtree(p)
        print(f"  rm -rf output/people/{d}  (done)")
    else:
        print(f"  would rm -rf output/people/{d}")

print()
print("Re-run commands (run after creds are confirmed in your shell):")
print(f"  echo \"creds: ${{AWS_BEARER_TOKEN_BEDROCK:+set}}\"")
for d in bad:
    print(f"  python run.py process --doc {d}")

if not apply:
    print()
    print("Add --apply to actually delete the output dirs above.")
