import json, sys
from pathlib import Path
p = Path("output/run_summary.json")
if not p.exists():
    sys.exit("no run_summary.json yet")
data = json.load(open(p))
runs = data.get("runs", [])
total = 0.0
print(f"{'doc_id':<30}  {'people':>6}  {'para':>5}  {'review':>6}  {'cost':>8}")
for r in runs:
    if "error" in r:
        print(f"{r['doc_id']:<30}  ERROR: {r['error']}")
        continue
    cost = (((r.get("usage") or {}).get("total") or {}).get("total") or {}).get("cost_usd", 0)
    rc = (r.get("review_counts") or {}).get("needs_review", 0)
    n_para = r.get("n_paraphrases", 0)
    print(f"{r['doc_id']:<30}  {r.get('n_people', 0):>6}  {n_para:>5}  {rc:>6}  ${cost:>7.4f}")
    total += cost
print(f"{'TOTAL':<30}                              ${total:>7.4f}")
