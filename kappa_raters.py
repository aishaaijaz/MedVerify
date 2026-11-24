"""Inter-rater agreement between the two team members on the claim labels.

1. Each member fills ONE column of claims.csv independently (rater1 / rater2) with:
   supported | contradicted | not_verifiable   (judged against the retrieved evidence text, NOT your medical knowledge).
2. python kappa_raters.py runs/C/claims.csv
   Prints % agreement, Cohen's kappa, the confusion matrix and the disagreements, and writes
   claims_adjudicated.csv where human_label = the agreed label (blank where you disagree: discuss and fill those in).
3. Then: python eval_verifier.py runs/C/claims_adjudicated.csv
"""
import sys
from pathlib import Path

import pandas as pd

OK = {"supported", "contradicted", "not_verifiable"}
p = Path(sys.argv[1])
df = pd.read_csv(p, dtype=str, keep_default_na=False)
for c in ("rater1", "rater2"):
    df[c] = df[c].str.strip().str.lower()
d = df[(df.rater1 != "") & (df.rater2 != "")].copy()
bad = d[~d.rater1.isin(OK) | ~d.rater2.isin(OK)]
if len(d) == 0:
    sys.exit("Fill the rater1 and rater2 columns first.")
if len(bad):
    sys.exit(f"Labels must be one of {sorted(OK)}. Problem rows: {list(bad.index + 2)} (CSV line numbers)")

n = len(d)
labels = sorted(OK)
po = (d.rater1 == d.rater2).mean()
pe = sum((d.rater1 == l).mean() * (d.rater2 == l).mean() for l in labels)
kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
print(f"{n} claims labelled by both raters")
print(f"Agreement {po:.1%} | Cohen's kappa {kappa:.3f}  (0.61-0.80 substantial, >0.80 almost perfect)\n")
print(pd.crosstab(d.rater1, d.rater2, rownames=["rater1"], colnames=["rater2"]).to_string(), "\n")
dis = d[d.rater1 != d.rater2]
print(f"{len(dis)} disagreements to discuss:")
print(dis[["pair_id", "claim_id", "claim", "rater1", "rater2"]].to_string(index=False, max_colwidth=70))
df["human_label"] = df.apply(lambda r: r.rater1 if r.rater1 and r.rater1 == r.rater2 else r.human_label, axis=1)
out = p.with_name("claims_adjudicated.csv")
df.to_csv(out, index=False)
print(f"\nSaved {out}. Fill human_label for the disagreements, then run eval_verifier.py on it.")
