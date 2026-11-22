"""Score the verifier against HAND labels.

1. Run run_batch.py with --verify; open runs/<name>/claims.csv.
2. Both team members fill `human_label` with one of: supported | contradicted | not_verifiable
   (judged against the retrieved evidence text, NOT your own medical knowledge). Use one agreed file after adjudication.
3. python eval_verifier.py runs/<name>/claims.csv

Verifier labels are collapsed to three classes: supported (supported, partially_supported),
problem (contradicted, unsupported_specific), not_verifiable. A human 'unsupported' counts as problem.
"""
import sys

import pandas as pd

MAP_V = {"supported": "supported", "partially_supported": "supported", "contradicted": "problem",
         "unsupported_specific": "problem", "not_verifiable": "not_verifiable"}
MAP_H = {"supported": "supported", "contradicted": "problem", "unsupported": "problem", "not_verifiable": "not_verifiable"}


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


df = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
df = df[df.human_label.str.strip() != ""].copy()
if df.empty:
    sys.exit("No hand labels found: fill the human_label column first.")
df["h"] = df.human_label.str.strip().str.lower().map(MAP_H)
if df.h.isna().any():
    sys.exit(f"Unknown human_label values: {sorted(set(df[df.h.isna()].human_label))}")
df["v"] = df.verifier_label.map(MAP_V)
print(f"{len(df)} hand-labelled claims\n")
print(pd.crosstab(df.h, df.v, rownames=["human"], colnames=["verifier"]).to_string(), "\n")
tp = ((df.v == "problem") & (df.h == "problem")).sum()
fp = ((df.v == "problem") & (df.h != "problem")).sum()
fn = ((df.v != "problem") & (df.h == "problem")).sum()
p, r = tp / (tp + fp) if tp + fp else float("nan"), tp / (tp + fn) if tp + fn else float("nan")
f1 = 2 * p * r / (p + r) if tp else float("nan")
print(f"Problem detection  precision {p:.3f} | recall {r:.3f} | F1 {f1:.3f}  (tp={tp}, fp={fp}, fn={fn})")
print(f"3-class agreement  {(df.v == df.h).mean():.1%} | Cohen's kappa {kappa(list(df.h), list(df.v)):.3f}")
print("\nBy verifier layer (agreement with human):")
print(df.groupby("layer").apply(lambda g: pd.Series({"n": len(g), "agree": (g.v == g.h).mean()}), include_groups=False).round(3).to_string())
