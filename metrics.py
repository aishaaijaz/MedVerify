"""Classical evaluation metrics for MedVerify runs (accuracy, precision/recall/F1, confusion matrices, bootstrap CIs).

  python metrics.py runs/A_M runs/Bpair_M runs/C_M [runs/other ...]      (each folder holds results.csv from run_batch.py)

Per run (main pairs only) it writes metrics_report.txt, confusion_severity.png and confusion_category.png into the run folder.
Across runs it writes runs/comparison.csv (one row per run) so the three conditions / models sit side by side.
Severity: 3 classes (major/moderate/minor); any other prediction (unknown, abstain, unparsed) is counted as 'other' = wrong.
Category: multi-label. Reported as exact-match accuracy, per-category P/R/F1, micro/macro-F1, and a confusion matrix over the
pairs that have exactly ONE true category (predicted column = the single predicted category, or 'multi/none').
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEVS = ["major", "moderate", "minor"]
CATS = ["absorption", "transport", "distribution", "metabolism", "excretion", "pd_additive", "pd_antagonistic"]
RNG = np.random.default_rng(42)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def sev_metrics(t, p):
    t, p = list(t), [x if x in SEVS else "other" for x in p]
    rows, f1s = {}, []
    for c in SEVS:
        tp = sum(a == c and b == c for a, b in zip(t, p))
        fp = sum(a != c and b == c for a, b in zip(t, p))
        fn = sum(a == c and b != c for a, b in zip(t, p))
        rows[c] = prf(tp, fp, fn) + (sum(a == c for a in t),)
        f1s.append(rows[c][2])
    cm = pd.DataFrame([[sum(a == r and b == c for a, b in zip(t, p)) for c in SEVS + ["other"]] for r in SEVS],
                      index=SEVS, columns=SEVS + ["other"])
    return {"acc": float(np.mean([a == b for a, b in zip(t, p)])), "macro_f1": float(np.mean(f1s)), "per_class": rows, "cm": cm}


def cat_sets(s):
    return {x for x in str(s).split(";") if x in CATS}


def cat_metrics(df):
    T, P = df.true_categories.map(cat_sets), df.pred_categories.map(cat_sets)
    rows, f1s, TP, FP, FN = {}, [], 0, 0, 0
    for c in CATS:
        tp = sum(c in a and c in b for a, b in zip(T, P))
        fp = sum(c not in a and c in b for a, b in zip(T, P))
        fn = sum(c in a and c not in b for a, b in zip(T, P))
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        rows[c] = prf(tp, fp, fn) + (sum(c in a for a in T),)
        if rows[c][3]:
            f1s.append(rows[c][2])
    single = [(next(iter(a)), next(iter(b)) if len(b) == 1 else "multi/none") for a, b in zip(T, P) if len(a) == 1]
    cols = CATS + ["multi/none"]
    cm = pd.DataFrame([[sum(x == r and y == c for x, y in single) for c in cols] for r in CATS], index=CATS, columns=cols)
    return {"exact": float(np.mean([a == b for a, b in zip(T, P)])), "micro_f1": prf(TP, FP, FN)[2],
            "macro_f1": float(np.mean(f1s)), "per_class": rows, "cm": cm.loc[cm.sum(axis=1) > 0]}


def boot_ci(df, fn, n=1000):
    vals = [fn(df.iloc[RNG.integers(0, len(df), len(df))]) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def heat(cm, title, path):
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * cm.shape[1], 1.0 + 0.7 * cm.shape[0]))
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=40, ha="right")
    ax.set_yticks(range(cm.shape[0]), cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center",
                    color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def table(d):
    return "\n".join(f"  {c:16s} P {v[0]:.2f}  R {v[1]:.2f}  F1 {v[2]:.2f}  (n={v[3]})" for c, v in d.items())


summary = []
for run in map(Path, sys.argv[1:]):
    df = pd.read_csv(run / "results.csv", dtype=str, keep_default_na=False)
    df = df[df.group == "main"].reset_index(drop=True) if "group" in df else df
    sm, cmx = sev_metrics(df.true_severity, df.pred_severity), cat_metrics(df)
    ci_s = boot_ci(df, lambda d: sev_metrics(d.true_severity, d.pred_severity)["acc"])
    ci_f = boot_ci(df, lambda d: sev_metrics(d.true_severity, d.pred_severity)["macro_f1"])
    ci_c = boot_ci(df, lambda d: cat_metrics(d)["exact"])
    txt = [f"RUN {run.name} | n={len(df)} main pairs | parseable labels: {(df.parsed == 'True').mean():.1%}", "",
           f"SEVERITY  accuracy {sm['acc']:.3f} (95% CI {ci_s[0]:.2f}-{ci_s[1]:.2f}) | macro-F1 {sm['macro_f1']:.3f} (CI {ci_f[0]:.2f}-{ci_f[1]:.2f})",
           table(sm["per_class"]), "  confusion matrix (rows = true):", sm["cm"].to_string(), "",
           f"CATEGORY exact-match {cmx['exact']:.3f} (CI {ci_c[0]:.2f}-{ci_c[1]:.2f}) | micro-F1 {cmx['micro_f1']:.3f} | macro-F1 {cmx['macro_f1']:.3f}",
           table(cmx["per_class"]), "  confusion matrix, pairs with exactly one true category:", cmx["cm"].to_string()]
    (run / "metrics_report.txt").write_text("\n".join(txt), encoding="utf-8")
    heat(sm["cm"], f"Severity - {run.name}", run / "confusion_severity.png")
    heat(cmx["cm"], f"Category (single-label pairs) - {run.name}", run / "confusion_category.png")
    print("\n".join(txt), "\n" + "=" * 60)
    summary.append({"run": run.name, "n": len(df), "sev_accuracy": round(sm["acc"], 3), "sev_macro_f1": round(sm["macro_f1"], 3),
                    "cat_exact": round(cmx["exact"], 3), "cat_micro_f1": round(cmx["micro_f1"], 3),
                    "cat_macro_f1": round(cmx["macro_f1"], 3), "parse_rate": round((df.parsed == "True").mean(), 3)})
out = Path(sys.argv[1]).parent / "comparison.csv"
pd.DataFrame(summary).to_csv(out, index=False)
print(pd.DataFrame(summary).to_string(index=False), f"\nSaved {out}")
