"""Re-score existing runs with the corrected format-failure logic (no model reruns needed).

  python rescore.py runs/A_llama3_2_3b runs/Bpair_llama3_2_3b ...      (or: python rescore.py runs/*)

Reads each results.csv, recomputes parsed / abstained / cat_exact / cat_overlap / sev_correct from the saved `answer` text,
keeps a backup as results.csv.bak, and prints format failures per run. Then run metrics.py as usual.
Answers that are empty, echo the prompt, or lack the CATEGORIES:/SEVERITY: lines are now FORMAT FAILURES, not abstentions.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

from retrieval import score_answer

for run in map(Path, sys.argv[1:]):
    f = run / "results.csv"
    if not f.exists():
        continue
    if not (run / "results.csv.bak").exists():
        shutil.copy(f, run / "results.csv.bak")
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    for i, r in df.iterrows():
        truth = None if r.true_severity == "none" else {"categories": [c for c in r.true_categories.split(";") if c], "severity": r.true_severity}
        for k, v in score_answer(r.answer, truth).items():
            df.at[i, k] = str(v)
    df.to_csv(f, index=False)
    m = df[df.group == "main"]
    n = df[df.group == "no_record"]
    print(f"{run.name:40s} main n={len(m)} | format failures {(m.parsed != 'True').sum():2d} | abstained {(m.abstained == 'True').sum():2d} | "
          f"cat_exact {(m.cat_exact == 'True').mean():.0%} | sev_acc {(m.sev_correct == 'True').mean():.0%} | "
          f"no-record correct abstention {(n.cat_exact == 'True').mean() if len(n) else float('nan'):.0%}")
