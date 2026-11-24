"""Collect result files from runs/ into results/ (and results.zip) for committing.

  python collect_results.py runs/A_qwen2_5_3b runs/Bpair_qwen2_5_3b ...

Copies only aggregate outputs: metrics_report.txt, summary.txt, confusion_severity.png, confusion_category.png, plus
runs/comparison.csv. It deliberately does NOT copy results.csv / claims.csv (per-pair data derived from the licensed databases).
Run metrics.py first so the files exist.
"""
import shutil
import sys
import zipfile
from pathlib import Path

out = Path("results")
out.mkdir(exist_ok=True)
for run in map(Path, sys.argv[1:]):
    dest = out / run.name
    dest.mkdir(exist_ok=True)
    for name in ("metrics_report.txt", "summary.txt", "confusion_severity.png", "confusion_category.png"):
        if (run / name).exists():
            shutil.copy(run / name, dest / name)
cmp = Path("runs") / "comparison.csv"
if cmp.exists():
    shutil.copy(cmp, out / "comparison.csv")
with zipfile.ZipFile("results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(out.rglob("*")):
        if f.is_file():
            z.write(f)
print(f"Collected {len(list(out.rglob('*')))} items into {out}/ and results.zip")
