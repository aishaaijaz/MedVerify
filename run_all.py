"""Run the whole pipeline in order: python run_all.py"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for step in ["01_ddinter_preprocess.py", "02_mecddi_preprocess.py", "03_link_ddinter.py", "04_eda.py"]:
    print(f"\n{'=' * 20} {step} {'=' * 20}")
    r = subprocess.run([sys.executable, str(here / step)], cwd=here)
    if r.returncode:
        sys.exit(f"{step} failed (exit {r.returncode}) - fix it before continuing.")
