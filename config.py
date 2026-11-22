"""Shared paths, settings and helpers for the MedVerify preprocessing pipeline.

Unzip the contents of this folder into the MedVerify project root. Expected layout:
    <ROOT>/data/raw/ddinter_downloads_code_A.csv ... _V.csv   (8 DDInter files)
    <ROOT>/MecDDI/*.csv                                       (9 MecDDI files; also found at data/MecDDI or data/raw/MecDDI)
No absolute paths are used anywhere.
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DDINTER = ROOT / "data" / "raw"
MANUAL_MAP = ROOT / "data" / "manual_name_map.csv"  # optional: ddinter_name_norm,mecddi_drug_id
PROC = ROOT / "data" / "processed"
DD_OUT = PROC / "ddinter"
OUT = PROC / "mecddi"
LINK_OUT = PROC / "linked"
EDA_OUT = PROC / "eda"

# Allow matching DDInter names to MecDDI synonyms (brand names etc.) when no primary-name match exists.
USE_SYNONYMS = True

SEVERITIES = ["major", "moderate", "minor", "unknown"]


def find_mecddi_dir():
    """Locate the folder holding the MecDDI CSVs (folder name 'MecDDI', any capitalization)."""
    for base in (ROOT, ROOT / "data", RAW_DDINTER):
        if base.exists():
            for p in base.iterdir():
                if p.is_dir() and p.name.lower() == "mecddi" and any(p.glob("*.csv")):
                    return p
    raise FileNotFoundError("MecDDI folder not found. Put the 9 MecDDI CSVs in <ROOT>/MecDDI/ (or data/MecDDI, data/raw/MecDDI).")


def norm_text(x):
    """Unicode-normalize, lowercase, strip quotes/whitespace, collapse internal spaces."""
    if x is None or pd.isna(x):
        return ""
    s = unicodedata.normalize("NFKC", str(x)).lower().strip().strip("\"'")
    return re.sub(r"\s+", " ", s)


def find_file(folder, *keywords):
    """Find exactly one CSV in `folder` whose name contains all keywords (spaces/underscores/hyphens ignored)."""
    hits = []
    for p in Path(folder).glob("*.csv"):
        name = re.sub(r"[^a-z0-9]+", " ", p.stem.lower())
        if all(k in name for k in keywords):
            hits.append(p)
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one CSV in {folder} matching {keywords}, found {[h.name for h in hits]}")
    return hits[0]


def make_key(a, b):
    """Order-independent pair key from two Series: 'min|max'."""
    lo = a.where(a < b, b)
    hi = b.where(a < b, a)
    return lo + "|" + hi
