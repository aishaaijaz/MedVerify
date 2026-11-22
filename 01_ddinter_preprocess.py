"""Step 1 - Clean and consolidate the 8 raw DDInter files.

Inputs  : data/raw/ddinter_downloads_code_*.csv  (columns: DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level)
Outputs : data/processed/ddinter/
            ddinter_lookup.csv             one row per unique interaction record, with provenance
            ddinter_drugs.csv              one row per drug (id, name, normalized name, number of pairs)
            ddinter_preprocessing_report.txt

Rules: only EXACT duplicate records are consolidated (the same pair listed in several category files).
Records are never merged merely for sharing a pair key; if one pair ever carried two different severities
both rows would be kept and flagged. 'Unknown' is a real severity class, not a missing value.
"""
import re

import pandas as pd

from config import DD_OUT, RAW_DDINTER, SEVERITIES, make_key, norm_text

REPORT = []


def log(msg=""):
    print(msg)
    REPORT.append(str(msg))


def check(name, ok, detail=""):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


DD_OUT.mkdir(parents=True, exist_ok=True)
files = sorted(p for p in RAW_DDINTER.glob("*.csv") if re.search(r"ddinter.*code", p.name, re.I))
if not files:
    raise SystemExit(f"No ddinter_downloads_code_*.csv files found in {RAW_DDINTER}")
log(f"DDInter files found: {[f.name for f in files]}")

REQUIRED = ["DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level"]
frames = []
for f in files:
    d = pd.read_csv(f, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    d.columns = [c.strip() for c in d.columns]
    check(f"{f.name}: required columns present", all(c in d.columns for c in REQUIRED))
    d = d[REQUIRED].apply(lambda s: s.str.strip())
    m = re.search(r"code_([A-Za-z])", f.name)
    d["source_file"] = f.name
    d["source_category"] = m.group(1).upper() if m else f.stem  # ATC letter taken from the file name
    frames.append(d)
raw = pd.concat(frames, ignore_index=True)
log(f"\nCombined rows: {len(raw)}")

# ---- normalize
raw["severity"] = raw.Level.map(norm_text)
raw["drug_a_name_norm"] = raw.Drug_A.map(norm_text)
raw["drug_b_name_norm"] = raw.Drug_B.map(norm_text)
check("No blank values in key fields", (raw[REQUIRED] == "").sum().sum() == 0)
check("Severity values all in expected set", raw.severity.isin(SEVERITIES).all(), f"found {sorted(raw.severity.unique())}")
check("No self-pairs (by ID)", (raw.DDInterID_A == raw.DDInterID_B).sum() == 0)

# ---- ID <-> name consistency (internal check only)
idname = pd.concat([raw[["DDInterID_A", "drug_a_name_norm"]].set_axis(["id", "name"], axis=1),
                    raw[["DDInterID_B", "drug_b_name_norm"]].set_axis(["id", "name"], axis=1)]).drop_duplicates()
check("Each DDInter ID has exactly one normalized name", idname.groupby("id").name.nunique().max() == 1)
check("Each normalized name has exactly one DDInter ID", idname.groupby("name").id.nunique().max() == 1)
name_by_id = idname.set_index("id").name.to_dict()
raw_name_by_id = pd.concat([raw[["DDInterID_A", "Drug_A"]].set_axis(["id", "n"], axis=1),
                            raw[["DDInterID_B", "Drug_B"]].set_axis(["id", "n"], axis=1)]).drop_duplicates("id").set_index("id").n.to_dict()

# ---- canonical pair key and orientation (by DDInter ID order)
raw["pair_key"] = make_key(raw.DDInterID_A, raw.DDInterID_B)
swap = raw.DDInterID_A > raw.DDInterID_B
raw["drug_a_id"] = raw.DDInterID_A.where(~swap, raw.DDInterID_B)
raw["drug_b_id"] = raw.DDInterID_B.where(~swap, raw.DDInterID_A)

# ---- consolidate exact duplicates, keep provenance
agg = (raw.groupby(["pair_key", "drug_a_id", "drug_b_id", "severity"], as_index=False)
       .agg(source_files=("source_file", lambda s: ";".join(sorted(set(s)))),
            source_categories=("source_category", lambda s: ";".join(sorted(set(s)))),
            source_record_count=("source_file", "size")))
agg["drug_a_name"], agg["drug_b_name"] = agg.drug_a_id.map(raw_name_by_id), agg.drug_b_id.map(raw_name_by_id)
agg["drug_a_name_norm"], agg["drug_b_name_norm"] = agg.drug_a_id.map(name_by_id), agg.drug_b_id.map(name_by_id)
agg["severity_conflict"] = agg.duplicated("pair_key", keep=False)
log(f"Unique interaction records: {len(agg)}  (consolidated {len(raw) - len(agg)} additional occurrences)")
check("One severity per pair (no conflicts)", not agg.severity_conflict.any(), f"{int(agg.severity_conflict.sum())} rows in conflict")
check("Pair keys unique", agg.pair_key.is_unique)

cols = ["pair_key", "drug_a_id", "drug_a_name", "drug_a_name_norm", "drug_b_id", "drug_b_name", "drug_b_name_norm",
        "severity", "severity_conflict", "source_files", "source_categories", "source_record_count"]
agg = agg[cols]
log("\nSeverity distribution (unique records):")
log(pd.concat([agg.severity.value_counts(), agg.severity.value_counts(normalize=True).round(4)], axis=1).to_string())
log(f"\nDistinct drugs: {len(name_by_id)}")

deg = pd.concat([agg.drug_a_id, agg.drug_b_id]).value_counts()
drugs = pd.DataFrame({"drug_id": list(name_by_id), "drug_name": [raw_name_by_id[i] for i in name_by_id],
                      "drug_name_norm": list(name_by_id.values())})
drugs["n_pairs"] = drugs.drug_id.map(deg).fillna(0).astype(int)

agg.to_csv(DD_OUT / "ddinter_lookup.csv", index=False)
drugs.to_csv(DD_OUT / "ddinter_drugs.csv", index=False)
(DD_OUT / "ddinter_preprocessing_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
log(f"\nSaved to {DD_OUT}")
if any(line.startswith("[FAIL]") for line in REPORT):
    raise SystemExit("One or more checks FAILED - see report.")
