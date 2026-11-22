"""Step 2 - Clean and structure the 9 MecDDI files.

Inputs  : the MecDDI folder (9 CSVs)  (drug info, pair list, 7 mechanism-category files)
Outputs : data/processed/mecddi/
            mecddi_drugs.csv                  one row per drug (clean metadata + normalized names/synonyms)
            mecddi_pair_mechanisms_long.csv   one row per (pair, mechanism category)
            mecddi_pairs.csv                  one row per pair; categories + multi-hot columns
            mecddi_preprocessing_report.txt   counts and PASS/FAIL checks

What MecDDI contains: drug pairs labelled with one or more of 7 mechanism CATEGORIES.
It has NO mechanism description text and NO severity, so severity comes from DDInter (step 2).
"""
import pandas as pd

from config import OUT, find_file, find_mecddi_dir, make_key, norm_text

REPORT = []


def log(msg=""):
    print(msg)
    REPORT.append(str(msg))


def check(name, ok, detail=""):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


# source label -> (short code, filename keywords)
CATS = {
    "Affected gastrointestinal absorption": ("absorption", ("gastrointestinal", "absorption")),
    "Affected cellular transport": ("transport", ("cellular", "transport")),
    "Affected organization distribution": ("distribution", ("organization", "distribution")),
    "Affected intra/extra-hepatic metabolism": ("metabolism", ("hepatic", "metabolism")),
    "Affected excretion pathways": ("excretion", ("excretion",)),
    "Pharmacodynamic additive effects": ("pd_additive", ("additive",)),
    "Pharmacodynamic antagonistic effects": ("pd_antagonistic", ("antagonistic",)),
}
CODES = [v[0] for v in CATS.values()]


def read(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda s: s.str.strip())


OUT.mkdir(parents=True, exist_ok=True)
MEC = find_mecddi_dir()
log(f"MecDDI folder: {MEC}")

# ---------------------------------------------------------------- 1. drug table
drug = read(find_file(MEC, "drug", "general", "information")).rename(columns={
    "Drug_ID": "drug_id", "Drug_Name": "drug_name", "PubChem_CID": "pubchem_cid", "Synonymous": "synonyms",
    "Therapeutic_Class": "therapeutic_class", "Drug_Type": "drug_type", "InChI": "inchi", "InChIKey": "inchikey",
    "Canonical_SMILES": "smiles", "CAS_Number": "cas_number", "Formula": "formula", "ChEBI": "chebi",
    "Disease_Class": "disease_class"})
log(f"Drug table: {len(drug)} rows, {drug.shape[1]} columns")

# '.' is used as a placeholder for 'none' in the source (e.g. Disease_Class); make it a real missing value.
for c in drug.columns.difference(["drug_id", "drug_name"]):
    drug[c] = drug[c].replace({".": pd.NA, "": pd.NA})

drug["drug_name_norm"] = drug.drug_name.map(norm_text)
# Synonyms are ';'-separated (brand names, INN/USAN forms, catalogue IDs). Keep a de-duplicated normalized list.
drug["synonyms_norm"] = drug.synonyms.map(
    lambda s: ";".join(dict.fromkeys(t for t in (norm_text(x) for x in str(s).split(";")) if t)) if pd.notna(s) else "")

check("Drug_ID unique", drug.drug_id.is_unique)
check("Normalized drug name unique", drug.drug_name_norm.is_unique)
check("No missing Drug_ID / Drug_Name", drug[["drug_id", "drug_name"]].notna().all().all())
log("Missing values per column (after '.' -> missing): " + str({c: int(v) for c, v in drug.isna().sum().items() if v}))
name_by_id = drug.set_index("drug_id").drug_name
ids = set(drug.drug_id)


def canon(df, label):
    """Validate IDs/names against the drug table and put each pair in canonical order (A id < B id)."""
    for side in "AB":
        check(f"{label}: {side}_Drug_ID all in drug table", df[f"{side}_Drug_ID"].isin(ids).all())
        bad = (df[f"{side}_Drug_Name"] != df[f"{side}_Drug_ID"].map(name_by_id)).sum()
        check(f"{label}: {side} name matches drug table for its ID", bad == 0, f"{bad} mismatches")
    check(f"{label}: no self-pairs", (df.A_Drug_ID == df.B_Drug_ID).sum() == 0)
    swap = df.A_Drug_ID > df.B_Drug_ID
    out = pd.DataFrame({
        "drug_a_id": df.A_Drug_ID.where(~swap, df.B_Drug_ID),
        "drug_b_id": df.B_Drug_ID.where(~swap, df.A_Drug_ID)})
    out["drug_a_name"] = out.drug_a_id.map(name_by_id)
    out["drug_b_name"] = out.drug_b_id.map(name_by_id)
    out["pair_key"] = make_key(df.A_Drug_ID, df.B_Drug_ID)
    return out


# ---------------------------------------------------------------- 2. pair list
pair_raw = read(find_file(MEC, "drug", "interaction", "pair"))
log(f"\nPair file: {len(pair_raw)} rows")
pairs = canon(pair_raw, "pairs")
n_dup = pairs.duplicated("pair_key").sum()
pairs = pairs.drop_duplicates("pair_key")
log(f"Duplicate pairs (A,B)/(B,A) removed: {n_dup}; unique pairs: {len(pairs)}")

# ---------------------------------------------------------------- 3. mechanism files
frames = []
for label, (code, kw) in CATS.items():
    f = find_file(MEC, *kw)
    d = read(f)
    check(f"{code}: single expected category label", set(d.Mechanism_Category) == {label}, f"file={f.name}, rows={len(d)}")
    c = canon(d, code)
    c["category_code"], c["category_label"], c["source_file"] = code, label, f.name
    frames.append(c)
mech = pd.concat(frames, ignore_index=True)
log(f"\nMechanism rows (all 7 files): {len(mech)}")

long = (mech.groupby(["pair_key", "drug_a_id", "drug_b_id", "drug_a_name", "drug_b_name",
                      "category_code", "category_label", "source_file"], as_index=False)
        .size().rename(columns={"size": "n_source_rows"}))
log(f"Duplicate (pair, category) rows collapsed: {len(mech) - len(long)}; unique (pair, category): {len(long)}")

# ---------------------------------------------------------------- 4. consistency between files
check("Every mechanism pair is in the pair file", set(long.pair_key) <= set(pairs.pair_key))
check("Every pair has at least one mechanism category", set(pairs.pair_key) <= set(long.pair_key))

# ---------------------------------------------------------------- 5. one row per pair (multi-label)
hot = pd.crosstab(long.pair_key, long.category_code).clip(upper=1)
for c in CODES:
    if c not in hot:
        hot[c] = 0
hot = hot[CODES].add_prefix("cat_")
cats = long.groupby("pair_key").category_code.agg(lambda s: ";".join(sorted(s))).rename("categories")
wide = pairs.set_index("pair_key").join(cats).join(hot).reset_index()
wide["n_categories"] = wide[[f"cat_{c}" for c in CODES]].sum(axis=1)

log("\nPairs per mechanism category (a pair can have several):")
log(long.category_code.value_counts().to_string())
log("\nCategories per pair:")
log(wide.n_categories.value_counts().sort_index().to_string())
log(f"\nDrugs appearing in pairs: {len(set(wide.drug_a_id) | set(wide.drug_b_id))} of {len(drug)}")

# ---------------------------------------------------------------- 6. save
drug.to_csv(OUT / "mecddi_drugs.csv", index=False)
long.to_csv(OUT / "mecddi_pair_mechanisms_long.csv", index=False)
wide.to_csv(OUT / "mecddi_pairs.csv", index=False)
(OUT / "mecddi_preprocessing_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
log(f"\nSaved to {OUT}")
if any(line.startswith("[FAIL]") for line in REPORT):
    raise SystemExit("One or more checks FAILED - see report.")
