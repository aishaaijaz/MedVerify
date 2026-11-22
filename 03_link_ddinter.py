"""Step 2 - Link MecDDI pairs to DDInter severity.

MecDDI and DDInter use different identifiers, so drugs are matched by NORMALIZED NAME:
  1. optional manual map (data/manual_name_map.csv: ddinter_name_norm,mecddi_drug_id)
  2. exact match on MecDDI primary drug name
  3. (if USE_SYNONYMS) match on a MecDDI synonym, only when it points to exactly one drug
A name that points to several MecDDI drugs is AMBIGUOUS and is never linked automatically.

Inputs  : data/processed/mecddi/*.csv (step 1) and data/processed/ddinter/ddinter_lookup.csv (step 1)
Outputs : data/processed/linked/
            ddinter_drug_map.csv         every DDInter drug name and how it matched
            synonym_matches_for_review.csv  links made via synonyms - check these by eye
            unmatched_ddinter_drugs.csv  names to review / add to the manual map (most frequent first)
            linked_pairs.csv             pairs present in BOTH sources: categories + severity
            linking_report.txt
"""
import pandas as pd

from config import (DD_OUT, LINK_OUT, MANUAL_MAP, OUT, USE_SYNONYMS, make_key, norm_text)

REPORT = []


def log(msg=""):
    print(msg)
    REPORT.append(str(msg))


LINK_OUT.mkdir(parents=True, exist_ok=True)
pairs = pd.read_csv(OUT / "mecddi_pairs.csv")
drugs = pd.read_csv(OUT / "mecddi_drugs.csv", dtype=str, keep_default_na=False)
dd = pd.read_csv(DD_OUT / "ddinter_lookup.csv", dtype=str, keep_default_na=False)


cols = {"drug_a": "drug_a_name_norm", "drug_b": "drug_b_name_norm", "severity": "severity"}
log(f"DDInter records: {len(dd)}")
dd["a_norm"], dd["b_norm"] = dd.drug_a_name_norm, dd.drug_b_name_norm

# ---------------------------------------------------------------- build MecDDI name index
primary, syn = {}, {}
for did, n, s in zip(drugs.drug_id, drugs.drug_name_norm, drugs.synonyms_norm):
    primary.setdefault(n, set()).add(did)
    for t in s.split(";"):
        if t:
            syn.setdefault(t, set()).add(did)
manual = {}
if MANUAL_MAP.exists():
    m = pd.read_csv(MANUAL_MAP, dtype=str)
    manual = {norm_text(a): b.strip() for a, b in zip(m.ddinter_name_norm, m.mecddi_drug_id)}
    log(f"Manual name map loaded: {len(manual)} entries")


def resolve(name):
    if name in manual:
        return manual[name], "manual"
    if name in primary:
        ids = primary[name]
        return (next(iter(ids)), "exact_name") if len(ids) == 1 else (None, "ambiguous")
    if USE_SYNONYMS and name in syn:
        ids = syn[name]
        return (next(iter(ids)), "synonym") if len(ids) == 1 else (None, "ambiguous")
    return None, "unmatched"


freq = pd.concat([dd.a_norm, dd.b_norm]).value_counts()
dmap = pd.DataFrame({"ddinter_name_norm": freq.index, "n_ddinter_rows": freq.values})
res = dmap.ddinter_name_norm.map(resolve)
dmap["mecddi_drug_id"] = [r[0] for r in res]
dmap["match_type"] = [r[1] for r in res]
dmap["mecddi_drug_name"] = dmap.mecddi_drug_id.map(drugs.set_index("drug_id").drug_name)
dmap.to_csv(LINK_OUT / "ddinter_drug_map.csv", index=False)
# Synonym matches are the riskiest links (a brand/catalogue name can be shared) - review them by eye.
dmap[dmap.match_type == "synonym"].to_csv(LINK_OUT / "synonym_matches_for_review.csv", index=False)
dmap[dmap.mecddi_drug_id.isna()].to_csv(LINK_OUT / "unmatched_ddinter_drugs.csv", index=False)

log(f"\nDDInter distinct drug names: {len(dmap)}")
log(dmap.match_type.value_counts().to_string())
log(f"Matched DDInter drugs: {dmap.mecddi_drug_id.notna().sum()} ({dmap.mecddi_drug_id.notna().mean():.1%})")

# ---------------------------------------------------------------- link pairs
idmap = dict(zip(dmap.ddinter_name_norm, dmap.mecddi_drug_id))
dd["a_id"], dd["b_id"] = dd.a_norm.map(idmap), dd.b_norm.map(idmap)
both = dd[dd.a_id.notna() & dd.b_id.notna() & (dd.a_id != dd.b_id)].copy()
both["pair_key"] = make_key(both.a_id, both.b_id)
log(f"\nDDInter rows: {len(dd)}; with both drugs matched: {len(both)} ({len(both) / len(dd):.1%})")

g = both.groupby("pair_key").agg(
    n_ddinter_rows=("severity", "size"),
    severity_values=("severity", lambda s: ";".join(sorted(set(s)))))
# Several DDInter rows can map to one MecDDI pair (e.g. two names for one drug). Never merge silently:
g["severity_conflict"] = g.severity_values.str.contains(";")
g["severity"] = g.severity_values.where(~g.severity_conflict, "conflict")
g = g.reset_index()

linked = pairs.merge(g.drop(columns="severity_values"), on="pair_key", how="inner")
linked.to_csv(LINK_OUT / "linked_pairs.csv", index=False)

in_mec = g.pair_key.isin(pairs.pair_key)
log(f"Distinct DDInter pairs (both drugs matched): {len(g)}")
log(f"  ...of which also in MecDDI: {in_mec.sum()} ({in_mec.mean():.1%})")
log(f"MecDDI pairs: {len(pairs)}; MecDDI pairs with a DDInter severity (= linked): {len(linked)} ({len(linked) / len(pairs):.1%})")
log(f"Severity conflicts (several DDInter rows, different labels): {int(linked.severity_conflict.sum())}")
log("\nSeverity distribution in linked pairs:")
log(linked.severity.value_counts().to_string())
log("\nMechanism-category counts in linked pairs:")
log(linked[[c for c in linked.columns if c.startswith("cat_")]].sum().to_string())
(LINK_OUT / "linking_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
log(f"\nSaved to {LINK_OUT}. Review unmatched_ddinter_drugs.csv; add fixes to data/manual_name_map.csv and re-run.")
