"""Build the fixed evaluation pair list (run once, then commit eval_pairs.csv and never change it).

  python make_pairs.py            -> eval_pairs.csv

50 main pairs: present in BOTH MecDDI and DDInter, severity Major/Moderate/Minor (17/17/16), at least 3 pairs for every
mechanism category where available, and no drug used more than twice (avoids hub-drug bias).
10 no-record pairs: both drugs known, but the pair is absent from MecDDI (tests whether the model abstains).
The file holds drug names only; reference labels are looked up from the databases at run time.
"""
import random

import pandas as pd

from retrieval import CATS, Evidence

SEED, MAX_PER_DRUG = 42, 2
QUOTA = {"major": 17, "moderate": 17, "minor": 16}
E = Evidence()
rng = random.Random(SEED)
P = E.pairs[E.pairs.severity.isin(QUOTA)].copy()
# keep only pairs whose names resolve unambiguously (so the apps and batch runner can find them)
ok = lambda r: E.resolve(E.name[r.drug_a_id]) == r.drug_a_id and E.resolve(E.name[r.drug_b_id]) == r.drug_b_id
P = P[P.apply(ok, axis=1)]
used, chosen, sev_n = {}, [], {s: 0 for s in QUOTA}


def take(key):
    r = P.loc[key]
    if key in chosen or used.get(r.drug_a_id, 0) >= MAX_PER_DRUG or used.get(r.drug_b_id, 0) >= MAX_PER_DRUG:
        return False
    if sev_n[r.severity] >= QUOTA[r.severity]:
        return False
    chosen.append(key)
    sev_n[r.severity] += 1
    for d in (r.drug_a_id, r.drug_b_id):
        used[d] = used.get(d, 0) + 1
    return True


# 1) category coverage: rarest categories first
for c in sorted(CATS, key=lambda c: (P["cat_" + c] == "1").sum()):
    cand = sorted(P.index[P["cat_" + c] == "1"])
    rng.shuffle(cand)
    got = sum(P.loc[k, "cat_" + c] == "1" for k in chosen)
    for k in cand:
        if got >= 3:
            break
        got += take(k)
# 2) fill remaining severity quotas
for s in QUOTA:
    cand = sorted(P.index[P.severity == s])
    rng.shuffle(cand)
    for k in cand:
        if sev_n[s] >= QUOTA[s]:
            break
        take(k)

rows = [{"pair_id": f"P{i:02d}", "pair_key": k, "drug_a": E.name[P.loc[k, "drug_a_id"]], "drug_b": E.name[P.loc[k, "drug_b_id"]],
         "group": "main"} for i, k in enumerate(chosen, 1)]

# no-record pairs among drugs already in the list (familiar names), absent from MecDDI
drugs = sorted({d for k in chosen for d in (P.loc[k, "drug_a_id"], P.loc[k, "drug_b_id"])})
nr = []
while len(nr) < 10:
    a, b = rng.sample(drugs, 2)
    k = E.key(a, b)
    if k not in E.pairs.index and k not in [x[0] for x in nr]:
        nr.append((k, a, b))
rows += [{"pair_id": f"N{i:02d}", "pair_key": k, "drug_a": E.name[a], "drug_b": E.name[b], "group": "no_record"}
         for i, (k, a, b) in enumerate(nr, 1)]
df = pd.DataFrame(rows)
df.to_csv("eval_pairs.csv", index=False)
m = P.loc[chosen]
print(f"{len(chosen)} main pairs | severity {m.severity.value_counts().to_dict()}")
print("category coverage:", {c: int((m['cat_' + c] == '1').sum()) for c in CATS})
print(f"distinct drugs: {len(drugs)}; max uses per drug: {max(used.values())}; no-record pairs: {len(nr)}")
