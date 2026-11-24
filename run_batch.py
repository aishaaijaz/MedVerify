"""Batch runner for MedVerify: run Condition A or B (optionally with Condition C verification) over many pairs.

Examples
  python run_batch.py --condition A --model qwen2.5:3b --n 20 --out runs/A_qwen
  python run_batch.py --condition B --mode heldout --model qwen2.5:3b --n 20 --out runs/B_qwen --verify
  python run_batch.py --condition B --mode heldout --model qwen2.5:3b --judge-model qwen2.5:3b --verify --out runs/C_qwen
  python run_batch.py --condition B --mode heldout --n 5 --verify --dry-run --out runs/test     (no Ollama needed)

With --pairs eval_pairs.csv the fixed evaluation list is used (groups: main = pairs with a record, no_record = abstention test).
Otherwise sampling: n pairs per severity class (major / moderate / minor) from pairs present in BOTH MecDDI and DDInter,
reproducible via --seed. Use the same --seed, --n, --temperature, --max-tokens for every condition and model.
Outputs: results.csv (one row per pair), claims.csv (one row per claim, blank human_label column), summary.txt.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from llm import chat
from retrieval import CATS, Evidence, make_prompt, make_prompt_zero, score_answer
from retrieval import evidence_block, record_text
from verifier import verify

SEVS = ["major", "moderate", "minor"]


def macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def stub_answer(ev, condition):
    """Offline stand-in for a model (dry run only): guesses the most common label in the evidence."""
    cat, sev = "metabolism", "moderate"
    if condition == "B" and ev["mode"] == "heldout":
        tot = {c: ev["profile_a"]["category_counts"][c] + ev["profile_b"]["category_counts"][c] for c in CATS}
        cat = max(tot, key=tot.get)
    elif condition == "B":
        r = ev["pair"] or {"categories": [cat], "severity": sev}
        cat, sev = r["categories"][0], r["severity"] or sev
    words = {"pd_additive": "pharmacodynamic additive", "pd_antagonistic": "pharmacodynamic antagonistic"}.get(cat, cat)
    return (f"### 1. Interaction status\nA recorded interaction is likely. Confidence low.\n### 2. Mechanism\n"
            f"The interaction involves the {words} category [E1].\n### 3. Expected effect\nExposure may change (not in evidence).\n"
            f"### 4. Severity\nThe severity is {sev} [E2].\n### 5. Key factual claims\nC1: The mechanism category is {words} [E1].\n"
            f"C2: The severity is {sev} [E2].\n### 6. Uncertainty and limitations\nStub.\n### 7. Safety note\nCheck references.\n"
            f"### Final labels\nCATEGORIES: {cat}\nSEVERITY: {sev.capitalize()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["A", "B"], required=True)
    ap.add_argument("--mode", choices=["pair", "heldout"], default="heldout", help="Condition B evidence mode")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--judge-model", default=None, help="enable the LLM judge in the verifier")
    ap.add_argument("--verify", action="store_true", help="apply Condition C verification to every answer")
    ap.add_argument("--n", type=int, default=20, help="pairs per severity class (random sample; ignored with --pairs)")
    ap.add_argument("--pairs", default=None, help="CSV with drug_a, drug_b, group (e.g. eval_pairs.csv); use for the real evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="no Ollama: use a stub generator and no judge")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    E = Evidence()
    if a.pairs:
        pl = pd.read_csv(a.pairs, dtype=str, keep_default_na=False)
        items = [(r.drug_a, r.drug_b, r.group, r.get("pair_id", "")) for _, r in pl.iterrows()]
    else:
        pairs = E.pairs[E.pairs.severity.isin(SEVS)]
        rng = random.Random(a.seed)
        items = []
        for s_ in SEVS:
            keys = sorted(pairs.index[pairs.severity == s_])
            items += [(E.name[E.pairs.loc[k].drug_a_id], E.name[E.pairs.loc[k].drug_b_id], "main", "") for k in rng.sample(keys, min(a.n, len(keys)))]
    chosen = items
    print(f"{len(chosen)} pairs | condition {a.condition} | mode {a.mode if a.condition == 'B' else '-'} | model {a.model}")

    judge = None
    if a.verify and a.judge_model and not a.dry_run:
        judge = lambda p: chat(a.judge_model, p, 0.0, 120)

    rows, claim_rows, skipped = [], [], 0
    t0 = time.time()
    for i, (na, nb, group, pid) in enumerate(chosen, 1):
        ev = E.retrieve(na, nb, "heldout" if a.condition == "A" else a.mode)
        if ev["status"] != "ok":  # name resolution failed: skip, report
            skipped += 1
            continue
        truth = ev["truth"] if "truth" in ev else ev["pair"]  # None when the pair has no record
        k = E.key(E.resolve(na), E.resolve(nb))
        prompt = make_prompt_zero(ev["a"], ev["b"]) if a.condition == "A" else make_prompt(ev)
        try:
            ans = stub_answer(ev, a.condition) if a.dry_run else chat(a.model, prompt, a.temperature, a.max_tokens)
        except Exception as e:
            print(f"  [{i}] generation failed: {e}")
            ans = ""
        sc = score_answer(ans, truth)
        tc = truth["categories"] if truth else []
        ts = truth["severity"] if truth else "none"
        row = {"pair_id": pid, "group": group, "pair_key": k, "drug_a": ev["a"], "drug_b": ev["b"],
               "true_categories": ";".join(tc), "true_severity": ts, **sc, "answer": ans}
        if a.verify and ans and group == "main":
            res = verify(ans, ev, truth, judge)
            row.update(verdict=res["verdict"], n_flagged=res["n_flagged"], n_claims=len(res["claims"]),
                       **{f"n_{kk}": v for kk, v in res["counts"].items()})
            for c in res["claims"]:
                claim_rows.append({"pair_id": pid, "pair_key": k, "claim_id": c["id"], "claim": c["text"], "source": c["source"],
                                   "verifier_label": c["label"], "layer": c["layer"], "reason": c["reason"],
                                   "evidence": evidence_block(ev) + "\n" + record_text(truth),
                                   "rater1": "", "rater2": "", "human_label": ""})
        rows.append(row)
        if i % 10 == 0:
            print(f"  {i}/{len(chosen)}  ({time.time() - t0:.0f}s)")
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)  # checkpoint
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    if claim_rows:
        pd.DataFrame(claim_rows).to_csv(out / "claims.csv", index=False)

    nr = df[df.group == "no_record"]
    df = df[df.group == "main"]
    L = [f"Condition {a.condition} | mode {a.mode if a.condition == 'B' else '-'} | model {a.model} | judge {a.judge_model} | "
         f"T={a.temperature} | max_tokens={a.max_tokens} | seed={a.seed} | dry_run={a.dry_run}",
         f"Main pairs evaluated: {len(df)} (skipped, unresolved names: {skipped}); FORMAT FAILURES (no label lines): {(~df.parsed.astype(bool)).mean():.1%}",
         f"Category exact-match: {df.cat_exact.mean():.1%} | category any-overlap: {df.cat_overlap.mean():.1%}",
         f"Severity accuracy: {df.sev_correct.mean():.1%} | severity macro-F1 (major/moderate/minor): "
         f"{macro_f1(df.true_severity, df.pred_severity, SEVS):.3f}",
         "Severity accuracy by true class: " + ", ".join(f"{s}: {df[df.true_severity == s].sev_correct.mean():.1%}" for s in SEVS),
         "Category recall by true category: " + ", ".join(
             f"{c}: {np.mean([c in p.split(';') for p in df[df.true_categories.str.split(';').apply(lambda x: c in x)].pred_categories]):.0%}"
             for c in CATS if df.true_categories.str.split(';').apply(lambda x: c in x).any()),
         f"Wrongly abstained on pairs WITH a record: {df.abstained.astype(bool).mean():.1%}"]
    if len(nr):
        L.append(f"No-record pairs (n={len(nr)}): correctly abstained {(nr.cat_exact & nr.sev_correct).mean():.1%}")
    if a.verify and "verdict" in df:
        wrong = ~(df.cat_exact & df.sev_correct)
        L += ["", "Verification (Condition C)",
              f"Answers flagged: {(df.verdict == 'FLAG').mean():.1%}",
              f"Flagged among answers with a wrong final label: {(df[wrong].verdict == 'FLAG').mean():.1%} (n={int(wrong.sum())})",
              f"Flagged among answers with correct final labels: {(df[~wrong].verdict == 'FLAG').mean():.1%} (n={int((~wrong).sum())})",
              f"Mean claims per answer: {df.n_claims.mean():.1f}; unsupported-detail claims per answer: {df.n_unsupported_specific.mean():.2f}; "
              f"contradicted per answer: {df.n_contradicted.mean():.2f}",
              "NOTE: category/severity checks compare with the same databases that define the reference, so they are exact by construction;",
              "what needs human labels is the free-text claim layer (claims.csv -> kappa_raters.py -> eval_verifier.py)."]
    (out / "summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    sys.exit(main())
