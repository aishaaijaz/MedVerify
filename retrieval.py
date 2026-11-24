"""Evidence retrieval for MedVerify Condition B (grounded).

Reads the tables produced by the preprocessing pipeline (data/processed/...).
Retrieval is exact-match lookup on structured tables, not semantic search.

Two evidence modes:
  "pair"     - the pair's own MecDDI categories and DDInter severity are supplied (explanation demo).
  "heldout"  - the pair's own labels are HIDDEN (leave-one-out); the model only sees each drug's OTHER known
               interactions as a category/severity profile plus examples. Use this for evaluation (no leakage).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
CATS = ["absorption", "transport", "distribution", "metabolism", "excretion", "pd_additive", "pd_antagonistic"]
CAT_TEXT = {"absorption": "affects absorption", "transport": "affects cellular transport",
            "distribution": "affects distribution", "metabolism": "affects metabolism",
            "excretion": "affects excretion", "pd_additive": "pharmacodynamic additive effect",
            "pd_antagonistic": "pharmacodynamic antagonistic effect"}


def norm(x):
    return " ".join(str(x).lower().strip().split())


class Evidence:
    def __init__(self):
        self.drugs = pd.read_csv(PROC / "mecddi" / "mecddi_drugs.csv", dtype=str, keep_default_na=False)
        mp = pd.read_csv(PROC / "mecddi" / "mecddi_pairs.csv", dtype=str, keep_default_na=False)
        lk = pd.read_csv(PROC / "linked" / "linked_pairs.csv", dtype=str, keep_default_na=False)
        sev = lk.set_index("pair_key").severity.to_dict()
        mp["severity"] = mp.pair_key.map(sev).fillna("")  # '' = no DDInter record for this pair
        self.pairs = mp.set_index("pair_key")
        self.name = self.drugs.set_index("drug_id").drug_name.to_dict()
        self.cls = self.drugs.set_index("drug_id").therapeutic_class.to_dict()
        # name -> id: primary names first, then unambiguous synonyms
        self.idx = {}
        for did, n in zip(self.drugs.drug_id, self.drugs.drug_name_norm):
            self.idx.setdefault(n, set()).add(did)
        self.syn = {}
        for did, s in zip(self.drugs.drug_id, self.drugs.synonyms_norm):
            for t in s.split(";"):
                if t:
                    self.syn.setdefault(t, set()).add(did)
        # partner index: drug_id -> list of pair_keys
        self.by_drug = {}
        for k, a, b in zip(self.pairs.index, self.pairs.drug_a_id, self.pairs.drug_b_id):
            self.by_drug.setdefault(a, []).append(k)
            self.by_drug.setdefault(b, []).append(k)

    def resolve(self, text):
        """Drug name -> MecDDI drug_id, or None if unknown/ambiguous."""
        n = norm(text)
        for table in (self.idx, self.syn):
            if n in table and len(table[n]) == 1:
                return next(iter(table[n]))
        return None

    @staticmethod
    def key(a, b):
        return "|".join(sorted([a, b]))

    def pair_record(self, ida, idb):
        k = self.key(ida, idb)
        if k not in self.pairs.index:
            return None
        r = self.pairs.loc[k]
        return {"categories": [c for c in r.categories.split(";") if c], "severity": r.severity or None}

    def profile(self, did, exclude_key=None, n_examples=4):
        """Summary of a drug's other known interactions (the held-out pair is excluded)."""
        keys = [k for k in self.by_drug.get(did, []) if k != exclude_key]
        sub = self.pairs.loc[keys]
        cat_counts = {c: int((sub["cat_" + c] == "1").sum()) for c in CATS}
        sev_counts = sub.severity.replace("", "no severity record").value_counts().to_dict()
        ex = sub.sort_values("pair_key").sample(min(n_examples, len(sub)), random_state=42) if len(sub) else sub
        examples = []
        for k, r in ex.iterrows():
            other = r.drug_b_id if r.drug_a_id == did else r.drug_a_id
            examples.append({"partner": self.name[other], "categories": r.categories.split(";"),
                             "severity": r.severity or "unknown/none"})
        return {"drug": self.name[did], "therapeutic_class": self.cls.get(did, ""), "n_known": len(sub),
                "category_counts": cat_counts, "severity_counts": sev_counts, "examples": examples}

    def retrieve(self, name_a, name_b, mode="pair"):
        ida, idb = self.resolve(name_a), self.resolve(name_b)
        if not ida or not idb:
            miss = [n for n, i in ((name_a, ida), (name_b, idb)) if not i]
            return {"status": "unknown_drug", "missing": miss}
        if ida == idb:
            return {"status": "same_drug"}
        rec = self.pair_record(ida, idb)
        out = {"status": "ok", "mode": mode, "a": self.name[ida], "b": self.name[idb],
               "pair_found": rec is not None}
        if mode == "pair":
            out["pair"] = rec
        else:
            k = self.key(ida, idb)
            out["profile_a"], out["profile_b"] = self.profile(ida, k), self.profile(idb, k)
            out["truth"] = rec  # kept for grading only; NEVER put in the prompt
        return out


def _profile_text(p):
    cc = ", ".join(f"{CAT_TEXT[c]}: {n}" for c, n in p["category_counts"].items() if n) or "none"
    sc = ", ".join(f"{k}: {v}" for k, v in p["severity_counts"].items()) or "none"
    ex = "\n".join(f"    - with {e['partner']}: {', '.join(CAT_TEXT.get(c, c) for c in e['categories'])}; severity {e['severity']}"
                   for e in p["examples"])
    return (f"  {p['drug']} (class: {p['therapeutic_class'] or 'n/a'}); {p['n_known']} other recorded interactions\n"
            f"    mechanism categories across them: {cc}\n    severities across them: {sc}\n    examples:\n{ex}")


def evidence_block(ev):
    """Plain-text evidence shown to the model. Never includes the held-out truth."""
    if ev["mode"] == "pair":
        if not ev["pair_found"]:
            return "No MecDDI/DDInter record exists for this pair."
        r = ev["pair"]
        cats = "; ".join(CAT_TEXT[c] for c in r["categories"])
        return (f"[E1] MecDDI mechanism categories for this pair: {cats}\n"
                f"[E2] DDInter severity for this pair: {r['severity'] or 'no DDInter record'}")
    return ("The record for this exact pair is NOT available. Evidence is each drug's other recorded interactions:\n"
            f"[E1]\n{_profile_text(ev['profile_a'])}\n[E2]\n{_profile_text(ev['profile_b'])}")


HEADINGS = """### 1. Interaction status
### 2. Mechanism
### 3. Expected effect
### 4. Severity
### 5. Key factual claims
### 6. Uncertainty and limitations
### 7. Safety note"""

FINAL = f"""### Final labels
CATEGORIES: <comma-separated from: {', '.join(CATS)}, or insufficient_evidence>
SEVERITY: <Major | Moderate | Minor | Unknown | insufficient_evidence>"""

STYLE = """Write in plain, short sentences for a student reader. Explain any technical term in a few words.
Start section 1 with one sentence that summarises the whole answer."""


def make_prompt(ev):
    """Condition B prompt: same section layout as Condition A, but grounded in retrieved evidence."""
    held = ev["mode"] == "heldout"
    sev_rule = ("The pair's own record is hidden. In section 4 give a cautious severity INFERRED from the drugs' other "
                "interactions and say clearly that it is an inference, not a database label."
                if held else
                "In section 4 report the DDInter severity exactly as given [E2] and say it is the recorded database label.")
    return f"""You are generating a research explanation of a possible drug–drug interaction.
This is CONDITION B (GROUNDED).

Drug 1: {ev['a']}
Drug 2: {ev['b']}

Evidence retrieved from MecDDI (mechanism categories) and DDInter (severity):
{evidence_block(ev)}

Rules:
- Base every statement on the evidence above and cite it as [E1] or [E2].
- MecDDI gives only mechanism CATEGORIES, not detailed mechanisms. Do not name enzymes, transporters, receptors or pathways unless the evidence names them. Anything beyond the evidence must be tagged "(not in evidence)".
- {sev_rule}
- If the evidence is insufficient, say so. Do not give patient-specific or dosing advice.
{STYLE}

Use these exact headings:

### 1. Interaction status
Is a clinically relevant interaction recorded, possible, not established, or uncertain? Give confidence.

### 2. Mechanism
Explain the mechanism CATEGORY from the evidence in plain language [E1]. Say clearly that the evidence does not give finer detail, if it does not.

### 3. Expected effect
Describe possible consequences. Separate what the evidence supports from general knowledge, which must be tagged "(not in evidence)".

### 4. Severity
Major / Moderate / Minor / Unknown, with its source.

### 5. Key factual claims
Up to five numbered, short, checkable claims written as "C1: ...", "C2: ...", each ending with [E1], [E2] or "(not in evidence)".

### 6. Uncertainty and limitations
What is uncertain and what would be needed to verify this.

### 7. Safety note
State that this must be checked against authoritative references by a qualified professional.

{FINAL}
"""


def make_prompt_zero(a, b):
    """Condition A prompt: names only, same section layout, plus the machine-readable final labels."""
    return f"""You are generating a research explanation of a possible drug–drug interaction.
This is CONDITION A (ZERO-SHOT).

Drug 1: {a}
Drug 2: {b}

You have NOT been given database records, mechanism evidence, severity labels, or patient details.
Use pretrained knowledge only. Do not invent certainty, citations, or database checks.
If unsure, say so. Do not give patient-specific treatment or dosing advice.
{STYLE}

Use these exact headings:

### 1. Interaction status
Is a clinically relevant interaction known, possible, not established, or uncertain? Give confidence.

### 2. Mechanism
Explain the mechanism if known. Mention enzymes, transporters or pathways only when reasonably confident.

### 3. Expected effect
Describe possible consequences. Separate established effects from theoretical ones.

### 4. Severity
Major / Moderate / Minor / Unknown. State that this is an unverified model estimate.

### 5. Key factual claims
Up to five numbered, short, checkable claims written as "C1: ...", "C2: ...".

### 6. Uncertainty and limitations
### 7. Safety note
State that this must be checked against authoritative references by a qualified professional.

{FINAL}
"""


def split_final(text):
    """Split an answer into (reader-facing part, final-labels part)."""
    k = text.lower().rfind("### final labels")
    return (text[:k].rstrip(), text[k:]) if k != -1 else (text, "")


def parse_sections(text):
    """Return {1: body, 2: body, ...} for '### n. Title' headings."""
    import re
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*#{2,4}\s*(\d)\.\s*", line)
        if m:
            cur = int(m.group(1)); out[cur] = []
        elif cur is not None and not line.lower().lstrip("# ").startswith("final labels"):
            out[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}


def record_text(rec):
    """Database record as text, for the verifier (never shown to the generator in held-out mode)."""
    if not rec:
        return "No MecDDI/DDInter record exists for this pair."
    cats = "; ".join(CAT_TEXT[c] for c in rec["categories"])
    return f"[DB] MecDDI categories: {cats}. DDInter severity: {rec['severity'] or 'no DDInter record'}."


def parse_labels(text):
    """Extract the machine-readable labels from a model answer (for automatic grading)."""
    cats, sev = [], None
    for line in text.splitlines():
        u = line.strip().lstrip("-*# ").lower()
        if u.startswith("categories:"):
            cats = [c.strip(" *`.") for c in u.split(":", 1)[1].split(",") if c.strip(" *`.") in CATS]
        elif u.startswith("severity:"):
            v = u.split(":", 1)[1].strip(" *`.<>")
            sev = v if v in ("major", "moderate", "minor", "unknown", "insufficient_evidence") else None
    return {"categories": cats, "severity": sev}


def score_answer(ans, truth):
    """Score one answer. `truth` = {'categories': [...], 'severity': ...} or None when the pair has no record.

    parsed     = the answer contains BOTH label lines (CATEGORIES: and SEVERITY:). Otherwise it is a FORMAT FAILURE
                 (empty text, echoed prompt, missing labels) and is never treated as an abstention.
    abstained  = labels present but no category and severity insufficient/unknown.
    """
    import re
    lab = parse_labels(ans or "")
    has = bool(re.search(r"(?im)^[\W_]*categories\s*:", ans or "")) and bool(re.search(r"(?im)^[\W_]*severity\s*:", ans or ""))
    abst = has and not lab["categories"] and lab["severity"] in (None, "insufficient_evidence", "unknown")
    out = {"pred_categories": ";".join(lab["categories"]), "pred_severity": lab["severity"] or "", "parsed": has, "abstained": abst}
    if truth:
        tc = set(truth["categories"])
        out.update(cat_exact=has and set(lab["categories"]) == tc, cat_overlap=has and bool(set(lab["categories"]) & tc),
                   sev_correct=has and lab["severity"] == truth["severity"])
    else:  # no record exists: the only correct behaviour is to abstain
        out.update(cat_exact=abst, cat_overlap=False, sev_correct=abst)
    return out
