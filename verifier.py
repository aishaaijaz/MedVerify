"""Condition C - claim-level verification layer.

Input : a generated answer (Condition A or B), the retrieved evidence, and the database record for the pair.
Output: one verdict per claim plus an overall FLAG / OK. The verifier FLAGS; it never rewrites the answer.

Per-claim checks, in this order (first one that applies decides):
  1. Honest tag      claim tagged "(not in evidence)"               -> not_verifiable   (not penalised)
  2. Unsupported specific  names an enzyme/transporter/etc. with no tag -> unsupported_specific (FLAG)
  3. Database rule   claim states a mechanism category or a severity -> supported / partially_supported / contradicted
                     (deterministic comparison with MecDDI / DDInter; no LLM involved)
  4. Evidence judge  anything else: an LLM judge (temperature 0) reads ONLY the evidence text -> supported /
                     contradicted / not_verifiable. If no judge is configured -> not_verifiable.
The two "Final labels" lines are checked with the same database rule.
"""
import re

from retrieval import CATS, evidence_block, parse_labels, parse_sections, record_text, split_final

FLAGGED = {"contradicted", "unsupported_specific"}

# category -> regex that signals a claim about that mechanism category
CAT_PATTERNS = {
    "absorption": r"\babsorption\b",
    "transport": r"\btransport",
    "distribution": r"\bdistribution\b|protein[- ]binding",
    "metabolism": r"\bmetaboli",
    "excretion": r"\bexcret|renal (?:clearance|elimination)",
    "pd_additive": r"\badditive\b|\bsynerg",
    "pd_antagonistic": r"\bantagonis|\bopposing\b",
}
# things MecDDI categories cannot support: specific enzymes, transporters, receptors
SPECIFIC = re.compile(r"\bCYP\s?\d\w*|\bUGT\s?\d\w*|\bP-?gp\b|P-glycoprotein|\bOATP\w*|\bOCT\d\b|\bBCRP\b|"
                      r"\bMAO-?[AB]\b|\bhERG\b|\bQT\b|\bserotonin syndrome\b", re.I)
NEGATION = re.compile(r"\b(not|no|none|never|without|unlikely)\b", re.I)
META = re.compile(r"^(the )?(evidence|record|database|mecddi|ddinter)\b.*\b(does not|doesn't|no finer|not give|only give|gives only)|"
                  r"\b(is|are) (an|a) (inference|estimate)\b|^(this|these) (must|should|is|are) ", re.I)
ABSENCE = re.compile(r"\b(no|not any|without any)\b[^.]{0,40}\b(record|entry|entries|interaction)\b[^.]{0,60}\b(mecddi|ddinter|database|evidence|recorded|listed|found)\b|"
                     r"\b(mecddi|ddinter|database)\b[^.]{0,60}\b(no|not)\b[^.]{0,30}\b(record|entry|list|contain)", re.I)
NOT_IN_EVIDENCE = re.compile(r"\(\s*not in evidence\s*\)", re.I)


# ------------------------------------------------------------------ claim extraction
def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*•\t") for p in parts if len(p.strip(" -*•\t")) > 15]


def extract_claims(answer, max_claims=12):
    """Claims = the numbered key claims (section 5) + sentences of the mechanism / effect / severity sections."""
    body, _ = split_final(answer)
    sec = parse_sections(body)
    claims, seen = [], set()

    def add(text, source):
        t = re.sub(r"\s+", " ", re.sub(r"^C\d+\s*[:.)-]\s*", "", text)).strip()
        k = t.lower()
        if len(t) > 10 and k not in seen and not META.search(t):
            seen.add(k)
            claims.append({"id": f"K{len(claims) + 1}", "text": t, "source": source,
                           "cites": sorted(set(re.findall(r"\[E\d\]", t))), "tagged": bool(NOT_IN_EVIDENCE.search(t))})

    for line in sec.get(5, "").splitlines():
        if re.match(r"\s*[-*]?\s*C\d+\s*[:.)]", line):
            add(line.strip("-* "), "key_claim")
    for n, name in ((2, "mechanism"), (3, "effect"), (4, "severity")):
        for sent in _sentences(sec.get(n, "")):
            add(sent, name)
    return claims[:max_claims]


# ------------------------------------------------------------------ deterministic database rule
def _claimed_categories(text):
    return [c for c, p in CAT_PATTERNS.items() if re.search(p, text, re.I)]


def _claimed_severity(text):
    m = re.findall(r"\b(major|moderate|minor)\b", text, re.I)
    return m[0].lower() if m and re.search(r"severity|severe|risk|level|rated|classified", text, re.I) else None


def check_categories(claimed, rec):
    """Compare a set of claimed categories with the MecDDI categories of the pair."""
    if rec is None:
        return "not_verifiable", "No database record exists for this pair."
    true = set(rec["categories"])
    claimed = set(claimed)
    if not claimed:
        return "not_verifiable", "No category stated."
    if claimed == true:
        return "supported", "Matches MecDDI categories."
    if claimed <= true:
        return "supported", f"Stated categories are in MecDDI (MecDDI lists more: {', '.join(sorted(true - claimed))})."
    if claimed & true:
        return "partially_supported", f"Overlaps MecDDI; extra stated: {', '.join(sorted(claimed - true))}."
    return "contradicted", f"MecDDI lists {', '.join(sorted(true))}, not {', '.join(sorted(claimed))}."


def check_severity(claimed, rec):
    if rec is None or not rec.get("severity"):
        return "not_verifiable", "No DDInter severity recorded for this pair."
    if claimed in ("unknown", "insufficient_evidence"):
        return "not_verifiable", "Model gave no definite severity."
    ok = claimed == rec["severity"]
    return ("supported", "Matches DDInter severity.") if ok else ("contradicted", f"DDInter records {rec['severity']}.")


# ------------------------------------------------------------------ LLM evidence judge
def judge_prompt(evidence, claim):
    return f"""You are a strict fact-checker. Decide whether the CLAIM is supported by the EVIDENCE only.
Do NOT use outside knowledge.

EVIDENCE:
{evidence}

CLAIM: {claim}

Answer with exactly three lines:
LABEL: supported | contradicted | not_verifiable
QUOTE: the exact words copied from the EVIDENCE that justify the label (write none for not_verifiable)
REASON: one short sentence
Rules: "supported" only if the evidence states it or directly implies it. "contradicted" only if the evidence states the opposite.
A claim that something does NOT exist is supported when the evidence also says it does not exist. Otherwise use "not_verifiable".
Example: CLAIM "No record exists for this pair." with EVIDENCE "No MecDDI/DDInter record exists for this pair." -> LABEL: supported"""


def _flat(x):
    return re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()


def parse_judge(text, evidence=""):
    """Parse the judge. A supported/contradicted label must be backed by a quote that really occurs in the evidence;
    otherwise it is downgraded to not_verifiable (guards against the judge misreading or inventing evidence)."""
    lab = re.search(r"LABEL:\s*(supported|contradicted|not_verifiable)", text, re.I)
    why = re.search(r"REASON:\s*(.+)", text, re.I)
    quote = re.search(r"QUOTE:\s*(.+)", text, re.I)
    label = lab.group(1).lower() if lab else "not_verifiable"
    reason = why.group(1).strip() if why else "Judge gave no clear reason."
    if label != "not_verifiable":
        q = _flat(quote.group(1)) if quote else ""
        if len(q) < 8 or q not in _flat(evidence):
            return "not_verifiable", "Judge's quote was not found in the evidence, so its verdict was not trusted."
    return label, reason


# ------------------------------------------------------------------ main entry point
def verify(answer, ev, record, judge=None):
    """Verify one answer.

    ev     : evidence dict from Evidence.retrieve (shown to the generator)
    record : database record {'categories':[...], 'severity':...} or None  (the verification reference)
    judge  : optional callable(prompt:str)->str, an LLM at temperature 0. None = no LLM layer.
    """
    evidence = evidence_block(ev) + "\n" + record_text(record) if record is not None else evidence_block(ev)
    results = []
    for c in extract_claims(answer):
        text, label, reason, layer = c["text"], None, "", ""
        negated = bool(NEGATION.search(text))
        if c["tagged"]:
            label, reason, layer = "not_verifiable", "Model marked this as outside the evidence.", "tag"
        elif SPECIFIC.search(text) and not re.search(SPECIFIC, evidence):
            label, reason, layer = ("unsupported_specific",
                                    f"Names '{SPECIFIC.search(text).group(0)}', which no retrieved source states.", "rule")
        elif ABSENCE.search(text) and (record is not None or ev.get("mode") == "pair" or not ev.get("pair_found", True)):
            absent = not ev.get("pair_found", record is not None)
            label, layer = ("supported", "Database has no record for this pair.") if absent else ("contradicted", "Database DOES have a record for this pair.")
            reason, layer = layer, "database"
        elif not negated and (_claimed_categories(text) or _claimed_severity(text)):
            cats, sev = _claimed_categories(text), _claimed_severity(text)
            verdicts = []
            if cats:
                verdicts.append(check_categories(cats, record))
            if sev:
                verdicts.append(check_severity(sev, record))
            order = ["contradicted", "partially_supported", "not_verifiable", "supported"]
            label = min((v[0] for v in verdicts), key=order.index)
            reason, layer = " ".join(v[1] for v in verdicts), "database"
        if label is None:
            if judge:
                label, reason = parse_judge(judge(judge_prompt(evidence, text)), evidence)
                layer = "llm_judge"
            else:
                label, reason, layer = "not_verifiable", "No judge configured.", "none"
        results.append({**c, "label": label, "reason": reason, "layer": layer})

    # Final labels (machine-readable lines)
    _, fin = split_final(answer)
    lab = parse_labels(fin or answer)
    final = {}
    if lab["categories"]:
        final["categories"] = check_categories(lab["categories"], record)
    else:
        final["categories"] = ("not_verifiable", "No categories parsed (or insufficient_evidence).")
    final["severity"] = check_severity(lab["severity"], record) if lab["severity"] else ("not_verifiable", "No severity parsed.")

    n = {k: sum(r["label"] == k for r in results) for k in
         ["supported", "partially_supported", "contradicted", "unsupported_specific", "not_verifiable"]}
    flagged = n["contradicted"] + n["unsupported_specific"] + sum(v[0] == "contradicted" for v in final.values())
    return {"claims": results, "final_labels": lab, "final_checks": final, "counts": n,
            "n_flagged": flagged, "verdict": "FLAG" if flagged else "OK"}
