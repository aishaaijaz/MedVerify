import pandas as pd
import streamlit as st

from llm import chat, installed_models
from retrieval import Evidence, evidence_block, make_prompt, parse_labels, split_final
from verifier import verify

st.set_page_config(page_title="MedVerify | Verified", page_icon="💊", layout="wide")
st.title("💊 MedVerify")
st.subheader("Condition C — Grounded + claim-level verification")
st.caption("Retrieve evidence → generate a grounded explanation → split it into claims → check every claim. "
           "The verifier flags problems; it never rewrites the answer.")
st.warning("Research prototype only. Do not use this output for diagnosis, prescribing, or patient care.")

ICON = {"supported": "✅ supported", "partially_supported": "🟡 partly supported", "contradicted": "❌ contradicted",
        "unsupported_specific": "🚩 unsupported detail", "not_verifiable": "❔ not verifiable"}
LAYER = {"database": "database rule", "rule": "specific-term rule", "llm_judge": "LLM judge", "tag": "model's own tag", "none": "-"}


@st.cache_resource
def load_evidence():
    return Evidence()


try:
    EV = load_evidence()
except FileNotFoundError as e:
    st.error(f"Processed data not found ({e}). Run the preprocessing pipeline first (python run_all.py).")
    st.stop()

with st.sidebar:
    st.header("Settings")
    mode_label = st.radio("Evidence mode", ["Pair record given", "Held-out (leave-one-out)"])
    mode = "pair" if mode_label.startswith("Pair") else "heldout"
    detected = installed_models()
    model = st.selectbox("Generator model", detected or ["qwen2.5:3b", "llama3.2:3b"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.select_slider("Maximum output tokens", [512, 768, 1024, 1536, 2048], value=1024)
    st.divider()
    st.header("Verifier")
    use_db = st.checkbox("Check against the database record", value=True,
                         help="Compares category/severity claims with MecDDI/DDInter for this pair. "
                              "Off = the verifier sees only the evidence the generator saw.")
    use_judge = st.checkbox("Use LLM judge for other claims", value=True,
                            help="A model at temperature 0 reads only the evidence. Slower; one call per claim.")
    judge_model = st.selectbox("Judge model", detected or ["qwen2.5:3b", "llama3.2:3b"],
                               index=detected.index(model) if model in detected else 0)

c1, c2 = st.columns(2)
drug_a = c1.text_input("Drug 1", placeholder="e.g. Warfarin")
drug_b = c2.text_input("Drug 2", placeholder="e.g. Fluconazole")

if st.button("Generate and verify", type="primary", use_container_width=True):
    ev = EV.retrieve(drug_a, drug_b, mode)
    if ev["status"] == "unknown_drug":
        st.error(f"Not found (or ambiguous) in MecDDI: {', '.join(ev['missing'])}. Try the generic name.")
    elif ev["status"] == "same_drug":
        st.error("Enter two different drugs.")
    else:
        try:
            with st.spinner(f"Generating with {model}..."):
                ans = chat(model, make_prompt(ev), temperature, max_tokens)
            record = (ev["pair"] if mode == "pair" else ev["truth"]) if use_db else None
            judge = (lambda p: chat(judge_model, p, 0.0, 120)) if use_judge else None
            with st.spinner("Verifying claims..."):
                res = verify(ans, ev, record, judge)
            st.session_state["c"] = (ev, ans, res, (model, judge_model if use_judge else "-", temperature, mode, use_db))
        except Exception as e:
            st.error(f"Ollama request failed: {e}. Is Ollama running and the model(s) pulled?")

if "c" in st.session_state:
    ev, ans, res, (m, jm, t, md, db) = st.session_state["c"]
    shown, _ = split_final(ans)
    cnt = res["counts"]
    if res["verdict"] == "FLAG":
        st.error(f"⚠️ FLAGGED — {res['n_flagged']} problem(s) found. Do not trust the flagged statements.")
    else:
        st.success("No contradictions or unsupported details found. (Unverifiable claims may remain.)")
    cols = st.columns(5)
    for col, (k, lab) in zip(cols, ICON.items()):
        col.metric(lab, cnt[k])

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### Model output")
        st.caption(f"{ev['a']} + {ev['b']} | generator {m} | judge {jm} | T={t} | mode={md} | database check={'on' if db else 'off'}")
        st.markdown(shown)
    with right:
        st.markdown("### Retrieved evidence")
        st.code(evidence_block(ev), language="text")

    st.markdown("### Claim-by-claim verification")
    df = pd.DataFrame([{"id": c["id"], "claim": c["text"], "verdict": ICON[c["label"]],
                        "checked by": LAYER[c["layer"]], "why": c["reason"]} for c in res["claims"]])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Final labels")
    f1, f2 = st.columns(2)
    lab = res["final_labels"]
    f1.write(f"**Categories:** {', '.join(lab['categories']) or '—'}  \n{ICON[res['final_checks']['categories'][0]]} — {res['final_checks']['categories'][1]}")
    f2.write(f"**Severity:** {lab['severity'] or '—'}  \n{ICON[res['final_checks']['severity'][0]]} — {res['final_checks']['severity'][1]}")

    st.download_button("Download claim table for hand-labelling (.csv)",
                       df.assign(human_label="").to_csv(index=False), file_name="claims_to_label.csv")

with st.expander("How the verifier works"):
    st.markdown("""
1. **Claims**: the numbered key claims plus the sentences of the mechanism, effect and severity sections.
2. **Honest tag**: a claim the model itself marked "(not in evidence)" is *not verifiable* (not penalised).
3. **Specific-term rule**: naming an enzyme/transporter (e.g. CYP…, P-gp) that no retrieved source states → *unsupported detail* (flag).
4. **Database rule**: a stated mechanism category or severity is compared with MecDDI / DDInter → supported / partly / contradicted.
5. **LLM judge**: all other claims are judged against the evidence text only (temperature 0).
6. **Verdict**: FLAG if any claim is contradicted or unsupported, or a final label contradicts the database.
""")
st.caption("MedVerify academic prototype • Local Ollama inference • Not for clinical use")
