import streamlit as st

from llm import chat, installed_models
from retrieval import Evidence, evidence_block, make_prompt, parse_labels, split_final

st.set_page_config(page_title="MedVerify | Grounded", page_icon="💊", layout="wide")
st.title("💊 MedVerify")
st.subheader("Condition B — Grounded DDI explanation")
st.caption("Evidence is retrieved from MecDDI (mechanism categories) and DDInter (severity) and supplied to the model.")
st.warning("Research prototype only. Do not use this output for diagnosis, prescribing, or patient care.")


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
    mode_label = st.radio("Evidence mode", ["Pair record given", "Held-out (leave-one-out)"],
                          help="Pair record: the pair's own categories and severity are supplied. "
                               "Held-out: they are hidden and the model sees only each drug's other interactions (use for evaluation).")
    mode = "pair" if mode_label.startswith("Pair") else "heldout"
    detected = installed_models()
    model = st.selectbox("Ollama model", detected or ["qwen2.5:3b", "llama3.2:3b"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.select_slider("Maximum output tokens", [512, 768, 1024, 1536, 2048], value=1024)
    st.caption("Keep settings identical to Condition A when comparing.")

c1, c2 = st.columns(2)
drug_a = c1.text_input("Drug 1", placeholder="e.g. Warfarin")
drug_b = c2.text_input("Drug 2", placeholder="e.g. Fluconazole")

if st.button("Retrieve evidence and generate", type="primary", use_container_width=True):
    ev = EV.retrieve(drug_a, drug_b, mode)
    if ev["status"] == "unknown_drug":
        st.error(f"Not found (or ambiguous) in MecDDI: {', '.join(ev['missing'])}. Try the generic name.")
    elif ev["status"] == "same_drug":
        st.error("Enter two different drugs.")
    else:
        with st.spinner(f"Generating locally with {model}..."):
            try:
                ans = chat(model, make_prompt(ev), temperature, max_tokens)
                st.session_state["ev_b"], st.session_state["ans_b"] = ev, ans
                st.session_state["meta_b"] = (ev["a"], ev["b"], model, temperature, max_tokens, mode)
            except Exception as e:  # connection, timeout, HTTP errors
                st.error(f"Ollama request failed: {e}. Is Ollama running and the model pulled?")

if st.session_state.get("ans_b"):
    ev, ans = st.session_state["ev_b"], st.session_state["ans_b"]
    a, b, m, t, n, md = st.session_state["meta_b"]
    shown, final = split_final(ans)
    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.markdown("### Retrieved evidence")
        st.code(evidence_block(ev), language="text")
        if md == "heldout":
            with st.expander("Reference labels (grading only; NOT shown to the model)"):
                st.write(ev["truth"] or "No record for this pair.")
    with right:
        st.markdown("### Model output")
        st.caption(f"{a} + {b} | {m} | T={t} | max tokens={n} | mode={md}")
        st.markdown(shown)
        with st.expander("Machine-readable labels"):
            st.write(parse_labels(ans))
        st.download_button("Download output (.txt)", ans, file_name="medverify_grounded_output.txt")

with st.expander("Condition B protocol"):
    st.markdown("- Retrieval is exact-match lookup on the cleaned MecDDI/DDInter tables.\n"
                "- Pair mode shows the pair's own record; held-out mode hides it (no leakage) for evaluation.\n"
                "- MecDDI has categories only, so detail beyond the categories is tagged '(not in evidence)'.\n"
                "- Same model, settings and section layout as Condition A.")
st.caption("MedVerify academic prototype • Local Ollama inference • Not for clinical use")
