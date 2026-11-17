import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434"

st.set_page_config(page_title="MedVerify | Zero-shot", page_icon="💊", layout="wide")
st.title("💊 MedVerify")
st.subheader("Condition A — Zero-shot DDI explanation")
st.caption("Only the two drug names are supplied. No MecDDI record, DDInter severity, retrieval context, or answer key is provided.")
st.warning("Research prototype only. Do not use this output for diagnosis, prescribing, or patient care.")

def installed_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", []) if m.get("name")]
    except requests.RequestException:
        return []

def make_prompt(a, b):
    return f"""You are generating a research explanation of a possible drug–drug interaction.
This is CONDITION A (ZERO-SHOT).

Drug 1: {a}
Drug 2: {b}

You have NOT been given database records, mechanism evidence, severity labels, or patient details.
Use pretrained knowledge only. Do not invent certainty, citations, or database checks.
Do not assume an interaction exists simply because two drugs are named.
If unsure, say so. Do not give patient-specific treatment or dosing advice.

Use these exact headings:

### 1. Interaction status
State whether a clinically relevant interaction is known, possible, not established, or uncertain. Give confidence.

### 2. Proposed mechanism
Explain the biological/pharmacological mechanism if known. Mention enzymes, transporters, receptors, or pathways only when reasonably confident. Otherwise say the mechanism is uncertain.

### 3. Expected effect
Describe possible consequences. Separate established effects from theoretical possibilities.

### 4. Model-estimated severity
Use Major / Moderate / Minor / Unknown only if there is a defensible basis. Clearly state this is an unverified model estimate, NOT a database-verified label.

### 5. Key factual claims
List up to five short, checkable factual claims.

### 6. Uncertainty and limitations
State what is uncertain and what evidence is needed to verify the explanation.

### 7. Safety note
State that the explanation must be checked against authoritative references by a qualified professional.
"""

with st.sidebar:
    st.header("Model settings")
    detected = installed_models()
    models = detected or ["qwen2.5:3b", "llama3.2:3b"]
    model = st.selectbox("Ollama model", models)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.select_slider("Maximum output tokens", [512, 768, 1024, 1536, 2048], value=1024)
    st.caption("Keep settings fixed when comparing experimental runs.")
    if detected:
        st.success(f"{len(detected)} local model(s) detected.")
    else:
        st.info("No local models detected. Start Ollama and pull a model.")

c1, c2 = st.columns(2)
with c1:
    drug_a = st.text_input("Drug 1", placeholder="e.g. Warfarin")
with c2:
    drug_b = st.text_input("Drug 2", placeholder="e.g. Fluconazole")

if st.button("Generate zero-shot explanation", type="primary", use_container_width=True):
    a, b = drug_a.strip(), drug_b.strip()
    if not a or not b:
        st.error("Enter both drug names.")
    elif a.casefold() == b.casefold():
        st.error("Enter two different drug names.")
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": make_prompt(a, b)}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        with st.spinner(f"Generating locally with {model}..."):
            try:
                r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
                r.raise_for_status()
                answer = r.json().get("message", {}).get("content", "").strip()
                if answer:
                    st.session_state["answer"] = answer
                    st.session_state["meta"] = (a, b, model, temperature, max_tokens)
                else:
                    st.error("Ollama returned an empty response.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Ollama. Make sure its local service is running.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try a smaller model or lower output-token limit.")
            except requests.RequestException as e:
                st.error(f"Ollama request failed: {e}")

if "answer" in st.session_state:
    st.divider()
    a, b, m, t, n = st.session_state["meta"]
    st.markdown("### Model output")
    st.caption(f"{a} + {b} | Model: {m} | Temperature: {t} | Max tokens: {n}")
    st.markdown(st.session_state["answer"])
    st.download_button("Download output (.txt)", st.session_state["answer"],
                       file_name="medverify_zero_shot_output.txt", mime="text/plain")

with st.expander("Condition A protocol"):
    st.markdown("- Input contains only two drug names.\n- No MecDDI or DDInter lookup occurs.\n- Severity is a model estimate, not ground truth.\n- Save model tag and generation settings for each run.\n- Compare outputs later against the independently prepared reference answer key.")

st.caption("MedVerify academic prototype • Local Ollama inference • Not for clinical use")
