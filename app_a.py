import streamlit as st

from llm import chat, installed_models
from retrieval import make_prompt_zero, parse_labels, split_final

st.set_page_config(page_title="MedVerify | Zero-shot", page_icon="💊", layout="wide")
st.title("💊 MedVerify")
st.subheader("Condition A — Zero-shot DDI explanation")
st.caption("Only the two drug names are supplied. No MecDDI record, DDInter severity, retrieval context, or answer key is provided.")
st.warning("Research prototype only. Do not use this output for diagnosis, prescribing, or patient care.")

with st.sidebar:
    st.header("Model settings")
    detected = installed_models()
    model = st.selectbox("Ollama model", detected or ["qwen2.5:3b", "llama3.2:3b"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.select_slider("Maximum output tokens", [512, 768, 1024, 1536, 2048], value=1024)
    st.caption("Keep settings identical to Conditions B and C when comparing.")
    st.success(f"{len(detected)} local model(s) detected.") if detected else st.info("No local models detected. Start Ollama and pull a model.")

c1, c2 = st.columns(2)
drug_a = c1.text_input("Drug 1", placeholder="e.g. Warfarin")
drug_b = c2.text_input("Drug 2", placeholder="e.g. Fluconazole")

if st.button("Generate zero-shot explanation", type="primary", use_container_width=True):
    a, b = drug_a.strip(), drug_b.strip()
    if not a or not b:
        st.error("Enter both drug names.")
    elif a.casefold() == b.casefold():
        st.error("Enter two different drug names.")
    else:
        with st.spinner(f"Generating locally with {model}..."):
            try:
                ans = chat(model, make_prompt_zero(a, b), temperature, max_tokens)
                if ans:
                    st.session_state["ans_a"] = ans
                    st.session_state["meta_a"] = (a, b, model, temperature, max_tokens)
                else:
                    st.error("Ollama returned an empty response.")
            except Exception as e:
                st.error(f"Ollama request failed: {e}. Is Ollama running and the model pulled?")

if st.session_state.get("ans_a"):
    ans = st.session_state["ans_a"]
    a, b, m, t, n = st.session_state["meta_a"]
    shown, _ = split_final(ans)
    st.divider()
    st.markdown("### Model output")
    st.caption(f"{a} + {b} | {m} | T={t} | max tokens={n}")
    st.markdown(shown)
    with st.expander("Machine-readable labels"):
        st.write(parse_labels(ans))
    st.download_button("Download output (.txt)", ans, file_name="medverify_zero_shot_output.txt")

st.caption("MedVerify academic prototype • Local Ollama inference • Not for clinical use")
