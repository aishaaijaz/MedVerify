# MedVerify — Condition A (Zero-shot) Demo

A small Streamlit app that asks a **locally run** LLM (via Ollama) to explain a possible drug–drug interaction from **only the two drug names**. It is the *ungrounded baseline* (Condition A) of the MedVerify study.

> **Research prototype only. Not for diagnosis, prescribing, or patient care.**

## What Condition A does and does not do
| Does | Does not |
|---|---|
| Send two drug names and a fixed prompt to a local Ollama model | Query MecDDI or DDInter |
| Show the model's structured answer (status, mechanism, effect, estimated severity, key claims, uncertainty) | Supply severity labels, retrieved evidence, or the answer key |
| Let you set model, temperature, and max tokens, and download the output | Verify any claim (that is Conditions B/C) |

The severity in the output is a **model estimate, not a database label**. Outputs are later compared against MecDDI (mechanism category) and DDInter (severity), and graded for faithfulness to those references, not clinical correctness.

## Requirements
- Python 3.9+
- [Ollama](https://ollama.com) installed and running (default endpoint `http://localhost:11434`)
- `requirements.txt`: `streamlit`, `requests`
- CPU-only machines work; 16 GB RAM is enough for small models

## Setup
```powershell
# 1. Pull two small starter models (run one at a time on 16 GB RAM)
ollama pull qwen2.5:3b
ollama pull llama3.2:3b

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app (Ollama must already be running)
streamlit run app.py
```
Streamlit opens at http://localhost:8501.

## Usage
1. Pick an installed model in the sidebar (the list is read from Ollama; fallback names are shown if none are detected).
2. Enter two different drug names (e.g. Warfarin, Fluconazole). Prefer names as they appear in DDInter/MecDDI so results can be matched later.
3. Click **Generate zero-shot explanation**.
4. Read the output; use **Download output (.txt)** to save it.

## Output format
The prompt forces seven headings: interaction status, proposed mechanism, expected effect, model-estimated severity (Major / Moderate / Minor / Unknown), key factual claims (max 5), uncertainty and limitations, safety note. The short **key factual claims** list is what the later verifier checks.

## Running experiments (reproducibility)
- Keep **temperature and max tokens fixed** across all runs (default 0.2 and 1024).
- Record per run: model tag, temperature, max tokens, drug pair, date. The app shows these under the output.
- Use identical settings for Conditions A, B and C so grounding is the only difference.
- For large batches, call the same prompt from a script rather than clicking through the UI.

## Troubleshooting
| Symptom | Fix |
|---|---|
| "Cannot connect to Ollama" | Start Ollama (open the app or `ollama serve`) |
| No models detected | `ollama pull qwen2.5:3b` then refresh the page |
| Timeout | Use a smaller model or lower max tokens; close other heavy apps |
| Empty response | Re-run; try a different model |

## Privacy and safety
- Inference stays on your machine; no API key or cloud call is used.
- Do not expose port 11434 publicly.
- Small models can invent mechanisms and confident-sounding errors; treat every output as unverified.

## Project context
Part of **MedVerify** (AI in Healthcare mini project, AISHA AIJAZ and SNEHA). Next: Condition B (grounded in MecDDI + DDInter) and Condition C (grounded + claim verification).