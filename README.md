# MedVerify
### A Retrieval-Grounded, Self-Verifying LLM System for Explaining Drug–Drug Interaction Mechanisms

**Course:** AI in Healthcare — Mini Project  
**Team members:** AISHA AIJAZ, SNEHA  
**Last updated:** 16 November 2026

> **Status:** Preprocessing of both datasets (DDInter, MecDDI) and linking are complete and tested. Test-set construction, model experiments, verification and the Streamlit demo are planned.

## 1. Project overview

MedVerify studies how well locally run LLMs state and explain the **mechanism category and severity** of drug–drug interactions (DDIs), and whether retrieval grounding and claim-level verification against two structured databases make those answers more accurate and traceable.

**Research question:** How do retrieval grounding in MecDDI (mechanism categories) and DDInter (severity), together with claim-level verification, affect the accuracy and evidence support of LLM-generated DDI mechanism and severity explanations?

### Conditions
1. **A — Ungrounded:** the model predicts category, severity and explanation from pretrained knowledge.
2. **B — Grounded:** retrieved evidence is supplied; the model must stay within it and say when evidence is insufficient. To avoid leakage, test pairs are held out and the model sees the drugs' other known interactions.
3. **C — Grounded + verified:** the output is split into atomic claims; category/severity claims are checked against MecDDI and DDInter, other claims against the retrieved evidence. The verifier flags claims; it does not rewrite them.

Two Ollama models form the core comparison; a third is stretch work. Models are chosen after a smoke test, and exact tags, settings and dates are recorded per run.

### Evaluation
- **Automatic:** category and severity predictions graded on thousands of linked pairs (accuracy, macro-F1, confusion matrices, per-class results).
- **Human:** 40–60 stratified linked pairs; both members grade the explanations (correct / partial / incorrect); Cohen's kappa; verifier precision/recall/F1 against hand-labelled claims.
- **Circularity:** reference and retrieval source are the same databases, so grounded results measure faithfulness; a subset is cross-checked against drug labels.

### Demo
A Streamlit research prototype (not a clinical tool) to compare A/B/C outputs with evidence and claim labels. It calls Ollama's local API; do not expose that endpoint publicly.

## 2. Data

| | DDInter 2.0 | MecDDI |
|---|---|---|
| Provides | Severity (Major/Moderate/Minor/Unknown) | Mechanism categories (7) |
| Raw files | 8 CSVs (`data/raw/`) | 9 CSVs (`MecDDI/`) |
| Raw rows | 222,383 | 178,406 pairs (+ category files) |
| After cleaning | 160,235 unique pairs, 1,939 drugs | 178,406 pairs, 1,922 drugs |
| Text explanations | None | None |

- DDInter severity: Moderate 60.33%, Unknown 18.61%, Major 16.80%, Minor 4.26%. `Unknown` is its own class, not a negative label.
- MecDDI labels per pair: 1 label 174,627; 2 labels 3,478; 3 labels 301. Most frequent: pharmacodynamic additive (109,271), metabolism (38,532), pharmacodynamic antagonistic (22,664); rarest: distribution (1,110), excretion (1,378).
- **Linking** (by normalized drug name): 1,894/1,939 DDInter drugs matched (97.7%); 119,601 pairs (67.0% of MecDDI pairs) have both category and severity; 3 severity conflicts flagged.
- Neither dataset has mechanism text; this is a stated limitation.

**Licensing:** DDInter is CC BY-NC-SA (verify on the official site). MecDDI states "all rights reserved"; cite the MecDDI paper (J. Chem. Inf. Model., 2023) and confirm permission before publishing results. Raw and processed data are excluded from Git.

Sources: https://ddinter2.scbdd.com/ , https://mecddi.idrblab.net/download

## 3. Repository layout

```text
MedVerify/
├── README.md, requirements.txt, .gitignore
├── config.py                   # shared paths and helpers
├── 01_ddinter_preprocess.py    # raw DDInter -> data/processed/ddinter/
├── 02_mecddi_preprocess.py     # raw MecDDI  -> data/processed/mecddi/
├── 03_link_ddinter.py          # link by drug name -> data/processed/linked/
├── 04_eda.py                   # tables/charts -> data/processed/eda/
├── run_all.py                  # runs steps 01-04
├── retrieval.py                # evidence lookup + Condition A/B prompts + label parser
├── llm.py                      # Ollama client
├── verifier.py                 # Condition C claim-level verification
├── app_b.py                    # Condition B demo (grounded)
├── app_c.py                    # Condition C demo (grounded + verified)
├── run_batch.py                # batch runs for A / B / C with automatic grading
├── eval_verifier.py            # verifier vs hand labels
├── zeroshot_demo/              # Condition A demo
├── data/                       # NOT in Git: data/raw/ (DDInter), processed outputs
└── MecDDI/                     # NOT in Git: the 9 MecDDI CSVs
```

## 4. Setup and running

```powershell
pip install -r requirements.txt
ollama pull <model-tag>          # at least two small models; run one at a time on 16 GB RAM
python run_all.py                # preprocessing (needs data/raw/ and MecDDI/)
streamlit run app_b.py           # Condition B
streamlit run app_c.py           # Condition C
```
Condition A: see `zeroshot_demo/`. Test the batch pipeline without Ollama:
`python run_batch.py --condition B --mode heldout --n 5 --verify --dry-run --out runs/test`

### The verifier (Condition C)
Each answer is split into claims. A claim is (1) not verifiable if the model tagged it "(not in evidence)", (2) flagged if it names an enzyme/transporter no source states, (3) compared with MecDDI/DDInter if it states a mechanism category or severity (or says no record exists), (4) otherwise judged against the evidence text by an LLM at temperature 0, whose verdict must be backed by a quote found in the evidence. The verifier flags; it never rewrites answers.

## 5. Evaluation

Same seed, sample size, temperature and token limit for every run.
```powershell
python run_batch.py --condition A --model <tag> --n 20 --out runs/A
python run_batch.py --condition B --mode pair    --model <tag> --n 20 --out runs/B_pair
python run_batch.py --condition B --mode heldout --model <tag> --n 20 --out runs/B_heldout
python run_batch.py --condition B --mode heldout --model <tag> --judge-model <tag> --verify --out runs/C
python eval_verifier.py runs/C/claims.csv     # after hand-labelling the human_label column
```
Metrics: category exact-match/overlap; severity accuracy and macro-F1; unsupported/contradicted claims per answer; verifier precision/recall/F1 and Cohen's kappa against hand labels (judged against the evidence, not clinical knowledge); abstention on pairs with no record.

## 6. Limitations
- MecDDI is a static snapshot (last updated Feb 2023); only categories, no text or severity.
- Name-based linking loses some drugs; results apply to the overlap (67% of MecDDI pairs).
- Circular reference (see above); severity classes are imbalanced.
- Small local models, small human-graded set, two raters. Not clinically validated.

## 7. Future work
- Consistency-based (SelfCheckGPT-style) verification as a comparison; a third or larger model; automatic correction stage; expert review before any real-world use.

## 8. Literature and references

The references below were consolidated from the supplied literature-review document and spreadsheet. Links for references 19 and 25–27 were checked against the linked publisher or arXiv records on 27 October 2026. Verify all other metadata and findings before formal citation.

### Dataset
1. Xiong, G. et al. (2022). **DDInter: An online drug–drug interaction database towards improving clinical decision-making and patient safety.** https://academic.oup.com/nar/article/50/D1/D1200/6389535
2. Tian, Y. et al. (2025). **DDInter 2.0: an enhanced drug interaction resource with expanded data coverage, new interaction types, and improved user interface.** https://academic.oup.com/nar/article/53/D1/D1356/7740584

### Classical and deep-learning DDI prediction
3. **Graph-based deep learning for drug-drug interaction prediction: a systematic review.** https://pmc.ncbi.nlm.nih.gov/articles/PMC13199350/
4. **SmileGNN: Drug–Drug Interaction Prediction Based on the SMILES and Graph Neural Network.** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8879716/
5. **Enhancing Drug-Drug Interaction Prediction Using Deep Attention Neural Networks (DANN-DDI).** https://www.biorxiv.org/content/10.1101/2021.03.16.435553.full.pdf
6. **A survey of deep learning methods for drug-drug interaction prediction.** https://www.academia.edu/145684345/A_survey_of_deep_learning_methods_for_drug_drug_interaction_prediction
7. **Deep graph contrastive learning model for drug-drug interaction prediction.** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11182529/

### LLMs for DDI identification
8. **Assessing the potential of ChatGPT-4 to accurately identify drug-drug interactions and provide clinical pharmacotherapy recommendations.** https://www.medrxiv.org/content/10.1101/2024.06.29.24309701.full.pdf
9. **Comparative evaluation of artificial intelligence platforms and drug interaction screening databases using real-world patient data.** https://pmc.ncbi.nlm.nih.gov/articles/PMC12712589/
10. **Drug-drug interaction identification using large language models.** https://www.medrxiv.org/content/10.64898/2025.12.03.25341549v1.full
11. **Navigating uncertainty matters: Evaluating large language models for drug-drug interaction identification.** https://pubmed.ncbi.nlm.nih.gov/42640124/
12. **AI platform in the context of drug-drug interactions.** https://www.dovepress.com/evaluating-the-sensitivity-specificity-and-accuracy-of-chatgpt-35-chat-peer-reviewed-fulltext-article-DHPS
13. **ChatGPT as a Digital Pharmacist: A Systematic Review and Meta-Analysis of Drug-Counselling Accuracy.** https://www.medrxiv.org/content/10.1101/2025.08.10.25333396.full.pdf
14. **Evaluating the capability of ChatGPT in predicting drug interactions.** https://pmc.ncbi.nlm.nih.gov/articles/PMC11602951/

### Medical hallucination and reliability
15. Kim, Y. et al. (2025). **Medical Hallucinations in Foundation Models and Their Impact on Healthcare.** https://arxiv.org/abs/2503.05777v1
16. **MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models.** https://arxiv.org/pdf/2502.14302
17. **MedHallBench: A New Benchmark for Assessing Hallucination in Medical Large Language Models.** https://arxiv.org/pdf/2412.18947
18. **Multi-model assurance analysis showing large language models are highly vulnerable to adversarial hallucination attacks during clinical decision support.** https://www.nature.com/articles/s43856-025-01021-3
19. Singhal, K. et al. (2023). **Large language models encode clinical knowledge.** Nature, 620, 172–180. https://www.nature.com/articles/s41586-023-06291-2

### Retrieval and verification
20. Vladika, J. et al. (2026). **Improving Reliability and Explainability of Medical Question Answering Through Atomic Fact-Checking in Retrieval-Augmented Large Language Models: Creation and Validation Study.** Journal of Medical Internet Research, 28, e92090. https://www.jmir.org/2026/1/e92090/
21. **Retrieval-Augmented Generation (RAG) in Healthcare: A Comprehensive Review.** https://www.mdpi.com/2673-2688/6/9/226
22. **Mitigating hallucinations in healthcare AI: a systematic review of evidence-based strategies.** https://pmc.ncbi.nlm.nih.gov/articles/PMC13470931/
23. **Representation Before Retrieval: Structured Patient Artifacts Reduce Hallucination in Clinical AI Systems.** https://www.medrxiv.org/content/10.64898/2026.02.13.26346256v1.full
24. **MEGA-RAG: a retrieval-augmented generation framework with multi-evidence guided answer refinement for mitigating hallucinations of LLMs in public health.** https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12540348/
25. Manakul, P., Liusie, A., & Gales, M. (2023). **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.** https://arxiv.org/abs/2303.08896
26. Niu, C., Wu, Y., et al. (2024). **RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.** https://arxiv.org/abs/2401.00396
27. Lewis, P. et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** https://arxiv.org/abs/2005.11401

## 9. Data and licensing

Raw and processed data are excluded from Git (see `.gitignore`). DDInter is CC BY-NC-SA (verify on the official site); MecDDI states "all rights reserved" - cite the MecDDI paper (J. Chem. Inf. Model., 2023) and confirm permission before publishing results. This is a research prototype, not a clinical tool.
