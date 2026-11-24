# MedVerify
### A Retrieval-Grounded, Self-Verifying LLM System for Explaining Drug–Drug Interaction Mechanisms

**Course:** AI in Healthcare — Mini Project  
**Team members:** AISHA AIJAZ, SNEHA  
**Last updated:** 16 November 2026

> **Status:** Preprocessing, Conditions A/B/C, the verifier and the batch evaluation are implemented and run on a fixed 50-pair test set for two Ollama models. Hand-labelling of claims (inter-rater kappa, verifier precision/recall) is in progress.

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
├── README.md, requirements.txt, .gitignore, EVALUATION.md
├── config.py                   # shared paths and helpers
├── 01_ddinter_preprocess.py    # raw DDInter -> data/processed/ddinter/
├── 02_mecddi_preprocess.py     # raw MecDDI  -> data/processed/mecddi/
├── 03_link_ddinter.py          # link by drug name -> data/processed/linked/
├── 04_eda.py                   # tables/charts -> data/processed/eda/
├── run_all.py                  # runs steps 01-04
├── retrieval.py                # evidence lookup, Condition A/B prompts, label parser, scoring
├── llm.py                      # Ollama client
├── verifier.py                 # claim-level verification (Condition C)
├── app_a.py / app_b.py / app_c.py   # Streamlit demos: zero-shot / grounded / grounded + verified
├── make_pairs.py, eval_pairs.csv    # fixed evaluation pair list (50 main + 10 no-record)
├── run_batch.py                # run A / B (+ verification) over the pair list, auto-grade
├── rescore.py                  # re-score saved runs (format-failure logic)
├── metrics.py                  # accuracy, P/R/F1, confusion matrices, bootstrap CIs
├── kappa_raters.py             # inter-rater agreement on claim labels
├── eval_verifier.py            # verifier vs hand labels
├── collect_results.py          # copy aggregate results into results/ (+ results.zip)
├── results/                    # committed aggregate results (no per-pair data)
├── data/                       # NOT in Git: data/raw/ (DDInter), processed outputs
├── MecDDI/                     # NOT in Git: the 9 MecDDI CSVs
└── runs/                       # NOT in Git: per-run outputs (answers, claims)
```

## 4. Setup and running

```powershell
pip install -r requirements.txt
ollama pull <model-tag>          # small instruction-tuned models; run one at a time on 16 GB RAM
python run_all.py                # preprocessing (needs data/raw/ and MecDDI/)
streamlit run app_a.py           # Condition A (zero-shot)
streamlit run app_b.py           # Condition B (grounded)
streamlit run app_c.py           # Condition C (grounded + verified)
```
Test the evaluation pipeline without Ollama: `python run_batch.py --condition B --mode heldout --pairs eval_pairs.csv --verify --dry-run --out runs/dry`

### The verifier (Condition C)
Each answer is split into claims. A claim is (1) not verifiable if the model tagged it "(not in evidence)", (2) flagged if it names an enzyme/transporter no source states, (3) compared with MecDDI/DDInter if it states a mechanism category or severity (or says no record exists), (4) otherwise judged against the evidence text by an LLM at temperature 0, whose verdict must be backed by a quote found in the evidence. The verifier flags; it never rewrites answers. The same judge model is used for every generator.

## 5. Evaluation

**Test set:** `eval_pairs.csv` - 50 pairs present in both MecDDI and DDInter (17 Major / 17 Moderate / 16 Minor; all mechanism categories covered; no drug used more than twice) plus 10 pairs with no record (abstention test). Fixed before any model was run.

**Conditions:** A = names only; B-pair = the pair's own record is supplied; B-held-out = the record is hidden and the model sees each drug's other known interactions; C = verification applied on top.

```powershell
$m = "<model-tag>"; $t = $m -replace '[:./]', '_'; $judge = "<judge-tag>"
python run_batch.py --condition A --model $m --pairs eval_pairs.csv --verify --out "runs/A_$t"
python run_batch.py --condition B --mode pair    --model $m --judge-model $judge --pairs eval_pairs.csv --verify --out "runs/Bpair_$t"
python run_batch.py --condition B --mode heldout --model $m --judge-model $judge --pairs eval_pairs.csv --verify --out "runs/Bheld_$t"
python rescore.py runs/A_$t runs/Bpair_$t runs/Bheld_$t     # only needed for runs made before the scoring fix
python metrics.py runs/A_$t runs/Bpair_$t runs/Bheld_$t
python collect_results.py runs/A_$t runs/Bpair_$t runs/Bheld_$t
```
Same temperature and token limit for every run. Metrics: accuracy, per-class precision/recall/F1, macro/micro-F1, 95% bootstrap confidence intervals, confusion matrices, format-failure rate, abstention behaviour. Claim-level labels (`claims.csv`) are labelled independently by both team members against the stored evidence text only (never medical knowledge): `supported / contradicted / not_verifiable`; `kappa_raters.py` gives Cohen's kappa and `eval_verifier.py` the verifier's precision/recall/F1.

### What counts as a hallucination
| Type | Definition | Measured by |
|---|---|---|
| Contradicted claim | States a category/severity that conflicts with MecDDI/DDInter, or says a record does/doesn't exist wrongly | database rule |
| Unsupported specific detail | Names an enzyme/transporter/mechanism term (CYP, UGT, P-gp, OATP...) that no retrieved source states, without tagging it "(not in evidence)" | term rule |
| Unsupported free-text claim | Presented as evidence-based but the evidence does not state it | LLM judge + hand labels |
| Failure to abstain | Definite answer for a pair with no record | automatic |

Not counted: claims honestly tagged "(not in evidence)" and expressions of uncertainty. "Unsupported" means no retrieved source states it, not that it is false. An answer missing the label lines is a **format failure**, reported separately and never counted as an abstention.

## 6. Results (summary)

Full tables and per-run reports are in `results/` (`RESULTS.md`, `comparison.csv`, `metrics_reports/`). Two Ollama models (Qwen 2.5 3B, Llama 3.2 3B), n = 50 pairs per run.

| Run | Severity acc | Category exact | Unsupported details / answer | Contradicted / answer |
|---|---|---|---|---|
| Qwen A (zero-shot) | 14% | 4% | 0.86 | 0.50 |
| Llama A (zero-shot) | 6% | 0% | 1.06 | 1.02 |
| Qwen B-pair | 96% | 88% | 0.00 | 0.00 |
| Llama B-pair | 90% | 54% | 0.08 | 0.10 |
| Qwen B-held-out | 12% | 6% | 0.00 | 1.90 |
| Llama B-held-out | 22% | 12% | 0.00 | 1.74 |

- Supplying the pair's record removes almost all invented specific details and raises accuracy to 90%+ (largely copying the record).
- With the record, Qwen follows the evidence better than Llama on category (non-overlapping confidence intervals); other model differences are within the intervals.
- Without the record, both small models perform poorly and no model is clearly better. Models rarely abstain when no record exists.
- A third model (BioMistral community build) produced no parseable answers and is excluded.

## 7. Limitations
- MecDDI is a static snapshot (last updated Feb 2023) with categories only: no explanatory text and no severity. Neither dataset has mechanism text.
- Name-based linking leaves 2.3% of DDInter drugs unmatched; only 67% of MecDDI pairs have a DDInter severity.
- The reference and the retrieval source are the same databases, so grounded results measure faithfulness to them, not clinical correctness. No clinician graded the outputs.
- The specific-term rule only catches terms on its list; the LLM judge is a small model whose verdicts are evaluated against hand labels.
- Small test set (50 pairs, wide confidence intervals); small local models; a held-out run may be affected by answer truncation (format failures).
- Not clinically validated; outputs must not guide patient care.

## 7b. Future work
- Consistency-based (SelfCheckGPT-style) verification as a comparison; larger or domain models; retry of format-failed answers with a higher token limit; an automatic correction stage; expert review before any real-world use.

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
