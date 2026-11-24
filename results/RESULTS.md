# MedVerify results (n = 50 fixed test pairs per run)

Scores are from `rescore.py` + `metrics.py` after the format-failure fix (an answer without both label lines is a format failure, never an abstention). Per-run detail (precision/recall/F1, confusion matrices, 95% bootstrap CIs) is in `metrics_reports/`. Reference labels come from MecDDI (category) and DDInter (severity), so these measure agreement with those databases, not clinical correctness; no clinician graded the outputs.

## Models and conditions
Two Ollama models were evaluated: Qwen 2.5 3B and Llama 3.2 3B. A third (BioMistral community build) produced no parseable answer in any run (50/50 format failures: it echoed the prompt), so it is excluded. Judge model for the claim checks: Qwen 2.5 3B for all runs.
- **A**: zero-shot (names only). **B-pair**: the pair's own MecDDI category and DDInter severity are supplied. **B-held-out**: that record is hidden; the model sees each drug's other known interactions. The verifier (Condition C) runs on top of A and B.

## Prediction performance
| Run | Severity acc (95% CI) | Severity macro-F1 | Category exact | Category micro-F1 | Format failures |
|---|---|---|---|---|---|
| Qwen A | 14% (6-24) | 0.14 | 4% | 0.33 | 8/50 |
| Llama A | 6% (0-14) | 0.08 | 0% | 0.22 | 5/50 |
| Qwen B-pair | 96% (90-100) | 0.98 | 88% | 0.89 | 2/50 |
| Llama B-pair | 90% (82-96) | 0.92 | 54% | 0.59 | 3/50 |
| Qwen B-held-out | 12% (4-20) | 0.14 | 6% | 0.30 | 19/50 |
| Llama B-held-out | 22% (12-34) | 0.24 | 12% | 0.27 | 5/50 |

## Hallucination indicators (verifier, per answer)
| Run | Unsupported specific details | Contradicted claims | Answers flagged |
|---|---|---|---|
| Qwen A | 0.86 | 0.50 | 70% |
| Llama A | 1.06 | 1.02 | 88% |
| Qwen B-pair | 0.00 | 0.00 | 0% |
| Llama B-pair | 0.08 | 0.10 | 12% |
| Qwen B-held-out | 0.00 | 1.90 | 82% |
| Llama B-held-out | 0.00 | 1.74 | 80% |

Definitions: *unsupported specific detail* = names an enzyme/transporter (CYP, UGT, P-gp, OATP...) that no retrieved source states; *contradicted* = category/severity claim that conflicts with MecDDI/DDInter. "Unsupported" means no retrieved source states it, not that it is false.

## Abstention on pairs with no record (n = 10)
Correct abstention: Qwen A 0%, B-pair 40%, B-held-out 0%; Llama A 0%, B-pair 10%, B-held-out 10%. Models usually answer even when no record exists.

## Findings
1. Supplying the pair's record raises severity accuracy to 90%+ and removes almost all invented specific details; this is largely copying the record, so it is a sanity check, not a measure of knowledge.
2. With the record, Qwen follows the evidence better than Llama on category (CI 78-96% vs 40-68%, non-overlapping); other model differences are within the confidence intervals.
3. Without the record (A and held-out) both small models perform poorly (below an always-"moderate" severity baseline and a most-common-category baseline); no model is clearly better.
4. Many severity "errors" are honest "Unknown" answers; Qwen's held-out run has 38% format failures (possibly truncation at the 1024-token limit), so its held-out score understates its ability.

## Not yet included
Hand-labelled claim evaluation (Cohen's kappa between the two raters; verifier precision/recall/F1 against human labels).
