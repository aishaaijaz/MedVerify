# MedVerify evaluation protocol

Fixed test set: `eval_pairs.csv` = 50 main pairs (in MecDDI and DDInter; severity 17 Major / 17 Moderate / 16 Minor; every mechanism category covered; no drug used more than twice) + 10 no-record pairs (abstention test). Commit it and never change it.

## 0. Smoke test (no Ollama)
`python run_batch.py --condition B --mode heldout --pairs eval_pairs.csv --verify --dry-run --out runs/dry`
The dry run is also a trivial baseline (most common category/severity), the bar the models must beat.

## 1. Runs (repeat for each of the 2 models; same temperature/max tokens everywhere)
```powershell
python run_batch.py --condition A --model M --pairs eval_pairs.csv --verify --out runs/A_M            # zero-shot (+ rule checks, no judge)
python run_batch.py --condition B --mode pair --model M --pairs eval_pairs.csv --out runs/Bpair_M      # grounded, record given
python run_batch.py --condition B --mode heldout --model M --judge-model M --pairs eval_pairs.csv --verify --out runs/C_M   # grounded held-out + verification
```
`runs/C_M` also gives the Condition B held-out scores (the verifier only adds flags). Test with a 3-row copy of the pairs file first.

## 2. Automatic metrics (`summary.txt` in each run)
Category exact/overlap and per-category recall; severity accuracy, per-class accuracy and macro-F1; abstention on the 10 no-record pairs; wrong abstention on pairs with a record; verifier flag rates and unsupported-detail/contradicted claims per answer.

## 3. Human labelling and kappa (runs/C_M/claims.csv)
1. Each member fills their own column (`rater1` / `rater2`) independently, for ALL claims: `supported | contradicted | not_verifiable`.
2. Rule: judge only against the retrieved evidence text ([E1]/[E2] shown in the app, or the `reason` column), never your own medical knowledge. A claim the evidence does not state = `not_verifiable`. "No record" does not support claims that two drugs do/don't interact.
3. `python kappa_raters.py runs/C_M/claims.csv` -> agreement, Cohen's kappa, disagreements; discuss and fill `human_label` in `claims_adjudicated.csv`.
4. `python eval_verifier.py runs/C_M/claims_adjudicated.csv` -> verifier precision/recall/F1 for problem detection, 3-class kappa vs humans, accuracy by layer. Also report the judge's false-support rate (human not_verifiable/contradicted but verifier supported).

## 4. What to report
Table per model: A vs B-pair vs B-held-out (+ trivial baseline) on category and severity; unsupported-detail claims per answer A vs B; verifier metrics; rater kappa; limitations (circular reference, no clinician grading, small judge model).

## 5. Models and classical metrics (added)
Run section 1 once per model tag (e.g. a Qwen 3B, a Llama 3B and, optionally, a BioMistral 7B community build; check the exact tag with `ollama list` / the Ollama site). A 7B model is slower and may not follow the output format: report its parseable-label rate.
Then, for every run folder: `python metrics.py runs/A_M runs/Bpair_M runs/C_M` -> accuracy, per-class precision/recall/F1, macro-F1, 95% bootstrap CIs, confusion matrices (PNG) and `runs/comparison.csv`.
Larger automatic sample (no hand labels needed): drop `--pairs` and use `--n 40` (120 pairs, stratified by severity) for A and B, then run metrics.py on those folders. Human claim-labelling stays on the 50 fixed pairs.
`claims.csv` now has an `evidence` column: the proof text each claim must be labelled against.
