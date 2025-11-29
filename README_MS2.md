# Milestone 2 overview and reproduction guide

This project bundles several baselines for multilingual language identification. The
modular implementation lives under `scripts/language_id_baselines/` and mirrors the
behaviour documented in `reports/Milestone 2 run.md`.

## High-level structure

- **Data preparation** – `scripts/prepare_multilingual_conllu.py` exports Wikipedia
  sentences into language-specific CoNLL-U files under `data/`.
- **Baseline evaluation** – `python -m scripts.language_id_baselines` runs:
  - A Unicode/diacritic/keyword heuristic model.
  - A character n-gram TF–IDF + multinomial logistic regression model.
  - An optional XLM-RoBERTa fine-tuning baseline (requires PyTorch + Transformers).
- **Reports** – console summaries plus optional JSON exports via the `--output-report`
  flag; the Milestone 2 reference output is stored in `reports/Milestone 2 run.md`.

## Reproducing key results

1. Ensure the `data/` tree is populated. If starting from scratch, run
   `python scripts/prepare_multilingual_conllu.py --output-dir data`.
2. Install dependencies:
   - Core baselines: `pip install scikit-learn`.
   - Transformer baseline (optional): `pip install torch datasets transformers`.
3. Execute the evaluation suite (mirrors the Milestone 2 configuration):

   ```bash
   python -m scripts.language_id_baselines \
     --data-root data \
     --max-sentences-per-language 2000 \
     --test-size 0.2 \
     --validation-size 0.1 \
     --xlmr-epochs 1 \
     --output-report reports/baseline_eval.json
   ```

4. Compare the console output or generated JSON to the reference metrics in
   `reports/Milestone 2 run.md`. If the Transformer stack is unavailable the XLM-R stage
   will be skipped automatically, matching the behaviour described in the milestone
   report.

These steps reproduce the headline accuracy, precision/recall/F1 tables, confusion
matrices, and representative misclassifications highlighted in Milestone 2.
