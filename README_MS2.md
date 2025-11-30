# Milestone 2: Multilingual Language Identification Baselines

This milestone refactors the original monolithic evaluator (`scripts/evaluate_language_id_baselines.py`) into a modular package under `scripts/refactored_language_id_baselines/` while preserving behaviour. The refactor clarifies how data flows through the heuristics, classical ML, and transformer-based baselines.

## File structure

```
scripts/
├─ evaluate_language_id_baselines.py          # Original script (unchanged)
└─ refactored_language_id_baselines/
   ├─ cli.py                                  # CLI + orchestration
   ├─ data.py                                 # Dataset ingestion
   ├─ evaluation.py                           # Baseline runners + splits
   ├─ logistic_regression.py                  # Char n-gram TF–IDF baseline
   ├─ metrics.py                              # Metric computation helpers
   ├─ reporting.py                            # Console + confusion-matrix output
   ├─ rule_based.py                           # Heuristic language identifier
   └─ xlmr.py                                 # XLM-RoBERTa fine-tuning
```

## How to reproduce the key Milestone 2 results

1. **Prepare data**: Ensure the multilingual Wikipedia snippets produced by `scripts/prepare_multilingual_conllu_stanza.py` are available under `data/<language>/*.conllu`.
2. **Install dependencies**:
   - Core: `pip install spacy scikit-learn matplotlib pandas pretty_confusion_matrix`
   - Transformers baseline: `pip install torch datasets transformers packaging`
3. **Run the refactored evaluator** (mirrors the Milestone 2 configuration):
   ```bash
   python -m scripts.refactored_language_id_baselines.cli \
     --data-root data \
     --max-sentences-per-language 2000 \
     --test-size 0.2 \
     --validation-size 0.1 \
     --random-seed 13 \
     --xlmr-epochs 1 \
     --xlmr-batch-size 8 \
     --xlmr-learning-rate 2e-5 \
     --xlmr-weight-decay 0.01
   ```
4. **Inspect outputs**:
   - Console logs report accuracy, precision/recall/F1, confusion matrices, and sampled misclassifications for each baseline.
   - Prettified confusion matrices are saved under `reports/` (e.g., `reports/confusion_matrix_xlmr_fine_tuning.png`).
   - To archive results, add `--output-report reports/milestone2_refactor.json`.
5. **Compare with Milestone 2 reference**: The expected runtime profile and qualitative outcomes are summarised in `reports/Milestone 2 run.md`.

## Notes

- The refactor keeps the original script intact for backwards compatibility.
- Each module is designed for focused reuse (e.g., swap in a different vectoriser without touching the CLI).
- Optional baselines are skipped automatically if their dependencies are not installed; install missing packages to enable them.

For a deeper module-by-module explanation, see `docs/language_id_baselines_refactor.md`.
