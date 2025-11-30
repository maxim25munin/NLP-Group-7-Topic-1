# Language Identification Baselines — Refactored Architecture

This document explains how the refactored multilingual language identification baselines are organised inside `scripts/refactored_language_id_baselines/`. The goal of the refactor is to make each concern—data loading, modelling, evaluation, and reporting—explicit and reusable without changing the original `scripts/evaluate_language_id_baselines.py` file.

## Directory layout

```
scripts/
├─ evaluate_language_id_baselines.py      # Original monolithic script (unchanged)
└─ refactored_language_id_baselines/
   ├─ __init__.py                         # Exposes the CLI entry point
   ├─ cli.py                              # Argument parsing and orchestration
   ├─ data.py                             # Dataset ingestion helpers
   ├─ evaluation.py                       # Baseline runners and data splitting
   ├─ logistic_regression.py              # Character n-gram TF–IDF pipeline
   ├─ metrics.py                          # Shared metrics and misclassification utilities
   ├─ reporting.py                        # Console + visual reporting helpers
   └─ rule_based.py                       # Hand-crafted heuristic classifier
```

The XLM-RoBERTa fine-tuning logic sits in `xlmr.py` next to the other modules. It is optional and only imported when running the deep-learning baseline.

## Module responsibilities

- **`cli.py`** — Provides the command-line interface, wires together data loading, model evaluation, reporting, and optional report export. It mirrors the behaviour of the original script while delegating work to specialised modules.
- **`data.py`** — Handles reading CoNLL-U files produced by `prepare_multilingual_conllu_stanza.py` and yields shuffled `SentenceExample` records ready for downstream processing.
- **`evaluation.py`** — Implements one function per baseline (`evaluate_rule_based`, `evaluate_logistic_regression`, `evaluate_xlmr`) and a `split_train_val_test` helper that maintains stratification.
- **`rule_based.py`** — Contains the heuristic language detector, including Unicode script checks, keyword/diacritic cues, and frequency-based priors.
- **`logistic_regression.py`** — Builds the scikit-learn pipeline that feeds character n-gram TF–IDF features into a multinomial logistic regression classifier.
- **`xlmr.py`** — Loads the Hugging Face transformers stack on demand, applies compatibility shims, and fine-tunes an XLM-RoBERTa sequence-classification head.
- **`metrics.py`** — Centralises metric computation (accuracy, precision/recall/F1, confusion matrix) and the extraction of representative misclassifications.
- **`reporting.py`** — Formats metrics for console output, renders prettified confusion matrices into `reports/`, and prints qualitative comparisons across baselines.

## Execution flow

1. `cli.main` parses arguments, logs configuration, and loads the multilingual dataset with `data.load_multilingual_dataset`.
2. `evaluation.split_train_val_test` partitions the corpus into training, validation, and test splits using stratified sampling.
3. Each baseline is executed via its `evaluate_*` function, which trains the corresponding model class or pipeline and returns a `BaselineResult` bundle.
4. `reporting.print_results` and `reporting.compare_models` display quantitative metrics, confusion matrices, and representative misclassifications, saving figures to `reports/`.
5. When `--output-report` is provided, the CLI serialises the full results (metrics + misclassification samples) as JSON for downstream analysis.

## Dependency notes

- The rule-based baseline depends on **spaCy** language resources for stopword lists.
- The classical baseline requires **scikit-learn** for vectorisation, training, and metrics.
- The deep-learning baseline requires **PyTorch**, **datasets**, and **transformers**; the module sets `USE_TF=0` and `TRANSFORMERS_NO_TF=1` to avoid TensorFlow initialisation.
- Confusion-matrix visualisation uses **matplotlib**, **pandas**, and **pretty_confusion_matrix**.

If optional dependencies are missing, only the corresponding baseline is skipped while the rest of the pipeline remains usable.

## Extending the baselines

- Add new heuristics by expanding the constants and scoring logic in `rule_based.py`.
- Swap or tune the character n-gram features by editing `build_logistic_regression_pipeline` in `logistic_regression.py`.
- Experiment with different transformer backbones or training schedules by adjusting the defaults in `xlmr.py` and passing alternative CLI flags (e.g., `--xlmr-epochs`).

## Data expectations and provenance

The refactor assumes the multilingual dataset prepared by `scripts/prepare_multilingual_conllu_stanza.py` resides under `data/` with one subdirectory per language containing CoNLL-U files. The runtime behaviour and expected outcomes mirror the Milestone 2 run captured in `reports/Milestone 2 run.md`.
