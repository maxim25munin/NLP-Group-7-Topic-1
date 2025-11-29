# Language identification baselines – refactored layout

This document explains how the refactored baseline evaluation package is organised and
how the pieces collaborate when running experiments.

## Overview

The `scripts/language_id_baselines` package mirrors the behaviour of the original
`evaluate_language_id_baselines.py` script while making responsibilities explicit:

1. **Data ingestion** – read multilingual CoNLL-U files created by
   `scripts/prepare_multilingual_conllu.py` into `SentenceExample` objects.
2. **Baselines** – train/predict with three approaches: rule-based heuristics, character
   n-gram TF–IDF + logistic regression, and XLM-R fine-tuning.
3. **Metrics and reports** – compute accuracy, precision/recall/F1, confusion matrices,
   and representative mistakes.
4. **CLI orchestration** – expose configuration flags and emit JSON summaries when
   requested.

## File structure

```text
scripts/language_id_baselines/
├── __init__.py           # Aggregates exports for consumers
├── __main__.py           # Runtime entry point (python -m scripts.language_id_baselines)
├── cli.py                # Argument parser
├── classical_ml.py       # Character n-gram TF–IDF + logistic regression pipeline
├── data.py               # Sentence iteration and dataset assembly
├── evaluation.py         # Baseline runners and result container
├── metrics.py            # Metrics, confusion matrices, and misclassification sampling
├── reporting.py          # Console rendering of metrics and trade-off summary
├── rule_based.py         # Unicode script/diacritic/keyword heuristic classifier
└── xlmr.py               # XLM-R fine-tuning wrapper with compatibility shims
```

## Module responsibilities and interactions

- **`data.py`** loads the Wikipedia-derived CoNLL-U files into memory. The
  `load_multilingual_dataset` function randomises sentence order and enforces optional
  per-language caps for quicker experiments.
- **`rule_based.py`** implements the handcrafted heuristic model. It analyses Unicode
  scripts, searches for language-specific diacritics, and counts functional keywords.
- **`classical_ml.py`** defines `build_logistic_regression_pipeline`, which pairs a
  character n-gram TF–IDF vectoriser with a multinomial logistic regression classifier.
- **`xlmr.py`** lazily imports PyTorch, Datasets, and Transformers, disables TensorFlow
  backends, and patches legacy `transformers` installations to keep the Trainer working.
  The `XLMRClassifier` encapsulates tokenisation, label mapping, and prediction.
- **`metrics.py`** supplies reusable helpers for computing metrics and extracting
  human-readable misclassification snippets.
- **`evaluation.py`** coordinates baseline training and wraps outputs into
  `BaselineResult` objects. It also emits friendly warnings when the Transformer stack is
  unavailable.
- **`reporting.py`** formats the evaluation outputs, printing per-model sections and a
  qualitative comparison of computational trade-offs.
- **`__main__.py`** wires the CLI, data loading, train/test splits, and all baselines
  together. It also serialises results to JSON when the `--output-report` flag is used.

## Running the baselines

Prerequisites:
- **Classical baselines** – install `scikit-learn`.
- **Transformer baseline** – additionally install `torch`, `datasets`, and
  `transformers` (see `requirements-transformers.txt` if present).

Example command (default settings mirror the previous single-file script):

```bash
python -m scripts.language_id_baselines \
  --data-root data \
  --max-sentences-per-language 2000 \
  --test-size 0.2 \
  --validation-size 0.1 \
  --xlmr-epochs 1
```

To persist metrics to disk, append `--output-report reports/baseline_eval.json`.

## Reproducing Milestone 2 results

The exact runtime outputs for the baseline comparisons are stored in
`reports/Milestone 2 run.md`. To regenerate comparable numbers:

1. Prepare data with `scripts/prepare_multilingual_conllu.py` (or re-use the existing
   `data/` tree if already populated).
2. Install dependencies for the baselines you want to run (classical only or classical +
   Transformer stack).
3. Run the command above; the defaults match the Milestone 2 configuration. If GPU
   resources are unavailable the XLM-R stage will be skipped with a warning, matching the
   behaviour documented in the milestone report.

The printed console output mirrors the Milestone 2 metrics: accuracy summary, precision
/ recall / F1 per language, confusion matrices, and sampled misclassifications.
