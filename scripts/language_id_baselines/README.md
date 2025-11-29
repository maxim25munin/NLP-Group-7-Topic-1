# Language identification baselines (refactored)

This package breaks down the original monolithic `evaluate_language_id_baselines.py` script
into focused modules that handle data loading, baseline implementations, evaluation
reporting, and the command-line interface. Run the suite with:

```bash
python -m scripts.language_id_baselines --data-root data
```

## Module structure

```text
scripts/language_id_baselines/
├── __init__.py                # Public exports for quick imports
├── __main__.py                # Entry point invoked via `python -m ...`
├── cli.py                     # Argument parsing
├── classical_ml.py            # Character n-gram TF–IDF + logistic regression
├── data.py                    # Dataset iteration helpers
├── evaluation.py              # Baseline runners and result container
├── metrics.py                 # Metrics, confusion matrix, misclassification samples
├── reporting.py               # Pretty-printers and comparison summary
├── rule_based.py              # Unicode/keyword heuristic identifier
└── xlmr.py                    # XLM-RoBERTa fine-tuning baseline
```

### Responsibilities

- **`data.py`** – Converts the CoNLL-U exports from `prepare_multilingual_conllu.py` into
  `SentenceExample` objects ready for train/test splits.
- **`rule_based.py`** – Encapsulates the Unicode script heuristics, diacritic cues, and
  keyword spotting logic.
- **`classical_ml.py`** – Provides the TF–IDF vectoriser and multinomial logistic
  regression pipeline.
- **`xlmr.py`** – Handles lazy imports for PyTorch/Transformers and wraps the
  fine-tuning workflow for `xlm-roberta-base`.
- **`metrics.py`** – Collects evaluation metrics, confusion matrices, and human-readable
  classification reports.
- **`evaluation.py`** – Orchestrates training/prediction for each baseline and packages
  their outputs as `BaselineResult` objects.
- **`reporting.py`** – Prints detailed per-model results and a short trade-off
  comparison.
- **`cli.py`** – Defines the command-line switches used throughout the scripts.
- **`__main__.py`** – Wires everything together for a single entry point.

### Notes on dependencies

- The classical baseline requires **scikit-learn**. Install with `pip install scikit-learn`.
- The XLM-R baseline additionally requires **PyTorch**, **datasets**, and
  **transformers**. Optional compatibility shims keep the Trainer working on older
  versions.
