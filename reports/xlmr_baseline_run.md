# Milestone 2 — Summary of Code Changes and Baseline Run

This document summarizes my changes made to the baseline script and the exact command used to generate the file ```baseline_results_full.json```.

## 1. Summary of Changes

### 1.1 Added TYPE_CHECKING import

To avoid slow or failing imports when `torch` is not installed, and to prevent unnecessary warnings, the following pattern was introduced:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Trainer, TrainingArguments
```

This ensures:
- Heavy `transformers` imports are only used for static type checking.
- Rule-based and classical ML baselines can run without deep learning dependencies.
- No runtime slowdown or import errors if PyTorch is absent.

### 1.2 Updated TrainingArguments parameter

The previous argument:

```python
evaluation_strategy = `epoch`
```

was replaced with:

```python
eval_strategy = `epoch`
```

This change is required for compatibility with my installed version of the `transformers` module (version `4.57.1`), where `evaluation_strategy` is not the expected keyword argument.

## 2. Prompt Used for baseline_results_full.json

The following command was used to generate the full results JSON:

```bash
python scripts/evaluate_language_id_baselines.py /
  --data-root data /
  --max-sentences-per-language 2000 /
  --test-size 0.2 /
  --validation-size 0.1 /
  --xlmr-epochs 2 /
  --xlmr-batch-size 8 /
  --xlmr-learning-rate 2e-5 /
  --xlmr-weight-decay 0.01 /
  --xlmr-output-dir xlmr_language_id /
  --output-report reports/baseline_results_full.json
```

This prompt runs all baselines (rule-based, logistic regression, and XLM-R fine-tuning) and stores the output in:

```reports/baseline_results_full.json```