"""Evaluation runners for the different baselines."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .classical_ml import build_logistic_regression_pipeline
from .metrics import collect_misclassifications, summarise_metrics
from .rule_based import RuleBasedIdentifier
from .xlmr import XLMRClassifier, TRANSFORMERS_IMPORT_ERROR, torch

LOGGER = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    name: str
    metrics: Dict[str, object]
    misclassifications: Dict[str, List[Tuple[str, str]]]


def evaluate_rule_based(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
) -> BaselineResult:
    LOGGER.info("Training rule-based baseline")
    model = RuleBasedIdentifier()
    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)
    labels = sorted(set(train_labels) | set(test_labels))
    metrics = summarise_metrics(test_labels, predictions, labels)
    misclassifications = collect_misclassifications(test_texts, test_labels, predictions)
    return BaselineResult("Rule-based heuristics", metrics, misclassifications)


def evaluate_logistic_regression(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
) -> BaselineResult:
    LOGGER.info("Training character n-gram logistic regression")
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)
    labels = sorted(set(train_labels) | set(test_labels))
    metrics = summarise_metrics(test_labels, predictions, labels)
    misclassifications = collect_misclassifications(test_texts, test_labels, predictions)
    return BaselineResult("Char n-gram logistic regression", metrics, misclassifications)


def evaluate_xlmr(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    val_texts: Sequence[str],
    val_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
    args: argparse.Namespace,
) -> BaselineResult:
    if torch is None:
        raise RuntimeError(
            "PyTorch and transformers are required for the XLM-R baseline. "
            "Install the optional dependencies with `pip install -r requirements-transformers.txt`."
        )
    LOGGER.info("Fine-tuning XLM-R model")
    classifier = XLMRClassifier(
        model_name="xlm-roberta-base",
        output_dir=args.xlmr_output_dir,
        num_train_epochs=args.xlmr_epochs,
        batch_size=args.xlmr_batch_size,
        learning_rate=args.xlmr_learning_rate,
        weight_decay=args.xlmr_weight_decay,
        seed=args.random_seed,
    )
    classifier.fit(train_texts, train_labels, val_texts, val_labels)
    predictions = classifier.predict(test_texts)
    labels = sorted(set(train_labels) | set(test_labels))
    metrics = summarise_metrics(test_labels, predictions, labels)
    misclassifications = collect_misclassifications(test_texts, test_labels, predictions)
    return BaselineResult("XLM-R fine-tuning", metrics, misclassifications)


def warn_missing_transformers() -> None:
    if torch is None and TRANSFORMERS_IMPORT_ERROR:
        LOGGER.warning(
            "PyTorch/transformers not available; skipping XLM-R baseline. "
            "Install the optional dependencies with `pip install -r requirements-transformers.txt`.",
            exc_info=TRANSFORMERS_IMPORT_ERROR,
        )
    elif torch is None:
        LOGGER.warning(
            "PyTorch/transformers not available; skipping XLM-R baseline. "
            "Install the optional dependencies with `pip install -r requirements-transformers.txt`."
        )
