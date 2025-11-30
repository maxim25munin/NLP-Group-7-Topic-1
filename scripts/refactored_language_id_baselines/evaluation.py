"""Evaluation entry points for the three language identification baselines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

from sklearn.model_selection import train_test_split

from .logistic_regression import build_logistic_regression_pipeline
from .metrics import collect_misclassifications, summarise_metrics
from .rule_based import RuleBasedIdentifier
from .xlmr import XLMRClassifier, transformers_available

LOGGER = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    name: str
    metrics: Dict[str, object]
    misclassifications: Dict[str, List[tuple[str, str]]]


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
    *,
    model_name: str,
    output_dir: str,
    num_train_epochs: float,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> BaselineResult:
    if not transformers_available():
        raise RuntimeError(
            "PyTorch and transformers are required for the XLM-R baseline. "
            "Install the optional dependencies with `pip install -r requirements-transformers.txt`."
        )
    LOGGER.info("Fine-tuning XLM-R model")
    classifier = XLMRClassifier(
        model_name=model_name,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    classifier.fit(train_texts, train_labels, val_texts, val_labels)
    predictions = classifier.predict(test_texts)
    labels = sorted(set(train_labels) | set(test_labels))
    metrics = summarise_metrics(test_labels, predictions, labels)
    misclassifications = collect_misclassifications(test_texts, test_labels, predictions)
    return BaselineResult("XLM-R fine-tuning", metrics, misclassifications)


def split_train_val_test(
    texts: Sequence[str],
    labels: Sequence[str],
    test_size: float,
    validation_size: float,
    seed: int,
):
    """Return train/validation/test splits while preserving label stratification."""

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    if validation_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=validation_size,
            random_state=seed,
            stratify=y_train,
        )
    else:
        X_val, y_val = X_test, y_test
    return X_train, X_val, X_test, y_train, y_val, y_test
