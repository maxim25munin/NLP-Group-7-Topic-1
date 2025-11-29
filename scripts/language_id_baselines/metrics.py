"""Metrics and error analysis helpers for baseline evaluation."""

from __future__ import annotations

import textwrap
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

try:  # Optional dependency used by metrics reporting
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "scikit-learn is required to run the baseline evaluation script."
        "Please install it via `pip install scikit-learn`."
    ) from exc


def collect_misclassifications(
    texts: Sequence[str],
    gold: Sequence[str],
    predicted: Sequence[str],
    max_per_label: int = 2,
) -> Dict[str, List[Tuple[str, str]]]:
    """Return representative misclassifications grouped by gold label."""

    collected: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for text, gold_label, pred_label in zip(texts, gold, predicted):
        if gold_label == pred_label:
            continue
        if len(collected[gold_label]) >= max_per_label:
            continue
        snippet = textwrap.shorten(text, width=180, placeholder="…")
        collected[gold_label].append((pred_label, snippet))
    return collected


def summarise_metrics(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, object]:
    report = classification_report(gold, predicted, labels=labels, zero_division=0, output_dict=True)
    accuracy = accuracy_score(gold, predicted)
    conf_mat = confusion_matrix(gold, predicted, labels=labels)
    return {"accuracy": accuracy, "classification_report": report, "confusion_matrix": conf_mat.tolist()}


def format_classification_report(report: Dict[str, Dict[str, float]]) -> str:
    headers = ["precision", "recall", "f1-score", "support"]
    lines = ["label           precision  recall  f1-score  support"]
    for label, metrics in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1-score", 0.0)
        support = int(metrics.get("support", 0))
        lines.append(f"{label:<15} {precision:>9.3f} {recall:>7.3f} {f1:>8.3f} {support:>8}")
    global_metrics = report.get("macro avg")
    if global_metrics:
        lines.append(
            f"{'macro avg':<15} {global_metrics.get('precision', 0.0):>9.3f}"
            f" {global_metrics.get('recall', 0.0):>7.3f}"
            f" {global_metrics.get('f1-score', 0.0):>8.3f}"
            f" {int(global_metrics.get('support', 0)):>8}"
        )
    weighted_metrics = report.get("weighted avg")
    if weighted_metrics:
        lines.append(
            f"{'weighted avg':<15} {weighted_metrics.get('precision', 0.0):>9.3f}"
            f" {weighted_metrics.get('recall', 0.0):>7.3f}"
            f" {weighted_metrics.get('f1-score', 0.0):>8.3f}"
            f" {int(weighted_metrics.get('support', 0)):>8}"
        )
    return "\n".join(lines)
