"""Reporting helpers for model metrics and qualitative analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from pretty_confusion_matrix import pp_matrix

from .metrics import format_classification_report

LOGGER = logging.getLogger(__name__)


def render_pretty_confusion_matrix(confusion: Sequence[Sequence[int]], labels: Sequence[str], title: str) -> Path:
    """Render and save a prettified confusion matrix heatmap."""

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in title).strip("_")
    slug = slug or "baseline"
    output_path = reports_dir / f"confusion_matrix_{slug}.png"

    df_cm = pd.DataFrame(confusion, index=labels, columns=labels)
    plt.figure(figsize=(8, 6))
    pp_matrix(df_cm, cmap="PuRd", figsize=(8, 6), fz=7)
    ax = plt.gca()
    white = to_rgba("white")
    for text in ax.texts:
        if to_rgba(text.get_color()) == white:
            text.set_color("black")
    plt.title(f"Confusion matrix: {title}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def print_results(result, labels: Sequence[str]) -> None:
    accuracy = result.metrics.get("accuracy", 0.0)
    print("\n" + "=" * 80)
    print(f"Results for {result.name}")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.4f}")
    report = result.metrics.get("classification_report")
    if isinstance(report, dict):
        print("\nClassification report:")
        print(format_classification_report(report))
    confusion = result.metrics.get("confusion_matrix")
    if isinstance(confusion, list):
        print("\nConfusion matrix (rows = gold, columns = predicted):")
        header = "{:<12}".format(" ") + " ".join(f"{label:<10}" for label in labels)
        print(header)
        for label, row in zip(labels, confusion):
            values = " ".join(f"{value:<10}" for value in row)
            print(f"{label:<12}{values}")
        pretty_path = render_pretty_confusion_matrix(confusion, labels, result.name)
        print(f"Saved prettified confusion matrix to {pretty_path}")
    if result.misclassifications:
        print("\nRepresentative misclassifications:")
        for gold_label in labels:
            examples = result.misclassifications.get(gold_label)
            if not examples:
                continue
            print(f"- Gold label {gold_label}:")
            for predicted_label, snippet in examples:
                print(f"    predicted {predicted_label:<10} :: {snippet}")


def compare_models(results) -> None:
    print("\n" + "#" * 80)
    print("Model comparison and qualitative notes")
    print("#" * 80)
    print("\nAccuracy overview:")
    for result in results:
        accuracy = result.metrics.get("accuracy", 0.0)
        print(f"- {result.name:<35} {accuracy:.4f}")
    print(
        "\nOperational trade-offs:\n"
        "* Rule-based heuristics: extremely cheap to run and interpretable, but they\n"
        "  depend on linguistic expertise to curate the cues and struggle to scale to\n"
        "  language families that share scripts or borrow vocabulary.\n"
        "* Char n-gram logistic regression: inexpensive to train and requires little\n"
        "  feature engineering. It scales to new languages as long as labelled data\n"
        "  is available, though it cannot leverage sub-word semantics beyond n-gram\n"
        "  statistics.\n"
        "* XLM-R fine-tuning: delivers the strongest accuracy in most scenarios and\n"
        "  generalises across scripts thanks to multilingual pretraining. The trade-off\n"
        "  is substantially higher computational cost and a dependency on GPU\n"
        "  resources, which may be prohibitive for rapid experimentation.\n"
    )
