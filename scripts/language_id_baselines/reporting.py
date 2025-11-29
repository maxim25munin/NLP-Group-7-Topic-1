"""Pretty-print helpers for evaluation outputs."""

from __future__ import annotations

from typing import Sequence

from .metrics import format_classification_report
from .evaluation import BaselineResult


def print_results(result: BaselineResult, labels: Sequence[str]) -> None:
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
    if result.misclassifications:
        print("\nRepresentative misclassifications:")
        for gold_label in labels:
            examples = result.misclassifications.get(gold_label)
            if not examples:
                continue
            print(f"- Gold label {gold_label}:")
            for predicted_label, snippet in examples:
                print(f"    predicted {predicted_label:<10} :: {snippet}")


def compare_models(results: Sequence[BaselineResult]) -> None:
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
