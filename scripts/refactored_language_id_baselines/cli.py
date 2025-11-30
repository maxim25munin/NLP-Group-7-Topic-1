"""Command-line entry point for evaluating multilingual language ID baselines."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

from .data import load_multilingual_dataset
from .evaluation import (
    BaselineResult,
    evaluate_logistic_regression,
    evaluate_rule_based,
    evaluate_xlmr,
    split_train_val_test,
)
from .reporting import compare_models, print_results

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing the language sub-folders with CoNLL-U files.",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Subset of languages to evaluate (default: all languages present in the data root).",
    )
    parser.add_argument(
        "--max-sentences-per-language",
        type=int,
        default=2000,
        help="Optional cap on the number of sentences per language to speed up experiments.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the data set aside for the held-out test set (default: 0.2).",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Proportion of the training data reserved for validation (default: 0.1).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=13,
        help="Random seed used for shuffling and model initialisation.",
    )
    parser.add_argument(
        "--xlmr-output-dir",
        type=Path,
        default=Path("./xlmr_language_id"),
        help="Directory used by the XLM-R Trainer to store checkpoints and logs.",
    )
    parser.add_argument(
        "--xlmr-epochs",
        type=float,
        default=1.0,
        help="Number of fine-tuning epochs for the XLM-R baseline (default: 1).",
    )
    parser.add_argument(
        "--xlmr-batch-size",
        type=int,
        default=8,
        help="Per-device batch size for XLM-R training and evaluation.",
    )
    parser.add_argument(
        "--xlmr-learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for XLM-R fine-tuning.",
    )
    parser.add_argument(
        "--xlmr-weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for XLM-R fine-tuning.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to save the evaluation report as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOGGER.info("Loading multilingual dataset from %s", args.data_root)
    examples = load_multilingual_dataset(
        args.data_root,
        languages=args.languages,
        max_sentences_per_language=args.max_sentences_per_language,
        seed=args.random_seed,
    )
    if not examples:
        raise SystemExit("No data available. Did you run prepare_multilingual_conllu.py?")

    texts = [example.text for example in examples]
    labels = [example.label for example in examples]

    LOGGER.info("Loaded %d sentences across %d languages", len(texts), len(set(labels)))
    if len(set(labels)) < 2:
        raise SystemExit("At least two distinct languages are required for evaluation.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        texts=texts,
        labels=labels,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.random_seed,
    )

    ordered_labels = sorted(set(labels))

    results: List[BaselineResult] = []  # type: ignore[name-defined]
    results.append(evaluate_rule_based(X_train, y_train, X_test, y_test))
    results.append(evaluate_logistic_regression(X_train, y_train, X_test, y_test))

    try:
        results.append(
            evaluate_xlmr(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                model_name="xlm-roberta-base",
                output_dir=str(args.xlmr_output_dir),
                num_train_epochs=args.xlmr_epochs,
                batch_size=args.xlmr_batch_size,
                learning_rate=args.xlmr_learning_rate,
                weight_decay=args.xlmr_weight_decay,
                seed=args.random_seed,
            )
        )
    except RuntimeError as exc:
        LOGGER.warning("Skipping XLM-R baseline: %s", exc)

    for result in results:
        print_results(result, ordered_labels)

    compare_models(results)

    if args.output_report:
        LOGGER.info("Writing report to %s", args.output_report)
        serialisable_results = []
        for result in results:
            serialisable_results.append(
                {
                    "name": result.name,
                    "metrics": result.metrics,
                    "misclassifications": result.misclassifications,
                }
            )
        args.output_report.write_text(json.dumps(serialisable_results, indent=2, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    main()
