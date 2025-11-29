"""Run all language identification baselines and compare their outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

from sklearn.model_selection import train_test_split

from .cli import parse_args
from .data import load_multilingual_dataset
from .evaluation import (
    BaselineResult,
    evaluate_logistic_regression,
    evaluate_rule_based,
    evaluate_xlmr,
    warn_missing_transformers,
)
from .reporting import compare_models, print_results
from .xlmr import torch

LOGGER = logging.getLogger(__name__)


def run_evaluations(args) -> List[BaselineResult]:
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

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=labels,
    )

    if args.validation_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=args.validation_size,
            random_state=args.random_seed,
            stratify=y_train,
        )
    else:
        X_val, y_val = X_test, y_test

    ordered_labels = sorted(set(labels))

    results: List[BaselineResult] = []
    results.append(evaluate_rule_based(X_train, y_train, X_test, y_test))
    results.append(evaluate_logistic_regression(X_train, y_train, X_test, y_test))

    if torch is not None:
        try:
            results.append(
                evaluate_xlmr(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    args,
                )
            )
        except RuntimeError as exc:
            LOGGER.warning("Skipping XLM-R baseline: %s", exc)
    else:
        warn_missing_transformers()

    for result in results:
        print_results(result, ordered_labels)

    compare_models(results)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_evaluations(args)

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
        Path(args.output_report).write_text(
            json.dumps(serialisable_results, indent=2, ensure_ascii=False), encoding="utf8"
        )


if __name__ == "__main__":
    import sys

    if sys.argv and sys.argv[0].endswith("ipykernel_launcher.py"):
        main([])
    else:
        main()
