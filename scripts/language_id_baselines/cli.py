"""Command-line argument parsing for baseline evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


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
        help="Proportion of the dataset reserved for testing (default: 0.2).",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Optional validation split from the training data (default: 0.1).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=13,
        help="Random seed for data shuffling and train/test splits.",
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
