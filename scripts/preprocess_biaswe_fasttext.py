"""Prepare BiaSWE dataset for fastText experiments.

This script reads the three CSV splits downloaded from Hugging Face under
``data/biaswe`` (train, validation, and test) and consolidates them into a
single CSV shaped like ``data/kazakh_hate_speech_fasttext.csv`` with columns
``text`` and ``label``. The ``label`` column is derived from a majority vote of
annotators' ``hate_speech`` answers in each row.

Example usage:
    python scripts/preprocess_biaswe_fasttext.py
    python scripts/preprocess_biaswe_fasttext.py --input-dir data/biaswe --output data/biaswe_fasttext.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_INPUT_DIR = Path("data/biaswe")
DEFAULT_OUTPUT = Path("data/biaswe_fasttext.csv")
SPLIT_FILENAMES = (
    "train-00000-of-00001.csv",
    "val-00000-of-00001.csv",
    "test-00000-of-00001.csv",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing BiaSWE CSV splits (default: data/biaswe).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the consolidated fastText-style CSV (default: data/biaswe_fasttext.csv).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unrecognised arguments: {' '.join(unknown)}")
    return args


def extract_majority_hate_label(annotation_blob: str) -> str | None:
    """Return majority vote of ``hate_speech`` labels or ``None`` if unavailable."""

    if not isinstance(annotation_blob, str) or not annotation_blob.strip():
        return None

    try:
        annotations = ast.literal_eval(annotation_blob)
    except (ValueError, SyntaxError):
        return None

    votes: list[str] = []
    for annotator in annotations.values():
        if not isinstance(annotator, dict):
            continue
        value = annotator.get("hate_speech")
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"yes", "no"}:
                votes.append(normalized)

    if not votes:
        return None

    tally = Counter(votes)
    label = "Yes" if tally["yes"] > tally["no"] else "No"
    return label


def load_split(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row.get("text", "").strip()
            label = extract_majority_hate_label(row.get("annotations", ""))
            if text and label:
                rows.append((text, label))
    return rows


def consolidate_splits(input_dir: Path, splits: Iterable[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for filename in splits:
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected split not found: {path}")
        print(f"Loading {path}")
        rows.extend(load_split(path))

    unique_rows = list(dict.fromkeys(rows))
    return unique_rows


def write_fasttext_csv(rows: list[tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output file already exists at {args.output}. Use --overwrite to replace it."
        )

    combined_rows = consolidate_splits(args.input_dir, SPLIT_FILENAMES)
    write_fasttext_csv(combined_rows, args.output)
    print(f"Wrote consolidated fastText CSV to {args.output}")


if __name__ == "__main__":
    main()
