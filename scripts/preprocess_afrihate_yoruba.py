"""
Prepare the AfriHate Yoruba dataset for fastText probing experiments.

The goal is to mirror the structure of `data/kazakh_hate_speech_fasttext.csv`
by exporting a CSV with two columns: `text` (input string) and `label`
(target class). The source file `data/afrihate_yor_all_splits.csv` bundles
train/validation/test splits; this script merges them while applying a few
light cleaning steps:

- drops rows missing text or labels
- collapses internal whitespace so each example lives on a single line
- removes duplicate text/label pairs

Example:
    python scripts/preprocess_afrihate_yoruba.py \
        --input data/afrihate_yor_all_splits.csv \
        --output data/afrihate_yoruba_fasttext.csv

After running the script, point fastText experiments to the generated CSV.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge AfriHate Yoruba splits and export a fastText-friendly CSV "
            "containing `text` and `label` columns."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/afrihate_yor_all_splits.csv"),
        help="Path to the combined AfriHate Yoruba CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/afrihate_yoruba_fasttext.csv"),
        help="Destination CSV with cleaned text/label columns.",
    )
    return parser.parse_args()


def normalize_whitespace(text: str) -> str:
    """Collapse internal whitespace to single spaces."""

    return " ".join(text.split())


def read_examples(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open(newline="", encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text = normalize_whitespace((row.get("tweet") or "").strip())
            label = (row.get("label") or "").strip()
            if not text or not label:
                continue
            yield text, label


def deduplicate(examples: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    unique: List[Tuple[str, str]] = []
    for text, label in examples:
        key = (text, label)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def write_clean_csv(output_path: Path, examples: List[Tuple[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "label"])
        for text, label in examples:
            writer.writerow([text, label])


def summarize(examples: Iterable[Tuple[str, str]]) -> str:
    counts = Counter(label for _, label in examples)
    total = sum(counts.values())
    parts = [f"{label}: {count}" for label, count in counts.most_common()]
    return f"{total} examples (" + ", ".join(parts) + ")"


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    rows = list(read_examples(args.input))
    unique_rows = deduplicate(rows)
    write_clean_csv(args.output, unique_rows)

    print("Wrote cleaned dataset to", args.output)
    print("Summary:", summarize(unique_rows))


if __name__ == "__main__":
    main()
