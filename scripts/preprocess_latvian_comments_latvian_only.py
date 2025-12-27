"""
Filter the fastText-ready Latvian comments dataset to Latvian-only entries.

This script trims `data/latvian_comments_fasttext.csv` down further by removing
entries labelled `rus`, keeping only comments marked with the `nat` label so the
resulting file contains only Latvian-language comments.

Example:
    python scripts/preprocess_latvian_comments_latvian_only.py \
        --input data/latvian_comments_fasttext.csv \
        --output data/latvian_comments_fasttext_nat_only.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter latvian_comments_fasttext.csv to include only Latvian (nat) "
            "entries for fastText experiments."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/latvian_comments_fasttext.csv"),
        help="Path to the fastText-ready Latvian comments CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/latvian_comments_fasttext_nat_only.csv"),
        help="Destination CSV containing only Latvian-language comments.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="nat",
        help="Label value to keep; defaults to 'nat' (Latvian).",
    )
    args, _ = parser.parse_known_args(argv)
    return args


def read_examples(path: Path, keep_label: str) -> Iterable[Tuple[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)

        required = {"text", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or not label:
                continue
            if label != keep_label:
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


def set_csv_field_size_limit() -> None:
    """Increase the CSV parser field size limit to handle long comments."""

    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int //= 2


def main() -> None:
    args = parse_args()

    set_csv_field_size_limit()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    rows = list(read_examples(args.input, args.label))
    unique_rows = deduplicate(rows)
    write_clean_csv(args.output, unique_rows)

    print("Wrote Latvian-only dataset to", args.output)
    print("Summary:", summarize(unique_rows))


if __name__ == "__main__":
    main()
