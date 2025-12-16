"""
Convert the Delfi Latvian comment dataset into a fastText-friendly CSV.

This script trims the original `lv-comments-2019.csv` down to two columns
`text` and `label`, mirroring the layout used by
`data/kazakh_hate_speech_fasttext.csv`. The `content` column becomes the text
field, and the `channel_language` column supplies the label.

Example:
    python scripts/preprocess_latvian_comments.py \\
        --input data/lv-comments-2019.csv \\
        --output data/latvian_comments_fasttext.csv
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
            "Reduce lv-comments-2019.csv to `text` and `label` columns for fastText "
            "experiments."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/lv-comments-2019.csv"),
        help="Path to the original Latvian comment dataset CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/latvian_comments_fasttext.csv"),
        help="Destination CSV containing text and label columns.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="nat",
        help=(
            "Filter to a specific channel_language value (e.g., 'nat' or 'rus'). "
            "Set to an empty string to keep all languages."
        ),
    )
    # Using parse_known_args prevents failures when the script is invoked from
    # environments (like Jupyter) that append additional arguments (e.g., the
    # kernel connection file).
    args, _ = parser.parse_known_args(argv)
    return args


def read_examples(path: Path, language: str | None) -> Iterable[Tuple[str, str]]:
    with path.open(newline="", encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            content = (row.get("content") or "").strip()
            label = (row.get("channel_language") or "").strip()
            if not content or not label:
                continue
            if language and label != language:
                continue
            yield content, label


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
    language_filter = args.language or None

    set_csv_field_size_limit()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    rows = list(read_examples(args.input, language_filter))
    unique_rows = deduplicate(rows)
    write_clean_csv(args.output, unique_rows)

    print("Wrote cleaned dataset to", args.output)
    print("Summary:", summarize(unique_rows))


if __name__ == "__main__":
    main()
