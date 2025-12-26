"""
Prepare the Urdu GSM8K dataset for fastText experiments.

The fastText probing notebooks expect a CSV with two columns: `text` (input
string) and `label` (target string). This script converts the GSM8K Urdu file
into that two-column format by mapping the question to `text` and the reasoning
answer to `label` while dropping rows missing either field. Duplicate
question/answer pairs are also removed.

Example:
    python scripts/preprocess_gsm8k_urdu.py \
        --input data/GSM8K_Urdu_all.csv \
        --output data/gsm8k_urdu_fasttext.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple


QUESTION_FIELD = "Question (Urdu)"
REASONING_FIELD = "Reasoning (Urdu)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the Urdu GSM8K CSV to a fastText-friendly CSV containing "
            "`text` (question) and `label` (reasoning answer) columns."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/GSM8K_Urdu_all.csv"),
        help="Path to the original GSM8K Urdu CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gsm8k_urdu_fasttext.csv"),
        help="Destination CSV with `text` and `label` columns.",
    )
    return parser.parse_args()


def normalize_field(value: str | None) -> str:
    """Trim whitespace and collapse internal newlines."""

    if not value:
        return ""
    cleaned = value.strip()
    return " ".join(cleaned.split())


def read_examples(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open(newline="", encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = normalize_field(row.get(QUESTION_FIELD))
            reasoning = normalize_field(row.get(REASONING_FIELD))
            if not question or not reasoning:
                continue
            yield question, reasoning


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


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    rows = list(read_examples(args.input))
    unique_rows = deduplicate(rows)
    write_clean_csv(args.output, unique_rows)

    print("Wrote cleaned dataset to", args.output)
    print("Total rows:", len(unique_rows))


if __name__ == "__main__":
    main()
