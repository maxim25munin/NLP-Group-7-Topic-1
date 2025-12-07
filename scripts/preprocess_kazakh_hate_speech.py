"""
Prepare the Kazakh hate-speech dataset for fastText probing experiments.

The notebook `scripts/fasttext_probing.ipynb` expects a CSV with two columns:
`text` (input string) and `label` (target class). This script converts the
original CSV in `data/Kazakh Language Dataset for Hate Speech Detection on Social Media Text/`
into that two-column format while cleaning a few artifacts:

- drops rows that are missing text or labels
- optionally uses the stemmed text supplied in the source file
- removes trailing English-only tags (e.g., ``propaganda``, ``radicalization``)
  that appear at the end of some messages to avoid label leakage
- collapses duplicate rows after cleaning

Example:
    python scripts/preprocess_kazakh_hate_speech.py \
        --input "data/Kazakh Language Dataset for Hate Speech Detection on Social Media Text/Data.csv" \
        --output data/kazakh_hate_speech_fasttext.csv

After running the script, point `KAZSANDRA_LOCAL_PATH` in
`scripts/fasttext_probing.ipynb` to the generated CSV to probe fastText on the
hate-speech labels.
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
            "Clean the Kazakh hate-speech dataset and export a fastText-friendly CSV "
            "containing `text` and `label` columns."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/Kazakh Language Dataset for Hate Speech Detection on Social Media Text/Data.csv"
        ),
        help="Path to the original dataset CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/kazakh_hate_speech_fasttext.csv"),
        help="Destination CSV with cleaned text/label columns.",
    )
    parser.add_argument(
        "--use-stemmed",
        action="store_true",
        help="Use the stemmed message column instead of the raw message text.",
    )
    return parser.parse_args()


def strip_trailing_english_tags(text: str) -> str:
    """Remove trailing ASCII-only tokens that act like annotation tags."""

    tokens: List[str] = text.strip().split()
    while tokens and tokens[-1].isascii() and tokens[-1].isalpha():
        tokens.pop()
    return " ".join(tokens)


def read_examples(path: Path, use_stemmed: bool = False) -> Iterable[Tuple[str, str]]:
    text_field = "message_stemmed" if use_stemmed else "message"
    with path.open(newline="", encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            raw_text = (row.get(text_field) or "").strip()
            label = (row.get("label_name") or "").strip()
            if not raw_text or not label:
                continue
            yield strip_trailing_english_tags(raw_text), label


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

    rows = list(read_examples(args.input, use_stemmed=args.use_stemmed))
    unique_rows = deduplicate(rows)
    write_clean_csv(args.output, unique_rows)

    print("Wrote cleaned dataset to", args.output)
    print("Summary:", summarize(unique_rows))


if __name__ == "__main__":
    main()
