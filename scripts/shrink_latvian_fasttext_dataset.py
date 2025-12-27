"""
Downsample the Latvian fastText dataset to roughly 20 MB.

The original `latvian_comments_fasttext_nat_only.csv` file is about 147 MB,
which is too large for quick experimentation. This script randomly samples a
subset of rows (with a reproducible seed) so the output CSV is approximately
20 MB, keeping the original `text`/`label` column layout.

Example:
    python scripts/shrink_latvian_fasttext_dataset.py \
        --input data/latvian_comments_fasttext_nat_only.csv \
        --output data/latvian_comments_fasttext_nat_only_20mb.csv \
        --target-mb 20
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

SAFETY_FRACTION = 0.98


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly downsample latvian_comments_fasttext_nat_only.csv so the "
            "resulting file is roughly 20 MB for fastText experiments."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/latvian_comments_fasttext_nat_only.csv"),
        help="Path to the Latvian fastText CSV containing text and label columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/latvian_comments_fasttext_nat_only_20mb.csv"),
        help="Destination path for the downsampled CSV.",
    )
    parser.add_argument(
        "--target-mb",
        type=float,
        default=20.0,
        help="Approximate target size of the output file in megabytes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to keep sampling reproducible.",
    )
    args, _ = parser.parse_known_args(argv)
    return args


def set_csv_field_size_limit() -> None:
    """Increase the CSV parser field size limit to handle long comments."""

    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int //= 2


def read_rows(path: Path) -> Tuple[List[Tuple[str, str]], List[int]]:
    with path.open(newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)

        required = {"text", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Input CSV is missing required columns: " + ", ".join(sorted(missing))
            )

        rows: List[Tuple[str, str]] = []
        sizes: List[int] = []
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or not label:
                continue
            rows.append((text, label))
            # Roughly account for a comma and newline per row to estimate size.
            sizes.append(len(text) + len(label) + 3)

    return rows, sizes


def select_sample(
    rows: Sequence[Tuple[str, str]],
    sizes: Sequence[int],
    target_bytes: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    if not rows:
        raise ValueError("Input CSV contains no valid rows to sample.")

    header_size = len("text,label\n")
    estimated_full_size = header_size + sum(sizes)
    if estimated_full_size <= target_bytes:
        return list(rows)

    average_row_size = sum(sizes) / len(sizes)
    available_payload = max(target_bytes - header_size, 0)
    target_payload = available_payload * SAFETY_FRACTION
    sample_size = int(target_payload / max(1.0, average_row_size))
    sample_size = max(1, min(sample_size, len(rows)))

    indices = rng.sample(range(len(rows)), sample_size)
    return [rows[i] for i in sorted(indices)]


def write_rows(path: Path, rows: Iterable[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "label"])
        for text, label in rows:
            writer.writerow([text, label])


def enforce_size_limit(
    output_path: Path, rows: List[Tuple[str, str]], target_bytes: int
) -> List[Tuple[str, str]]:
    current_rows = rows
    while len(current_rows) > 1:
        write_rows(output_path, current_rows)
        actual_size = output_path.stat().st_size
        if actual_size <= target_bytes:
            return current_rows

        trim_ratio = target_bytes / actual_size
        trimmed_count = max(1, int(len(current_rows) * trim_ratio * SAFETY_FRACTION))
        if trimmed_count >= len(current_rows):
            trimmed_count = len(current_rows) - 1
        current_rows = current_rows[:trimmed_count]

    write_rows(output_path, current_rows)
    return current_rows


def summarize(rows: Iterable[Tuple[str, str]]) -> str:
    count = 0
    for count, _ in enumerate(rows, start=1):
        pass
    return f"{count} examples"


def main() -> None:
    args = parse_args()
    set_csv_field_size_limit()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    target_bytes = int(args.target_mb * 1024 * 1024)
    rng = random.Random(args.seed)

    rows, sizes = read_rows(args.input)
    sampled_rows = select_sample(rows, sizes, target_bytes, rng)
    final_rows = enforce_size_limit(args.output, sampled_rows, target_bytes)

    final_size = args.output.stat().st_size
    print("Wrote downsampled dataset to", args.output)
    print("Summary:", summarize(final_rows))
    print("Approximate size:", f"{final_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
