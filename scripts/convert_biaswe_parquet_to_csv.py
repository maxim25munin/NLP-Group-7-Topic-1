"""Convert BiaSWE parquet splits to CSV files.

This utility reads the Swedish misogyny detection dataset stored under
``data/biaswe`` (or another directory you specify) and writes CSV copies of each
parquet file. The output files keep the same base name as the parquet source
(e.g. ``train-00000-of-00001.parquet`` -> ``train-00000-of-00001.csv``).

Example usage:
    python scripts/convert_biaswe_parquet_to_csv.py
    python scripts/convert_biaswe_parquet_to_csv.py --input-dir data/biaswe --output-dir data/biaswe/csv

The script requires ``pandas`` with either ``pyarrow`` or ``fastparquet``
installed for reading parquet files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_INPUT_DIR = Path("data/biaswe")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing BiaSWE parquet files (default: data/biaswe).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write CSV files. Defaults to the input directory so the "
            "CSV files sit alongside the parquet sources."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files if they already exist.",
    )

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unrecognised arguments: {' '.join(unknown)}")
    return args


def convert_files(parquet_files: Iterable[Path], output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for parquet_path in sorted(parquet_files):
        output_path = output_dir / f"{parquet_path.stem}.csv"
        if output_path.exists() and not overwrite:
            raise SystemExit(
                f"CSV already exists at {output_path}. Use --overwrite to replace it."
            )

        print(f"Converting {parquet_path} -> {output_path}")
        dataframe = pd.read_parquet(parquet_path)
        dataframe.to_csv(output_path, index=False)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    parquet_files = list(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files found in {input_dir}")

    convert_files(parquet_files, output_dir, args.overwrite)
    print("Conversion complete.")


if __name__ == "__main__":
    main()
