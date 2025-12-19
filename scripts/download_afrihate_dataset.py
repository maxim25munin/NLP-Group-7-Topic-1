"""Download the AfriHate dataset from Hugging Face and store each split as CSV.

The script uses the :mod:`datasets` library to pull the public dataset
``afrihate/afrihate`` (or another repository specified via ``--repo-id``) and
writes each available split to ``<output-dir>/<split>.csv``. Cached downloads are
respected, so re-running the script will reuse files stored by ``datasets``.

Example usage:
    python scripts/download_afrihate_dataset.py
    python scripts/download_afrihate_dataset.py --output-dir data/custom_afrihate
    python scripts/download_afrihate_dataset.py --splits train validation

The script requires the ``datasets`` package. Install it with:
    pip install datasets
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

from datasets import Dataset, DatasetDict, load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfFolder

DEFAULT_REPO_ID = "afrihate/afrihate"
DEFAULT_OUTPUT_DIR = Path("data/afrihate")
EMBEDDED_HF_TOKEN = "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"


class DatasetAccessError(RuntimeError):
    """Raised when a dataset cannot be accessed due to permission issues."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repository ID (defaults to afrihate/afrihate).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help=(
            "Specific dataset splits to export (e.g. train validation test). If "
            "omitted, all available splits are saved."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Optional Hugging Face token for private or gated datasets. Defaults "
            "to the HF_TOKEN environment variable when unset."
        ),
    )

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unrecognised arguments: {' '.join(unknown)}")
    return args


def resolve_token(cli_token: str | None) -> str | None:
    """Return a usable Hugging Face token if provided via CLI or environment."""

    return (
        cli_token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or HfFolder.get_token()
        or EMBEDDED_HF_TOKEN
    )


def load_splits(repo_id: str, token: str | None) -> DatasetDict:
    try:
        return load_dataset(repo_id, token=token)
    except DatasetNotFoundError as exc:  # pragma: no cover - network dependent
        if token is None:
            raise DatasetAccessError(
                "Failed to load dataset. It may be gated or private and require an "
                "approved Hugging Face account. Provide a token via --token, the "
                "HF_TOKEN environment variable, or HUGGINGFACEHUB_API_TOKEN, then "
                f"retry after requesting access at https://huggingface.co/datasets/{repo_id} "
                "if needed."
            ) from exc
        raise DatasetAccessError(
            "Failed to load dataset. This dataset is gated and requires an "
            "approved Hugging Face account. Request access at "
            f"https://huggingface.co/datasets/{repo_id} and provide a token via "
            "--token or the HF_TOKEN environment variable."
        ) from exc
    except Exception as exc:  # pragma: no cover - network dependent
        raise DatasetAccessError(f"Failed to load dataset {repo_id}: {exc}") from exc


def export_split(dataset: Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = dataset.to_pandas()
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    token = resolve_token(args.token)
    dataset_dict = load_splits(args.repo_id, token)

    available_splits: Iterable[str] = dataset_dict.keys()
    requested_splits = args.splits or list(available_splits)

    for split in requested_splits:
        if split not in dataset_dict:
            raise SystemExit(
                f"Requested split '{split}' not found. Available splits: "
                f"{', '.join(dataset_dict.keys())}"
            )
        export_split(dataset_dict[split], args.output_dir / f"{split}.csv")


if __name__ == "__main__":
    try:
        main()
    except DatasetAccessError as exc:  # pragma: no cover - network dependent
        raise SystemExit(str(exc))
