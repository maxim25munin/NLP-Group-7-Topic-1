"""Download the Swedish misogyny detection dataset (BiaSWE) from Hugging Face.

The script fetches files from the public Hugging Face dataset repository
``AI-Sweden-Models/BiaSWE`` and stores them in a local directory. By default it
pulls common tabular/text artefacts (CSV, TSV, JSONL, JSON, TXT, and Parquet),
but you can optionally narrow the download to specific filenames or extensions.

Example usage:
    python scripts/download_biaswe_dataset.py --output-dir data/biaswe
    python scripts/download_biaswe_dataset.py --include train.csv --include dev.csv

The script requires the ``huggingface_hub`` package. Install it with:
    pip install huggingface_hub
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

DEFAULT_REPO_ID = "AI-Sweden-Models/BiaSWE"
DEFAULT_OUTPUT_DIR = Path("data/biaswe")
DEFAULT_EXTENSIONS = (".csv", ".tsv", ".jsonl", ".json", ".txt", ".parquet")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repository ID to download from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to store downloaded files (defaults to data/biaswe).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help=(
            "File name (or extension like .csv) to fetch. May be passed multiple "
            "times; defaults to common data formats if omitted."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token if the dataset is private.",
    )

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unrecognised arguments: {' '.join(unknown)}")
    return args


def select_files(repo_id: str, includes: Iterable[str] | None, token: str | None) -> List[str]:
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
    except HfHubHTTPError as exc:  # pragma: no cover - network dependent
        raise SystemExit(f"Unable to list files for {repo_id}: {exc}") from exc

    if includes:
        selections = [inc.lower() for inc in includes]
        matched = [
            f for f in files
            if any(f.lower() == inc or f.lower().endswith(inc) for inc in selections)
        ]
        if not matched:
            requested = ", ".join(includes)
            raise SystemExit(
                f"No files in {repo_id} matched the requested includes: {requested}."
            )
        return matched

    default_matches = [f for f in files if f.lower().endswith(DEFAULT_EXTENSIONS)]
    if default_matches:
        return default_matches
    return files


def download_files(repo_id: str, files: Iterable[str], output_dir: Path, token: str | None) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths: List[Path] = []

    for file in files:
        print(f"Downloading {file}...")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                repo_type="dataset",
                token=token,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
        except HfHubHTTPError as exc:  # pragma: no cover - network dependent
            raise SystemExit(f"Failed to download {file} from {repo_id}: {exc}") from exc

        downloaded_paths.append(Path(local_path))
    return downloaded_paths


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    files_to_download = select_files(args.repo_id, args.include, args.token)
    downloaded = download_files(args.repo_id, files_to_download, args.output_dir, args.token)

    print("Download complete. Saved files:")
    for path in downloaded:
        print(f" - {path}")


if __name__ == "__main__":
    main()
