"""Utility to download the pre-trained fastText Latvian binary vectors.

The script downloads the compressed fastText binary model for Latvian from the
public fastText repository and decompresses it into ``data/latvian`` by
default. It can be used directly from the command line or imported in a
Jupyter notebook to ensure the model is available locally.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path
from urllib.request import urlopen

DEFAULT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.lv.300.bin.gz"
DEFAULT_OUTPUT_DIR = Path("data/latvian")
DEFAULT_ARCHIVE_NAME = "cc.lv.300.bin.gz"
DEFAULT_MODEL_NAME = "cc.lv.300.bin"


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream a file from ``url`` to ``destination``.

    Parameters
    ----------
    url:
        Source URL for the file.
    destination:
        Local path to write the downloaded content to.
    chunk_size:
        Number of bytes to read per iteration while streaming the download.
    """

    with urlopen(url) as response, destination.open("wb") as output_file:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output_file.write(chunk)


def decompress_gzip(source: Path, target: Path, buffer_size: int = 1024 * 1024) -> None:
    """Decompress a ``.gz`` archive to ``target`` using streaming IO."""

    with gzip.open(source, "rb") as compressed_file, target.open("wb") as binary_file:
        shutil.copyfileobj(compressed_file, binary_file, buffer_size)


def ensure_latvian_fasttext_model(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_url: str = DEFAULT_MODEL_URL,
    archive_name: str = DEFAULT_ARCHIVE_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Path:
    """Download and decompress the Latvian fastText binary model if missing.

    Parameters
    ----------
    output_dir:
        Directory where the downloaded archive and model will be stored.
    model_url:
        URL of the compressed fastText binary model.
    archive_name:
        Filename for the downloaded archive within ``output_dir``.
    model_name:
        Filename for the decompressed binary model within ``output_dir``.

    Returns
    -------
    Path
        Location of the decompressed fastText binary model.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / archive_name
    model_path = output_dir / model_name

    if model_path.exists():
        return model_path

    print(f"Downloading Latvian fastText model from {model_url}...")
    download_file(model_url, archive_path)

    print(f"Decompressing {archive_path} to {model_path}...")
    decompress_gzip(archive_path, model_path)

    archive_path.unlink(missing_ok=True)
    print("Download complete.")
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store the downloaded model (default: data/latvian).",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_MODEL_URL,
        help="Custom URL for the compressed fastText binary model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = ensure_latvian_fasttext_model(output_dir=args.output_dir, model_url=args.url)
    print(f"Latvian fastText model is available at: {model_path}")


if __name__ == "__main__":
    main()
