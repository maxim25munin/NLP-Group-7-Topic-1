"""Utility to download the pre-trained Wolof fastText binary vectors.

The script downloads compressed fastText binary models from the public
fastText repository and decompresses them into the Wolof directory under
``data`` by default. It can be used from the command line or imported in a
Jupyter notebook to ensure the required model is available locally.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path
from urllib.request import urlopen

DEFAULT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.wo.300.bin.gz"
DEFAULT_OUTPUT_DIR = Path("data/wolof")
DEFAULT_ARCHIVE_NAME = "cc.wo.300.bin.gz"
DEFAULT_MODEL_NAME = "cc.wo.300.bin"


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


def ensure_fasttext_model(
    output_dir: Path | None = None,
    model_url: str | None = None,
    archive_name: str | None = None,
    model_name: str | None = None,
) -> Path:
    """Download and decompress the Wolof fastText binary model.

    Parameters
    ----------
    output_dir:
        Directory where the downloaded archive and model will be stored. Falls
        back to the Wolof default when not provided.
    model_url:
        URL of the compressed fastText binary model. Falls back to the Wolof
        default when not provided.
    archive_name:
        Filename for the downloaded archive within ``output_dir``. Falls back
        to the Wolof default when not provided.
    model_name:
        Filename for the decompressed binary model within ``output_dir``. Falls
        back to the Wolof default when not provided.

    Returns
    -------
    Path
        Location of the decompressed fastText binary model.
    """

    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    model_url = model_url or DEFAULT_MODEL_URL
    archive_name = archive_name or DEFAULT_ARCHIVE_NAME
    model_name = model_name or DEFAULT_MODEL_NAME

    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / archive_name
    model_path = output_dir / model_name

    if model_path.exists():
        return model_path

    print(f"Downloading Wolof fastText model from {model_url}...")
    download_file(model_url, archive_path)

    print(f"Decompressing {archive_path} to {model_path}...")
    decompress_gzip(archive_path, model_path)

    archive_path.unlink(missing_ok=True)
    print("Download complete.")
    return model_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments while tolerating unknown Jupyter flags."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store the downloaded model. Defaults to the Wolof directory.",
    )
    parser.add_argument(
        "--url",
        help="Custom URL for the compressed fastText binary model.",
    )

    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unrecognized arguments: {' '.join(unknown)}")
    return args


def main() -> None:
    args = parse_args()
    model_path = ensure_fasttext_model(
        output_dir=args.output_dir,
        model_url=args.url,
    )
    print(f"Wolof fastText model is available at: {model_path}")


if __name__ == "__main__":
    main()
