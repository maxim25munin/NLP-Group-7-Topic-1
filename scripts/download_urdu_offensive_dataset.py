"""
Download and preprocess the Urdu offensive-language dataset hosted on Zenodo
(record 7207438) into a fastText-friendly CSV with `text` and `label` columns.

The script mirrors the Kazakh preprocessing workflow used elsewhere in the
project by providing:
- automatic download of a public Zenodo artifact (no authentication required)
- heuristics to locate the main CSV/TSV payload within the record or inside a
  provided zip archive
- cleaning and normalisation of label values
- deduplication and optional row limiting to grab a manageable chunk

Example usage:
    python scripts/download_urdu_offensive_dataset.py \
        --output data/urdu_offensive_fasttext.csv

If the environment blocks outbound network traffic, use the `--metadata-json`
flag to point the script at a previously downloaded `records/<id>.json` metadata
file so the preprocessing logic can still run on a local copy of the data.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

RECORD_ID = "7207438"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
DEFAULT_OUTPUT = Path("data/urdu_offensive_fasttext.csv")
DEFAULT_DOWNLOAD_DIR = Path("data/zenodo_urdu_offensive")

TEXT_FIELD_CANDIDATES = [
    "tweet",
    "text",
    "comment",
    "content",
    "post",
    "message",
]
LABEL_FIELD_CANDIDATES = [
    "label",
    "target",
    "class",
    "category",
    "is_offensive",
    "offensive",
]


@dataclass
class DownloadedFile:
    path: Path
    metadata: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Urdu offensive-language dataset from Zenodo and export a "
            "fastText-friendly CSV containing text and label columns."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV for the cleaned dataset.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Where to place downloaded artifacts.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default=None,
        help="Override auto-detection and use this column as the text field.",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default=None,
        help="Override auto-detection and use this column as the label field.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Only keep the first N rows after cleaning (grabs a manageable chunk).",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help=(
            "Path to a pre-downloaded Zenodo metadata JSON. Useful when network "
            "access is blocked but the raw dataset is already present locally."
        ),
    )
    parser.add_argument(
        "--artifact-name",
        type=str,
        default=None,
        help=(
            "Name of the file within the Zenodo record to download. Defaults to the "
            "first CSV/TSV file found in the metadata."
        ),
    )
    return parser.parse_args()


def fetch_metadata(path: Optional[Path] = None) -> Dict:
    if path:
        with path.open() as fh:
            return json.load(fh)

    try:
        with urlopen(API_URL) as response:  # type: ignore[call-arg]
            return json.load(response)
    except URLError as exc:  # pragma: no cover - network dependent
        raise SystemExit(
            "Failed to retrieve Zenodo metadata. If running offline, supply a "
            "local JSON via --metadata-json."
        ) from exc


def select_file(metadata: Dict, artifact_name: Optional[str]) -> Dict:
    files = metadata.get("files") or []
    if not files:
        raise SystemExit("No files listed in Zenodo metadata; cannot proceed.")

    if artifact_name:
        for f in files:
            if f.get("key") == artifact_name:
                return f
        raise SystemExit(f"Requested artifact '{artifact_name}' not found in record.")

    preferred_exts = (".csv", ".tsv", ".txt")
    for ext in preferred_exts:
        for f in files:
            key = f.get("key", "")
            if key.lower().endswith(ext):
                return f

    return files[0]


def download_file(file_info: Dict, dest_dir: Path) -> DownloadedFile:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file_info.get("key", "downloaded_file"))
    dest_path = dest_dir / filename
    download_url = file_info["links"]["self"]

    try:
        with urlopen(download_url) as response, dest_path.open("wb") as out:  # type: ignore[call-arg]
            shutil.copyfileobj(response, out)
    except URLError as exc:  # pragma: no cover - network dependent
        raise SystemExit(
            "Download failed. Verify network access or download the file manually "
            "and re-run the script with --artifact-name pointing to its name."
        ) from exc

    return DownloadedFile(path=dest_path, metadata=file_info)


def maybe_extract_zip(file: DownloadedFile, dest_dir: Path) -> Path:
    if not zipfile.is_zipfile(file.path):
        return file.path

    with zipfile.ZipFile(file.path) as archive:
        members = [m for m in archive.namelist() if not m.endswith("/")]
        if not members:
            raise SystemExit("Zip archive is empty; nothing to extract.")

        preferred_exts = (".csv", ".tsv", ".txt")
        chosen: Optional[str] = None
        for ext in preferred_exts:
            for member in members:
                if member.lower().endswith(ext):
                    chosen = member
                    break
            if chosen:
                break
        if chosen is None:
            chosen = members[0]

        dest_dir.mkdir(parents=True, exist_ok=True)
        destination = dest_dir / Path(chosen).name
        with destination.open("wb") as out:
            out.write(archive.read(chosen))
        return destination


def sniff_dialect(path: Path) -> csv.Dialect:
    sample_bytes = path.read_bytes()[:4096]
    sample_text = sample_bytes.decode("utf8", errors="ignore")
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=[",", "\t", ";", "|"])
    except csv.Error:
        return csv.get_dialect("excel")


def select_field(field_name: Optional[str], candidates: List[str], columns: List[str]) -> str:
    if field_name:
        if field_name not in columns:
            raise SystemExit(f"Specified field '{field_name}' not found in columns: {columns}")
        return field_name

    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise SystemExit(
        "Could not auto-detect the required column. Please specify --text-field "
        "and --label-field explicitly."
    )


def normalise_label(raw: str) -> str:
    cleaned = raw.strip()
    lowered = cleaned.lower()
    mapping = {
        "0": "not-offensive",
        "non-offensive": "not-offensive",
        "not_offensive": "not-offensive",
        "not offensive": "not-offensive",
        "normal": "not-offensive",
        "1": "offensive",
        "offensive": "offensive",
        "hate speech": "offensive",
        "hate_speech": "offensive",
        "hate": "offensive",
        "abusive": "offensive",
        "off": "offensive",
        "not": "not-offensive",
    }
    return mapping.get(lowered, cleaned)


def read_rows(
    path: Path, text_field: str, label_field: str, dialect: Optional[csv.Dialect] = None
) -> Iterable[Tuple[str, str]]:
    dialect = dialect or sniff_dialect(path)
    with path.open(newline="", encoding="utf8", errors="ignore") as fh:
        reader = csv.DictReader(fh, dialect=dialect)
        for row in reader:
            text = (row.get(text_field) or "").strip()
            label = (row.get(label_field) or "").strip()
            if not text or not label:
                continue
            yield text, normalise_label(label)


def deduplicate(rows: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    unique: List[Tuple[str, str]] = []
    for text, label in rows:
        key = (text, label)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def write_clean_csv(output_path: Path, rows: List[Tuple[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "label"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    metadata = fetch_metadata(args.metadata_json)
    selected = select_file(metadata, args.artifact_name)
    downloaded = download_file(selected, args.download_dir)
    raw_path = maybe_extract_zip(downloaded, args.download_dir)

    dialect = sniff_dialect(raw_path)
    with raw_path.open(newline="", encoding="utf8", errors="ignore") as fh:
        reader = csv.DictReader(fh, dialect=dialect)
        header = reader.fieldnames or []
    if not header:
        raise SystemExit("Could not read header from downloaded dataset; aborting.")
    text_field = select_field(args.text_field, TEXT_FIELD_CANDIDATES, header)
    label_field = select_field(args.label_field, LABEL_FIELD_CANDIDATES, header)

    rows = deduplicate(read_rows(raw_path, text_field, label_field, dialect))
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]

    write_clean_csv(args.output, rows)
    print(f"Wrote cleaned dataset to {args.output}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
