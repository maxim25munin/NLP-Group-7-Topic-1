"""Prepare multilingual Wikipedia datasets in CoNLL-U format.

This script downloads the Wikimedia Wikipedia dumps via the HuggingFace
`wikimedia/wikipedia` dataset, performs light cleaning and tokenisation, and
outputs a CoNLL-U file suitable for downstream language identification
experiments.
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import re
from typing import Iterable, Iterator, List, Optional

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None

# Dedicated module-level logger so the script can integrate with parent logging
# configurations while still producing informative progress messages.
LOGGER = logging.getLogger(__name__)

LANGUAGE_CONFIG = {
    "de": {
        "subset": "20231101.de",
        "output": "data/german/german_wikipedia.conllu",
        "sent_id_prefix": "de",
    },
    "en": {
        "subset": "20231101.en",
        "output": "data/english/english_wikipedia.conllu",
        "sent_id_prefix": "en",
    },
    "fr": {
        "subset": "20231101.fr",
        "output": "data/french/french_wikipedia.conllu",
        "sent_id_prefix": "fr",
    },
    "kk": {
        "subset": "20231101.kk",
        "output": "data/kazakh/kazakh_wikipedia.conllu",
        "sent_id_prefix": "kk",
    },
    "lv": {
        "subset": "20231101.lv",
        "output": "data/latvian/latvian_wikipedia.conllu",
        "sent_id_prefix": "lv",
    },
    "sv": {
        "subset": "20231101.sv",
        "output": "data/swedish/swedish_wikipedia.conllu",
        "sent_id_prefix": "sv",
    },
    "sw": {
        "subset": "20231101.sw",
        "output": "data/swahili/swahili_wikipedia.conllu",
        "sent_id_prefix": "sw",
    },
    "ur": {
        "subset": "20231101.ur",
        "output": "data/urdu/urdu_wikipedia.conllu",
        "sent_id_prefix": "ur",
    },
    "wo": {
        "subset": "20231101.wo",
        "output": "data/wolof/wolof_wikipedia.conllu",
        "sent_id_prefix": "wo",
    },
    "yo": {
        "subset": "20231101.yo",
        "output": "data/yoruba/yoruba_wikipedia.conllu",
        "sent_id_prefix": "yo",
    },
}

CATEGORY_LABELS = (
    "Категория",
    "Category",
    "File",
    "Файл",
    "Image",
    "Сурет",
    "زمرہ",
    "زمرہ جات",
    "تصنيف",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define and parse command-line options for the script."""

    # We create a parser that uses the module docstring as the help description.
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--language",
        choices=sorted(LANGUAGE_CONFIG.keys()),
        default="kk",
        help="Two-letter language code to process (default: kk).",
    )

    # Dataset configuration options.
    parser.add_argument(
        "--subset",
        default=None,
        help="Wikimedia Wikipedia subset identifier (default: language specific)",
    )

    # Constraints that govern the amount of data exported.
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=10000,
        help="Maximum number of sentences to export (default: 10k)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=3,
        help="Minimum number of tokens required for a sentence to be kept",
    )

    # Output configuration for the generated CoNLL-U file.
    parser.add_argument(
        "--output",
        default=None,
        help="Output CoNLL-U path (default: language specific)",
    )

    # Randomisation settings to make data ordering deterministic.
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used to deterministically shuffle the dataset",
    )

    args = parser.parse_args(argv)

    config = LANGUAGE_CONFIG[args.language]
    if args.subset is None:
        args.subset = config["subset"]
    if args.output is None:
        args.output = config["output"]

    return args


# Compiled regular expressions that strip away MediaWiki formatting artefacts.
REF_TAG_PATTERN = re.compile(r"<ref[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
DOUBLE_BRACKET_PATTERN = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")
CURLY_PATTERN = re.compile(r"\{\{[^{}]*\}\}")
COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
CATEGORY_PATTERN = re.compile(
    r"^\s*(?:" + "|".join(re.escape(label) for label in CATEGORY_LABELS) + r"):[^\n]+$",
    re.MULTILINE,
)
WIKI_APOSTROPHE_PATTERN = re.compile(r"'{2,}")
DIGIT_GAP_PATTERN = re.compile(r"(?<=\d)\s+(?=\d)")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Regular expressions for tokenisation and sentence segmentation.
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?؟۔])\s+")


def clean_text(text: str) -> str:
    """Remove common MediaWiki artefacts and normalise whitespace."""

    # Guard clause for empty strings to avoid running the subsequent logic.
    if not text:
        return ""

    # Convert HTML entities (e.g., &amp;) into their literal characters.
    text = html.unescape(text)

    # Remove footnote references, HTML comments, and template blocks that are
    # typically not useful for plain-text modelling.
    text = REF_TAG_PATTERN.sub(" ", text)
    text = COMMENT_PATTERN.sub(" ", text)

    # Templates may be nested; keep stripping until nothing changes.
    previous = None
    while previous != text:
        previous = text
        text = CURLY_PATTERN.sub(" ", text)

    # Replace wiki link markup with the human-readable label that appears in the
    # second capture group.
    text = DOUBLE_BRACKET_PATTERN.sub(r"\1", text)

    # Strip remaining HTML tags and category markers.
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = CATEGORY_PATTERN.sub(" ", text)

    # Collapse stylistic apostrophes and remove gaps inside numbers.
    text = WIKI_APOSTROPHE_PATTERN.sub("", text)
    text = DIGIT_GAP_PATTERN.sub("", text)

    # Normalise all whitespace to a single space before trimming the edges.
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split cleaned text into individual sentences."""

    # Return early if there is nothing to segment.
    if not text:
        return []

    # Split on sentence-ending punctuation while keeping punctuation marks in
    # the resulting segments.
    segments = SENTENCE_BOUNDARY_PATTERN.split(text)

    # Remove any leftover leading/trailing whitespace and drop empty segments.
    sentences = [segment.strip() for segment in segments if segment.strip()]
    return sentences


def tokenize(sentence: str) -> List[str]:
    """Split a sentence into individual tokens using a simple regex."""

    # Capture both alphanumeric tokens and punctuation marks.
    tokens = TOKEN_PATTERN.findall(sentence)

    # Filter out empty strings that may appear due to stray whitespace.
    return [token for token in tokens if token.strip()]


def iter_sentences(
    dataset: Iterable[dict],
    max_sentences: int,
    min_tokens: int,
) -> Iterator[tuple[str, int, str, List[str]]]:
    """Yield cleaned, tokenised sentences alongside metadata."""

    # Track how many sentences have been produced so we can stop at the limit.
    count = 0

    # Iterate over each article in the dataset.
    for row in dataset:
        text = clean_text(row.get("text", ""))

        # Enumerate the sentences for stable sentence IDs.
        for idx, sentence in enumerate(split_sentences(text)):
            tokens = tokenize(sentence)

            # Skip sentences that are too short for meaningful processing.
            if len(tokens) < min_tokens:
                continue

            # Emit the metadata and token list to the caller.
            yield row["id"], idx, sentence, tokens
            count += 1

            # Stop once we have produced the requested number of sentences.
            if count >= max_sentences:
                return


def write_conllu(
    sentences: Iterable[tuple[str, int, str, List[str]]],
    output_path: str,
    sent_id_prefix: str,
) -> None:
    """Write the provided sentences to a CoNLL-U formatted file."""

    # Ensure the output directory exists before attempting to write the file.
    directory = os.path.dirname(output_path) or "."
    os.makedirs(directory, exist_ok=True)

    # Open the destination file in text mode with UTF-8 encoding.
    with open(output_path, "w", encoding="utf-8") as f:
        for article_id, sent_idx, sentence, tokens in sentences:
            # Compose a stable sentence identifier that combines article and index.
            sent_id = f"{sent_id_prefix}-{article_id}-{sent_idx+1}"
            f.write(f"# sent_id = {sent_id}\n")
            f.write(f"# text = {sentence}\n")

            # CoNLL-U requires one token per line; we only fill the ID and FORM
            # columns and leave the rest blank for downstream tools.
            for i, token in enumerate(tokens, start=1):
                f.write(f"{i}\t{token}\t_\t_\t_\t_\t_\t_\t_\t_\n")

            # Separate sentences with a blank line per the CoNLL-U specification.
            f.write("\n")


def main(argv: Optional[List[str]] = None) -> None:
    """Orchestrate the end-to-end conversion pipeline."""

    # Parse the CLI arguments (or injected argv) and configure logging.
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Always pull the dataset from Hugging Face to ensure consistent coverage.
    if load_dataset is None:
        raise RuntimeError("datasets library is required to download the dataset")

    LOGGER.info("Loading dataset subset %s for language %s", args.subset, args.language)
    dataset = load_dataset("wikimedia/wikipedia", args.subset, split="train")

    # Shuffle the dataset to introduce randomness while keeping determinism.
    dataset = dataset.shuffle(seed=args.seed)
    LOGGER.info("Dataset loaded: %d rows", len(dataset))

    # Stream the cleaned sentences and write them to disk in CoNLL-U format.
    sentences = iter_sentences(dataset, args.max_sentences, args.min_tokens)
    sent_id_prefix = LANGUAGE_CONFIG[args.language]["sent_id_prefix"]
    LOGGER.info("Writing CoNLL-U to %s", args.output)
    write_conllu(sentences, args.output, sent_id_prefix)
    LOGGER.info("Done")


if __name__ == "__main__":
    # Detect whether the script was launched inside a Jupyter notebook. In that
    # scenario we ignore the notebook's command-line arguments and supply an
    # empty list so `argparse` falls back to all defaults.
    import sys

    if sys.argv and sys.argv[0].endswith("ipykernel_launcher.py"):
        main([])
    else:
        main()
