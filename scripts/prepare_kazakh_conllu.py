"""Prepare Kazakh Wikipedia dataset in CoNLL-U format.

This script downloads the Wikimedia Kazakh Wikipedia dump via the
HuggingFace `wikimedia/wikipedia` dataset, performs light cleaning
and tokenisation, and outputs a CoNLL-U file suitable for downstream
language identification experiments.
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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define and parse command-line options for the script."""

    # We create a parser that uses the module docstring as the help description.
    parser = argparse.ArgumentParser(description=__doc__)

    # Dataset configuration options.
    parser.add_argument(
        "--subset",
        default="20231101.kk",
        help="Wikimedia Wikipedia subset identifier (default: 20231101.kk)",
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
        default="data/kazakh/kazakh_wikipedia.conllu",
        help="Output CoNLL-U path",
    )

    # Randomisation settings to make data ordering deterministic.
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used to deterministically shuffle the dataset",
    )

    # Allow the caller to pass a custom argv (e.g., from notebooks or tests).
    return parser.parse_args(argv)


# Compiled regular expressions that strip away MediaWiki formatting artefacts.
REF_TAG_PATTERN = re.compile(r"<ref[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
DOUBLE_BRACKET_PATTERN = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")
CURLY_PATTERN = re.compile(r"\{\{[^{}]*\}\}")
COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
CATEGORY_PATTERN = re.compile(r"^\s*(?:Категория|File|Файл|Image|Сурет):[^\n]+$", re.MULTILINE)
WIKI_APOSTROPHE_PATTERN = re.compile(r"'{2,}")
DIGIT_GAP_PATTERN = re.compile(r"(?<=\d)\s+(?=\d)")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Regular expressions for tokenisation and sentence segmentation.
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


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
) -> None:
    """Write the provided sentences to a CoNLL-U formatted file."""

    # Ensure the output directory exists before attempting to write the file.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open the destination file in text mode with UTF-8 encoding.
    with open(output_path, "w", encoding="utf-8") as f:
        for article_id, sent_idx, sentence, tokens in sentences:
            # Compose a stable sentence identifier that combines article and index.
            sent_id = f"kk-{article_id}-{sent_idx+1}"
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

    LOGGER.info("Loading dataset subset %s", args.subset)
    dataset = load_dataset("wikimedia/wikipedia", args.subset, split="train")
    
    # Shuffle the dataset to introduce randomness while keeping determinism.
    dataset = dataset.shuffle(seed=args.seed)
    LOGGER.info("Dataset loaded: %d rows", len(dataset))

    # Stream the cleaned sentences and write them to disk in CoNLL-U format.
    sentences = iter_sentences(dataset, args.max_sentences, args.min_tokens)
    LOGGER.info("Writing CoNLL-U to %s", args.output)
    write_conllu(sentences, args.output)
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
