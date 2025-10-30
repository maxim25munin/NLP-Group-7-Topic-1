"""Prepare multilingual Wikipedia datasets in CoNLL-U format.

This script downloads the Wikimedia Wikipedia dumps via the HuggingFace
`wikimedia/wikipedia` dataset, performs light cleaning and tokenisation, and
outputs a CoNLL-U file suitable for downstream language identification
experiments. The exported CoNLL-U rows are heuristically enriched so that all
ten columns contain reasonable placeholder annotations, even when gold-standard
linguistic analyses are unavailable.
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import re
import string
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None

try:
    import stanza  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    stanza = None

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


@dataclass
class SentenceRecord:
    """Container for a processed sentence ready to be written to CoNLL-U."""

    article_id: str
    sent_idx: int
    text: str
    tokens: List[str]
    stanza_sentence: Optional["stanza.models.common.doc.Sentence"] = None


_STANZA_PIPELINES: dict[str, "stanza.Pipeline"] = {}


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

    parser.add_argument(
        "--disable-stanza",
        action="store_true",
        help="Disable the Stanza pipeline and fall back to heuristic annotations",
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

PUNCTUATION_CHARS = set(string.punctuation + "„“”«»–—…¡¿؟،؛۔’‘″″")


def _format_word_id(word_id: object) -> str:
    """Normalise Stanza word identifiers into their CoNLL-U string form."""

    if isinstance(word_id, tuple):
        return ".".join(str(part) for part in word_id)
    return str(word_id)


def get_stanza_pipeline(language: str) -> "stanza.Pipeline":
    """Initialise (or reuse) a Stanza pipeline for the requested language."""

    if stanza is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Stanza is required for high-accuracy preprocessing. Install stanza to continue."
        )

    if language in _STANZA_PIPELINES:
        return _STANZA_PIPELINES[language]

    preferred_processor_sets = [
        "tokenize,mwt,pos,lemma,depparse",
        "tokenize,pos,lemma,depparse",
        "tokenize,pos,lemma",
        "tokenize,pos",
        "tokenize",
    ]

    last_error: Optional[Exception] = None
    for processors in preferred_processor_sets:
        try:
            stanza.download(language, processors=processors, verbose=False)
            pipeline = stanza.Pipeline(
                language,
                processors=processors,
                tokenize_no_ssplit=False,
                use_gpu=False,
                verbose=False,
            )
            _STANZA_PIPELINES[language] = pipeline
            return pipeline
        except Exception as exc:  # pragma: no cover - depends on local stanza data
            last_error = exc
            continue

    raise RuntimeError(
        f"Unable to initialise a Stanza pipeline for language '{language}'."
    ) from last_error


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


def guess_upos(token: str) -> str:
    """Heuristically assign a coarse universal POS tag for the token."""

    if not token:
        return "X"

    if token.isdigit():
        return "NUM"

    if all(char in PUNCTUATION_CHARS for char in token):
        return "PUNCT"

    if token.replace(".", "").replace(",", "").isdigit():
        return "NUM"

    if token.isalpha():
        if token.istitle():
            return "PROPN"
        if token.isupper() and len(token) > 1:
            return "PROPN"
        return "NOUN"

    return "SYM"


def guess_lemma(token: str, upos: str) -> str:
    """Derive a simple lemma by normalising alphabetic tokens."""

    if upos == "PUNCT" or not token:
        return token

    if token.isalpha():
        return token.lower()

    return token


def guess_feats(token: str, upos: str) -> str:
    """Generate a small set of illustrative morphological features."""

    features: List[str] = []

    if upos == "NUM":
        features.append("NumType=Card")
    elif upos == "PUNCT":
        features.append("PunctType=UNK")
    elif upos == "PROPN":
        features.append("Proper=Yes")

    if token.isalpha():
        if token.islower():
            features.append("LetterCase=Lower")
        elif token.isupper():
            features.append("LetterCase=Upper")
        elif token.istitle():
            features.append("LetterCase=Title")

    if not features:
        return "Feature=NA"

    return "|".join(sorted(set(features)))


def iter_sentences(
    dataset: Iterable[dict],
    max_sentences: int,
    min_tokens: int,
    language: str,
    use_stanza: bool,
) -> Iterator[SentenceRecord]:
    """Yield cleaned, tokenised sentences alongside metadata."""

    pipeline = get_stanza_pipeline(language) if use_stanza else None

    # Track how many sentences have been produced so we can stop at the limit.
    count = 0

    # Iterate over each article in the dataset.
    for row in dataset:
        text = clean_text(row.get("text", ""))

        if not text:
            continue

        article_id = str(
            row.get(
                "id",
                row.get(
                    "title",
                    row.get("url", row.get("pageid", row.get("wikidata_id", "unknown"))),
                ),
            )
        )

        if pipeline is not None:
            doc = pipeline(text)
            sentence_iterator = doc.sentences
        else:
            sentence_iterator = split_sentences(text)

        # Enumerate the sentences for stable sentence IDs.
        for idx, sentence in enumerate(sentence_iterator):
            if pipeline is not None:
                words = [word for token in sentence.tokens for word in token.words]
                if len(words) < min_tokens:
                    continue

                tokens = [word.text for word in words]
                sentence_text = sentence.text or " ".join(tokens)
                record = SentenceRecord(
                    article_id=article_id,
                    sent_idx=idx,
                    text=sentence_text,
                    tokens=tokens,
                    stanza_sentence=sentence,
                )
            else:
                tokens = tokenize(sentence)

                # Skip sentences that are too short for meaningful processing.
                if len(tokens) < min_tokens:
                    continue

                record = SentenceRecord(
                    article_id=article_id,
                    sent_idx=idx,
                    text=sentence,
                    tokens=tokens,
                )

            # Emit the metadata and token list to the caller.
            yield record
            count += 1

            # Stop once we have produced the requested number of sentences.
            if count >= max_sentences:
                return


def _append_misc(base_misc: Optional[str], additions: List[str]) -> str:
    """Merge additional key/value pairs into an existing MISC field."""

    misc_parts: List[str] = []
    if base_misc and base_misc != "_":
        misc_parts.extend(part for part in base_misc.split("|") if part)

    misc_parts.extend(additions)
    return "|".join(dict.fromkeys(misc_parts)) if misc_parts else "_"


def _write_stanza_sentence(
    handle,
    record: SentenceRecord,
    sent_id_prefix: str,
) -> None:
    """Serialise a SentenceRecord backed by Stanza annotations to CoNLL-U."""

    assert record.stanza_sentence is not None

    for token in record.stanza_sentence.tokens:
        if len(token.words) > 1:
            start = _format_word_id(token.words[0].id)
            end = _format_word_id(token.words[-1].id)
            handle.write(f"{start}-{end}\t{token.text}\t_\t_\t_\t_\t_\t_\t_\t_\n")

        for word in token.words:
            idx = _format_word_id(word.id)
            lemma = word.lemma or "_"
            upos = word.upos or "_"
            xpos = word.xpos or "_"
            feats = word.feats or "_"

            head = word.head if word.head is not None else 0
            head_str = str(head)
            deprel = word.deprel or ("root" if head == 0 else "dep")

            deps = word.deps
            if isinstance(deps, list):
                deps = "|".join(f"{h}:{rel}" for h, rel in deps)
            if not deps:
                deps = "0:root" if head == 0 else f"{head}:{deprel}"

            misc = _append_misc(word.misc, [f"TokenId={idx}", f"Lang={sent_id_prefix}"])

            handle.write(
                "\t".join(
                    [
                        idx,
                        word.text,
                        lemma,
                        upos,
                        xpos,
                        feats,
                        head_str,
                        deprel,
                        deps,
                        misc,
                    ]
                )
                + "\n"
            )


def _write_heuristic_sentence(
    handle,
    record: SentenceRecord,
    sent_id_prefix: str,
) -> None:
    """Serialise a heuristically processed sentence to CoNLL-U."""

    last_content_idx = 0
    for i, token in enumerate(record.tokens, start=1):
        upos = guess_upos(token)
        xpos = upos
        lemma = guess_lemma(token, upos)
        feats = guess_feats(token, upos)

        if upos != "PUNCT":
            last_content_idx = i

        if i == 1:
            head = 0
            deprel = "root"
        elif upos == "PUNCT" and last_content_idx:
            head = last_content_idx
            deprel = "punct"
        else:
            head = i - 1
            deprel = "dep"

        deps = f"{head}:{deprel}" if head else "0:root"
        misc = f"TokenId={i}|Lang={sent_id_prefix}"

        handle.write(
            "\t".join(
                [
                    str(i),
                    token,
                    lemma,
                    upos,
                    xpos,
                    feats,
                    str(head),
                    deprel,
                    deps,
                    misc,
                ]
            )
            + "\n"
        )


def write_conllu(
    sentences: Iterable[SentenceRecord],
    output_path: str,
    sent_id_prefix: str,
) -> None:
    """Write the provided sentences to a CoNLL-U formatted file."""

    # Ensure the output directory exists before attempting to write the file.
    directory = os.path.dirname(output_path) or "."
    os.makedirs(directory, exist_ok=True)

    # Open the destination file in text mode with UTF-8 encoding.
    with open(output_path, "w", encoding="utf-8") as f:
        for record in sentences:
            # Compose a stable sentence identifier that combines article and index.
            sent_id = f"{sent_id_prefix}-{record.article_id}-{record.sent_idx + 1}"
            f.write(f"# sent_id = {sent_id}\n")
            f.write(f"# text = {record.text}\n")

            if record.stanza_sentence is not None:
                _write_stanza_sentence(f, record, sent_id_prefix)
            else:
                _write_heuristic_sentence(f, record, sent_id_prefix)

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
    use_stanza = not args.disable_stanza
    if use_stanza:
        try:
            get_stanza_pipeline(args.language)
        except RuntimeError as exc:
            LOGGER.warning(
                "Falling back to heuristic preprocessing because Stanza initialisation failed: %s",
                exc,
            )
            use_stanza = False

    sentences = iter_sentences(
        dataset,
        args.max_sentences,
        args.min_tokens,
        args.language,
        use_stanza,
    )
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
