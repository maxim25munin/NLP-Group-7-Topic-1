"""Dataset loading utilities for the refactored baselines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence


@dataclass
class SentenceExample:
    """Container for a single sentence and its language label."""

    text: str
    label: str


def iter_conllu_sentences(path: Path) -> Iterator[str]:
    """Yield raw sentence texts from a CoNLL-U file."""

    buffer: List[str] = []
    for line in path.read_text(encoding="utf8").splitlines():
        if line.startswith("# text = "):
            buffer.append(line[len("# text = ") :])
        elif line.startswith("#"):
            continue
        elif not line.strip():
            if buffer:
                yield " ".join(buffer).strip()
                buffer = []
        else:
            continue
    if buffer:
        yield " ".join(buffer).strip()


def load_multilingual_dataset(
    data_root: Path,
    languages: Optional[Sequence[str]] = None,
    max_sentences_per_language: Optional[int] = None,
    seed: int = 13,
) -> List[SentenceExample]:
    """Load the multilingual dataset from the repository's ``data/`` tree."""

    rng = random.Random(seed)
    examples: List[SentenceExample] = []

    language_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    for lang_dir in language_dirs:
        language = lang_dir.name
        if languages and language not in languages:
            continue
        conllu_files = sorted(lang_dir.glob("*.conllu"))
        if not conllu_files:
            continue
        sentences: List[str] = []
        for conllu in conllu_files:
            sentences.extend(iter_conllu_sentences(conllu))
        if max_sentences_per_language is not None:
            rng.shuffle(sentences)
            sentences = sentences[:max_sentences_per_language]
        examples.extend(SentenceExample(text=sent, label=language) for sent in sentences)
    rng.shuffle(examples)
    return examples
