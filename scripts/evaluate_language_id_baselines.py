"""Evaluate multiple baselines for the multilingual language identification task.

The script reads multilingual Wikipedia snippets prepared with
``scripts/prepare_multilingual_conllu.py`` and trains three families of
baselines:

* A hand-crafted rule-based classifier that relies on Unicode script
  inspection, language-specific diacritics, and frequent functional words.
* A classical machine-learning baseline that feeds character n-gram TF–IDF
  features into a multinomial logistic regression classifier.
* A deep-learning baseline that fine-tunes an `XLM-RoBERTa` sequence
  classification head via Hugging Face `transformers`.

Each system is evaluated quantitatively (accuracy, precision/recall/F1, and a
confusion matrix) and qualitatively through a sample of misclassified sentences.
At the end of the run the script prints a short comparison that highlights the
trade-offs between the competing approaches.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import inspect
import matplotlib.pyplot as plt
import random
import re
import textwrap
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd
from matplotlib.colors import to_rgba
from pretty_confusion_matrix import pp_matrix

try:
    import spacy
except Exception as exc:  # pragma: no cover - external dependency
    raise SystemExit(
        "spaCy is required to run the rule-based evaluation script. "
        "Install it via `pip install spacy`."
    ) from exc

try:  # Optional dependency used by the logistic regression baseline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "scikit-learn is required to run the baseline evaluation script."
        "Please install it via `pip install scikit-learn`."
    ) from exc

# The deep-learning baseline depends on PyTorch and the Hugging Face
# transformers stack. We import them lazily so the script can still evaluate the
# classical baselines in environments where GPU support is unavailable.
TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None

try:  # pragma: no cover - heavy dependency initialisation
    # Explicitly disable the TensorFlow backend in Hugging Face `transformers`.
    #
    # Users running the notebook on Windows reported crashes when the
    # `Trainer` import tried to load TensorFlow shared libraries that are not
    # available in their environment.  Setting the environment flags keeps the
    # library in its PyTorch-only mode while retaining the optional dependency
    # for users who do have TensorFlow installed.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    from packaging import version

    transformers_version: Optional[version.Version] = None
    try:
        transformers_version = version.parse(importlib_metadata.version("transformers"))
    except importlib_metadata.PackageNotFoundError:
        pass

    try:
        import huggingface_hub

        min_hf_version = version.parse("0.34.0")
        hf_version = version.parse(huggingface_hub.__version__)

        if hf_version < min_hf_version:
            raise ImportError(
                "huggingface_hub version is too old; please upgrade with "
                "`pip install -U \"huggingface_hub>=0.34.0\"`."
            )

        if (
            transformers_version is not None
            and transformers_version < version.parse("4.45.0")
            and hf_version >= version.parse("1.0.0")
        ):
            raise ImportError(
                "Installed transformers "
                f"{transformers_version} expects huggingface_hub<1.0.0. "
                "Please upgrade transformers (e.g., `pip install -U transformers`) "
                "or install a compatible hub release (<1.0.0)."
            )
    except ImportError:
        raise

    import torch
    from datasets import Dataset

    try:
        # NOTE: Some environments ship an older version of Hugging Face
        # ``transformers`` that predates the ``is_torch_greater_or_equal`` utility
        # function.  Recent releases of the library import this helper from
        # ``transformers.utils`` when initialising the :class:`~transformers.Trainer`
        # class.  If the function is missing the import raises an ``ImportError``
        # even though the rest of the API works as expected.  To keep the training
        # baseline usable without forcing a specific ``transformers`` version we
        # provide a tiny compatibility shim before importing the trainer-related
        # classes.
        import transformers
    except ImportError as exc:
        if "huggingface-hub" in str(exc) and transformers_version is not None:
            raise ImportError(
                "The installed transformers build is incompatible with the current "
                "huggingface_hub release. Upgrade transformers to >=4.45.0 (see "
                "docs/requirements-transformers.txt) or downgrade huggingface_hub "
                "to <1.0.0."
            ) from exc
        raise

    if not hasattr(transformers.utils, "is_torch_greater_or_equal"):
        try:
            from packaging import version
        except Exception:  # pragma: no cover - packaging is part of std envs
            version = None

        def _is_torch_greater_or_equal(min_version: str) -> bool:
            """Return ``True`` if the installed torch version satisfies ``min_version``.

            The real helper was introduced in ``transformers`` 4.38.  Older
            versions that still rely on :class:`~transformers.Trainer` do not
            ship the utility, so we emulate the behaviour that recent releases
            expect.  This mirrors the logic used inside ``transformers`` and is
            sufficient for the training loop implemented in this repository.
            """

            if torch is None:
                return False
            if version is None:
                # Fallback to a very small parser that covers the ``MAJOR.MINOR``
                # patterns we use in this project.
                def _parse(ver: str) -> tuple[int, ...]:
                    return tuple(int(part) for part in ver.split(".") if part.isdigit())

                return _parse(torch.__version__) >= _parse(min_version)

            return version.parse(torch.__version__) >= version.parse(min_version)

        transformers.utils.is_torch_greater_or_equal = _is_torch_greater_or_equal

    
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except Exception as exc:  # pragma: no cover - optional dependency
    torch = None
    Dataset = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    Trainer = None
    TrainingArguments = None
    TRANSFORMERS_IMPORT_ERROR = exc

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

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
            # Other comment lines are ignored.
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
    """Load the multilingual dataset from the repository's `data/` tree."""

    rng = random.Random(seed)
    examples: List[SentenceExample] = []

    language_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    for lang_dir in language_dirs:
        language = lang_dir.name
        if languages and language not in languages:
            continue
        conllu_files = sorted(lang_dir.glob("*.conllu"))
        if not conllu_files:
            LOGGER.warning("No CoNLL-U files found for language %s", language)
            continue
        sentences = []
        for conllu in conllu_files:
            sentences.extend(iter_conllu_sentences(conllu))
        if max_sentences_per_language is not None:
            rng.shuffle(sentences)
            sentences = sentences[:max_sentences_per_language]
        examples.extend(SentenceExample(text=sent, label=language) for sent in sentences)
    rng.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------

LANGUAGE_SPECIAL_CHARACTERS: Dict[str, str] = {
    "german": "äöüßÄÖÜ",  # German umlauts and eszett
    "french": "àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ",  # French diacritics
    "swedish": "åäöÅÄÖ",  # Swedish special characters
    "latvian": "āčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ",  # Latvian diacritics
    "kazakh": "әөүұқғңһіӘӨҮҰҚҒҢҺІ",  # Cyrillic letters prominent in Kazakh
    "wolof": "ëñËÑ",  # Wolof
    "yoruba": "ẹọṣńáéíóúÀÁÈÉÌÍÒÓÙÚṢẸỌŃ",  # Yoruba tonal marks (partial list)
}

LANGUAGE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "german": ("und", "der", "die", "nicht"),
    "english": ("the", "and", "is", "was"),
    "french": ("le", "la", "les", "des"),
    "swedish": ("och", "det", "som", "inte"),
    "latvian": ("un", "kas", "par", "ar"),
    "swahili": ("ya", "kwa", "na", "cha"),
    "wolof": ("ci", "ak", "la", "nga"),
    "yoruba": ("ni", "ati", "ṣe", "jẹ"),
}

LANGUAGE_SCRIPTS: Dict[str, str] = {
    "kazakh": "Cyrillic",
    "urdu": "Arabic",
}

LANGUAGE_SPACY_CODES: Dict[str, str] = {
    "german": "de",
    "english": "en",
    "french": "fr",
    "swedish": "sv",
    "latvian": "lv",
    "swahili": "sw",
    "wolof": "wo",
    "yoruba": "yo",
    "kazakh": "kk",
    "urdu": "ur",
}


class RuleBasedIdentifier:
    """Heuristic classifier for language identification using spaCy and regex."""

    def __init__(self) -> None:
        self.priors: Dict[str, float] = {}
        self._nlp_cache: Dict[str, spacy.language.Language] = {}
        self._stopword_cache: Dict[str, Set[str]] = {}
        self.top_tokens: Dict[str, Set[str]] = {}
        self.keyword_patterns: Dict[str, List[re.Pattern[str]]] = {
            language: [
                re.compile(rf"\b{re.escape(keyword)}\b", flags=re.IGNORECASE)
                for keyword in keywords
            ]
            for language, keywords in LANGUAGE_KEYWORDS.items()
        }
        self.diacritic_patterns: Dict[str, re.Pattern[str]] = {
            language: re.compile(f"[{re.escape(chars)}]")
            for language, chars in LANGUAGE_SPECIAL_CHARACTERS.items()
            if chars
        }

    @staticmethod
    def _dominant_script(text: str) -> Optional[str]:
        counts: Counter[str] = Counter()
        for char in text:
            if not char.strip():
                continue
            try:
                name = unicodedata.name(char)
            except ValueError:
                continue
            if "ARABIC" in name:
                counts["Arabic"] += 1
            elif "CYRILLIC" in name:
                counts["Cyrillic"] += 1
            elif "LATIN" in name:
                counts["Latin"] += 1
            elif "GREEK" in name:
                counts["Greek"] += 1
            else:
                counts["Other"] += 1
        if not counts:
            return None
        script, _ = counts.most_common(1)[0]
        return script

    def _get_nlp(self, language: str) -> spacy.language.Language:
        code = LANGUAGE_SPACY_CODES.get(language, "xx")
        if code not in self._nlp_cache:
            try:
                nlp = spacy.blank(code)
            except Exception:
                nlp = spacy.blank("xx")
            self._nlp_cache[code] = nlp
        return self._nlp_cache[code]

    def _stopword_ratio(self, text: str, language: str) -> float:
        stopwords = self._language_stopwords(language)
        tokens = [token.lower() for token in re.findall(r"\b\w+\b", text)]
        if not tokens:
            return 0.0
        stopword_hits = sum(token in stopwords for token in tokens)
        return stopword_hits / len(tokens)

    def _keyword_hits(self, text: str, language: str) -> int:
        patterns = self.keyword_patterns.get(language, [])
        return sum(len(pattern.findall(text)) for pattern in patterns)

    def _diacritic_hits(self, text: str, language: str) -> int:
        pattern = self.diacritic_patterns.get(language)
        if not pattern:
            return 0
        return len(pattern.findall(text))

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        label_counts = Counter(labels)
        total = sum(label_counts.values())
        self.priors = {label: count / total for label, count in label_counts.items()}

        token_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        for text, label in zip(texts, labels):
            tokens = [t.lower() for t in re.findall(r"\b\w+\b", text) if len(t) > 1]
            token_counts[label].update(tokens)

        self.top_tokens = {
            label: {token for token, _ in counter.most_common(40)} for label, counter in token_counts.items()
        }

    def _score_language(self, text: str, language: str) -> float:
        score = math.log(self.priors.get(language, 1e-6))

        dominant_script = self._dominant_script(text)
        script_expectation = LANGUAGE_SCRIPTS.get(language)
        if script_expectation and dominant_script == script_expectation:
            score += 2.0
        elif script_expectation and dominant_script and dominant_script != script_expectation:
            score -= 4.0

        keyword_hits = self._keyword_hits(text, language)
        score += keyword_hits * 1.0

        diacritic_hits = self._diacritic_hits(text, language)
        score += diacritic_hits * 1.2

        stop_ratio = self._stopword_ratio(text, language)
        score += stop_ratio * 2.0

        overlap_score = self._token_overlap(text, language)
        score += overlap_score * 1.2

        # Penalise languages that rely on diacritics when the sentence uses only ASCII
        if not diacritic_hits and language in LANGUAGE_SPECIAL_CHARACTERS:
            if all(ord(c) < 128 for c in text):
                score -= 0.5
        return score

    def _language_stopwords(self, language: str) -> Set[str]:
        if language in self._stopword_cache:
            return self._stopword_cache[language]

        code = LANGUAGE_SPACY_CODES.get(language, "")
        stopwords: Set[str] = set()
        if code:
            try:
                module = import_module(f"spacy.lang.{code}")
                if hasattr(module, "STOP_WORDS"):
                    stopwords = set(getattr(module, "STOP_WORDS"))
            except Exception:
                stopwords = set()
        self._stopword_cache[language] = stopwords
        return stopwords

    def _token_overlap(self, text: str, language: str) -> float:
        """Return fraction of tokens appearing in the language's frequent vocabulary."""

        vocab = self.top_tokens.get(language)
        if not vocab:
            return 0.0
        tokens = [token.lower() for token in re.findall(r"\b\w+\b", text)]
        if not tokens:
            return 0.0
        hits = sum(token in vocab for token in tokens)
        return hits / len(tokens)

    def predict(self, texts: Sequence[str]) -> List[str]:
        predictions = []
        for text in texts:
            scores = {label: self._score_language(text, label) for label in self.priors}
            if not scores:
                predictions.append("unknown")
                continue
            best_label = max(scores.items(), key=lambda item: item[1])[0]
            predictions.append(best_label)
        return predictions


# ---------------------------------------------------------------------------
# Machine learning baseline (character n-gram logistic regression)
# ---------------------------------------------------------------------------


def build_logistic_regression_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
    )
    classifier = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


# ---------------------------------------------------------------------------
# Deep learning baseline (XLM-R fine-tuning)
# ---------------------------------------------------------------------------


class XLMRClassifier:
    """Fine-tunes an XLM-RoBERTa sequence classification model."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        output_dir: Path = Path("./xlmr_language_id"),
        num_train_epochs: float = 1.0,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        seed: int = 13,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "The transformers baseline requires PyTorch and transformers to be installed."
            )
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.model: Optional[AutoModelForSequenceClassification] = None

    def _encode_dataset(self, dataset: Dataset) -> Dataset:
        def tokenize_function(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        return dataset.map(tokenize_function, batched=True)

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[str],
        eval_texts: Sequence[str],
        eval_labels: Sequence[str],
    ) -> None:
        unique_labels = sorted(set(train_labels) | set(eval_labels))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        train_dataset = Dataset.from_dict(
            {"text": list(train_texts), "label": [self.label2id[label] for label in train_labels]}
        )
        eval_dataset = Dataset.from_dict(
            {"text": list(eval_texts), "label": [self.label2id[label] for label in eval_labels]}
        )

        encoded_train = self._encode_dataset(train_dataset)
        encoded_eval = self._encode_dataset(eval_dataset)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            label2id=self.label2id,
            id2label=self.id2label,
        )

        # Some environments ship older ``transformers`` versions whose
        # ``TrainingArguments`` constructor does not support the modern
        # ``evaluation_strategy``/``logging_strategy`` flags.  Build the kwargs
        # dynamically to stay compatible with both old and new releases.
        args_signature = inspect.signature(TrainingArguments.__init__).parameters
        training_kwargs = dict(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            seed=self.seed,
        )

        optional_args = {
            "evaluation_strategy": "epoch",
            "logging_strategy": "epoch",
            "save_strategy": "no",
            "load_best_model_at_end": False,
        }
        for name, value in optional_args.items():
            if name in args_signature:
                training_kwargs[name] = value

        # Fall back to the legacy flag used by very old transformers releases
        # when ``evaluation_strategy`` is unavailable.
        if "evaluation_strategy" not in training_kwargs and "evaluate_during_training" in args_signature:
            training_kwargs["evaluate_during_training"] = True

        training_args = TrainingArguments(**training_kwargs)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_train,
            eval_dataset=encoded_eval,
        )
        trainer.train()
        self.trainer = trainer

    def predict(self, texts: Sequence[str]) -> List[str]:
        if self.model is None:
            raise RuntimeError("The model has not been trained yet.")
        dataset = Dataset.from_dict({"text": list(texts)})
        encoded_dataset = self._encode_dataset(dataset)
        predictions = self.trainer.predict(encoded_dataset).predictions
        predicted_ids = predictions.argmax(axis=-1)
        return [self.id2label[idx] for idx in predicted_ids]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def collect_misclassifications(
    texts: Sequence[str],
    gold: Sequence[str],
    predicted: Sequence[str],
    max_per_label: int = 2,
) -> Dict[str, List[Tuple[str, str]]]:
    """Return representative misclassifications grouped by gold label."""

    collected: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for text, gold_label, pred_label in zip(texts, gold, predicted):
        if gold_label == pred_label:
            continue
        if len(collected[gold_label]) >= max_per_label:
            continue
        snippet = textwrap.shorten(text, width=180, placeholder="…")
        collected[gold_label].append((pred_label, snippet))
    return collected


def summarise_metrics(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, object]:
    report = classification_report(gold, predicted, labels=labels, zero_division=0, output_dict=True)
    accuracy = accuracy_score(gold, predicted)
    conf_mat = confusion_matrix(gold, predicted, labels=labels)
    return {"accuracy": accuracy, "classification_report": report, "confusion_matrix": conf_mat.tolist()}


def format_classification_report(report: Dict[str, Dict[str, float]]) -> str:
    headers = ["precision", "recall", "f1-score", "support"]
    lines = ["label           precision  recall  f1-score  support"]
    for label, metrics in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1-score", 0.0)
        support = int(metrics.get("support", 0))
        lines.append(f"{label:<15} {precision:>9.3f} {recall:>7.3f} {f1:>8.3f} {support:>8}")
    global_metrics = report.get("macro avg")
    if global_metrics:
        lines.append(
            f"{'macro avg':<15} {global_metrics.get('precision', 0.0):>9.3f}"
            f" {global_metrics.get('recall', 0.0):>7.3f}"
            f" {global_metrics.get('f1-score', 0.0):>8.3f}"
            f" {int(global_metrics.get('support', 0)):>8}"
        )
    weighted_metrics = report.get("weighted avg")
    if weighted_metrics:
        lines.append(
            f"{'weighted avg':<15} {weighted_metrics.get('precision', 0.0):>9.3f}"
            f" {weighted_metrics.get('recall', 0.0):>7.3f}"
            f" {weighted_metrics.get('f1-score', 0.0):>8.3f}"
            f" {int(weighted_metrics.get('support', 0)):>8}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing the language sub-folders with CoNLL-U files.",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Subset of languages to evaluate (default: all languages present in the data root).",
    )
    parser.add_argument(
        "--max-sentences-per-language",
        type=int,
        default=2000,
        help="Optional cap on the number of sentences per language to speed up experiments.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the data set aside for the held-out test set (default: 0.2).",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Proportion of the training data reserved for validation when training the transformer.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=13,
        help="Random seed used for shuffling and model initialisation.",
    )
    parser.add_argument(
        "--xlmr-output-dir",
        type=Path,
        default=Path("./xlmr_language_id"),
        help="Directory used by the XLM-R Trainer to store checkpoints and logs.",
    )
    parser.add_argument(
        "--xlmr-epochs",
        type=float,
        default=1.0,
        help="Number of fine-tuning epochs for the XLM-R baseline (default: 1).",
    )
    parser.add_argument(
        "--xlmr-batch-size",
        type=int,
        default=8,
        help="Per-device batch size for XLM-R training and evaluation.",
    )
    parser.add_argument(
        "--xlmr-learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for XLM-R fine-tuning.",
    )
    parser.add_argument(
        "--xlmr-weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for XLM-R fine-tuning.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to save the evaluation report as JSON.",
    )
    return parser.parse_args(argv)


@dataclass
class BaselineResult:
    name: str
    metrics: Dict[str, object]
    misclassifications: Dict[str, List[Tuple[str, str]]]


def evaluate_rule_based(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
) -> BaselineResult:
    LOGGER.info("Training rule-based baseline")
    model = RuleBasedIdentifier()
    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)
    labels = sorted(set(train_labels) | set(test_labels))
    metrics = summarise_metrics(test_labels, predictions, labels)
    misclassifications = collect_misclassifications(test_texts, test_labels, predictions)
    return BaselineResult("Rule-based heuristics", metrics, misclassifications)


def evaluate_logistic_regression(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
) -> BaselineResult:
    LOGGER.info("Training character n-gram logistic regression")
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)
    labels = sorted(set(train_labels) | set(test_labels))
    metrics = summarise_metrics(test_labels, predictions, labels)
    misclassifications = collect_misclassifications(test_texts, test_labels, predictions)
    return BaselineResult("Char n-gram logistic regression", metrics, misclassifications)


def evaluate_xlmr(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    val_texts: Sequence[str],
    val_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
    args: argparse.Namespace,
) -> BaselineResult:
    if torch is None:
        raise RuntimeError(
            "PyTorch and transformers are required for the XLM-R baseline. "
            "Install the optional dependencies with `pip install -r requirements-transformers.txt`."
        )
    LOGGER.info("Fine-tuning XLM-R model")
    classifier = XLMRClassifier(
        model_name="xlm-roberta-base",
        output_dir=args.xlmr_output_dir,
        num_train_epochs=args.xlmr_epochs,
        batch_size=args.xlmr_batch_size,
        learning_rate=args.xlmr_learning_rate,
        weight_decay=args.xlmr_weight_decay,
        seed=args.random_seed,
    )
    classifier.fit(train_texts, train_labels, val_texts, val_labels)
    predictions = classifier.predict(test_texts)
    labels = sorted(set(train_labels) | set(test_labels))
    metrics = summarise_metrics(test_labels, predictions, labels)
    misclassifications = collect_misclassifications(test_texts, test_labels, predictions)
    return BaselineResult("XLM-R fine-tuning", metrics, misclassifications)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def render_pretty_confusion_matrix(
    confusion: Sequence[Sequence[int]], labels: Sequence[str], title: str
) -> Path:
    """Render and save a prettified confusion matrix heatmap.

    The output image is stored under ``reports/`` with a slugified version of
    the baseline name to make locating the visualisation straightforward.
    """

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in title).strip("_")
    slug = slug or "baseline"
    output_path = reports_dir / f"confusion_matrix_{slug}.png"

    df_cm = pd.DataFrame(confusion, index=labels, columns=labels)
    plt.figure(figsize=(8, 6))
    pp_matrix(df_cm, cmap="PuRd", figsize=(8, 6), fz=7)
    ax = plt.gca()
    white = to_rgba("white")
    for text in ax.texts:
        if to_rgba(text.get_color()) == white:
            text.set_color("black")
    plt.title(f"Confusion matrix: {title}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def print_results(result: BaselineResult, labels: Sequence[str]) -> None:
    accuracy = result.metrics.get("accuracy", 0.0)
    print("\n" + "=" * 80)
    print(f"Results for {result.name}")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.4f}")
    report = result.metrics.get("classification_report")
    if isinstance(report, dict):
        print("\nClassification report:")
        print(format_classification_report(report))
    confusion = result.metrics.get("confusion_matrix")
    if isinstance(confusion, list):
        print("\nConfusion matrix (rows = gold, columns = predicted):")
        header = "{:<12}".format(" ") + " ".join(f"{label:<10}" for label in labels)
        print(header)
        for label, row in zip(labels, confusion):
            values = " ".join(f"{value:<10}" for value in row)
            print(f"{label:<12}{values}")
        pretty_path = render_pretty_confusion_matrix(confusion, labels, result.name)
        print(f"Saved prettified confusion matrix to {pretty_path}")
    if result.misclassifications:
        print("\nRepresentative misclassifications:")
        for gold_label in labels:
            examples = result.misclassifications.get(gold_label)
            if not examples:
                continue
            print(f"- Gold label {gold_label}:")
            for predicted_label, snippet in examples:
                print(f"    predicted {predicted_label:<10} :: {snippet}")


def compare_models(results: Sequence[BaselineResult]) -> None:
    print("\n" + "#" * 80)
    print("Model comparison and qualitative notes")
    print("#" * 80)
    print("\nAccuracy overview:")
    for result in results:
        accuracy = result.metrics.get("accuracy", 0.0)
        print(f"- {result.name:<35} {accuracy:.4f}")
    print(
        "\nOperational trade-offs:\n"
        "* Rule-based heuristics: extremely cheap to run and interpretable, but they\n"
        "  depend on linguistic expertise to curate the cues and struggle to scale to\n"
        "  language families that share scripts or borrow vocabulary.\n"
        "* Char n-gram logistic regression: inexpensive to train and requires little\n"
        "  feature engineering. It scales to new languages as long as labelled data\n"
        "  is available, though it cannot leverage sub-word semantics beyond n-gram\n"
        "  statistics.\n"
        "* XLM-R fine-tuning: delivers the strongest accuracy in most scenarios and\n"
        "  generalises across scripts thanks to multilingual pretraining. The trade-off\n"
        "  is substantially higher computational cost and a dependency on GPU\n"
        "  resources, which may be prohibitive for rapid experimentation.\n"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    LOGGER.info("Loading multilingual dataset from %s", args.data_root)
    examples = load_multilingual_dataset(
        args.data_root,
        languages=args.languages,
        max_sentences_per_language=args.max_sentences_per_language,
        seed=args.random_seed,
    )
    if not examples:
        raise SystemExit("No data available. Did you run prepare_multilingual_conllu.py?")

    texts = [example.text for example in examples]
    labels = [example.label for example in examples]

    LOGGER.info("Loaded %d sentences across %d languages", len(texts), len(set(labels)))
    if len(set(labels)) < 2:
        raise SystemExit("At least two distinct languages are required for evaluation.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=labels,
    )

    if args.validation_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=args.validation_size,
            random_state=args.random_seed,
            stratify=y_train,
        )
    else:
        X_val, y_val = X_test, y_test

    ordered_labels = sorted(set(labels))

    results: List[BaselineResult] = []
    results.append(evaluate_rule_based(X_train, y_train, X_test, y_test))
    results.append(evaluate_logistic_regression(X_train, y_train, X_test, y_test))

    if torch is not None:
        try:
            results.append(
                evaluate_xlmr(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    args,
                )
            )
        except RuntimeError as exc:
            LOGGER.warning("Skipping XLM-R baseline: %s", exc)
    else:
        if TRANSFORMERS_IMPORT_ERROR:
            import_error = str(TRANSFORMERS_IMPORT_ERROR)
            compatibility_hint = ""

            if "huggingface_hub" in import_error:
                compatibility_hint = (
                    " Detected a transformers/huggingface_hub version mismatch. "
                    "Upgrade transformers to >=4.45.0 with ``pip install -U \"transformers>=4.45.0\"`` "
                    "or install a compatible hub release with ``pip install -U \"huggingface_hub<1.0.0\"``."
                )

            LOGGER.warning(
                "PyTorch/transformers not available; skipping XLM-R baseline. "
                "Install the optional dependencies with `pip install -r requirements-transformers.txt`. "
                "Import error: %s.%s",
                import_error,
                compatibility_hint,
            )
        else:
            LOGGER.warning(
                "PyTorch/transformers not available; skipping XLM-R baseline. "
                "Install the optional dependencies with `pip install -r requirements-transformers.txt`."
            )

    for result in results:
        print_results(result, ordered_labels)

    compare_models(results)

    if args.output_report:
        LOGGER.info("Writing report to %s", args.output_report)
        serialisable_results = []
        for result in results:
            serialisable_results.append(
                {
                    "name": result.name,
                    "metrics": result.metrics,
                    "misclassifications": result.misclassifications,
                }
            )
        args.output_report.write_text(json.dumps(serialisable_results, indent=2, ensure_ascii=False), encoding="utf8")

if __name__ == "__main__":
    # Detect whether the script was launched inside a Jupyter notebook. In that
    # scenario we ignore the notebook's command-line arguments and supply an
    # empty list so `argparse` falls back to all defaults.
    import sys

    if sys.argv and sys.argv[0].endswith("ipykernel_launcher.py"):
        main([])
    else:
        main()
