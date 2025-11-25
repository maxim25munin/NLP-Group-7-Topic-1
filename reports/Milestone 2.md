```python
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
import random
import textwrap
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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
    
    import torch
    from datasets import Dataset

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
    "german": (" und ", " der ", " die ", " nicht "),
    "english": (" the ", " and ", " is ", " was "),
    "french": (" le ", " la ", " les ", " des "),
    "swedish": (" och ", " det ", " som ", " inte "),
    "latvian": (" un ", " kas ", " par ", " ar "),
    "swahili": (" ya ", " kwa ", " na ", " cha "),
    "wolof": (" ci ", " ak ", " la ", " nga "),
    "yoruba": (" ni ", " ati ", " ṣe ", " jẹ "),
}

LANGUAGE_SCRIPTS: Dict[str, str] = {
    "kazakh": "Cyrillic",
    "urdu": "Arabic",
}


class RuleBasedIdentifier:
    """Heuristic classifier for language identification."""

    def __init__(self) -> None:
        self.priors: Dict[str, float] = {}

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

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        label_counts = Counter(labels)
        total = sum(label_counts.values())
        self.priors = {label: count / total for label, count in label_counts.items()}

    def _score_language(self, text: str, language: str) -> float:
        score = math.log(self.priors.get(language, 1e-6))

        dominant_script = self._dominant_script(text)
        script_expectation = LANGUAGE_SCRIPTS.get(language)
        if script_expectation and dominant_script == script_expectation:
            score += 2.0
        elif script_expectation and dominant_script and dominant_script != script_expectation:
            score -= 4.0

        special_chars = LANGUAGE_SPECIAL_CHARACTERS.get(language, "")
        if special_chars:
            char_hits = sum(text.count(char) for char in special_chars)
            score += char_hits * 1.2

        keywords = LANGUAGE_KEYWORDS.get(language, ())
        if keywords:
            keyword_hits = sum(text.lower().count(keyword.strip()) for keyword in keywords)
            score += keyword_hits * 0.8

        # Penalise languages that rely on diacritics when the sentence uses only ASCII
        if not special_chars and all(ord(c) < 128 for c in text):
            score += 0.3
        return score

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
            LOGGER.warning(
                "PyTorch/transformers not available; skipping XLM-R baseline. "
                "Install the optional dependencies with `pip install -r requirements-transformers.txt`.",
                exc_info=TRANSFORMERS_IMPORT_ERROR,
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
```

    C:\Users\Maxim\conda\lib\site-packages\torchvision\io\image.py:13: UserWarning: Failed to load image Python extension: '[WinError 127] The specified procedure could not be found'If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
      warn(
    INFO: Loading multilingual dataset from data
    INFO: Loaded 20000 sentences across 10 languages
    INFO: Training rule-based baseline
    INFO: Training character n-gram logistic regression
    C:\Users\Maxim\conda\lib\site-packages\sklearn\linear_model\_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
      warnings.warn(
    INFO: Fine-tuning XLM-R model
    


    Map:   0%|          | 0/14400 [00:00<?, ? examples/s]



    Map:   0%|          | 0/1600 [00:00<?, ? examples/s]


    Some weights of XLMRobertaForSequenceClassification were not initialized from the model checkpoint at xlm-roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    C:\Users\Maxim\conda\lib\site-packages\torch\utils\data\dataloader.py:666: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
      warnings.warn(warn_msg)
    



    <div>

      <progress value='1800' max='1800' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [1800/1800 2:23:46, Epoch 1/1]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1800</td>
      <td>0.312500</td>
    </tr>
  </tbody>
</table><p>



    Map:   0%|          | 0/4000 [00:00<?, ? examples/s]






    
    ================================================================================
    Results for Rule-based heuristics
    ================================================================================
    Accuracy: 0.1000
    
    Classification report:
    label           precision  recall  f1-score  support
    english heuristic     0.000   0.000    0.000      400
    french stanza       0.000   0.000    0.000      400
    german heuristic     0.100   1.000    0.182      400
    kazakh stanza       0.000   0.000    0.000      400
    latvian stanza      0.000   0.000    0.000      400
    swahili heuristic     0.000   0.000    0.000      400
    swedish stanza      0.000   0.000    0.000      400
    urdu stanza         0.000   0.000    0.000      400
    wolof stanza        0.000   0.000    0.000      400
    yoruba heuristic     0.000   0.000    0.000      400
    macro avg           0.010   0.100    0.018     4000
    weighted avg        0.010   0.100    0.018     4000
    
    Confusion matrix (rows = gold, columns = predicted):
                english heuristic french stanza german heuristic kazakh stanza latvian stanza swahili heuristic swedish stanza urdu stanza wolof stanza yoruba heuristic
    english heuristic0          0          400        0          0          0          0          0          0          0         
    french stanza0          0          400        0          0          0          0          0          0          0         
    german heuristic0          0          400        0          0          0          0          0          0          0         
    kazakh stanza0          0          400        0          0          0          0          0          0          0         
    latvian stanza0          0          400        0          0          0          0          0          0          0         
    swahili heuristic0          0          400        0          0          0          0          0          0          0         
    swedish stanza0          0          400        0          0          0          0          0          0          0         
    urdu stanza 0          0          400        0          0          0          0          0          0          0         
    wolof stanza0          0          400        0          0          0          0          0          0          0         
    yoruba heuristic0          0          400        0          0          0          0          0          0          0         
    
    Representative misclassifications:
    - Gold label english heuristic:
        predicted german heuristic :: University of Chicago Press, 1953 Harry Kalven Jr.
        predicted german heuristic :: Benjamin Mark Seymour (born 16 April 1999) is an English professional footballer who plays as a forward for National League South club Hampton & Richmond Borough.
    - Gold label french stanza:
        predicted german heuristic :: Références Bibliographie Ignacio URÍA, Viento norte.
        predicted german heuristic :: Entre-temps, KB Saliout fusionne avec l'usine Khrounitchev pour former le GKNPZ Khrounitchev.
    - Gold label kazakh stanza:
        predicted german heuristic :: Дереккөздер География және геодезия
        predicted german heuristic :: 14-16 ғ. көптеген мемлекеттерде ордендер мен медальдар жасала бастады.
    - Gold label latvian stanza:
        predicted german heuristic :: Platons bieži tiek minēts kā visbagātākais informācijas avots par Sokrata dzīvi un filozofiju.
        predicted german heuristic :: Barību meklē uz ūdensaugiem, gan augošiem ūdenī, gan krastā.
    - Gold label swahili heuristic:
        predicted german heuristic :: Tazama pia Orodha ya visiwa vya Tanzania Tanbihi Viungo vya nje Geonames.org Visiwa vya Tanzania Ziwa Viktoria Mkoa wa Kagera
        predicted german heuristic :: Kata ya Naipanga imeundwa na vijiji vinne (4) ambavyo ni Naipanga, Joshoni, Congo na Nagaga.
    - Gold label swedish stanza:
        predicted german heuristic :: Källor Berg i Litauen
        predicted german heuristic :: Fotboll, konditionsträning och jogging har varit de huvudsakliga sportintressena i hans liv.
    - Gold label urdu stanza:
        predicted german heuristic :: پھر جب اٹھارہ سو ستاون کی جنگ آزادی واقع ہوئی تب نواب بھوپال نے اپنی ریاست میں امن و امان قائم رکھنے کے لیےانگریزی فوج کا ساتھ دیا ،
        predicted german heuristic :: انہوں نے کہا کہ شرکت کرنے والی ٹیمیں لاہور لائنز، پشاور ڈیرز، کراچی ڈولفنز، کوئٹہ پینتھرز اور اسلام آباد شاہین شامل ہیں۔
    - Gold label wolof stanza:
        predicted german heuristic :: Bañ a nelaw guddi
        predicted german heuristic :: Bamu mujj di ku xarañe bépp xam-xamu Lislaam, te it di kuy jëfe ngirum Tassawuuf ci anam gu xóot, Jigéen ci lislaam:
    - Gold label yoruba heuristic:
        predicted german heuristic :: Ní ọdún 2014, Boafo kópa nínu eré oníṣókí kan tí àkọ́lé rẹ̀ jẹ́ Bus Nut.
        predicted german heuristic :: Iṣẹ́ orílẹ̀-ède Nàìjíríà Arábìnrin náà gbá bọ́ọ̀lù fún ikọ̀ agbábọ́ọ̀lù sínú agbọ̀n ti àwọn obìnrin First Bank ti orílẹ̀-ède Nàìjíríà ti ìlú Èkó tí a mọ̀ sí Elephant Girls nígbà…
    
    ================================================================================
    Results for Char n-gram logistic regression
    ================================================================================
    Accuracy: 0.9677
    
    Classification report:
    label           precision  recall  f1-score  support
    english heuristic     0.881   0.963    0.920      400
    french stanza       0.970   0.975    0.973      400
    german heuristic     0.929   0.955    0.942      400
    kazakh stanza       0.983   1.000    0.991      400
    latvian stanza      0.995   0.988    0.991      400
    swahili heuristic     0.976   0.927    0.951      400
    swedish stanza      0.990   0.980    0.985      400
    urdu stanza         1.000   1.000    1.000      400
    wolof stanza        0.983   0.983    0.983      400
    yoruba heuristic     0.981   0.907    0.943      400
    macro avg           0.969   0.968    0.968     4000
    weighted avg        0.969   0.968    0.968     4000
    
    Confusion matrix (rows = gold, columns = predicted):
                english heuristic french stanza german heuristic kazakh stanza latvian stanza swahili heuristic swedish stanza urdu stanza wolof stanza yoruba heuristic
    english heuristic385        2          5          2          1          0          1          0          1          3         
    french stanza3          390        2          0          0          1          0          0          3          1         
    german heuristic5          4          382        4          0          2          1          0          1          1         
    kazakh stanza0          0          0          400        0          0          0          0          0          0         
    latvian stanza3          0          2          0          395        0          0          0          0          0         
    swahili heuristic16         1          6          0          0          371        2          0          2          2         
    swedish stanza2          2          4          0          0          0          392        0          0          0         
    urdu stanza 0          0          0          0          0          0          0          400        0          0         
    wolof stanza1          2          1          1          0          2          0          0          393        0         
    yoruba heuristic22         1          9          0          1          4          0          0          0          363       
    
    Representative misclassifications:
    - Gold label english heuristic:
        predicted yoruba heuristic :: University of Chicago Press, 1953 Harry Kalven Jr.
        predicted swedish stanza :: Author Stephen J.
    - Gold label french stanza:
        predicted english heuristic :: Discographie The Wrestling Album (1985) Piledriver - The Wrestling Album 2 (1987) WWF Full Metal (1996) WWF The Music, Vol. 2 (1997) (1997) WWF The Music, Vol. 3 (1998) WWF The…
        predicted wolof stanza :: Napoli sacra.
    - Gold label german heuristic:
        predicted kazakh stanza :: 306–312.
        predicted english heuristic :: Vera religio vindicata contra omnis generis incredulos, Stahel, Würzburg 1771.
    - Gold label latvian stanza:
        predicted german heuristic :: Geschichte der Böhmischen Provinz der Gesellschaft Jesu, I, Wien 1910; S. Polčin.
        predicted german heuristic :: Vācu ordeņa Prūsijas mestri 1229.-1239. Hermans Balke (Hermann Balk) 1239.-1244. Heinrihs no Veidas (Heinrich von Weida) 1244.-1246. Popo no Osternas (Poppo von Osterna)…
    - Gold label swahili heuristic:
        predicted german heuristic :: Oppenheimer, J.R.
        predicted english heuristic :: Miaka ya 1980 Miaka ya 1990 Miaka ya 2000 Kifo chake Tuzo za Academy Filmografia The Autobiography of Miss Jane Pittman (1974) Heartbeeps (Oscar Nomination) (1981) The Thing…
    - Gold label swedish stanza:
        predicted english heuristic :: Diskografi Studioalbum Living In A Box (1987) Gatecrashing (1989) Samlingsalbum The Best of Living in a Box (1999) The Very Best of Living in a Box (2003) Living In A Box - The…
        predicted english heuristic :: Proceedings of the Royal Society of Edinburgh 20:76-93.
    - Gold label wolof stanza:
        predicted english heuristic :: Lees bind ci moom Kathleen Sheldon, Historical Dictionary of Women in Sub-Saharan Africa, The Scarecrow Press, Inc., 2005, 448 p.
        predicted kazakh stanza :: — 376 с.
    - Gold label yoruba heuristic:
        predicted english heuristic :: This could be determined by the kind of job this person does or wealth.
        predicted german heuristic :: Lehnigk, On the Hurwitz matrix, Zeitschrift für Angewandte Mathematik und Physik (ZAMP), May 1970 Bernard A.
    
    ================================================================================
    Results for XLM-R fine-tuning
    ================================================================================
    Accuracy: 0.9653
    
    Classification report:
    label           precision  recall  f1-score  support
    english heuristic     0.878   0.955    0.915      400
    french stanza       0.973   0.975    0.974      400
    german heuristic     0.912   0.960    0.935      400
    kazakh stanza       0.995   1.000    0.998      400
    latvian stanza      1.000   0.988    0.994      400
    swahili heuristic     0.951   0.927    0.939      400
    swedish stanza      0.990   0.978    0.984      400
    urdu stanza         1.000   1.000    1.000      400
    wolof stanza        0.985   0.980    0.982      400
    yoruba heuristic     0.981   0.890    0.933      400
    macro avg           0.966   0.965    0.965     4000
    weighted avg        0.966   0.965    0.965     4000
    
    Confusion matrix (rows = gold, columns = predicted):
                english heuristic french stanza german heuristic kazakh stanza latvian stanza swahili heuristic swedish stanza urdu stanza wolof stanza yoruba heuristic
    english heuristic382        1          10         0          0          6          1          0          0          0         
    french stanza2          390        3          1          0          2          1          0          0          1         
    german heuristic3          4          384        0          0          4          0          0          1          4         
    kazakh stanza0          0          0          400        0          0          0          0          0          0         
    latvian stanza3          0          2          0          395        0          0          0          0          0         
    swahili heuristic14         1          8          0          0          371        2          0          3          1         
    swedish stanza4          0          4          0          0          1          391        0          0          0         
    urdu stanza 0          0          0          0          0          0          0          400        0          0         
    wolof stanza0          4          2          1          0          0          0          0          392        1         
    yoruba heuristic27         1          8          0          0          6          0          0          2          356       
    
    Representative misclassifications:
    - Gold label english heuristic:
        predicted german heuristic :: University of Chicago Press, 1953 Harry Kalven Jr.
        predicted swahili heuristic :: Author Stephen J.
    - Gold label french stanza:
        predicted german heuristic :: Discographie The Wrestling Album (1985) Piledriver - The Wrestling Album 2 (1987) WWF Full Metal (1996) WWF The Music, Vol. 2 (1997) (1997) WWF The Music, Vol. 3 (1998) WWF The…
        predicted swahili heuristic :: Le chahut, 1977, .
    - Gold label german heuristic:
        predicted french stanza :: Lausanne 1998.
        predicted swahili heuristic :: 122–264.
    - Gold label latvian stanza:
        predicted german heuristic :: Geschichte der Böhmischen Provinz der Gesellschaft Jesu, I, Wien 1910; S. Polčin.
        predicted english heuristic :: Beznoteces ezers.
    - Gold label swahili heuristic:
        predicted german heuristic :: Oppenheimer, J.R.
        predicted german heuristic :: Maria T.
    - Gold label swedish stanza:
        predicted swahili heuristic :: Diskografi Studioalbum Living In A Box (1987) Gatecrashing (1989) Samlingsalbum The Best of Living in a Box (1999) The Very Best of Living in a Box (2003) Living In A Box - The…
        predicted english heuristic :: Proceedings of the Royal Society of Edinburgh 20:76-93.
    - Gold label wolof stanza:
        predicted kazakh stanza :: — 376 с.
        predicted french stanza :: Yero Sylla, Grammatical Relations and Fula Syntax, Los Angeles, University of California, 1979 (Thèse PhD) Louis Léon César Faidherbe, Vocabulaire d'environ 1,500 mots français…
    - Gold label yoruba heuristic:
        predicted english heuristic :: This could be determined by the kind of job this person does or wealth.
        predicted german heuristic :: Lehnigk, On the Hurwitz matrix, Zeitschrift für Angewandte Mathematik und Physik (ZAMP), May 1970 Bernard A.
    
    ################################################################################
    Model comparison and qualitative notes
    ################################################################################
    
    Accuracy overview:
    - Rule-based heuristics               0.1000
    - Char n-gram logistic regression     0.9677
    - XLM-R fine-tuning                   0.9653
    
    Operational trade-offs:
    * Rule-based heuristics: extremely cheap to run and interpretable, but they
      depend on linguistic expertise to curate the cues and struggle to scale to
      language families that share scripts or borrow vocabulary.
    * Char n-gram logistic regression: inexpensive to train and requires little
      feature engineering. It scales to new languages as long as labelled data
      is available, though it cannot leverage sub-word semantics beyond n-gram
      statistics.
    * XLM-R fine-tuning: delivers the strongest accuracy in most scenarios and
      generalises across scripts thanks to multilingual pretraining. The trade-off
      is substantially higher computational cost and a dependency on GPU
      resources, which may be prohibitive for rapid experimentation.
    
    
