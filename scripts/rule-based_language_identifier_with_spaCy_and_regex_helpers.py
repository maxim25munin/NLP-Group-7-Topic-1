"""Evaluate the rule-based language identifier with spaCy and regex helpers.

This script loads multilingual Wikipedia snippets prepared with
``scripts/prepare_multilingual_conllu.py`` and evaluates a single baseline:

* A rule-based classifier that combines Unicode script inspection, language
  specific diacritics, frequent functional words, and spaCy token statistics.
  Regular expressions are used to match keywords with proper word boundaries
  and to count language-specific diacritic characters.

The evaluation reports accuracy, precision/recall/F1, a confusion matrix, and
representative misclassifications for quick qualitative inspection.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import textwrap
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
    import spacy
except Exception as exc:  # pragma: no cover - external dependency
    raise SystemExit(
        "spaCy is required to run the rule-based evaluation script. "
        "Install it via `pip install spacy`."
    ) from exc

try:  # Optional dependency used for evaluation metrics
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "scikit-learn is required to run the evaluation script. "
        "Please install it via `pip install scikit-learn`."
    ) from exc

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
        nlp = self._get_nlp(language)
        doc = nlp(text)
        tokens = [token for token in doc if token.is_alpha]
        if not tokens:
            return 0.0
        stopwords = sum(token.is_stop for token in tokens)
        return stopwords / len(tokens)

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
        score += stop_ratio * 1.5

        # Penalise languages that rely on diacritics when the sentence uses only ASCII
        if not diacritic_hits and language in LANGUAGE_SPECIAL_CHARACTERS:
            if all(ord(c) < 128 for c in text):
                score -= 0.5
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
        help="Proportion of the dataset to allocate to the test split.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.0,
        help="Unused placeholder kept for compatibility; validation is not needed for the rule-based model.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=13,
        help="Random seed for dataset shuffling and splits.",
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

    ordered_labels = sorted(set(labels))

    result = evaluate_rule_based(X_train, y_train, X_test, y_test)
    print_results(result, ordered_labels)

    if args.output_report:
        LOGGER.info("Writing report to %s", args.output_report)
        serialisable_result = {
            "name": result.name,
            "metrics": result.metrics,
            "misclassifications": result.misclassifications,
        }
        args.output_report.write_text(json.dumps(serialisable_result, indent=2, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    # Detect whether the script was launched inside a Jupyter notebook. In that
    # scenario we ignore the notebook's command-line arguments and supply an
    # empty list so `argparse` falls back to all defaults.
    import sys

    if sys.argv and sys.argv[0].endswith("ipykernel_launcher.py"):
        main([])
    else:
        main()
