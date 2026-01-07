"""Evaluate fastText versus character n-gram baselines on OOD language ID.

This script intentionally avoids transformer dependencies so it can run quickly
inside a Jupyter notebook environment. It compares:
- fastText averaged word embeddings + LogisticRegression
- Character n-gram TF-IDF + LogisticRegression
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:  # pragma: no cover - optional dependency
    import fasttext
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("fastText is required for this script. Install it via `pip install fasttext`.") from exc

RANDOM_SEED = 13
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class SentenceExample:
    text: str
    label: str


def iter_conllu_sentences(path: Path) -> Iterable[str]:
    """Yield raw sentence strings from a CoNLL-U file."""

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


def load_multilingual_wikipedia(
    data_root: Path,
    languages: Sequence[str],
    max_sentences_per_language: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Load Wikipedia sentences and language labels into a DataFrame."""

    rng = random.Random(seed)
    examples: List[SentenceExample] = []

    for lang in sorted(languages):
        lang_dir = data_root / lang
        conllu_files = sorted(lang_dir.glob("*.conllu"))
        if not conllu_files:
            warnings.warn(f"No CoNLL-U files found for language: {lang}")
            continue
        sentences: List[str] = []
        for conllu in conllu_files:
            sentences.extend(iter_conllu_sentences(conllu))
        if max_sentences_per_language is not None:
            rng.shuffle(sentences)
            sentences = sentences[:max_sentences_per_language]
        examples.extend(SentenceExample(text=s, label=lang) for s in sentences)

    rng.shuffle(examples)
    if not examples:
        raise ValueError(
            "No Wikipedia sentences were loaded. Ensure data/<lang>/*.conllu files exist for the selected languages."
        )
    return pd.DataFrame([e.__dict__ for e in examples])


def load_hate_speech_dataset(
    path: Path, language: str, text_column: str = "text", label_column: str = "label"
) -> pd.DataFrame:
    """Load an OOD hate-speech/social-media dataset and tag it with a language label."""

    if not path.exists():
        raise FileNotFoundError(f"Expected OOD file for {language}: {path}")

    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise ValueError(f"Expected a '{text_column}' column in {path}")

    domain_col = f"{language}_domain_label"
    if label_column in df.columns and label_column != domain_col:
        df = df.rename(columns={label_column: domain_col})
    elif label_column not in df.columns:
        df[domain_col] = np.nan

    if text_column != "text":
        df = df.rename(columns={text_column: "text"})

    df["label"] = language
    return df[["text", "label", domain_col]]


def load_fasttext_models(
    model_dir: Path, languages: Sequence[str], code_lookup: Optional[Dict[str, str]] = None
) -> Dict[str, fasttext.FastText._FastText]:
    """Load fastText models for the specified languages."""

    code_lookup = code_lookup or {}
    models: Dict[str, fasttext.FastText._FastText] = {}
    for lang in languages:
        code = code_lookup.get(lang, lang[:2])
        path = model_dir / f"cc.{code}.300.bin"
        if not path.exists():
            warnings.warn(f"Missing fastText model: {path}")
            continue
        models[lang] = fasttext.load_model(path.as_posix())
    if not models:
        raise FileNotFoundError("No fastText models were loaded. Please download cc.<lang>.300.bin files.")
    return models


def get_sentence_embedding(text: str, model: fasttext.FastText._FastText) -> np.ndarray:
    """Compute a sentence embedding by averaging token vectors."""

    tokens = text.split()
    if not tokens:
        return np.zeros(model.get_dimension(), dtype=np.float32)
    vectors: List[np.ndarray] = [model.get_word_vector(tok) for tok in tokens]
    return np.mean(vectors, axis=0)


def extract_fasttext_features(
    texts: Sequence[str],
    models: Dict[str, fasttext.FastText._FastText],
    language_labels: Optional[Sequence[str]] = None,
    language_hint: Optional[str] = None,
) -> np.ndarray:
    """Convert sentences to feature matrices using language-specific models."""

    if language_hint:
        if language_hint not in models:
            raise ValueError(f"language_hint={language_hint!r} not found in loaded models: {sorted(models)}")
        default_model = models[language_hint]
    else:
        default_model = None

    features: List[np.ndarray] = []
    for i, text in enumerate(texts):
        model = None
        if language_labels is not None and i < len(language_labels):
            lang = language_labels[i]
            if lang not in models:
                raise ValueError(
                    f"No fastText model loaded for language {lang!r}. Provide a language_hint or load the missing model."
                )
            model = models[lang]
        elif default_model is not None:
            model = default_model
        else:
            raise ValueError(
                "No language labels were provided and no language_hint was set; cannot select a fastText model for embedding."
            )
        features.append(get_sentence_embedding(text, model))
    return np.vstack(features)


def train_fasttext_classifier(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    models: Dict[str, fasttext.FastText._FastText],
) -> LogisticRegression:
    features = extract_fasttext_features(train_texts, models, language_labels=train_labels)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    clf.fit(features, train_labels)
    return clf


def evaluate_fasttext_classifier(
    clf: LogisticRegression,
    texts: Sequence[str],
    labels: Sequence[str],
    models: Dict[str, fasttext.FastText._FastText],
    language_hint: Optional[str] = None,
) -> Dict[str, object]:
    language_labels = None if language_hint else labels
    features = extract_fasttext_features(texts, models, language_labels=language_labels, language_hint=language_hint)
    preds = clf.predict(features)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=sorted(set(labels) | set(preds)))
    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": preds,
    }


def train_char_ngram_classifier(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    ngram_range: Tuple[int, int],
    min_df: int,
    max_features: Optional[int],
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
    )
    train_features = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    clf.fit(train_features, train_labels)
    return vectorizer, clf


def evaluate_char_ngram_classifier(
    vectorizer: TfidfVectorizer,
    clf: LogisticRegression,
    texts: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, object]:
    features = vectorizer.transform(texts)
    preds = clf.predict(features)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=sorted(set(labels) | set(preds)))
    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": preds,
    }


def summarize_metrics(name: str, eval_result: Dict[str, object]) -> Dict[str, float]:
    report = eval_result["report"]
    macro = report.get("macro avg", {})
    return {
        "Model": name,
        "Accuracy": eval_result["accuracy"],
        "Macro Precision": macro.get("precision", 0.0),
        "Macro Recall": macro.get("recall", 0.0),
        "Macro F1": macro.get("f1-score", 0.0),
    }


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fastText and character n-gram language ID on OOD hate speech/social media data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing Wikipedia CoNLL-U files.")
    parser.add_argument("--fasttext-model-dir", type=Path, default=Path("models/fasttext"), help="Directory with cc.<lang>.300.bin models.")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["kazakh", "latvian", "swedish", "yoruba", "urdu"],
        help="Languages to include from the Wikipedia dataset.",
    )
    parser.add_argument(
        "--ood-files",
        type=json.loads,
        default={
            "kazakh": "data/kazakh_hate_speech_fasttext.csv",
            "yoruba": "data/afrihate_yoruba_fasttext.csv",
            "latvian": "data/latvian_comments_fasttext_nat_only_20mb.csv",
            "swedish": "data/biaswe_fasttext.csv",
            "urdu": "data/gsm8k_urdu_fasttext.csv",
        },
        help="JSON mapping of language -> CSV path for OOD datasets.",
    )
    parser.add_argument(
        "--fasttext-language-codes",
        type=json.loads,
        default={"kazakh": "kk", "latvian": "lv", "swedish": "sv", "yoruba": "yo", "urdu": "ur"},
        help="JSON mapping of language -> ISO code used to locate cc.<code>.300.bin files.",
    )
    parser.add_argument("--max-sentences", type=int, default=2000, help="Cap sentences per language for Wikipedia data.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test proportion for Wikipedia data.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation proportion taken from the training split.")
    parser.add_argument("--char-ngram-min", type=int, default=3, help="Minimum character n-gram size.")
    parser.add_argument("--char-ngram-max", type=int, default=5, help="Maximum character n-gram size.")
    parser.add_argument("--char-min-df", type=int, default=2, help="Minimum document frequency for char n-grams.")
    parser.add_argument(
        "--char-max-features",
        type=int,
        default=200000,
        help="Maximum number of character features (lower this for faster notebook runs).",
    )
    known_args, unknown_args = parser.parse_known_args(args=args)
    if unknown_args:
        warnings.warn(f"Ignoring unrecognised arguments: {unknown_args}")
    return known_args


def main() -> None:
    args = parse_args()

    if isinstance(args.ood_files, str):
        ood_files_raw = json.loads(args.ood_files)
    else:
        ood_files_raw = args.ood_files
    ood_files = {lang: Path(path) for lang, path in ood_files_raw.items()}

    fasttext_codes = (
        json.loads(args.fasttext_language_codes)
        if isinstance(args.fasttext_language_codes, str)
        else args.fasttext_language_codes
    )

    ood_sets: Dict[str, pd.DataFrame] = {}
    for lang in args.languages:
        path = ood_files.get(lang)
        if path is None:
            warnings.warn(f"No OOD path configured for language: {lang}")
            continue
        try:
            ood_sets[lang] = load_hate_speech_dataset(path, language=lang)
            print(f"Loaded {len(ood_sets[lang])} {lang} OOD sentences (held out for OOD evaluation)")
        except FileNotFoundError as exc:
            warnings.warn(str(exc))

    if not ood_sets:
        raise FileNotFoundError("No OOD datasets were loaded. Add CSVs to data/<language>_*.csv or adjust --ood-files.")

    ood_df = pd.concat(ood_sets.values(), ignore_index=True)
    print(f"Combined OOD examples: {len(ood_df)} across {len(ood_sets)} languages")

    wiki_df = load_multilingual_wikipedia(
        args.data_dir, languages=args.languages, max_sentences_per_language=args.max_sentences, seed=RANDOM_SEED
    )
    print(f"Loaded {len(wiki_df)} Wikipedia sentences across {wiki_df.label.nunique()} languages")

    train_df, test_df = train_test_split(
        wiki_df, test_size=args.test_size, random_state=RANDOM_SEED, stratify=wiki_df.label
    )
    train_df, val_df = train_test_split(
        train_df, test_size=args.val_size, random_state=RANDOM_SEED, stratify=train_df.label
    )
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    fasttext_models = load_fasttext_models(args.fasttext_model_dir, languages=args.languages, code_lookup=fasttext_codes)
    print(f"Loaded fastText models for: {', '.join(sorted(fasttext_models))}")

    fasttext_clf = train_fasttext_classifier(train_df.text.tolist(), train_df.label.tolist(), fasttext_models)
    char_vectorizer, char_clf = train_char_ngram_classifier(
        train_df.text.tolist(),
        train_df.label.tolist(),
        ngram_range=(args.char_ngram_min, args.char_ngram_max),
        min_df=args.char_min_df,
        max_features=args.char_max_features,
    )

    id_fasttext = evaluate_fasttext_classifier(
        fasttext_clf, test_df.text.tolist(), test_df.label.tolist(), fasttext_models
    )
    id_char = evaluate_char_ngram_classifier(char_vectorizer, char_clf, test_df.text.tolist(), test_df.label.tolist())

    print(f"fastText in-distribution accuracy: {id_fasttext['accuracy']:.4f}")
    print(f"Char n-gram in-distribution accuracy: {id_char['accuracy']:.4f}")

    for lang, df in sorted(ood_sets.items()):
        ood_fasttext = evaluate_fasttext_classifier(
            fasttext_clf,
            df.text.tolist(),
            df.label.tolist(),
            fasttext_models,
            language_hint=lang,
        )
        ood_char = evaluate_char_ngram_classifier(char_vectorizer, char_clf, df.text.tolist(), df.label.tolist())
        print(
            "fastText OOD ({lang}) accuracy: {acc:.4f} | macro F1: {f1:.4f}".format(
                lang=lang,
                acc=ood_fasttext["accuracy"],
                f1=ood_fasttext["report"]["macro avg"]["f1-score"],
            )
        )
        print(
            "Char n-gram OOD ({lang}) accuracy: {acc:.4f} | macro F1: {f1:.4f}".format(
                lang=lang,
                acc=ood_char["accuracy"],
                f1=ood_char["report"]["macro avg"]["f1-score"],
            )
        )

    combined_fasttext = evaluate_fasttext_classifier(
        fasttext_clf, ood_df.text.tolist(), ood_df.label.tolist(), fasttext_models
    )
    combined_char = evaluate_char_ngram_classifier(char_vectorizer, char_clf, ood_df.text.tolist(), ood_df.label.tolist())

    print(
        "fastText macro OOD accuracy: {acc:.4f} | macro F1: {f1:.4f}".format(
            acc=combined_fasttext["accuracy"],
            f1=combined_fasttext["report"]["macro avg"]["f1-score"],
        )
    )
    print(
        "Char n-gram macro OOD accuracy: {acc:.4f} | macro F1: {f1:.4f}".format(
            acc=combined_char["accuracy"],
            f1=combined_char["report"]["macro avg"]["f1-score"],
        )
    )

    results = [
        summarize_metrics("fastText + LogisticRegression (ID)", id_fasttext),
        summarize_metrics("Char n-gram + LogisticRegression (ID)", id_char),
        summarize_metrics("fastText + LogisticRegression (OOD)", combined_fasttext),
        summarize_metrics("Char n-gram + LogisticRegression (OOD)", combined_char),
    ]

    comparison = pd.DataFrame(results)
    print("\nSummary:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
