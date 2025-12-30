"""Evaluate XLM-RoBERTa versus fastText on non-English OOD language identification."""

from __future__ import annotations

import argparse
import json
import inspect
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:  # pragma: no cover - optional dependency
    import fasttext
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit("fastText is required for this script. Install it via `pip install fasttext`.") from exc

# Lazily import transformers stack for the XLM-R baseline
TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None
try:  # pragma: no cover - heavy dependency initialisation
    from packaging import version

    try:
        import huggingface_hub

        if version.parse(huggingface_hub.__version__) >= version.parse("1.0.0"):
            raise ImportError(
                "huggingface_hub>=1.0 detected; transformers in this project requires "
                "huggingface_hub<1.0. Install a compatible version with "
                "`pip install \"huggingface_hub<1.0\"`."
            )
    except ImportError:
        # Either the package is missing (handled below) or already incompatible.
        pass

    import torch
    from datasets import Dataset
    import transformers

    if not hasattr(transformers.utils, "is_torch_greater_or_equal"):

        def _is_torch_greater_or_equal(min_version: str) -> bool:
            if torch is None:
                return False
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
) -> Tuple[LogisticRegression, np.ndarray]:
    features = extract_fasttext_features(train_texts, models, language_labels=train_labels)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    clf.fit(features, train_labels)
    return clf, features


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
    return {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist(), "predictions": preds}


def collect_misclassifications(
    texts: Sequence[str], labels: Sequence[str], preds: Sequence[str], limit: int = 20
) -> pd.DataFrame:
    indices = [i for i, (y, p) in enumerate(zip(labels, preds)) if y != p]
    sampled = indices[:limit]
    return pd.DataFrame(
        {
            "text": [texts[i] for i in sampled],
            "true_label": [labels[i] for i in sampled],
            "predicted_label": [preds[i] for i in sampled],
            "token_count": [len(texts[i].split()) for i in sampled],
        }
    )


class XLMRClassifier:
    """Fine-tunes an XLM-RoBERTa sequence classification model."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        output_dir: Path | str = Path("reports/xlmr_ood_language_id"),
        num_train_epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        seed: int = RANDOM_SEED,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "The transformers baseline requires PyTorch and transformers to be installed."
            )
        self.model_name = model_name
        self.output_dir = Path(output_dir)
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
            return self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

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


def evaluate_xlmr_classifier(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    val_texts: Sequence[str],
    val_labels: Sequence[str],
    test_texts: Sequence[str],
    test_labels: Sequence[str],
    ood_texts: Sequence[str],
    ood_labels: Sequence[str],
    model_name: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    if torch is None:
        raise RuntimeError(
            "PyTorch and transformers are required for the XLM-R baseline. "
            "Install optional dependencies with `pip install -r requirements-transformers.txt`."
        )

    classifier = XLMRClassifier(
        model_name=model_name,
        output_dir=output_dir,
        num_train_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    classifier.fit(train_texts, train_labels, val_texts, val_labels)

    def _evaluate(texts: Sequence[str], labels: Sequence[str]) -> Dict[str, object]:
        preds = classifier.predict(texts)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(labels, preds, labels=sorted(set(labels) | set(preds)))
        return {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist(), "predictions": preds}

    id_eval = _evaluate(test_texts, test_labels)
    ood_eval = _evaluate(ood_texts, ood_labels)
    return id_eval, ood_eval


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Jupyter and some IDEs inject additional arguments (for example the ``-f``
    flag used by IPython to pass the connection file). ``argparse`` raises a
    ``SystemExit`` when it encounters unexpected flags, which made the script
    unusable from a notebook. Using ``parse_known_args`` lets us ignore
    irrelevant parameters while still validating the recognised options.
    """
    parser = argparse.ArgumentParser(
        description="Compare XLM-RoBERTa and fastText language ID on OOD hate speech/social media data.",
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
    parser.add_argument("--xlmr-model", type=str, default="xlm-roberta-base", help="Hugging Face model name for XLM-RoBERTa.")
    parser.add_argument("--xlmr-epochs", type=int, default=1, help="Number of fine-tuning epochs.")
    parser.add_argument("--xlmr-batch-size", type=int, default=4, help="Per-device batch size.")
    parser.add_argument("--xlmr-learning-rate", type=float, default=5e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--xlmr-weight-decay", type=float, default=0.01, help="Weight decay for fine-tuning.")
    parser.add_argument("--xlmr-output-dir", type=Path, default=Path("reports/xlmr_ood_language_id"), help="Directory for XLM-R checkpoints/logs.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test proportion for Wikipedia data.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation proportion taken from the training split.")
    parser.add_argument(
        "--skip-xlmr",
        action="store_true",
        help="Skip XLM-R fine-tuning (useful when transformers dependencies are unavailable).",
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

    fasttext_codes = json.loads(args.fasttext_language_codes) if isinstance(args.fasttext_language_codes, str) else args.fasttext_language_codes

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

    fasttext_clf, _ = train_fasttext_classifier(train_df.text.tolist(), train_df.label.tolist(), fasttext_models)

    id_eval = evaluate_fasttext_classifier(
        fasttext_clf, test_df.text.tolist(), test_df.label.tolist(), fasttext_models
    )
    print(f"fastText in-distribution accuracy: {id_eval['accuracy']:.4f}")

    ood_evals: Dict[str, Dict[str, object]] = {}
    for lang, df in sorted(ood_sets.items()):
        ood_eval = evaluate_fasttext_classifier(
            fasttext_clf,
            df.text.tolist(),
            df.label.tolist(),
            fasttext_models,
            language_hint=lang,
        )
        ood_evals[lang] = ood_eval
        print(f"fastText OOD accuracy ({lang}): {ood_eval['accuracy']:.4f}")

    combined_ood_eval = evaluate_fasttext_classifier(
        fasttext_clf, ood_df.text.tolist(), ood_df.label.tolist(), fasttext_models
    )
    print(f"fastText macro OOD accuracy: {combined_ood_eval['accuracy']:.4f}")

    results = [
        {
            "Model": "fastText + LogisticRegression",
            "ID Accuracy": id_eval["accuracy"],
            "OOD Accuracy": combined_ood_eval["accuracy"],
        }
    ]

    if args.skip_xlmr:
        print("Skipping XLM-R evaluation (per --skip-xlmr).")
    elif TRANSFORMERS_IMPORT_ERROR is not None:
        warnings.warn(
            "PyTorch/transformers not available; skipping XLM-R baseline. "
            f"Import error: {TRANSFORMERS_IMPORT_ERROR}"
        )
    else:
        xlmr_id_eval, xlmr_ood_eval = evaluate_xlmr_classifier(
            train_df.text.tolist(),
            train_df.label.tolist(),
            val_df.text.tolist(),
            val_df.label.tolist(),
            test_df.text.tolist(),
            test_df.label.tolist(),
            ood_df.text.tolist(),
            ood_df.label.tolist(),
            model_name=args.xlmr_model,
            output_dir=args.xlmr_output_dir,
            epochs=args.xlmr_epochs,
            batch_size=args.xlmr_batch_size,
            learning_rate=args.xlmr_learning_rate,
            weight_decay=args.xlmr_weight_decay,
        )
        print(f"XLM-R in-distribution accuracy: {xlmr_id_eval['accuracy']:.4f}")
        print(f"XLM-R macro OOD accuracy: {xlmr_ood_eval['accuracy']:.4f}")
        results.append(
            {
                "Model": "XLM-R fine-tuning",
                "ID Accuracy": xlmr_id_eval["accuracy"],
                "OOD Accuracy": xlmr_ood_eval["accuracy"],
            }
        )

    comparison = pd.DataFrame(results)
    comparison["Performance Drop"] = comparison["ID Accuracy"] - comparison["OOD Accuracy"]
    print("\nSummary:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
