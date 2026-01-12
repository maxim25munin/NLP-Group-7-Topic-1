# Final Solution: fastText Primary + Character n-gram Fallback for High-OOV Sentences

This notebook documents the **final ensemble solution** we propose for OOD-robust language identification.

**Key idea ('the ensemble is the safest bet'):**
- Use **fastText + Logistic Regression** as the primary model (strong in-domain accuracy).
- For **high-OOV (out-of-vocabulary) sentences**, fall back to a **character n-gram TF-IDF + Logistic Regression** model, which is more robust to noisy, misspelled, or code-mixed text.

This notebook is based on the experimental pipeline in `scripts/evaluate_ood_fasttext_vs_char_ngrams.py` and extends it with a **routing/ensemble strategy** that mirrors our slide notes: *When we fused both approaches—using fastText as primary and falling back to character n-grams for high-OOV sentences—macro F1 jumped to 0.80.*

---
## 1. Setup
The code below keeps dependencies intentionally lightweight and mirrors the baseline script. Install `fasttext` before running this notebook.


```python
from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import fasttext

RANDOM_SEED = 13
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
```

---
## 2. Data Utilities
We load: 
- **Wikipedia (in-distribution)** CoNLL-U sentences for training/validation/testing.
- **OOD datasets** (hate speech/social media) for stress testing robustness.


```python
@dataclass
class SentenceExample:
    text: str
    label: str


def iter_conllu_sentences(path: Path) -> Iterable[str]:
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
```

---
## 3. Feature Extractors (fastText + Char n-grams)
fastText embeddings are averaged per sentence. The character n-gram model uses TF-IDF.


```python
def load_fasttext_model(
    model_dir: Path, language: str, code_lookup: Optional[Dict[str, str]] = None
) -> Optional[fasttext.FastText._FastText]:
    code_lookup = code_lookup or {}
    code = code_lookup.get(language, language[:2])
    path = model_dir / f"cc.{code}.300.bin"
    if not path.exists():
        warnings.warn(f"Missing fastText model: {path}")
        return None
    return fasttext.load_model(path.as_posix())


def load_fasttext_models(
    model_dir: Path, languages: Sequence[str], code_lookup: Optional[Dict[str, str]] = None
) -> Dict[str, fasttext.FastText._FastText]:
    models: Dict[str, fasttext.FastText._FastText] = {}
    for lang in languages:
        model = load_fasttext_model(model_dir, lang, code_lookup=code_lookup)
        if model is None:
            continue
        models[lang] = model
    if not models:
        raise FileNotFoundError("No fastText models were loaded. Please download cc.<lang>.300.bin files.")
    return models


def get_sentence_embedding(text: str, model: fasttext.FastText._FastText) -> np.ndarray:
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
    default_language: Optional[str] = None,
) -> np.ndarray:
    if language_hint:
        if language_hint not in models:
            raise ValueError(f"language_hint={language_hint!r} not found in loaded models: {sorted(models)}")
        default_model = models[language_hint]
    elif default_language:
        if default_language not in models:
            raise ValueError(
                f"default_language={default_language!r} not found in loaded models: {sorted(models)}"
            )
        default_model = models[default_language]
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
                "No language labels were provided and no language_hint/default_language was set; "
                "cannot select a fastText model for embedding."
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
```

---
## 4. OOV Routing Logic
We quantify **OOV ratio** for a sentence using the fastText vocabulary. When the ratio exceeds a threshold, we route the sample to the character n-gram classifier instead of fastText.

This is the *fusion* described in the slide notes: fastText as primary, char n-gram fallback when fastText is likely unreliable.


```python
def sentence_oov_ratio(text: str, model: fasttext.FastText._FastText) -> float:
    tokens = text.split()
    if not tokens:
        return 1.0
    oov_count = sum(1 for tok in tokens if model.get_word_id(tok) < 0)
    return oov_count / len(tokens)


def ensemble_predict(
    texts: Sequence[str],
    fasttext_clf: LogisticRegression,
    char_clf: LogisticRegression,
    char_vectorizer: TfidfVectorizer,
    models: Dict[str, fasttext.FastText._FastText],
    language_labels: Optional[Sequence[str]] = None,
    language_hint: Optional[str] = None,
    default_language: Optional[str] = None,
    oov_threshold: float = 0.3,
) -> np.ndarray:
    preds: List[str] = []
    for i, text in enumerate(texts):
        if language_labels is not None and i < len(language_labels):
            lang_key = language_labels[i]
        elif language_hint is not None:
            lang_key = language_hint
        else:
            lang_key = default_language

        if lang_key is None:
            raise ValueError("Need a language label/hint/default to select fastText model.")
        if lang_key not in models:
            raise ValueError(f"No fastText model loaded for language {lang_key!r}.")

        oov_ratio = sentence_oov_ratio(text, models[lang_key])
        if oov_ratio >= oov_threshold:
            char_features = char_vectorizer.transform([text])
            preds.append(char_clf.predict(char_features)[0])
        else:
            ft_features = extract_fasttext_features([text], models, language_labels=[lang_key])
            preds.append(fasttext_clf.predict(ft_features)[0])
    return np.array(preds)


def evaluate_ensemble(
    texts: Sequence[str],
    labels: Sequence[str],
    fasttext_clf: LogisticRegression,
    char_clf: LogisticRegression,
    char_vectorizer: TfidfVectorizer,
    models: Dict[str, fasttext.FastText._FastText],
    language_labels: Optional[Sequence[str]] = None,
    language_hint: Optional[str] = None,
    default_language: Optional[str] = None,
    oov_threshold: float = 0.3,
) -> Dict[str, object]:
    preds = ensemble_predict(
        texts,
        fasttext_clf,
        char_clf,
        char_vectorizer,
        models,
        language_labels=language_labels,
        language_hint=language_hint,
        default_language=default_language,
        oov_threshold=oov_threshold,
    )
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=sorted(set(labels) | set(preds)))
    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": preds,
    }
```

---
## 5. Configuration
Set paths for data and fastText models, plus languages and OOD datasets. These defaults mirror the baseline script.


```python
DATA_DIR = Path("data")
FASTTEXT_MODEL_DIR = Path("models/fasttext")

LANGUAGES = ["kazakh", "latvian", "swedish", "yoruba", "urdu"]
FASTTEXT_LANGUAGE_CODES = {"kazakh": "kk", "latvian": "lv", "swedish": "sv", "yoruba": "yo", "urdu": "ur"}

FASTTEXT_DEFAULT_LANGUAGE = LANGUAGES[0]

OOD_FILES = {
    "kazakh": Path("data/kazakh_hate_speech_fasttext.csv"),
    "yoruba": Path("data/afrihate_yoruba_fasttext.csv"),
    "latvian": Path("data/latvian_comments_fasttext_nat_only_20mb.csv"),
    "swedish": Path("data/biaswe_fasttext.csv"),
    "urdu": Path("data/gsm8k_urdu_fasttext.csv"),
}

MAX_SENTENCES = 2000
TEST_SIZE = 0.2
VAL_SIZE = 0.1

CHAR_NGRAM_RANGE = (3, 5)
CHAR_MIN_DF = 2
CHAR_MAX_FEATURES = 200000

OOV_THRESHOLD = 0.3
```

---
## 6. Load Data


```python
ood_sets: Dict[str, pd.DataFrame] = {}
for lang in LANGUAGES:
    path = OOD_FILES.get(lang)
    if path is None:
        continue
    try:
        ood_sets[lang] = load_hate_speech_dataset(path, language=lang)
        print(f"Loaded {len(ood_sets[lang])} {lang} OOD sentences")
    except FileNotFoundError as exc:
        warnings.warn(str(exc))

if not ood_sets:
    raise FileNotFoundError("No OOD datasets were loaded. Update OOD_FILES with available data.")

ood_df = pd.concat(ood_sets.values(), ignore_index=True)
print(f"Combined OOD examples: {len(ood_df)} across {len(ood_sets)} languages")

wiki_df = load_multilingual_wikipedia(
    DATA_DIR, languages=LANGUAGES, max_sentences_per_language=MAX_SENTENCES, seed=RANDOM_SEED
)
print(f"Loaded {len(wiki_df)} Wikipedia sentences across {wiki_df.label.nunique()} languages")

train_df, test_df = train_test_split(
    wiki_df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=wiki_df.label
)
train_df, val_df = train_test_split(
    train_df, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=train_df.label
)
print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
```

    Loaded 10150 kazakh OOD sentences
    Loaded 110607 latvian OOD sentences
    Loaded 450 swedish OOD sentences
    Loaded 4856 yoruba OOD sentences
    Loaded 6365 urdu OOD sentences
    Combined OOD examples: 132428 across 5 languages
    Loaded 10000 Wikipedia sentences across 5 languages
    Train size: 7200, Val size: 800, Test size: 2000
    

---
## 7. Train Base Models


```python
fasttext_models = load_fasttext_models(FASTTEXT_MODEL_DIR, languages=LANGUAGES, code_lookup=FASTTEXT_LANGUAGE_CODES)
print(f"Loaded fastText models for: {', '.join(sorted(fasttext_models))}")

fasttext_clf = train_fasttext_classifier(train_df.text.tolist(), train_df.label.tolist(), fasttext_models)
char_vectorizer, char_clf = train_char_ngram_classifier(
    train_df.text.tolist(),
    train_df.label.tolist(),
    ngram_range=CHAR_NGRAM_RANGE,
    min_df=CHAR_MIN_DF,
    max_features=CHAR_MAX_FEATURES,
)
```

    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    

    Loaded fastText models for: kazakh, latvian, swedish, urdu, yoruba
    

    C:\Users\Maxim\conda\lib\site-packages\sklearn\linear_model\_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
      warnings.warn(
    C:\Users\Maxim\conda\lib\site-packages\sklearn\linear_model\_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
      warnings.warn(
    

---
## 8. Evaluate the Ensemble (FastText Primary + Char Fallback)
We evaluate on both in-distribution test data and OOD data. The ensemble routes high-OOV samples to the character n-gram model.


```python
id_ensemble = evaluate_ensemble(
    test_df.text.tolist(),
    test_df.label.tolist(),
    fasttext_clf,
    char_clf,
    char_vectorizer,
    fasttext_models,
    default_language=FASTTEXT_DEFAULT_LANGUAGE,
    oov_threshold=OOV_THRESHOLD,
)

print("Ensemble (ID) Accuracy:", round(id_ensemble["accuracy"], 4))
print("Ensemble (ID) Macro Precision:", round(id_ensemble["report"]["macro avg"]["precision"], 4))
print("Ensemble (ID) Macro Recall:", round(id_ensemble["report"]["macro avg"]["recall"], 4))
print("Ensemble (ID) Macro F1:", round(id_ensemble["report"]["macro avg"]["f1-score"], 4))

ood_ensemble = evaluate_ensemble(
    ood_df.text.tolist(),
    ood_df.label.tolist(),
    fasttext_clf,
    char_clf,
    char_vectorizer,
    fasttext_models,
    default_language=FASTTEXT_DEFAULT_LANGUAGE,
    oov_threshold=OOV_THRESHOLD,
)

print("Ensemble (OOD) Accuracy:", round(ood_ensemble["accuracy"], 4))
print("Ensemble (OOD) Macro Precision:", round(ood_ensemble["report"]["macro avg"]["precision"], 4))
print("Ensemble (OOD) Macro Recall:", round(ood_ensemble["report"]["macro avg"]["recall"], 4))
print("Ensemble (OOD) Macro F1:", round(ood_ensemble["report"]["macro avg"]["f1-score"], 4))

```

    Ensemble (ID) Accuracy: 0.9755
    Ensemble (ID) Macro Precision: 0.9767
    Ensemble (ID) Macro Recall: 0.9755
    Ensemble (ID) Macro F1: 0.9754
    Ensemble (OOD) Accuracy: 0.9764
    Ensemble (OOD) Macro Precision: 0.8449
    Ensemble (OOD) Macro Recall: 0.9839
    Ensemble (OOD) Macro F1: 0.9007
    

---
## 9. Inspect OOV Routing Behavior (Optional)
This helper cell checks how often we route to the fallback for each dataset.


```python
def routing_stats(
    texts: Sequence[str],
    labels: Sequence[str],
    models: Dict[str, fasttext.FastText._FastText],
    default_language: str,
) -> pd.DataFrame:
    rows = []
    for text, label in zip(texts, labels):
        ratio = sentence_oov_ratio(text, models[default_language])
        rows.append({"label": label, "oov_ratio": ratio})
    df = pd.DataFrame(rows)
    return df

id_routing = routing_stats(test_df.text.tolist(), test_df.label.tolist(), fasttext_models, FASTTEXT_DEFAULT_LANGUAGE)
ood_routing = routing_stats(ood_df.text.tolist(), ood_df.label.tolist(), fasttext_models, FASTTEXT_DEFAULT_LANGUAGE)

print("ID routing: % fallback", (id_routing.oov_ratio >= OOV_THRESHOLD).mean())
print("OOD routing: % fallback", (ood_routing.oov_ratio >= OOV_THRESHOLD).mean())
```

    ID routing: % fallback 0.8055
    OOD routing: % fallback 0.920386927235932
    

---
## 10. Summary
This notebook demonstrates the **final ensemble solution**: a fastText-first classifier that switches to character n-grams for high-OOV sentences.

**Why this is our safest bet:**
- fastText is strong on clean, in-distribution data.
- Character n-grams handle noisy, code-mixed, or misspelled text that fastText struggles with.
- Routing based on OOV ratio operationalizes our empirical insight from the slides.

You can tune `OOV_THRESHOLD` (e.g., 0.2–0.4) to match the validation set or replicate the macro F1 improvements reported in `docs/speaker-notes.md`.
