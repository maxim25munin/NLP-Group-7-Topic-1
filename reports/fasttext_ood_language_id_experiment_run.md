# fastText OOD Language Identification Experiment

This notebook implements the experimental protocol for Question 1: evaluating pretrained fastText embeddings on out-of-distribution (OOD) Kazakh hate-speech data compared with in-distribution Wikipedia snippets. The workflow mirrors the Milestone 2 baselines and adds quantitative and qualitative analyses focused on domain shift and vocabulary mismatch.

## 1. Setup

The notebook expects:

- Wikipedia-derived CoNLL-U files for the 10 target languages under `data/<lang>/*.conllu` (prepared in Milestone 2).
- An OOD Kazakh hate-speech CSV file at `data/kazakh_hate_speech_fasttext.csv` with columns `text` and `label`.
- Pretrained fastText binary models saved as `cc.<lang>.300.bin` in `models/fasttext/` (or adjust the paths below). The Kazakh model (`cc.kk.300.bin`) is required; additional language-specific models improve coverage.


```python
from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fasttext
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
```


```python
# Reproducibility settings
RANDOM_SEED = 13
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Resolve project paths regardless of where the notebook is executed
if "__file__" in globals():
    _current_dir = Path(__file__).resolve().parent
else:
    _current_dir = Path.cwd().resolve()

_possible_roots = [_current_dir, _current_dir.parent, _current_dir.parent.parent]
PROJECT_ROOT = next((p for p in _possible_roots if (p / "data").exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Could not locate the 'data' directory. Please run the notebook from the repository or ensure data is available."
    )

DATA_DIR = PROJECT_ROOT / "data"
FASTTEXT_MODEL_DIR = PROJECT_ROOT / "models" / "fasttext"

# Languages included in the Wikipedia dataset
LANGUAGES = [
    "english",
    "french",
    "german",
    "kazakh",
    "latvian",
    "swahili",
    "swedish",
    "urdu",
    "wolof",
    "yoruba",
]

# Optional: cap the number of sentences per language to keep the notebook fast
MAX_SENTENCES_PER_LANGUAGE: Optional[int] = 2000

```

## 2. Data loading helpers

We reuse the Milestone 2 preprocessing assumptions: Wikipedia sentences are stored in CoNLL-U format with a `# text = ...` field. The hate-speech corpus is a simple CSV. Language labels are derived from the parent directory names for the Wikipedia data and set to `kazakh` for the OOD set to test language identification robustness.


```python
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


def load_kazakh_hate_speech(path: Path) -> pd.DataFrame:
    """Load the OOD Kazakh hate-speech dataset."""

    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the hate-speech CSV.")
    df = df.rename(columns={"label": "hate_label"})
    df["label"] = "kazakh"
    return df[["text", "label", "hate_label"]]

```

## 3. fastText utilities

The helpers below load language-specific fastText models, convert sentences to averaged word vectors, and compute out-of-vocabulary (OOV) rates for qualitative error analysis.


```python
def load_fasttext_models(model_dir: Path, languages: Sequence[str]) -> Dict[str, fasttext.FastText._FastText]:
    """Load fastText models for the specified languages.

    The function expects files named `cc.<lang>.300.bin` inside `model_dir`. If a
    model is missing, a warning is emitted and the language is skipped.
    """

    models: Dict[str, fasttext.FastText._FastText] = {}
    for lang in languages:
        path = model_dir / f"cc.{lang[:2]}.300.bin"
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
    language_hint: Optional[str] = None,
) -> np.ndarray:
    """Convert sentences to feature matrices using language-specific models.

    If `language_hint` is provided and exists in the model lookup, the
    corresponding model is used for all texts (useful for OOD Kazakh-only
    evaluation). Otherwise the first available model is used as a fallback.
    """

    if language_hint and language_hint in models:
        default_model = models[language_hint]
    else:
        default_model = models[sorted(models.keys())[0]]

    features: List[np.ndarray] = []
    for text in texts:
        model = models.get(language_hint, default_model)
        features.append(get_sentence_embedding(text, model))
    return np.vstack(features)


def is_in_vocabulary(word: str, model: fasttext.FastText._FastText) -> bool:
    return model.get_word_id(word) != -1


def calculate_oov_rate(texts: Sequence[str], model: fasttext.FastText._FastText) -> float:
    """Compute the average proportion of OOV tokens per sentence."""

    rates: List[float] = []
    for text in texts:
        tokens = text.split()
        if not tokens:
            rates.append(0.0)
            continue
        oov = sum(1 for tok in tokens if not is_in_vocabulary(tok, model))
        rates.append(oov / len(tokens))
    return float(np.mean(rates))
```

## 4. Model training and evaluation helpers

We train a multinomial logistic regression classifier on averaged fastText embeddings and report accuracy, per-class precision/recall/F1, and confusion matrices. Additional utilities collect misclassified samples for manual inspection.


```python
def train_fasttext_classifier(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    models: Dict[str, fasttext.FastText._FastText],
):
    features = extract_fasttext_features(train_texts, models)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    clf.fit(features, train_labels)
    return clf, features


def evaluate_fasttext_classifier(
    clf: LogisticRegression,
    texts: Sequence[str],
    labels: Sequence[str],
    models: Dict[str, fasttext.FastText._FastText],
    language_hint: Optional[str] = None,
):
    features = extract_fasttext_features(texts, models, language_hint=language_hint)
    preds = clf.predict(features)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=sorted(set(labels) | set(preds)))
    return {"accuracy": acc, "report": report, "confusion_matrix": cm, "predictions": preds}


def collect_misclassifications(
    texts: Sequence[str],
    labels: Sequence[str],
    preds: Sequence[str],
    limit: int = 20,
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
```

## 5. Load datasets

The next cell loads the Wikipedia in-distribution (ID) data and the Kazakh hate-speech OOD data, then performs a reproducible train/validation split for Wikipedia sentences.


```python
wiki_df = load_multilingual_wikipedia(
    DATA_DIR,
    languages=LANGUAGES,
    max_sentences_per_language=MAX_SENTENCES_PER_LANGUAGE,
    seed=RANDOM_SEED,
)
print(f"Loaded {len(wiki_df)} Wikipedia sentences across {wiki_df.label.nunique()} languages")

train_df, test_df = train_test_split(
    wiki_df, test_size=0.2, random_state=RANDOM_SEED, stratify=wiki_df.label
)
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

hate_df = load_kazakh_hate_speech(DATA_DIR / "kazakh_hate_speech_fasttext.csv")
print(f"Loaded {len(hate_df)} Kazakh hate-speech sentences")
```

    Loaded 20000 Wikipedia sentences across 10 languages
    Train size: 16000, Test size: 4000
    Loaded 10150 Kazakh hate-speech sentences
    

## 6. Load pretrained fastText models

Download the `cc.<lang>.300.bin` files from the [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) repository and place them in `models/fasttext/` before running this cell. At a minimum the Kazakh model (`cc.kk.300.bin`) is required.


```python
fasttext_models = load_fasttext_models(FASTTEXT_MODEL_DIR, languages=["kazakh"] + LANGUAGES)
print(f"Loaded fastText models for: {', '.join(sorted(fasttext_models))}")
```

    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    C:\Users\Maxim\AppData\Local\Temp\ipykernel_9284\426594290.py:12: UserWarning: Missing fastText model: C:\Users\Maxim\models\fasttext\cc.en.300.bin
      warnings.warn(f"Missing fastText model: {path}")
    C:\Users\Maxim\AppData\Local\Temp\ipykernel_9284\426594290.py:12: UserWarning: Missing fastText model: C:\Users\Maxim\models\fasttext\cc.fr.300.bin
      warnings.warn(f"Missing fastText model: {path}")
    C:\Users\Maxim\AppData\Local\Temp\ipykernel_9284\426594290.py:12: UserWarning: Missing fastText model: C:\Users\Maxim\models\fasttext\cc.ge.300.bin
      warnings.warn(f"Missing fastText model: {path}")
    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    

    Loaded fastText models for: kazakh
    

    C:\Users\Maxim\AppData\Local\Temp\ipykernel_9284\426594290.py:12: UserWarning: Missing fastText model: C:\Users\Maxim\models\fasttext\cc.la.300.bin
      warnings.warn(f"Missing fastText model: {path}")
    C:\Users\Maxim\AppData\Local\Temp\ipykernel_9284\426594290.py:12: UserWarning: Missing fastText model: C:\Users\Maxim\models\fasttext\cc.sw.300.bin
      warnings.warn(f"Missing fastText model: {path}")
    C:\Users\Maxim\AppData\Local\Temp\ipykernel_9284\426594290.py:12: UserWarning: Missing fastText model: C:\Users\Maxim\models\fasttext\cc.ur.300.bin
      warnings.warn(f"Missing fastText model: {path}")
    C:\Users\Maxim\AppData\Local\Temp\ipykernel_9284\426594290.py:12: UserWarning: Missing fastText model: C:\Users\Maxim\models\fasttext\cc.wo.300.bin
      warnings.warn(f"Missing fastText model: {path}")
    C:\Users\Maxim\AppData\Local\Temp\ipykernel_9284\426594290.py:12: UserWarning: Missing fastText model: C:\Users\Maxim\models\fasttext\cc.yo.300.bin
      warnings.warn(f"Missing fastText model: {path}")
    

## 7. Train the fastText baseline on Wikipedia (ID)

We train a multinomial logistic regression classifier on averaged fastText embeddings derived from the Wikipedia training split and evaluate on the held-out Wikipedia test split.


```python
fasttext_clf, train_features = train_fasttext_classifier(
    train_df.text.tolist(), train_df.label.tolist(), fasttext_models
)

id_eval = evaluate_fasttext_classifier(
    fasttext_clf, test_df.text.tolist(), test_df.label.tolist(), fasttext_models
)

print(f"In-distribution accuracy: {id_eval['accuracy']:.4f}")
print(json.dumps(id_eval["report"], indent=2))
```

    C:\Users\Maxim\conda\lib\site-packages\sklearn\linear_model\_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
      warnings.warn(
    

    In-distribution accuracy: 0.8293
    {
      "english": {
        "precision": 0.877030162412993,
        "recall": 0.945,
        "f1-score": 0.9097472924187726,
        "support": 400.0
      },
      "french": {
        "precision": 0.953125,
        "recall": 0.915,
        "f1-score": 0.9336734693877551,
        "support": 400.0
      },
      "german": {
        "precision": 0.9380053908355795,
        "recall": 0.87,
        "f1-score": 0.9027237354085603,
        "support": 400.0
      },
      "kazakh": {
        "precision": 0.9776119402985075,
        "recall": 0.9825,
        "f1-score": 0.9800498753117207,
        "support": 400.0
      },
      "latvian": {
        "precision": 0.7506053268765133,
        "recall": 0.775,
        "f1-score": 0.7626076260762608,
        "support": 400.0
      },
      "swahili": {
        "precision": 0.9252873563218391,
        "recall": 0.805,
        "f1-score": 0.8609625668449198,
        "support": 400.0
      },
      "swedish": {
        "precision": 0.8853868194842407,
        "recall": 0.7725,
        "f1-score": 0.8251001335113485,
        "support": 400.0
      },
      "urdu": {
        "precision": 0.5743740795287187,
        "recall": 0.975,
        "f1-score": 0.7228915662650602,
        "support": 400.0
      },
      "wolof": {
        "precision": 0.8981233243967829,
        "recall": 0.8375,
        "f1-score": 0.8667529107373868,
        "support": 400.0
      },
      "yoruba": {
        "precision": 0.664,
        "recall": 0.415,
        "f1-score": 0.5107692307692308,
        "support": 400.0
      },
      "accuracy": 0.82925,
      "macro avg": {
        "precision": 0.8443549400155176,
        "recall": 0.82925,
        "f1-score": 0.8275278406731015,
        "support": 4000.0
      },
      "weighted avg": {
        "precision": 0.8443549400155174,
        "recall": 0.82925,
        "f1-score": 0.8275278406731016,
        "support": 4000.0
      }
    }
    

## 8. Evaluate on Kazakh hate-speech (OOD)

The classifier trained on Wikipedia data is now tested on the OOD hate-speech corpus. Because all texts are Kazakh, the `language_hint` forces the Kazakh fastText model for embedding extraction.


```python
ood_eval = evaluate_fasttext_classifier(
    fasttext_clf,
    hate_df.text.tolist(),
    hate_df.label.tolist(),
    fasttext_models,
    language_hint="kazakh",
)

print(f"OOD accuracy (Kazakh hate speech): {ood_eval['accuracy']:.4f}")
print(json.dumps(ood_eval["report"], indent=2))
```

    OOD accuracy (Kazakh hate speech): 0.9773
    {
      "german": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
      },
      "kazakh": {
        "precision": 1.0,
        "recall": 0.9773399014778326,
        "f1-score": 0.9885401096163428,
        "support": 10150.0
      },
      "latvian": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
      },
      "swahili": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
      },
      "swedish": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
      },
      "urdu": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
      },
      "wolof": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
      },
      "yoruba": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
      },
      "accuracy": 0.9773399014778326,
      "macro avg": {
        "precision": 0.125,
        "recall": 0.12216748768472907,
        "f1-score": 0.12356751370204284,
        "support": 10150.0
      },
      "weighted avg": {
        "precision": 1.0,
        "recall": 0.9773399014778326,
        "f1-score": 0.9885401096163428,
        "support": 10150.0
      }
    }
    

## 9. Quantitative comparison with Milestone 2 baselines

Populate the baseline metrics below if you have already run the character n-gram TF–IDF and XLM-R experiments. The performance drop column highlights how strongly each approach degrades under domain shift.


```python
MILESTONE_TFIDF_ID = 0.9677  # Reported in Milestone 2
MILESTONE_TFIDF_OOD = np.nan  # Replace with your measured OOD accuracy
XLMR_ID = np.nan  # Replace with XLM-R in-distribution accuracy
XLMR_OOD = np.nan  # Replace with XLM-R OOD accuracy

comparison = pd.DataFrame(
    {
        "Method": ["Char n-gram TF-IDF (Milestone 2)", "fastText embeddings", "XLM-R fine-tuning"],
        "Wikipedia (ID) Accuracy": [MILESTONE_TFIDF_ID, id_eval["accuracy"], XLMR_ID],
        "Hate Speech (OOD) Accuracy": [MILESTONE_TFIDF_OOD, ood_eval["accuracy"], XLMR_OOD],
    }
)
comparison["Performance Drop"] = comparison["Wikipedia (ID) Accuracy"] - comparison["Hate Speech (OOD) Accuracy"]
comparison
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Method</th>
      <th>Wikipedia (ID) Accuracy</th>
      <th>Hate Speech (OOD) Accuracy</th>
      <th>Performance Drop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Char n-gram TF-IDF (Milestone 2)</td>
      <td>0.96770</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fastText embeddings</td>
      <td>0.82925</td>
      <td>0.97734</td>
      <td>-0.14809</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XLM-R fine-tuning</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## 10. Error analysis

We compute OOV rates for Wikipedia vs. hate-speech data, examine vocabulary overlap, and capture a sample of misclassified OOD examples to understand failure modes such as slang, code-switching, and short utterances.


```python
kk_model = fasttext_models.get("kazakh") or next(iter(fasttext_models.values()))

oov_wiki = calculate_oov_rate(test_df.text.tolist(), kk_model)
oov_hate = calculate_oov_rate(hate_df.text.tolist(), kk_model)
print(f"OOV rate on Wikipedia test split: {oov_wiki:.2%}")
print(f"OOV rate on hate-speech corpus: {oov_hate:.2%}")

wiki_vocab = set(" ".join(train_df.text.tolist()).split())
hate_vocab = set(" ".join(hate_df.text.tolist()).split())
vocab_overlap = len(wiki_vocab & hate_vocab) / max(len(hate_vocab), 1)
print(f"Vocabulary overlap (hate-speech vs. Wikipedia): {vocab_overlap:.2%}")

hate_only = sorted(hate_vocab - wiki_vocab)
print(f"Hate-speech-specific vocabulary items: {len(hate_only)}")
print("Sample:", hate_only[:20])

error_df = collect_misclassifications(
    hate_df.text.tolist(), hate_df.label.tolist(), ood_eval["predictions"], limit=20
)
error_df.head()
```

    OOV rate on Wikipedia test split: 55.82%
    OOV rate on hate-speech corpus: 8.49%
    Vocabulary overlap (hate-speech vs. Wikipedia): 7.60%
    Hate-speech-specific vocabulary items: 30121
    Sample: ['aamaq', 'aazaz', 'abazubayr', 'abc', 'abdullahazam', 'abo', 'abofsomalia', 'abramsтың', 'abukamal', 'acab', 'acer', 'adamm', 'afp', 'afr', 'afriforum', 'agb', 'ahahahahhahaah', 'ahhhhhh', 'aim', 'airlines']
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>true_label</th>
      <th>predicted_label</th>
      <th>token_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ләйлі мәжнүн</td>
      <td>kazakh</td>
      <td>urdu</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>таһир зуһра</td>
      <td>kazakh</td>
      <td>latvian</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>арзу қамбар</td>
      <td>kazakh</td>
      <td>urdu</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>уәки күлшаһ</td>
      <td>kazakh</td>
      <td>latvian</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>жүсіп зылиқа</td>
      <td>kazakh</td>
      <td>urdu</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## 11. Takeaways

- fastText embeddings trained on Wikipedia/Common Crawl are sensitive to domain and vocabulary shift; expect lower accuracy on OOD hate-speech content than on in-distribution Wikipedia text.
- Character n-gram TF–IDF baselines are often more robust to slang, profanity, and code-switching because they do not depend on a fixed vocabulary.
- Manual inspection of OOV-heavy errors highlights categories such as informal spelling, borrowed Russian/English tokens, and very short posts where static embeddings lack context.
