# fastText OOD Language Identification Experiment

This notebook follows the Question 1 guidance by evaluating pretrained fastText embeddings on five languages: Kazakh, Latvian, Swedish, Yoruba, and Urdu. Wikipedia-derived CoNLL-U data are used for Latvian, Swedish, Yoruba, and Urdu, while the OOD hate-speech/social-media corpora come from `data/kazakh_hate_speech_fasttext.csv` and `data/latvian_comments_fasttext.csv`. The goal is to illustrate how relying on pretrained fastText embeddings for language identification can break when confronting non-Wikipedia, out-of-distribution content across multiple languages.

## 1. Setup

The notebook expects:

- Wikipedia-derived CoNLL-U files for Latvian, Swedish, Yoruba, and Urdu under `data/<lang>/*.conllu`.
- OOD hate-speech/social-media CSV files at `data/kazakh_hate_speech_fasttext.csv` and `data/latvian_comments_fasttext.csv` with columns `text` and `label`.
- Pretrained fastText binary models saved as `cc.<lang>.300.bin` in `models/fasttext/` (or adjust the paths below). Models are provided for all five target languages.


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
WIKI_LANGUAGES = ["kazakh", "latvian", "swedish", "yoruba", "urdu"]
OOD_LANGUAGES = ["kazakh", "latvian"]
LANGUAGES = sorted(set(WIKI_LANGUAGES + OOD_LANGUAGES))

FASTTEXT_LANGUAGE_CODES: Dict[str, str] = {
    "kazakh": "kk",
    "latvian": "lv",
    "swedish": "sv",
    "yoruba": "yo",
    "urdu": "ur",
}

OOD_FILES: Dict[str, Path] = {
    "kazakh": DATA_DIR / "kazakh_hate_speech_fasttext.csv",
    "latvian": DATA_DIR / "latvian_comments_fasttext.csv",
}

# Optional: cap the number of sentences per language to keep the notebook fast
MAX_SENTENCES_PER_LANGUAGE: Optional[int] = 2000

```

## 2. Data loading helpers

We reuse the Milestone 2 preprocessing assumptions: Wikipedia sentences are stored in CoNLL-U format with a `# text = ...` field. The hate-speech corpora are simple CSVs. Language labels are derived from the parent directory names for the Wikipedia data and set explicitly for each OOD set to test language identification robustness across Kazakh and Latvian.


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
            buffer.append(line[len("# text = " ) :])
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

```

## 3. fastText utilities

The helpers below load language-specific fastText models, convert sentences to averaged word vectors, and compute out-of-vocabulary (OOV) rates for qualitative error analysis.


```python
def load_fasttext_models(
    model_dir: Path, languages: Sequence[str], code_lookup: Optional[Dict[str, str]] = None
) -> Dict[str, fasttext.FastText._FastText]:
    """Load fastText models for the specified languages.

    The function expects files named `cc.<lang>.300.bin` inside `model_dir`. If a
    model is missing, a warning is emitted and the language is skipped.
    """

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
    """Convert sentences to feature matrices using language-specific models.

    If `language_hint` is provided, that model is used for all texts (useful for OOD
    Kazakh-only evaluation). Otherwise the function attempts to match each sample's
    language label to a loaded model and will raise an error if the model is
    missing to avoid silently falling back to an unintended language.
    """
    if language_hint:
        if language_hint not in models:
            raise ValueError(
                f"language_hint={language_hint!r} not found in loaded models: {sorted(models)}"
            )
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
):
    language_labels = None if language_hint else labels
    features = extract_fasttext_features(texts, models, language_labels=language_labels, language_hint=language_hint)
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

The next cell loads the Wikipedia in-distribution (ID) data for Kazakh, Latvian, Swedish, Yoruba, and Urdu, then performs a reproducible train/validation split. It also loads the Kazakh and Latvian hate-speech/social-media OOD datasets, which are held out entirely for OOD evaluation.


```python
ood_sets: Dict[str, pd.DataFrame] = {}
for lang in OOD_LANGUAGES:
    path = OOD_FILES.get(lang)
    if path is None:
        warnings.warn(f"No OOD path configured for language: {lang}")
        continue
    try:
        ood_sets[lang] = load_hate_speech_dataset(path, language=lang)
        print(f"Loaded {len(ood_sets[lang])} {lang} OOD sentences (held out for OOD evaluation)")
    except FileNotFoundError as exc:
        warnings.warn(str(exc))

if not ood_sets:
    raise FileNotFoundError("No OOD datasets were loaded. Add CSVs to data/<language>_*.csv or adjust OOD_FILES.")

# Combined view for downstream evaluation
ood_df = pd.concat(ood_sets.values(), ignore_index=True)
print(f"Combined OOD examples: {len(ood_df)} across {len(ood_sets)} languages")

wiki_df = load_multilingual_wikipedia(
    DATA_DIR,
    languages=WIKI_LANGUAGES,
    max_sentences_per_language=MAX_SENTENCES_PER_LANGUAGE,
    seed=RANDOM_SEED,
)
print(f"Loaded {len(wiki_df)} Wikipedia sentences across {wiki_df.label.nunique()} languages")

train_df, test_df = train_test_split(
    wiki_df, test_size=0.2, random_state=RANDOM_SEED, stratify=wiki_df.label
)
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")


```

    Loaded 10150 kazakh OOD sentences (held out for OOD evaluation)
    Loaded 1281158 latvian OOD sentences (held out for OOD evaluation)
    Combined OOD examples: 1291308 across 2 languages
    Loaded 10000 Wikipedia sentences across 5 languages
    Train size: 8000, Test size: 2000
    

## 6. Load pretrained fastText models

Download the `cc.<lang>.300.bin` files from the [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) repository and place them in `models/fasttext/` before running this cell. All five languages used in the experiment need to be available.


```python
fasttext_models = load_fasttext_models(
    FASTTEXT_MODEL_DIR, languages=LANGUAGES, code_lookup=FASTTEXT_LANGUAGE_CODES
)
print(f"Loaded fastText models for: {', '.join(sorted(fasttext_models))}")
```

    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    

    Loaded fastText models for: kazakh, latvian, swedish, urdu, yoruba
    

    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    

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
    

    In-distribution accuracy: 0.9980
    {
      "kazakh": {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 400.0
      },
      "latvian": {
        "precision": 0.9950124688279302,
        "recall": 0.9975,
        "f1-score": 0.9962546816479401,
        "support": 400.0
      },
      "swedish": {
        "precision": 1.0,
        "recall": 0.9975,
        "f1-score": 0.9987484355444305,
        "support": 400.0
      },
      "urdu": {
        "precision": 1.0,
        "recall": 1.0,
        "f1-score": 1.0,
        "support": 400.0
      },
      "yoruba": {
        "precision": 0.995,
        "recall": 0.995,
        "f1-score": 0.995,
        "support": 400.0
      },
      "accuracy": 0.998,
      "macro avg": {
        "precision": 0.998002493765586,
        "recall": 0.998,
        "f1-score": 0.9980006234384741,
        "support": 2000.0
      },
      "weighted avg": {
        "precision": 0.998002493765586,
        "recall": 0.998,
        "f1-score": 0.9980006234384742,
        "support": 2000.0
      }
    }
    

## 8. Evaluate on hate-speech/social-media corpora (OOD)

The classifier trained on Wikipedia data is tested on the held-out hate-speech/social-media corpora. Because each corpus is monolingual, the `language_hint` forces the matching fastText model for embedding extraction, revealing how poorly the Wikipedia-trained embeddings transfer for each target language.


```python
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
    print(f"OOD accuracy ({lang} hate speech/social media): {ood_eval['accuracy']:.4f}")
    print(json.dumps(ood_eval["report"], indent=2))

# Combined OOD macro view without language hints to show overall transfer gaps
combined_ood_eval = evaluate_fasttext_classifier(
    fasttext_clf, ood_df.text.tolist(), ood_df.label.tolist(), fasttext_models
)
print(f"Macro OOD accuracy across languages: {combined_ood_eval['accuracy']:.4f}")
print(json.dumps(combined_ood_eval["report"], indent=2))


```

    OOD accuracy (kazakh hate speech/social media): 0.9959
    {
      "kazakh": {
        "precision": 1.0,
        "recall": 0.9958620689655172,
        "f1-score": 0.9979267449896337,
        "support": 10150.0
      },
      "yoruba": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
      },
      "accuracy": 0.9958620689655172,
      "macro avg": {
        "precision": 0.5,
        "recall": 0.4979310344827586,
        "f1-score": 0.49896337249481687,
        "support": 10150.0
      },
      "weighted avg": {
        "precision": 1.0,
        "recall": 0.9958620689655172,
        "f1-score": 0.9979267449896337,
        "support": 10150.0
      }
    }
    OOD accuracy (latvian hate speech/social media): 0.9504
    {
      "kazakh": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
      },
      "latvian": {
        "precision": 1.0,
        "recall": 0.9504417097656963,
        "f1-score": 0.9745912477229288,
        "support": 1281158.0
      },
      "yoruba": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
      },
      "accuracy": 0.9504417097656963,
      "macro avg": {
        "precision": 0.3333333333333333,
        "recall": 0.3168139032552321,
        "f1-score": 0.3248637492409763,
        "support": 1281158.0
      },
      "weighted avg": {
        "precision": 1.0,
        "recall": 0.9504417097656963,
        "f1-score": 0.9745912477229289,
        "support": 1281158.0
      }
    }
    Macro OOD accuracy across languages: 0.9508
    {
      "kazakh": {
        "precision": 0.9988142292490119,
        "recall": 0.9958620689655172,
        "f1-score": 0.9973359644795264,
        "support": 10150.0
      },
      "latvian": {
        "precision": 1.0,
        "recall": 0.9504417097656963,
        "f1-score": 0.9745912477229288,
        "support": 1281158.0
      },
      "yoruba": {
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0,
        "support": 0.0
      },
      "accuracy": 0.950798725013707,
      "macro avg": {
        "precision": 0.6662714097496706,
        "recall": 0.6487679262437379,
        "f1-score": 0.6573090707341517,
        "support": 1291308.0
      },
      "weighted avg": {
        "precision": 0.9999906795488586,
        "recall": 0.950798725013707,
        "f1-score": 0.9747700268175209,
        "support": 1291308.0
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
      <td>0.9677</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fastText embeddings</td>
      <td>0.9980</td>
      <td>0.950442</td>
      <td>0.047558</td>
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

We compute OOV rates for Wikipedia vs. the OOD corpora, examine vocabulary overlap by language, and capture a sample of misclassified OOD examples to understand failure modes such as slang, code-switching, and short utterances.


```python
oov_rows: List[Dict[str, object]] = []
for lang, model in fasttext_models.items():
    wiki_subset = test_df[test_df.label == lang]
    if wiki_subset.empty:
        continue

    wiki_oov = calculate_oov_rate(wiki_subset.text.tolist(), model)
    ood_subset = ood_sets.get(lang)
    ood_oov = calculate_oov_rate(ood_subset.text.tolist(), model) if ood_subset is not None else np.nan

    train_subset = train_df[train_df.label == lang]
    wiki_vocab = set(" ".join(train_subset.text.tolist()).split())
    ood_vocab = set(" ".join(ood_subset.text.tolist()).split()) if ood_subset is not None else set()
    vocab_overlap = len(wiki_vocab & ood_vocab) / max(len(ood_vocab), 1) if ood_vocab else np.nan

    oov_rows.append(
        {
            "language": lang,
            "wiki_oov_rate": wiki_oov,
            "ood_oov_rate": ood_oov,
            "vocab_overlap": vocab_overlap,
            "ood_unique_terms": len(ood_vocab - wiki_vocab) if ood_vocab else np.nan,
        }
    )

summary_df = pd.DataFrame(oov_rows)
print(summary_df)

error_frames: List[pd.DataFrame] = []
for lang, eval_result in ood_evals.items():
    errors = collect_misclassifications(
        ood_sets[lang].text.tolist(),
        ood_sets[lang].label.tolist(),
        eval_result["predictions"],
        limit=20,
    ).assign(language=lang)
    error_frames.append(errors)

error_df = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
error_df.head()

```

      language  wiki_oov_rate  ood_oov_rate  vocab_overlap  ood_unique_terms
    0   kazakh       0.144956      0.084897       0.070526           30299.0
    1  latvian       0.159612      0.313091       0.003104         3177654.0
    2  swedish       0.128034           NaN            NaN               NaN
    3     urdu       0.091078           NaN            NaN               NaN
    4   yoruba       0.298034           NaN            NaN               NaN
    




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
      <th>language</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ұнатсаң лайк</td>
      <td>kazakh</td>
      <td>yoruba</td>
      <td>2</td>
      <td>kazakh</td>
    </tr>
    <tr>
      <th>1</th>
      <td>лубөйің алш</td>
      <td>kazakh</td>
      <td>yoruba</td>
      <td>2</td>
      <td>kazakh</td>
    </tr>
    <tr>
      <th>2</th>
      <td>тeгін лайкты aямайықшы</td>
      <td>kazakh</td>
      <td>yoruba</td>
      <td>3</td>
      <td>kazakh</td>
    </tr>
    <tr>
      <th>3</th>
      <td>өшірмей тұрғанда оқыңыздар https vk cc xho</td>
      <td>kazakh</td>
      <td>yoruba</td>
      <td>7</td>
      <td>kazakh</td>
    </tr>
    <tr>
      <th>4</th>
      <td>тапсырыс беру сілтемесі https vk cc mhvfi</td>
      <td>kazakh</td>
      <td>yoruba</td>
      <td>7</td>
      <td>kazakh</td>
    </tr>
  </tbody>
</table>
</div>



## 11. Takeaways

- fastText embeddings trained on Wikipedia/Common Crawl are sensitive to domain and vocabulary shift; expect lower accuracy on OOD hate-speech content than on in-distribution Wikipedia text.
- Character n-gram TF-IDF baselines are often more robust to slang, profanity, and code-switching because they do not depend on a fixed vocabulary.
- Manual inspection of OOV-heavy errors highlights how domain-specific slang and transliteration variants can break pretrained embeddings.
