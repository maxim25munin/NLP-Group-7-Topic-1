# Latvian and Yoruba fastText OOV histograms


This notebook computes per-sentence out-of-vocabulary (OOV) ratios for Latvian and Yoruba using their fastText models,
contrasting Wikipedia sentences with out-of-distribution (OOD) social-media corpora. It mirrors the analysis in the
fastText OOD language identification report and produces histograms for presentation-ready coverage illustrations.



```python

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import fasttext
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RANDOM_SEED = 13
np.random.seed(RANDOM_SEED)

current_dir = Path.cwd().resolve()
project_root_candidates = [current_dir, current_dir.parent, current_dir.parent.parent]
PROJECT_ROOT = next((p for p in project_root_candidates if (p / 'data').exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not locate the 'data' directory. Please run inside the repository.")

DATA_DIR = PROJECT_ROOT / 'data'
FASTTEXT_MODEL_DIR = PROJECT_ROOT / 'models' / 'fasttext'

LANG_CONFIG: Dict[str, Dict[str, object]] = {
    'latvian': {
        'code': 'lv',
        'ood_path': DATA_DIR / 'latvian_comments_fasttext_nat_only_20mb.csv',
    },
    'yoruba': {
        'code': 'yo',
        'ood_path': DATA_DIR / 'afrihate_yoruba_fasttext.csv',
    },
}
MAX_WIKI_SENTENCES: Optional[int] = 3000
BINS = np.linspace(0, 1, 21)

```


```python

def iter_conllu_sentences(path: Path) -> Iterable[str]:
    """Yield raw sentence strings from a CoNLL-U file."""
    buffer: List[str] = []
    for line in path.read_text(encoding='utf8').splitlines():
        if line.startswith('# text = '):
            buffer.append(line[len('# text = ' ) :])
        elif line.startswith('#'):
            continue
        elif not line.strip():
            if buffer:
                yield ' '.join(buffer).strip()
                buffer = []
        else:
            continue
    if buffer:
        yield ' '.join(buffer).strip()

def load_wikipedia_sentences(
    data_root: Path, language: str, max_sentences: Optional[int] = None
) -> List[str]:
    """Load Wikipedia sentences for a language from its CoNLL-U files."""
    lang_dir = data_root / language
    conllu_files = sorted(lang_dir.glob(f"{language}_wikipedia*.conllu"))
    if not conllu_files:
        raise FileNotFoundError(f"No Wikipedia CoNLL-U files found for {language} in {lang_dir}")
    sentences: List[str] = []
    for path in conllu_files:
        for sent in iter_conllu_sentences(path):
            sentences.append(sent)
            if max_sentences is not None and len(sentences) >= max_sentences:
                return sentences
    return sentences

def collect_fasttext_paths(
    model_dir: Path, languages: Sequence[str], code_lookup: Optional[Dict[str, str]] = None
) -> tuple[Dict[str, Path], List[tuple[str, str, Path]]]:
    """Return expected fastText binary paths and any missing ones."""
    code_lookup = code_lookup or {}
    model_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    missing: List[tuple[str, str, Path]] = []
    for lang in languages:
        code = code_lookup.get(lang, lang[:2])
        path = model_dir / f'cc.{code}.300.bin'
        paths[lang] = path
        if not path.exists():
            missing.append((lang, code, path))
    return paths, missing

def format_fasttext_download_instructions(
    missing: Sequence[tuple[str, str, Path]], model_dir: Path
) -> str:
    """Return human-friendly guidance for retrieving fastText binaries."""
    lines = [
        "fastText binary models are required for this notebook.",
        f"Place cc.<code>.300.bin files inside {model_dir} (create the directory if needed).",
        "Download archives from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/ with commands like:",
        f"    mkdir -p {model_dir.as_posix()}",
    ]
    for lang, code, path in missing:
        url = f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{code}.300.bin.gz"
        lines.append(f"    wget -O {path.as_posix()}.gz {url}")
        lines.append(f"    gunzip {path.as_posix()}.gz")
    return "\n".join(lines)

def load_fasttext_models(
    model_dir: Path, languages: Sequence[str], code_lookup: Optional[Dict[str, str]] = None
) -> Dict[str, fasttext.FastText._FastText]:
    """Load fastText models for the specified languages.

    Models are expected to follow the cc.<code>.300.bin naming convention inside model_dir.
    """
    code_lookup = code_lookup or {}
    models: Dict[str, fasttext.FastText._FastText] = {}
    paths, missing = collect_fasttext_paths(model_dir, languages, code_lookup=code_lookup)
    if missing:
        missing_lines = "\n".join(f"- {lang}: expected at {path}" for lang, _, path in missing)
        instructions = format_fasttext_download_instructions(missing, model_dir)
        raise FileNotFoundError(
            f"Missing fastText model binaries:\n{missing_lines}\n\n{instructions}"
        )
    for lang, path in paths.items():
        models[lang] = fasttext.load_model(path.as_posix())
    return models

def is_in_vocabulary(word: str, model: fasttext.FastText._FastText) -> bool:
    return model.get_word_id(word) != -1

def sentence_oov_fraction(text: str, model: fasttext.FastText._FastText) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    oov = sum(1 for tok in tokens if not is_in_vocabulary(tok, model))
    return oov / len(tokens)

def collect_oov_fractions(texts: Sequence[str], model: fasttext.FastText._FastText) -> List[float]:
    return [sentence_oov_fraction(text, model) for text in texts]

def plot_oov_hist(ax, values: Sequence[float], language: str, split: str):
    mean_value = float(np.mean(values)) if len(values) else 0.0
    ax.hist(values, bins=BINS, color='#4a90e2', edgecolor='black')
    ax.axvline(mean_value, color='crimson', linestyle='--', linewidth=2, label=f'mean={mean_value:.3f}')
    ax.set_title(f"{language.title()} - {split}")
    ax.set_xlabel('OOV fraction per sentence')
    ax.set_ylabel('Count')
    ax.set_xlim(0, 1)
    ax.legend()


```


```python

language_codes = {lang: cfg['code'] for lang, cfg in LANG_CONFIG.items()}
fasttext_models = load_fasttext_models(FASTTEXT_MODEL_DIR, LANG_CONFIG.keys(), code_lookup=language_codes)

datasets: Dict[str, Dict[str, List[str]]] = {}
for lang, cfg in LANG_CONFIG.items():
    wiki = load_wikipedia_sentences(DATA_DIR, lang, max_sentences=MAX_WIKI_SENTENCES)
    ood_df = pd.read_csv(cfg['ood_path'])
    if 'text' not in ood_df.columns:
        raise ValueError(f"Expected a 'text' column in {cfg['ood_path']}")
    datasets[lang] = {
        'wikipedia': wiki,
        'ood': ood_df['text'].astype(str).tolist(),
    }
    print(f"Loaded {len(wiki)} Wikipedia and {len(datasets[lang]['ood'])} OOD sentences for {lang}.")

```

    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
    

    Loaded 3000 Wikipedia and 110607 OOD sentences for latvian.
    Loaded 3000 Wikipedia and 4856 OOD sentences for yoruba.
    


```python

oov_measurements: Dict[str, Dict[str, List[float]]] = {}
summary_rows: List[Dict[str, object]] = []
for lang, splits in datasets.items():
    model = fasttext_models[lang]
    wiki_oov = collect_oov_fractions(splits['wikipedia'], model)
    ood_oov = collect_oov_fractions(splits['ood'], model)
    oov_measurements[lang] = {'wikipedia': wiki_oov, 'ood': ood_oov}
    for split_name, values in [('wikipedia', wiki_oov), ('ood', ood_oov)]:
        summary_rows.append(
            {
                'language': lang,
                'split': split_name,
                'mean_oov': float(np.mean(values)),
                'median_oov': float(np.median(values)),
                'num_sentences': len(values),
            }
        )
summary_df = pd.DataFrame(summary_rows)
summary_df

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
      <th>language</th>
      <th>split</th>
      <th>mean_oov</th>
      <th>median_oov</th>
      <th>num_sentences</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>latvian</td>
      <td>wikipedia</td>
      <td>0.148960</td>
      <td>0.133333</td>
      <td>3000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>latvian</td>
      <td>ood</td>
      <td>0.240599</td>
      <td>0.206897</td>
      <td>110607</td>
    </tr>
    <tr>
      <th>2</th>
      <td>yoruba</td>
      <td>wikipedia</td>
      <td>0.283485</td>
      <td>0.285714</td>
      <td>3000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>yoruba</td>
      <td>ood</td>
      <td>0.379225</td>
      <td>0.357143</td>
      <td>4856</td>
    </tr>
  </tbody>
</table>
</div>




```python

num_langs = len(LANG_CONFIG)
fig, axes = plt.subplots(num_langs, 2, figsize=(12, 4 * num_langs), sharex=True, sharey=True)
axes = np.array(axes).reshape(num_langs, 2)
for row_idx, (lang, splits) in enumerate(oov_measurements.items()):
    plot_oov_hist(axes[row_idx, 0], splits['wikipedia'], lang, 'Wikipedia')
    plot_oov_hist(axes[row_idx, 1], splits['ood'], lang, 'OOD social media')
fig.suptitle('Latvian and Yoruba fastText OOV coverage', fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.97))

```


    
![png](output_5_0.png)
    

