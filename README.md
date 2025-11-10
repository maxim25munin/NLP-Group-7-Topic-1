# NLP-Topic-1-Milestone-2

## Baseline evaluation script

[`scripts/evaluate_language_id_baselines.py`](scripts/evaluate_language_id_baselines.py)
implements three complementary baselines for the milestone 2 classification
task:

1. A rule-based system that combines Unicode script inspection with
   language-specific diacritics and keyword cues.
2. A classical machine-learning pipeline that feeds character n-gram TF–IDF
   features into a multinomial logistic regression classifier.
3. A deep-learning approach that fine-tunes an XLM-RoBERTa sequence
   classification head using Hugging Face `transformers`.

Each baseline reports quantitative metrics (accuracy, precision/recall/F1, and a
confusion matrix), qualitative misclassification examples, and an operational
comparison of scalability and cost. Example usage:

```bash
python scripts/evaluate_language_id_baselines.py \
  --data-root data \
  --max-sentences-per-language 2000 \
  --output-report reports/baseline_results.json
```

Fine-tuning the transformer requires optional dependencies from
`requirements-transformers.txt` (PyTorch, `transformers`, `datasets`, and
`accelerate`). Install them with:

```bash
pip install -r requirements-transformers.txt
```

The script automatically skips the deep-learning baseline when the dependencies
are unavailable.

# NLP-Topic-1-Milestone-1

Topic 1: Language Identification &mdash; Milestone 1 focuses on preparing core
text datasets. This repository currently contains a script that produces
multilingual CoNLL-U formatted samples of Wikipedia articles to prototype the
preprocessing pipeline used in downstream experiments. The script downloads raw
dumps from Hugging Face, removes MediaWiki-specific markup, segments the text
into sentences, and writes token-level annotations enriched either with Stanza
parses (when available) or deterministic heuristics. This ensures that every
CoNLL-U column is populated for experimentation, even when full linguistic
analyses are not obtainable.

## Repository layout

* `scripts/prepare_multilingual_conllu_stanza.py` &mdash; downloads, cleans,
  tokenises, and exports Wikipedia articles as CoNLL-U sentences for a range of
  languages. When the optional [Stanza](https://stanfordnlp.github.io/stanza/)
  models can be loaded, the script reuses their word tokens, lemmas,
  part-of-speech tags, features, and dependency parses. Otherwise it falls back
  to deterministic heuristics that fabricate lemmas, UPOS/XPOS tags,
  morphological features, dependency heads, and `TokenId`/`Lang` metadata so
  that downstream tools expecting fully populated CoNLL-U rows continue to
  work.
* `data/<language>/<language>_wikipedia.conllu` &mdash; default output locations of the
  processed sentences for each supported language (see table below).
* `docs/data_preparation.md` &mdash; additional notes on dataset construction and
  observed quality issues.

## Requirements

The preparation script targets Python 3.9+ and relies on the following Python
packages:

* `datasets` (required when downloading from Hugging Face)
* `stanza` (optional but recommended for higher-quality linguistic annotation)

To install the extra dependencies into your environment:

```bash
python -m pip install datasets stanza
```

## Running the converter

The script exports up to 10,000 sentences by default. It downloads the public
`wikimedia/wikipedia` dataset hosted on Hugging Face, applies light cleaning,
tokenises the text, and writes the result to disk as a CoNLL-U corpus. Kazakh
(`kk`) is the default language so running the script without arguments
recreates the dataset used in our experiments.

Each sentence in the output file is preceded by stable `# sent_id` and `# text`
comments. When Stanza annotations are available the script reuses their rich
token metadata; otherwise tokens receive heuristic lemmas, universal POS tags,
and dependency arcs that follow a simple chain (with punctuation attaching to
the most recent non-punctuation token). Output directories are created
automatically when needed.

Supported language codes, their default dataset subsets, and output paths:

| Language | Subset ID     | Default output path                       |
|----------|---------------|-------------------------------------------|
| de       | `20231101.de` | `data/german/german_wikipedia.conllu`     |
| en       | `20231101.en` | `data/english/english_wikipedia.conllu`   |
| fr       | `20231101.fr` | `data/french/french_wikipedia.conllu`     |
| kk       | `20231101.kk` | `data/kazakh/kazakh_wikipedia.conllu`     |
| lv       | `20231101.lv` | `data/latvian/latvian_wikipedia.conllu`   |
| sv       | `20231101.sv` | `data/swedish/swedish_wikipedia.conllu`   |
| sw       | `20231101.sw` | `data/swahili/swahili_wikipedia.conllu`   |
| ur       | `20231101.ur` | `data/urdu/urdu_wikipedia.conllu`         |
| wo       | `20231101.wo` | `data/wolof/wolof_wikipedia.conllu`       |
| yo       | `20231101.yo` | `data/yoruba/yoruba_wikipedia.conllu`     |

```bash
# Export the default Kazakh sample (10k sentences, >=3 tokens)
python scripts/prepare_multilingual_conllu_stanza.py

# Generate a German sample with custom limits
python scripts/prepare_multilingual_conllu_stanza.py \
  --language de \
  --max-sentences 25000 \
  --min-tokens 5

# Override both the Hugging Face subset and the output location
python scripts/prepare_multilingual_conllu_stanza.py \
  --language sw \
  --subset 20231101.sw \
  --output data/swahili/custom_sw_sample.conllu
```

Key options include:

* `--language` &mdash; language to process (defaults to `kk`/Kazakh).
* `--subset` &mdash; Hugging Face dataset subset identifier (defaults to the
  language-specific value shown above).
* `--max-sentences` &mdash; upper bound on the number of exported sentences.
* `--min-tokens` &mdash; minimum token length for a sentence to be kept.
* `--output` &mdash; destination path for the generated CoNLL-U file.
* `--seed` &mdash; random seed for shuffling prior to extraction.
* `--disable-stanza` &mdash; force the heuristic pipeline even when Stanza is
  installed.

The script requires network access to download the dataset. If `datasets` is
missing, it raises a runtime error prompting you to install the package. When
Stanza cannot be initialised for the requested language the script logs a
warning and automatically switches to the heuristic mode.

## Notebook usage

The script detects when it is launched inside a Jupyter notebook (via
`ipykernel_launcher.py`) and automatically ignores the notebook's own command line
arguments. You can therefore run the pipeline in a notebook cell with:

```python
from scripts.prepare_multilingual_conllu_stanza import main

main([])  # exports using default settings (Kazakh sample)
```

Pass a list of arguments to override specific options, mirroring the CLI
behaviour:

```python
main([
    "--language", "en",
    "--max-sentences", "500",
    "--output", "data/english/sample.conllu",
])
```
