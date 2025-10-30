# NLP-Topic-1-Milestone-1

Topic 1: Language Identification &mdash; Milestone 1 focuses on preparing core
text datasets. This repository currently contains a script that produces
multilingual CoNLL-U formatted samples of Wikipedia articles to prototype the
preprocessing pipeline used in downstream experiments. The script downloads
raw dumps from Hugging Face, applies light MediaWiki-aware cleaning, and
creates token-level placeholder annotations so every CoNLL-U column is
populated for experimentation.

## Repository layout

* `scripts/prepare_multilingual_conllu` &mdash; downloads, cleans, tokenises, and exports
  Wikipedia articles as CoNLL-U sentences for a range of languages. The export
  step fabricates lemmas, UPOS/XPOS tags, morphological features, dependency
  heads, and `TokenId`/`Lang` metadata using deterministic heuristics so that
  downstream tools expecting fully populated CoNLL-U rows continue to work.
* `data/<language>/<language>_wikipedia.conllu` &mdash; default output locations of the
  processed sentences for each supported language (see table below).
* `docs/data_preparation.md` &mdash; additional notes on dataset construction and
  observed quality issues.

## Requirements

The preparation script targets Python 3.9+ and relies on the following Python
package:

* `datasets` (required when downloading from Hugging Face)

To install the extra dependency into your environment:

```bash
python -m pip install datasets
```

## Running the converter

The script exports up to 10,000 sentences by default. It downloads the public
`wikimedia/wikipedia` dataset hosted on Hugging Face, applies light cleaning,
tokenises the text, and writes the result to disk as a CoNLL-U corpus.

Each sentence in the output file is preceded by stable `# sent_id` and `# text`
comments. Tokens receive basic guesses for lemmas and universal POS tags, and
the dependency structure defaults to a simple chain (or attaches punctuation to
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
python scripts/prepare_multilingual_conllu.py

# Generate a German sample with custom limits
python scripts/prepare_multilingual_conllu.py \
  --language de \
  --max-sentences 25000 \
  --min-tokens 5

# Override both the Hugging Face subset and the output location
python scripts/prepare_multilingual_conllu.py \
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

The script requires network access to download the dataset. If the `datasets`
package is unavailable, install it as shown above before running the converter.

When the optional dependency cannot be imported, the script raises a runtime
error prompting you to install it before retrying.

## Notebook usage

The script detects when it is launched inside a Jupyter notebook (via
`ipykernel_launcher.py`) and automatically ignores the notebook's own command line
arguments. You can therefore run the pipeline in a notebook cell with:

```python
from scripts.prepare_multilingual_conllu import main

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
