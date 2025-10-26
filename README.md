# NLP-Topic-1-Milestone-1-Kazakh

Topic 1: Language Identification &mdash; Milestone 1 focuses on preparing core
text datasets. This repository currently contains a script that produces a
CoNLL-U formatted sample of the Kazakh Wikipedia to prototype the preprocessing
pipeline.

## Repository layout

* `scripts/prepare_kazakh_conllu.py` &mdash; converts Kazakh Wikipedia articles into a
  CoNLL-U corpus.
* `data/kazakh/kazakh_wikipedia.conllu` &mdash; default output location of the processed
  sentences.
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
`wikimedia/wikipedia` dataset hosted on Hugging Face and converts it into a
lightly cleaned, tokenised CoNLL-U corpus.

```bash
# Download from Hugging Face and write to data/kazakh/kazakh_wikipedia.conllu
python scripts/prepare_kazakh_conllu.py

# Increase the number of exported sentences and require longer sentences
python scripts/prepare_kazakh_conllu.py --max-sentences 25000 --min-tokens 5
```

The script requires network access to download the dataset. If the `datasets`
package is unavailable, install it as shown above before running the converter.

## Notebook usage

The script detects when it is launched inside a Jupyter notebook (via
`ipykernel_launcher.py`) and automatically ignores the notebook's own command line
arguments. You can therefore run the pipeline in a notebook cell with:

```python
from scripts.prepare_kazakh_conllu import main

main([])  # exports using default settings
```

Pass a list of arguments to override specific options, mirroring the CLI
behaviour:

```python
main(["--max-sentences", "500", "--output", "data/kazakh/sample.conllu"])
```
