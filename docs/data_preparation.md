# Kazakh Wikipedia Dataset Preparation

This document summarises how the Kazakh Wikipedia sample dataset used for
Milestone&nbsp;1 was produced.

## Data source

* **Primary source:** Hugging Face `wikimedia/wikipedia` dump (`20231101.kk`).
* **Local snapshot:** For reproducibility inside offline environments the
  processed CoNLL-U export is committed to the repository under
  `data/kazakh/kazakh_wikipedia.conllu`.

## Processing pipeline

The preprocessing is implemented in `scripts/prepare_kazakh_conllu.py`.  A
recent run was executed on a workstation with internet access using the
following steps:

1. **Load raw articles.** The script relies on the `datasets` library to stream
   the Hugging Face subset specified via `--subset` (default `20231101.kk`).
2. **Clean MediaWiki markup.** The helper removes `<ref>` blocks, HTML tags,
   HTML comments, template braces, category/file lines, and MediaWiki link
   syntax (`[[link|text]]`). Consecutive apostrophes (`''`, `'''`) used for
   emphasis are stripped, and gaps inside digit sequences such as `1 350 000`
   are collapsed into `1350000`.
3. **Sentence segmentation.** A lightweight regex split on sentence-final
   punctuation (`.?!`) is applied.
4. **Tokenisation.** Tokens are extracted with the pattern `\w+|[^\w\s]`,
   preserving punctuation as individual tokens.
5. **CoNLL-U serialisation.** Each sentence is written with a `# sent_id` and
   `# text` header, followed by tab-separated token lines where linguistic
   annotations are left as `_` placeholders.

Example command:

```bash
python scripts/prepare_kazakh_conllu.py \
  --subset 20231101.kk \
  --max-sentences 300 \
  --output data/kazakh/kazakh_wikipedia.conllu
```

> **Note:** The command requires outbound HTTPS access to download from
> Hugging Face. When working in a restricted environment, run the script
> elsewhere and commit the resulting CoNLL-U file.

## Observed data quality issues

Manual inspection of the CoNLL-U file highlighted the following artefacts that
should be addressed:

* **Hyphenated compounds:** words such as `мұнай-газ` are tokenised into three
  pieces (`мұнай`, `-`, `газ`). Downstream models may prefer merged tokens or a
  special handling strategy.
* **Ellipsis handling:** the naive tokeniser keeps `...` as three individual
  `.` tokens.
* **Sentence segmentation:** abbreviation- or quote-heavy sentences could be
  split incorrectly because the segmenter only looks for punctuation followed by
  whitespace.

Addressing these points (e.g., improved segmentation rules, normalising
ellipsis, and dedicated treatment of hyphenated words) will improve data quality
for downstream language identification experiments.
