# Multilingual Wikipedia Dataset Preparation

This document summarises how the Wikipedia sample datasets committed under the
`data/` directory were produced for Milestone&nbsp;1.

## Data source

* **Primary source:** Hugging Face `wikimedia/wikipedia` dumps (e.g.
  `20231101.en`, `20231101.kk`, `20231101.sw`).
* **Local snapshot:** Processed CoNLL-U exports are versioned inside this
  repository under `data/<language>/<language>_wikipedia.conllu`.

The languages currently covered by the script are **de, en, fr, kk, lv, sv, sw,
ur, wo, yo**.  Each language has its own default subset identifier, output path
and `sent_id` prefix baked into the configuration.

## Processing pipeline

The preprocessing is implemented in `scripts/prepare_multilingual_conllu.py`.
The script expects internet access to download the selected subset via the
`datasets` library and performs the following steps:

1. **Parse configuration.** Command-line options define the language, subset,
   output path, record limits, and random seed.  Language-specific defaults are
   resolved automatically when optional flags are omitted.
2. **Load raw articles.** The Hugging Face dataset `wikimedia/wikipedia` is
   loaded using the requested subset and shuffled deterministically via
   `Dataset.shuffle(seed)`.
3. **Clean MediaWiki markup.** The helper removes `<ref>` blocks, HTML comments,
   nested template braces `{{…}}`, HTML tags, category/file lines (multiple
   languages), and MediaWiki link syntax (`[[target|label]]`).  Consecutive
   apostrophes used for emphasis are collapsed and whitespace inside digit
   sequences (e.g. `1 350 000`) is removed before normalising all whitespace.
4. **Sentence segmentation.** Cleaned text is split with the regular expression
   `(?<=[.!?؟۔])\s+`, which works for both Latin-script and right-to-left
   punctuation used by the supported languages.
5. **Tokenisation.** Tokens are extracted with the pattern `\w+|[^\w\s]`,
   keeping punctuation marks as standalone tokens and skipping empty strings.
6. **Filtering.** Sentences shorter than `--min-tokens` (default: 3 tokens) are
   discarded and the script stops after `--max-sentences` (default: 10 000)
   emitted sentences.
7. **CoNLL-U serialisation.** Each sentence is written with a stable
   `# sent_id` combining the language prefix, article id, and sentence index, as
   well as a `# text` comment.  Token lines only populate the `ID` and `FORM`
   columns while leaving linguistic annotations as `_` placeholders.  Sentences
   are separated by blank lines per the CoNLL-U specification.

Example command producing the default Kazakh export:

```bash
python scripts/prepare_multilingual_conllu.py \
  --language kk \
  --max-sentences 300 \
  --output data/kazakh/kazakh_wikipedia.conllu
```

> **Note:** The command requires outbound HTTPS access to download from Hugging
> Face.  When working offline, run the script on a networked machine and commit
> the resulting CoNLL-U file.

## Observed data quality issues

Manual inspection of the generated files highlights recurring artefacts worth
addressing in future iterations:

* **Hyphenated compounds:** words such as `мұнай-газ` or `Deutsch-Französisch`
  are split into three tokens (`token`, `-`, `token`).  Downstream models may
  require custom handling.
* **Ellipsis handling:** the regex tokeniser keeps `...` as three individual
  `.` tokens.
* **Sentence segmentation:** abbreviation- or quote-heavy sentences can still be
  split incorrectly because segmentation is driven purely by punctuation
  followed by whitespace.

Enhancing segmentation heuristics, normalising ellipses, and introducing
language-specific tokenisation rules would improve data quality for future
language identification experiments.
