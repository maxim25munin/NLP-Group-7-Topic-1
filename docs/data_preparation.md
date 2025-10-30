# Multilingual Wikipedia Dataset Preparation

This document summarises how the Wikipedia sample datasets committed under the
`data/` directory were produced for Milestone&nbsp;1 and how to reproduce or extend
them with the accompanying preprocessing script.

## Prerequisites

* **Python packages:** Install the Hugging Face `datasets` library with
  `pip install datasets` before running the script.
* **Network access:** Downloading subsets from Hugging Face requires outbound
  HTTPS connectivity.  If you need to work offline, run the script on a
  networked machine first and commit the generated CoNLL-U files to this
  repository.

## Data source

* **Primary source:** Hugging Face `wikimedia/wikipedia` dumps (e.g.
  `20231101.en`, `20231101.kk`, `20231101.sw`).
* **Local snapshot:** Processed CoNLL-U exports are versioned inside this
  repository under `data/<language>/<language>_wikipedia.conllu`.

The languages currently covered by the script are **de, en, fr, kk, lv, sv, sw,
ur, wo, yo**.  Each language has its own default subset identifier, output path
and `sent_id` prefix baked into the configuration:

| Language code | Default subset | Output path | `sent_id` prefix |
| ------------- | -------------- | ----------- | ---------------- |
| `de`          | `20231101.de`  | `data/german/german_wikipedia.conllu`   | `de` |
| `en`          | `20231101.en`  | `data/english/english_wikipedia.conllu` | `en` |
| `fr`          | `20231101.fr`  | `data/french/french_wikipedia.conllu`   | `fr` |
| `kk`          | `20231101.kk`  | `data/kazakh/kazakh_wikipedia.conllu`   | `kk` |
| `lv`          | `20231101.lv`  | `data/latvian/latvian_wikipedia.conllu` | `lv` |
| `sv`          | `20231101.sv`  | `data/swedish/swedish_wikipedia.conllu` | `sv` |
| `sw`          | `20231101.sw`  | `data/swahili/swahili_wikipedia.conllu` | `sw` |
| `ur`          | `20231101.ur`  | `data/urdu/urdu_wikipedia.conllu`       | `ur` |
| `wo`          | `20231101.wo`  | `data/wolof/wolof_wikipedia.conllu`     | `wo` |
| `yo`          | `20231101.yo`  | `data/yoruba/yoruba_wikipedia.conllu`   | `yo` |

All of these defaults can be overridden with command-line arguments when
running the script.

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
7. **Heuristic linguistic enrichment.** Each token receives lightweight
   Universal Dependencies style annotations derived from simple rules:
   * `UPOS` and `XPOS` are inferred from casing, character classes, and common
     punctuation patterns.
   * Lemmas are lowercased forms of alphabetic tokens, leaving other symbols
     untouched.
   * Morphological `FEATS` mark properties such as `NumType=Card`,
     `Proper=Yes`, and coarse letter-case information when applicable.
   * Dependency heads default to the previous token while punctuation attaches
     to the nearest non-punctuation token; root tokens point to `0`.
   * Miscellaneous metadata includes a deterministic `TokenId` and language
     code.
8. **CoNLL-U serialisation.** Each sentence is written with a stable `# sent_id`
   combining the language prefix, article id, and sentence index, as well as a
   `# text` comment.  All ten CoNLL-U columns are filled using the heuristics
   above, and sentences are separated by blank lines per the CoNLL-U
   specification.

### Command-line interface

The script exposes several options to customise the export:

* `--language`: Two-letter language code defined in the table above (default:
  `kk`).
* `--subset`: Hugging Face subset identifier.  When omitted, the language
  default is used.
* `--max-sentences`: Maximum number of sentences to export (default: 10 000).
* `--min-tokens`: Minimum number of tokens a sentence must contain to be kept
  (default: 3).
* `--output`: Destination CoNLL-U file (default: language-specific path).
* `--seed`: Random seed controlling dataset shuffling (default: 13).

Example command producing the default Kazakh export:

```bash
python scripts/prepare_multilingual_conllu.py \
  --language kk \
  --max-sentences 300 \
  --min-tokens 5 \
  --seed 21 \
  --output data/kazakh/kazakh_wikipedia.conllu
```

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
