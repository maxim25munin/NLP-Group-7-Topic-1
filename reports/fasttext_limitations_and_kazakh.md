# fastText Embeddings on Non-English, Non-Wikipedia Data

## Why pretrained fastText vectors are a poor fit for this project

* **Domain mismatch** – The official fastText models are trained on Wikipedia and/or Common Crawl. Our multilingual corpora already show heavy code-switching, orthographic variation, and genre differences (tweets, forums, news rewrites). Static vectors trained on encyclopedic text cannot adapt to those lexical shifts, so downstream classifiers may over-index on well-formed tokens and miss noisy user-generated cues.
* **Limited contextual awareness** – fastText vectors are static; they cannot represent polysemy or context-dependent signals (e.g., Yoruba tone-marked words whose meaning flips when diacritics are dropped). The existing baselines in `scripts/refactored_language_id_baselines/` rely on character n-grams precisely to stay robust to such variation, something raw fastText averages cannot guarantee.
* **Script and normalization gaps** – Under-resourced languages often rely on digraphs, combining marks, or mixed Latin/Cyrillic scripts. Without careful Unicode normalization, the subword hashing in pretrained fastText models will split characters in unexpected ways, causing unpredictable representations for common tokens.
* **Lack of task alignment** – Using averaged pretrained embeddings as features for a non-DL classifier assumes linear separability in that space. For short, noisy sentences the signal-to-noise ratio is low; character n-gram TF–IDF features or task-specific fine-tuning tend to outperform raw averages because they emphasize discriminative surface cues.

## Attempted OOD evaluation (blocked by offline environment)

The plan was to probe fastText on a non-Wikipedia dataset (e.g., Yoruba social-media sentiment from the AfroSenti/AFRISENTI collection) by averaging pretrained vectors and training a logistic regression classifier. However, the environment cannot reach external package or model hosts (HTTPS requests to PyPI and Hugging Face are rejected with `403 Forbidden`), so the required `datasets`, `fasttext`, and pretrained vector downloads are unavailable. We therefore could not execute the experiment locally.

## Implications and next steps

1. When connectivity is restored, download the Yoruba (or other under-resourced) social-media dataset and the corresponding fastText vectors. Measure accuracy/F1 against the existing character n-gram baseline to quantify the gap.
2. Perform manual error inspection on misclassified posts, focusing on code-switching, diacritic stripping, and spelling variation to illustrate where static vectors fail.
3. Consider lightweight domain adaptation (e.g., continued training on in-domain text) or fallback to subword TF–IDF features when pretrained vectors underperform.

## fastText availability for Kazakh

fastText publishes pretrained vectors for Kazakh (`cc.kk.300.bin` / `.vec`) trained on Common Crawl and Wikipedia. These models are compatible with the Python `fasttext` loader and can be fine-tuned or averaged for downstream tasks once download access is available.
