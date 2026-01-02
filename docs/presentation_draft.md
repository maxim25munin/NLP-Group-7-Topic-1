# Language Identification Project: Presentation Draft

## Project overview
- **Goal:** Build and evaluate multilingual sentence-level language identification spanning Latin and Cyrillic scripts.
- **Datasets:** Multilingual Wikipedia samples exported to CoNLL-U via `prepare_multilingual_conllu_stanza.py`, with Stanza annotations when available and heuristic fallbacks elsewhere. Evaluation uses 4,000 held-out sentences (400 per language) across 10 labels, as prepared in Milestone 1 and refined in Milestone 2.
- **Languages:** German (de), English (en), French (fr), Swedish (sv), Latvian (lv), Swahili (sw), Wolof (wo), Yoruba (yo), Kazakh (kk), Urdu (ur).

## Approaches compared
- **Rule-based heuristics**
  - Unicode script detection, diacritic patterns, cue words/affixes; computationally cheap and interpretable.
- **Character n-gram logistic regression (TF–IDF)**
  - Lightweight classical ML baseline on character n-grams; strong accuracy with small footprint.
- **XLM-RoBERTa fine-tuning**
  - Multilingual transformer for sequence classification; robust across scripts but compute-heavy.
- **fastText averaged embeddings (Q1 study)**
  - Multinomial logistic regression on pretrained word vectors; high ID accuracy but OOD brittleness due to OOV coverage gaps.

## Key results
- **Milestone 2 (in-domain Wikipedia evaluation, 10 languages)**
  - Rule-based: 0.89 accuracy.
  - Char n-gram logistic regression: **0.968 accuracy** (best overall) with near-perfect recall on several classes.
  - XLM-R fine-tuning: 0.966 accuracy; similar confusion profile, slightly better French/Swedish recall.
  - Insight: n-gram model offers best accuracy–cost balance; rules remain useful for diagnostics; XLM-R for harder domains.
- **Q1 fastText OOD study (hate speech/social media, 5 languages)**
  - Wikipedia (ID) accuracy: 0.998; OOD combined accuracy: **0.9863** (–0.0117 drop).
  - Per-language OOD accuracy: 0.9959 (kk), 0.9840 (lv), 1.0000 (sv, ur), 0.9996 (yo).
  - OOV rates highlight brittleness: Latvian 24.1% OOV (469k unseen terms); Yoruba 37.9% OOV (20,801 unseen terms).
  - Insight: headline metrics mask coverage fragility; pair fastText with character-level features and report macro metrics for OOD robustness.

## Error analysis highlights
- Rule-based errors: Latin-script overlap drags Kazakh/Latvian/Yoruba into German; numeric lists lack cues.
- Char n-gram errors: confusions among orthographically similar pairs (English↔Yoruba, Swedish↔English) and short numeric snippets.
- XLM-R errors: swaps between Swahili/Wolof and German for borrowed-vocabulary sentences; over-indexing on high-resource patterns.
- fastText OOD errors: short or transliterated posts (e.g., two-token Kazakh) mislabelled as Yoruba; OOV-driven fragility despite high accuracy.

## Recommendations for stakeholders
- **Primary baseline:** Character n-gram logistic regression for best accuracy–efficiency trade-off (Milestone 2).
- **Diagnostic fallback:** Maintain rule-based heuristics for interpretability and rapid checks; extend with richer cues for Cyrillic variants.
- **High-performance/shifted domains:** Deploy XLM-R when domain shift or code-switching warrants transformer robustness, accepting higher compute costs.
- **OOD coverage checks:** When using pretrained embeddings (e.g., fastText), audit OOV rates per target domain and pair with character-level features to mitigate brittleness.

## Next steps for the presentation
- Add visual confusion matrices comparing baselines on the 10-language Wikipedia split.
- Include OOV histograms for Latvian and Yoruba to illustrate fastText coverage gaps.
- Summarise deployment considerations (latency, hardware) for n-gram vs. XLM-R models.
- Prepare speaker notes emphasising when to trade accuracy for interpretability or compute efficiency.
