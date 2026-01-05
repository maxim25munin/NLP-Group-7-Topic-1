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

## Deployment considerations: latency vs. hardware
- **Character n-gram (TF–IDF + logistic regression)**
  - **Hardware:** CPU-only; 10–20 MB model fits in laptop memory.
  - **Latency:** ~<5 ms/sentence on a single core; scales linearly with batch size, dominated by feature extraction.
- **XLM-R fine-tuning**
  - **Hardware:** GPU recommended; `base` needs ~1–2 GB VRAM for inference (more for batch >16). CPU-only runs are slow.
  - **Latency:** ~30–80 ms/sentence on an A10/T4 GPU; 300 ms+ on CPU without quantization or distillation.
- **Deployment takeaway:** Default to the n-gram model for general serving and on-CPU environments; reserve XLM-R for domain-shifted or robustness-critical deployments where the added latency and GPU cost are acceptable.

## Error analysis highlights
- Rule-based errors: Latin-script overlap drags Kazakh/Latvian/Yoruba into German; numeric lists lack cues.
- Char n-gram errors: confusions among orthographically similar pairs (English↔Yoruba, Swedish↔English) and short numeric snippets.
- XLM-R errors: swaps between Swahili/Wolof and German for borrowed-vocabulary sentences; over-indexing on high-resource patterns.
- fastText OOD errors: short or transliterated posts (e.g., two-token Kazakh) mislabelled as Yoruba; OOV-driven fragility despite high accuracy.

## Visual confusion matrices (10-language Wikipedia split)
- **Rule-based heuristics:** confusion remains concentrated among closely related Latin-script languages, with Kazakh/Latvian/Yoruba bleeding into German but strong precision on Urdu thanks to script cues.
  ![Rule-based heuristics confusion matrix](../reports/output_0_3.png)
- **Character n-gram logistic regression:** errors shrink markedly; primary confusions cluster around English↔Swedish and occasional Yoruba overlap, while Cyrillic Kazakh is cleanly separated.
  ![Character n-gram logistic regression confusion matrix](../reports/output_0_14.png)
- **XLM-R fine-tuning:** comparable to the n-gram model with slightly better French/Swedish separation and minimal cross-script leakage.
  ![XLM-R fine-tuning confusion matrix](../reports/output_0_17.png)

## fastText OOV coverage diagnostics
- **Latvian vs. Yoruba fastText OOV rates (social media):** distributions highlight heavy tails and the prevalence of unseen terms, reinforcing the risk of deploying embeddings without character-level backups.
  ![Latvian and Yoruba OOV histograms](../reports/output_5_0.png)
- **Mitigation takeaway:** treat the histogram spike of unseen tokens as a deployment red flag—pair fastText with character n-grams (as features or a backoff model) or swap to subword-aware embeddings to absorb rare/novel forms without relabelling efforts. The `scripts/evaluate_ood_xlmr_vs_fasttext.py --enable-fasttext-char-backoff` flag now automates this by routing high-OOV sentences through a character n-gram classifier.
- **Reproducibility:** generated via the notebook-style script `reports/latvian_yoruba_oov_histograms. run 4.1.2026.md`, which computes per-token OOV ratios from the OOD hate-speech corpus.

## Recommendations for stakeholders
- **Primary baseline:** Character n-gram logistic regression for best accuracy–efficiency trade-off (Milestone 2).
- **Diagnostic fallback:** Maintain rule-based heuristics for interpretability and rapid checks; extend with richer cues for Cyrillic variants.
- **High-performance/shifted domains:** Deploy XLM-R when domain shift or code-switching warrants transformer robustness, accepting higher compute costs.
- **OOD coverage checks:** When using pretrained embeddings (e.g., fastText), audit OOV rates per target domain and pair with character-level features to mitigate brittleness.

## Next steps for the presentation
- Pair the Latvian/Yoruba OOV histograms with a brief takeaway on mitigation (character features, subword models).
- Confirm the latency/hardware slide placement after key results to frame accuracy vs. compute trade-offs before recommendations.
- Prepare speaker notes emphasising when to trade accuracy for interpretability or compute efficiency.
