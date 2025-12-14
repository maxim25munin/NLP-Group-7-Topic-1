# fastText OOD Language ID: Q1 Findings

## Why pretrained fastText embeddings are risky as non-DL features

- **Metrics can look deceptively strong on single-language OOD tests.** The OOD Kazakh hate-speech set contains only Kazakh texts, so accuracy remains high at 0.9959 with just a -0.0021 drop from in-domain evaluation.[^ood-results][^tfidf-comparison] Macro precision/recall still collapse to 0.50 because the classifier effectively predicts only Kazakh, masking cross-language weakness.
- **OOV coverage remains uneven.** Using the Kazakh fastText model, the Wikipedia test split shows a 63.79% OOV rate, while the hate-speech corpus sits at 8.49% with 30,226 vocabulary items unseen in Wikipedia.[^oov] Character n-gram features avoid this reliance on pretrained token coverage.
- **Short or slang-heavy messages are brittle.** Misclassified OOD samples (e.g., two-token posts labeled as Yoruba) show that limited context and transliteration variants can still confuse the averaged embeddings despite high headline accuracy.[^error-examples]

## Quantitative evaluation on OOD Kazakh hate speech

- **Setup.** Multinomial logistic regression trained on averaged fastText vectors from 10,000 Wikipedia sentences (8,000 train / 2,000 test) across five languages and evaluated against 10,150 held-out Kazakh hate-speech sentences using the Kazakh embedding model for all OOD texts.[^setup] The hate-speech dataset is sourced from the PeerJ Computer Science study on offensive language detection.[^kazakh-source]
- **Results.**
  - Wikipedia (ID) accuracy: **0.9980** with macro F1 **0.998**; Yoruba is the weakest class with precision and recall at **0.995**.[^id-performance]
  - Kazakh hate speech (OOD) accuracy: **0.9959**, yet macro precision/recall fall to **≈0.50** because the single-language evaluation never tests non-Kazakh predictions.[^ood-results] The high headline number should not be interpreted as cross-language robustness.
  - Performance drop vs. Milestone TF–IDF: the table shows a **-0.0021** difference between in-domain and OOD accuracy for fastText, driven by the single-language OOD setup rather than genuine robustness; the Milestone 2 TF–IDF baseline remains at 0.9677 in-domain with no reported OOD score.[^tfidf-comparison]

## Manual error analysis highlights

- **OOV remains meaningful despite high accuracy.** A 63.79% OOV rate on the Wikipedia test split versus 8.49% on hate speech, along with 30,226 hate-speech-only vocabulary items, underscores how coverage differences can destabilize embeddings when labels are more diverse.[^oov]
- **Single-token/short posts still fail.** The misclassified two-token hate-speech sample labeled Yoruba illustrates how transliteration and brevity can override the model's otherwise strong metrics.[^error-examples]

## Recommendation

Use character n-gram features or contextual models (e.g., XLM-R) as baselines. If fastText is retained, pair it with character-level features, verify coverage on target domains, and report macro metrics to avoid overstating robustness on single-language OOD evaluations.

[^setup]: Experiment configuration and dataset sizes from the fastText OOD language ID run (lines 301–329) in `reports/fasttext_ood_language_id_experiment. run 14.12.2025.md`.
[^id-performance]: In-domain Wikipedia accuracy and per-language observations (lines 354–420) in `reports/fasttext_ood_language_id_experiment. run 14.12.2025.md`.
[^ood-results]: OOD Kazakh hate-speech performance metrics and macro averages (lines 424–470) in `reports/fasttext_ood_language_id_experiment. run 14.12.2025.md`.
[^tfidf-comparison]: Comparison with Milestone TF–IDF baseline and performance drop calculation (lines 474–546) in `reports/fasttext_ood_language_id_experiment. run 14.12.2025.md`.
[^oov]: OOV analysis covering Wikipedia and hate-speech vocabularies (lines 556–584) in `reports/fasttext_ood_language_id_experiment. run 14.12.2025.md`.
[^error-examples]: Manual error analysis excerpt on short OOD posts (lines 615–649) in `reports/fasttext_ood_language_id_experiment. run 14.12.2025.md`.
[^kazakh-source]: Source dataset link for Kazakh hate speech: https://peerj.com/articles/cs-3027/#supplemental-information.
