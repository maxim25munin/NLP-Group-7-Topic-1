# fastText OOD Language ID: Q1 Findings

## Why pretrained fastText embeddings are risky as non-DL features

- **Metrics can look strong even when the task is under-specified.** The OOD Kazakh hate-speech set is single-language, so accuracy rises to 0.9995 and the comparison table shows only a -0.0007 drop from in-domain testing.[^ood-results][^tfidf-comparison] However, the macro precision/recall collapse to 0.50 because the classifier never predicts non-Kazakh labels, revealing that the apparent robustness is an artifact of the data, not true language-discrimination performance.
- **OOV coverage remains uneven.** Using the Kazakh fastText model, the Wikipedia test split still shows a 42.40% OOV rate, while the hate-speech corpus sits at 9.09% with 4,633 vocabulary items unseen in Wikipedia.[^oov] Character n-gram features avoid this reliance on pretrained token coverage.
- **Short or slang-heavy messages are brittle.** Misclassified OOD samples (e.g., a two-token post labeled as Yoruba) show that limited context and transliteration variants can still confuse the averaged embeddings despite high headline accuracy.[^error-examples]

## Quantitative evaluation on OOD Kazakh hate speech

- **Setup.** Multinomial logistic regression trained on averaged fastText vectors from 12,896 combined Wikipedia and Kazakh hate-speech sentences across five languages; evaluated on a 3,224-sample Wikipedia test split and a 2,030-sample Kazakh hate-speech OOD split that forces the Kazakh embedding model for all texts.[^setup] The hate-speech dataset is sourced from the PeerJ Computer Science study on offensive language detection.[^kazakh-source]
- **Results.**
  - Wikipedia (ID) accuracy: **0.9988** with macro F1 **0.999**; Yoruba is the weakest class with precision 0.99 but perfect recall.[^id-performance]
  - Kazakh hate speech (OOD) accuracy: **0.9995**, yet macro precision/recall drop to **0.50** because the single-language evaluation never tests non-Kazakh predictions.[^ood-results] The high headline number should not be interpreted as cross-language robustness.
  - Performance drop vs. Milestone TF–IDF: the table shows a **-0.0007** difference between in-domain and OOD accuracy for fastText, driven by the single-language OOD setup rather than genuine robustness; the Milestone 2 TF–IDF baseline remains at 0.9677 in-domain with no reported OOD score.[^tfidf-comparison]

## Manual error analysis highlights

- **OOV remains meaningful despite high accuracy.** A 42.40% OOV rate on the Wikipedia test split versus 9.09% on hate speech, along with 4,633 hate-speech-only vocabulary items, underscores how coverage differences can destabilize embeddings when labels are more diverse.[^oov]
- **Single-token/short posts still fail.** The misclassified two-token hate-speech sample labeled Yoruba illustrates how transliteration and brevity can override the model's otherwise strong metrics.[^error-examples]

## Recommendation

Use character n-gram features or contextual models (e.g., XLM-R) as baselines. If fastText is retained, pair it with character-level features, verify coverage on target domains, and report macro metrics to avoid overstating robustness on single-language OOD evaluations.

[^setup]: Experiment configuration and dataset sizes from the fastText OOD language ID run (lines 313–317, 341–353) in `reports/fasttext_ood_language_id_experiment_run. 13.12.2025.md`.
[^id-performance]: In-domain Wikipedia accuracy and per-language observations (lines 363–407) in `reports/fasttext_ood_language_id_experiment_run. 13.12.2025.md`.
[^ood-results]: OOD Kazakh hate-speech performance metrics and macro averages (lines 430–455) in `reports/fasttext_ood_language_id_experiment_run. 13.12.2025.md`.
[^tfidf-comparison]: Comparison with Milestone TF–IDF baseline and performance drop calculation (lines 460–523) in `reports/fasttext_ood_language_id_experiment_run. 13.12.2025.md`.
[^oov]: OOV analysis covering Wikipedia and hate-speech vocabularies (lines 545–570) in `reports/fasttext_ood_language_id_experiment_run. 13.12.2025.md`.
[^error-examples]: Manual error analysis excerpt on short OOD posts (lines 575–607) in `reports/fasttext_ood_language_id_experiment_run. 13.12.2025.md`.
[^kazakh-source]: Source dataset link for Kazakh hate speech: https://peerj.com/articles/cs-3027/#supplemental-information.
