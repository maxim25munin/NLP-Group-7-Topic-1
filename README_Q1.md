# fastText OOD Language ID: Q1 Findings

## Why pretrained fastText embeddings are risky as non-DL features

- **Headline accuracy still masks domain brittleness.** Wikipedia training across five languages reaches 0.9980 accuracy, yet OOD hate-speech/social-media results range from 0.9840 (Latvian) to 1.0000 (Swedish/Urdu) with a combined macro OOD accuracy of 0.9863—a **0.0117 drop** from in-domain performance.[^id-performance][^ood-results][^tfidf-comparison]
- **OOV coverage diverges by language and domain.** Latvian OOD text shows a **24.1% OOV rate** with 469,222 vocabulary items unseen in Wikipedia, while Yoruba hits **37.9% OOV** and 20,801 OOD-only terms; Kazakh remains lower at 8.49% OOV with 30,299 unique OOD terms.[^oov] Character n-gram features avoid this reliance on pretrained token coverage.
- **Short or slang-heavy messages remain brittle.** Misclassified OOD samples (e.g., two-token Kazakh posts labeled as Yoruba) show that limited context and transliteration variants can still confuse the averaged embeddings despite high headline accuracy.[^error-examples]

## Quantitative evaluation on OOD hate speech/social media

- **Setup.** Multinomial logistic regression trained on averaged fastText vectors from 10,000 Wikipedia sentences (8,000 train / 2,000 test) across five languages and evaluated against 10,150 Kazakh, 110,607 Latvian, 450 Swedish, 4,856 Yoruba, and 6,365 Urdu hate-speech/social-media sentences using language-specific embedding models.[^setup] The hate-speech dataset is sourced from the PeerJ Computer Science study on offensive language detection.[^kazakh-source]
- **Results.**
  - Wikipedia (ID) accuracy: **0.9980** with macro F1 **0.998**; Yoruba is the weakest class with precision and recall at **0.995**.[^id-performance]
  - OOD accuracies by language: **0.9959** (Kazakh), **0.9840** (Latvian), **1.0000** (Swedish), **1.0000** (Urdu), **0.9996** (Yoruba).[^ood-results]
  - Combined OOD accuracy across all five languages: **0.9863**, a **0.0117** absolute drop from Wikipedia performance; Milestone 2 TF–IDF remains at 0.9677 in-domain with no reported OOD score.[^ood-results][^tfidf-comparison]

## Manual error analysis highlights

- **OOV remains meaningful despite high accuracy.** Latvian OOD text shows a 24.1% OOV rate and 469,222 OOD-only vocabulary items, while Yoruba hits 37.9% OOV with 20,801 unique OOD terms and Kazakh sits at 8.49% OOV with 30,299 unique OOD terms, underscoring how coverage differences can destabilize embeddings when labels are more diverse.[^oov]
- **Single-token/short posts still fail.** The misclassified two-token Kazakh sample labeled Yoruba illustrates how transliteration and brevity can override the model's otherwise strong metrics.[^error-examples]

## Recommendation

Use character n-gram features or contextual models (e.g., XLM-R) as baselines. If fastText is retained, pair it with character-level features, verify coverage on target domains, and report macro metrics to avoid overstating robustness on single-language OOD evaluations. The five-language hate-speech evaluation already surfaces coverage gaps (e.g., Latvian vs. Yoruba OOV), but expanding OOD sources further can strengthen per-language diagnostics and ensure fair comparisons across baselines.

## Hypothesis check: are pretrained fastText embeddings a safe non-DL baseline?

The Q1 hypothesis warns that pretrained fastText embeddings may be brittle when repurposed as fixed features. The five-language OOD study mostly supports this caution: despite a modest 1.17-point accuracy drop from Wikipedia to hate-speech/social-media domains, the aggregate hides structural fragility. Latvian OOD accuracy falls to 0.9840 with a 24.1% OOV rate and 469,222 unseen terms, while Yoruba shows an even higher 37.9% OOV rate despite near-perfect accuracy—evidence that coverage gaps can resurface on more varied tasks.【F:reports/fasttext_ood_language_id_experiment. run 28.12.2025.md†L326-L373】【F:reports/fasttext_ood_language_id_experiment. run 28.12.2025.md†L527-L565】【F:reports/fasttext_ood_language_id_experiment. run 28.12.2025.md†L819-L824】 The manual errors—short Kazakh posts mislabeled as Yoruba—underline how transliteration and sparse context can still derail the embeddings.【F:reports/fasttext_ood_language_id_experiment. run 28.12.2025.md†L856-L895】 In sum, while headline metrics look strong, the error modes and OOV skew validate the hypothesis: pretrained fastText vectors alone are not a dependable non-DL baseline without complementary character-level features and per-domain coverage checks.

[^setup]: Experiment configuration and dataset sizes from the fastText OOD language ID run (lines 330–373) in `reports/fasttext_ood_language_id_experiment. run 28.12.2025.md`.
[^id-performance]: In-domain Wikipedia accuracy and per-language observations (lines 422–467) in `reports/fasttext_ood_language_id_experiment. run 28.12.2025.md`.
[^ood-results]: Per-language and combined OOD performance metrics (lines 499–690) in `reports/fasttext_ood_language_id_experiment. run 28.12.2025.md`.
[^tfidf-comparison]: Comparison with Milestone TF–IDF baseline and performance drop calculation (lines 693–748) in `reports/fasttext_ood_language_id_experiment. run 28.12.2025.md`.
[^oov]: OOV analysis covering Wikipedia and hate-speech vocabularies plus unique term counts (lines 777–824) in `reports/fasttext_ood_language_id_experiment. run 28.12.2025.md`.
[^error-examples]: Manual error analysis excerpt on short OOD posts (lines 856–895) in `reports/fasttext_ood_language_id_experiment. run 28.12.2025.md`.
[^kazakh-source]: Source dataset link for Kazakh hate speech: https://peerj.com/articles/cs-3027/#supplemental-information.
