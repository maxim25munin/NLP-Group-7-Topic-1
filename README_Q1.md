# fastText OOD Language ID: Q1 Findings

## Why pretrained fastText embeddings are risky as non-DL features

- **Headline accuracy hides cross-language fragility.** Wikipedia training across five languages reaches 0.9980 accuracy, yet OOD hate-speech results swing from 0.9959 (Kazakh) to 0.9504 (Latvian), and the combined OOD macro accuracy is 0.9508—an overall **0.0476 drop** from in-domain performance.[^id-performance][^ood-results][^tfidf-comparison]
- **OOV coverage is highly uneven by domain.** Latvian hate-speech/social-media text shows a **31.3% OOV rate** and over 3.1 million vocabulary items unseen in Wikipedia, versus 8.49% OOV for Kazakh and 30,299 unique Kazakh OOD terms.[^oov] Character n-gram features avoid this reliance on pretrained token coverage.
- **Short or slang-heavy messages are brittle.** Misclassified OOD samples (e.g., two-token Kazakh posts labeled as Yoruba) show that limited context and transliteration variants can still confuse the averaged embeddings despite high headline accuracy.[^error-examples]

## Quantitative evaluation on OOD hate speech/social media

- **Setup.** Multinomial logistic regression trained on averaged fastText vectors from 10,000 Wikipedia sentences (8,000 train / 2,000 test) across five languages and evaluated against 10,150 held-out Kazakh hate-speech sentences plus 1,281,158 Latvian social-media/hate-speech comments using language-specific embedding models.[^setup] The hate-speech dataset is sourced from the PeerJ Computer Science study on offensive language detection.[^kazakh-source]
- **Results.**
  - Wikipedia (ID) accuracy: **0.9980** with macro F1 **0.998**; Yoruba is the weakest class with precision and recall at **0.995**.[^id-performance]
  - OOD Kazakh accuracy: **0.9959**, but macro precision/recall still collapse to **≈0.50** because the evaluation is monolingual.[^ood-results]
  - OOD Latvian accuracy: **0.9504**, revealing a substantial domain gap for social-media text.[^ood-results]
  - Combined OOD accuracy across Kazakh and Latvian: **0.9508**, a **0.0476** absolute drop from Wikipedia performance; Milestone 2 TF–IDF remains at 0.9677 in-domain with no reported OOD score.[^tfidf-comparison]

## Manual error analysis highlights

- **OOV remains meaningful despite high accuracy.** Latvian OOD text shows a 31.3% OOV rate and 3.18 million OOD-only vocabulary items, while Kazakh sits at 8.49% OOV with 30,299 unique OOD terms, underscoring how coverage differences can destabilize embeddings when labels are more diverse.[^oov]
- **Single-token/short posts still fail.** The misclassified two-token Kazakh sample labeled Yoruba illustrates how transliteration and brevity can override the model's otherwise strong metrics.[^error-examples]

## Recommendation

Use character n-gram features or contextual models (e.g., XLM-R) as baselines. If fastText is retained, pair it with character-level features, verify coverage on target domains, and report macro metrics to avoid overstating robustness on single-language OOD evaluations. To make OOD assessment meaningful beyond Kazakh and Latvian, add hate-speech datasets for Urdu, Yoruba, and Swedish so that:

- **Cross-language robustness is observable.** Multi-language OOD sets prevent inflated accuracy from a single-language benchmark and reveal transfer gaps for each language.
- **Coverage and slang issues are quantified per language.** Additional corpora let you measure OOV gaps, transliteration, and short-message brittleness in each target language instead of inferring from Kazakh alone.
- **Baselines are fairly compared.** Parallel OOD scores across languages allow character n-gram and contextual models to be judged against fastText on balanced macro metrics.

[^setup]: Experiment configuration and dataset sizes from the fastText OOD language ID run (lines 341–365) in `reports/fasttext_ood_language_id_experiment. run 16.12.2025.md`.
[^id-performance]: In-domain Wikipedia accuracy and per-language observations (lines 392–458) in `reports/fasttext_ood_language_id_experiment. run 16.12.2025.md`.
[^ood-results]: OOD Kazakh and Latvian performance metrics plus combined macro view (lines 462–579) in `reports/fasttext_ood_language_id_experiment. run 16.12.2025.md`.
[^tfidf-comparison]: Comparison with Milestone TF–IDF baseline and performance drop calculation (lines 641–652) in `reports/fasttext_ood_language_id_experiment. run 16.12.2025.md`.
[^oov]: OOV analysis covering Wikipedia and hate-speech vocabularies (lines 666–721) in `reports/fasttext_ood_language_id_experiment. run 16.12.2025.md`.
[^error-examples]: Manual error analysis excerpt on short OOD posts (lines 726–759) in `reports/fasttext_ood_language_id_experiment. run 16.12.2025.md`.
[^kazakh-source]: Source dataset link for Kazakh hate speech: https://peerj.com/articles/cs-3027/#supplemental-information.
