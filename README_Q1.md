# fastText OOD Language ID: Q1 Findings

## Why pretrained fastText embeddings are risky as non-DL features

- **Domain-specific slang and code-switching are underrepresented in Wikipedia/Common Crawl pretraining.** Even after training on 20k Wikipedia sentences across 10 languages, the fastText+logistic baseline reaches 0.83 ID accuracy in-domain, but drops sharply on languages with poorer coverage (e.g., Yoruba recall 0.41).[^id-performance] Character n-gram baselines from Milestone 2 are less sensitive to this vocabulary gap because they do not depend on token lookup.
- **Out-of-vocabulary (OOV) exposure is severe for real-world text.** Kazakh hate-speech posts show only 7.6% vocabulary overlap with Wikipedia and introduce 30k unseen tokens; OOV rates spike to 55.82% even on the held-out Wikipedia split when using the Kazakh model as a single encoder.[^oov] Static embeddings discard these unknown pieces, erasing morphological cues critical for smaller or polysynthetic languages.
- **Language-ambiguous short texts confuse averaged embeddings.** Misclassified OOD samples are typically two-token cultural references that the model maps to Urdu or Latvian because the averaged vectors lack context and subword coverage is insufficient.[^error-examples] Non-DL baselines built on such sentence averages will replicate these boundary errors.

## Quantitative evaluation on OOD Kazakh hate speech

- **Setup.** Logistic regression trained on averaged fastText vectors from Wikipedia sentences in 10 languages; evaluated in-domain and on 10,150 OOD Kazakh hate-speech messages using the Kazakh fastText model for all OOD texts.[^setup] The hate-speech dataset is sourced from the PeerJ Computer Science study on offensive language detection.[^kazakh-source] Missing fastText binaries for most languages further highlight practical coverage issues.
- **Results.**
  - Wikipedia (ID) accuracy: **0.8293** overall, with strong Kazakh/French performance but weak Yoruba and Urdu due to vocabulary gaps.[^id-performance]
  - Kazakh hate speech (OOD) accuracy: **0.9773** using the Kazakh model, but the metric is inflated because the test set is single-language; macro precision/recall are 0.125/0.122, indicating the classifier predicts only Kazakh when out-of-distribution classes are absent.[^ood-results] A realistic multi-language OOD mix would expose larger degradation.
  - Performance drop vs. Milestone TF–IDF: fastText lags the character n-gram baseline by ~14.8 points in-domain (0.829 vs. 0.968) and lacks evidence of cross-domain robustness.[^tfidf-comparison]

## Manual error analysis highlights

- **High OOV on Wikipedia split (55.82%)** shows that even curated corpora contain many tokens absent from the pretrained Kazakh model, leading to noisy averages.[^oov]
- **Hate-speech-specific slang (30,121 unique tokens)**, including Latin-script acronyms and Russian loanwords, are unseen in Wikipedia and hence unmodeled by static vectors.[^oov]
- **Shortest OOD posts (2 tokens)** such as "ләйлі мәжнүн" are mapped to Urdu or Latvian because the embeddings lack disambiguating context; subword segmentation alone is insufficient for names and poetic references.[^error-examples]

## Recommendation

Use character n-gram features or contextual models (e.g., XLM-R) as baselines. If fastText is retained, restrict it to languages with verified coverage, augment with domain-specific training, and combine with subword character models to handle slang and code-switching.

[^setup]: Experiment configuration and dataset details from the fastText OOD language ID run (report lines 277–520). See `reports/fasttext_ood_language_id_experiment_run.md`.
[^id-performance]: In-domain Wikipedia accuracy and per-language observations (lines 340–437) in `reports/fasttext_ood_language_id_experiment_run.md`.
[^ood-results]: OOD Kazakh hate-speech performance metrics (lines 445–520) in `reports/fasttext_ood_language_id_experiment_run.md`.
[^tfidf-comparison]: Comparison with Milestone TF–IDF baseline (lines 524–596) in `reports/fasttext_ood_language_id_experiment_run.md`.
[^oov]: OOV analysis covering Wikipedia and hate-speech vocabularies (lines 601–634) in `reports/fasttext_ood_language_id_experiment_run.md`.
[^error-examples]: Manual error analysis excerpts on short OOD posts (lines 664–701) in `reports/fasttext_ood_language_id_experiment_run.md`.
[^kazakh-source]: Source dataset link for Kazakh hate speech: https://peerj.com/articles/cs-3027/#supplemental-information.
