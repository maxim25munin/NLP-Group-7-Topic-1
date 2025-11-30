# Milestone 2 Report: Multilingual Language Identification

## Experimental setup
- **Task**: sentence-level language identification across 10 language/domain labels spanning Latin and Cyrillic scripts.
- **Data**: multilingual Wikipedia snippets preprocessed with Stanza using `scripts/prepare_multilingual_conllu_stanza.py`, with 4,000 held-out sentences (400 per label) for evaluation.
- **Baselines evaluated** via `scripts/evaluate_language_id_baselines.py`:
  1. **Rule-based heuristics** using Unicode script detection, diacritics, and curated cue words and affixes.
  2. **Character n-gram logistic regression** with TF–IDF features (non-DL).
  3. **XLM-RoBERTa fine-tuning** for multilingual sequence classification (DL).
- **Code runtime** is documented in `reports/Milestone 2 run.md`.

## Quantitative results
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- |
| Rule-based heuristics | 0.8925 | 0.929 | 0.893 | 0.900 |
| Char n-gram logistic regression | 0.9677 | 0.969 | 0.968 | 0.968 |
| XLM-R fine-tuning | 0.9663 | 0.968 | 0.966 | 0.966 |

**Class-level patterns**
- The **rule-based system** now leverages more targeted cues: it excels on German and Swahili heuristic variants but still leaks Kazakh, Latvian, and Yoruba examples into German due to shared Latin characters and named entities.
- The **character n-gram model** achieves the best overall accuracy, with perfect recall on Urdu and Kazakh and high precision on most labels; residual errors mirror orthographic similarity (e.g., English ↔ Yoruba) and short numeric snippets.
- **XLM-R** closely trails the n-gram model while slightly improving French/Swedish recall; its confusion profile is similar but shows more swaps between Swahili/Wolof and German for sentences dominated by borrowed vocabulary or names.

## Qualitative error analysis
- **Rule-based heuristics**: Mislabelled cases concentrate where affix and diacritic cues are absent. Kazakh and Latvian Cyrillic or transliterated segments are pulled into German, and English numeric lists lack signals for the heuristic rules.
- **Character n-grams**: Errors typically involve near-neighbour scripts or shared Latin strings (English → Yoruba heuristic, Swedish → English heuristic) and encyclopedic lists with few distinctive n-grams.
- **XLM-R fine-tuning**: Transformer predictions better separate stanza vs. heuristic variants but occasionally over-index on high-resource patterns, sending Swahili or Wolof into German or English when context is sparse or dominated by loanwords.

## Operational comparison
- **Rule-based heuristics**: Minimal compute and fully interpretable; surprisingly competitive (89% accuracy) after expanding affix and script cues, but scaling to new languages still demands linguistic expertise and manual maintenance.
- **Character n-gram logistic regression**: Lightweight training/inference with small memory footprint; scales with labelled data and requires little language expertise beyond tokenisation. May struggle with domain shift or unseen scripts.
- **XLM-R fine-tuning**: Strong accuracy and cross-script robustness; requires GPUs, longer training times, and higher serving costs. Model size complicates on-device deployment but offers adaptability to new domains with limited data.

## Takeaways for Milestone 2 Improvement
- **Adopt the char n-gram logistic regression** as the primary baseline: it edges out XLM-R in accuracy while remaining cheap and simple to deploy.
- **Use XLM-R selectively** for settings with extreme domain shift or code-switching where transfer learning benefits may justify the compute overhead.
- **Iterate on the rule-based system** as a fallback validator: its interpretability is valuable for rapid diagnostics, but it needs richer cues (e.g., character bigrams for Cyrillic) to avoid over-predicting German on transliterated text.
