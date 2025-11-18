# Milestone 2 Report: Multilingual Language Identification

## Experimental setup
- **Task**: sentence-level language identification across 10 language/domain labels spanning Latin and Cyrillic scripts.
- **Data preparation**: multilingual Wikipedia snippets preprocessed with Stanza using the provided `scripts/prepare_multilingual_conllu.py` utility. Each language contributed up to 400 held-out sentences for evaluation.
- **Baselines evaluated** via `scripts/evaluate_language_id_baselines.py`:
  1. **Rule-based heuristics** using script detection, diacritics, and curated cue words.
  2. **Character n-gram logistic regression** with TF–IDF features.
  3. **XLM-RoBERTa fine-tuning** for sequence classification (multilingual transformer).

## Quantitative results
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- |
| Rule-based heuristics | 0.0990 | 0.022 | 0.099 | 0.035 |
| Char n-gram logistic regression | 0.5481 | 0.544 | 0.548 | 0.544 |
| XLM-R fine-tuning | 0.5829 | 0.528 | 0.583 | 0.489 |

**Class-level patterns**
- The **rule-based system** only succeeds on Swahili and Yoruba variants, with all other languages mislabelled, demonstrating the fragility of hand-crafted cues when languages share scripts or lack distinctive diacritics.
- The **character n-gram model** achieves strong recall for high-resource scripts (e.g., German 0.965 recall; English 0.968 recall) but confuses closely related varieties (French vs. French stanza) and misroutes Yoruba variants.
- **XLM-R** improves overall accuracy relative to the logistic baseline and excels on stanza-domain variants (recall ≥0.94 for French stanza, Swedish stanza, Kazakh stanza, Urdu1), but underperforms on narrow classes with limited distinctive context (French1 recall 0.003; Wolof1 recall 0.000; Swedish1 recall 0.028).

## Qualitative error analysis
- **Rule-based heuristics**: Without reliable diacritic or script signals, the classifier collapses many languages into Yoruba or Swahili, leading to systematic over-prediction of those labels. Cross-script confusion (e.g., Cyrillic Kazakh to Yoruba1) highlights the limitation of simplistic Unicode checks when transliteration or shared characters occur.
- **Character n-grams**: Errors mostly stem from closely related variants that share orthography and vocabulary (French stanza ↔ French1; Wolof stanza ↔ Wolof1; Kazakh stanza ↔ Kazakh1). Occasional leakage into high-frequency classes (Swahili) suggests the model captures surface patterns but lacks semantic grounding.
- **XLM-R fine-tuning**: The transformer better separates stanza vs. non-stanza in many cases but still collapses minority classes (Wolof1, French1, Swedish1) into more frequent neighbours. Misclassifications often arise when sentences contain named entities or numerical content that offer little language-specific signal.

## Operational comparison
- **Rule-based heuristics**: Lowest computational cost and fully interpretable, but scaling to new languages demands manual linguistic expertise and brittle cue lists.
- **Character n-gram logistic regression**: Light-weight training and inference; scales with labelled data and minimal feature engineering. However, it cannot exploit subword semantics and struggles when languages share near-identical orthography.
- **XLM-R fine-tuning**: Highest accuracy and robust cross-script generalisation, yet requires GPU resources, longer training cycles, and incurs higher serving costs. Model size also complicates deployment on resource-constrained devices.

## Takeaways for Milestone 2 Improvement
- Prioritise **transformer-based modelling** while addressing minority-class performance via data balancing, domain-specific augmentation, or label grouping.
- Explore **hybrid approaches** where rule-based cues guardrail transformer predictions in low-confidence regions, preserving interpretability for critical errors.
- Implement **lightweight distillation or adapter-based fine-tuning** to retain transformer accuracy with lower runtime cost, enabling broader deployment scenarios.
