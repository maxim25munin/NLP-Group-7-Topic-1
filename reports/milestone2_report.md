# Milestone 2 Report: Multilingual Language Identification

## Experimental setup
- **Task**: sentence-level language identification across 10 language/domain labels spanning Latin and Cyrillic scripts.
- **Data**: multilingual Wikipedia snippets preprocessed with Stanza using the provided `scripts/prepare_multilingual_conllu.py` utility. Each language contributed up to 400 held-out sentences for evaluation.
- **Baselines evaluated** via `scripts/evaluate_language_id_baselines.py`:
  1. **Rule-based heuristics** using script detection, diacritics, and curated cue words.
  2. **Character n-gram logistic regression** with TF–IDF features.
  3. **XLM-RoBERTa fine-tuning** for sequence classification (multilingual transformer).

## Quantitative results
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| --- | --- | --- | --- | --- |
| Rule-based heuristics | 0.1000 | 0.010 | 0.100 | 0.018 |
| Char n-gram logistic regression | 0.9677 | 0.969 | 0.968 | 0.968 |
| XLM-R fine-tuning | 0.9653 | 0.966 | 0.965 | 0.965 |

**Class-level patterns**
- The **rule-based system** collapses nearly all inputs into the German heuristic class, revealing how brittle hand-crafted cues become when languages share scripts or lack distinctive diacritics.
- The **character n-gram model** delivers high recall across languages (e.g., Urdu 1.00, Kazakh 1.00) but still confuses closely related varieties such as English vs. Swedish stanza or Yoruba vs. English when surface forms are similar.
- **XLM-R** matches the n-gram baseline overall but differs on error profiles: it separates stanza vs. non-stanza labels more effectively while occasionally drifting into high-frequency classes when context is sparse.

## Qualitative error analysis
- **Rule-based heuristics**: With most sentences mapped to German heuristic, errors include English/French/Kazakh text mislabeled as German, underscoring the weakness of relying solely on affixes or Unicode ranges.
- **Character n-grams**: Misclassifications tend to involve near-neighbour languages (e.g., English → Yoruba heuristic, French stanza → English heuristic) or short numeric/name-heavy snippets that lack discriminative n-grams.
- **XLM-R fine-tuning**: Transformer predictions improve stanza separation but still misroute some minority-class samples (e.g., Yoruba heuristic → English or German). Named entities and template-like lists remain challenging because they provide little semantic signal specific to a language.

## Operational comparison
- **Rule-based heuristics**: Lowest computational cost and fully interpretable, but scaling to new languages demands manual linguistic expertise and brittle cue lists; performance is unusable for this task.
- **Character n-gram logistic regression**: Light-weight training and inference; scales with labelled data and minimal feature engineering. However, it cannot exploit subword semantics and may overfit to high-frequency orthographic patterns.
- **XLM-R fine-tuning**: High accuracy and robust cross-script generalisation, yet requires GPU resources, longer training cycles, and higher serving costs. Model size complicates deployment on resource-constrained devices.

## Takeaways for Milestone 2 Improvement
- **Deploy the char n-gram logistic regression** as the default baseline: it matches XLM-R performance with far lower cost and complexity.
- **Use XLM-R selectively** for domains with limited training data or heavy code-switching, where transfer learning can pay off despite higher compute demands.
- **Future work**: explore adapter-based or distilled transformer variants to narrow the efficiency gap, and investigate hybrid strategies where lightweight heuristics guardrail model confidence for low-resource classes.
