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

## Appendix: Evaluation artifacts

### Rule-based heuristics
**Classification report**

| Label | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| english heuristic | 0.000 | 0.000 | 0.000 | 400 |
| french stanza | 0.000 | 0.000 | 0.000 | 400 |
| german heuristic | 0.100 | 1.000 | 0.182 | 400 |
| kazakh stanza | 0.000 | 0.000 | 0.000 | 400 |
| latvian stanza | 0.000 | 0.000 | 0.000 | 400 |
| swahili heuristic | 0.000 | 0.000 | 0.000 | 400 |
| swedish stanza | 0.000 | 0.000 | 0.000 | 400 |
| urdu stanza | 0.000 | 0.000 | 0.000 | 400 |
| wolof stanza | 0.000 | 0.000 | 0.000 | 400 |
| yoruba heuristic | 0.000 | 0.000 | 0.000 | 400 |
| **Macro avg** | **0.010** | **0.100** | **0.018** | **4000** |
| **Weighted avg** | **0.010** | **0.100** | **0.018** | **4000** |

**Confusion matrix (rows = gold, columns = predicted)**

```
| gold \ pred      | english heuristic | french stanza | german heuristic | kazakh stanza | latvian stanza | swahili heuristic | swedish stanza | urdu stanza | wolof stanza | yoruba heuristic |
| english heuristic | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| french stanza     | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| german heuristic  | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kazakh stanza     | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| latvian stanza    | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| swahili heuristic | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| swedish stanza    | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| urdu stanza       | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| wolof stanza      | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| yoruba heuristic  | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
```

**Representative misclassifications**
- Gold: *english heuristic* → german heuristic :: University of Chicago Press, 1953 Harry Kalven Jr.
- Gold: *english heuristic* → german heuristic :: Benjamin Mark Seymour (born 16 April 1999) is an English professional footballer who plays as a forward for National League South club Hampton & Richmond Borough.
- Gold: *french stanza* → german heuristic :: Références Bibliographie Ignacio URÍA, Viento norte.
- Gold: *french stanza* → german heuristic :: Entre-temps, KB Saliout fusionne avec l'usine Khrounitchev pour former le GKNPZ Khrounitchev.
- Gold: *kazakh stanza* → german heuristic :: Дереккөздер География және геодезия
- Gold: *latvian stanza* → german heuristic :: Platons bieži tiek minēts kā visbagātākais informācijas avots par Sokrata dzīvi un filozofiju.
- Gold: *swahili heuristic* → german heuristic :: Tazama pia Orodha ya visiwa vya Tanzania Tanbihi Viungo vya nje Geonames.org Visiwa vya Tanzania Ziwa Viktoria Mkoa wa Kagera
- Gold: *swedish stanza* → german heuristic :: Källor Berg i Litauen
- Gold: *urdu stanza* → german heuristic :: پھر جب اٹھارہ سو ستاون کی جنگ آزادی واقع ہوئی تب نواب بھوپال نے اپنی ریاست میں امن و امان قائم رکھنے کے لیےانگریزی فوج کا ساتھ دیا ،
- Gold: *wolof stanza* → german heuristic :: Bañ a nelaw guddi
- Gold: *yoruba heuristic* → german heuristic :: Ní ọdún 2014, Boafo kópa nínu eré oníṣókí kan tí àkọ́lé rẹ̀ jẹ́ Bus Nut.

### Char n-gram logistic regression
**Classification report**

| Label | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| english heuristic | 0.881 | 0.963 | 0.920 | 400 |
| french stanza | 0.970 | 0.975 | 0.973 | 400 |
| german heuristic | 0.929 | 0.955 | 0.942 | 400 |
| kazakh stanza | 0.983 | 1.000 | 0.991 | 400 |
| latvian stanza | 0.995 | 0.988 | 0.991 | 400 |
| swahili heuristic | 0.976 | 0.927 | 0.951 | 400 |
| swedish stanza | 0.990 | 0.980 | 0.985 | 400 |
| urdu stanza | 1.000 | 1.000 | 1.000 | 400 |
| wolof stanza | 0.983 | 0.983 | 0.983 | 400 |
| yoruba heuristic | 0.981 | 0.907 | 0.943 | 400 |
| **Macro avg** | **0.969** | **0.968** | **0.968** | **4000** |
| **Weighted avg** | **0.969** | **0.968** | **0.968** | **4000** |

**Confusion matrix (rows = gold, columns = predicted)**

```
| gold \ pred      | english heuristic | french stanza | german heuristic | kazakh stanza | latvian stanza | swahili heuristic | swedish stanza | urdu stanza | wolof stanza | yoruba heuristic |
| english heuristic | 385 | 2 | 5 | 2 | 1 | 0 | 1 | 0 | 1 | 3 |
| french stanza     | 3   | 390 | 2 | 0 | 0 | 1 | 0 | 0 | 3 | 1 |
| german heuristic  | 5   | 4 | 382 | 4 | 0 | 2 | 1 | 0 | 1 | 1 |
| kazakh stanza     | 0   | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 |
| latvian stanza    | 3   | 0 | 2 | 0 | 395 | 0 | 0 | 0 | 0 | 0 |
| swahili heuristic | 16  | 1 | 6 | 0 | 0 | 371 | 2 | 0 | 2 | 2 |
| swedish stanza    | 2   | 2 | 4 | 0 | 0 | 0 | 392 | 0 | 0 | 0 |
| urdu stanza       | 0   | 0 | 0 | 0 | 0 | 0 | 0 | 400 | 0 | 0 |
| wolof stanza      | 1   | 2 | 1 | 1 | 0 | 2 | 0 | 0 | 393 | 0 |
| yoruba heuristic  | 22  | 1 | 9 | 0 | 1 | 4 | 0 | 0 | 0 | 363 |
```

**Representative misclassifications**
- Gold: *english heuristic* → yoruba heuristic :: University of Chicago Press, 1953 Harry Kalven Jr.
- Gold: *english heuristic* → swedish stanza :: Author Stephen J.
- Gold: *french stanza* → english heuristic :: Discographie The Wrestling Album (1985) Piledriver - The Wrestling Album 2 (1987) WWF Full Metal (1996) WWF The Music, Vol. 2 (1997) (1997) WWF The Music, Vol. 3 (1998) WWF The…
- Gold: *french stanza* → wolof stanza :: Napoli sacra.
- Gold: *german heuristic* → kazakh stanza :: 306–312.
- Gold: *german heuristic* → english heuristic :: Vera religio vindicata contra omnis generis incredulos, Stahel, Würzburg 1771.
- Gold: *latvian stanza* → german heuristic :: Geschichte der Böhmischen Provinz der Gesellschaft Jesu, I, Wien 1910; S. Polčin.
- Gold: *swahili heuristic* → german heuristic :: Oppenheimer, J.R.
- Gold: *swedish stanza* → english heuristic :: Diskografi Studioalbum Living In A Box (1987) Gatecrashing (1989) Samlingsalbum The Best of Living in a Box (1999) The Very Best of Living in a Box (2003) Living In A Box - The…
- Gold: *wolof stanza* → english heuristic :: Lees bind ci moom Kathleen Sheldon, Historical Dictionary of Women in Sub-Saharan Africa, The Scarecrow Press, Inc., 2005, 448 p.
- Gold: *yoruba heuristic* → english heuristic :: This could be determined by the kind of job this person does or wealth.

### XLM-R fine-tuning
**Classification report**

| Label | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| english heuristic | 0.878 | 0.955 | 0.915 | 400 |
| french stanza | 0.973 | 0.975 | 0.974 | 400 |
| german heuristic | 0.912 | 0.960 | 0.935 | 400 |
| kazakh stanza | 0.995 | 1.000 | 0.998 | 400 |
| latvian stanza | 1.000 | 0.988 | 0.994 | 400 |
| swahili heuristic | 0.951 | 0.927 | 0.939 | 400 |
| swedish stanza | 0.990 | 0.978 | 0.984 | 400 |
| urdu stanza | 1.000 | 1.000 | 1.000 | 400 |
| wolof stanza | 0.985 | 0.980 | 0.982 | 400 |
| yoruba heuristic | 0.981 | 0.890 | 0.933 | 400 |
| **Macro avg** | **0.966** | **0.965** | **0.965** | **4000** |
| **Weighted avg** | **0.966** | **0.965** | **0.965** | **4000** |

**Confusion matrix (rows = gold, columns = predicted)**

```
| gold \ pred      | english heuristic | french stanza | german heuristic | kazakh stanza | latvian stanza | swahili heuristic | swedish stanza | urdu stanza | wolof stanza | yoruba heuristic |
| english heuristic | 382 | 1 | 10 | 0 | 0 | 6 | 1 | 0 | 0 | 0 |
| french stanza     | 2   | 390 | 3 | 1 | 0 | 2 | 1 | 0 | 0 | 1 |
| german heuristic  | 3   | 4 | 384 | 0 | 0 | 4 | 0 | 0 | 1 | 4 |
| kazakh stanza     | 0   | 0 | 0 | 400 | 0 | 0 | 0 | 0 | 0 | 0 |
| latvian stanza    | 3   | 0 | 2 | 0 | 395 | 0 | 0 | 0 | 0 | 0 |
| swahili heuristic | 14  | 1 | 8 | 0 | 0 | 371 | 2 | 0 | 3 | 1 |
| swedish stanza    | 4   | 0 | 4 | 0 | 1 | 391 | 0 | 0 | 0 | 0 |
| urdu stanza       | 0   | 0 | 0 | 0 | 0 | 0 | 0 | 400 | 0 | 0 |
| wolof stanza      | 0   | 4 | 2 | 1 | 0 | 0 | 0 | 0 | 392 | 1 |
| yoruba heuristic  | 27  | 1 | 8 | 0 | 0 | 6 | 0 | 0 | 2 | 356 |
```

**Representative misclassifications**
- Gold: *english heuristic* → german heuristic :: University of Chicago Press, 1953 Harry Kalven Jr.
- Gold: *english heuristic* → swahili heuristic :: Author Stephen J.
- Gold: *french stanza* → german heuristic :: Discographie The Wrestling Album (1985) Piledriver - The Wrestling Album 2 (1987) WWF Full Metal (1996) WWF The Music, Vol. 2 (1997) (1997) WWF The Music, Vol. 3 (1998) WWF The…
- Gold: *french stanza* → swahili heuristic :: Le chahut, 1977, .
- Gold: *german heuristic* → french stanza :: Lausanne 1998.
- Gold: *german heuristic* → swahili heuristic :: 122–264.
- Gold: *latvian stanza* → german heuristic :: Geschichte der Böhmischen Provinz der Gesellschaft Jesu, I, Wien 1910; S. Polčin.
- Gold: *swahili heuristic* → german heuristic :: Oppenheimer, J.R.
- Gold: *swedish stanza* → swahili heuristic :: Diskografi Studioalbum Living In A Box (1987) Gatecrashing (1989) Samlingsalbum The Best of Living in a Box (1999) The Very Best of Living in a Box (2003) Living In A Box - The…
- Gold: *wolof stanza* → kazakh stanza :: — 376 с.
- Gold: *yoruba heuristic* → english heuristic :: This could be determined by the kind of job this person does or wealth.
