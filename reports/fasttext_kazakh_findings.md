# Limitations of pretrained fastText embeddings on Kazakh hate speech data

This memo summarises the probing experiment in `scripts/fasttext_probing_with_kazakh_hate_speech.ipynb`, which evaluates pretrained fastText embeddings on non-English, out-of-distribution data. The study targets the Kazakh hate speech corpus (10,150 instances across racism, bullying, violent, nazism, and neutral labels) rather than Wikipedia text in order to stress-test the embeddings under domain shift.

## Experimental setup
- **Embeddings**: Pretrained Kazakh fastText vectors (`cc.kk.300.bin`).
- **Features**: Sentence representations built by averaging whitespace-tokenised word vectors.
- **Classifier**: Multiclass logistic regression trained on the averaged vectors.
- **Data**: Balanced Wikipedia sample (for fallback) and the primary Kazakh hate speech dataset; the latter is prioritised when available.

## Quantitative performance
- Held-out **accuracy: 0.718** on the hate speech test split.
- Macro-averaged **precision/recall/F1: 0.72/0.72/0.72**.
- Per-class recall varies from **0.60 (nazism)** to **0.80 (racism)**, indicating uneven coverage across labels.

## Error analysis highlights
- Misclassifications cluster around thematically related classes: many violent posts are mislabelled as nazism despite distinct intents, suggesting that generic embeddings blur topic-specific nuances.
- Confusion persists even with clear lexical cues, implying that surface-level similarity (e.g., shared political vocabulary) outweighs context or pragmatics in the static vectors.

## Why fastText alone is risky for this project
1. **Domain mismatch**: The embeddings are trained on Wikipedia, but hate speech posts contain slang, threats, and culturally specific references absent from the training domain. The 0.72 macro-F1 shows substantial information loss when transferred to social-media-style text.
2. **Label sensitivity**: Near-overlapping semantics (violent vs. nazism) are not well separated by static averages, which undermines reliability for safety-critical moderation tasks.
3. **Morphological coverage**: Kazakh is morphologically rich; averaging static word vectors (even with subword support) cannot capture inflection-driven meaning shifts or multiword expressions, leading to recall drops for some labels.
4. **Limited robustness to out-of-vocabulary phrasing**: The classifier relies on token overlap with Wikipedia-derived vectors; colloquialisms or code-switched segments in the hate speech corpus remain underrepresented.

## Implications and next steps
- Use fastText features only as a rough baseline; pair them with character n-grams or contextual encoders to reduce domain gap.
- Consider fine-tuning multilingual transformers on the target corpus to model context and morphology more effectively.
- Expand manual error inspection to quantify which lexical fields (e.g., political slogans vs. threats) drive the most confusion and to guide data augmentation.
