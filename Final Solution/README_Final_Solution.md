# Final Solution: fastText Primary + Character n-gram Fallback

This README summarizes the run details for the final ensemble notebook in `Final Solution/final_solution_fasttext_char_fallback.ipynb`, which uses **fastText + Logistic Regression** as the primary model and falls back to a **character n-gram TF-IDF + Logistic Regression** model when the sentence OOV ratio is high. The detailed run log lives in `Final Solution/final_solution_fasttext_char_fallback_12Jan.md`.

## Notebook Purpose
The final solution targets robust language identification across in-domain (Wikipedia) and OOD (hate speech/social media) data by routing high-OOV sentences to a character n-gram classifier that is more tolerant of noise and misspellings.

## Environment
- Python dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `fasttext`
- Random seed: `13`
- Key datasets:
  - In-domain: Wikipedia CoNLL-U sentences under `data/<language>/*.conllu`
  - OOD: language-specific CSVs defined in the notebook configuration

## Data Configuration
- Languages: `kazakh`, `latvian`, `swedish`, `yoruba`, `urdu`
- fastText model files: `models/fasttext/cc.<lang>.300.bin` with language code mapping
- OOD datasets:
  - `data/kazakh_hate_speech_fasttext.csv`
  - `data/afrihate_yoruba_fasttext.csv`
  - `data/latvian_comments_fasttext_nat_only_20mb.csv`
  - `data/biaswe_fasttext.csv`
  - `data/gsm8k_urdu_fasttext.csv`

## Training Setup
- Max sentences per language: `2000`
- Train/validation/test split: `0.7 / 0.1 / 0.2`
- Character n-gram TF-IDF settings:
  - `ngram_range=(3, 5)`
  - `min_df=2`
  - `max_features=200000`
- OOV routing threshold: `0.3`

## Results (from `final_solution_fasttext_char_fallback_12Jan.md`)
**In-domain (Wikipedia test set)**
- Accuracy: `0.9755`
- Macro Precision: `0.9767`
- Macro Recall: `0.9755`
- Macro F1: `0.9754`

**OOD (combined datasets)**
- Accuracy: `0.9764`
- Macro Precision: `0.8449`
- Macro Recall: `0.9839`
- Macro F1: `0.9007`

## Routing Behavior
- ID routing to fallback: `80.55%`
- OOD routing to fallback: `92.04%`

## How to Reproduce
1. Download the fastText models for the target languages into `models/fasttext/`.
2. Ensure the Wikipedia and OOD datasets are available at the paths listed above.
3. Run `Final Solution/final_solution_fasttext_char_fallback.ipynb` end-to-end.

For step-by-step outputs and exact prints, refer to `Final Solution/final_solution_fasttext_char_fallback_12Jan.md`.
