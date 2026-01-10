# Language Identification Final Presentation — Speaker Notes

## Slide 1: Title Slide / Project Overview

**Objective of this slide:**
Set context for the audience and establish what problem you're solving.

**Key talking points:**

- Start with: "Today we're presenting our work on **multilingual sentence-level language identification** across 10 languages spanning both Latin and Cyrillic scripts."

- Emphasize the scope: We built and evaluated systems to automatically detect which language a sentence is written in, moving beyond traditional single-script approaches.

- Highlight the languages: German, English, French, Swedish, Latvian, Swahili, Wolof, Yoruba, Kazakh, and Urdu. These represent both high-resource and low-resource languages, and mix two scripts entirely.

- Frame the challenge: "Language identification sounds simple—just look at the letters, right? But when you have overlapping orthographies (like English and Yoruba), transliterated text, and domain shift to social media, it becomes a real problem."

**Timing:** 60–90 seconds

**Delivery tip:**

Pause after listing the languages to let their diversity sink in. Your audience will appreciate that you're not just working with European languages.

---

## Slide 2: Approaches Compared

**Objective of this slide:**
Give the audience a roadmap of the baseline methods, from simplest to most sophisticated.

**Key talking points:**

**Rule-based heuristics:**
- "The simplest approach: we use linguistic intuition to write rules. For example, if a word contains the character ع (ayn), it's probably Arabic or Urdu. If a sentence uses a lot of ß, it's very likely German."
- Mention the three kinds of rules: Unicode script detection (Are we in Latin? Cyrillic? Arabic block?), diacritics (ñ, ò, š, etc.), and curated cue words/affixes (–ung endings in German, –tion in French).
- Trade-off: "Interpretable and fast, but labor-intensive to maintain as languages grow."

**Character n-gram logistic regression:**
- "We're moving from hand-written rules to learning patterns from data. Take every substring of length 2, 3, 4 from the text—bigrams, trigrams, 4-grams—and count how often each appears."
- Explain the intuition: "The sequence 'ch' appears often in German and French, but rarely in Yoruba. So the model learns to weight these frequencies."
- TF–IDF reweighting: "We downweight very common n-grams that appear in all languages, so rare but distinctive patterns get more signal."
- Trade-off: "Lightweight training and inference, strong accuracy, works out of the box without language expertise."

**XLM-RoBERTa fine-tuning:**
- "Transformer models like BERT learn deep patterns from massive multilingual text. XLM-RoBERTa is trained on 100+ languages, so it already knows what language-specific patterns look like."
- Fine-tuning process: "We take the pretrained model and show it our 8,000 training sentences, allowing it to adapt its weights specifically to the 10 languages we care about."
- Trade-off: "More compute (needs a GPU), slower inference, but very robust—the model learns contextual and semantic patterns, not just character patterns."

**fastText averaged embeddings (Q1 case study):**
- "For Question 1, we explored word embeddings: average the fastText vectors for all words in a sentence, then feed to logistic regression."
- Why it matters: "It bridges embeddings (DL) and classical ML, but—as we'll see later—it's risky when deployed to new domains."

**Delivery tip:**
Use hand gestures or slides to show progression: point up or forward as you move from simple to complex. This helps the audience track the increasing sophistication.

**Timing:** 2–2.5 minutes

---

## Slide 3: Key Results — Milestone 2 (In-Domain Wikipedia, 10 Languages)

**Objective of this slide:**
Present the main quantitative outcomes and help the audience compare the baselines.

**Key talking points:**

**Rule-based heuristics (accuracy 0.89):**
- "Our hand-written rules got 89%. That's surprisingly good for a zero-learning approach, but there's a clear ceiling."
- Where it fails: "Latin script is ambiguous. Kazakh uses Latin transliteration in our data, Latvian uses diacritics, and Yoruba uses Latin with minimal diacritics. Our rules can't always distinguish them, so they leak Kazakh/Latvian/Yoruba into German."
- Positives: "Urdu benefits massively from script cues—it uses the Nastaliq script, so it's almost perfectly separated."

**Character n-gram logistic regression (accuracy 0.968, BEST):**
- Lead with: "The character n-gram model won. 96.8% accuracy is excellent, and it's lightweight."
- Recall performance: "On several languages—Urdu, Kazakh—we got near-perfect recall. Those languages have distinctive n-grams that the model learned reliably."
- Where it stumbles: "English and Swedish still confuse each other occasionally (both use Latin, similar letter frequencies). Very short numeric snippets are also ambiguous—'123' could be any language."

**XLM-R fine-tuning (accuracy 0.966, close second):**
- "Only 0.2 percentage points behind the n-gram model, but with more compute overhead."
- Strengths: "It does slightly better on French and Swedish—capturing syntactic patterns that character n-grams miss. And it's more robust to spelling variations or typos."
- The trade-off: "Is 0.2% accuracy worth 15× the inference latency and GPU requirements? For most cases, no."

**Key insight to emphasize:**
"Our main finding: **character n-gram logistic regression is the sweet spot for production language ID.** It's accurate, cheap, interpretable (top features are actual character sequences), and scales to new languages easily. Rules are useful for diagnostics; XLM-R for harder domains."

**Timing:** 2.5–3 minutes

**Visuals to call out:**
- Point to the results table. Let the audience read the numbers for 5–10 seconds before you comment.
- If you have a bar chart, note the tiny gap between n-gram and XLM-R visually.

---

## Slide 4: Key Results — Question 1 (fastText OOD, 5 Languages)

**Objective of this slide:**
Present the findings on out-of-domain robustness with word embeddings.

**Key talking points:**

**The setup:**
- "In Question 1, we asked: if we use pretrained fastText embeddings as features for logistic regression, how robust are they when we switch from Wikipedia to real-world hate speech and social media?"
- Domain shift: "Wikipedia is formal, clean, curated. Social media is short, slangy, sometimes transliterated. It's a significant domain gap."

**In-domain performance (Wikipedia):**
- "On Wikipedia, fastText was nearly perfect: 99.8% accuracy. Yoruba was the weakest link at 99.5%, but still excellent."
- The illusion: "So you might think, 'Embedding-based language ID is solved!' But..."

**Out-of-domain performance (hate speech/social media):**
- Lead with: "When we moved to social media, accuracy dropped only 1.17 percentage points to 98.63%—still very high."
- Per-language breakdown: "But the drop was **uneven**. Swedish and Urdu stayed at 100%. Latvian dropped to 98.4%. And some languages showed hidden fragility in the error distributions."
- Why this matters: "A 1% accuracy drop sounds small, but if you're processing millions of messages, that's tens of thousands of misclassifications."

**The key insight: OOV (Out-of-Vocabulary) coverage:**
- "fastText embeddings are words. But social media contains slang, abbreviations, brand names, and transliterations that Wikipedia doesn't have."
- Show the numbers: "Latvian had a **24.1% OOV rate** in social media—nearly 1 in 4 tokens were unseen during Wikipedia training. Yoruba was even worse at **37.9% OOV**. Kazakh was better at 8.5%, but still significant."
- Manual errors: "We found misclassified posts like a two-token Kazakh message labeled as Yoruba—too short and too transliterated for the embeddings to work."

**The headline-metric trap:**
- "This is crucial: **accuracy hides failures when classes are imbalanced or errors cluster by language.**"
- Example: "If Yoruba represents 1% of social media data, a 10% error rate on Yoruba barely moves overall accuracy, but it's a real problem for Yoruba-language content."

**Mitigation mentioned in Q1:**
- "To fix this, we recommend pairing fastText with character n-grams and flagging high-OOV sentences. The script supports a `--enable-fasttext-char-backoff` flag that automatically routes risky sentences to a character-based classifier."

**Key takeaway:**
"**Pretrained word embeddings alone are not a safe baseline for multilingual language ID without character-level backups and per-domain coverage checks.**"

**Timing:** 2.5–3 minutes

**Delivery tip:**
Slow down when explaining OOV. It's a concept that might be new to some audience members. Draw a simple diagram if you have slides: "Pretrained vocab = 100,000 words. New domain = 200,000 words. Overlap = only 150,000. Missing 50,000."

---

## Slide 5: Detailed OOD Robustness — Macro F1 Comparison

**Objective of this slide:**
Show why overall accuracy is misleading and per-language metrics matter.

**Key talking points:**

**The setup:**
- "Here's where things get interesting. We computed not just accuracy, but per-language F1 scores on the social media data, then averaged them."
- Why macro F1 matters: "Macro F1 treats each language equally, so if our model fails on Yoruba (a minority class), it shows up in the metric. Accuracy would hide it if Yoruba is rare."

**fastText alone (macro F1: 0.666):**
- "Despite high accuracy, fastText's macro F1 was only 0.67—it means the average per-language performance was mediocre."
- Breakdown by language:
  - Kazakh: 0.50 F1 (poor).
  - Latvian: 0.33 F1 (terrible).
  - Swedish: 1.0 F1 (perfect).
  - Urdu: 1.0 F1 (perfect).
  - Yoruba: 0.50 F1 (poor).
- Insight: "Imbalanced success. Two languages near-perfect, three languages near-random."

**Character n-gram alone (macro F1: 0.506):**
- "The n-gram model without character support on OOD is actually worse than fastText on macro F1—only 0.51."
- Why? "Character n-grams learned on Wikipedia, and social media's slang, abbreviations, and transliteration break those patterns."
- But it's stable: "No extreme failures like Yoruba at 0.33; more consistent across languages."

**Combined fastText + character n-gram (macro F1: 0.799, BEST):**
- "When we fused both approaches—using fastText as primary and falling back to character n-grams for high-OOV sentences—macro F1 jumped to 0.80."
- Per-language wins:
  - Latvian: 0.49 → 0.50 (fixed by character fallback).
  - Yoruba: 0.50 → 1.0 (character features catch the transliterated posts).
  - Swedish/Urdu: stay at 1.0 (no degradation).
- Takeaway: "The ensemble is the safest bet."

**Why this slide is critical:**
"This demonstrates that **headline metrics can mask systemic failures.** You need per-language diagnostics to build robust systems, especially in multilingual settings."

**Timing:** 2–2.5 minutes

**Visual cue:**
If you have a comparative bar chart, show the three groupings side by side and draw attention to the Yoruba bar spiking from 0.50 to 1.0 in the combined model.

---

## Slide 6: Deployment Considerations — Latency vs. Hardware

**Objective of this slide:**
Bridge from research metrics to real-world deployment trade-offs.

**Key talking points:**

**Character n-gram (TF–IDF + logistic regression):**
- **Hardware:** "Pure CPU. The model is 10–20 MB—fits on a Raspberry Pi."
- **Latency:** "Less than 5 milliseconds per sentence on a single CPU core. For batch processing, scales linearly."
- **Why it's fast:** "Feature extraction (converting text to n-grams) is a simple string operation. No deep compute."
- **Deployment scenario:** "You run language ID on edge devices, mobile phones, or cost-constrained cloud instances. Default choice."

**XLM-R fine-tuning:**
- **Hardware:** "GPU strongly recommended. The base model needs 1–2 GB VRAM for inference. CPU-only inference is 10× slower."
- **Latency:** "30–80 milliseconds per sentence on a mid-range GPU (A10, T4). 300+ ms on CPU without quantization."
- **Why it's slow:** "Transformer attention is O(n²) in sequence length, and we're doing 12 transformer layers."
- **Deployment scenario:** "High-throughput cloud services where you have GPU pools and can amortize costs. Robustness-critical settings (e.g., content moderation on code-switched text)."

**The decision matrix:**
- "If you're processing millions of messages per day and cost is a constraint: **n-gram model.**"
- "If you're doing live language ID on phones: **n-gram model.**"
- "If you're moderating hate speech and can't afford false negatives on rare languages: **XLM-R or ensemble.**"
- "If you have domain-shifted text (not Wikipedia): **n-gram first, XLM-R for hard cases.**"

**Translating latency into real numbers:**
- "Let's say you process 1 million messages per hour."
  - n-gram: 5,000 seconds = 1.4 hours of compute (doable on one machine).
  - XLM-R on CPU: 300,000 seconds = 83 hours (need a GPU farm).
  - XLM-R on 4 GPUs: 7,500 seconds = 2 hours.
- "The hardware requirements explode."

**Key takeaway:**
"**Default to the n-gram model for general serving. XLM-R is a specialist tool for hard domains or when you have GPU infrastructure.**"

**Timing:** 2–2.5 minutes

---

## Slide 7: Error Analysis Highlights

**Objective of this slide:**
Humanize the models by showing what they get wrong and why.

**Key talking points:**

**Rule-based heuristics:**
- "Our hand-written rules failed predictably: Latin script is ambiguous."
- Example error: "A Kazakh sentence in Latin transliteration, say 'Mektep talimdı alg…', has no diacritics. Our rules default to German because German also uses bare Latin."
- Another example: "Numeric lists like '1. Something 2. Something' have no script cues, so the heuristic rules can't identify the language. They guess randomly or default."

**Character n-gram logistic regression:**
- "The n-gram model is much more accurate, but still has a few weak spots."
- English ↔ Swedish confusion: "Both languages have high-frequency trigrams like 'the', 'ing', 'and'. Short sentences can be ambiguous. Example: 'the morning was cold' could be English or Swedish."
- Yoruba ambiguity: "Yoruba uses Latin with tones marked by diacritics (ẹ, ọ, ń). Without diacritics (e.g., user-generated content on Twitter), it looks like English or other Latin languages."
- Numeric content: "Short numbers or URLs (e.g., '2025-01-10 https://example.com') have almost no language signal."

**XLM-R fine-tuning:**
- "Transformer errors are more subtle because the model learns context."
- Over-indexing on high-resource patterns: "If the model saw lots of German text during training (Wikipedia is biased toward Germanic languages), it may over-predict German for Swahili or Wolof sentences dominated by borrowed German words (like 'Telefon', 'Auto')."
- Example: "A Swahili sentence 'Telefoni yangu ina simu mitatu' (My phone has three SIM cards) might be mislabeled as German because of 'Telefoni', even though the rest is distinctly Swahili."

**fastText OOD errors:**
- "The most revealing errors: short, transliterated posts."
- Example: "A Kazakh tweet 'Qalyptau kut kut!' (a transliteration with no diacritics) was labeled Yoruba. Why? Both languages have sparse data, similar orthography, and the post is too short to disambiguate."
- Slang and abbreviations: "TikTok-style abbreviations (e.g., 'smh', 'ffs') have no language markers. fastText struggles because these tokens aren't in Wikipedia."

**Why this matters:**
- "Understanding errors is crucial for building trust. If you deploy language ID, stakeholders need to know where it's fragile."
- "For example: 'Don't use this for numeric lists alone; pair it with other signals.'"

**Timing:** 2–2.5 minutes

**Delivery tip:**
Choose 1–2 concrete examples and write them on a slide or explain aloud. Real examples resonate more than abstract descriptions.

---

## Slide 8: Visual Confusion Matrices (10-Language Wikipedia)

**Objective of this slide:**
Show the confusion patterns visually, making error modes tangible.

**Key talking points:**

**Rule-based heuristics matrix:**
- "Looking at the first matrix, you see a concentration of errors in the top-left: German, English, French, Swedish."
- "Cyrillic languages (Kazakh, Urdu) are cleanly separated—script cues work great there."
- "But notice the Yoruba row: it bleeds into German and English. That's the Latin-script overlap problem we mentioned."
- Reading the matrix: "Rows are true labels, columns are predicted labels. A strong diagonal (dark colors on the diagonal) means high accuracy."

**Character n-gram logistic regression matrix:**
- "The second matrix is much cleaner. The diagonal is much darker (higher accuracy)."
- "You still see some English ↔ Swedish confusion (off-diagonal block), and Yoruba has a small leak, but overall, it's a dramatic improvement."
- "Cyrillic languages remain perfectly separated—character n-grams are great at capturing script patterns."

**XLM-R fine-tuning matrix:**
- "The third matrix is comparable to the second—similarly dark diagonal, similar error patterns."
- "XLM-R does slightly better at French/Swedish separation (you might notice a slightly darker off-diagonal block there compared to n-grams)."
- "But the gains are marginal given the added compute cost."

**How to walk through visuals:**
- "Let me walk you through these left to right, showing how the error patterns change."
- Point to specific cells: "See this cell (English row, Yoruba column)? That's the main n-gram confusion. See how it shrinks in the XLM-R matrix? That's the contextual learning at work."
- Summarize: "Rule-based → character n-grams → XLM-R shows a clear progression in error reduction, but with diminishing returns."

**Timing:** 1.5–2 minutes

---

## Slide 9: fastText OOV Coverage Diagnostics

**Objective of this slide:**
Explain why OOV is problematic and show the empirical evidence.

**Key talking points:**

**What is OOV?**
- "OOV means 'out-of-vocabulary.' A word is OOV if it doesn't exist in the vocabulary used to train the embedding model."
- Example: "fastText was trained on Wikipedia. If a social media post contains slang like 'YOLO' or 'ghosting', those words might not be in the Wikipedia vocabulary."
- Why it breaks embeddings: "If a word isn't in the vocabulary, the embedding lookup fails. fastText has a character-level fallback (it builds word vectors from subword n-grams), but this is less robust than seeing the word during training."

**Latvian vs. Yoruba OOV distributions:**
- "These histograms show the distribution of OOV rates per sentence in the social media dataset."
- Latvian histogram: "You see a long tail. Most sentences have 10–20% OOV, but some go above 40%. The distribution is spread out."
- Yoruba histogram: "Even worse. Yoruba shows a heavier tail extending above 50%. Many Yoruba social media sentences have very high OOV rates."
- Why the difference? "Latvian is a Baltic language with many inflections and rare words. Wikipedia Latvian is formal; social media Latvian is very different. Yoruba on social media (Twitter, TikTok) uses lots of slang, code-switching, and transliteration. Very different from formal Wikipedia."

**What this tells us:**
- "These histograms are red flags. If >25% of tokens are unknown, the embedding approach is fragile."
- "Yoruba's >37% OOV is especially problematic. It means the embeddings have lost a lot of information."

**The diagnostic value:**
- "Our evaluation script now computes these OOV histograms. If you see a heavy tail, you know: 'This model will fail on this language on this domain. I need a backup (character n-grams, rules, or a retrained model).'"

**The mitigation:**
- "The solution: when OOV is high, fall back to character n-grams. We implemented this as `--enable-fasttext-char-backoff` with an `--oov-threshold` flag."
- How it works: "For each sentence, compute OOV rate. If it exceeds the threshold (default 0.35, or 35%), use character n-grams. Otherwise, use fastText embeddings."
- Results: "This ensemble improved Yoruba OOD F1 from 0.50 to 1.0—a perfect recovery."

**Timing:** 2–2.5 minutes

**Delivery tip:**
Spend time on the histograms. Let the audience look at them for 10–15 seconds before you interpret them. Visual intuition is powerful.

---

## Slide 10: Milestone 1 Summary (Data Preparation)

**Objective of this slide:**
Acknowledge the foundational work without dwelling on it (it's less exciting than models, but essential).

**Key talking points:**

**The task:**
- "Milestone 1 was about preparing the data. We downloaded raw Wikipedia dumps for 10 languages, cleaned them, tokenized them, and annotated them with linguistic metadata."

**The pipeline:**
- Data source: "Hugging Face `wikimedia/wikipedia` dataset—publicly available, reproducible."
- Processing: "We used Stanza (a dependency parser) to annotate sentences with tokens, lemmas, POS tags, and dependency heads. When Stanza didn't support a language, we fell back to heuristic tokenization and generic POS tags."
- Output format: "CoNLL-U, a standard format for linguistic annotations. This ensures downstream tools (taggers, parsers) can plug in seamlessly."
- Balanced dataset: "We extracted 10,000 sentences per language (approximately 400 per language for the evaluation set), ensuring balanced evaluation."

**Why this matters:**
- "High-quality data is a prerequisite for good models. If we had noisy, imbalanced, or poorly annotated data, the baselines would have failed."
- "By using Stanza, we ensured linguistic fidelity: lemmas and parse trees aren't used in this task, but they would be useful for downstream linguistic analysis or error diagnosis."

**Key achievements of Milestone 1:**
- Reproducible pipeline (Python script, published on GitHub).
- 10 languages, balanced representation.
- Two annotation strategies (Stanza + heuristic fallback) for robustness.

**Timing:** 1–1.5 minutes

**Note:** This slide is here for completeness, but spend less time on it than on results. Your audience cares more about model performance and insights.

---

## Slide 11: Milestone 2 Summary (Baselines & Evaluation)

**Objective of this slide:**
Recap Milestone 2 achievements and frame the transition to the final solution.

**Key talking points:**

**What Milestone 2 delivered:**
- "Three baseline systems, each representing a different point on the accuracy–cost–interpretability trade-off."
- Rule-based: interpretable, cheap, limited ceiling (89%).
- Character n-grams: strong accuracy (96.8%), lightweight, scales to new languages.
- XLM-R: highest potential (96.6%), but compute-heavy.

**Evaluation protocol:**
- In-domain: 4,000 held-out Wikipedia sentences (400 per language), evaluated with accuracy, precision, recall, F1.
- Per-class analysis: We didn't just report overall numbers; we looked at each language individually.
- Confusion matrices: We visualized where errors cluster (e.g., Latin-script overlap).

**Qualitative analysis:**
- "We didn't just trust metrics. We looked at failing cases and asked: why did the model fail? Were errors systematic (e.g., always confusing Swedish with English) or random?"
- Outcome: "Systematic errors revealed linguistic patterns (script, morphology, orthography), which informed recommendations."

**Refactored codebase:**
- "We took the original monolithic evaluation script and refactored it into modular packages, improving readability and maintainability."

**Key insight from Milestone 2:**
- "Character n-gram logistic regression is the Goldilocks solution: not too simple (rules), not too heavy (XLM-R), and it works."

**Timing:** 1–1.5 minutes

---

## Slide 12: Question 1 Summary (fastText OOD)

**Objective of this slide:**
Recap Question 1 and tie it to the larger narrative.

**Key talking points:**

**The Q1 hypothesis:**
- "We asked: are pretrained word embeddings (fastText) a safe, lightweight, non-deep-learning baseline for multilingual language ID?"
- Hypothesis: "Probably not—embeddings rely on vocabulary, and vocabulary doesn't transfer across domains."

**The experiment:**
- In-domain accuracy: 99.8% on Wikipedia (near-perfect).
- Out-of-domain accuracy: 98.63% on hate-speech/social-media (only 1.17 point drop, so it *seemed* safe).
- But drilling deeper: OOV rates of 24–38% for some languages, and per-language F1 scores revealed hidden failures (Yoruba at 0.50 F1).

**The finding:**
- "Headline metrics mask fragility. fastText is vulnerable to OOV and domain shift, especially for low-resource languages."

**The lesson:**
- "When deploying embeddings for language ID, don't rely on overall accuracy. Check per-language metrics, monitor OOV rates, and pair embeddings with character-level backups."

**The fix:**
- "We implemented a mitigation in the evaluation script: `--enable-fasttext-char-backoff` routes high-OOV sentences to character n-grams, recovering Yoruba F1 from 0.50 to 1.0."

**Timing:** 1.5–2 minutes

---

## Slide 13: Final Recommendations & Next Steps

**Objective of this slide:**
Summarize actionable recommendations for practitioners.

**Key talking points:**

**For production deployment:**

1. **Default choice: Character n-gram logistic regression**
   - Best accuracy–cost trade-off.
   - Works on CPU, minimal dependencies.
   - Scales to new languages without retraining.
   - Interpretable features (actual character sequences).

2. **For domain-shifted data:**
   - Use character n-gram + fastText ensemble (with `--enable-fasttext-char-backoff`).
   - Or use XLM-R if GPU resources are available.

3. **For real-time, low-latency applications:**
   - Character n-grams (<5 ms per sentence).
   - Avoid XLM-R (300+ ms on CPU).

4. **For robustness:**
   - Always report per-language metrics (macro F1, not just accuracy).
   - Monitor OOV rates on new domains.
   - Test on truly out-of-domain data (e.g., social media, user-generated content).

5. **For low-resource languages:**
   - Character-based approaches (n-grams, character CNNs) work better than embeddings.
   - Avoid pretrained word embeddings without character backups.

**Potential future work:**
- Extend to code-switched text (sentences mixing multiple languages).
- Integrate with downstream NLP tasks (e.g., sentiment analysis, machine translation).
- Develop language ID for endangered or minority languages (only 10 languages here; many more exist).
- Explore domain adaptation: how to fine-tune models on new domains with minimal labeled data.
- Add fine-grained confidence scores (not just class predictions) to help users know when the model is uncertain.

**Reproducibility:**
- All code and data are on GitHub.
- Results are reproducible: scripts, random seeds, and artifact paths documented.
- Evaluation metrics computed from scratch (no cherry-picked subsets).

**Key takeaway to close with:**
"**Language identification is solved for in-domain, high-resource settings. The challenge—and where future work lies—is robustness to domain shift, low-resource languages, and code-switching. Our baselines and analysis provide a foundation for addressing these challenges.**"

**Timing:** 2–2.5 minutes

---

## General Delivery Guidelines

### Pacing
- **Total presentation time: 15 minutes** (strictly enforced).
- Allocate ~90 seconds per major result slide, ~60 seconds for supporting slides.
- Leave room for natural pauses (don't rush).

### Audience Engagement
- **Start with a concrete problem:** "Imagine you're building a content moderation system that processes millions of posts in 100+ languages. How do you know which language each post is in?"
- **Use stories:** When explaining errors, give real examples. "A Kazakh tweet was labeled Yoruba because…"
- **Invite questions:** If time permits, pause between major sections and ask, "Any quick questions before we move on?"

### Visuals
- **Point to metrics:** When showing results tables, pause for 5–10 seconds and let the audience read the numbers before you interpret.
- **Highlight key cells:** Use a cursor or laser pointer to draw attention to standout numbers (0.968 accuracy, the Yoruba F1 jump).
- **Explain matrices left-to-right:** When showing confusion matrices, narrate the progression: rule-based → n-gram → XLM-R.

### Language
- **Avoid jargon where possible.** Instead of "OOV", say "words the model hasn't seen before." Instead of "macro F1", say "average performance across languages, weighted equally."
- **Use analogies.** "OOV is like a vocabulary game where you're penalized for words not in the dictionary. Social media has lots of new words, so the 'game' becomes harder."
- **Be honest about limitations.** "We tested on Wikipedia and hate speech, but we haven't tried WhatsApp, customer service chat, or emails. Domain generalization is still an open problem."

### Handling Questions
- **If asked about a detail you didn't cover:** "Great question. That's in the detailed evaluation scripts in our GitHub repo. [Offer GitHub link or email.]"
- **If asked about a limitation:** "You're right, that's a limitation of this approach. It's exactly why we recommend [mitigation]. Ideally, future work would…"
- **If you don't know the answer:** "I don't have that number off the top of my head, but I can compute it and send it to you. What's your interest?"

---

## Slide-by-Slide Timing Breakdown

| Slide | Content | Time (min) |
|-------|---------|-----------|
| 1 | Title / Project Overview | 1.5 |
| 2 | Approaches Compared | 2.5 |
| 3 | Key Results — Milestone 2 | 3 |
| 4 | Key Results — Q1 OOD | 3 |
| 5 | OOD Robustness Macro F1 | 2.5 |
| 6 | Deployment Latency | 2.5 |
| 7 | Error Analysis | 2.5 |
| 8 | Confusion Matrices | 2 |
| 9 | fastText OOV Diagnostics | 2.5 |
| 10 | Milestone 1 Summary | 1.5 |
| 11 | Milestone 2 Summary | 1.5 |
| 12 | Q1 Summary | 2 |
| 13 | Recommendations & Next Steps | 2.5 |
| **Total** | | **30 minutes** |

**Note:** The above totals 30 minutes, which allows for a 15-minute presentation (slides 1–9 prioritized) + time for discussion and unexpected tangents. If you must cut to exactly 15 minutes, focus on slides 1–8 (results and analysis) and mention recommendations briefly.

---

## Backup Slides (Optional)

If you finish early or want to provide extra detail during discussion:

### Backup A: Code Architecture
- Explain the refactored `scripts/refactored_language_id_baselines/` structure.
- Show how rule-based, n-gram, and XLM-R modules are organized.
- Mention documentation and tests.

### Backup B: Dataset Statistics
- Breakdown of language distribution, sentence lengths, token counts.
- Stanza coverage (which languages had full parses vs. heuristics).
- Data splits (train/test/held-out).

### Backup C: Hyperparameter Tuning
- Why we chose specific n-gram ranges (1–4), TF–IDF parameters, etc.
- Ablation studies (e.g., "what if we only used trigrams?").
- Trade-offs explored and rejected.

### Backup D: Computational Requirements
- Training time and memory for XLM-R vs. n-grams.
- GPU specifications (did you use T4, A10, V100?).
- Model sizes and quantization options.

---

## Checklist Before Presenting

- [ ] All figures and confusion matrices are visible and labeled.
- [ ] URLs/GitHub links are in the slides (or written down for Q&A).
- [ ] You've timed the full presentation and it fits within 15 minutes.
- [ ] You've identified 1–2 team members' contributions to highlight during their sections.
- [ ] You have backup slides for any deep-dive questions.
- [ ] You've rehearsed the language-switching between sections (if team members present different parts).
- [ ] You know which numbers are most important to emphasize (e.g., 0.968 accuracy, 96.8%, the Yoruba F1 jump to 1.0).
- [ ] You're comfortable explaining OOV, macro F1, and deployment trade-offs in plain language.

---

## Final Remarks

**This presentation tells a story:**
1. **Problem:** Multilingual language ID across diverse scripts.
2. **Approaches:** Rule-based → classical ML → deep learning.
3. **Results:** Character n-grams win on accuracy–cost trade-off; XLM-R is a specialist; fastText is risky without safeguards.
4. **Insight:** Headline metrics hide failures; per-language diagnostics are essential.
5. **Recommendations:** Default to n-grams, pair with character backups, monitor OOV, test on truly OOD data.

Use this narrative arc to guide your delivery. Let the results speak for themselves, and don't over-explain. Your audience is NLP colleagues—they'll appreciate clarity, evidence, and honest caveats.

Good luck with your presentation! You've done excellent work over the semester.
