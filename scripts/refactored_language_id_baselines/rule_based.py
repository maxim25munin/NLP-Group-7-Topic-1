"""Rule-based heuristic classifier for language identification."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import import_module
from typing import Dict, List, Optional, Sequence, Set, Tuple

import spacy

LOGGER = logging.getLogger(__name__)

LANGUAGE_SPECIAL_CHARACTERS: Dict[str, str] = {
    "german": "äöüßÄÖÜ",  # German umlauts and eszett
    "french": "àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ",  # French diacritics
    "swedish": "åäöÅÄÖ",  # Swedish special characters
    "latvian": "āčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ",  # Latvian diacritics
    "kazakh": "әөүұқғңһіӘӨҮҰҚҒҢҺІ",  # Cyrillic letters prominent in Kazakh
    "wolof": "ëñËÑ",  # Wolof
    "yoruba": "ẹọṣńáéíóúÀÁÈÉÌÍÒÓÙÚṢẸỌŃ",  # Yoruba tonal marks (partial list)
}

LANGUAGE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "german": ("und", "der", "die", "nicht"),
    "english": ("the", "and", "is", "was"),
    "french": ("le", "la", "les", "des"),
    "swedish": ("och", "det", "som", "inte"),
    "latvian": ("un", "kas", "par", "ar"),
    "swahili": ("ya", "kwa", "na", "cha"),
    "wolof": ("ci", "ak", "la", "nga"),
    "yoruba": ("ni", "ati", "ṣe", "jẹ"),
}

LANGUAGE_SCRIPTS: Dict[str, str] = {
    "kazakh": "Cyrillic",
    "urdu": "Arabic",
}

LANGUAGE_SPACY_CODES: Dict[str, str] = {
    "german": "de",
    "english": "en",
    "french": "fr",
    "swedish": "sv",
    "latvian": "lv",
    "swahili": "sw",
    "wolof": "wo",
    "yoruba": "yo",
    "kazakh": "kk",
    "urdu": "ur",
}


@dataclass
class RuleBasedIdentifier:
    """Heuristic classifier for language identification using spaCy and regex."""

    priors: Dict[str, float] = None
    _nlp_cache: Dict[str, spacy.language.Language] = None
    _stopword_cache: Dict[str, Set[str]] = None
    top_tokens: Dict[str, Set[str]] = None
    keyword_patterns: Dict[str, List[re.Pattern[str]]] = None
    diacritic_patterns: Dict[str, re.Pattern[str]] = None

    def __post_init__(self) -> None:
        self.priors = {}
        self._nlp_cache = {}
        self._stopword_cache = {}
        self.top_tokens = {}
        self.keyword_patterns = {
            language: [
                re.compile(rf"\b{re.escape(keyword)}\b", flags=re.IGNORECASE)
                for keyword in keywords
            ]
            for language, keywords in LANGUAGE_KEYWORDS.items()
        }
        self.diacritic_patterns = {
            language: re.compile(f"[{re.escape(chars)}]")
            for language, chars in LANGUAGE_SPECIAL_CHARACTERS.items()
            if chars
        }

    @staticmethod
    def _dominant_script(text: str) -> Optional[str]:
        counts: Counter[str] = Counter()
        for char in text:
            if not char.strip():
                continue
            try:
                name = unicodedata.name(char)
            except ValueError:
                continue
            if "ARABIC" in name:
                counts["Arabic"] += 1
            elif "CYRILLIC" in name:
                counts["Cyrillic"] += 1
            elif "LATIN" in name:
                counts["Latin"] += 1
            elif "GREEK" in name:
                counts["Greek"] += 1
            else:
                counts["Other"] += 1
        if not counts:
            return None
        script, _ = counts.most_common(1)[0]
        return script

    def _get_nlp(self, language: str) -> spacy.language.Language:
        code = LANGUAGE_SPACY_CODES.get(language, "xx")
        if code not in self._nlp_cache:
            nlp = spacy.blank(code)
            self._nlp_cache[code] = nlp
        return self._nlp_cache[code]

    def _stopword_ratio(self, text: str, language: str) -> float:
        stopwords = self._language_stopwords(language)
        tokens = [token.lower() for token in re.findall(r"\b\w+\b", text)]
        if not tokens:
            return 0.0
        stopword_hits = sum(token in stopwords for token in tokens)
        return stopword_hits / len(tokens)

    def _keyword_hits(self, text: str, language: str) -> int:
        patterns = self.keyword_patterns.get(language, [])
        return sum(len(pattern.findall(text)) for pattern in patterns)

    def _diacritic_hits(self, text: str, language: str) -> int:
        pattern = self.diacritic_patterns.get(language)
        if not pattern:
            return 0
        return len(pattern.findall(text))

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        label_counts = Counter(labels)
        total = sum(label_counts.values())
        self.priors = {label: count / total for label, count in label_counts.items()}

        token_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        for text, label in zip(texts, labels):
            tokens = [t.lower() for t in re.findall(r"\b\w+\b", text) if len(t) > 1]
            token_counts[label].update(tokens)

        self.top_tokens = {
            label: {token for token, _ in counter.most_common(40)} for label, counter in token_counts.items()
        }

    def _score_language(self, text: str, language: str) -> float:
        score = math.log(self.priors.get(language, 1e-6))

        dominant_script = self._dominant_script(text)
        script_expectation = LANGUAGE_SCRIPTS.get(language)
        if script_expectation and dominant_script == script_expectation:
            score += 2.0
        elif script_expectation and dominant_script and dominant_script != script_expectation:
            score -= 4.0

        keyword_hits = self._keyword_hits(text, language)
        score += keyword_hits * 1.0

        diacritic_hits = self._diacritic_hits(text, language)
        score += diacritic_hits * 1.2

        stop_ratio = self._stopword_ratio(text, language)
        score += stop_ratio * 2.0

        overlap_score = self._token_overlap(text, language)
        score += overlap_score * 1.2

        if not diacritic_hits and language in LANGUAGE_SPECIAL_CHARACTERS:
            if all(ord(c) < 128 for c in text):
                score -= 0.5
        return score

    def _language_stopwords(self, language: str) -> Set[str]:
        if language in self._stopword_cache:
            return self._stopword_cache[language]

        code = LANGUAGE_SPACY_CODES.get(language, "")
        stopwords: Set[str] = set()
        if code:
            module = import_module(f"spacy.lang.{code}")
            if hasattr(module, "STOP_WORDS"):
                stopwords = set(getattr(module, "STOP_WORDS"))
        self._stopword_cache[language] = stopwords
        return stopwords

    def _token_overlap(self, text: str, language: str) -> float:
        """Return fraction of tokens appearing in the language's frequent vocabulary."""

        vocab = self.top_tokens.get(language)
        if not vocab:
            return 0.0
        tokens = [token.lower() for token in re.findall(r"\b\w+\b", text)]
        if not tokens:
            return 0.0
        hits = sum(token in vocab for token in tokens)
        return hits / len(tokens)

    def predict(self, texts: Sequence[str]) -> List[str]:
        predictions = []
        for text in texts:
            scores = {label: self._score_language(text, label) for label in self.priors}
            if not scores:
                predictions.append("unknown")
                continue
            best_label = max(scores.items(), key=lambda item: item[1])[0]
            predictions.append(best_label)
        return predictions
