"""Rule-based heuristics for language identification."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Sequence

LANGUAGE_SPECIAL_CHARACTERS: Dict[str, str] = {
    "german": "äöüßÄÖÜ",
    "french": "àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ",
    "swedish": "åäöÅÄÖ",
    "latvian": "āčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ",
    "kazakh": "әөүұқғңһіӘӨҮҰҚҒҢҺІ",
    "wolof": "ëñËÑ",
    "yoruba": "ẹọṣńáéíóúÀÁÈÉÌÍÒÓÙÚṢẸỌŃ",
}

LANGUAGE_KEYWORDS: Dict[str, Sequence[str]] = {
    "german": (" und ", " der ", " die ", " nicht "),
    "english": (" the ", " and ", " is ", " was "),
    "french": (" le ", " la ", " les ", " des "),
    "swedish": (" och ", " det ", " som ", " inte "),
    "latvian": (" un ", " kas ", " par ", " ar "),
    "swahili": (" ya ", " kwa ", " na ", " cha "),
    "wolof": (" ci ", " ak ", " la ", " nga "),
    "yoruba": (" ni ", " ati ", " ṣe ", " jẹ "),
}

LANGUAGE_SCRIPTS: Dict[str, str] = {
    "kazakh": "Cyrillic",
    "urdu": "Arabic",
}


class RuleBasedIdentifier:
    """Heuristic classifier for language identification."""

    def __init__(self) -> None:
        self.priors: Dict[str, float] = {}

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

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> None:
        label_counts = Counter(labels)
        total = sum(label_counts.values())
        self.priors = {label: count / total for label, count in label_counts.items()}

    def _score_language(self, text: str, language: str) -> float:
        score = math.log(self.priors.get(language, 1e-6))

        dominant_script = self._dominant_script(text)
        script_expectation = LANGUAGE_SCRIPTS.get(language)
        if script_expectation and dominant_script == script_expectation:
            score += 2.0
        elif script_expectation and dominant_script and dominant_script != script_expectation:
            score -= 4.0

        special_chars = LANGUAGE_SPECIAL_CHARACTERS.get(language, "")
        if special_chars:
            char_hits = sum(text.count(char) for char in special_chars)
            score += char_hits * 1.2

        keywords = LANGUAGE_KEYWORDS.get(language, ())
        if keywords:
            keyword_hits = sum(text.lower().count(keyword.strip()) for keyword in keywords)
            score += keyword_hits * 0.8

        if not special_chars and all(ord(c) < 128 for c in text):
            score += 0.3
        return score

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
