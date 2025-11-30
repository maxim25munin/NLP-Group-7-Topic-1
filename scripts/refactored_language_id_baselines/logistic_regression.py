"""Character n-gram logistic regression baseline."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_logistic_regression_pipeline() -> Pipeline:
    """Return a scikit-learn pipeline for the character n-gram baseline."""

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
    )
    classifier = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
