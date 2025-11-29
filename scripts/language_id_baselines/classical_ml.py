"""Classical machine learning baselines for language identification."""

from __future__ import annotations

try:  # Optional dependency used by the logistic regression baseline
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "scikit-learn is required to run the baseline evaluation script."
        "Please install it via `pip install scikit-learn`."
    ) from exc


def build_logistic_regression_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=2,
    )
    classifier = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
