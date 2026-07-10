"""TF-IDF + LogisticRegression section classifier.

Labels: HEADER, CONTACT, EXPERIENCE, EDUCATION, LINE_ITEM, TOTAL, OTHER.

The classifier is trained on synthetic labelled snippets from :mod:`docintel.data`.
Evaluation in the tests uses a held-out split / cross-validation so the reported
accuracy is leakage-safe (the model never sees the evaluation snippets).
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import make_training_data

__all__ = ["SectionClassifier", "LABELS"]

LABELS = ["HEADER", "CONTACT", "EXPERIENCE", "EDUCATION", "LINE_ITEM", "TOTAL", "OTHER"]


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    analyzer="word",
                    token_pattern=r"(?u)\b\w+\b|[$%@]",
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, C=4.0),
            ),
        ]
    )


class SectionClassifier:
    """Wraps a scikit-learn pipeline with fit / predict / persistence."""

    def __init__(self) -> None:
        self.pipeline: Pipeline = _build_pipeline()
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, texts: list[str], labels: list[str]) -> SectionClassifier:
        self.pipeline.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, texts: list[str]) -> list[str]:
        self._check_fitted()
        return list(self.pipeline.predict(texts))

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]

    def predict_proba(self, texts: list[str]):
        self._check_fitted()
        return self.pipeline.predict_proba(texts)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SectionClassifier is not fitted; call fit() or train_default().")

    @classmethod
    def train_default(cls, n_per_label: int = 120, seed: int = 7) -> SectionClassifier:
        """Train on the synthetic corpus and return a fitted classifier."""
        texts, labels = make_training_data(n_per_label=n_per_label, seed=seed)
        return cls().fit(texts, labels)

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> SectionClassifier:
        obj = cls()
        obj.pipeline = joblib.load(path)
        obj._fitted = True
        return obj
