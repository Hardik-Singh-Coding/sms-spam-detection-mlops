from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range = (1, 2),
                    min_df = 2,
                    max_df = 0.95,
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )