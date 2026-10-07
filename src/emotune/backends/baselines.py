"""Cheap reference classifiers that need only scikit-learn."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..data import Example
from ..labels import INVALID, LABELS
from .base import Prediction

LEXICON = {
    "sadness": {"sad", "lonely", "heartbroken", "miserable", "hopeless", "gloomy", "devastated", "empty", "unhappy",
                "depressed", "grief", "cry", "hurt"},
    "joy": {"happy", "delighted", "cheerful", "thrilled", "content", "glad", "ecstatic", "pleased", "joyful", "great"},
    "love": {"love", "loving", "affectionate", "adored", "tender", "devoted", "cherished", "romantic", "caring", "sweet"},
    "anger": {"angry", "furious", "irritated", "outraged", "resentful", "annoyed", "bitter", "hostile", "mad", "hate"},
    "fear": {"afraid", "terrified", "anxious", "nervous", "frightened", "uneasy", "panicked", "scared", "worried"},
    "surprise": {"surprised", "amazed", "astonished", "shocked", "stunned", "startled", "speechless", "curious"},
}


class KeywordBaseline:
    """Counts lexicon words. A tie or no match gives ``invalid``, so the baseline is honest about what it misses."""

    name = "keyword"

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        out = []
        for t in texts:
            words = re.findall(r"[a-z]+", t.casefold())
            counts = np.array([sum(w in LEXICON[name] for w in words) for name in LABELS], dtype=float)
            if counts.max() == 0 or (counts == counts.max()).sum() > 1:
                out.append(Prediction(INVALID, scores=tuple(counts)))
            else:
                out.append(Prediction(LABELS[int(counts.argmax())], scores=tuple(counts)))
        return out


class TfidfBaseline:
    """TF-IDF word and bigram features with logistic regression, in one pipeline fit on train only."""

    name = "tfidf-logreg"

    def __init__(self, seed: int = 0, C: float = 4.0) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed)),
        ])
        self.fitted = False

    def fit(self, train: Sequence[Example]) -> "TfidfBaseline":
        self.pipeline.fit([e.text for e in train], [e.label for e in train])
        self.fitted = True
        return self

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        if not self.fitted:
            raise RuntimeError("fit the baseline first")
        proba = self.pipeline.predict_proba(list(texts))
        classes = list(self.pipeline.classes_)
        out = []
        for row in proba:
            full = np.zeros(len(LABELS))
            full[classes] = row
            out.append(Prediction(LABELS[int(full.argmax())], scores=tuple(float(x) for x in full)))
        return out
