"""TF-IDF extractive summarizer built on scikit-learn."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessor import split_sentences


class TfidfSummarizer:
    """Rank sentences by TF-IDF weight and return the top ones in order.

    Parameters
    ----------
    n_sentences : int, optional
        Exact number of sentences in the summary. If None, ``ratio`` is used.
    ratio : float, optional
        Fraction of the document's sentences to keep (0 < ratio <= 1).
        Ignored when ``n_sentences`` is given.
    """

    def __init__(self, n_sentences=None, ratio=0.3):
        if n_sentences is not None and n_sentences < 1:
            raise ValueError("n_sentences must be >= 1")
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.n_sentences = n_sentences
        self.ratio = ratio
        # English stop words are dropped so that filler words ("the", "is",
        # "and") do not inflate sentence scores. sublinear_tf dampens the
        # effect of a single word repeated many times.
        self._vectorizer = TfidfVectorizer(
            stop_words="english", sublinear_tf=True, norm="l2"
        )

    def _target_count(self, n_total):
        if self.n_sentences is not None:
            return min(self.n_sentences, n_total)
        return max(1, int(round(n_total * self.ratio)))

    def summarize(self, text):
        """Return the extractive summary of ``text`` as a single string."""
        sentences = split_sentences(text)
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0]

        # One TF-IDF row per sentence; columns are vocabulary terms.
        tfidf = self._vectorizer.fit_transform(sentences)
        # Sentence score = sum of its terms' TF-IDF weights, divided by the
        # number of terms so that very long sentences are not favored
        # simply for having more words.
        weights = np.asarray(tfidf.sum(axis=1)).ravel()
        lengths = np.asarray((tfidf > 0).sum(axis=1)).ravel()
        scores = weights / np.maximum(lengths, 1)

        k = self._target_count(len(sentences))
        # Indices of the k highest-scoring sentences, then back into
        # original document order so the summary reads naturally.
        top_idx = np.argsort(scores)[-k:]
        top_idx = sorted(top_idx)
        return " ".join(sentences[i] for i in top_idx)

    def sentence_scores(self, text):
        """Return (sentence, score) pairs sorted by score, highest first.

        Useful for inspecting *why* the summarizer picked what it picked.
        """
        sentences = split_sentences(text)
        if not sentences:
            return []
        tfidf = self._vectorizer.fit_transform(sentences)
        weights = np.asarray(tfidf.sum(axis=1)).ravel()
        lengths = np.asarray((tfidf > 0).sum(axis=1)).ravel()
        scores = weights / np.maximum(lengths, 1)
        ranked = sorted(zip(sentences, scores), key=lambda p: p[1], reverse=True)
        return [(s, round(float(sc), 4)) for s, sc in ranked]
