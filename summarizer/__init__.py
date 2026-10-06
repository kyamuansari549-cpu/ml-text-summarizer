"""ML Text Summarizer — extractive text summarization using TF-IDF.

Built during the Industrial Internship (Data Science, AI & ML using Python)
at Academy of Skill Development, June–July 2026.

Method: each sentence of the input document is scored by the sum of the
TF-IDF weights of its words (computed with scikit-learn). The top-ranked
sentences are returned in their original order, forming an extractive
summary that preserves the document's own wording.
"""

from .tfidf_summarizer import TfidfSummarizer
from .preprocessor import split_sentences, clean_text

__all__ = ["TfidfSummarizer", "split_sentences", "clean_text"]
__version__ = "1.0.0"
