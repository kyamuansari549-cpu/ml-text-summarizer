"""Basic tests for the TF-IDF summarizer. Run with: pytest"""

from summarizer import TfidfSummarizer, split_sentences

TEXT = (
    "Machine learning is a branch of artificial intelligence. "
    "It enables computers to learn patterns from data without being explicitly programmed. "
    "Supervised learning uses labeled examples to train models. "
    "Unsupervised learning finds hidden structure in unlabeled data. "
    "Deep learning relies on neural networks with many layers. "
    "These models power applications like image recognition and machine translation."
)


def test_summary_has_requested_sentence_count():
    s = TfidfSummarizer(n_sentences=2).summarize(TEXT)
    assert len(split_sentences(s)) == 2


def test_ratio_keeps_fraction():
    s = TfidfSummarizer(ratio=0.5).summarize(TEXT)
    # 6 sentences * 0.5 = 3
    assert len(split_sentences(s)) == 3


def test_summary_preserves_original_order():
    s = TfidfSummarizer(n_sentences=3).summarize(TEXT)
    positions = [TEXT.index(sent) for sent in split_sentences(s)]
    assert positions == sorted(positions)


def test_summary_uses_document_words_only():
    s = TfidfSummarizer(n_sentences=2).summarize(TEXT)
    for sent in split_sentences(s):
        assert sent in TEXT


def test_empty_text_returns_empty_string():
    assert TfidfSummarizer(n_sentences=2).summarize("") == ""


def test_single_sentence_returned_as_is():
    assert TfidfSummarizer(n_sentences=3).summarize("Hello world.") == "Hello world."
