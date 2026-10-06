"""Text preprocessing helpers: sentence splitting and cleaning."""

import re


def split_sentences(text):
    """Split raw text into a list of sentences.

    Uses a regex that splits after sentence-ending punctuation (. ! ?)
    when followed by whitespace and a capital letter or digit. This keeps
    abbreviations like "e.g." mostly intact for typical input text.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def clean_text(text):
    """Lowercase text and strip characters that carry no meaning for TF-IDF.

    Keeps letters, digits and spaces only. Punctuation is removed because
    the vectorizer tokenizes on word boundaries anyway.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
