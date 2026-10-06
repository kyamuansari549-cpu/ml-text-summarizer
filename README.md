# Summarize Text with Machine Learning

Extractive text summarization using **TF-IDF** (scikit-learn). Built during the
Industrial Internship — *Data Science, AI & Machine Learning using Python* —
at the **Academy of Skill Development** (June–July 2026).

## How it works

1. The document is split into sentences and cleaned.
2. scikit-learn's `TfidfVectorizer` computes a TF-IDF weight for every word in
   every sentence (English stop words removed).
3. Each sentence is scored by its length-normalized sum of TF-IDF weights —
   sentences containing rare, document-specific words score highest.
4. The top-ranked sentences are returned **in their original order**, so the
   summary reads naturally and never invents wording (extractive, not
   abstractive).

## Project structure

```
ml-text-summarizer/
├── summarizer/
│   ├── __init__.py          # public API
│   ├── preprocessor.py      # sentence splitting + cleaning
│   ├── tfidf_summarizer.py  # TfidfSummarizer class (core logic)
│   └── utils.py             # file I/O helpers
├── examples/
│   └── sample_article.txt   # demo document
├── tests/
│   └── test_summarizer.py   # pytest suite
├── main.py                  # command-line interface
└── requirements.txt
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Summarize a file into 3 sentences:

```bash
python main.py --file examples/sample_article.txt --sentences 3
```

Keep 25% of the sentences and save to a file:

```bash
python main.py --file notes.txt --ratio 0.25 --output summary.txt
```

Use it from Python:

```python
from summarizer import TfidfSummarizer

with open("article.txt") as f:
    text = f.read()

summary = TfidfSummarizer(n_sentences=3).summarize(text)
print(summary)
```

Run the tests:

```bash
pytest
```

## Tech stack

Python · scikit-learn · NumPy · pandas · pytest
