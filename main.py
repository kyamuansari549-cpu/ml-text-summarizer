#!/usr/bin/env python3
"""Command-line interface for the ML text summarizer.

Examples
--------
Summarize a file into 3 sentences::

    python main.py --file examples/sample_article.txt --sentences 3

Keep 25% of the sentences and save the summary::

    python main.py --file notes.txt --ratio 0.25 --output summary.txt
"""

import argparse

from summarizer import TfidfSummarizer
from summarizer.utils import read_text_file, write_text_file


def main():
    parser = argparse.ArgumentParser(
        description="Extractive text summarization using TF-IDF (scikit-learn)."
    )
    parser.add_argument("--file", required=True, help="Path to the input .txt file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sentences", type=int, default=None,
                       help="Exact number of sentences in the summary")
    group.add_argument("--ratio", type=float, default=0.3,
                       help="Fraction of sentences to keep (default: 0.3)")
    parser.add_argument("--output", default=None,
                        help="Write the summary to this file instead of stdout")
    args = parser.parse_args()

    text = read_text_file(args.file)
    summarizer = TfidfSummarizer(n_sentences=args.sentences, ratio=args.ratio)
    summary = summarizer.summarize(text)

    if args.output:
        write_text_file(args.output, summary)
        print(f"Summary written to {args.output}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
