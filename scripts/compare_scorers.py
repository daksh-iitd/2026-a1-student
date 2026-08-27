#!/usr/bin/env python
"""
scripts/compare_scorers.py — evaluate VSM (TF-IDF cosine) against tuned
BM25 on the same dev queries/qrels, for the report's required "table
comparing Boolean/VSM vs. BM25 performance on the dev set" (assignment
Section 6). Boolean AND/OR itself isn't included here: boolean_search()
returns an unranked set, so nDCG@10/MAP@10 (both rank-sensitive) aren't
well-defined for it without an arbitrary ordering convention — it's a
correctness-checked candidate filter, not a competing ranked scorer.

Usage:
    python scripts/compare_scorers.py \\
        --corpus data/full/corpus.jsonl \\
        --queries data/full/queries_dev.tsv \\
        --qrels data/full/qrels_dev.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries
from submission import bm25, boolean_vsm
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

K1, B = 2.5, 0.6  # from scripts/tune_bm25.py


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    index = InvertedIndex()
    index.build(corpus)
    bm25.build(index)
    boolean_vsm.build(index)

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    bm25_run = {qid: bm25.score(text, args.k, k1=K1, b=B) for qid, text in queries}
    vsm_run = {qid: boolean_vsm.vsm_score(text, args.k) for qid, text in queries}

    bm25_agg = evaluate_run(bm25_run, qrels, k=args.k)["aggregate"]
    vsm_agg = evaluate_run(vsm_run, qrels, k=args.k)["aggregate"]

    print(f"{'Scorer':<10} {'nDCG@10':>10} {'MAP@10':>10} {'MRR':>10} {'P@10':>10}")
    for name, agg in (("BM25", bm25_agg), ("VSM", vsm_agg)):
        print(f"{name:<10} {agg['ndcg@10']:10.4f} {agg['map@10']:10.4f} {agg['mrr']:10.4f} {agg[f'p@{args.k}']:10.4f}")


if __name__ == "__main__":
    main()
