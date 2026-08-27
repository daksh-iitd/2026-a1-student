#!/usr/bin/env python
"""
scripts/tune_bm25_plus.py — sweep BM25+'s delta parameter (Lv & Zhai,
2011) on dev topics, holding k1/b at their already-tuned values
(scripts/tune_bm25.py). delta=0 reduces exactly to plain BM25, so the
delta=0 row here should reproduce the known 0.6672 nDCG@10 baseline
exactly, as a sanity check.

Usage:
    python scripts/tune_bm25_plus.py \\
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
from submission import bm25
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

K1, B = 2.5, 0.6
DELTA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--delta-grid", type=float, nargs="+", default=DELTA_GRID)
    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus} ...")
    corpus = load_corpus(args.corpus)
    index = InvertedIndex()
    index.build(corpus)
    bm25.build(index)

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    results = []
    for delta in args.delta_grid:
        run = {qid: bm25.score(text, args.k, k1=K1, b=B, delta=delta) for qid, text in queries}
        agg = evaluate_run(run, qrels, k=args.k)["aggregate"]
        results.append((delta, agg["ndcg@10"], agg["map@10"]))
        print(f"delta={delta:<5.2f}  nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}")

    best = max(results, key=lambda r: r[1])
    print(f"\nBest: delta={best[0]}  nDCG@10={best[1]:.4f}")
    print(f"vs. delta=0 (plain BM25) baseline: delta {best[1]-results[0][1]:+.4f}")


if __name__ == "__main__":
    main()
