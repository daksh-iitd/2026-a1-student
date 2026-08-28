#!/usr/bin/env python
"""
scripts/tune_coordination.py — sweep custom_scorer's coordination-level
bonus weight (gamma) on dev topics, holding k1/b/delta/w at their
already-tuned values. gamma=0 reproduces the known 0.690 nDCG@10 blend
baseline exactly, as a sanity check.

Usage:
    python scripts/tune_coordination.py \\
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
from submission import custom_scorer
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

K1, B, DELTA, W = 2.5, 0.6, 0.75, 0.8
GAMMA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--gamma-grid", type=float, nargs="+", default=GAMMA_GRID)
    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus} ...")
    corpus = load_corpus(args.corpus)
    index = InvertedIndex()
    index.build(corpus)
    custom_scorer.build(index)

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    results = []
    for gamma in args.gamma_grid:
        run = {
            qid: custom_scorer.score(text, args.k, k1=K1, b=B, delta=DELTA, w=W, gamma=gamma)
            for qid, text in queries
        }
        agg = evaluate_run(run, qrels, k=args.k)["aggregate"]
        results.append((gamma, agg["ndcg@10"], agg["map@10"]))
        tag = "  (gamma=0 sanity check, should match 0.6896)" if gamma == 0.0 else ""
        print(f"gamma={gamma:<5.2f}  nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}{tag}")

    best = max(results, key=lambda r: r[1])
    base = next(r for r in results if r[0] == 0.0)
    print(f"\nBest: gamma={best[0]}  nDCG@10={best[1]:.4f}")
    print(f"vs. gamma=0 (no coordination bonus): {'HELPS' if best[1] > base[1] else 'DOES NOT HELP'} (delta {best[1]-base[1]:+.4f})")


if __name__ == "__main__":
    main()
