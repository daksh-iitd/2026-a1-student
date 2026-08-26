#!/usr/bin/env python
"""
scripts/tune_bm25.py — sweep BM25's k1 and b on a dev query/qrels set and
report nDCG@10 (the assignment's primary leaderboard metric, Section 7)
for each combination.

Not part of the graded submission (build_index/load_index/retrieve in
submission/retrieve.py are) — a local dev tool for producing the
"parameter search procedure for k1 and b" and its plot, required by the
assignment report (Section 6). Builds the index once and reuses it across
the whole grid (re-running harness.run_harness per combination would
rebuild the index every time, which is wasted work since k1/b only affect
scoring, not indexing), and drives submission.bm25 directly rather than
the two-subprocess harness path, since the on-disk persistence contract
isn't what's being exercised here — only ranking quality is.

Usage:
    python scripts/tune_bm25.py \\
        --corpus data/full/corpus.jsonl \\
        --queries data/full/queries_dev.tsv \\
        --qrels data/full/qrels_dev.txt \\
        --out runs/bm25_sweep.csv

Tune only on dev topics, never on held-out ones (assignment Section 7,
"Report quality": tuning methodology must be sound) — held-out topics
aren't distributed to you anyway (data/README.md).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries
from submission import bm25
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

DEFAULT_K1_GRID = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]
DEFAULT_B_GRID = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10, help="retrieval depth (matches nDCG@10/MAP@10 cutoff)")
    parser.add_argument("--k1-grid", type=float, nargs="+", default=DEFAULT_K1_GRID)
    parser.add_argument("--b-grid", type=float, nargs="+", default=DEFAULT_B_GRID)
    parser.add_argument("--out", default=None, help="CSV path to write the full sweep results")
    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus} ...")
    corpus = load_corpus(args.corpus)
    index = InvertedIndex()
    index.build(corpus)
    bm25.build(index)
    print(f"Indexed {index.N} documents.")

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    results = []
    total = len(args.k1_grid) * len(args.b_grid)
    done = 0
    for k1 in args.k1_grid:
        for b in args.b_grid:
            run = {qid: bm25.score(text, args.k, k1=k1, b=b) for qid, text in queries}
            agg = evaluate_run(run, qrels, k=args.k)["aggregate"]
            results.append(
                {"k1": k1, "b": b, "ndcg@10": agg["ndcg@10"], "map@10": agg["map@10"], "mrr": agg["mrr"]}
            )
            done += 1
            print(
                f"[{done}/{total}] k1={k1:.2f} b={b:.2f}  "
                f"nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}"
            )

    results.sort(key=lambda r: r["ndcg@10"], reverse=True)
    print("\nTop 10 by nDCG@10:")
    print(f"{'k1':>6} {'b':>6} {'nDCG@10':>10} {'MAP@10':>10} {'MRR':>10}")
    for r in results[:10]:
        print(f"{r['k1']:6.2f} {r['b']:6.2f} {r['ndcg@10']:10.4f} {r['map@10']:10.4f} {r['mrr']:10.4f}")

    best = results[0]
    print(f"\nBest: k1={best['k1']}, b={best['b']}  (nDCG@10={best['ndcg@10']:.4f})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["k1", "b", "ndcg@10", "map@10", "mrr"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nFull sweep written to {args.out}")


if __name__ == "__main__":
    main()
