#!/usr/bin/env python
"""
scripts/tune_bm25_vsm_blend.py — sweep a static (non-PRF, no query
expansion) linear blend of BM25 and TF-IDF/cosine VSM:

    score(d) = w * BM25_norm(d) + (1-w) * VSM_norm(d)

Unlike submission/custom_scorer.py's Rocchio PRF (rejected: hurt on
every tested configuration, see report.tex \S5), this blends two
independent, already-implemented rankers with no query-side expansion,
so it isn't vulnerable to PRF's topic-drift failure mode. BM25's raw
scores are unbounded and VSM's are cosine similarities in [0,1] — not
comparable on a shared scale — so each is per-query min-max normalized
into [0,1] before blending. w=1.0 (pure BM25) should reproduce the known
0.6672 nDCG@10 baseline exactly, since min-max normalization is
monotonic and preserves ranking order; this is used as a sanity check.

Usage:
    python scripts/tune_bm25_vsm_blend.py \\
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

K1, B = 2.5, 0.6
W_GRID = [1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.0]


def minmax_normalize(d):
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi == lo:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus} ...")
    corpus = load_corpus(args.corpus)
    index = InvertedIndex()
    index.build(corpus)
    bm25.build(index)
    boolean_vsm.build(index)

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    BIG = 10**9

    bm25_raw = {qid: dict(bm25.score(text, BIG, k1=K1, b=B)) for qid, text in queries}
    vsm_raw = {qid: dict(boolean_vsm.vsm_score(text, BIG)) for qid, text in queries}
    bm25_norm = {qid: minmax_normalize(d) for qid, d in bm25_raw.items()}
    vsm_norm = {qid: minmax_normalize(d) for qid, d in vsm_raw.items()}

    def combined_run(w, k=args.k):
        run = {}
        for qid, _text in queries:
            bn, vn = bm25_norm[qid], vsm_norm[qid]
            union = set(bn) | set(vn)
            scored = [(d, w * bn.get(d, 0.0) + (1 - w) * vn.get(d, 0.0)) for d in union]
            scored.sort(key=lambda item: item[1], reverse=True)
            run[qid] = scored[:k]
        return run

    results = []
    for w in W_GRID:
        agg = evaluate_run(combined_run(w), qrels, k=args.k)["aggregate"]
        results.append((w, agg["ndcg@10"], agg["map@10"]))
        tag = "  (pure BM25, sanity check)" if w == 1.0 else ("  (pure VSM)" if w == 0.0 else "")
        print(f"w={w:<5.2f}  nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}{tag}")

    best = max(results, key=lambda r: r[1])
    pure_bm25 = next(r for r in results if r[0] == 1.0)
    print(f"\nBest: w={best[0]}  nDCG@10={best[1]:.4f}")
    print(f"vs. pure BM25 (w=1.0, nDCG@10={pure_bm25[1]:.4f}): delta {best[1]-pure_bm25[1]:+.4f}")


if __name__ == "__main__":
    main()
