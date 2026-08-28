#!/usr/bin/env python
"""
scripts/tune_joint_blend.py — joint grid search over k1, b, delta, w
together (previously tuned in two separate stages: k1/b via
tune_bm25.py, then delta/w on top via tune_bm25_vsm_blend.py +
tune_bm25_plus.py). Checks whether a true joint optimum differs from
stacking the two staged results.

VSM's raw scores don't depend on k1/b/delta at all, so they're computed
ONCE per query and reused across the whole grid — only BM25's raw scores
need recomputing per (k1,b,delta) combination.

Usage:
    python scripts/tune_joint_blend.py \\
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

K1_GRID = [2.0, 2.5, 3.0]
B_GRID = [0.5, 0.6, 0.7]
DELTA_GRID = [0.5, 0.75, 1.0]
W_GRID = [0.7, 0.8, 0.9]

CURRENT_BEST = {"k1": 2.5, "b": 0.6, "delta": 0.75, "w": 0.8}
CURRENT_BEST_NDCG = 0.6892  # from the last full harness run through retrieve()


def minmax(d):
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

    vsm_raw = {qid: boolean_vsm.raw_scores(text) for qid, text in queries}
    vsm_norm = {qid: minmax(d) for qid, d in vsm_raw.items()}

    results = []
    total = len(K1_GRID) * len(B_GRID) * len(DELTA_GRID) * len(W_GRID)
    done = 0
    for k1 in K1_GRID:
        for b in B_GRID:
            for delta in DELTA_GRID:
                bm25_raw = {qid: bm25.raw_scores(text, k1=k1, b=b, delta=delta) for qid, text in queries}
                bm25_norm = {qid: minmax(d) for qid, d in bm25_raw.items()}
                for w in W_GRID:
                    run = {}
                    for qid, _text in queries:
                        bn, vn = bm25_norm[qid], vsm_norm[qid]
                        union = set(bn) | set(vn)
                        scored = sorted(
                            ((d, w * bn.get(d, 0.0) + (1 - w) * vn.get(d, 0.0)) for d in union),
                            key=lambda x: -x[1],
                        )[: args.k]
                        run[qid] = scored
                    agg = evaluate_run(run, qrels, k=args.k)["aggregate"]
                    results.append({"k1": k1, "b": b, "delta": delta, "w": w, "ndcg@10": agg["ndcg@10"]})
                    done += 1
                    if done % 9 == 0:
                        print(f"[{done}/{total}] k1={k1} b={b} delta={delta}  best-w-so-far nDCG@10={agg['ndcg@10']:.4f}")

    results.sort(key=lambda r: r["ndcg@10"], reverse=True)
    print(f"\n{'k1':>5} {'b':>5} {'delta':>6} {'w':>5} {'nDCG@10':>10}")
    for r in results[:10]:
        print(f"{r['k1']:>5} {r['b']:>5} {r['delta']:>6} {r['w']:>5} {r['ndcg@10']:>10.4f}")

    best = results[0]
    print(f"\nCurrent staged-tuning result: {CURRENT_BEST}  nDCG@10={CURRENT_BEST_NDCG:.4f}")
    print(f"Best from joint search: {best}  nDCG@10={best['ndcg@10']:.4f}")
    delta_ndcg = best["ndcg@10"] - CURRENT_BEST_NDCG
    print(f"Delta: {delta_ndcg:+.4f}  -> {'JOINT SEARCH WINS' if delta_ndcg > 0.002 else 'NO MEANINGFUL DIFFERENCE (staged tuning already near-optimal)'}")


if __name__ == "__main__":
    main()
