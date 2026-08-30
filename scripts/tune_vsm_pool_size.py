#!/usr/bin/env python
"""
scripts/tune_vsm_pool_size.py — sweep custom_scorer's vsm_pool_size
(how many of BM25+'s top candidates the VSM pass is restricted to,
instead of rescoring every document any query term matches) on dev
topics, holding k1/b/delta/w/gamma at their already-tuned values.

Unlike the other tune_*.py scripts, this one is checking a performance
optimization for a quality regression, not searching for a quality
improvement — so both nDCG@10 *and* mean query latency are reported per
pool size. A very large pool_size (larger than any query's candidate
set) reproduces the unrestricted boolean_vsm.raw_scores() pass exactly,
used here as the baseline everything else is compared against.

Usage:
    python scripts/tune_vsm_pool_size.py \\
        --corpus data/full/corpus.jsonl \\
        --queries data/full/queries_dev.tsv \\
        --qrels data/full/qrels_dev.txt
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries
from submission import custom_scorer
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

K1, B, DELTA, W, GAMMA = 2.5, 0.6, 0.75, 0.8, 0.0
POOL_SIZE_GRID = [100, 200, 500, 1000, 2000, 5000, 200_000]  # 200,000 ~= unrestricted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pool-size-grid", type=int, nargs="+", default=POOL_SIZE_GRID)
    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus} ...")
    corpus = load_corpus(args.corpus)
    index = InvertedIndex()
    index.build(corpus)
    custom_scorer.build(index)

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    results = []
    for pool_size in args.pool_size_grid:
        run = {}
        latencies = []
        for qid, text in queries:
            t0 = time.perf_counter()
            run[qid] = custom_scorer.score(
                text, args.k, k1=K1, b=B, delta=DELTA, w=W, gamma=GAMMA, vsm_pool_size=pool_size
            )
            latencies.append((time.perf_counter() - t0) * 1000)
        agg = evaluate_run(run, qrels, k=args.k)["aggregate"]
        mean_latency = statistics.mean(latencies)
        results.append((pool_size, agg["ndcg@10"], agg["map@10"], mean_latency))
        label = "unrestricted" if pool_size == 200_000 else str(pool_size)
        print(
            f"pool={label:<14s} nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}"
            f"  mean_latency={mean_latency:6.1f}ms"
        )

    baseline = next(r for r in results if r[0] == 200_000)
    print(f"\nUnrestricted baseline: nDCG@10={baseline[1]:.4f}, {baseline[3]:.1f}ms/query")
    for pool_size, ndcg, _map10, latency in results:
        if pool_size == 200_000:
            continue
        speedup = baseline[3] / latency if latency else float("inf")
        print(
            f"pool={pool_size:<7d} nDCG@10 delta={ndcg - baseline[1]:+.4f}"
            f"  speedup={speedup:.1f}x"
        )


if __name__ == "__main__":
    main()
