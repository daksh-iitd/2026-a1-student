#!/usr/bin/env python
"""
scripts/tune_rrf.py — Reciprocal Rank Fusion (Cormack, Clarke & Buettcher,
2009) as an alternative to the current min-max-normalize-then-linearly-
blend approach (submission/custom_scorer.py) for combining BM25+ and VSM.

RRF combines rankers by RANK POSITION, not raw score:
    score(d) = sum_s  1 / (k_rrf + rank_s(d))
for each system s the document appears in (rank_s undefined -> that term
is 0). Never touches raw scores at all, so it sidesteps the min-max
blend's real weakness: one document with an extreme BM25 score can
compress every other document's normalized score toward 0, distorting
the blend. RRF can't be distorted that way, since only relative rank
order matters, not score magnitude.

Honest cost noted up front: RRF needs each system's RANKED list, not
just raw scores, so unlike the additive blend (one sort total, after the
"Improving Query Latency" fix), this needs 2 sorts (one per system) plus
1 more for the final combined ranking -- the same 3-sort cost profile the
latency fix removed for the additive blend, reintroduced here because
rank fusion is structurally different from score fusion, not because of
a similar implementation oversight.

Usage:
    python scripts/tune_rrf.py \\
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

K1, B, DELTA = 2.5, 0.6, 0.75  # current tuned BM25+ params, held fixed
K_RRF_GRID = [10, 20, 40, 60, 80, 100, 150]


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

    bm25_ranks = {}
    vsm_ranks = {}
    for qid, text in queries:
        bm25_ranked = sorted(bm25.raw_scores(text, k1=K1, b=B, delta=DELTA).items(), key=lambda x: -x[1])
        vsm_ranked = sorted(boolean_vsm.raw_scores(text).items(), key=lambda x: -x[1])
        bm25_ranks[qid] = {doc_id: i + 1 for i, (doc_id, _s) in enumerate(bm25_ranked)}
        vsm_ranks[qid] = {doc_id: i + 1 for i, (doc_id, _s) in enumerate(vsm_ranked)}

    def rrf_run(k_rrf, k=args.k):
        run = {}
        for qid, _text in queries:
            br, vr = bm25_ranks[qid], vsm_ranks[qid]
            candidates = set(br) | set(vr)
            scored = []
            for doc_id in candidates:
                s = 0.0
                if doc_id in br:
                    s += 1.0 / (k_rrf + br[doc_id])
                if doc_id in vr:
                    s += 1.0 / (k_rrf + vr[doc_id])
                scored.append((doc_id, s))
            scored.sort(key=lambda item: item[1], reverse=True)
            run[qid] = scored[:k]
        return run

    print(f"\n{'k_rrf':>7} {'nDCG@10':>10} {'MAP@10':>10}")
    results = []
    for k_rrf in K_RRF_GRID:
        agg = evaluate_run(rrf_run(k_rrf), qrels, k=args.k)["aggregate"]
        results.append((k_rrf, agg["ndcg@10"], agg["map@10"]))
        print(f"{k_rrf:>7} {agg['ndcg@10']:>10.4f} {agg['map@10']:>10.4f}")

    best = max(results, key=lambda r: r[1])
    print(f"\nBest RRF: k_rrf={best[0]}  nDCG@10={best[1]:.4f}")
    print(f"[reference] current additive blend (w=0.8): nDCG@10=0.6892")
    print(f"RRF vs. additive blend: {'RRF WINS' if best[1] > 0.6892 else 'ADDITIVE BLEND WINS'} (delta {best[1]-0.6892:+.4f})")


if __name__ == "__main__":
    main()
