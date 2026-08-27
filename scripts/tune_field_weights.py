#!/usr/bin/env python
"""
scripts/tune_field_weights.py — research experiment: does BM25F-style
field weighting

    score(q, d) = w_title * BM25(q, title) + w_body * BM25(q, abstract)

beat single-field BM25 over the concatenated title+abstract text
(nDCG@10 = 0.6672 at k1=2.5, b=0.6, from scripts/tune_bm25.py)?

Not part of the graded submission — see scripts/build_titled_corpus.py's
docstring for why (the real corpus format guarantees only one flat
`text` field; this needs ir_datasets' true title/abstract split, which
only exists for local experimentation on the trec-covid placeholder
data).

Builds two separate InvertedIndex objects (title-only, abstract-only)
and runs BM25 over each ONCE per query with an unbounded k, caching the
full per-doc score dict for both fields. Every (w_title, w_body)
combination in the sweep is then just a cheap weighted recombination of
those two cached dicts — no rescoring needed per grid point.

Usage:
    python scripts/build_titled_corpus.py
    python scripts/tune_field_weights.py \\
        --corpus data/full/corpus_titled.jsonl \\
        --queries data/full/queries_dev.tsv \\
        --qrels data/full/qrels_dev.txt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries
from submission import bm25
from submission.indexer import InvertedIndex

K1, B = 2.5, 0.6  # reuse the single-field-tuned values for both fields as a first pass
W_TITLE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def load_titled_corpus(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append((obj["doc_id"], obj.get("title", ""), obj.get("text", "")))
    return docs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, help="output of scripts/build_titled_corpus.py")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading titled corpus from {args.corpus} ...")
    docs = load_titled_corpus(args.corpus)
    title_corpus = [(doc_id, title) for doc_id, title, _text in docs]
    body_corpus = [(doc_id, text) for doc_id, _title, text in docs]

    title_index = InvertedIndex()
    title_index.build(title_corpus)
    body_index = InvertedIndex()
    body_index.build(body_corpus)
    print(f"Indexed {title_index.N} docs (title field, avgdl={title_index.avg_doc_len:.1f} tokens)")
    print(f"Indexed {body_index.N} docs (abstract field, avgdl={body_index.avg_doc_len:.1f} tokens)")

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    BIG = 10**9  # k large enough that score() returns every matched doc, unbounded

    bm25.build(title_index)
    title_scores = {qid: dict(bm25.score(text, BIG, k1=K1, b=B)) for qid, text in queries}
    bm25.build(body_index)
    body_scores = {qid: dict(bm25.score(text, BIG, k1=K1, b=B)) for qid, text in queries}

    def combined_run(w_title, w_body, k=args.k):
        run = {}
        for qid, _text in queries:
            ts, bs = title_scores[qid], body_scores[qid]
            union = set(ts) | set(bs)
            scored = [(d, w_title * ts.get(d, 0.0) + w_body * bs.get(d, 0.0)) for d in union]
            scored.sort(key=lambda item: item[1], reverse=True)
            run[qid] = scored[:k]
        return run

    def ndcg_map(w_title, w_body):
        agg = evaluate_run(combined_run(w_title, w_body), qrels, k=args.k)["aggregate"]
        return agg["ndcg@10"], agg["map@10"]

    print(f"\n{'config':<28} {'nDCG@10':>10} {'MAP@10':>10}")
    abstract_only = ndcg_map(0.0, 1.0)
    print(f"{'abstract only (w_title=0)':<28} {abstract_only[0]:10.4f} {abstract_only[1]:10.4f}")
    title_only = ndcg_map(1.0, 0.0)
    print(f"{'title only (w_body=0)':<28} {title_only[0]:10.4f} {title_only[1]:10.4f}")
    print(f"{'[reference] concat. text':<28} {'0.6672':>10} {'0.0166':>10}  (single-field BM25, scripts/tune_bm25.py)")
    print()

    results = []
    for w_title in W_TITLE_GRID:
        ndcg, map10 = ndcg_map(w_title, 1.0)
        results.append((w_title, ndcg, map10))
        print(f"w_title={w_title:<5.2f} w_body=1.00       {ndcg:10.4f} {map10:10.4f}")

    best = max(results, key=lambda r: r[1])
    print(f"\nBest w_title (w_body=1.0 fixed): {best[0]}  nDCG@10={best[1]:.4f}")
    print(f"vs. abstract-only:  {'HELPS' if best[1] > abstract_only[0] else 'DOES NOT HELP'} (delta {best[1]-abstract_only[0]:+.4f})")
    print(f"vs. concatenated-text baseline (0.6672): {'HELPS' if best[1] > 0.6672 else 'DOES NOT BEAT IT'} (delta {best[1]-0.6672:+.4f})")


if __name__ == "__main__":
    main()
