#!/usr/bin/env python
"""
scripts/tune_prefix_weights.py — submission-safe field-weighting variant.

scripts/tune_field_weights.py showed BM25F-style field weighting with
*true* title/abstract fields helps (+0.019 nDCG@10 over concatenated-text
BM25) — but that experiment depends on ir_datasets' title/text split,
which the real grading corpus (format: {"doc_id", "text"} only, per
data/README.md) is not guaranteed to provide. A naive "split text on the
first period" heuristic was tried as a substitute and rejected: titles in
this corpus don't end in periods, so the split mostly captures
"title + first sentence of abstract" together (only ~5% exact match
against the true title, avg 34 tokens vs. the true ~10).

This script tries a punctuation-independent alternative instead: treat
the first `prefix_len` *tokens* of the already-tokenized text (after the
standard stopword/stemming pipeline) as a pseudo-title, and the rest as
body. This works on any corpus in the documented format — no title
field, no sentence-boundary assumption — so, unlike
tune_field_weights.py, a result here could actually be wired into
retrieve() if it holds up.

Usage:
    python scripts/tune_prefix_weights.py \\
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
from submission.indexer import InvertedIndex, tokenize

K1, B = 2.5, 0.6
PREFIX_LEN_GRID = [8, 10, 12, 15, 20]
W_TITLE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def split_prefix(corpus, prefix_len):
    """corpus: list of (doc_id, text). Returns (prefix_corpus, rest_corpus)
    as lists of (doc_id, pseudo_text), splitting each doc's *tokenized*
    stream (stopwords/stemming already applied) at `prefix_len` tokens.
    Rejoining tokens with spaces and re-tokenizing downstream in
    InvertedIndex.build() is idempotent here since the tokens are already
    lowercase/alnum/stemmed."""
    prefix_corpus, rest_corpus = [], []
    for doc_id, text in corpus:
        tokens = tokenize(text)
        prefix_corpus.append((doc_id, " ".join(tokens[:prefix_len])))
        rest_corpus.append((doc_id, " ".join(tokens[prefix_len:])))
    return prefix_corpus, rest_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, help="the STANDARD corpus.jsonl (doc_id, text only)")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus} ...")
    corpus = load_corpus(args.corpus)
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    BIG = 10**9

    best_overall = None
    for prefix_len in PREFIX_LEN_GRID:
        prefix_corpus, rest_corpus = split_prefix(corpus, prefix_len)
        prefix_index = InvertedIndex()
        prefix_index.build(prefix_corpus)
        rest_index = InvertedIndex()
        rest_index.build(rest_corpus)

        bm25.build(prefix_index)
        prefix_scores = {qid: dict(bm25.score(text, BIG, k1=K1, b=B)) for qid, text in queries}
        bm25.build(rest_index)
        rest_scores = {qid: dict(bm25.score(text, BIG, k1=K1, b=B)) for qid, text in queries}

        def combined_run(w_prefix, w_rest, k=args.k):
            run = {}
            for qid, _text in queries:
                ps, rs = prefix_scores[qid], rest_scores[qid]
                union = set(ps) | set(rs)
                scored = [(d, w_prefix * ps.get(d, 0.0) + w_rest * rs.get(d, 0.0)) for d in union]
                scored.sort(key=lambda item: item[1], reverse=True)
                run[qid] = scored[:k]
            return run

        for w_prefix in W_TITLE_GRID:
            agg = evaluate_run(combined_run(w_prefix, 1.0), qrels, k=args.k)["aggregate"]
            ndcg = agg["ndcg@10"]
            print(f"prefix_len={prefix_len:3d} w_prefix={w_prefix:<5.2f} w_rest=1.00   nDCG@10={ndcg:.4f}  MAP@10={agg['map@10']:.4f}")
            if best_overall is None or ndcg > best_overall[0]:
                best_overall = (ndcg, prefix_len, w_prefix)

    print(f"\n[reference] concatenated-text single-field BM25: nDCG@10=0.6672")
    print(f"[reference] true title/abstract BM25F (upper bound, not submittable): nDCG@10=0.6861")
    ndcg, prefix_len, w_prefix = best_overall
    print(f"\nBest prefix-heuristic config: prefix_len={prefix_len}, w_prefix={w_prefix}  nDCG@10={ndcg:.4f}")
    print(f"vs. concatenated-text baseline: {'HELPS' if ndcg > 0.6672 else 'DOES NOT HELP'} (delta {ndcg-0.6672:+.4f})")
    print(f"fraction of the true-field upper-bound gain recovered: {(ndcg-0.6672)/(0.6861-0.6672):.1%}" if ndcg > 0.6672 else "")


if __name__ == "__main__":
    main()
