#!/usr/bin/env python
"""
scripts/tune_bm25f_prefix.py — two things, using scripts/bm25f_core.py's
proper single-saturation BM25F against corpora in the STANDARD
{"doc_id", "text"}-only format (data/README.md) — no ir_datasets title
field assumed:

1. On trec-covid: does the better formula recover more of the true-field
   ceiling (0.6861, tune_field_weights.py) from a heuristic token-prefix
   pseudo-title than the earlier naive-blend attempt did (which only
   recovered 5.5% of it — tune_prefix_weights.py, best nDCG@10=0.6682)?

2. On fiqa: the graceful-degradation path this whole design needs to be
   safe against an unknown held-out corpus. fiqa's documents have NO
   title field at all at the ir_datasets source level (confirmed: its
   GenericDoc type only has doc_id/text) — every title_text passed to
   FieldIndex.build() here is "", exactly simulating "field detection
   found nothing" rather than applying a prefix heuristic that has no
   natural meaning for fiqa's short forum-style passages. Checks: no
   crash (no division by zero on avg_len_title=0), and the result is not
   worse than plain BM25 on the same corpus.

Usage:
    python scripts/tune_bm25f_prefix.py --mode trec-covid \\
        --corpus data/full/corpus.jsonl \\
        --queries data/full/queries_dev.tsv \\
        --qrels data/full/qrels_dev.txt
    python scripts/tune_bm25f_prefix.py --mode fiqa-no-title \\
        --corpus data/full/fiqa/corpus.jsonl \\
        --queries data/full/fiqa/queries_dev.tsv \\
        --qrels data/full/fiqa/qrels_dev.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bm25f_core import FieldIndex, bm25f_raw_scores  # noqa: E402
from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries
from submission import bm25 as plain_bm25
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex, tokenize

K1 = 2.5
PREFIX_LEN_GRID = [8, 10, 12, 15]
W_TITLE_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def evaluate(index, queries, qrels, k, k1, w_title, b_title, b_body):
    run = {
        qid: sorted(
            bm25f_raw_scores(text, index, k1=k1, w_title=w_title, b_title=b_title, b_body=b_body).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        for qid, text in queries
    }
    return evaluate_run(run, qrels, k=k)["aggregate"]


def run_trec_covid_prefix(args):
    corpus = load_corpus(args.corpus)
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    print(f"{'prefix_len':>10} {'w_title':>8} {'nDCG@10':>10} {'MAP@10':>10}")
    best = None
    for prefix_len in PREFIX_LEN_GRID:
        prefix_docs = []
        for doc_id, text in corpus:
            tokens = tokenize(text)
            prefix_docs.append((doc_id, " ".join(tokens[:prefix_len]), " ".join(tokens[prefix_len:])))
        index = FieldIndex()
        index.build(prefix_docs)

        for w_title in W_TITLE_GRID:
            agg = evaluate(index, queries, qrels, args.k, K1, w_title, b_title=0.2, b_body=0.4)
            print(f"{prefix_len:>10} {w_title:>8.2f} {agg['ndcg@10']:>10.4f} {agg['map@10']:>10.4f}")
            if best is None or agg["ndcg@10"] > best[0]:
                best = (agg["ndcg@10"], prefix_len, w_title)

    print(f"\n[reference] plain single-field BM25:                              nDCG@10=0.6672")
    print(f"[reference] true-field ceiling (tune_field_weights.py):           nDCG@10=0.6861")
    print(f"[reference] earlier naive-blend prefix heuristic (tune_prefix_weights.py): nDCG@10=0.6682")
    ndcg, prefix_len, w_title = best
    print(f"\nBest here: prefix_len={prefix_len}, w_title={w_title}  nDCG@10={ndcg:.4f}")
    recovered = (ndcg - 0.6672) / (0.6861 - 0.6672) * 100 if ndcg > 0.6672 else 0.0
    print(f"Fraction of true-field ceiling recovered: {recovered:.1f}% (earlier naive-blend heuristic recovered 5.5%)")


def run_fiqa_no_title(args):
    corpus = load_corpus(args.corpus)
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    # Simulates "field detection found no title for this corpus" --
    # every title_text is "", exactly as it would be if build_index()
    # inspected the corpus and concluded there's nothing to split.
    no_title_docs = [(doc_id, "", text) for doc_id, text in corpus]
    index = FieldIndex()
    index.build(no_title_docs)
    print(f"avg_len_title (should be exactly 0.0): {index.avg_len_title}")
    assert index.avg_len_title == 0.0, "expected zero title tokens across the whole title-less corpus"

    agg = evaluate(index, queries, qrels, args.k, K1, w_title=1.0, b_title=0.2, b_body=0.4)
    print(f"BM25F, title-less fallback path: nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}")

    plain_index = InvertedIndex()
    plain_index.build(corpus)
    plain_bm25.build(plain_index)
    plain_run = {qid: plain_bm25.score(text, args.k, k1=K1, b=0.6) for qid, text in queries}
    plain_agg = evaluate_run(plain_run, qrels, k=args.k)["aggregate"]
    print(f"Plain BM25 (same corpus, same k1):   nDCG@10={plain_agg['ndcg@10']:.4f}  MAP@10={plain_agg['map@10']:.4f}")
    print(f"\nNo crash on avg_len_title=0.0: PASS")
    print(f"BM25F fallback vs. plain BM25: {'MATCHES/CLOSE' if abs(agg['ndcg@10']-plain_agg['ndcg@10']) < 0.01 else 'DIVERGES'} (delta {agg['ndcg@10']-plain_agg['ndcg@10']:+.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["trec-covid", "fiqa-no-title"], required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "trec-covid":
        run_trec_covid_prefix(args)
    else:
        run_fiqa_no_title(args)


if __name__ == "__main__":
    main()
