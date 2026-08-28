#!/usr/bin/env python
"""
scripts/tune_bm25f_true_fields.py — staged parameter search for proper
single-saturation BM25F (scripts/bm25f_core.py) against trec-covid's
TRUE title/abstract split (scripts/build_titled_corpus.py's output),
answering: does a properly-designed BM25F beat the earlier naive
two-pass blend (scripts/tune_field_weights.py, best nDCG@10=0.6861) on
the exact same data?

Staged search, not a full 4D grid ("more than 50 dev queries can
resolve" per the design this follows): (1) sweep w_title with b_title,
b_body, k1 at sensible starting defaults; (2) sweep b_body at the best
w_title; (3) sweep b_title at the best w_title/b_body; (4) a small local
refinement grid around the result.

Usage:
    python scripts/build_titled_corpus.py   # if corpus_titled.jsonl doesn't exist yet
    python scripts/tune_bm25f_true_fields.py \\
        --corpus data/full/corpus_titled.jsonl \\
        --queries data/full/queries_dev.tsv \\
        --qrels data/full/qrels_dev.txt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bm25f_core import FieldIndex, bm25f_raw_scores  # noqa: E402  (sys.path set above)
from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries

K1 = 2.5  # from scripts/tune_bm25.py; not re-swept here to keep the search staged/bounded


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading titled corpus from {args.corpus} ...")
    docs = load_titled_corpus(args.corpus)
    index = FieldIndex()
    index.build(docs)
    print(f"Indexed {index.N} docs. avg title len={index.avg_len_title:.1f}, avg body len={index.avg_len_body:.1f}")

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    # Stage 0: reference points
    body_only = evaluate(index, queries, qrels, args.k, K1, w_title=0.0, b_title=0.2, b_body=0.4)
    print(f"\nBody-only (w_title=0):  nDCG@10={body_only['ndcg@10']:.4f}")
    print("[reference] plain single-field BM25 (concatenated text):     nDCG@10=0.6672")
    print("[reference] earlier naive two-pass blend (tune_field_weights.py): nDCG@10=0.6861\n")

    # Stage 1: sweep w_title, others at defaults
    b_title, b_body = 0.2, 0.4
    print(f"Stage 1: sweep w_title (b_title={b_title}, b_body={b_body}, k1={K1})")
    best_w, best_ndcg = 0.0, body_only["ndcg@10"]
    for w_title in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        agg = evaluate(index, queries, qrels, args.k, K1, w_title, b_title, b_body)
        print(f"  w_title={w_title:<5.2f}  nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}")
        if agg["ndcg@10"] > best_ndcg:
            best_ndcg, best_w = agg["ndcg@10"], w_title
    print(f"  -> best w_title={best_w}  (nDCG@10={best_ndcg:.4f})")

    # Stage 2: sweep b_body at best w_title
    print(f"\nStage 2: sweep b_body (w_title={best_w}, b_title={b_title}, k1={K1})")
    best_bbody, best_ndcg2 = b_body, best_ndcg
    for cand_b_body in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]:
        agg = evaluate(index, queries, qrels, args.k, K1, best_w, b_title, cand_b_body)
        print(f"  b_body={cand_b_body:<5.2f}  nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}")
        if agg["ndcg@10"] > best_ndcg2:
            best_ndcg2, best_bbody = agg["ndcg@10"], cand_b_body
    print(f"  -> best b_body={best_bbody}  (nDCG@10={best_ndcg2:.4f})")

    # Stage 3: sweep b_title at best w_title/b_body
    print(f"\nStage 3: sweep b_title (w_title={best_w}, b_body={best_bbody}, k1={K1})")
    best_btitle, best_ndcg3 = b_title, best_ndcg2
    for cand_b_title in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]:
        agg = evaluate(index, queries, qrels, args.k, K1, best_w, cand_b_title, best_bbody)
        print(f"  b_title={cand_b_title:<5.2f}  nDCG@10={agg['ndcg@10']:.4f}  MAP@10={agg['map@10']:.4f}")
        if agg["ndcg@10"] > best_ndcg3:
            best_ndcg3, best_btitle = agg["ndcg@10"], cand_b_title
    print(f"  -> best b_title={best_btitle}  (nDCG@10={best_ndcg3:.4f})")

    print(f"\nFinal: w_title={best_w}, b_title={best_btitle}, b_body={best_bbody}, k1={K1}")
    print(f"nDCG@10={best_ndcg3:.4f}")
    print(f"vs. earlier naive two-pass blend (0.6861): {'BETTER' if best_ndcg3 > 0.6861 else 'NOT BETTER'} (delta {best_ndcg3-0.6861:+.4f})")
    print(f"vs. plain single-field BM25 (0.6672): delta {best_ndcg3-0.6672:+.4f}")


if __name__ == "__main__":
    main()
