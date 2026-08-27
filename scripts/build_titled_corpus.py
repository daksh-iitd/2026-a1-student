#!/usr/bin/env python
"""
scripts/build_titled_corpus.py — research-only data prep for the field-
weighting experiment (BM25F-style: w_title*BM25(q,title) +
w_body*BM25(q,abstract)).

scripts/download_full_corpus.py writes data/full/corpus.jsonl with title
and abstract already concatenated into one `text` field (ir_datasets'
`doc.default_text()`), matching the {"doc_id", "text"} format
build_index() is actually contracted to receive (data/README.md) — the
real grading corpus is not guaranteed to carry a separate title field, so
this script's output is NOT fed to submission.retrieve.build_index(); it
exists purely so scripts/tune_field_weights.py can evaluate whether field
weighting is worth pursuing at all, using ir_datasets' true (title, text)
split for beir/trec-covid.

Usage:
    python scripts/build_titled_corpus.py --out data/full/corpus_titled.jsonl
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="beir/trec-covid")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "full", "corpus_titled.jsonl"))
    args = parser.parse_args()

    import ir_datasets

    dataset = ir_datasets.load(args.dataset)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for doc in dataset.docs_iter():
            title = getattr(doc, "title", "") or ""
            text = getattr(doc, "text", "") or ""
            f.write(json.dumps({"doc_id": doc.doc_id, "title": title, "text": text}) + "\n")
            n += 1
    print(f"Wrote {n} (doc_id, title, text) records to {args.out}")


if __name__ == "__main__":
    main()
