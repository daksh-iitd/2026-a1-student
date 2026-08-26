"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5 of the
assignment spec, "Submission Interface & Conformance Checking") — do not
rename them, change their signatures, or move them out of this file.

    build_index(corpus_path: str, index_dir: str) -> None
        Called once, in its own process, with the path to a corpus.jsonl
        file (see data/README.md) and a directory to write your index
        into. Build whatever index and statistics you need, and WRITE
        THEM TO index_dir. The harness runs build_index() and
        load_index()/retrieve() in two SEPARATE processes on purpose (see
        harness/run_harness.py's module docstring) — nothing you only
        hold in memory here survives into load_index(). This call is
        timed as your "index build time" efficiency metric. The harness
        also measures the on-disk byte size of index_dir once this
        returns — that's your "index size" score (assignment Section 7),
        so write only what retrieve() actually needs, and consider
        compressing it.

    load_index(index_dir: str) -> None
        Called once, in a fresh process, before any retrieve() calls.
        Reconstruct everything retrieve() needs by reading index_dir —
        and only index_dir; there is no leftover state from
        build_index() to fall back on. Timed as your "index load time".

    retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]
        Called once per query, only after load_index() has run in the
        same process. Return up to k (doc_id, score) pairs, sorted by
        score descending (highest score = most relevant). This is exactly
        the ranking the harness scores with nDCG@10 / MAP@10. doc_id values
        must be ones that appeared in the corpus passed to build_index().

This file ships with a trivial, fully-working baseline — return the first
k documents in the order build_index() saw them, ignoring the query
entirely — wired up below. It actually persists to disk and reloads
correctly, so it exercises the full build -> disk -> fresh process -> load
-> query path end-to-end from your very first commit. Its scores will be
close to zero; replace the logic, but keep the same
persist-in-build / reconstruct-in-load shape.
"""
from typing import List, Tuple

from submission import bm25, boolean_vsm
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

# ---------------------------------------------------------------------------
# Module-level state. load_index() populates this (via bm25.build() /
# boolean_vsm.build()); retrieve() reads it. build_index() runs in a
# SEPARATE process and cannot rely on this state surviving into
# load_index()/retrieve() — everything needed at query time is written to
# index_dir in build_index() (via InvertedIndex.save()) and read back in
# load_index() (via InvertedIndex.load()).
# ---------------------------------------------------------------------------
_LOADED = False


def build_index(corpus_path: str, index_dir: str) -> None:
    """Load the corpus, build the inverted index, and persist it to
    `index_dir`.

    Runs once, in its own process, before load_index() ever runs. Heavy
    one-time work — tokenising the whole corpus, building postings lists,
    computing collection statistics — belongs here, not in retrieve(), so
    it doesn't get charged against your per-query latency.
    """
    corpus = load_corpus(corpus_path)
    index = InvertedIndex()
    index.build(corpus)
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    """Reconstruct the inverted index, reading only from `index_dir`, and
    prepare the scorers. Runs once, in a fresh process, before any
    retrieve() calls — there is no leftover state from build_index() to
    rely on.
    """
    global _LOADED
    index = InvertedIndex.load(index_dir)
    bm25.build(index)
    boolean_vsm.build(index)
    _LOADED = True


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, best first.

    Uses BM25 as the primary scorer — it carries 70% + 10% of the
    leaderboard weight (nDCG@10, MAP@10), and Boolean/VSM is graded
    separately for correctness rather than being the competition entry.

    k1/b below were chosen by a grid sweep (scripts/tune_bm25.py) over
    the released dev topics/qrels, maximizing nDCG@10 — see
    runs/bm25_sweep.csv for the full grid. Re-run the sweep and update
    these if the corpus or dev topics change; a value tuned on one
    corpus is not guaranteed to transfer to another.
    """
    if not _LOADED:
        raise RuntimeError(
            "retrieve() called before load_index(); the harness always "
            "calls build_index(corpus_path, index_dir) and then "
            "load_index(index_dir) — in that order, in two separate "
            "processes — before any retrieve() calls. If you're testing "
            "manually, do the same."
        )

    return bm25.score(query, k, k1=2.5, b=0.6)
