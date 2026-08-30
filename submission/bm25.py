"""
submission/bm25.py — Okapi BM25 ranking.

Required component (assignment Section 4.1): "a BM25 implementation with
tunable k1 and b." See the assignment background (Section 3) for the
Robertson & Walker / Robertson & Zaragoza references this is based on.

BM25 score for a query Q = q1...qn against document D:

    score(D, Q) = sum_i  IDF(qi) * ( tf(qi, D) * (k1 + 1) )
                                   / ( tf(qi, D) + k1 * (1 - b + b * |D| / avgdl) )

A standard IDF variant (Robertson-Sparck Jones, +1-smoothed so it stays
non-negative even for terms occurring in more than half the corpus):

    IDF(qi) = ln( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )

where:
    N        = number of documents in the corpus
    df(qi)   = number of documents containing qi
    tf(qi,D) = term frequency of qi in D
    |D|      = length of D in tokens
    avgdl    = average document length across the corpus

k1 (typically 1.2-2.0) controls term-frequency saturation; b (in [0, 1])
controls document-length normalisation strength. Both must be exposed as
parameters, not hard-coded — you need to sweep them for your report
(assignment Section 8, "parameter search procedure for k1, b").
"""
import heapq
import math
from typing import Dict, List, Optional, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None
_IDF: Dict[str, float] = {}


def build(index: InvertedIndex) -> None:
    """Precompute per-term IDF values (Robertson-Sparck Jones, +1-smoothed)
    from the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""
    global _INDEX, _IDF
    _INDEX = index
    N = index.N
    _IDF = {}
    for term in index.postings:
        df = index.document_frequency(term)
        _IDF[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)


def raw_scores(query: str, k1: float = 1.2, b: float = 0.75, delta: float = 0.0) -> Dict[str, float]:
    """Every matched doc_id's BM25(+) score for `query`, unsorted.

    Factored out of score() so callers that need to combine this with
    another ranker (submission/custom_scorer.py's BM25+/VSM blend) can
    skip the sort-then-slice step entirely and do exactly one sort over
    the combined result, instead of one sort here that's immediately
    thrown away plus a second sort over the union. Profiling
    custom_scorer.score() on the full corpus showed this redundant sort
    was ~80ms/query of pure overhead — see scripts/tune_bm25_vsm_blend.py
    and report.tex's "Improving Query Latency" section.

    `delta` (default 0.0, i.e. plain BM25) switches this to BM25+ (Lv &
    Zhai, 2011, "Lower-Bounding Term Frequency Normalization"): adds a
    small constant to every matched term's contribution, so a term's
    presence in a long document is never driven arbitrarily close to
    zero purely by length normalization — standard BM25's saturation
    curve can otherwise under-reward long relevant documents relative to
    short ones containing the same term once. See
    scripts/tune_bm25_plus.py for the delta sweep."""
    if _INDEX is None:
        raise RuntimeError("bm25.build() must be called before bm25.raw_scores()/score().")

    avgdl = _INDEX.avg_doc_len or 1.0
    scores: Dict[str, float] = {}
    for term in tokenize(query):
        postings = _INDEX.postings.get(term)
        if not postings:
            continue
        idf = _IDF.get(term, 0.0)
        for doc_id, tf in postings.items():
            dl = _INDEX.doc_len.get(doc_id, 0)
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * ((tf * (k1 + 1)) / denom + delta)
    return scores


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75, delta: float = 0.0) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked,
    highest score first. See raw_scores() for the delta/BM25+ explanation.

    Uses heapq.nlargest rather than a full sort: profiling on the full
    corpus showed queries can match 37K-101K documents, but only the top
    k (typically 10) are ever needed. A full sort is O(n log n) over the
    whole candidate set; nlargest is O(n log k) — measured 5-6x faster at
    these candidate-set sizes for k=10, with byte-identical output (it's
    a stable top-k, same tie-break order as sorted()'s)."""
    scores = raw_scores(query, k1=k1, b=b, delta=delta)
    return heapq.nlargest(k, scores.items(), key=lambda item: item[1])
