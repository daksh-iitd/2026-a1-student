"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1): "supports conjunctive/
disjunctive Boolean queries and a cosine-similarity vector-space ranking
with a TF-IDF weighting scheme of your choice."

Two independent pieces to implement:

1. Boolean retrieval: given a query, treat it as an AND (conjunctive) or
   OR (disjunctive) combination of terms and return the matching document
   set — no ranking, just set membership. Useful as a fast candidate
   filter and as a sanity check ("does my index even find the right
   documents for this query?").

2. Vector-space ranking: represent the query and each candidate document
   as TF-IDF weighted vectors and rank by cosine similarity. A standard
   TF-IDF weight for term t in document d:

       w(t, d) = tf(t, d) * log( N / df(t) )

   (log base is your choice — just be consistent), and cosine similarity
   between query vector q and document vector d:

       sim(q, d) = (q . d) / (||q|| * ||d||)

Both pieces should read from the same InvertedIndex you build in
indexer.py.
"""
import math
from typing import Dict, List, Optional, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None
_IDF: Dict[str, float] = {}
_DOC_NORMS: Dict[str, float] = {}


def build(index: InvertedIndex) -> None:
    """Precompute per-term IDF and per-document TF-IDF vector norms from
    the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""
    global _INDEX, _IDF, _DOC_NORMS
    _INDEX = index
    N = index.N
    _IDF = {}
    for term in index.postings:
        df = index.document_frequency(term)
        _IDF[term] = math.log(N / df) if df else 0.0

    norms_sq: Dict[str, float] = {}
    for term, doc_tf in index.postings.items():
        idf = _IDF[term]
        for doc_id, tf in doc_tf.items():
            w = tf * idf
            norms_sq[doc_id] = norms_sq.get(doc_id, 0.0) + w * w
    _DOC_NORMS = {doc_id: math.sqrt(v) for doc_id, v in norms_sq.items()}


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Return the (unranked) list of doc_ids matching `query`, treating it
    as a conjunction (`mode="and"`) or disjunction (`mode="or"`) of its
    terms."""
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build() must be called before boolean_search().")
    if mode not in ("and", "or"):
        raise ValueError(f"Unknown mode: {mode!r}, expected 'and' or 'or'")

    terms = set(tokenize(query))
    if not terms:
        return []

    doc_sets = [set(_INDEX.postings.get(term, {}).keys()) for term in terms]
    if mode == "and":
        result = doc_sets[0]
        for s in doc_sets[1:]:
            result = result & s
    else:
        result = set()
        for s in doc_sets:
            result |= s
    return list(result)


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by
    TF-IDF cosine similarity, highest score first."""
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build() must be called before vsm_score().")

    q_tf: Dict[str, int] = {}
    for term in tokenize(query):
        q_tf[term] = q_tf.get(term, 0) + 1
    q_weights = {term: tf * _IDF.get(term, 0.0) for term, tf in q_tf.items()}
    q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
    if q_norm == 0:
        return []

    dot: Dict[str, float] = {}
    for term, qw in q_weights.items():
        if qw == 0:
            continue
        postings = _INDEX.postings.get(term)
        if not postings:
            continue
        idf = _IDF[term]
        for doc_id, tf in postings.items():
            dot[doc_id] = dot.get(doc_id, 0.0) + qw * (tf * idf)

    scored: List[Tuple[str, float]] = []
    for doc_id, d in dot.items():
        dnorm = _DOC_NORMS.get(doc_id, 0.0)
        if dnorm == 0:
            continue
        scored.append((doc_id, d / (dnorm * q_norm)))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
