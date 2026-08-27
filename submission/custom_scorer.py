"""
submission/custom_scorer.py — static BM25+ / VSM blend.

Not required, but this is explicitly called out in the assignment
(Section 4.1) as "where separation in the leaderboard tends to happen":
any linear or non-linear combination of your Boolean/VSM and BM25
signals, additional features (e.g. proximity/bigram overlap), or your
own heuristic.

This is the scorer actually wired into retrieve() (see
submission/retrieve.py). Two independent signals, blended:

    score(d) = w * BM25plus_norm(d) + (1-w) * VSM_norm(d)

BM25+ (Lv & Zhai, 2011) is plain BM25 (submission/bm25.py) with a small
constant `delta` added to every matched term's contribution, correcting
for BM25's tendency to under-reward long relevant documents purely from
length normalization. VSM is TF-IDF/cosine similarity
(submission/boolean_vsm.py). The two are on incomparable scales (BM25 is
unbounded, VSM cosine is in [0,1]), so each is min-max normalized to
[0,1] per query before blending.

This is a STATIC blend of two independently-scored rankings — not the
Rocchio pseudo-relevance-feedback approach tried earlier and rejected
(see report.tex, "Custom Scorer Exploration"): PRF re-scores against an
*expanded query* built from an initial retrieval pass, which is what let
topic drift creep in on this corpus. Blending two rankers that each score
the *original* query independently doesn't share that failure mode, and
scripts/tune_bm25_vsm_blend.py + a follow-up delta/w grid search (see
report.tex) confirmed a real, compounding gain from both components:
nDCG@10 rose from 0.667 (plain BM25) to ~0.690 on the dev set at
delta=0.75, w=0.8 — comparable in size to the original k1/b tuning gain.

Cost: this runs a full VSM pass per query in addition to BM25's, so mean
query latency is roughly double a bare bm25.score() call.
"""
from typing import Dict, List, Optional, Tuple

from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex

_BUILT = False


def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(). Ensures both underlying scorers'
    caches (bm25._IDF, boolean_vsm._IDF/_DOC_NORMS) are populated,
    regardless of whether retrieve.load_index() also calls
    bm25.build()/boolean_vsm.build() directly — calling build() twice on
    the same index is idempotent, so this stays correct either way."""
    global _BUILT
    bm25.build(index)
    boolean_vsm.build(index)
    _BUILT = True


def _minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {doc_id: 1.0 for doc_id in scores}
    return {doc_id: (v - lo) / (hi - lo) for doc_id, v in scores.items()}


def score(
    query: str,
    k: int,
    k1: float = 2.5,
    b: float = 0.6,
    delta: float = 0.75,
    w: float = 0.8,
) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs, ranked by the BM25+/VSM
    blend described in the module docstring. k1/b/delta/w default to the
    values found by the dev-set grid search (report.tex)."""
    if not _BUILT:
        raise RuntimeError("custom_scorer.build() must be called before custom_scorer.score().")

    # bm25.raw_scores()/boolean_vsm.raw_scores() return every matched
    # doc_id's score, UNSORTED — score()/vsm_score() would sort the same
    # dict and then have that sort thrown away here anyway, since
    # normalization needs the whole set (not just a top-k) and the final
    # ranking is only decided by the one sort below, over the combined
    # scores. Calling the *_score() wrappers instead cost ~80ms/query of
    # pure wasted sorting on the full corpus — see report.tex's
    # "Improving Query Latency" section.
    bm25_raw = bm25.raw_scores(query, k1=k1, b=b, delta=delta)
    vsm_raw = boolean_vsm.raw_scores(query)

    bm25_norm = _minmax_normalize(bm25_raw)
    vsm_norm = _minmax_normalize(vsm_raw)

    candidates = set(bm25_norm) | set(vsm_norm)
    combined = [
        (doc_id, w * bm25_norm.get(doc_id, 0.0) + (1 - w) * vsm_norm.get(doc_id, 0.0))
        for doc_id in candidates
    ]
    combined.sort(key=lambda item: item[1], reverse=True)
    return combined[:k]
