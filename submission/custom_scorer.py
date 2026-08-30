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
import heapq
from typing import Dict, List, Optional, Tuple

from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex, tokenize

_BUILT = False
_INDEX: Optional[InvertedIndex] = None


def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(). Ensures both underlying scorers'
    caches (bm25._IDF, boolean_vsm._IDF/_DOC_NORMS) are populated,
    regardless of whether retrieve.load_index() also calls
    bm25.build()/boolean_vsm.build() directly — calling build() twice on
    the same index is idempotent, so this stays correct either way. Also
    keeps its own reference to `index`, for the coordination-level bonus
    (see _coordination_levels()), which needs raw postings directly
    rather than either sub-scorer's derived (IDF-weighted) state."""
    global _BUILT, _INDEX
    bm25.build(index)
    boolean_vsm.build(index)
    _INDEX = index
    _BUILT = True


def _coordination_levels(query: str) -> Dict[str, float]:
    """For every doc_id matching at least one query term, the fraction of
    *distinct* query terms it contains (Salton's "coordination level
    match"): 1.0 means every query term is present, 0.5 means half are.
    A cheap, position-free proxy for "does this document actually
    address every part of the query," motivated by the error analysis in
    report.tex — BM25/VSM alone treat query terms independently and give
    no credit (or penalty) for how many of them a document actually
    covers together."""
    terms = set(tokenize(query))
    if not terms or _INDEX is None:
        return {}
    counts: Dict[str, int] = {}
    for term in terms:
        for doc_id in _INDEX.postings.get(term, {}):
            counts[doc_id] = counts.get(doc_id, 0) + 1
    n_terms = len(terms)
    return {doc_id: c / n_terms for doc_id, c in counts.items()}


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
    gamma: float = 0.0,
    vsm_pool_size: int = 5000,
) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs, ranked by the BM25+/VSM
    blend described in the module docstring, plus `gamma` times the
    coordination-level bonus (_coordination_levels()) added on top.
    `gamma=0.0` (default) reproduces the plain BM25+/VSM blend exactly —
    see scripts/tune_coordination.py for the dev-set sweep that decides
    whether a nonzero gamma is actually worth shipping.

    `vsm_pool_size` restricts the VSM pass to the top `vsm_pool_size`
    BM25+ candidates (boolean_vsm.raw_scores_for_candidates()) instead of
    rescoring every document any query term matches
    (boolean_vsm.raw_scores()) a second time — a document outside the
    pool is scored as VSM=0, same as one VSM found no term overlap with
    at all. Swept on the full dev set directly against nDCG@10 (not just
    assumed safe): pool=1000 cost ~0.003-0.004 nDCG@10 versus
    unrestricted; pool=5000 matched or slightly *exceeded* the
    unrestricted score (0.692 vs 0.691, both runs, small differences
    from the pool's candidate-set-dependent min-max normalization
    changing per query, not a bug) while cutting mean query latency from
    159.5ms to 65.9ms in the same sweep (~2.4x) — see report.tex's
    efficiency section for the full pool-size table."""
    if not _BUILT:
        raise RuntimeError("custom_scorer.build() must be called before custom_scorer.score().")

    # bm25.raw_scores()/boolean_vsm.raw_scores() return every matched
    # doc_id's score, UNSORTED — score()/vsm_score() would sort the same
    # dict and then have that sort thrown away here anyway, since
    # normalization needs the whole set (not just a top-k) and the final
    # ranking is only decided by the one heapq.nlargest below, over the
    # combined scores. Calling the *_score() wrappers instead cost
    # ~80ms/query of pure wasted sorting on the full corpus — see
    # report.tex's "Improving Query Latency" section.
    bm25_raw = bm25.raw_scores(query, k1=k1, b=b, delta=delta)
    vsm_candidate_ids = [
        doc_id for doc_id, _ in heapq.nlargest(vsm_pool_size, bm25_raw.items(), key=lambda item: item[1])
    ]
    vsm_raw = boolean_vsm.raw_scores_for_candidates(query, vsm_candidate_ids)

    bm25_norm = _minmax_normalize(bm25_raw)
    vsm_norm = _minmax_normalize(vsm_raw)
    coordination = _coordination_levels(query) if gamma else {}

    candidates = set(bm25_norm) | set(vsm_norm)
    combined = (
        (
            doc_id,
            w * bm25_norm.get(doc_id, 0.0)
            + (1 - w) * vsm_norm.get(doc_id, 0.0)
            + gamma * coordination.get(doc_id, 0.0),
        )
        for doc_id in candidates
    )
    return heapq.nlargest(k, combined, key=lambda item: item[1])
