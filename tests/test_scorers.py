"""
tests/test_scorers.py — self-authored correctness tests for the required
components (assignment Section 4.1 / Section 7 grading rubric: "Boolean/
VSM and BM25 retrievers are both correctly implemented and independently
verifiable... graded by unit tests against known small examples, not
leaderboard score alone").

Every expected value below is computed independently from the formula
each module's own docstring states — via explicit arithmetic in this
file, not by calling the function under test — the same style
tests/test_metrics.py already uses (e.g. `expected_idcg = 3.0 +
1/math.log2(3) + 0.0`). A shared 3-document toy corpus is reused across
BM25/Boolean/VSM so the same hand-checkable numbers back every test.

Toy corpus (all three terms deliberately have identical document
frequency, df=2, so IDF is uniform across terms in this fixture --
isolates term-frequency and length-normalization behavior for the hand
check without also needing different IDF values to track):
    d1: "cat dog cat"          (cat:2, dog:1          | len=3)
    d2: "dog bird"              (dog:1, bird:1         | len=2)
    d3: "cat bird bird bird"    (cat:1, bird:3          | len=4)
N=3, avgdl=3.0. df(cat)=2 {d1,d3}, df(dog)=2 {d1,d2}, df(bird)=2 {d2,d3}.
"""
import math

from submission import bm25, boolean_vsm, custom_scorer
from submission.indexer import InvertedIndex

TOY_CORPUS = [
    ("d1", "cat dog cat"),
    ("d2", "dog bird"),
    ("d3", "cat bird bird bird"),
]


def _build_index():
    index = InvertedIndex()
    index.build(TOY_CORPUS)
    return index


# ---------------------------------------------------------------------------
# BM25 (submission/bm25.py)
# ---------------------------------------------------------------------------

def test_bm25_matches_hand_computed_scores_for_single_term_query():
    index = _build_index()
    bm25.build(index)

    k1, b = 1.2, 0.75
    N, avgdl = 3, 3.0
    df_cat = 2  # {d1, d3}
    idf_cat = math.log((N - df_cat + 0.5) / (df_cat + 0.5) + 1)  # = ln(1.6)

    # d1: tf(cat)=2, dl=3 -> denom = 2 + 1.2*(1-0.75+0.75*3/3) = 2 + 1.2*1.0 = 3.2
    expected_d1 = idf_cat * (2 * (k1 + 1)) / (2 + k1 * (1 - b + b * 3 / avgdl))
    # d3: tf(cat)=1, dl=4 -> denom = 1 + 1.2*(1-0.75+0.75*4/3) = 1 + 1.2*1.25 = 2.5
    expected_d3 = idf_cat * (1 * (k1 + 1)) / (1 + k1 * (1 - b + b * 4 / avgdl))

    results = bm25.score("cat", k=10, k1=k1, b=b)
    scores = dict(results)

    assert "d2" not in scores, "d2 doesn't contain 'cat' and must not be scored"
    assert math.isclose(scores["d1"], expected_d1, rel_tol=1e-9), f"expected {expected_d1}, got {scores['d1']}"
    assert math.isclose(scores["d3"], expected_d3, rel_tol=1e-9), f"expected {expected_d3}, got {scores['d3']}"
    # d1 has higher tf and a shorter-than-average document -> must outrank d3.
    assert [doc_id for doc_id, _score in results] == ["d1", "d3"]


def test_bm25_multi_term_query_sums_per_term_contributions():
    index = _build_index()
    bm25.build(index)

    k1, b = 1.2, 0.75
    N, avgdl = 3, 3.0
    idf = math.log((N - 2 + 0.5) / (2 + 0.5) + 1)  # every term here has df=2

    def contrib(tf, dl):
        return idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

    # d1 (dl=3): cat tf=2, dog tf=1 -> both terms contribute
    expected_d1 = contrib(2, 3) + contrib(1, 3)
    # d2 (dl=2): dog tf=1 only ("cat" absent)
    expected_d2 = contrib(1, 2)

    scores = dict(bm25.score("cat dog", k=10, k1=k1, b=b))
    assert math.isclose(scores["d1"], expected_d1, rel_tol=1e-9)
    assert math.isclose(scores["d2"], expected_d2, rel_tol=1e-9)


def test_bm25_unknown_term_contributes_nothing():
    index = _build_index()
    bm25.build(index)
    # "cat" alone vs. "cat xyzzy" (xyzzy is out-of-vocabulary) must score identically.
    with_unknown = dict(bm25.score("cat xyzzy", k=10))
    without_unknown = dict(bm25.score("cat", k=10))
    assert with_unknown == without_unknown


def test_bm25_empty_query_returns_no_results():
    index = _build_index()
    bm25.build(index)
    assert bm25.score("", k=10) == []


def test_bm25_delta_zero_is_plain_bm25():
    # BM25+ (Lv & Zhai) with delta=0 must reduce exactly to plain BM25 --
    # this is the module's own documented default behavior.
    index = _build_index()
    bm25.build(index)
    plain = bm25.score("cat dog", k=10, k1=1.5, b=0.6)
    explicit_zero_delta = bm25.score("cat dog", k=10, k1=1.5, b=0.6, delta=0.0)
    assert plain == explicit_zero_delta


def test_bm25_delta_strictly_increases_scores_for_matched_documents():
    # BM25+'s delta adds a strictly positive constant (idf * delta) to
    # every matched term's contribution, so any delta > 0 must raise
    # every matched document's score relative to delta=0, never lower it.
    index = _build_index()
    bm25.build(index)
    base = dict(bm25.score("cat dog", k=10, k1=1.2, b=0.75, delta=0.0))
    boosted = dict(bm25.score("cat dog", k=10, k1=1.2, b=0.75, delta=1.0))
    assert set(base) == set(boosted)
    for doc_id in base:
        assert boosted[doc_id] > base[doc_id]


# ---------------------------------------------------------------------------
# Boolean AND/OR (submission/boolean_vsm.py)
# ---------------------------------------------------------------------------

def test_boolean_and_requires_every_term_present():
    index = _build_index()
    boolean_vsm.build(index)
    # Only d1 contains both "cat" and "dog".
    assert set(boolean_vsm.boolean_search("cat dog", mode="and")) == {"d1"}


def test_boolean_or_requires_any_term_present():
    index = _build_index()
    boolean_vsm.build(index)
    # d1 (cat, dog), d2 (dog), d3 (cat) all match at least one term.
    assert set(boolean_vsm.boolean_search("cat dog", mode="or")) == {"d1", "d2", "d3"}


def test_boolean_and_no_match_returns_empty():
    index = _build_index()
    boolean_vsm.build(index)
    # No single document contains all of cat, dog, AND bird together.
    assert boolean_vsm.boolean_search("cat dog bird", mode="and") == []


def test_boolean_search_rejects_unknown_mode():
    index = _build_index()
    boolean_vsm.build(index)
    try:
        boolean_vsm.boolean_search("cat", mode="xor")
        assert False, "expected ValueError for an unsupported mode"
    except ValueError:
        pass


def test_boolean_search_empty_query_returns_empty():
    index = _build_index()
    boolean_vsm.build(index)
    assert boolean_vsm.boolean_search("", mode="and") == []
    assert boolean_vsm.boolean_search("", mode="or") == []


# ---------------------------------------------------------------------------
# VSM / TF-IDF cosine similarity (submission/boolean_vsm.py)
# ---------------------------------------------------------------------------

def test_vsm_matches_hand_computed_cosine_similarity():
    index = _build_index()
    boolean_vsm.build(index)

    N = 3
    idf = math.log(N / 2)  # every term here has df=2; boolean_vsm's IDF is plain log(N/df), no +0.5 smoothing

    # Document TF-IDF vectors, by hand from the corpus definition.
    d1 = {"cat": 2 * idf, "dog": 1 * idf}
    d2 = {"dog": 1 * idf, "bird": 1 * idf}
    d3 = {"cat": 1 * idf, "bird": 3 * idf}
    q = {"cat": 1 * idf, "dog": 1 * idf}

    def norm(vec):
        return math.sqrt(sum(w * w for w in vec.values()))

    def cosine(a, b):
        dot = sum(a.get(t, 0.0) * w for t, w in b.items())
        return dot / (norm(a) * norm(b))

    expected_d1 = cosine(d1, q)
    expected_d2 = cosine(d2, q)  # only "dog" overlaps
    expected_d3 = cosine(d3, q)  # only "cat" overlaps

    results = dict(boolean_vsm.vsm_score("cat dog", k=10))
    assert math.isclose(results["d1"], expected_d1, rel_tol=1e-9)
    assert math.isclose(results["d2"], expected_d2, rel_tol=1e-9)
    assert math.isclose(results["d3"], expected_d3, rel_tol=1e-9)
    # d1 shares both query terms and has the highest cat-weight -> must rank first.
    ranked_ids = [doc_id for doc_id, _score in boolean_vsm.vsm_score("cat dog", k=10)]
    assert ranked_ids[0] == "d1"


def test_vsm_cosine_scores_are_bounded_by_one():
    # Cosine similarity is mathematically bounded in [-1, 1], and every
    # weight here is non-negative (TF and IDF both >= 0), so scores must
    # actually fall in [0, 1].
    index = _build_index()
    boolean_vsm.build(index)
    for _doc_id, score in boolean_vsm.vsm_score("cat dog bird", k=10):
        assert -1e-9 <= score <= 1.0 + 1e-9


def test_vsm_empty_query_returns_no_results():
    index = _build_index()
    boolean_vsm.build(index)
    assert boolean_vsm.vsm_score("", k=10) == []


def test_vsm_unknown_term_only_query_returns_no_results():
    index = _build_index()
    boolean_vsm.build(index)
    # "xyzzy" has zero document frequency -> IDF defaults to 0.0 -> the
    # query vector's norm is 0 -> no results, not a crash.
    assert boolean_vsm.vsm_score("xyzzy", k=10) == []


def test_vsm_raw_scores_for_candidates_matches_raw_scores_restricted_to_the_same_set():
    # raw_scores_for_candidates() takes a different code path (candidates
    # outer loop, terms inner loop) specifically for speed on a small
    # candidate pool (submission/custom_scorer.py's vsm_pool_size) -- it
    # must still compute the exact same cosine similarity as raw_scores()
    # for any candidate both would have scored anyway.
    index = _build_index()
    boolean_vsm.build(index)
    full = boolean_vsm.raw_scores("cat dog bird")
    restricted = boolean_vsm.raw_scores_for_candidates("cat dog bird", ["d1", "d3"])
    assert set(restricted) <= {"d1", "d3"}
    for doc_id in restricted:
        assert math.isclose(restricted[doc_id], full[doc_id], rel_tol=1e-9)


def test_vsm_raw_scores_for_candidates_excludes_documents_outside_the_pool():
    # d2 matches "dog" and would appear in the unrestricted raw_scores(),
    # but is deliberately left out of the candidate pool here -- it must
    # not appear in the restricted result even though it's a real match.
    index = _build_index()
    boolean_vsm.build(index)
    full = boolean_vsm.raw_scores("dog")
    assert "d2" in full  # sanity check on the fixture itself
    restricted = boolean_vsm.raw_scores_for_candidates("dog", ["d1"])
    assert "d2" not in restricted


def test_vsm_raw_scores_for_candidates_empty_pool_or_query():
    index = _build_index()
    boolean_vsm.build(index)
    assert boolean_vsm.raw_scores_for_candidates("cat dog", []) == {}
    assert boolean_vsm.raw_scores_for_candidates("xyzzy", ["d1", "d2", "d3"]) == {}


# ---------------------------------------------------------------------------
# Cross-cutting: determinism and result well-formedness (mirrors the
# harness's own conformance checks, applied directly to each scorer
# rather than through the full build_index/load_index/retrieve path).
# ---------------------------------------------------------------------------

def test_bm25_and_vsm_are_deterministic():
    index = _build_index()
    bm25.build(index)
    boolean_vsm.build(index)
    assert bm25.score("cat dog bird", k=10) == bm25.score("cat dog bird", k=10)
    assert boolean_vsm.vsm_score("cat dog bird", k=10) == boolean_vsm.vsm_score("cat dog bird", k=10)


def test_bm25_and_vsm_never_return_more_than_k_or_duplicate_doc_ids():
    index = _build_index()
    bm25.build(index)
    boolean_vsm.build(index)
    for k in (0, 1, 2, 10):
        bm25_results = bm25.score("cat dog bird", k=k)
        vsm_results = boolean_vsm.vsm_score("cat dog bird", k=k)
        assert len(bm25_results) <= k
        assert len(vsm_results) <= k
        assert len({doc_id for doc_id, _s in bm25_results}) == len(bm25_results)
        assert len({doc_id for doc_id, _s in vsm_results}) == len(vsm_results)


def test_bm25_and_vsm_results_are_sorted_descending():
    index = _build_index()
    bm25.build(index)
    boolean_vsm.build(index)
    bm25_scores = [s for _d, s in bm25.score("cat dog bird", k=10)]
    vsm_scores = [s for _d, s in boolean_vsm.vsm_score("cat dog bird", k=10)]
    assert bm25_scores == sorted(bm25_scores, reverse=True)
    assert vsm_scores == sorted(vsm_scores, reverse=True)


# ---------------------------------------------------------------------------
# submission/custom_scorer.py — this is what retrieve() actually calls.
# Validates the min-max-normalize-then-blend wiring is correct, using
# bm25.raw_scores()/boolean_vsm.raw_scores() (already verified above
# against hand-computed values) as trusted ground truth for what's being
# combined -- a layered check: lower-level formulas verified by hand,
# higher-level composition verified against the now-trusted lower layer.
# ---------------------------------------------------------------------------

def _minmax(scores):
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def test_custom_scorer_blend_matches_manual_recombination():
    index = _build_index()
    custom_scorer.build(index)

    k1, b, delta, w = 1.2, 0.75, 0.0, 0.6
    bm25_raw = bm25.raw_scores("cat dog", k1=k1, b=b, delta=delta)
    vsm_raw = boolean_vsm.raw_scores("cat dog")
    bm25_norm = _minmax(bm25_raw)
    vsm_norm = _minmax(vsm_raw)
    candidates = set(bm25_norm) | set(vsm_norm)
    expected = {
        doc_id: w * bm25_norm.get(doc_id, 0.0) + (1 - w) * vsm_norm.get(doc_id, 0.0) for doc_id in candidates
    }

    results = dict(custom_scorer.score("cat dog", k=10, k1=k1, b=b, delta=delta, w=w))
    assert set(results) == set(expected)
    for doc_id in expected:
        assert math.isclose(results[doc_id], expected[doc_id], rel_tol=1e-9)


def test_custom_scorer_gamma_zero_ignores_coordination_bonus():
    # gamma=0.0 is custom_scorer.score()'s default specifically because
    # scripts/tune_coordination.py found every nonzero gamma hurts
    # nDCG@10 on the dev set (report.tex) -- this pins that the default
    # actually reproduces the plain blend, not just documents the intent.
    index = _build_index()
    custom_scorer.build(index)
    default_gamma = custom_scorer.score("cat dog", k=10, k1=1.2, b=0.75)
    explicit_zero_gamma = custom_scorer.score("cat dog", k=10, k1=1.2, b=0.75, gamma=0.0)
    assert default_gamma == explicit_zero_gamma


def test_custom_scorer_vsm_pool_size_restricts_vsm_to_bm25s_top_candidates():
    # vsm_pool_size=1 should only let BM25's single top-scoring candidate
    # keep a real VSM contribution; every other matched document must be
    # blended as if VSM found nothing for it (vsm_norm defaults to 0.0),
    # even though boolean_vsm.raw_scores() unrestricted would have scored
    # it. This pins that the pool restriction actually changes behavior
    # -- not just that it exists and happens to be a no-op.
    index = _build_index()
    custom_scorer.build(index)

    k1, b, delta, w = 1.2, 0.75, 0.0, 0.5
    bm25_raw = bm25.raw_scores("cat dog", k1=k1, b=b, delta=delta)
    top_doc_id = max(bm25_raw, key=bm25_raw.get)

    restricted = dict(custom_scorer.score("cat dog", k=10, k1=k1, b=b, delta=delta, w=w, vsm_pool_size=1))
    unrestricted = dict(
        custom_scorer.score("cat dog", k=10, k1=k1, b=b, delta=delta, w=w, vsm_pool_size=1000)
    )

    assert set(restricted) == set(unrestricted) == set(bm25_raw)
    # The pool winner's score is unaffected by the restriction...
    assert math.isclose(restricted[top_doc_id], unrestricted[top_doc_id], rel_tol=1e-9)
    # ...but at least one other matched document must differ, since its
    # real VSM contribution was dropped to 0.0 by the pool restriction.
    others = set(bm25_raw) - {top_doc_id}
    assert others, "fixture must have more than one BM25 candidate for this test to be meaningful"
    assert any(not math.isclose(restricted[d], unrestricted[d], rel_tol=1e-9) for d in others)


def test_custom_scorer_falls_back_to_bm25_only_when_vsm_finds_nothing():
    # An out-of-vocabulary-only query: bm25_raw and vsm_raw are both
    # empty, so custom_scorer must return [] rather than raising (e.g.
    # from a division by zero in a norm computation).
    index = _build_index()
    custom_scorer.build(index)
    assert custom_scorer.score("xyzzy", k=10) == []


def test_custom_scorer_is_deterministic_and_well_formed():
    index = _build_index()
    custom_scorer.build(index)
    for k in (0, 1, 2, 10):
        results = custom_scorer.score("cat dog bird", k=k)
        assert custom_scorer.score("cat dog bird", k=k) == results
        assert len(results) <= k
        assert len({doc_id for doc_id, _s in results}) == len(results)
        scores = [s for _d, s in results]
        assert scores == sorted(scores, reverse=True)
