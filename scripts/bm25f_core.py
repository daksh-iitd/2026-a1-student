"""
scripts/bm25f_core.py — shared BM25F (Robertson, Zaragoza & Taylor, 2004)
implementation for the field-weighting experiments in
scripts/tune_bm25f_true_fields.py and scripts/tune_bm25f_prefix.py.

Research-only: not part of the graded submission — see report.tex's
"Field-Weighting Experiment" section for why a persisted field split
isn't safely deployable against the actual {doc_id, text}-only corpus
contract (data/README.md). This module exists to test whether a
properly-designed single-saturation BM25F beats the earlier naive "two
separate BM25 passes, linearly blended" approach (which double-saturates
a term that appears in both fields) on the SAME title/abstract data,
before deciding whether the formula improvement is worth persisting for
real — and to test the graceful-degradation fallback for corpora with no
title field at all (e.g. fiqa).

One shared vocabulary, one postings dict per term — no vocabulary
duplication the way a naive "title_word"/"body_word" scheme would cause:
term -> {doc_id: (tf_title, tf_body)}.

Formula (single saturation, matching the BM25F literature):
    tf~_t(d) = sum_f  w_f * tf_{t,f}(d) / (1 + b_f * (len_f(d)/avglen_f - 1))
    score(d, Q) = sum_{t in Q}  idf_t * tf~_t(d) * (k1+1) / (k1 + tf~_t(d))

Single saturation is the point: a term appearing once in the title and
once in the body gets ONE dose of the steep part of the curve (via the
combined tf~), not two separate saturated contributions summed —
unlike scripts/tune_field_weights.py's earlier approach, which ran two
independent BM25 passes (each already saturated) and linearly blended
the results.

idf_t is global, computed over documents containing t in ANY field (not
per-field), using the same +1-smoothed Robertson-Sparck-Jones form as
submission/bm25.py, for consistency and to guarantee non-negativity.
"""
import math
from typing import Dict, List, Tuple

from submission.indexer import tokenize


class FieldIndex:
    def __init__(self):
        self.postings: Dict[str, Dict[str, Tuple[int, int]]] = {}  # term -> {doc_id: (tf_title, tf_body)}
        self.len_title: Dict[str, int] = {}
        self.len_body: Dict[str, int] = {}
        self.N: int = 0
        self.avg_len_title: float = 0.0
        self.avg_len_body: float = 0.0
        self.idf: Dict[str, float] = {}

    def build(self, corpus: List[Tuple[str, str, str]]) -> None:
        """corpus: list of (doc_id, title_text, body_text). title_text
        may be "" for a title-less document — len_title becomes 0 and,
        per the scoring function below, that document's title field
        contributes nothing to its score (graceful degradation, tested
        explicitly against fiqa in scripts/tune_bm25f_prefix.py)."""
        self.postings = {}
        self.len_title = {}
        self.len_body = {}
        total_title, total_body = 0, 0
        for doc_id, title_text, body_text in corpus:
            title_tokens = tokenize(title_text) if title_text else []
            body_tokens = tokenize(body_text) if body_text else []
            self.len_title[doc_id] = len(title_tokens)
            self.len_body[doc_id] = len(body_tokens)
            total_title += len(title_tokens)
            total_body += len(body_tokens)

            title_tf: Dict[str, int] = {}
            for tok in title_tokens:
                title_tf[tok] = title_tf.get(tok, 0) + 1
            body_tf: Dict[str, int] = {}
            for tok in body_tokens:
                body_tf[tok] = body_tf.get(tok, 0) + 1

            for term in set(title_tf) | set(body_tf):
                self.postings.setdefault(term, {})[doc_id] = (title_tf.get(term, 0), body_tf.get(term, 0))

        self.N = len(corpus)
        self.avg_len_title = (total_title / self.N) if self.N else 0.0
        self.avg_len_body = (total_body / self.N) if self.N else 0.0

        self.idf = {}
        for term, doc_tfs in self.postings.items():
            df = len(doc_tfs)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)


def bm25f_raw_scores(
    query: str,
    index: FieldIndex,
    k1: float = 1.2,
    w_title: float = 1.0,
    b_title: float = 0.2,
    b_body: float = 0.4,
) -> Dict[str, float]:
    """w_body is fixed at 1.0 (only the ratio to w_title matters — a
    separate free w_body would be a redundant degree of freedom, per
    BM25F's own design). Returns every matched doc_id's score, unsorted.

    Division-by-zero safety: `len_title/avg_len_title` is only ever
    evaluated when tf_title > 0 for this specific document, which can
    only happen if avg_len_title > 0 too (a document can't have nonzero
    title tokens if the collection-wide average is exactly zero) — so
    the `or 1.0` fallback below is defensive redundancy, not the actual
    mechanism that prevents the crash; it's cheap insurance kept for
    clarity. Verified directly against fiqa (genuinely title-less: every
    title_text is "") in scripts/tune_bm25f_prefix.py."""
    w_body = 1.0
    avg_title = index.avg_len_title or 1.0
    avg_body = index.avg_len_body or 1.0

    scores: Dict[str, float] = {}
    for term in tokenize(query):
        postings = index.postings.get(term)
        if not postings:
            continue
        idf = index.idf.get(term, 0.0)
        for doc_id, (tf_title, tf_body) in postings.items():
            norm_title = 0.0
            if tf_title:
                len_title = index.len_title.get(doc_id, 0)
                norm_title = w_title * tf_title / (1 + b_title * (len_title / avg_title - 1))
            norm_body = 0.0
            if tf_body:
                len_body = index.len_body.get(doc_id, 0)
                norm_body = w_body * tf_body / (1 + b_body * (len_body / avg_body - 1))

            tf_tilde = norm_title + norm_body
            if tf_tilde <= 0:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_tilde * (k1 + 1) / (k1 + tf_tilde)
    return scores
