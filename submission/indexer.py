"""
submission/indexer.py — build your inverted index here.

This is one of the required components (assignment Section 4.1): you must
build the inverted index yourself, without an existing search/indexing
library (Lucene, Elasticsearch, Pyserini, Whoosh, etc.).

A `tokenize()` helper is provided below purely so that tokenization is
consistent across your Boolean/VSM and BM25 scorers —
feel free to replace it (e.g. add stemming or stopword removal), just make
sure every scorer that reads this index was built with the same tokenizer.

Everything else — the postings representation, what per-document and
collection statistics you track, whether you add positions for
proximity/phrase features — is your design decision. `InvertedIndex`
below sketches a minimal, obviously-sufficient shape; you do not have to
use it, but if you do, filling in `build()` and `document_frequency()` is
enough to support Boolean/VSM and BM25.

Persistence (assignment Section 4.1 / Section 7 "index size" scoring):
`build_index()` in retrieve.py runs in one process and `load_index()` runs
in a separate, later one — so whatever this index needs at query time must
round-trip through `save()`/`load()` below, not just live as Python
attributes. The on-disk byte size of what `save()` writes is graded
directly (smaller, relative to the class median, scores better), so a
compact postings encoding is worth more here than in most course
assignments — see the `save()` docstring for concrete starting points.
"""
import json
import os
import re
from typing import Dict, List, Tuple

from submission import vbyte
from submission.stemmer import normalize_tokens

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric tokenization, followed by English stopword
    removal and Porter stemming (submission/stemmer.py) so that e.g.
    "running"/"runs"/"runner" conflate to a shared term and common
    function words ("the", "of", ...) don't pollute postings lists or
    dilute TF-IDF/BM25 scores. Every scorer (BM25, Boolean/VSM) calls this
    same function, so they all see an identical token stream."""
    return normalize_tokens(_TOKEN_RE.findall(text.lower()))


class InvertedIndex:
    """A minimal inverted index skeleton. Extend the data structures here
    however your design needs (e.g. term positions for phrase/proximity
    scoring, a more compact postings representation for the efficiency
    bonus) — this is a starting point, not a fixed schema.
    """

    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: term_freq}
        self.doc_len: Dict[str, int] = {}  # doc_id -> number of tokens
        self.doc_text: Dict[str, str] = {}  # doc_id -> raw text (handy for VSM/debugging)
        self.N: int = 0  # number of documents
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """corpus: list of (doc_id, text) pairs, e.g. from
        submission.corpus_utils.load_corpus().

        Tokenizes each document, populates self.postings, self.doc_len,
        self.N, and self.avg_doc_len. Raw document text is intentionally
        not retained (self.doc_text stays empty) — BM25/VSM only need
        term frequencies and length statistics, and keeping raw text
        around would bloat the persisted index for no scoring benefit.
        """
        self.postings = {}
        self.doc_len = {}
        total_len = 0
        for doc_id, text in corpus:
            tokens = tokenize(text)
            self.doc_len[doc_id] = len(tokens)
            total_len += len(tokens)
            tf_counts: Dict[str, int] = {}
            for tok in tokens:
                tf_counts[tok] = tf_counts.get(tok, 0) + 1
            for term, tf in tf_counts.items():
                self.postings.setdefault(term, {})[doc_id] = tf
        self.N = len(corpus)
        self.avg_doc_len = (total_len / self.N) if self.N else 0.0

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once."""
        return len(self.postings.get(term, {}))

    def save(self, index_dir: str) -> None:
        """Persist everything document_frequency() / your scorers need to
        `index_dir`, so `load()` can reconstruct this object in a fresh
        process with no memory of `build()` ever having run. Called from
        retrieve.build_index().

        The on-disk byte size of whatever you write here is graded
        directly (assignment Section 7, "index size", relative to the
        class median). This implementation takes the full path suggested
        by the docstring this replaced: doc_ids are assigned small integer
        ids (stored once), each postings list is sorted by integer doc id
        and gap-encoded (store the delta from the previous doc id, not the
        absolute id), and — the step that used to be missing — every
        integer (gaps, term frequencies, doc lengths, per-term postings
        byte-lengths) is VByte-encoded (submission/vbyte.py) into raw
        binary instead of a JSON array of ASCII digits. A JSON int costs
        1 separator byte plus 1 ASCII byte per digit; small VByte ints
        (the common case for gaps and term frequencies) cost exactly 1
        byte. Five files, each single-purpose so nothing forces text and
        binary data to share one JSON blob:
          - meta.json:          N, avg_doc_len, n_docs, n_terms (tiny; the
                                 only real JSON left, because there's
                                 nothing bulk here to save bytes on)
          - doc_ids.txt:        doc_ids, newline-joined, in ascending
                                 int-id order
          - doc_len.bin:        VByte(doc_len[i]) for i in doc_ids order
          - vocab_terms.txt:    postings' terms, newline-joined, sorted
          - vocab_lengths.bin:  VByte(byte length of each term's slice in
                                 postings.bin), same order as
                                 vocab_terms.txt — a term's *offset* is
                                 never stored: since every slice is
                                 written back-to-back with no gaps,
                                 load() reconstructs offsets as a running
                                 sum of the preceding lengths
          - postings.bin:       each term's (gap, tf) pairs, VByte-encoded
                                 and concatenated in vocab_terms.txt order
        """
        os.makedirs(index_dir, exist_ok=True)

        doc_ids = sorted(self.doc_len.keys())
        for doc_id in doc_ids:
            if "\n" in doc_id:
                raise ValueError(f"doc_id {doc_id!r} contains a newline; doc_ids.txt can't round-trip it")
        doc_id_to_int = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        doc_len_arr = [self.doc_len[doc_id] for doc_id in doc_ids]

        terms = sorted(self.postings.keys())
        postings_blob = bytearray()
        term_lengths: List[int] = []
        for term in terms:
            doc_tf = self.postings[term]
            int_ids = sorted(doc_id_to_int[doc_id] for doc_id in doc_tf)
            values: List[int] = []
            prev = 0
            for int_id in int_ids:
                doc_id = doc_ids[int_id]
                values.append(int_id - prev)
                values.append(doc_tf[doc_id])
                prev = int_id
            term_bytes = vbyte.encode_uints(values)
            term_lengths.append(len(term_bytes))
            postings_blob += term_bytes

        meta = {"N": self.N, "avg_doc_len": self.avg_doc_len, "n_docs": len(doc_ids), "n_terms": len(terms)}
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, separators=(",", ":"))
        with open(os.path.join(index_dir, "doc_ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(doc_ids))
        with open(os.path.join(index_dir, "doc_len.bin"), "wb") as f:
            f.write(vbyte.encode_uints(doc_len_arr))
        with open(os.path.join(index_dir, "vocab_terms.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(terms))
        with open(os.path.join(index_dir, "vocab_lengths.bin"), "wb") as f:
            f.write(vbyte.encode_uints(term_lengths))
        with open(os.path.join(index_dir, "postings.bin"), "wb") as f:
            f.write(bytes(postings_blob))

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct an InvertedIndex purely from what save() wrote to
        `index_dir`. Called in a fresh process — do not rely on any state
        other than what's actually on disk in `index_dir`."""
        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)

        index = cls()
        index.N = meta["N"]
        index.avg_doc_len = meta["avg_doc_len"]
        n_docs = meta["n_docs"]
        n_terms = meta["n_terms"]

        with open(os.path.join(index_dir, "doc_ids.txt"), encoding="utf-8") as f:
            content = f.read()
        doc_ids: List[str] = content.split("\n") if content else []

        with open(os.path.join(index_dir, "doc_len.bin"), "rb") as f:
            doc_len_bytes = f.read()
        doc_len_arr, _ = vbyte.decode_uints(doc_len_bytes, n_docs)
        index.doc_len = dict(zip(doc_ids, doc_len_arr))

        with open(os.path.join(index_dir, "vocab_terms.txt"), encoding="utf-8") as f:
            content = f.read()
        terms: List[str] = content.split("\n") if content else []

        with open(os.path.join(index_dir, "vocab_lengths.bin"), "rb") as f:
            lengths_bytes = f.read()
        term_lengths, _ = vbyte.decode_uints(lengths_bytes, n_terms)

        with open(os.path.join(index_dir, "postings.bin"), "rb") as f:
            postings_blob = f.read()

        index.postings = {}
        offset = 0
        for term, length in zip(terms, term_lengths):
            values = vbyte.decode_all_uints(postings_blob[offset : offset + length])
            offset += length
            doc_tf: Dict[str, int] = {}
            int_id = 0
            for i in range(0, len(values), 2):
                int_id += values[i]
                tf = values[i + 1]
                doc_tf[doc_ids[int_id]] = tf
            index.postings[term] = doc_tf

        return index
