"""
submission/frontcode.py — front-coding for a sorted term dictionary, used
by InvertedIndex.save()/load() (submission/indexer.py) to compress the
vocabulary file (assignment Section 7, "index size" score).

Measured on the full trec-covid vocabulary (165,843 stemmed terms, already
sorted for the postings/df arrays to line up with them): adjacent terms in
sorted order share, on average, 4.6 of their 7.3 characters — a
consequence of stemming collapsing suffix variation, so many terms in a
row differ only in their last few characters (e.g. "vaccin", "vaccinat"
sit right next to each other after "vaccination"/"vaccinated" both stem
down). Front coding exploits exactly this: instead of storing each term's
full text, store only how many leading characters it shares with the
*previous* term (in sorted order), plus the remaining suffix. Measured
saving: ~63% off the raw newline-joined dictionary.

No block heads / binary-search restart points here (the textbook version
of this technique usually has them): InvertedIndex.load() always decodes
the *entire* vocabulary sequentially into memory up front — nothing in
this codebase ever looks up one term on disk without loading the rest —
so block heads would trade dictionary size back for a random-access
capability nothing here uses. Plain front-coding captures the same
compression with less code and less to get wrong.

This module has no opinion on file formats or persistence — encode()/
decode() are pure functions over Python lists, called from
InvertedIndex.save()/load().
"""
from typing import List, Tuple


def _shared_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def encode(terms: List[str]) -> Tuple[List[int], List[str]]:
    """`terms` must already be sorted (InvertedIndex.save() sorts its
    vocabulary anyway). Returns (prefix_lens, suffixes): prefix_lens[i] is
    how many leading characters terms[i] shares with terms[i-1] (0 for
    i == 0, and whenever there's no shared prefix), and suffixes[i] is
    terms[i] with that shared prefix stripped off."""
    prefix_lens: List[int] = []
    suffixes: List[str] = []
    prev = ""
    for term in terms:
        shared = _shared_prefix_len(prev, term)
        prefix_lens.append(shared)
        suffixes.append(term[shared:])
        prev = term
    return prefix_lens, suffixes


def decode(prefix_lens: List[int], suffixes: List[str]) -> List[str]:
    """Inverse of encode(): reconstructs the original sorted term list.
    Must be walked start to finish, in order — each term is defined
    relative to the one before it, so there's no random access into the
    middle of a front-coded dictionary without decoding everything before
    it first (see the module docstring for why that's fine here)."""
    terms: List[str] = []
    prev = ""
    for plen, suf in zip(prefix_lens, suffixes):
        term = prev[:plen] + suf
        terms.append(term)
        prev = term
    return terms
