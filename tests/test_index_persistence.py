"""
tests/test_index_persistence.py — self-authored correctness tests for the
VByte encoding (submission/vbyte.py) and the InvertedIndex save()/load()
round trip (submission/indexer.py) built on top of it.

Not part of the harness's own conformance/metrics checks (those live in
test_interface_conformance.py / test_metrics.py) — these exist so a bug
in the on-disk compression format fails loudly and locally, on hand-
checkable examples, rather than showing up as a silent ranking-quality
regression at grading time.
"""
import os
import tempfile

from submission import vbyte
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

TOY_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "toy", "corpus.jsonl")


# ---------------------------------------------------------------------------
# vbyte.py
# ---------------------------------------------------------------------------

def test_encode_decode_uint_round_trip_small_values():
    for n in [0, 1, 2, 5, 100, 127]:
        encoded = vbyte.encode_uint(n)
        assert len(encoded) == 1, f"{n} should fit in exactly 1 byte, got {len(encoded)}"
        decoded, next_pos = vbyte.decode_uint(encoded, 0)
        assert decoded == n
        assert next_pos == len(encoded)


def test_encode_decode_uint_round_trip_at_byte_boundaries():
    # 127 = 0b01111111 is the largest 1-byte value; 128 = 0b10000000 must
    # spill into a 2nd byte. Same pattern one level up: 16383 = 2^14-1 is
    # the largest 2-byte value, 16384 needs a 3rd byte.
    cases = [126, 127, 128, 129, 16383, 16384, 16385, 2_000_000]
    for n in cases:
        encoded = vbyte.encode_uint(n)
        decoded, next_pos = vbyte.decode_uint(encoded, 0)
        assert decoded == n, f"round-trip failed for {n}"
        assert next_pos == len(encoded)


def test_encode_uint_rejects_negative():
    try:
        vbyte.encode_uint(-1)
        assert False, "expected ValueError for a negative input"
    except ValueError:
        pass


def test_encode_uints_and_decode_all_uints_round_trip():
    values = [0, 1, 127, 128, 300, 16383, 16384, 999999, 42]
    encoded = vbyte.encode_uints(values)
    assert vbyte.decode_all_uints(encoded) == values


def test_decode_uints_stops_after_count_and_reports_next_pos():
    # Two VByte regions back to back (mirrors how indexer.py packs a
    # term's postings slice inside one shared postings.bin buffer):
    # decode_uints() must be able to read exactly `count` ints from the
    # first region and hand back a position the caller can resume from.
    first = vbyte.encode_uints([1, 2, 3])
    second = vbyte.encode_uints([400, 5])
    buf = first + second

    values, pos = vbyte.decode_uints(buf, count=3, pos=0)
    assert values == [1, 2, 3]
    assert pos == len(first)

    values2, pos2 = vbyte.decode_uints(buf, count=2, pos=pos)
    assert values2 == [400, 5]
    assert pos2 == len(buf)


def test_encode_uints_empty_list_is_empty_bytes():
    assert vbyte.encode_uints([]) == b""
    assert vbyte.decode_all_uints(b"") == []


# ---------------------------------------------------------------------------
# InvertedIndex.save() / load() round trip
# ---------------------------------------------------------------------------

def _build_toy_index() -> InvertedIndex:
    corpus = load_corpus(TOY_CORPUS_PATH)
    index = InvertedIndex()
    index.build(corpus)
    return index


def test_save_load_round_trip_preserves_all_index_state():
    original = _build_toy_index()
    with tempfile.TemporaryDirectory() as index_dir:
        original.save(index_dir)
        reloaded = InvertedIndex.load(index_dir)

    assert reloaded.N == original.N
    assert reloaded.avg_doc_len == original.avg_doc_len
    assert reloaded.doc_len == original.doc_len
    assert set(reloaded.postings.keys()) == set(original.postings.keys())
    for term in original.postings:
        assert reloaded.postings[term] == original.postings[term], f"postings mismatch for term {term!r}"


def test_save_writes_expected_files_and_no_leftover_json_blob():
    index = _build_toy_index()
    with tempfile.TemporaryDirectory() as index_dir:
        index.save(index_dir)
        written = set(os.listdir(index_dir))

    assert written == {
        "meta.json",
        "doc_ids.txt",
        "doc_len.bin",
        "vocab_terms.txt",
        "vocab_lengths.bin",
        "postings.bin",
    }


def test_document_frequency_matches_after_round_trip():
    original = _build_toy_index()
    with tempfile.TemporaryDirectory() as index_dir:
        original.save(index_dir)
        reloaded = InvertedIndex.load(index_dir)

    for term in original.postings:
        assert reloaded.document_frequency(term) == original.document_frequency(term)


def test_empty_corpus_round_trips_without_crashing():
    index = InvertedIndex()
    index.build([])
    with tempfile.TemporaryDirectory() as index_dir:
        index.save(index_dir)
        reloaded = InvertedIndex.load(index_dir)

    assert reloaded.N == 0
    assert reloaded.doc_len == {}
    assert reloaded.postings == {}


def test_vbyte_encoding_is_smaller_than_naive_json_for_the_toy_index():
    import json as _json

    index = _build_toy_index()
    with tempfile.TemporaryDirectory() as index_dir:
        index.save(index_dir)
        vbyte_size = sum(
            os.path.getsize(os.path.join(index_dir, fn)) for fn in os.listdir(index_dir)
        )

    naive_json_size = len(
        _json.dumps(
            {
                "doc_len": index.doc_len,
                "postings": index.postings,
            }
        ).encode("utf-8")
    )
    assert vbyte_size < naive_json_size, (
        f"VByte-encoded index ({vbyte_size}B) should beat a naive JSON dump "
        f"({naive_json_size}B) even on a 20-document toy corpus"
    )
