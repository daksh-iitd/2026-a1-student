"""
submission/vbyte.py — variable-byte (VByte) integer encoding, used by
InvertedIndex.save()/load() (submission/indexer.py) to compress the
persisted index (assignment Section 7, "index size" score).

Standard VByte scheme (Manning/Raghavan/Schütze, Introduction to
Information Retrieval, sec. 5.3.2): each non-negative integer is split
into 7-bit groups, least-significant group first; each group is stored
in one byte, with the high bit (0x80) set to 1 on the LAST byte of the
number (marking where it ends) and 0 on every byte before it. Small
numbers — the common case for doc-id gaps and term frequencies in an
inverted index's postings — take 1 byte each instead of the several
ASCII digit bytes plus a separator that a JSON integer costs.
"""
from typing import List, Tuple


def encode_uint(n: int) -> bytes:
    """Encode one non-negative int as a VByte byte sequence."""
    if n < 0:
        raise ValueError(f"encode_uint requires n >= 0, got {n}")
    out = bytearray()
    while True:
        out.append(n & 0x7F)
        n >>= 7
        if n == 0:
            break
    out[-1] |= 0x80  # high bit marks the last byte of this number
    return bytes(out)


def encode_uints(values: List[int]) -> bytes:
    """Encode a sequence of non-negative ints, concatenated in order."""
    out = bytearray()
    for n in values:
        out += encode_uint(n)
    return bytes(out)


def decode_uint(data: bytes, pos: int) -> Tuple[int, int]:
    """Decode one VByte int starting at `pos`. Returns (value, next_pos)."""
    n = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        n |= (byte & 0x7F) << shift
        if byte & 0x80:
            break
        shift += 7
    return n, pos


def decode_uints(data: bytes, count: int, pos: int = 0) -> Tuple[List[int], int]:
    """Decode exactly `count` consecutive VByte ints starting at `pos`.
    Returns (values, next_pos) — next_pos lets callers chain multiple
    VByte-encoded regions back to back in one buffer."""
    values = []
    for _ in range(count):
        n, pos = decode_uint(data, pos)
        values.append(n)
    return values, pos


def decode_all_uints(data: bytes) -> List[int]:
    """Decode every VByte int in `data`, from position 0 to the end —
    used when the count isn't tracked separately (e.g. a self-contained
    per-term postings slice)."""
    values = []
    pos = 0
    n_bytes = len(data)
    while pos < n_bytes:
        n, pos = decode_uint(data, pos)
        values.append(n)
    return values
