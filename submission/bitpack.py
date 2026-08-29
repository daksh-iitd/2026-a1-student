"""
submission/bitpack.py — fixed-width 2-bit code packing, used by
InvertedIndex.save()/load() (submission/indexer.py) to compress the
term-frequency stream of the persisted index (assignment Section 7,
"index size" score).

Motivation (measured on the full trec-covid index, see report.tex): 72.6%
of all postings have tf == 1, and 95.8% have tf <= 4 — but plain VByte
(submission/vbyte.py) still spends a full byte on nearly every one of
them, since VByte's minimum unit is one byte. A term's tf only needs to
distinguish "1, 2, 3, or 4-or-more" for the overwhelming majority of
postings, which fits in 2 bits instead of 8.

Encoding: each tf value is first reduced to a 2-bit *code* by the caller
(indexer.py): code = min(tf, 4) - 1, i.e. 0/1/2 mean tf is exactly 1/2/3,
and 3 is an escape meaning "the real value is >= 4, stored separately" in
a parallel VByte-encoded escape list, consumed in the same order the
escape codes are encountered. This module only handles the packing of
the 2-bit codes themselves, 4 per byte, least-significant pair first;
it has no opinion on what the codes mean.
"""
from typing import List


def pack_2bit(codes: List[int]) -> bytes:
    """Pack a list of 2-bit codes (each in [0, 3]) 4-to-a-byte. The final
    byte is zero-padded if `len(codes)` isn't a multiple of 4 — callers
    must track the true count separately (e.g. total posting count in
    meta.json) since the padding bits aren't distinguishable from real
    zero codes."""
    out = bytearray()
    buf = 0
    nbits = 0
    for code in codes:
        if not 0 <= code <= 3:
            raise ValueError(f"pack_2bit requires codes in [0, 3], got {code}")
        buf |= code << nbits
        nbits += 2
        if nbits == 8:
            out.append(buf)
            buf = 0
            nbits = 0
    if nbits:
        out.append(buf)
    return bytes(out)


def unpack_2bit(data: bytes, count: int) -> List[int]:
    """Unpack exactly `count` 2-bit codes from `data`, in the order
    pack_2bit() wrote them. `count` must be supplied by the caller (see
    pack_2bit's docstring on why it can't be inferred from `len(data)`
    alone)."""
    codes: List[int] = []
    byte_idx = 0
    bit_idx = 0
    for _ in range(count):
        byte = data[byte_idx]
        codes.append((byte >> bit_idx) & 0x3)
        bit_idx += 2
        if bit_idx == 8:
            bit_idx = 0
            byte_idx += 1
    return codes
