"""Megatron .bin/.idx reader. The .bin is a flat token stream; the .idx (little-endian) holds a
9-byte header "MMIDIDX\\x00\\x00", uint64 version=1, uint8 dtype code, uint64 n_seqs, uint64 n_docs,
int32 sizes[n_seqs], int64 pointers[n_seqs], int64 doc_idx[n_docs]. One document per sequence."""

import struct

import numpy as np

_HEADER = b"MMIDIDX\x00\x00"
_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16}


def _read_header(path_prefix):
    with open(path_prefix + ".idx", "rb") as f:
        assert f.read(9) == _HEADER, f"bad header in {path_prefix}.idx"
        assert struct.unpack("<Q", f.read(8))[0] == 1
        code = struct.unpack("<B", f.read(1))[0]
        n_seqs, n_docs = struct.unpack("<QQ", f.read(16))
        return _DTYPES[code], n_seqs, n_docs, f.tell()


def open_megatron(path_prefix):
    """<prefix>.bin as a flat token memmap."""
    return np.memmap(path_prefix + ".bin", mode="r", dtype=_read_header(path_prefix)[0])


def read_doc_offsets(path_prefix):
    """Document boundaries in token space: int64[n_docs + 1], doc d spans [off[d], off[d+1])."""
    _, n_seqs, n_docs, end = _read_header(path_prefix)
    buf = np.memmap(path_prefix + ".idx", mode="r")
    sizes = np.frombuffer(buf, dtype=np.int32, count=n_seqs, offset=end)
    doc_idx = np.frombuffer(buf, dtype=np.int64, count=n_docs, offset=end + sizes.nbytes + 8 * n_seqs)
    seq_offsets = np.zeros(n_seqs + 1, dtype=np.int64)
    np.cumsum(sizes, out=seq_offsets[1:])
    return seq_offsets[doc_idx]


def window_cu_seqlens(doc_offsets, starts, block_size):
    """cu_seqlens (int32, over the flattened batch) for token windows [a, a + block_size): every window
    start is a boundary, plus every document start strictly inside a window."""
    cu = [0]
    for i, a in enumerate(starts):
        a = int(a)
        lo = np.searchsorted(doc_offsets, a, side="right")
        hi = np.searchsorted(doc_offsets, a + block_size, side="left")
        cu.extend((doc_offsets[lo:hi] - a + i * block_size).tolist())
        cu.append((i + 1) * block_size)
    return np.asarray(cu, dtype=np.int32)


class MegatronWriter:
    """Writes documents (uint16 token ids) as <prefix>.bin/.idx."""

    def __init__(self, path_prefix):
        self.path_prefix, self.sizes, self.total_tokens = path_prefix, [], 0
        self.bin = open(path_prefix + ".bin", "wb")

    def add_document(self, tokens):
        self.bin.write(np.asarray(tokens, dtype=np.uint16).tobytes())
        self.sizes.append(len(tokens))
        self.total_tokens += len(tokens)

    def finalize(self):
        self.bin.close()
        write_idx(self.path_prefix + ".idx", np.asarray(self.sizes, dtype=np.int32))


def write_idx(path, sizes):
    pointers = np.zeros(len(sizes), dtype=np.int64)
    np.cumsum(sizes[:-1].astype(np.int64) * 2, out=pointers[1:])
    with open(path, "wb") as f:
        f.write(_HEADER + struct.pack("<QB", 1, 8) + struct.pack("<QQ", len(sizes), len(sizes) + 1))
        f.write(sizes.tobytes() + pointers.tobytes() + np.arange(len(sizes) + 1, dtype=np.int64).tobytes())
