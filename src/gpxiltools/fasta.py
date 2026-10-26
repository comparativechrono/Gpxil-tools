"""Streaming FASTA reading into ref-plane codes.

Sequences are converted with a 256-entry lookup table, so a whole chromosome
is encoded in well under a second. Unlike a loop that skips characters it
does not recognise, the position of every base is preserved: soft-masked
(lowercase) bases keep their code plus the soft-mask bit, and N runs stay N.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
from dataclasses import dataclass
from typing import BinaryIO, Iterator

import numpy as np

from .codes import ASCII_TO_REF, INVALID, REF_TO_ASCII

_WHITESPACE = b"\n\r\t \x0b\x0c"
_BLOCK = 1 << 24


class FastaError(ValueError):
    pass


@dataclass
class FastaRecord:
    name: str
    description: str
    codes: np.ndarray  # uint8 ref-plane codes, one per base
    converted: int = 0  # characters that were not IUPAC symbols and became N

    @property
    def length(self) -> int:
        return int(self.codes.shape[0])


def open_maybe_gzip(path) -> BinaryIO:
    """Open a plain, gzip or bgzip file for binary reading."""
    fh = open(path, "rb")
    magic = fh.read(2)
    fh.seek(0)
    if magic == b"\x1f\x8b":
        return gzip.GzipFile(fileobj=fh)  # bgzip is multi-member gzip
    return fh


def encode_sequence(seq: bytes, *, name: str = "", offset: int = 0, invalid: str = "error", counts: list | None = None) -> np.ndarray:
    """ASCII sequence (no whitespace) -> uint8 ref codes.

    Characters that are not IUPAC symbols raise :class:`FastaError`, or with
    ``invalid='N'`` become N; their number is appended to ``counts``.
    """
    raw = np.frombuffer(seq, dtype=np.uint8)
    codes = ASCII_TO_REF[raw]
    bad = np.flatnonzero(codes == INVALID)
    if bad.size:
        if invalid == "error":
            i = int(bad[0])
            raise FastaError(
                f"{name}: invalid sequence character {chr(raw[i])!r} at offset {offset + i}; "
                "use invalid='N' (gpxil encode --invalid-to-n) to convert such characters to N"
            )
        codes[bad] = 0x0F
    if counts is not None:
        counts.append(int(bad.size))
    return codes


def iter_fasta(path, *, invalid: str = "error") -> Iterator[FastaRecord]:
    """Yield one :class:`FastaRecord` per sequence in a FASTA file."""
    with open_maybe_gzip(path) as fh:
        yield from _iter_fasta_stream(fh, invalid=invalid)


def _iter_fasta_stream(fh: BinaryIO, *, invalid: str) -> Iterator[FastaRecord]:
    header: bytes | None = None
    parts: list[bytes] = []
    pending = b""
    while True:
        block = fh.read(_BLOCK)
        if not block:
            data, pending, final = pending, b"", True
        else:
            data = pending + block
            cut = data.rfind(b"\n")
            if cut == -1:
                pending = data
                continue
            data, pending, final = data[: cut + 1], data[cut + 1 :], False
        pos = 0
        n = len(data)
        while pos < n:
            if data.startswith(b">", pos):
                if header is not None:
                    yield _make_record(header, parts, invalid)
                eol = data.find(b"\n", pos)
                eol = n if eol == -1 else eol
                header = data[pos + 1 : eol].rstrip(b"\r")
                parts = []
                pos = eol + 1
                continue
            nxt = data.find(b"\n>", pos)
            end = n if nxt == -1 else nxt + 1
            if header is None:
                if data[pos:end].strip():
                    raise FastaError("sequence data before the first '>' header line")
            else:
                parts.append(data[pos:end])
            pos = end
        if final:
            break
    if header is not None:
        yield _make_record(header, parts, invalid)


def _make_record(header: bytes, parts: list[bytes], invalid: str) -> FastaRecord:
    text = header.decode("utf-8", "replace").strip()
    if not text:
        raise FastaError("empty FASTA header")
    name, _, desc = text.partition(" ")
    # Encode block by block into one preallocated array, releasing each text
    # block as soon as it is encoded: peak memory stays near 2 bytes per base.
    total = 0
    for i, p in enumerate(parts):
        parts[i] = p = p.translate(None, _WHITESPACE)
        total += len(p)
    codes = np.empty(total, dtype=np.uint8)
    counts: list[int] = []
    pos = 0
    for i, p in enumerate(parts):
        k = len(p)
        codes[pos : pos + k] = encode_sequence(p, name=name, offset=pos, invalid=invalid, counts=counts)
        parts[i] = b""
        pos += k
    parts.clear()
    return FastaRecord(name=name, description=desc.strip(), codes=codes, converted=sum(counts))


def read_fasta(path, *, invalid: str = "error") -> list[FastaRecord]:
    return list(iter_fasta(path, invalid=invalid))


def sequence_digests(codes: np.ndarray) -> dict[str, str]:
    """MD5 (as in SAM ``@SQ M5``) and GA4GH refget ``sha512t24u`` of a sequence.

    Both digests are over the uppercase sequence, so they identify the
    reference independently of soft-masking.
    """
    md5_h = hashlib.md5()
    sha_h = hashlib.sha512()
    step = 1 << 24
    for s in range(0, codes.shape[0], step):
        upper = REF_TO_ASCII[np.bitwise_and(codes[s : s + step], 0x0F)].tobytes()
        md5_h.update(upper)
        sha_h.update(upper)
    md5 = md5_h.hexdigest()
    sha = sha_h.digest()[:24]
    refget = "SQ." + base64.urlsafe_b64encode(sha).decode("ascii")
    return {"md5": md5, "refget": refget}


def write_fasta(out: io.TextIOBase | BinaryIO, name: str, seq: bytes, *, line_width: int = 60, description: str = "") -> None:
    """Write one FASTA record (``seq`` is ASCII bytes)."""
    header = f">{name}" + (f" {description}" if description else "") + "\n"
    binary = not isinstance(out, io.TextIOBase)
    write = out.write
    if binary:
        write(header.encode())
        for i in range(0, len(seq), line_width):
            write(seq[i : i + line_width] + b"\n")
    else:
        write(header)
        for i in range(0, len(seq), line_width):
            write(seq[i : i + line_width].decode("ascii") + "\n")
