"""Features read from BED-like files, and the stretches of a gpxil file that commands scan.

Shared by ``gpxil runs``, ``stack``, ``spectrum``, ``skew`` and ``annotate``:

- :func:`read_bed` reads intervals (BED, plain or gzipped; 0-based, the end
  excluded) with their name, score and strand when the file has them;
- :func:`targets` gives what a command scans: the regions asked for, read as
  ``gpxil stats`` reads them, or else every sequence of the file, each base
  once (windows that overlap or touch are joined: spec 3.1);
- :func:`on_views` puts features on those stretches, by the sequence's name
  as the file finds it (``chr1`` and ``1`` alike) and at the stretch's own
  offsets, so that positions on a window are its parent's.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass, field

import numpy as np

from . import codes as C

CHUNK = 1 << 22  #: bases read at a time


@dataclass
class Feature:
    """One interval of a BED-like file: ``[start, end)``, 0-based."""

    contig: str
    start: int
    end: int
    name: str = ""
    score: float | None = None
    strand: str = "."
    fields: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start


def _open_text(path):
    path = os.fspath(path)
    with open(path, "rb") as fh:
        magic = fh.read(2)
    return gzip.open(path, "rt") if magic == b"\x1f\x8b" else open(path)


def read_bed(path, *, columns: tuple[int, int, int] = (0, 1, 2), one_based: bool = False) -> list[Feature]:
    """The intervals of a BED file (plain or gzipped): track, browser and ``#`` lines are skipped. ``columns`` gives
    the (0-based) columns of the contig, start and end, for tables laid out otherwise (UCSC's rmsk: 5, 6, 7); a BED's
    name, score and strand are columns 4, 5 and 6. ``one_based``: the starts are 1-based (GFF-like)."""
    out = []
    ci, si, ei = columns
    with _open_text(path) as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\r\n").split("\t")
            if len(f) < 3:
                f = line.split()
            try:
                contig, start, end = f[ci], int(f[si]) - (1 if one_based else 0), int(f[ei])
            except (IndexError, ValueError):
                raise ValueError(f"{path}, line {n}: not a BED interval ({line.strip()[:60]!r})") from None
            if start < 0 or end < start:
                raise ValueError(f"{path}, line {n}: start {start} and end {end} are not an interval")
            name = f[3] if len(f) > 3 and columns == (0, 1, 2) else ""
            score = None
            if len(f) > 4 and columns == (0, 1, 2):
                try:
                    score = float(f[4])
                except ValueError:
                    score = None
            strand = f[5] if len(f) > 5 and columns == (0, 1, 2) and f[5] in ("+", "-", ".") else "."
            out.append(Feature(contig, start, end, name, score, strand, f))
    return out


def targets(gfile, regions=None) -> list[tuple]:
    """What to scan: ``(view, start, end)`` at the view's offsets. ``regions``, ``(name, start, end)`` as
    ``GpxilFile.region_of`` gives them (start and end None for a whole contig), are read as ``gpxil stats`` reads
    them: a sequence named that only windows hold is each of its windows, and a region is read through the windows
    that overlap or touch the one that holds it. Without regions, every sequence of the file, each base once."""
    if not regions:
        return [(v, 0, v.length) for v in gfile.stretches()]
    out = []
    for name, s, e in regions:
        if s is None and e is None and gfile.find(name) is None and len(gfile.windows_of(name)) > 1:
            parts = [(w, 0, w.length) for w in gfile.windows_of(name)]
        else:
            parts = [gfile.locate(name, s, e, join=True)]
        out += [gfile.view(c, a, b) for c, a, b in parts]
    return out


def where(view, offset: int) -> int:
    """A view's offset as a position in its sequence (0-based)."""
    return view.start + offset


def sequence_name(gfile, name: str) -> str | None:
    """The name of the sequence a BED's contig name is, in this file (as the file finds names: ``1`` is ``chr1``),
    or None if the file holds no bases of it."""
    c = gfile.find(name)
    if c is not None:
        return c.parent
    wins = gfile.windows_of(name)
    return wins[0].parent if wins else None


def on_views(gfile, features: list[Feature], views: list[tuple]) -> tuple[list[list[tuple[Feature, int, int]]], int]:
    """The features on each scanned stretch: for each ``(view, start, end)``, the features that overlap it, as
    ``(feature, a, b)`` at the view's offsets, clipped to ``[start, end)``; and how many features fall on none."""
    by_seq: dict[str, list[Feature]] = {}
    cache: dict[str, str | None] = {}
    for f in features:
        if f.contig not in cache:
            cache[f.contig] = sequence_name(gfile, f.contig)
        seq = cache[f.contig]
        if seq is not None:
            by_seq.setdefault(seq, []).append(f)
    out: list[list[tuple[Feature, int, int]]] = []
    placed: set[int] = set()
    for v, s, e in views:
        got = []
        lo, hi = v.start + s, v.start + e
        for f in by_seq.get(v.parent, []):
            a, b = max(f.start, lo), min(f.end, hi)
            if a < b or (f.start == f.end and lo <= f.start < hi):
                got.append((f, a - v.start, max(b, a) - v.start))
                placed.add(id(f))
        out.append(got)
    return out, sum(id(f) not in placed for f in features)


def mask_of(n: int, intervals, offset: int = 0) -> np.ndarray:
    """A boolean array of length ``n``: True on the intervals ``(a, b)`` (shifted by ``-offset``, clipped)."""
    delta = np.zeros(n + 1, dtype=np.int32)
    for a, b in intervals:
        a, b = max(0, a - offset), min(n, b - offset)
        if a < b:
            delta[a] += 1
            delta[b] -= 1
    return np.cumsum(delta[:-1]) > 0


# ------------------------------------------------------------------ bases

#: IUPAC letter -> 4-bit set
SET_OF = {s: i for i, s in enumerate(C.IUPAC) if s.isalpha()}
COMPLEMENT_SET = np.array([((c & 1) << 3) | ((c & 2) << 1) | ((c & 4) >> 1) | ((c & 8) >> 3) for c in range(16)], dtype=np.uint8)
LETTER = np.frombuffer(C.IUPAC.encode(), dtype=np.uint8)  #: 4-bit set -> IUPAC letter (0: '-')


def base_set(text: str) -> int:
    """A set of bases as a 4-bit mask, from IUPAC letters: ``AG``, ``R`` (A or G), ``GA``, ``S``..."""
    m = 0
    for ch in str(text).strip().upper():
        if ch == "U":
            ch = "T"
        if ch not in SET_OF:
            raise ValueError(f"{text!r}: give bases as IUPAC letters (A, C, G, T, or R, Y, S, W, K, M, B, D, H, V, N)")
        m |= SET_OF[ch]
    if not m:
        raise ValueError("an empty set of bases")
    return m


def set_name(mask: int) -> str:
    """The letters of a 4-bit set of bases, in the order ACGT: 5 -> 'AG'."""
    return "".join(b for b, bit in zip("ACGT", (1, 2, 4, 8)) if mask & bit)


def reverse_complement_codes(codes: np.ndarray) -> np.ndarray:
    """Reference codes (4-bit sets, soft-mask bit kept) of the other strand, read 5' to 3'."""
    codes = np.asarray(codes, dtype=np.uint8)
    return (COMPLEMENT_SET[codes & 0x0F] | (codes & C.SOFTMASK))[::-1]
