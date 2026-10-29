"""Features read from BED-like files, and the stretches of a gpxil file that commands scan.

Shared by ``gpxil runs``, ``stack``, ``spectrum``, ``skew`` and ``annotate``:

- :func:`read_features` reads intervals from BED files (plain or gzipped;
  0-based, the end excluded), with their name, score and strand when the
  file has them, or a RepeatMasker table's repeats (:func:`read_repeats`);
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
from dataclasses import dataclass

import numpy as np

from . import codes as C

CHUNK = 1 << 22  #: bases read at a time


@dataclass(slots=True)
class Feature:
    """One interval of a BED-like file: ``[start, end)``, 0-based."""

    contig: str
    start: int
    end: int
    name: str = ""
    score: float | None = None
    strand: str = "."
    fields: tuple = ()  #: a BED line's columns after the sixth

    @property
    def length(self) -> int:
        return self.end - self.start


def _open_text(path):
    path = os.fspath(path)
    with open(path, "rb") as fh:
        magic = fh.read(2)
    return gzip.open(path, "rt") if magic == b"\x1f\x8b" else open(path)


class _Keep:
    """Which features to keep as a file is read (a name matching ``match``, a length of ``min_length`` or more),
    and one copy of each name and contig, which many features share."""

    def __init__(self, match: str | None = None, min_length: int = 0):
        import re

        try:
            self.pat = re.compile(match) if match else None
        except re.error as exc:
            raise ValueError(f"{match!r} is not a regular expression: {exc}") from None
        self.min_length = min_length
        self.names: dict[str, bool] = {}
        self.shared: dict[str, str] = {}

    def __call__(self, name: str, length: int) -> bool:
        if length < self.min_length:
            return False
        if self.pat is None:
            return True
        ok = self.names.get(name)
        if ok is None:
            ok = self.names[name] = bool(self.pat.search(name))
        return ok

    def one(self, text: str) -> str:
        return self.shared.setdefault(text, text)


def _skip(line: str) -> bool:
    return not line.strip() or line.startswith(("#", "track", "browser"))


def read_bed(path, *, match: str | None = None, min_length: int = 0) -> list[Feature]:
    """The intervals of a BED file (plain or gzipped; track, browser and ``#`` lines skipped), with their name,
    score and strand when it has them (columns 4, 5 and 6). ``match``, ``min_length``: as :func:`chosen`, as the
    file is read."""
    keep = _Keep(match, min_length)
    out = []
    with _open_text(path) as fh:
        for n, line in enumerate(fh, 1):
            if _skip(line):
                continue
            f = line.rstrip("\r\n").split("\t")
            if len(f) < 3:
                f = line.split()
            try:
                contig, start, end = f[0], int(f[1]), int(f[2])
            except (IndexError, ValueError):
                raise ValueError(f"{path}, line {n}: not a BED interval ({line.strip()[:60]!r})") from None
            if start < 0 or end < start:
                raise ValueError(f"{path}, line {n}: start {start} and end {end} are not an interval")
            name = f[3] if len(f) > 3 else ""
            if not keep(name, end - start):
                continue
            score = None
            if len(f) > 4:
                try:
                    score = float(f[4])
                except ValueError:
                    score = None
            strand = f[5] if len(f) > 5 and f[5] in ("+", "-") else "."
            out.append(Feature(keep.one(contig), start, end, keep.one(name), score, strand, tuple(f[6:])))
    return out


def _repeat_name(cls: str, family: str, name: str) -> str:
    """A repeat's name as a path: class, family (when it isn't the class again), name: 'SINE/Alu/AluSx'."""
    parts = [cls] + ([family] if family and family != cls else []) + [name]
    return "/".join(p for p in parts if p)


def _whole(text: str) -> bool:
    return text.lstrip("-").isdigit()


def _rmsk_kind(line: str) -> str | None:
    """How a RepeatMasker table lays out its lines, from one: ``ucsc1`` (UCSC's rmsk, tab-separated, 0-based, its
    first column ``bin``), ``ucsc0`` (without ``bin``), ``out`` (RepeatMasker's ``.out``: separated by spaces,
    1-based, ``C`` for the - strand); None if the line is none of them."""
    t = line.rstrip("\r\n").split("\t")
    for k in (1, 0):
        if len(t) >= 16 + k and t[8 + k] in ("+", "-") and all(_whole(t[i + k]) for i in (1, 5, 6, 7)):
            return f"ucsc{k}"
    w = line.split()
    if len(w) >= 15 and w[8] in ("+", "C") and all(_whole(w[i]) for i in (0, 5, 6)):
        try:
            float(w[1])
        except ValueError:
            return None
        return "out"
    return None


def _rmsk_line(kind: str, line: str) -> tuple:
    """``(contig, start, end, strand, name, class, family, divergence %)`` of a line laid out as ``kind`` says."""
    if kind == "out":
        w = line.split()
        cls, _, family = w[10].partition("/")
        return w[4], int(w[5]) - 1, int(w[6]), {"+": "+", "C": "-"}[w[8]], w[9], cls, family, float(w[1])
    k = int(kind[-1])
    t = line.rstrip("\r\n").split("\t")
    return t[4 + k], int(t[5 + k]), int(t[6 + k]), {"+": "+", "-": "-"}[t[8 + k]], t[9 + k], t[10 + k], t[11 + k], int(t[1 + k]) / 10


def read_repeats(path, *, match: str | None = None, min_length: int = 0) -> list[Feature]:
    """The repeats of a RepeatMasker table (plain or gzipped): UCSC's rmsk (``rmsk.txt.gz``, or the Table Browser's
    output) or RepeatMasker's own ``.out``. A repeat's name is its class, family and name, ``SINE/Alu/AluSx``
    (``Simple_repeat/(GA)n``: a family named as its class is left out); its score is its divergence from the
    repeat's consensus, in %, which grows with its age. ``match``, ``min_length``: as :func:`chosen`, as the table
    is read (a genome's table has millions of lines)."""
    keep = _Keep(match, min_length)
    names: dict[tuple, str] = {}
    out = []
    kind = None
    with _open_text(path) as fh:
        for n, line in enumerate(fh, 1):
            if _skip(line):
                continue
            if kind is None:
                kind = _rmsk_kind(line)
                if kind is None:
                    if n <= 3 and not line.split()[0].isdigit():
                        continue  # (a .out file begins with three lines of headings)
                    raise ValueError(f"{path}, line {n}: not a RepeatMasker line ({line.strip()[:60]!r})")
            try:
                contig, start, end, strand, rep, cls, family, div = _rmsk_line(kind, line)
            except (IndexError, KeyError, ValueError):
                raise ValueError(f"{path}, line {n}: not a RepeatMasker line like those before it ({line.strip()[:60]!r})") from None
            if start < 0 or end < start:
                raise ValueError(f"{path}, line {n}: start {start} and end {end} are not an interval")
            key = (cls, family, rep)
            name = names.get(key)
            if name is None:
                name = names[key] = _repeat_name(cls, family, rep)
            if keep(name, end - start):
                out.append(Feature(keep.one(contig), start, end, name, div, strand))
    return out


def layout_of(path) -> str:
    """Whether a file of features is a RepeatMasker table (``'rmsk'``) or BED (``'bed'``), by its first lines."""
    with _open_text(path) as fh:
        for n, line in enumerate(fh, 1):
            if _skip(line):
                continue
            if _rmsk_kind(line) is not None:
                return "rmsk"
            if n > 3 or line.split()[0].isdigit() or len(line.split("\t")) >= 3:
                break  # (a BED line; a .out file's headings come first, then its lines)
    return "bed"


def read_features(path, layout: str = "auto", *, match: str | None = None, min_length: int = 0) -> list[Feature]:
    """Features from a BED file (``layout='bed'``) or a RepeatMasker table (``'rmsk'``: :func:`read_repeats`);
    ``'auto'`` tells them apart by their first lines (:func:`layout_of`). ``match``, ``min_length``: as
    :func:`chosen`."""
    if layout not in ("auto", "bed", "rmsk"):
        raise ValueError("layout must be 'auto', 'bed' or 'rmsk'")
    if layout == "auto":
        layout = layout_of(path)
    read = read_repeats if layout == "rmsk" else read_bed
    return read(path, match=match, min_length=min_length)


def chosen(features: list[Feature], *, match: str | None = None, min_length: int = 0) -> list[Feature]:
    """The features whose name matches the regular expression ``match`` (searched for anywhere in it:
    ``'/Alu/'``, ``'^SINE/Alu/AluY'``) and that are ``min_length`` bases or more."""
    keep = _Keep(match, min_length)
    return [f for f in features if keep(f.name, f.length)]


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


class Intervals:
    """Intervals of one sequence, merged where they overlap or touch, in order: ``starts``, ``ends`` (0-based, the
    ends excluded). :meth:`mask` cuts a piece of them as a boolean array."""

    def __init__(self, pairs=()):
        pairs = sorted((int(a), int(b)) for a, b in pairs if b > a)
        starts, ends = [], []
        for a, b in pairs:
            if ends and a <= ends[-1]:
                ends[-1] = max(ends[-1], b)
            else:
                starts.append(a)
                ends.append(b)
        self.starts = np.array(starts, dtype=np.int64)
        self.ends = np.array(ends, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.size)

    @property
    def bases(self) -> int:
        return int((self.ends - self.starts).sum())

    def mask(self, a: int, b: int) -> np.ndarray:
        """Whether each position of ``[a, b)`` is in an interval."""
        n = max(0, b - a)
        i, j = int(np.searchsorted(self.ends, a, side="right")), int(np.searchsorted(self.starts, b, side="left"))
        if i >= j:
            return np.zeros(n, dtype=bool)
        if j - i < 4096:  # (few: a slice each)
            out = np.zeros(n, dtype=bool)
            for x, y in zip((self.starts[i:j] - a).tolist(), (self.ends[i:j] - a).tolist()):
                out[max(0, x) : min(n, y)] = True
            return out
        delta = np.zeros(n + 1, dtype=np.int32)
        np.add.at(delta, np.clip(self.starts[i:j] - a, 0, n), 1)
        np.add.at(delta, np.clip(self.ends[i:j] - a, 0, n), -1)
        return np.cumsum(delta[:-1]) > 0


def by_sequence(gfile, features, *, strand: str | None = None) -> tuple[dict[str, Intervals], int]:
    """The features as :class:`Intervals`, by the name of the sequence the file holds them on (as the file finds
    names: ``1`` is ``chr1``); with ``strand`` ('+' or '-'), only the features on it ('.' counts as '+'). Returns
    them and how many features are on no sequence of the file."""
    names: dict[str, str | None] = {}
    pairs: dict[str, list] = {}
    unplaced = 0
    for f in features:
        if strand is not None and (f.strand == "-") != (strand == "-"):
            continue
        if f.contig not in names:
            names[f.contig] = sequence_name(gfile, f.contig)
        seq = names[f.contig]
        if seq is None:
            unplaced += 1
            continue
        pairs.setdefault(seq, []).append((f.start, f.end))
    return {k: Intervals(v) for k, v in pairs.items()}, unplaced


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
