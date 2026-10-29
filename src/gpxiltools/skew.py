"""Strand symmetry along a genome (``gpxil skew``): base composition and its skews in windows, and how far each
strand's k-mers are from their reverse complements' counts.

Chargaff's second parity rule: over long stretches of one strand, A is as
common as T and C as G, and each k-mer as its reverse complement. Where a
stretch breaks it, something has treated the strands differently:
replication (the skews change sign at origins), transcription (genes'
coding strands lean to G and T), repeats inserted in one direction (an Alu's
poly(A)), or runs of one strand's bases, as the GA-sequences are.

In windows of ``window`` bases, set at multiples of it along each sequence
(so files and regions line up), counted on the + strand from the
reference's A, C, G and T:

- ``gc``: (G + C) / (A + C + G + T);
- ``gc_skew``: (G - C) / (G + C); ``at_skew``: (A - T) / (A + T);
- ``ry_skew``: purines less pyrimidines, (A + G - C - T) / (A + C + G + T);
- ``km_skew``: keto less amino bases, (G + T - A - C) / (A + C + G + T);
- ``asymmetry_k`` for each k asked for: over each pair of a k-mer w and
  its reverse complement w' (palindromes, w = w', left out), how far their
  counts differ, over their count: sum |n(w) - n(w')| / sum (n(w) + n(w')),
  0 when every k-mer is as common as its reverse complement, 1 when no
  k-mer's reverse complement is there;
- ``cumulative_gc_skew``: the sum of ``gc_skew`` along the sequence, whose
  turning points are where replication's direction changes, in bacteria.

A k-mer is counted in the window of its first base, when all of its bases
count. ``unmasked``: soft-masked bases (repeats) don't count. ``features``:
only bases in them count, and with ``stranded`` each is read on its
feature's strand (so the skews of genes are those of their coding strands:
transcription's); a base under features on both strands counts on each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import codes as C
from .features import CHUNK, Intervals, by_sequence, targets

SKEWS = ("gc", "gc_skew", "at_skew", "ry_skew", "km_skew")
_BASE4 = np.full(256, -1, dtype=np.int8)  #: a reference code (soft-masked or not) -> 0..3 for A, C, G, T; -1 else
for _i, _b in enumerate((C.BASE_A, C.BASE_C, C.BASE_G, C.BASE_T)):
    _BASE4[_b] = _BASE4[_b | C.SOFTMASK] = _i


def _rc_table(k: int) -> np.ndarray:
    idx = np.arange(4**k)
    out = np.zeros(4**k, dtype=np.int64)
    for j in range(k):
        out += (3 - (idx // 4**j) % 4) * 4 ** (k - 1 - j)
    return out


@dataclass
class _Window:
    acgt: np.ndarray  #: A, C, G, T
    other: int  #: other bases (N, IUPAC codes for several)
    masked: int  #: soft-masked A, C, G, T (counted, unless ``unmasked``)
    kmers: dict  #: k -> counts of each k-mer


class SkewCounter:
    """Counts by window, chunk by chunk, handing windows back as they are done (see the module's notes)."""

    def __init__(self, window: int = 100_000, ks=(1, 2, 3, 4), *, unmasked: bool = False):
        if window < 1:
            raise ValueError("a window is 1 base or more")
        ks = sorted({int(k) for k in ks})
        if not ks or any(k < 1 or k > 6 for k in ks):
            raise ValueError("k-mers of 1 to 6 bases")
        self.window, self.ks, self.unmasked = window, ks, unmasked
        self.counted = sorted({1, *ks})  #: (1-mers: the composition)
        self.kmax = self.counted[-1]
        self.rc = {k: _rc_table(k) for k in self.counted}
        self.pending: dict[tuple[str, int], _Window] = {}  #: windows that may still be added to
        self.sums: dict[str, _Window] = {}  #: each sequence's counts, over all its windows
        self.lengths: dict[str, int] = {}  #: each sequence's length, where known (its last window ends there)
        self._cum: dict[str, float] = {}

    def _new(self) -> _Window:
        return _Window(np.zeros(4, dtype=np.int64), 0, 0, {k: np.zeros(4**k, dtype=np.int64) for k in self.ks})

    def add(self, seq: str, pos: int, codes: np.ndarray, n: int, plus: np.ndarray | None = None, minus: np.ndarray | None = None):
        """Bases ``codes`` of the sequence ``seq`` from position ``pos``: the first ``n`` counted, the rest (up to the
        longest k less one) there to finish k-mers. ``plus`` and ``minus`` (over all of ``codes``): the bases of
        features on each strand that count (None: every base, on the + strand)."""
        W = self.window
        b4 = _BASE4[codes]
        ok = b4 >= 0
        if self.unmasked:
            ok &= (codes & C.SOFTMASK) == 0
        first = pos // W
        win = (np.arange(n, dtype=np.int64) + (pos - first * W)) // W
        nwin = int(win[-1]) + 1 if n else 0
        strands = [(ok, False)] if plus is None else [(ok & plus, False), (ok & minus, True)]
        kmax = self.kmax
        size = codes.size
        kmers = {k: np.zeros((nwin, 4**k), dtype=np.int64) for k in self.counted}
        # (each position's longest k-mer, its first base the most significant: a shorter one is its first bases)
        v = np.maximum(b4, 0).astype(np.int64)
        longest = np.zeros(n, dtype=np.int64)
        for t in range(kmax):
            m = max(0, min(n, size - t))
            longest *= 4
            longest[:m] += v[t : t + m]
        for good, flip in strands:
            if not good.any():
                continue
            bad = np.concatenate(([0], np.cumsum(~good, dtype=np.int64)))
            whole = np.zeros(n, dtype=bool)  # (the longest k-mer from here counts: all its bases do)
            m = max(0, min(n, size - kmax + 1))
            whole[:m] = bad[kmax : kmax + m] == bad[:m]
            sel = np.flatnonzero(whole)
            top = np.bincount(win[sel] * 4**kmax + longest[sel], minlength=nwin * 4**kmax).reshape(nwin, 4**kmax)
            # (where it doesn't, shorter k-mers may: few positions, near a base that doesn't count)
            short = np.flatnonzero(good[:n] & ~whole)
            for k in self.counted:
                got = top.reshape(nwin, 4**k, 4 ** (kmax - k)).sum(axis=2) if k < kmax else top
                if short.size and k < kmax:
                    q = short[short + k <= size]
                    q = q[bad[q + k] == bad[q]]
                    if q.size:
                        got = got + np.bincount(win[q] * 4**k + (longest[q] >> (2 * (kmax - k))), minlength=nwin * 4**k).reshape(nwin, 4**k)
                kmers[k] += got[:, self.rc[k]] if flip else got
        acgt = kmers[1]
        nib = codes[:n] & 0x0F
        counted = np.ones(n, dtype=bool) if plus is None else (plus[:n] | minus[:n])
        other = np.bincount(win[np.flatnonzero(counted & (nib != 0) & (b4[:n] < 0))], minlength=nwin)
        masked = np.bincount(win[np.flatnonzero(counted & (b4[:n] >= 0) & ((codes[:n] & C.SOFTMASK) != 0))], minlength=nwin)
        total = self.sums.setdefault(seq, self._new())
        for i in range(nwin):
            if not (acgt[i].any() or other[i] or masked[i]):
                continue
            w = self.pending.get((seq, first + i))
            if w is None:
                w = self.pending[(seq, first + i)] = self._new()
            for x in (w, total):
                x.acgt += acgt[i]
                x.other += int(other[i])
                x.masked += int(masked[i])
                for k in self.ks:
                    x.kmers[k] += kmers[k][i]

    def finish(self, seq: str | None = None, before: int | None = None) -> list[dict]:
        """The rows of the windows that are done, in order: those of the sequence ``seq`` that end at or before
        ``before`` (None: all of them; ``seq`` None: every window still open)."""
        keys = sorted(
            (k for k in self.pending if (seq is None or k[0] == seq) and (before is None or (k[1] + 1) * self.window <= before)),
            key=lambda k: (list(self.sums).index(k[0]), k[1]),
        )
        out = []
        for key in keys:
            win = self.pending.pop(key)
            name, w = key
            end = (w + 1) * self.window
            d = {"sequence": name, "start": w * self.window, "end": min(end, self.lengths.get(name, end))}
            d.update(measures(win.acgt, win.kmers, self.rc))
            d["other"], d["soft_masked"] = win.other, win.masked
            cum = self._cum.get(name, 0.0) + (d["gc_skew"] or 0.0)
            self._cum[name] = cum
            d["cumulative_gc_skew"] = round(cum, 6)
            out.append(d)
        return out

    def totals(self) -> dict:
        """The measures over each sequence, and over them all."""
        out = {}
        acgt = np.zeros(4, dtype=np.int64)
        km = {k: np.zeros(4**k, dtype=np.int64) for k in self.ks}
        for seq, x in self.sums.items():
            out[seq] = measures(x.acgt, x.kmers, self.rc)
            acgt += x.acgt
            for k in self.ks:
                km[k] += x.kmers[k]
        if self.sums:
            out["all"] = measures(acgt, km, self.rc)
        return out


def measures(acgt: np.ndarray, kmers: dict, rc: dict) -> dict:
    """Composition, skews and asymmetries of counts: A, C, G, T, and k-mers (see the module's notes)."""
    a, c, g, t = (int(x) for x in acgt)
    n = a + c + g + t

    def ratio(x, y):
        return round(x / y, 6) if y else None

    d = {"bases": n, "A": a, "C": c, "G": g, "T": t}
    d["gc"] = ratio(g + c, n)
    d["gc_skew"] = ratio(g - c, g + c)
    d["at_skew"] = ratio(a - t, a + t)
    d["ry_skew"] = ratio(a + g - c - t, n)
    d["km_skew"] = ratio(g + t - a - c, n)
    for k, counts in sorted(kmers.items()):
        r = rc[k]
        pal = r == np.arange(r.size)
        total = int(counts[~pal].sum())  # (each pair's difference is counted twice below, once from each side)
        d[f"asymmetry_{k}"] = ratio(int(np.abs(counts - counts[r])[~pal].sum()), 2 * total)
    return d


def iter_skew(
    gfile,
    regions=None,
    *,
    window: int = 100_000,
    ks=(1, 2, 3, 4),
    unmasked: bool = False,
    features=None,
    stranded: bool = False,
    chunk: int = CHUNK,
    counter: SkewCounter | None = None,
):
    """The windows of a gpxil file's reference, as rows (dicts), in order along each sequence (see the module's
    notes): ``regions`` as ``GpxilFile.region_of`` gives them (default every sequence of the file); ``features``
    (:class:`~gpxiltools.features.Feature`) to count only bases in them. Give a :class:`SkewCounter` as
    ``counter`` for its ``totals()`` afterwards."""
    counter = counter or SkewCounter(window, ks, unmasked=unmasked)
    window, extra = counter.window, max(counter.ks) - 1
    plus = minus = None
    if features is not None:
        if stranded:
            plus, _ = by_sequence(gfile, features, strand="+")
            minus, _ = by_sequence(gfile, features, strand="-")
        else:
            plus, _ = by_sequence(gfile, features)
            minus = {}
    none = Intervals()
    # (what to scan, along each sequence in order, each base once, regions that overlap joined: so that a window is
    # done once the scan is past it)
    jobs = [(v.parent, v.start + s, v.start + e, v, s, e) for v, s, e in targets(gfile, regions)]
    # (chunks end at windows' ends; and hold few enough windows that their k-mers' counts stay small)
    step = max(window, min(chunk // window, max(1, (1 << 22) // 4**counter.kmax)) * window)
    last = None
    for seq, a0, _, view, s, e in jobs:
        if last is not None and last != seq:
            yield from counter.finish(last)
        yield from counter.finish(seq, before=a0)
        last = seq
        if not view.covers("ref", s, e):
            raise ValueError(f"{view.region(s, e)} has no reference: give it with --reference")
        if view.parent_length is not None:
            counter.lengths[seq] = int(view.parent_length)
        off = view.start
        lo = s
        while lo < e:
            hi = min(e, ((off + lo) // window) * window - off + step)  # (at a window's end, so windows are done whole)
            b = min(e, hi + extra)  # (bases past the chunk finish its last k-mers; but not past the region)
            codes = view.read("ref", lo, b)
            pl = mi = None
            if plus is not None:
                pl = plus.get(seq, none).mask(off + lo, off + b)
                mi = minus.get(seq, none).mask(off + lo, off + b)
            counter.add(seq, off + lo, codes, hi - lo, pl, mi)
            yield from counter.finish(seq, before=off + hi)
            lo = hi
    yield from counter.finish()


# ------------------------------------------------------------------ the picture

#: diverging, for skews: blue (negative), a grey nothing, red (positive)
_NEG, _MID, _POS = np.array([0x18, 0x4F, 0x95]), np.array([0xF0, 0xEF, 0xEC]), np.array([0xB3, 0x26, 0x1E])
#: sequential, for gc and the asymmetries: light to dark blue
_LO, _HI = np.array([0xCD, 0xE2, 0xFB]), np.array([0x0D, 0x36, 0x6B])
NO_VALUE = np.array([255, 255, 255])


def _diverging(x: np.ndarray, limit: float) -> np.ndarray:
    f = np.clip(x / limit, -1, 1)[..., None] if limit else np.zeros(x.shape + (1,))
    out = np.where(f < 0, _MID + (-f) * (_NEG - _MID), _MID + f * (_POS - _MID))
    return out


def _sequential(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    f = np.clip((x - lo) / (hi - lo), 0, 1)[..., None] if hi > lo else np.zeros(x.shape + (1,))
    return _LO + f * (_HI - _LO)


def scales(rows: list[dict], names) -> dict:
    """Each measure's scale: ``(kind, lo, hi)``, a skew's symmetric about 0 to the 99th percentile of its size."""
    out = {}
    for m in names:
        v = np.array([r[m] for r in rows if r.get(m) is not None], dtype=float)
        if m.endswith("skew") and m != "cumulative_gc_skew":
            lim = float(np.percentile(np.abs(v), 99)) if v.size else 0.0
            out[m] = ("diverging", -lim, lim)
        elif m == "cumulative_gc_skew":
            lim = float(np.abs(v).max()) if v.size else 0.0
            out[m] = ("diverging", -lim, lim)
        else:
            lo, hi = (float(np.percentile(v, 1)), float(np.percentile(v, 99))) if v.size else (0.0, 1.0)
            out[m] = ("sequential", lo, hi)
    return out


def picture(rows: list[dict], names, *, window: int, track: int = 12, title: str = ""):
    """A map of the windows: for each sequence, a track for each measure, a pixel a window along it; and a scale for
    each measure. A :class:`PIL.Image.Image`."""
    from PIL import Image, ImageDraw

    seqs = list(dict.fromkeys(r["sequence"] for r in rows))
    sc = scales(rows, names)
    ncol = max((r["start"] // window + 1 for r in rows), default=1)
    left, top, gap = 12 + 7 * max([len(s) for s in seqs] + [4]), 30, 6
    label_w = 7 * max(len(m) for m in names) + 16
    width = left + ncol + label_w + 12
    block = len(names) * track + gap
    legend_h = 30 * len(names) + 10
    height = top + len(seqs) * block + legend_h
    img = Image.new("RGB", (max(width, 380, 7 * len(title) + 24), height), (255, 255, 255))
    px = np.asarray(img).copy()
    for si, seq in enumerate(seqs):
        mine = [r for r in rows if r["sequence"] == seq]
        y0 = top + si * block
        for mi, m in enumerate(names):
            kind, lo, hi = sc[m]
            vals = np.full(ncol, np.nan)
            for r in mine:
                if r[m] is not None:
                    vals[r["start"] // window] = r[m]
            col = _diverging(np.nan_to_num(vals), hi) if kind == "diverging" else _sequential(np.nan_to_num(vals), lo, hi)
            col[np.isnan(vals)] = NO_VALUE
            y = y0 + mi * track
            px[y : y + track - 1, left : left + ncol] = np.round(col).astype(np.uint8)[None, :, :]
    img = Image.fromarray(px)
    d = ImageDraw.Draw(img)
    if title:
        d.text((12, 8), title, fill=(0, 0, 0))
    for si, seq in enumerate(seqs):
        y0 = top + si * block
        d.text((12, y0 + (len(names) * track) // 2 - 6), seq, fill=(0, 0, 0))
        for mi, m in enumerate(names):
            d.text((left + ncol + 8, y0 + mi * track), m, fill=(90, 90, 90))
    y = top + len(seqs) * block + 4
    for m in names:
        kind, lo, hi = sc[m]
        bar = np.linspace(lo, hi, 200)
        col = _diverging(bar, hi) if kind == "diverging" else _sequential(bar, lo, hi)
        px2 = np.asarray(img).copy()
        px2[y : y + 10, 12:212] = np.round(col).astype(np.uint8)[None, :, :]
        img = Image.fromarray(px2)
        d = ImageDraw.Draw(img)
        d.text((220, y - 1), f"{m}: {lo:.3g} to {hi:.3g}" + (" (0 grey)" if kind == "diverging" else ""), fill=(0, 0, 0))
        y += 30
    return img


def write_rows(out, rows: list[dict]) -> None:
    keys = list(rows[0]) if rows else ["sequence", "start", "end"]
    out.write("\t".join(keys) + "\n")
    for r in rows:
        out.write("\t".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")
