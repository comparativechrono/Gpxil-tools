"""Runs of a given composition: the longest stretches of the reference whose bases all belong to a set (``gpxil runs``).

A set of bases is given in IUPAC letters: ``AG`` (or ``R``) for purines, as
in Albrecht-Buehler's pure GA-sequences (PLoS ONE 2008: more than 50 bases,
``--min 51``); ``CT`` (``Y``), ``A`` for runs of one base, ``AT`` (``W``),
``GC`` (``S``)... A base belongs to the set when its own IUPAC set does (an
R in the reference belongs to ``AG``; an N to no set but ``ACGT``); soft
masking is ignored. A run is maximal: the bases either side of it don't
belong to the set, or there are none.

With ``both_strands`` the runs of the set on the other strand are found too:
on the + strand they are runs of the complementary set (a run of ``AG`` on
the - strand is one of ``CT`` on the +), reported with strand ``-``. A set
that is its own complement (``AT``, ``GC``, ``ACGT``) has the same runs on
both strands, reported once, with strand ``.``.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .features import CHUNK, COMPLEMENT_SET, base_set, set_name, targets


@dataclass
class Run:
    sequence: str  #: the sequence's name (a window's parent's)
    start: int  #: 0-based, in the sequence
    end: int
    strand: str  #: '+', '-' or '.'

    @property
    def length(self) -> int:
        return self.end - self.start


def _member(mask: int) -> np.ndarray:
    """Reference code -> whether its base belongs to the set ``mask`` (lookup table over all 256 codes)."""
    codes = np.arange(256)
    nib = codes & 0x0F
    return (nib != 0) & ((nib & ~mask & 0x0F) == 0)


class _Follow:
    """The runs of one set along a stretch, followed from chunk to chunk."""

    def __init__(self, member: np.ndarray, strand: str, min_length: int):
        self.member, self.strand, self.min_length = member, strand, min_length
        self.open_at: int | None = None

    def feed(self, codes: np.ndarray, s: int, last: bool) -> list[tuple[int, int, str]]:
        """The runs ended in the chunk ``codes`` that starts at ``s`` (``last``: the stretch ends with it)."""
        e = s + codes.size
        x = self.member[codes].view(np.int8)
        d = np.diff(np.concatenate(([0], x, [0])))
        starts = np.flatnonzero(d == 1) + s
        ends = np.flatnonzero(d == -1) + s
        out = []
        if self.open_at is not None:
            if starts.size and starts[0] == s:
                starts[0] = self.open_at  # (the run goes on from the chunk before)
            elif s - self.open_at >= self.min_length:
                out.append((self.open_at, s, self.strand))
            self.open_at = None
        if ends.size and ends[-1] == e and not last:
            self.open_at = int(starts[-1])  # (it may go on in the next chunk)
            starts, ends = starts[:-1], ends[:-1]
        keep = (ends - starts) >= self.min_length  # (most runs are short: left out before they become Python numbers)
        out += [(a, b, self.strand) for a, b in zip(starts[keep].tolist(), ends[keep].tolist())]
        return out


_SINGLE = np.zeros(256, dtype=bool)
for _code in (1, 2, 4, 8):
    _SINGLE[_code] = _SINGLE[_code | 0x10] = True


class RunFinder:
    """The runs of the set ``of`` (IUPAC letters) of ``min_length`` bases or more, in the regions (``(name, start,
    end)``, as ``GpxilFile.region_of`` gives them; default: every sequence of the file, each base once).
    :meth:`runs` yields them as :class:`Run`, in order along each sequence, a run cut where a region ends; after it,
    ``acgt`` is how many bases of the regions are A, C, G or T."""

    def __init__(self, gfile, regions=None, *, of: str = "AG", min_length: int = 51, both_strands: bool = False, chunk: int = CHUNK):
        if min_length < 1:
            raise ValueError("the minimum length must be 1 or more")
        self.gfile, self.regions, self.chunk = gfile, regions, chunk
        self.of, self.min_length = of, min_length
        mask = base_set(of)
        comp = int(COMPLEMENT_SET[mask])
        self.sets = [(mask, "+")]
        if both_strands:
            self.sets = [(mask, ".")] if comp == mask else [(mask, "+"), (comp, "-")]
        self.acgt = 0

    def runs(self) -> Iterator[Run]:
        self.acgt = 0
        for view, s, e in targets(self.gfile, self.regions):
            if not view.covers("ref", s, e):
                raise ValueError(f"{view.region(s, e)} has no reference: give it with --reference")
            follow = [_Follow(_member(m), strand, self.min_length) for m, strand in self.sets]
            pending: list[tuple[int, int, str]] = []  # (a heap: runs ended, waiting for any run still open before them)
            for a in range(s, e, self.chunk):
                b = min(e, a + self.chunk)
                codes = view.read("ref", a, b)
                self.acgt += int(_SINGLE[codes].sum())
                for f in follow:
                    for r in f.feed(codes, a, b == e):
                        heapq.heappush(pending, r)
                first_open = min((f.open_at for f in follow if f.open_at is not None), default=None)
                while pending and (first_open is None or pending[0][0] < first_open):
                    x, y, strand = heapq.heappop(pending)
                    yield Run(view.parent, view.start + x, view.start + y, strand)


def iter_runs(gfile, regions=None, *, of: str = "AG", min_length: int = 51, both_strands: bool = False, chunk: int = CHUNK):
    """The runs (:class:`Run`) of :class:`RunFinder`."""
    return RunFinder(gfile, regions, of=of, min_length=min_length, both_strands=both_strands, chunk=chunk).runs()


class Summary:
    """Counts of runs as they go by: how many on each strand, their bases and lengths."""

    def __init__(self):
        self.lengths: dict[str, list[int]] = {}

    def add(self, run: Run) -> None:
        self.lengths.setdefault(run.strand, []).append(run.length)

    def as_dict(self, of: str, min_length: int, acgt: int | None = None) -> dict:
        allv = np.array([x for v in self.lengths.values() for x in v], dtype=np.int64)
        d = {
            "of": set_name(base_set(of)),
            "min_length": min_length,
            "runs": int(allv.size),
            "by_strand": {k: len(v) for k, v in sorted(self.lengths.items())},
            "bases": int(allv.sum()),
            "longest": int(allv.max()) if allv.size else None,
            "length_quantiles": {q: float(np.percentile(allv, q)) for q in (50, 90, 99)} if allv.size else None,
        }
        if acgt is not None:
            d["acgt_bases"] = acgt
            d["per_mb"] = round(1e6 * allv.size / acgt, 3) if acgt else None
        return d


def write_bed(out, runs, summary: Summary | None = None, *, of: str = "AG") -> int:
    """Runs as BED6 (name: the set; score: the length, at most 1000; strand), to a text stream. Returns how many."""
    name = set_name(base_set(of))
    n = 0
    for r in runs:
        out.write(f"{r.sequence}\t{r.start}\t{r.end}\t{name}\t{min(r.length, 1000)}\t{r.strand}\n")
        if summary is not None:
            summary.add(r)
        n += 1
    return n


def format_summary(d: dict) -> str:
    strands = ", ".join(
        f"{n:,} on {'the + strand' if k == '+' else 'the - strand' if k == '-' else 'both strands'}" for k, n in d["by_strand"].items()
    )
    out = f"{d['runs']:,} runs of {d['of']} of {d['min_length']} bases or more" + (f" ({strands})" if strands else "")
    if d["runs"]:
        q = d["length_quantiles"]
        out += f"; {d['bases']:,} bases; lengths: median {q[50]:g}, 90 % {q[90]:g}, 99 % {q[99]:g}, longest {d['longest']:,}"
    if d.get("per_mb") is not None:
        out += f"; {d['per_mb']:g} a megabase of A, C, G and T"
    return out
