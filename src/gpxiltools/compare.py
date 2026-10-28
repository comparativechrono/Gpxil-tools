"""Genomes compared base by base: two gpxil files, or a child's with both parents' (a trio).

Files on the same reference have their planes aligned to it base for base, so
genomes are compared array by array, without aligning sequences or matching
VCF records.

An *allele* here is what one haplotype has at a base: its state (the
reference, another base, deleted, COMPLEX, ...) together with the bases
inserted after it and the data of a COMPLEX allele. A base is *called* in a
file where every haplotype present is known, as in :mod:`gpxiltools.stats`,
leaving out the calls a file written with callable regions keeps outside them
(spec 6.1): runs of called bases among bases not called, every base of them
covered by a call (``all_calls`` keeps them).

**Two genomes** (``gpxil compare A B``). At each base called in both, the
genotypes are compared as unordered pairs of alleles, so A/G equals G/A,
phased or not, and one copy (haplotype 2 ABSENT) equals two copies of the
same allele. Bases where the two differ that are no more than ``gap`` bases
(50) apart form one stretch, which takes in the whole of any deletion it
touches and the whole of any repeat in which its insertions and deletions
could be written as well. Where the two files could spell the same pair of
haplotype sequences over a stretch, they differ only in how the variants are
written (an insertion anchored at the other end of a repeat, say), and the
stretch counts as the same. Phasing isn't compared, so a file's heterozygous
pieces are tried either way round: runs of bases that differ in one way (a
deletion on one haplotype, or another base), and insertions apart from the
base they follow. Each file's events, counted as ``gpxil stats`` counts them
(a substitution at a base, a run of deleted bases, an insertion, a COMPLEX
allele), are then looked for in the other: an event is *found* if every base
it spans is the same in both, or written differently; otherwise it is
*missing* from the other file, or the other file has *another genotype*
there. Events over bases the other file didn't call are *not compared*. With
B taken as the truth, precision is the share of A's compared events found in
B, and recall the share of B's found in A.

**A child and both parents** (``gpxil compare CHILD --father F --mother M``).
At each base called in all three, the child's alleles must be one from each
parent (or with one copy, as on a son's X, from either). Where they aren't,
the stretch around the error, with every variant of the three near it, is
spelled out: if the child's two sequences could be one of the father's and
one of the mother's, the three write it differently. Otherwise the base is a
*Mendelian error*, and a *candidate new mutation* if the child has an allele
neither parent has. Where the child carries a variant and only one way of
taking an allele from each parent fits, its *parent of origin* is known: the
father, the mother, or both (a homozygous variant). Where a parent's calls
are phased, the child's genotype often shows which of that parent's two
haplotypes it received; a change from one to the other along a phase set is
a *crossover* in that parent's meiosis, or an error in the phasing or the
calls, so changes not backed by at least three sites in a row on each side
are left out.

Files are read a piece at a time (a megabase), each piece cut where no file
has a variant within ``gap`` bases, so nothing compared is cut in two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import codes as C

CHUNK = 1 << 20  #: bases read at a time (each piece cut where no file has a variant nearby)
GAP = 50  #: bases where two files differ no more than this far apart are one stretch
MAX_PIECES = 12  #: heterozygous pieces of a stretch tried either way round; a stretch with more is taken as written
MIN_RUN = 3  #: sites in a row from one of a parent's haplotypes, on each side of a crossover
EXAMPLES = 20  #: differences, or Mendelian errors, described
FADE = 0.9  #: how far a picture's reference is faded to white, behind the classes

_IS_SUB = np.zeros(16, dtype=bool)
_IS_SUB[list(C.HAP_SUB_CODES)] = True
_IS_VAR = _IS_SUB.copy()
_IS_VAR[[C.HAP_DEL, C.HAP_COMPLEX]] = True
_ABSENT = C.HAP_ABSENT
_INS = C.FLAG_INS1 | C.FLAG_INS2
# a gt byte's two haplotype states, reserved states read as UNKNOWN (spec, reader rule 4)
_H1 = C.normalize_states(np.arange(256, dtype=np.uint8) >> 4)
_H2 = C.normalize_states(np.arange(256, dtype=np.uint8) & 0x0F)
_GT_VAR = _IS_VAR[_H1] | _IS_VAR[_H2]
_KEY = np.int32  #: allele keys: state | id << 4
_MAX_IDS = 1 << 27


# ---------------------------------------------------------------- the files, as alleles


class _Ids:
    """Numbers for what 4 bits of state can't hold: the bases inserted after a base, and a COMPLEX allele's data.

    Shared by the files compared, so the same allele has the same key in each.
    """

    def __init__(self) -> None:
        self._ids: dict = {}
        self.items: list = [(None, None)]

    def __call__(self, ins: str | None, cplx: str | None) -> int:
        k = (ins, cplx)
        i = self._ids.get(k)
        if i is None:
            if len(self.items) >= _MAX_IDS:
                raise ValueError("too many different insertions and COMPLEX alleles to compare")
            i = self._ids[k] = len(self.items)
            self.items.append(k)
        return i


@dataclass
class _Chunk:
    """One file over [s, e): states, flags, allele keys (state | id << 4), and what is called, a variant, phased."""

    s: int
    e: int
    h1: np.ndarray
    h2: np.ndarray
    flags: np.ndarray
    k1: np.ndarray
    k2: np.ndarray
    called: np.ndarray
    var: np.ndarray
    phased: np.ndarray
    ps: np.ndarray | None

    def cut(self, s: int, e: int) -> _Chunk:
        a, b = s - self.s, e - self.s

        def part(x):
            return None if x is None else x[a:b]

        return _Chunk(s, e, *(part(getattr(self, n)) for n in ("h1", "h2", "flags", "k1", "k2", "called", "var", "phased", "ps")))


class _Member:
    """One file's contig, read a piece at a time as allele keys."""

    def __init__(self, contig, ids: _Ids, *, want_ps: bool = False, confident: bool = False):
        self.c = contig
        self.ids = ids
        self.confident = confident  # (calls outside the regions called with confidence left out)
        self.has_gt = contig.has("gt")
        self.has_flags = contig.has("flags")
        self.want_ps = want_ps and contig.has("ps")
        self.ins: dict[tuple[int, int], str] = {}
        self.cplx: dict[tuple[int, int], str] = {}
        for ent in contig.sidecar() if self.has_gt else ():
            if ent.kind == "INS":
                self.ins[(ent.offset, ent.hap)] = ent.data
            elif ent.kind == "COMPLEX":
                self.cplx[(ent.offset, ent.hap)] = ent.data
        self._extra = []  # per haplotype: the offsets with an entry, sorted, and their (inserted bases, COMPLEX data)
        for hap in (1, 2):
            offs = sorted({o for o, h in self.ins if h == hap} | {o for o, h in self.cplx if h == hap})
            self._extra.append((np.array(offs, dtype=np.int64), [(self.ins.get((o, hap)), self.cplx.get((o, hap))) for o in offs]))
        self.ins_at = np.array(sorted({o for o, _ in self.ins}), dtype=np.int64)
        self.cplx_at = np.array(sorted({o for o, _ in self.cplx}), dtype=np.int64)

    def read(self, s: int, e: int) -> _Chunk:
        n = e - s
        if not self.has_gt:  # (no calls: nothing called)
            unknown = np.full(n, C.HAP_UNKNOWN, dtype=np.uint8)
            key = unknown.astype(_KEY)
            no = np.zeros(n, dtype=bool)
            return _Chunk(s, e, unknown, unknown, np.zeros(n, dtype=np.uint8), key, key, no, no, no, None)
        gt = self.c.read("gt", s, e)
        h1, h2 = np.take(_H1, gt), np.take(_H2, gt)
        flags = self.c.read("flags", s, e) if self.has_flags else np.zeros(n, dtype=np.uint8)
        k1, k2 = h1.astype(_KEY), h2.astype(_KEY)
        for (offs, extras), key, h, bit in zip(self._extra, (k1, k2), (h1, h2), (C.FLAG_INS1, C.FLAG_INS2)):
            a, b = np.searchsorted(offs, (s, e)).tolist()
            for o, (ins, cplx) in zip(offs[a:b].tolist(), extras[a:b]):
                i = o - s
                ins = ins if flags[i] & bit else None
                cplx = cplx if h[i] == C.HAP_COMPLEX else None
                if ins is not None or cplx is not None:
                    key[i] |= self.ids(ins, cplx) << 4
        called = (h1 != C.HAP_UNKNOWN) & (h2 != C.HAP_UNKNOWN)
        if self.confident:
            _drop_islands(called, (flags & C.FLAG_VARIANT_SITE) != 0)
        var = np.take(_GT_VAR, gt) | ((flags & _INS) != 0)
        phased = (flags & C.FLAG_PHASED) != 0
        ps = self.c.read("ps", s, e) if self.want_ps and phased.any() else None
        return _Chunk(s, e, h1, h2, flags, k1, k2, called, var, phased, ps)

    def events(self, ch: _Chunk, lo: int, hi: int):
        """The events that start in [lo, hi), as gpxil stats counts them: arrays of kinds (0 a substitution, 1 an
        insertion or deletion, 2 a COMPLEX allele), starts and ends. A homozygous event is one event."""
        s = ch.s
        a, b = lo - s, hi - s
        kinds, starts, ends = [], [], []

        def add(kind, x, y):
            kinds.append(np.full(len(x), kind, dtype=np.int8))
            starts.append(np.asarray(x, dtype=np.int64))
            ends.append(np.asarray(y, dtype=np.int64))

        h1, h2 = ch.h1[a:b], ch.h2[a:b]
        sub = np.flatnonzero((np.take(_IS_SUB, h1) | np.take(_IS_SUB, h2)) & (h1 != C.HAP_UNKNOWN) & (h2 != C.HAP_UNKNOWN)) + lo
        add(0, sub, sub + 1)
        runs = set()  # (the same run on both haplotypes, once)
        for h in (ch.h1, ch.h2):
            x, y = _runs_of(h == C.HAP_DEL)
            keep = (x >= a) & (x < b)
            runs.update(zip((x[keep] + s).tolist(), (y[keep] + s).tolist()))
        runs = sorted(runs)
        add(1, [x for x, _ in runs], [y for _, y in runs])
        for kind, at, table in ((1, self.ins_at, self.ins), (2, self.cplx_at, self.cplx)):
            i, j = np.searchsorted(at, (lo, hi)).tolist()
            pos = []
            for o in at[i:j].tolist():
                got = [table[(o, hap)] for hap in (1, 2) if (o, hap) in table]
                pos += [o] * (1 if len(got) == 2 and got[0] == got[1] else len(got))
            pos = np.array(pos, dtype=np.int64)
            add(kind, pos, pos + 1)
        return np.concatenate(kinds), np.concatenate(starts), np.concatenate(ends)


def _runs_of(mask: np.ndarray):
    """The starts and ends of the runs of True."""
    if not mask.size:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    d = np.diff(mask.view(np.int8))
    starts, ends = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [mask.size]))
    return starts, ends


def _drop_islands(called: np.ndarray, site: np.ndarray) -> None:
    """Leave out calls outside the regions called with confidence: runs of called bases between bases not called,
    every base of them covered by a call (VARIANT_SITE). A file written with callable regions keeps the calls outside
    them (spec 6.1), each an island among bases not called."""
    starts, ends = _runs_of(called)
    if not starts.size:
        return
    # (each run and the bases not called after it: all covered by a call or not called, so the run is all covered)
    island = np.minimum.reduceat((site | ~called).view(np.int8), starts) == 1
    for x, y in zip(starts[island].tolist(), ends[island].tolist()):
        called[x:y] = False


def _quiet(noisy: np.ndarray, a: int, b: int, xs: np.ndarray, gap: int, length: int) -> np.ndarray:
    """For each x in ``xs``: whether a piece can be cut at x, with no variant in [x - gap, x + gap). ``noisy``
    covers [a, b); the ends of the contig are always cuts."""
    cs = np.concatenate(([0], np.cumsum(noisy, dtype=np.int32)))
    lo = np.maximum(a, xs - gap)
    hi = np.minimum(b, xs + gap)
    seen = ((xs - gap >= a) | (a == 0)) & ((xs + gap <= b) | (b == length))
    return ((cs[hi - a] - cs[lo - a] == 0) & seen) | (xs == 0) | (xs == length)


def _noisy(members, a: int, b: int) -> np.ndarray:
    noisy = np.zeros(b - a, dtype=bool)
    for m in members:
        noisy |= m.read(a, b).var
    return noisy


def _pieces(members, start: int, end: int, length: int, chunk: int, gap: int):
    """[start, end) read a piece at a time, each piece cut where no file has a variant within ``gap`` bases, so
    nothing compared is cut in two: (s, e, the files' chunks over [s, e)). The first piece may begin before
    ``start``, and the last end after ``end``."""
    s, look = start, 16 * gap
    while s > 0:  # (the first cut, at or before start)
        a, b = max(0, s - look), min(length, start + gap)
        xs = np.arange(a + gap if a else 0, start + 1)
        ok = np.flatnonzero(_quiet(_noisy(members, a, b), a, b, xs, gap, length))
        if ok.size:
            s = int(xs[ok[-1]])
            break
        look *= 4
    while s < end:
        target, look = min(end, max(s + chunk, start + 1)), 4 * gap
        while True:
            a, b = max(0, s - gap), min(length, target + look)
            chunks = [m.read(a, b) for m in members]
            noisy = np.zeros(b - a, dtype=bool)
            for ch in chunks:
                noisy |= ch.var
            xs = np.arange(target, b + 1)
            near = max(a, target - gap)  # (only the bases near the candidates)
            ok = np.flatnonzero(_quiet(noisy[near - a :], near, b, xs, gap, length))
            if ok.size:
                x = int(xs[ok[0]])
                break
            look *= 4
        yield s, x, [ch.cut(s, x) for ch in chunks]
        s = x


def _check_same_reference(contigs) -> None:
    first = contigs[0]
    for c in contigs[1:]:
        if c.length != first.length:
            raise ValueError(f"{c.name} has {c.length:,} bases in one file and {first.length:,} in another: not the same reference")
        for k in ("md5", "refget"):
            if k in c.entry and k in first.entry and c.entry[k] != first.entry[k]:
                raise ValueError(f"{c.name} has another {k} in one file than in another: not the same reference")


def _ref_source(contigs):
    """A contig with the reference's bases (a ref plane, or a reference attached), or None."""
    return next((c for c in contigs if c.has("ref")), None)


def _ref_reader(ref, s: int, e: int):
    """The reference's bases over [s, e) (4-bit codes), read when first asked for; None without a reference."""
    got: list = []

    def read():
        if ref is None:
            return None
        if not got:
            got.append(np.bitwise_and(ref.read("ref", s, e), 0x0F))
        return got[0]

    return read


def _common_regions(gfiles, regions):
    """(the contig's name in each file, start, end) of the regions asked for, or of every contig all the files have."""
    if regions:
        out = []
        for name, s, e in regions:
            cs = [g[name] for g in gfiles]
            e = cs[0].length if e is None else e
            if not (0 <= s < e <= cs[0].length):
                raise IndexError(f"{name}:{s + 1}-{e} is outside {cs[0].name} (length {cs[0].length:,})")
            out.append(([c.name for c in cs], s, e))
        return out
    out = []
    for c in gfiles[0]:
        names = [c.name]
        for g in gfiles[1:]:
            hit = next((n for n in (c.name, *c.entry.get("aliases", [])) if n in g), None)
            if hit is None:
                break
            names.append(g[hit].name)
        else:
            out.append((names, 0, c.length))
    if not out:
        raise ValueError("the files have no contig in common")
    return out


def _confident(g, all_calls: bool) -> bool:
    """Whether to leave out a file's calls outside its regions called with confidence: it was written with them."""
    s = g.metadata.get("sample")
    return not all_calls and bool(isinstance(s, dict) and s.get("callable_regions"))


def _need_samples(named) -> None:
    for who, g in named:
        if not g.sample_id:
            raise ValueError(f"{who}, {g.path}, has no sample: only a reference")


def _name(g) -> str:
    return str(g.path).rstrip("/").rsplit("/", 1)[-1]


def _runs(mask: np.ndarray) -> int:
    """How many runs of True."""
    return int(mask[:1].sum() + (mask[1:] & ~mask[:-1]).sum())


# ---------------------------------------------------------------- two genomes

_KINDS = ("substitutions", "indels", "complex")

#: the classes of a picture of two genomes (index 0xF0 + class), with sRGB and CMYK colours
PAIR_CLASSES = [
    ("the same, or no variant (the reference, faded)", None),
    ("a variant in both, the same", ((0, 150, 70), (0.85, 0.0, 0.9, 0.1))),
    ("only in A", ((220, 30, 30), (0.0, 0.9, 0.9, 0.05))),
    ("only in B", ((30, 90, 220), (0.9, 0.6, 0.0, 0.0))),
    ("in both, another genotype", ((255, 150, 0), (0.0, 0.45, 1.0, 0.0))),
    ("the same, written differently", ((0, 175, 195), (0.8, 0.0, 0.25, 0.0))),
    ("not called in both", ((185, 185, 185), (0.0, 0.0, 0.0, 0.3))),
]


def _event_counts() -> dict:
    return {k: {side: dict(total=0, found=0, missing=0, other_genotype=0, not_compared=0) for side in "ab"} for k in _KINDS}


@dataclass
class PairTally:
    """Counts of two genomes compared, over one or more regions. Tallies add up (``a + b``)."""

    bases: int = 0
    both_called: int = 0
    a_only_called: int = 0
    b_only_called: int = 0
    neither_called: int = 0
    var_same: int = 0
    var_written_differently: int = 0
    var_only_a: int = 0
    var_only_b: int = 0
    var_different: int = 0
    stretches_written_differently: int = 0
    stretches_different: int = 0
    stretches_unchecked: int = 0  #: (no reference to spell them with)
    events: dict = field(default_factory=_event_counts)
    examples: list = field(default_factory=list)  #: the first stretches where the two differ

    def __add__(self, other: PairTally) -> PairTally:
        out = PairTally()
        for name in self.__dataclass_fields__:
            if name == "examples":
                out.examples = (self.examples + other.examples)[:EXAMPLES]
            elif name != "events":
                setattr(out, name, getattr(self, name) + getattr(other, name))
        for k in _KINDS:
            for side in "ab":
                out.events[k][side] = {x: self.events[k][side][x] + other.events[k][side][x] for x in self.events[k][side]}
        return out

    def as_dict(self) -> dict:
        def share(n, d):
            return round(n / d, 6) if d else None

        events = {}
        for k in _KINDS:
            a, b = self.events[k]["a"], self.events[k]["b"]
            p = share(a["found"], a["found"] + a["missing"] + a["other_genotype"])
            r = share(b["found"], b["found"] + b["missing"] + b["other_genotype"])
            f1 = round(2 * p * r / (p + r), 6) if p is not None and r is not None and p + r else None
            events[k] = {"a": dict(a), "b": dict(b), "precision": p, "recall": r, "f1": f1}
        either = self.var_same + self.var_written_differently + self.var_only_a + self.var_only_b + self.var_different
        return {
            "bases": self.bases,
            "called": {
                "both": self.both_called,
                "a_only": self.a_only_called,
                "b_only": self.b_only_called,
                "neither": self.neither_called,
            },
            "variant_bases": {
                "either": either,
                "same": self.var_same,
                "written_differently": self.var_written_differently,
                "only_a": self.var_only_a,
                "only_b": self.var_only_b,
                "other_genotype": self.var_different,
                "agreement": share(self.var_same + self.var_written_differently, either),
            },
            "stretches": {
                "written_differently": self.stretches_written_differently,
                "different": self.stretches_different,
                "not_checked": self.stretches_unchecked,
            },
            "events": events,
            "examples": self.examples,
        }


def _parts(keys: np.ndarray, ref4: np.ndarray, items: list) -> tuple[list[str], list[str]]:
    """One haplotype over a stretch, base by base: what each base spells (nothing where deleted or absent), and the
    bases inserted after it."""
    bases, inserted = [], []
    for x, r in zip(keys.tolist(), ref4.tolist()):
        st, extra = x & 0xF, x >> 4
        ins, cplx = items[extra]
        if st == C.HAP_REF:
            bases.append(C.IUPAC[r])
        elif _IS_SUB[st]:
            bases.append(C.IUPAC[st])
        elif st == C.HAP_COMPLEX:
            bases.append(f"<{cplx or ''}>")
        elif st == C.HAP_UNKNOWN:
            bases.append("?")
        else:
            bases.append("")
        inserted.append(ins or "")
    return bases, inserted


def _spell(keys: np.ndarray, ref4: np.ndarray, items: list) -> str:
    """One haplotype's sequence over a stretch, from its allele keys."""
    bases, inserted = _parts(keys, ref4, items)
    return "".join(x + y for x, y in zip(bases, inserted))


# kinds of state, for cutting a stretch into pieces: the reference, another base, deleted, COMPLEX, absent, unknown
_KIND = np.full(16, 6, dtype=np.int8)
_KIND[C.HAP_REF], _KIND[list(C.HAP_SUB_CODES)], _KIND[C.HAP_DEL], _KIND[C.HAP_COMPLEX], _KIND[C.HAP_ABSENT] = 0, 1, 2, 3, 4


def _haplotype_pairs(k1: np.ndarray, k2: np.ndarray, ref4: np.ndarray, items: list) -> set:
    """The pairs of sequences a file could spell over a stretch: its heterozygous pieces taken either way round, as
    phasing isn't compared. A piece is a run of bases where the two haplotypes differ in the same way (a deletion on
    one, say, or another base), or an insertion on one haplotype and not the other: so records next to each other,
    or at one base, can be taken apart, as unphased calls allow."""
    b1, i1 = _parts(k1, ref4, items)
    b2, i2 = _parts(k2, ref4, items)
    n = len(b1)
    kind = (_KIND[k1 & 0xF] * 8 + _KIND[k2 & 0xF]).tolist()
    own_base, own_ins = [-1] * n, [-1] * n  # (the piece each base, and each insertion, belongs to)
    pieces = 0
    for j in range(n):
        if b1[j] != b2[j]:
            if j and own_base[j - 1] >= 0 and kind[j - 1] == kind[j]:
                own_base[j] = own_base[j - 1]
            else:
                own_base[j], pieces = pieces, pieces + 1
        if i1[j] != i2[j]:
            own_ins[j], pieces = pieces, pieces + 1
    if pieces > MAX_PIECES:  # (too many to try every way round: as written)
        own_base, own_ins, pieces = [-1] * n, [-1] * n, 0
    out = set()
    for mask in range(1 << max(0, pieces - 1)):  # (the first piece stays: turning all round is the same pair)
        turn = [False] + [bool(mask >> j & 1) for j in range(pieces - 1)]
        one, two = [], []
        for j in range(n):
            tb = own_base[j] >= 0 and turn[own_base[j]]
            ti = own_ins[j] >= 0 and turn[own_ins[j]]
            one.append((b2[j] if tb else b1[j]) + (i2[j] if ti else i1[j]))
            two.append((b1[j] if tb else b2[j]) + (i1[j] if ti else i2[j]))
        x, y = "".join(one), "".join(two)
        out.add((x, y) if x <= y else (y, x))
    return out


_CODE = {c: i for i, c in enumerate(C.IUPAC)}


def _slide(ref4: np.ndarray, x: int, y: int, ins: str | None) -> tuple[int, int]:
    """The bases over which an insertion (of ``ins`` after base x) or a deletion (of [x, y)) could be written as well,
    moved along a repeat of the reference: [start, end)."""
    n = len(ref4)
    if ins is not None:
        s = [_CODE.get(b, 0) for b in ins.upper()]
        m = len(s)
        r = 0
        while x + 1 + r < n and ref4[x + 1 + r] == s[r % m]:
            r += 1
        left = 0
        while x - left >= 0 and ref4[x - left] == s[(m - 1 - left) % m]:
            left += 1
        return x - left, x + r + 1
    r = 0
    while y + r < n and ref4[x + r] == ref4[y + r]:
        r += 1
    left = 0
    while x - 1 - left >= 0 and ref4[x - 1 - left] == ref4[y - 1 - left]:
        left += 1
    return x - left, y + r


def _widened(windows, chunks, ref4: np.ndarray, items, gap: int, within: np.ndarray) -> list[list[int]]:
    """Windows widened over the repeats in which their insertions and deletions (in any of the files) could be written
    as well, but not beyond the bases ``within`` (called in all) around them; then merged where they come within
    ``gap`` bases of each other. So an indel written at the two ends of a long repeat is still one stretch."""
    n = len(ref4)
    out: list[list[int]] = []
    outside = np.flatnonzero(~within)
    for w0, w1 in windows:
        i = int(np.searchsorted(outside, w0))
        floor = int(outside[i - 1]) + 1 if i else 0
        j = int(np.searchsorted(outside, w1))
        ceil = int(outside[j]) if j < outside.size else n
        while True:
            a, b = w0, w1
            for ch in chunks:
                for h, k, bit in ((ch.h1, ch.k1, C.FLAG_INS1), (ch.h2, ch.k2, C.FLAG_INS2)):
                    xs, ys = _runs_of(h[w0:w1] == C.HAP_DEL)
                    found = [(x, y, None) for x, y in zip((xs + w0).tolist(), (ys + w0).tolist())]
                    for q in (np.flatnonzero(ch.flags[w0:w1] & bit) + w0).tolist():
                        ins = items[int(k[q]) >> 4][0]
                        if ins:
                            found.append((q, q + 1, ins))
                    for x, y, ins in found:
                        lo, hi = _slide(ref4, x, y, ins)
                        a, b = min(a, lo), max(b, hi)
            a, b = min(w0, max(a, floor)), max(w1, min(b, ceil))
            if (a, b) == (w0, w1):
                break
            w0, w1 = a, b
        out.append([w0, w1])
    out.sort()
    merged: list[list[int]] = []
    for w0, w1 in out:
        if merged and w0 - (merged[-1][1] - 1) <= gap:
            merged[-1][1] = max(merged[-1][1], w1)
        else:
            merged.append([w0, w1])
    return merged


def _two_copies(ch: _Chunk):
    """One copy (the other haplotype ABSENT) taken as two copies of its allele: how genotypes are compared."""
    k1, k2 = ch.k1, ch.k2
    a1, a2 = ch.h1 == _ABSENT, ch.h2 == _ABSENT
    if not (a1.any() or a2.any()):
        return k1, k2
    return np.where(a1 & ~a2, k2, k1), np.where(a2 & ~a1, k1, k2)


def _pair_piece(ma: _Member, mb: _Member, A: _Chunk, B: _Chunk, items, ref4_of, lo, hi, gap, t: PairTally, classes=None, collect=None):
    """Compare a piece (cut where neither file has a variant nearby) and count what is in [lo, hi)."""
    s, n = A.s, A.e - A.s
    a1, a2 = _two_copies(A)
    b1, b2 = _two_copies(B)
    both = A.called & B.called
    same = ((a1 == b1) & (a2 == b2)) | ((a1 == b2) & (a2 == b1))
    equiv = np.zeros(n, dtype=bool)
    pos = np.flatnonzero(both & ~same)
    if pos.size:
        deleted = (A.h1 == C.HAP_DEL) | (A.h2 == C.HAP_DEL) | (B.h1 == C.HAP_DEL) | (B.h2 == C.HAP_DEL)
        stretches: list[list[int]] = []
        for grp in np.split(pos, np.flatnonzero(np.diff(pos) > gap) + 1):
            w0, w1 = int(grp[0]), int(grp[-1]) + 1
            while w0 > 0 and deleted[w0 - 1]:
                w0 -= 1
            while w1 < n and deleted[w1]:
                w1 += 1
            if stretches and w0 - (stretches[-1][1] - 1) <= gap:  # (no more than gap from the last base of the one before)
                stretches[-1][1] = max(stretches[-1][1], w1)
            else:
                stretches.append([w0, w1])
        ref4 = ref4_of()
        if ref4 is not None:
            stretches = _widened(stretches, (A, B), ref4, items, gap, both)
        for w0, w1 in stretches:
            counted = lo <= s + w0 < hi
            if ref4 is None:
                t.stretches_unchecked += counted
                alike = False
            elif _haplotype_pairs(a1[w0:w1], a2[w0:w1], ref4[w0:w1], items) & _haplotype_pairs(b1[w0:w1], b2[w0:w1], ref4[w0:w1], items):
                equiv[w0:w1] = True
                t.stretches_written_differently += counted
                alike = True
            else:
                t.stretches_different += counted
                alike = False
            if counted and collect is not None:
                collect.append((ma.c.name, s + w0 + 1, s + w1, alike))
            if counted and not alike and len(t.examples) < EXAMPLES:
                j = w0 + int(np.argmax(both[w0:w1] & ~same[w0:w1]))  # (the first base where they differ)
                r = int(ref4[j]) if ref4 is not None else None
                t.examples.append(
                    {"position": s + w0 + 1, "end": s + w1, "a": _genotype_text(A, j, r, items), "b": _genotype_text(B, j, r, items)}
                )
    a, b = lo - s, hi - s
    w = slice(a, b)
    ca, cb, bo = A.called[w], B.called[w], both[w]
    t.bases += b - a
    t.both_called += int(bo.sum())
    t.a_only_called += int((ca & ~cb).sum())
    t.b_only_called += int((~ca & cb).sum())
    t.neither_called += int((~ca & ~cb).sum())
    va, vb, sm, eq = A.var[w], B.var[w], same[w], equiv[w]
    anyv = bo & (va | vb)
    differ = anyv & ~sm & ~eq
    t.var_same += int((anyv & sm).sum())
    t.var_written_differently += int((anyv & ~sm & eq).sum())
    t.var_only_a += int((differ & va & ~vb).sum())
    t.var_only_b += int((differ & vb & ~va).sum())
    t.var_different += int((differ & va & vb).sum())
    if classes is not None:
        cls = np.zeros(b - a, dtype=np.uint8)
        cls[anyv & sm] = 1
        cls[differ & va & ~vb] = 2
        cls[differ & vb & ~va] = 3
        cls[differ & va & vb] = 4
        cls[anyv & ~sm & eq] = 5
        cls[~bo] = 6
        classes.append(cls)
    # each file's events, looked for in the other
    ok = same | equiv
    for me, ch, other, side in ((ma, A, B, "a"), (mb, B, A, "b")):
        kinds, x, y = me.events(ch, lo, hi)
        x, y = x - s, y - s
        one = y - x == 1
        not_compared, found, there = ~both[x], ok[x], other.var[x]  # (right for the events on one base)
        for i in np.flatnonzero(~one).tolist():  # (deletions: every base they span)
            sl = slice(x[i], y[i])
            not_compared[i], found[i], there[i] = not both[sl].all(), ok[sl].all(), other.var[sl].any()
        found &= ~not_compared
        other_gt = ~not_compared & ~found & there
        missing = ~not_compared & ~found & ~other_gt
        for k, name in enumerate(_KINDS):
            sel = kinds == k
            e = t.events[name][side]
            e["total"] += int(sel.sum())
            e["found"] += int((sel & found).sum())
            e["missing"] += int((sel & missing).sum())
            e["other_genotype"] += int((sel & other_gt).sum())
            e["not_compared"] += int((sel & not_compared).sum())


def compare_pair(a, b, regions=None, *, chunk: int = CHUNK, gap: int = GAP, all_calls: bool = False, stretches: list | None = None) -> dict:
    """Two open gpxil files compared, B taken as the truth for precision and recall: per region and in all.

    ``regions`` are (contig, start, end), 0-based with the end not included (None for the contig's end); by
    default every contig both files have. A file written with callable regions also holds the calls outside them;
    they aren't compared unless ``all_calls``. ``stretches``, a list, collects every stretch where the two differ
    base by base: (contig, start, end (1-based, inclusive), whether they are the same written differently).
    """
    _need_samples((("A", a), ("B", b)))
    ids = _Ids()
    out, total = [], PairTally()
    for (na, nb), s, e in _common_regions([a, b], regions):
        ca, cb = a[na], b[nb]
        _check_same_reference([ca, cb])
        t = PairTally()
        ma, mb = _Member(ca, ids, confident=_confident(a, all_calls)), _Member(cb, ids, confident=_confident(b, all_calls))
        ref = _ref_source([ca, cb])
        for s0, e0, (A, B) in _pieces([ma, mb], s, e, ca.length, chunk, gap):
            lo, hi = max(s, s0), min(e, e0)
            if lo < hi:
                _pair_piece(ma, mb, A, B, ids.items, _ref_reader(ref, s0, e0), lo, hi, gap, t, collect=stretches)
        out.append({"contig": ca.name, "start": s + 1, "end": e, **t.as_dict()})
        total = total + t
    return {
        "mode": "pair",
        "a": {"file": _name(a), "sample": a.sample_id},
        "b": {"file": _name(b), "sample": b.sample_id},
        "reference": (a.metadata.get("reference") or {}).get("name"),
        "gap": gap,
        "all_calls": all_calls,
        "regions": out,
        "total": {"regions": len(out), **total.as_dict()},
    }


# ---------------------------------------------------------------- a child and both parents

#: the classes of a picture of a trio (index 0xF0 + class), with sRGB and CMYK colours
TRIO_CLASSES = [
    ("no variant in the child (the reference, faded)", None),
    ("from the father", ((40, 90, 220), (0.85, 0.6, 0.0, 0.0))),
    ("from the father's haplotype 2", ((120, 170, 250), (0.45, 0.2, 0.0, 0.0))),
    ("from the mother", ((225, 50, 140), (0.05, 0.85, 0.15, 0.0))),
    ("from the mother's haplotype 2", ((245, 150, 200), (0.0, 0.45, 0.0, 0.0))),
    ("from both parents", ((120, 50, 170), (0.55, 0.85, 0.0, 0.0))),
    ("from either parent: can't be told", ((140, 140, 100), (0.15, 0.15, 0.45, 0.3))),
    ("consistent, written differently", ((0, 175, 195), (0.8, 0.0, 0.25, 0.0))),
    ("a Mendelian error", ((215, 0, 0), (0.0, 1.0, 1.0, 0.1))),
    ("a candidate new mutation", ((255, 200, 0), (0.0, 0.2, 1.0, 0.0))),
    ("not called in all three", ((185, 185, 185), (0.0, 0.0, 0.0, 0.3))),
]
_TRIO_SOMETIMES = (2, 4)  #: (a parent's haplotype 2: in a picture's key only where it shows)


def _parent_counts(phased: bool = False) -> dict:
    return {"phased": phased, "sites": 0, "phase_sets": 0, "changes": 0, "crossovers": 0, "at": []}


@dataclass
class TrioTally:
    """Counts of a child compared with both parents, over one or more regions. Tallies add up (``a + b``)."""

    bases: int = 0
    all_called: int = 0
    child_variant: int = 0
    from_father: int = 0
    from_mother: int = 0
    from_both: int = 0
    origin_unknown: int = 0
    error_bases: int = 0
    error_sites: int = 0
    new_bases: int = 0
    new_sites: int = 0
    written_differently_bases: int = 0
    written_differently_stretches: int = 0
    examples: list = field(default_factory=list)
    parents: dict = field(default_factory=lambda: {"father": _parent_counts(), "mother": _parent_counts()})

    def __add__(self, other: TrioTally) -> TrioTally:
        out = TrioTally()
        for name in self.__dataclass_fields__:
            if name not in ("examples", "parents"):
                setattr(out, name, getattr(self, name) + getattr(other, name))
        out.examples = (self.examples + other.examples)[:EXAMPLES]
        for p in out.parents:
            x, y = self.parents[p], other.parents[p]
            out.parents[p] = {k: (x[k] or y[k]) if k == "phased" else x[k] + y[k] for k in x}
        return out

    def as_dict(self) -> dict:
        return {
            "bases": self.bases,
            "all_called": self.all_called,
            "child_variant_bases": self.child_variant,
            "from_father": self.from_father,
            "from_mother": self.from_mother,
            "from_both": self.from_both,
            "origin_unknown": self.origin_unknown,
            "mendelian_errors": {
                "bases": self.error_bases,
                "sites": self.error_sites,
                "new_mutation_bases": self.new_bases,
                "new_mutation_sites": self.new_sites,
                "examples": self.examples,
            },
            "written_differently": {"stretches": self.written_differently_stretches, "bases": self.written_differently_bases},
            "haplotypes": {p: (dict(x) if x["phased"] else None) for p, x in self.parents.items()},
        }


def _allele_text(x: int, r: int | None, items) -> str:
    st, extra = x & 0xF, x >> 4
    ins, cplx = items[extra]
    if st == C.HAP_ABSENT:
        return ""
    if st == C.HAP_REF:
        txt = C.IUPAC[r] if r is not None else "ref"
    elif st == C.HAP_DEL:
        txt = "-"
    elif st == C.HAP_UNKNOWN:
        txt = "."
    elif st == C.HAP_COMPLEX:
        txt = "<" + (cplx or "complex")[:24] + ">"
    else:
        txt = C.IUPAC[st]
    return txt + (f"+{ins[:24]}" if ins else "")


def _genotype_text(ch: _Chunk, j: int, r, items) -> str:
    alleles = [x for x in (_allele_text(int(ch.k1[j]), r, items), _allele_text(int(ch.k2[j]), r, items)) if x]
    return ("|" if ch.phased[j] else "/").join(alleles)


def _haplotypes(ch: _Chunk, w0: int, w1: int, ref4: np.ndarray, items) -> tuple[set, set | None]:
    """What a file's haplotypes could spell over [w0, w1) of its chunk: the sequences, and the pairs of them (None
    where it has one copy)."""
    k1, k2, r = ch.k1[w0:w1], ch.k2[w0:w1], ref4[w0:w1]
    if (k2 == _ABSENT).all():
        return {_spell(k1, r, items)}, None
    pairs = _haplotype_pairs(k1, k2, r, items)
    return {x for p in pairs for x in p}, pairs


def _trio_consistent(chs, w0: int, w1: int, ref4: np.ndarray, items) -> bool:
    """Whether the child's sequences over a stretch could be one of the father's and one of the mother's."""
    (c_one, c_pairs), (f_one, _), (m_one, _) = (_haplotypes(ch, w0, w1, ref4, items) for ch in chs)
    if c_pairs is None:
        return bool(c_one & (f_one | m_one))
    return any((x in f_one and y in m_one) or (y in f_one and x in m_one) for x, y in c_pairs)


def _crossovers(sites: np.ndarray, contig: str, phased: bool, min_run: int = MIN_RUN) -> dict:
    """Which of a parent's haplotypes the child received, at sites (position, phase set, haplotype): the changes
    along each phase set, and those backed by ``min_run`` sites in a row on each side (crossovers)."""
    out = _parent_counts(phased)
    if not len(sites):
        return out
    out["sites"] = len(sites)
    sites = sites[np.lexsort((sites[:, 0], sites[:, 1]))]
    bounds = [0, *(np.flatnonzero(np.diff(sites[:, 1]) != 0) + 1).tolist(), len(sites)]
    out["phase_sets"] = len(bounds) - 1
    for a, b in zip(bounds, bounds[1:]):
        pos, hap = sites[a:b, 0], sites[a:b, 2]
        edges = [0, *(np.flatnonzero(np.diff(hap) != 0) + 1).tolist(), len(hap)]
        out["changes"] += len(edges) - 2
        kept: list[list[int]] = []
        for x, y in zip(edges, edges[1:]):
            if y - x < min_run:
                continue
            if kept and kept[-1][0] == hap[x]:
                kept[-1][2] = y
            else:
                kept.append([int(hap[x]), x, y])
        for (_, _, y0), (_, x1, _) in zip(kept, kept[1:]):
            out["crossovers"] += 1
            out["at"].append([contig, int(pos[y0 - 1]) + 1, int(pos[x1]) + 1])
    return out


def _trio_piece(chs, items, ref4_of, lo: int, hi: int, gap: int, t: TrioTally, sites: dict, classes=None, collect=None):
    """Compare a piece of the child's, the father's and the mother's files and count what is in [lo, hi)."""
    Cc, F, M = chs
    s, n = Cc.s, Cc.e - Cc.s
    c1, c2 = Cc.k1, Cc.k2
    f1, f2, m1, m2 = F.k1, F.k2, M.k1, M.k2
    f2p, m2p, c2p = f2 != _ABSENT, m2 != _ABSENT, c2 != _ABSENT

    def fathers(x):
        return (x == f1) | ((x == f2) & f2p)

    def mothers(x):
        return (x == m1) | ((x == m2) & m2p)

    allc = Cc.called & F.called & M.called
    one = ~c2p
    f1c, m1c, f2c, m2c = fathers(c1), mothers(c1), fathers(c2), mothers(c2)
    opt1, opt2 = f1c & m2c, f2c & m1c  # (the child's first allele from the father and the second from the mother, or the other way)
    consistent = np.where(one, f1c | m1c, opt1 | opt2)
    err = allc & ~consistent
    # stretches around the errors, with every variant of the three near them: consistent as sequences?
    written = np.zeros(n, dtype=bool)
    ref4 = ref4_of() if err.any() else None
    if ref4 is not None:
        pos = np.flatnonzero(Cc.var | F.var | M.var | err)
        breaks = np.flatnonzero(np.diff(pos) > gap) + 1
        firsts, lasts = pos[np.concatenate(([0], breaks))], pos[np.concatenate((breaks - 1, [pos.size - 1]))] + 1
        ks = np.unique(np.searchsorted(firsts, np.flatnonzero(err), side="right") - 1).tolist()
        for w0, w1 in _widened([[int(firsts[k]), int(lasts[k])] for k in ks], chs, ref4, items, gap, allc):
            if _trio_consistent(chs, w0, w1, ref4, items):
                written[w0:w1] = True
                t.written_differently_stretches += lo <= s + w0 < hi
    rewritten = err & written
    err &= ~written
    new = err & (((c1 != 0) & (c1 != _ABSENT) & ~f1c & ~m1c) | (c2p & (c2 != 0) & ~f2c & ~m2c))
    # the father's and the mother's allele, where only one way fits (-1: can't be told)
    only1, only2 = opt1 & ~opt2, opt2 & ~opt1
    pat = np.where(only1, c1, np.where(only2, c2, -1))
    mat = np.where(only1, c2, np.where(only2, c1, -1))
    hom = c2p & (c1 == c2) & opt1
    pat, mat = np.where(hom, c1, pat), np.where(hom, c1, mat)
    pat = np.where(one, np.where(f1c & ~m1c, c1, -1), pat)
    mat = np.where(one, np.where(m1c & ~f1c, c1, -1), mat)
    cvar = allc & Cc.var
    ok = cvar & consistent
    vf, vm = (pat != -1) & (pat != 0), (mat != -1) & (mat != 0)
    from_both, from_f, from_m = ok & vf & vm, ok & vf & ~vm, ok & vm & ~vf
    unknown = ok & ~vf & ~vm
    a, b = lo - s, hi - s
    w = slice(a, b)
    t.bases += b - a
    t.all_called += int(allc[w].sum())
    t.child_variant += int(cvar[w].sum())
    t.from_father += int(from_f[w].sum())
    t.from_mother += int(from_m[w].sum())
    t.from_both += int(from_both[w].sum())
    t.origin_unknown += int(unknown[w].sum())
    t.error_bases += int(err[w].sum())
    t.error_sites += _runs(err[w])
    t.new_bases += int(new[w].sum())
    t.new_sites += _runs(new[w])
    t.written_differently_bases += int(rewritten[w].sum())
    if err[w].any() and (len(t.examples) < EXAMPLES or collect is not None):
        e = err[w]
        for i in np.flatnonzero(e & ~np.concatenate(([False], e[:-1]))).tolist():
            j = a + i
            r = int(ref4[j]) if ref4 is not None else None
            ex = {
                "position": s + j + 1,
                "child": _genotype_text(Cc, j, r, items),
                "father": _genotype_text(F, j, r, items),
                "mother": _genotype_text(M, j, r, items),
                "new": bool(new[j]),
            }
            if len(t.examples) < EXAMPLES:
                t.examples.append(ex)
            if collect is not None:
                collect[1].append({"contig": collect[0], **ex})
            elif len(t.examples) >= EXAMPLES:
                break
    # which of a phased parent's haplotypes the child received, where the parent is heterozygous and it shows
    inherited = {}
    for parent, P, allele in (("father", F, pat), ("mother", M, mat)):
        p1, p2, ph = P.k1[w], P.k2[w], P.phased[w]
        if ph.any():
            t.parents[parent]["phased"] = True
        allele = allele[w]
        h = np.where(allele == p1, 1, np.where(allele == p2, 2, 0))
        site = allc[w] & consistent[w] & (allele != -1) & (p1 != p2) & (p2 != _ABSENT) & ph & (h > 0)
        inherited[parent] = np.where(site, h, 0)
        if site.any():
            idx = np.flatnonzero(site)
            ps = P.ps[w][idx].astype(np.int64) if P.ps is not None else np.zeros(idx.size, dtype=np.int64)
            sites[parent].append(np.stack([idx + lo, ps, h[idx]], axis=1))
    if classes is not None:
        cls = np.zeros(b - a, dtype=np.uint8)
        x = from_f[w]
        cls[x] = np.where(inherited["father"][x] == 2, 2, 1)
        x = from_m[w]
        cls[x] = np.where(inherited["mother"][x] == 2, 4, 3)
        cls[from_both[w]] = 5
        cls[unknown[w]] = 6
        cls[rewritten[w]] = 7
        cls[err[w]] = 8
        cls[new[w]] = 9
        cls[~allc[w]] = 10
        classes.append(cls)


def _trio_region(ms, ids: _Ids, ref, s: int, e: int, length: int, chunk: int, gap: int, classes=None, errors=None) -> TrioTally:
    t = TrioTally()
    sites: dict = {"father": [], "mother": []}
    for s0, e0, chs in _pieces(ms, s, e, length, chunk, gap):
        lo, hi = max(s, s0), min(e, e0)
        if lo < hi:
            collect = None if errors is None else (ms[0].c.name, errors)
            _trio_piece(chs, ids.items, _ref_reader(ref, s0, e0), lo, hi, gap, t, sites, classes, collect)
    for p in t.parents:
        got = np.concatenate(sites[p]) if sites[p] else np.zeros((0, 3), dtype=np.int64)
        t.parents[p] = _crossovers(got, ms[0].c.name, t.parents[p]["phased"])
    return t


def compare_trio(
    child, father, mother, regions=None, *, chunk: int = CHUNK, gap: int = GAP, all_calls: bool = False, errors: list | None = None
) -> dict:
    """A child's gpxil file compared with both parents': per region and in all (``regions`` and ``all_calls`` as for
    :func:`compare_pair`). ``errors``, a list, collects every Mendelian error, as described in the result's examples."""
    _need_samples((("the child's file", child), ("the father's file", father), ("the mother's file", mother)))
    ids = _Ids()
    out, total = [], TrioTally()
    for names, s, e in _common_regions([child, father, mother], regions):
        cs = [g[n] for g, n in zip((child, father, mother), names)]
        _check_same_reference(cs)
        trio = (child, father, mother)
        ms = [_Member(c, ids, want_ps=k > 0, confident=_confident(trio[k], all_calls)) for k, c in enumerate(cs)]
        t = _trio_region(ms, ids, _ref_source(cs), s, e, cs[0].length, chunk, gap, errors=errors)
        out.append({"contig": cs[0].name, "start": s + 1, "end": e, **t.as_dict()})
        total = total + t
    return {
        "mode": "trio",
        "child": {"file": _name(child), "sample": child.sample_id},
        "father": {"file": _name(father), "sample": father.sample_id},
        "mother": {"file": _name(mother), "sample": mother.sample_id},
        "reference": (child.metadata.get("reference") or {}).get("name"),
        "gap": gap,
        "all_calls": all_calls,
        "regions": out,
        "total": {"regions": len(out), **total.as_dict()},
    }


# ---------------------------------------------------------------- pictures


def picture(
    gfiles,
    contig: str,
    start: int = 0,
    end: int | None = None,
    *,
    bases_per_row=None,
    palette="classic",
    chunk=CHUNK,
    gap=GAP,
    all_calls=False,
):
    """A picture of a region, one pixel a base: of two files (``[a, b]``) in the classes of :data:`PAIR_CLASSES`, or
    of a trio (``[child, father, mother]``) in those of :data:`TRIO_CLASSES`, over the reference faded."""
    from .palettes import ref_colours
    from .render import Rendering, _wrap

    trio = len(gfiles) == 3
    ((names, start, end),) = _common_regions(gfiles, [(contig, start, end)])
    cs = [g[n] for g, n in zip(gfiles, names)]
    _check_same_reference(cs)
    ids = _Ids()
    ref = _ref_source(cs)
    classes: list = []
    if trio:
        ms = [_Member(c, ids, want_ps=k > 0, confident=_confident(g, all_calls)) for k, (c, g) in enumerate(zip(cs, gfiles))]
        _trio_region(ms, ids, ref, start, end, cs[0].length, chunk, gap, classes)
    else:
        ms = [_Member(c, ids, confident=_confident(g, all_calls)) for c, g in zip(cs, gfiles)]
        t = PairTally()
        for s0, e0, (A, B) in _pieces(ms, start, end, cs[0].length, chunk, gap):
            lo, hi = max(start, s0), min(end, e0)
            if lo < hi:
                _pair_piece(ms[0], ms[1], A, B, ids.items, _ref_reader(ref, s0, e0), lo, hi, gap, t, classes)
    cls = np.concatenate(classes) if classes else np.zeros(end - start, dtype=np.uint8)
    table = TRIO_CLASSES if trio else PAIR_CLASSES
    rgb_lut, cmyk_lut = ref_colours(palette)
    rgb = (rgb_lut.astype(float) + (255 - rgb_lut.astype(float)) * FADE).astype(np.uint8)
    cmyk = (cmyk_lut.astype(float) * (1 - FADE)).astype(np.uint8)
    present = set(np.unique(cls).tolist())
    legend = [(table[0][0], C.BASE_A)]
    for k, (label, colours) in enumerate(table):
        if colours is None:
            continue
        rgb[0xF0 + k] = colours[0]
        cmyk[0xF0 + k] = np.round(np.array(colours[1]) * 255)
        if not (trio and k in _TRIO_SOMETIMES) or k in present:
            legend.append((label, 0xF0 + k))
    base = ref.read("ref", start, end) if ref is not None else np.full(end - start, C.BASE_N, dtype=np.uint8)
    idx = np.where(cls == 0, base, 0xF0 + cls).astype(np.uint8)
    bpr = bases_per_row or cs[0].width
    return Rendering(_wrap(idx, bpr, C.NO_BASE), rgb, cmyk, legend, bpr, 1, start, end, f"{cs[0].name}:{start + 1:,}-{end:,}")


# ---------------------------------------------------------------- as text


def _pct(n, d=None) -> str:
    if d is not None:
        n = n / d if d else None
    if n is None:
        return "–"
    if 0.9999 < n < 1:  # (not quite all, nor quite none)
        return "> 99.99 %"
    if 0 < n < 0.0001:
        return "< 0.01 %"
    return f"{100 * n:.2f} %"


def _n(n: int, one: str, many: str | None = None) -> str:
    return f"{n:,} {one if n == 1 else many or one + 's'}"


def _block_pair(label: str, d: dict) -> list[str]:
    n = d["bases"]
    c, v, st = d["called"], d["variant_bases"], d["stretches"]
    rows = [
        (
            "called",
            f"in both {c['both']:,} ({_pct(c['both'], n)}) · in A only {c['a_only']:,} · in B only {c['b_only']:,} · in neither {c['neither']:,}",
        ),
        ("variant bases", f"{v['either']:,} called in both with a variant in either"),
        (
            "",
            f"the same {v['same'] + v['written_differently']:,} ({_pct(v['agreement'])}), of them written differently {v['written_differently']:,}",
        ),
        ("", f"only in A {v['only_a']:,} · only in B {v['only_b']:,} · another genotype {v['other_genotype']:,}"),
    ]
    for ex in d["examples"][:5]:
        at = f"{ex['position']:,}" + (f"-{ex['end']:,}" if ex["end"] > ex["position"] else "")
        rows.append(("", f"at {at}: A {ex['a']}, B {ex['b']}"))
    if len(d["examples"]) > 5:
        rows.append(("", f"... ({len(d['examples'])} in the JSON)"))
    if st["not_checked"]:
        rows.append(("", f"{_n(st['not_checked'], 'stretch', 'stretches')} not checked for being written differently (no reference)"))
    for k in _KINDS:
        e = d["events"][k]
        a, b = e["a"], e["b"]
        if not (a["total"] or b["total"]):
            continue
        rows += [
            (k, f"A {a['total']:,}: found in B {a['found']:,} · not in B {a['missing']:,} · another genotype {a['other_genotype']:,}"),
            ("", f"B {b['total']:,}: found in A {b['found']:,} · not in A {b['missing']:,} · another genotype {b['other_genotype']:,}"),
            ("", f"precision {_pct(e['precision'])} · recall {_pct(e['recall'])} · F1 {_pct(e['f1'])} (B as the truth)"),
        ]
        if a["not_compared"] or b["not_compared"]:
            rows.append(("", f"not compared, where the other file didn't call: A {a['not_compared']:,} · B {b['not_compared']:,}"))
    return [f"{label} · {n:,} bases"] + [f"  {k:<16}{x}" for k, x in rows]


def _block_trio(label: str, d: dict) -> list[str]:
    n = d["bases"]
    me = d["mendelian_errors"]
    rows = [
        ("called", f"in all three {d['all_called']:,} ({_pct(d['all_called'], n)})"),
        ("child's variants", _n(d["child_variant_bases"], "base")),
        ("", f"from the father {d['from_father']:,} · from the mother {d['from_mother']:,} · from both {d['from_both']:,}"),
        ("", f"from either, can't be told {d['origin_unknown']:,}"),
        (
            "Mendelian errors",
            f"{_n(me['sites'], 'site')} ({_n(me['bases'], 'base')}), {me['new_mutation_sites']:,} of them candidate new mutations",
        ),
    ]
    for ex in me["examples"][:5]:
        rows.append(
            ("", f"at {ex['position']:,}: child {ex['child']}, father {ex['father']}, mother {ex['mother']}{' (new)' if ex['new'] else ''}")
        )
    if len(me["examples"]) > 5:
        rows.append(("", f"... ({len(me['examples'])} in the JSON)"))
    wd = d["written_differently"]
    if wd["stretches"]:
        rows.append(
            (
                "written differently",
                f"{_n(wd['stretches'], 'stretch', 'stretches')} ({_n(wd['bases'], 'base')}) consistent as sequences, not base by base",
            )
        )
    for parent in ("father", "mother"):
        x = d["haplotypes"][parent]
        head = f"{parent}'s haplotypes"
        if x is None:
            rows.append((head, "not phased, so which one the child received isn't known"))
            continue
        at = ", ".join(f"{p0:,}–{p1:,}" for _, p0, p1 in x["at"][:4]) + (", ..." if len(x["at"]) > 4 else "")
        rows.append((head, f"{_n(x['crossovers'], 'crossover')}{' (between ' + at + ')' if at else ''}"))
        rows.append(("", f"{_n(x['changes'], 'change')} at {_n(x['sites'], 'site')} in {_n(x['phase_sets'], 'phase set')}"))
    return [f"{label} · {n:,} bases"] + [f"  {k:<20}{x}" for k, x in rows]


def format_compare(result: dict) -> str:
    """The result of :func:`compare_pair` or :func:`compare_trio` as a table for people."""
    trio = result["mode"] == "trio"
    if trio:
        head = ", ".join(f"{w} {result[w]['file']} ({result[w]['sample']})" for w in ("child", "father", "mother"))
    else:
        head = f"A {result['a']['file']} ({result['a']['sample']}), B {result['b']['file']} ({result['b']['sample']})"
    if result.get("reference"):
        head += f" · {result['reference']}"
    block = _block_trio if trio else _block_pair
    lines = [head, ""]
    regs = result["regions"]
    for r in regs:
        lines += block(f"{r['contig']}:{r['start']:,}-{r['end']:,}", r)
    if len(regs) > 1:
        lines += ["", *block(f"all {len(regs)} regions", result["total"])]
    return "\n".join(lines)
