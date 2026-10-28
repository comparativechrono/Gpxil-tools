"""What a gpxil file holds, counted: the reference, the calls, the variants, phasing and quality.

Counts are made from the planes a piece at a time, so a whole genome needs
little memory, and they work the same for a region as for a whole contig:

- **reference**: the bases A, C, G and T, N, gaps (no base), other IUPAC
  codes and soft-masked bases; GC content is C and G over A, C, G and T.
- **called**: bases where every haplotype present is known; **not called**
  where none is (UNKNOWN); **partly called** where one is and one is not.
  At a base with one copy (haplotype 2 ABSENT) the one haplotype counts.
- **substitutions**, at bases where a haplotype carries another base and none
  is unknown: homozygous where every haplotype present carries the same base
  (one copy included), heterozygous otherwise (the other haplotype the
  reference, another base, or deleted). Each different base carried is one
  allele, a transition (A↔G, C↔T) or a transversion; their ratio needs the
  reference.
- **deletions**: a run of deleted bases on a haplotype is one deletion,
  homozygous when the other haplotype has the same run (or there is one
  copy). **Insertions**: an insertion after a base, homozygous when both
  haplotypes have the same inserted bases (or there is one copy). An event
  belongs to the region where it starts, with all of its bases.
- **complex**: COMPLEX alleles, counted by their sidecar entries.
- **phasing**: the heterozygous substitutions whose call was phased; and the
  phase sets, the distinct ``ps`` values of phased bases (all phased calls
  are one phase set when there is no ``ps`` plane), each spanning its first
  to its last base in the region. N50 is the span at which the sets, from
  the longest down, cover half the total span.
- **gq** and **dp**: the distribution over the bases that have a value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import codes as C

CHUNK = 1 << 22  #: bases read at a time
_SUBS = np.array(C.HAP_SUB_CODES, dtype=np.uint8)
_SINGLE = np.zeros(16, dtype=bool)
_SINGLE[list(C.HAP_SUB_CODES)] = True


@dataclass
class Tally:
    """Counts for one or more regions. Tallies add up (``a + b``)."""

    regions: int = 0
    bases: int = 0
    has_ref: bool = False
    has_gt: bool = False
    ref_codes: np.ndarray = field(default_factory=lambda: np.zeros(32, dtype=np.int64))
    called: int = 0
    not_called: int = 0
    partly_called: int = 0
    one_copy: int = 0
    filtered: int = 0
    snv_het: int = 0
    snv_hom: int = 0
    snv_het_phased: int = 0
    transitions: int = 0
    transversions: int = 0
    ins_het: int = 0
    ins_hom: int = 0
    ins_bases: int = 0
    del_het: int = 0
    del_hom: int = 0
    del_bases: int = 0
    complex_het: int = 0
    complex_hom: int = 0
    phase_spans: list = field(default_factory=list)  #: the span of each phase set
    gq: np.ndarray | None = None  #: histogram of the values 0-254
    dp: np.ndarray | None = None

    def __add__(self, other: Tally) -> Tally:
        out = Tally()
        for name in self.__dataclass_fields__:
            a, b = getattr(self, name), getattr(other, name)
            if isinstance(a, bool):
                value = a or b
            elif a is None or b is None:
                value = b if a is None else a
            else:
                value = a + b  # (ints, arrays and lists)
            setattr(out, name, value)
        return out

    def as_dict(self) -> dict:
        """The counts as plain data, with the fractions, ratios and quartiles worked out."""
        rc = self.ref_codes
        d: dict = {"bases": self.bases}
        if self.has_ref:
            acgt = [int(rc[b] + rc[b | C.SOFTMASK]) for b in C.HAP_SUB_CODES]
            n = int(rc[C.BASE_N] + rc[C.BASE_N | C.SOFTMASK])
            gaps = int(rc[C.NO_BASE] + rc[C.SOFTMASK])
            known = sum(acgt)
            d["reference"] = {
                "A": acgt[0],
                "C": acgt[1],
                "G": acgt[2],
                "T": acgt[3],
                "N": n,
                "gaps": gaps,
                "ambiguous": int(rc.sum()) - known - n - gaps,
                "soft_masked": int(rc[C.SOFTMASK + 1 :].sum()),
                "gc": round((acgt[1] + acgt[2]) / known, 4) if known else None,
            }
        else:
            d["reference"] = None
        if not self.has_gt:
            for key in ("calls", "substitutions", "insertions", "deletions", "complex", "phasing", "gq", "dp"):
                d[key] = None
            return d
        d["calls"] = {
            "called": self.called,
            "not_called": self.not_called,
            "partly_called": self.partly_called,
            "one_copy": self.one_copy,
            "filtered": self.filtered,
        }
        ts, tv = self.transitions, self.transversions
        d["substitutions"] = {
            "heterozygous": self.snv_het,
            "homozygous": self.snv_hom,
            "transitions": ts if self.has_ref else None,
            "transversions": tv if self.has_ref else None,
            "ti_tv": round(ts / tv, 3) if self.has_ref and tv else None,
        }
        d["insertions"] = {"heterozygous": self.ins_het, "homozygous": self.ins_hom, "bases": self.ins_bases}
        d["deletions"] = {"heterozygous": self.del_het, "homozygous": self.del_hom, "bases": self.del_bases}
        d["complex"] = {"heterozygous": self.complex_het, "homozygous": self.complex_hom}
        spans = sorted(self.phase_spans, reverse=True)
        n50, run = None, 0
        for s in spans:
            run += s
            if 2 * run >= sum(spans):
                n50 = s
                break
        d["phasing"] = {
            "heterozygous_substitutions_phased": self.snv_het_phased,
            "fraction": round(self.snv_het_phased / self.snv_het, 4) if self.snv_het else None,
            "phase_sets": len(spans),
            "n50": n50,
            "largest": spans[0] if spans else None,
        }
        d["gq"] = _distribution(self.gq)
        d["dp"] = _distribution(self.dp)
        return d


def _distribution(hist: np.ndarray | None) -> dict | None:
    if hist is None:
        return None
    n = int(hist.sum())
    if not n:
        return {"bases": 0, "mean": None, "min": None, "q1": None, "median": None, "q3": None, "max": None}
    cum = np.cumsum(hist)
    present = np.flatnonzero(hist)

    def quantile(f: float) -> int:  # the smallest value with at least this fraction of the bases at or below it
        return int(np.searchsorted(cum, f * n, side="left"))

    return {
        "bases": n,
        "mean": round(float((hist * np.arange(hist.size)).sum() / n), 2),
        "min": int(present[0]),
        "q1": quantile(0.25),
        "median": quantile(0.5),
        "q3": quantile(0.75),
        "max": int(present[-1]),
    }


class _Runs:
    """Runs of deleted bases on one haplotype, followed from chunk to chunk."""

    def __init__(self, open_before: bool):
        self.start: int | None = -1 if open_before else None  # start of a run still open; -1: it began before the region
        self.one_copy = False
        self.runs: list[tuple[int, int, bool]] = []  # (start, end, one copy)

    def _close(self, end: int) -> None:
        if self.start is not None and self.start >= 0:
            self.runs.append((self.start, end, self.one_copy))
        self.start = None

    def feed(self, dele: np.ndarray, absent2: np.ndarray, s: int) -> None:
        """The deleted bases of the chunk starting at ``s`` (in the region)."""
        x = dele.view(np.int8)
        prev = np.concatenate(([1 if self.start is not None else 0], x[:-1]))
        starts = np.flatnonzero((x == 1) & (prev == 0)).tolist()
        ends = np.flatnonzero((x == 0) & (prev == 1)).tolist()
        if self.start is not None:
            if not ends:
                return  # the open run goes on through the chunk
            self._close(s + ends.pop(0))
        for i, a in enumerate(starts):  # the remaining starts and ends alternate
            if i < len(ends):
                self.runs.append((s + a, s + ends[i], bool(absent2[a])))
            else:
                self.start, self.one_copy = s + a, bool(absent2[a])

    def follow(self, dele: np.ndarray, s: int, contig_end: int) -> None:
        """A chunk after the region: only the run that began in the region is followed, to its end."""
        stop = np.flatnonzero(~dele)
        if stop.size:
            self._close(s + int(stop[0]))
        elif s + dele.size >= contig_end:
            self._close(contig_end)

    @property
    def following(self) -> bool:
        return self.start is not None and self.start >= 0


def region_tally(contig, start: int = 0, end: int | None = None, *, chunk: int = CHUNK) -> Tally:
    """Count one region ``[start, end)`` (0-based) of a :class:`~gpxiltools.container.Contig`."""
    end = contig.length if end is None else end
    if not (0 <= start < end <= contig.length):
        raise IndexError(f"{contig.name}:{start + 1}-{end} is outside {contig.name} (length {contig.length:,})")
    t = Tally(regions=1, bases=end - start, has_ref=contig.covers("ref", start, end), has_gt=contig.has("gt"))
    if not t.has_gt:
        if t.has_ref:
            for s in range(start, end, chunk):
                t.ref_codes += np.bincount(contig.read("ref", s, min(end, s + chunk)), minlength=32)[:32]
        return t
    has = {p: contig.has(p) for p in ("flags", "gq", "dp", "ps")}
    t.gq = np.zeros(255, dtype=np.int64) if has["gq"] else None
    t.dp = np.zeros(255, dtype=np.int64) if has["dp"] else None
    if start > 0:
        before = contig.read("gt", start - 1, start)
        open_before = [C.normalize_states(f(before))[0] == C.HAP_DEL for f in (C.hap1, C.hap2)]
    else:
        open_before = [False, False]
    runs = [_Runs(o) for o in open_before]
    ins_one_copy: dict[int, bool] = {}
    blocks: dict[int, list[int]] = {}
    for s in range(start, end, chunk):
        e = min(end, s + chunk)
        gt = contig.read("gt", s, e)
        h1, h2 = C.normalize_states(C.hap1(gt)), C.normalize_states(C.hap2(gt))
        del gt
        flags = contig.read("flags", s, e) if has["flags"] else np.zeros(e - s, dtype=np.uint8)
        absent1, absent2 = h1 == C.HAP_ABSENT, h2 == C.HAP_ABSENT
        unk1, unk2 = h1 == C.HAP_UNKNOWN, h2 == C.HAP_UNKNOWN
        none_known = unk1 & (unk2 | absent2)
        some_unknown = unk1 | unk2
        t.not_called += int(none_known.sum())
        t.partly_called += int((some_unknown & ~none_known).sum())
        t.called += int((~some_unknown).sum())
        t.one_copy += int(absent2.sum())
        t.filtered += int(((flags & C.FLAG_FILTERED) != 0).sum())
        sub1, sub2 = np.isin(h1, _SUBS), np.isin(h2, _SUBS)
        snv = (sub1 | sub2) & ~some_unknown
        hom = snv & ((sub1 & sub2 & (h1 == h2)) | (sub1 & absent2) | (sub2 & absent1))
        het = snv & ~hom
        phased = (flags & C.FLAG_PHASED) != 0
        t.snv_hom += int(hom.sum())
        t.snv_het += int(het.sum())
        t.snv_het_phased += int((het & phased).sum())
        if t.has_ref:
            ref = contig.read("ref", s, e)
            t.ref_codes += np.bincount(ref, minlength=32)[:32]
            ref4 = np.bitwise_and(ref, 0x0F)
            single = _SINGLE[ref4]
            a1 = snv & sub1 & single
            a2 = snv & sub2 & single & ~(sub1 & (h1 == h2))
            ts = int(np.isin((ref4 | h1)[a1], (5, 10)).sum() + np.isin((ref4 | h2)[a2], (5, 10)).sum())  # A|G, C|T
            t.transitions += ts
            t.transversions += int(a1.sum() + a2.sum()) - ts
        for r, h in zip(runs, (h1, h2)):
            r.feed(h == C.HAP_DEL, absent2, s)
        for i in np.flatnonzero((flags & (C.FLAG_INS1 | C.FLAG_INS2)) != 0).tolist():
            ins_one_copy[s + i] = bool(absent2[i])
        for p in ("gq", "dp"):
            if has[p]:
                v = contig.read(p, s, e)
                getattr(t, p)[:] += np.bincount(v[v != C.QUAL_MISSING], minlength=255)[:255]
        if phased.any():
            pos = np.flatnonzero(phased)
            ids = contig.read("ps", s, e)[pos] if has["ps"] else np.zeros(pos.size, dtype=np.uint32)
            vals, first = np.unique(ids, return_index=True)
            last = ids.size - 1 - np.unique(ids[::-1], return_index=True)[1]
            for v, f, lst in zip(vals.tolist(), first.tolist(), last.tolist()):
                lo, hi = s + int(pos[f]), s + int(pos[lst])
                b = blocks.setdefault(v, [lo, hi])
                b[0], b[1] = min(b[0], lo), max(b[1], hi)
    # deletions that begin in the region and go on past it: followed to their ends, or to the end of the contig
    s = end
    while any(r.following for r in runs) and s < contig.length:
        e = min(contig.length, s + chunk)
        gt = contig.read("gt", s, e)
        for r, f in zip(runs, (C.hap1, C.hap2)):
            if r.following:
                r.follow(C.normalize_states(f(gt)) == C.HAP_DEL, s, contig.length)
        s = e
    for r in runs:
        if r.following:
            r._close(contig.length)  # (a deletion to the end of what the contig holds)
    # deletions: homozygous when both haplotypes have the same run, or there is one copy
    same = {(a, b) for a, b, _ in runs[0].runs} & {(a, b) for a, b, _ in runs[1].runs}
    for k, r in enumerate(runs):
        for a, b, one in r.runs:
            if (a, b) in same and k == 1:
                continue  # (counted with haplotype 1)
            if (a, b) in same or one:
                t.del_hom += 1
            else:
                t.del_het += 1
            t.del_bases += b - a
    # insertions and COMPLEX alleles, from the sidecar
    events: dict[tuple[str, int], dict[int, str]] = {}
    for ent in contig.sidecar():
        if start <= ent.offset < end and ent.kind in ("INS", "COMPLEX"):
            events.setdefault((ent.kind, ent.offset), {})[ent.hap] = ent.data
    for (kind, off), haps in events.items():
        same_both = len(haps) == 2 and haps[1] == haps[2]
        if kind == "INS":
            if same_both:
                t.ins_hom += 1
                t.ins_bases += len(haps[1])
                continue
            for data in haps.values():
                if ins_one_copy.get(off, False):
                    t.ins_hom += 1
                else:
                    t.ins_het += 1
                t.ins_bases += len(data)
        elif same_both or (list(haps) == [1] and _one_copy_at(contig, off)):
            t.complex_hom += 1
        else:
            t.complex_het += len(haps)
    t.phase_spans = [hi - lo + 1 for lo, hi in blocks.values()]
    return t


def _one_copy_at(contig, offset: int) -> bool:
    return bool(C.hap2(contig.read("gt", offset, offset + 1))[0] == C.HAP_ABSENT)


def _wanted(gfile, regions) -> list[tuple]:
    """The regions asked for, each ``(contig, start, end)`` at the contig's own offsets: a sequence named that only
    windows hold (spec 3.1) is each of them; a region that windows hold together, their stretch."""
    out = []
    for name, s, e in regions or [(n, None, None) for n in gfile.contigs]:
        if s is None and e is None and gfile.find(name) is None and len(gfile.windows_of(name)) > 1:
            out += [(w, 0, w.length) for w in gfile.windows_of(name)]
        else:
            out.append(gfile.locate(name, s, e, join=True))
    return out


def _through(gfile, c, s: int, e: int) -> tuple:
    """A region read through the file's join of the contigs of its sequence that holds it (a Stretch of windows that
    overlap or touch), so that an event cut by the end of one window is counted whole: (view, start, end)."""
    return gfile.view(c, s, e)


def tally(gfile, regions=None, *, chunk: int = CHUNK) -> list[tuple]:
    """Tallies of ``regions``: ``(contig, start, end)``, 0-based, start and end None for the whole contig, end None
    for the rest of it (default: every contig). On a window (spec 3.1) they are positions in its parent, and a
    sequence named that only windows hold is each of its windows. A region is read through the file's windows that
    overlap or touch the one that holds it, so an event that one cuts is counted whole, in the region it begins in.
    Returns ``(contig, start, end, Tally)`` with the :class:`~gpxiltools.container.Contig` (or the
    :class:`~gpxiltools.container.Stretch` of windows that hold it together) and its own offsets."""
    return [(c, s, e, region_tally(*_through(gfile, c, s, e), chunk=chunk)) for c, s, e in _wanted(gfile, regions)]


def _in_all(gfile, parts, chunk: int) -> Tally:
    """The tally of all the regions together, each base once: where regions of a sequence overlap (windows that
    overlap, say, or the sequence and a window of it), their union is counted, read as one stretch."""
    total = None
    seqs: dict[str, list] = {}
    for c, s, e, t in parts:
        seqs.setdefault(c.parent, []).append((c, s, e, t))
    for name, ps in seqs.items():
        spans = sorted((c.start + s, c.start + e) for c, s, e, _ in ps)
        if all(a[1] < b[0] for a, b in zip(spans, spans[1:])):
            got = [t for *_, t in ps]  # (no two overlap or touch: the sum of them)
        else:
            union: list[list[int]] = []
            for a, b in spans:
                if union and a <= union[-1][1]:
                    union[-1][1] = max(union[-1][1], b)
                else:
                    union.append([a, b])
            views = gfile.stretches(name)  # (the file's contigs of the sequence, joined where they overlap or touch)
            got = []
            for a, b in union:
                v = next(v for v in views if v.start <= a and b <= v.end)
                got.append(region_tally(v, a - v.start, b - v.start, chunk=chunk))
        for t in got:
            total = t if total is None else total + t
    return total


def stats(gfile, regions=None, *, chunk: int = CHUNK) -> dict:
    """What the file holds, counted per region and in all, as plain data (:func:`format_stats` makes a table of it).
    In all, each base is counted once, where regions overlap."""
    parts = tally(gfile, regions, chunk=chunk)
    total = _in_all(gfile, parts, chunk)
    meta = gfile.metadata
    return {
        "file": str(gfile.path).rstrip("/").rsplit("/", 1)[-1],
        "sample": gfile.sample_id,
        "reference": (meta.get("reference") or {}).get("name"),
        "regions": [_where(c, s, e) | t.as_dict() for c, s, e, t in parts],
        "total": {"regions": len(parts), **total.as_dict()},
    }


def _where(c, s: int, e: int) -> dict:
    """A region in its sequence's coordinates (a window's: its parent's), and the window it is in."""
    out = {"contig": c.parent, "start": c.start + s + 1, "end": c.start + e}
    if c.window is not None:
        out["window"] = c.name
    return out


# ----------------------------------------------------------------- as a table


def _pct(n: int | None, d: int) -> str:
    return "–" if n is None or not d else f"{100 * n / d:.1f} %"


def _span(n: int | None) -> str:
    if n is None:
        return "–"
    return f"{n / 1e6:.1f} Mb" if n >= 1e6 else f"{n / 1e3:.1f} kb" if n >= 1e3 else f"{n} bp"


def _quality(q: dict | None) -> str | None:
    if q is None:
        return None
    if not q["bases"]:
        return "no values"
    return f"median {q['median']}, quartiles {q['q1']}–{q['q3']}, mean {q['mean']}; {q['bases']:,} bases with a value"


def _block(label: str, d: dict) -> list[str]:
    n = d["bases"]
    rows = [(label, f"{n:,} bases")]
    r = d["reference"]
    if r:
        acgt = ", ".join(f"{b} {_pct(r[b], n)}" for b in "ACGT")
        rest = (
            f"; N {_pct(r['N'], n)}"
            + (f"; gaps {r['gaps']:,}" if r["gaps"] else "")
            + (f"; other codes {r['ambiguous']:,}" if r["ambiguous"] else "")
        )
        gc = f" (GC {100 * r['gc']:.1f} %)" if r["gc"] is not None else ""
        rows.append(("reference", f"{acgt}{gc}{rest}; soft-masked {_pct(r['soft_masked'], n)}"))
    c = d["calls"]
    if c is None:
        rows.append(("sample", "none: the file holds the reference only"))
        return [f"{rows[0][0]} · {rows[0][1]}"] + [f"  {k:<18}{v}" for k, v in rows[1:]]
    called = f"{c['called']:,} bases ({_pct(c['called'], n)}); not called {c['not_called']:,} ({_pct(c['not_called'], n)})"
    if c["partly_called"]:
        called += f"; partly called {c['partly_called']:,}"
    rows.append(("called", called))
    if c["one_copy"]:
        rows.append(("one copy", f"{c['one_copy']:,} bases"))
    s = d["substitutions"]
    rows.append(
        (
            "substitutions",
            f"{s['heterozygous']:,} heterozygous, {s['homozygous']:,} homozygous" + (f"; Ti/Tv {s['ti_tv']:.2f}" if s["ti_tv"] else ""),
        )
    )
    for key in ("insertions", "deletions"):
        e = d[key]
        rows.append((key, f"{e['heterozygous']:,} heterozygous, {e['homozygous']:,} homozygous; {e['bases']:,} bases"))
    x = d["complex"]
    if x["heterozygous"] or x["homozygous"]:
        rows.append(("complex", f"{x['heterozygous']:,} heterozygous, {x['homozygous']:,} homozygous"))
    p = d["phasing"]
    if p["phase_sets"]:
        sets = f"{p['phase_sets']:,} phase set{'s' if p['phase_sets'] != 1 else ''}"
        sets += f", N50 {_span(p['n50'])}, largest {_span(p['largest'])}" if p["phase_sets"] > 1 else f" over {_span(p['largest'])}"
        rows.append(
            ("phasing", f"{_pct(p['heterozygous_substitutions_phased'], s['heterozygous'])} of heterozygous substitutions phased; {sets}")
        )
    else:
        rows.append(("phasing", "none"))
    for key, name in (("gq", "genotype quality"), ("dp", "read depth")):
        q = _quality(d[key])
        if q:
            rows.append((name, q))
    if c["filtered"]:
        rows.append(("filtered", f"{c['filtered']:,} bases"))
    return [f"{rows[0][0]} · {rows[0][1]}"] + [f"  {k:<18}{v}" for k, v in rows[1:]]


def _label(r: dict) -> str:
    return f"{r['contig']}:{r['start']:,}-{r['end']:,}"


def format_stats(result: dict) -> str:
    """The result of :func:`stats` as a table for people."""
    head = (
        result["file"]
        + (f" · sample {result['sample']}" if result["sample"] else "")
        + (f" · {result['reference']}" if result["reference"] else "")
    )
    regs = result["regions"]
    if len(regs) == 1:
        return "\n".join([head, "", *_block(_label(regs[0]), regs[0])])
    width = max(len(_label(r)) for r in regs) + 2
    lines = [
        head,
        "",
        f"{'region':<{width}}{'bases':>14}{'called':>9}{'het subs':>10}{'hom subs':>10}{'ins':>8}{'del':>8}{'Ti/Tv':>7}{'phased':>8}",
    ]
    for r in regs:
        c, s = r["calls"], r["substitutions"]
        if c is None:
            lines.append(f"{_label(r):<{width}}{r['bases']:>14,}  reference only")
            continue
        ins, dele = r["insertions"], r["deletions"]
        phased = r["phasing"]["fraction"]
        titv = f"{s['ti_tv']:.2f}" if s["ti_tv"] else "–"
        lines.append(
            f"{_label(r):<{width}}{r['bases']:>14,}{100 * c['called'] / r['bases']:>8.1f}%{s['heterozygous']:>10,}{s['homozygous']:>10,}"
            f"{ins['heterozygous'] + ins['homozygous']:>8,}{dele['heterozygous'] + dele['homozygous']:>8,}"
            f"{titv:>7}{'–' if phased is None else f'{100 * phased:.0f}%':>8}"
        )
    return "\n".join([*lines, "", *_block(f"all {len(regs)} regions", result["total"])])
