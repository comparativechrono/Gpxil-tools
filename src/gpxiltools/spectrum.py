"""Substitution spectra in regions against the rest of a genome, context by context (``gpxil spectrum``).

Do the sample's substitutions in a set of regions (GA-sequences, Alu tails,
promoters ...) come as often as their sequence leads one to expect? A base's
chance of a substitution depends on its neighbours (the C of a CpG changes
about ten times as often as other C's), so the regions' expected count is
built context by context: for each context (the base and ``k // 2`` bases
either side; 3 bases by default) and each substitution in it, its rate over
the background (by default every called base outside the regions), times how
many bases of the regions have that context. The observed count over the
expected is the ratio; its 95 % interval and p-value come from an exact
binomial test of the regions' substitutions among all those counted (in the
regions and the background), which allows for the background's counts
being a sample too.

Contexts are read on the strand where the middle base is a pyrimidine (C or
T: the 96 classes of mutational signatures, with ``k`` 3) or, with
``centre='purine'``, a purine (A or G: the strand of a GA-sequence). With
``stranded``, the regions' substitutions are read on each feature's own
strand (12 kinds, A>C to T>G; a feature without a strand on the + strand),
the background's rates staying blind to strands, which shows an asymmetry
between the strands (transcription's, say); a base under features on both
strands counts once on each.

Which bases count: those where the sample's genotype is known on both
haplotypes, and its reference base and its context are all A, C, G and T.
In a file written with regions called with confidence, calls outside them
(islands among bases not called: spec 6.1) are left out, as ``gpxil
compare`` leaves them out (``all_calls`` keeps them); ``callable`` (a BED)
narrows the bases further.

``isolated``: only substitutions with no other variant (a substitution, a
deleted base, an insertion, a COMPLEX allele; either haplotype) within that
many bases either side count, in the regions and the background alike, so
that a change written as several substitutions (in a repeat, say) doesn't
count as several. A site with two different alternate bases counts as two
substitutions.

Also counted: transitions and transversions; heterozygous and homozygous
(or haploid) alleles; insertions and deletions per called base: a base
counts once if a deletion begins there on either haplotype (each
haplotype's own: two different deletions that overlap are two) or an
insertion follows it.
"""

from __future__ import annotations

import math

import numpy as np

from . import codes as C
from .features import CHUNK, Intervals, by_sequence, targets

ACGT = "ACGT"
CENTRES = ("pyrimidine", "purine")
IDX4 = np.full(256, -1, dtype=np.int64)  #: a base's code -> 0..3 for A, C, G, T (3 - i its complement); -1 else
for _i, _b in enumerate((C.BASE_A, C.BASE_C, C.BASE_G, C.BASE_T)):
    IDX4[_b] = _i
_IS_SUB = np.zeros(256, dtype=bool)
_IS_SUB[list(C.HAP_SUB_CODES)] = True
_INS = C.FLAG_INS1 | C.FLAG_INS2
_TRANSITION = {(0, 2), (2, 0), (1, 3), (3, 1)}


# ------------------------------------------------------------------ exact tests


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta function (Numerical Recipes, 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1_000_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(ln) * _betacf(a, b, x) / a
    return 1.0 - math.exp(ln) * _betacf(b, a, 1.0 - x) / b


def _beta_quantile(q: float, a: float, b: float) -> float:
    """The x where I_x(a, b) = q, by bisection on the log-odds."""
    lo, hi = -750.0, 750.0
    for _ in range(200):
        mid = (lo + hi) / 2
        x = 1.0 / (1.0 + math.exp(-mid))
        if betainc(a, b, x) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 1.0 / (1.0 + math.exp(-(lo + hi) / 2))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return betainc(n - k, k + 1, 1.0 - p)


def binom_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper and Pearson's exact interval for a binomial proportion, k of n."""
    alpha = 1.0 - level
    lo = 0.0 if k == 0 else _beta_quantile(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_quantile(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def binom_test(k: int, n: int, p: float) -> float:
    """The two-sided p-value of k of n against p: twice the smaller tail (at most 1). Each tail is computed as itself,
    P(X <= k) = I_{1-p}(n - k, k + 1) and P(X >= k) = I_p(k, n - k + 1), so a small one isn't lost to rounding."""
    if n == 0:
        return 1.0
    low = binom_cdf(k, n, p)
    high = betainc(k, n - k + 1, p) if k > 0 else 1.0
    return min(1.0, 2.0 * min(low, high))


def rate_ratio(observed: int, background: int, weight: float, level: float = 0.95) -> dict:
    """How many times as often something happens in the regions as the background predicts, with an exact interval
    and p-value: ``observed`` events in the regions, ``background`` in the background, and ``weight`` the regions'
    events the background predicts for each of its own (the expected count over the background's count). The test is
    of the regions' share of all the events: ``observed`` of ``observed + background``, against
    ``weight / (1 + weight)``."""
    if background == 0 or weight <= 0:
        return {"ratio": None, "interval": None, "p": None}
    n = observed + background
    lo, hi = binom_interval(observed, n, level)

    def ratio(p: float) -> float | None:
        return None if p >= 1.0 else p / (1.0 - p) / weight

    return {
        "ratio": observed / (background * weight),
        "interval": [ratio(lo), ratio(hi)],
        "p": binom_test(observed, n, weight / (1.0 + weight)),
    }


# ------------------------------------------------------------------ counting


def _rc_table(k: int) -> np.ndarray:
    """Context index (k bases, the first the most significant, 4 a digit) -> its reverse complement's."""
    idx = np.arange(4**k)
    out = np.zeros(4**k, dtype=np.int64)
    for j in range(k):
        out += (3 - (idx // 4**j) % 4) * 4 ** (k - 1 - j)
    return out


#: a reference code (soft-masked or not) -> 0..3 for A, C, G, T; -1 for anything else
_BASE4 = IDX4[np.arange(256) & 0x0F].astype(np.int8)
#: a gt byte -> the same with reserved states read as UNKNOWN (spec 10, reader rule 4)
_NORMAL_GT = ((C.normalize_states(np.arange(256) >> 4).astype(np.uint16) << 4) | C.normalize_states(np.arange(256) & 0x0F)).astype(np.uint8)
_VARIES = np.zeros(16, dtype=bool)  #: a haplotype's state differs from the reference (a substitution, DEL, COMPLEX)
_VARIES[[x for x in C.VALID_HAP_STATES if x not in (C.HAP_REF, C.HAP_UNKNOWN, C.HAP_ABSENT)]] = True
_VARIANT_GT = _VARIES[_NORMAL_GT >> 4] | _VARIES[_NORMAL_GT & 0x0F]  #: a gt byte -> either haplotype varies


class _Counts:
    """Counts over a set of bases: bases by context; substitution alleles by context and alternate base (all, and
    homozygous); called A/C/G/T bases and the insertion and deletion events at them."""

    def __init__(self, k: int):
        n = 4**k
        self.bases = np.zeros(n, dtype=np.int64)
        self.subs = np.zeros((n, 4), dtype=np.int64)
        self.hom = np.zeros((n, 4), dtype=np.int64)
        self.called = 0
        self.indels = 0

    def add_sites(self, ctx, keep, alleles) -> None:
        """Substitution sites at contexts ``ctx`` (those ``keep`` counts); ``alleles``: ``(present, alt, hom)``
        arrays over the sites, each allele of them."""
        n = self.bases.size
        for present, alt, hom in alleles:
            sel = keep & present
            if sel.any():
                idx = ctx[sel] * 4 + alt[sel]
                self.subs += np.bincount(idx, minlength=4 * n).reshape(n, 4)
                h = hom[sel]
                if h.any():
                    self.hom += np.bincount(idx[h], minlength=4 * n).reshape(n, 4)


class SpectrumCounter:
    """Counts a sample's substitutions in regions and a background, chunk by chunk (see the module's notes)."""

    def __init__(self, k: int = 3, *, centre: str = "pyrimidine", isolated: int | None = None, stranded: bool = False):
        if k < 1 or k % 2 == 0 or k > 9:
            raise ValueError("the context is an odd number of bases, 1 to 9 (3: a base and its neighbours)")
        if centre not in CENTRES:
            raise ValueError(f"the centre is one of {', '.join(CENTRES)}")
        if isolated is not None and isolated < 1:
            raise ValueError("--isolated takes a distance of 1 base or more")
        self.k, self.h, self.centre, self.isolated, self.stranded = k, k // 2, centre, isolated, stranded
        self.rc = _rc_table(k).astype(np.int32)
        mid = (np.arange(4**k) // 4**self.h) % 4
        self.mid = mid
        self.flip = np.isin(mid, (0, 2) if centre == "pyrimidine" else (1, 3))  #: contexts read on the other strand
        self.canonical = np.where(self.flip, self.rc, np.arange(4**k)).astype(np.int32)
        self.margin = max(self.h, isolated or 0) + 1
        self.inside, self.background = _Counts(k), _Counts(k)
        self.on_strands = _Counts(k) if stranded else None
        self.notes: dict = {}  #: what was counted where (set by :func:`count_spectrum`)

    def add_chunk(self, ref, gt, flags, called, inside, background, plus=None, minus=None, *, unknown_before: bool = False) -> None:
        """Counts over the middle of padded arrays (``self.margin`` bases more either side, which give contexts and
        neighbours): ``called`` the bases that count, ``inside`` and ``background`` the regions' and background's
        bases (over the middle only); ``plus`` and ``minus``, with ``stranded``, the regions' bases on each strand.
        ``unknown_before``: the base before the middle isn't known (a window's first base), so a deletion there
        isn't counted as beginning there."""
        m, k, h = self.margin, self.k, self.h
        n = ref.size - 2 * m
        mid = slice(m, m + n)
        nctx = 4**k
        r4 = _BASE4[ref]
        ctx = r4[m - h : m - h + n].astype(np.int32)
        ok = ctx >= 0
        for t in range(1, k):
            v = r4[m - h + t : m - h + t + n]
            ok &= v >= 0
            ctx = ctx * 4 + v
        ctx[~ok] = 0
        g = _NORMAL_GT[gt]
        h1, h2 = g >> 4, g & 0x0F
        keep = called[mid] & ok
        canon = self.canonical[ctx]
        acgt = r4[mid] >= 0
        d1, d2 = h1 == C.HAP_DEL, h2 == C.HAP_DEL
        indel = (d1[mid] & ~d1[m - 1 : m - 1 + n]) | (d2[mid] & ~d2[m - 1 : m - 1 + n]) | ((flags[mid] & _INS) != 0)
        if unknown_before and n:
            indel[0] = (flags[m] & _INS) != 0
        # (the substitution sites, which are few: their alleles)
        x1, x2 = h1[mid], h2[mid]
        site = np.flatnonzero(_IS_SUB[x1] | _IS_SUB[x2])
        if self.isolated and site.size:
            var = np.flatnonzero(_VARIANT_GT[g] | ((flags & _INS) != 0))  # (at the padded arrays' offsets)
            at = site + m
            near = np.searchsorted(var, at + self.isolated, side="right") - np.searchsorted(var, at - self.isolated, side="left")
            site = site[near == 1]  # (no variant near it but itself)
        s1, s2 = x1[site], x2[site]
        sub1, sub2 = _IS_SUB[s1], _IS_SUB[s2]
        second = sub2 & ~(sub1 & (s2 == s1))  # (a second allele: hap 2's, if it isn't hap 1's)
        hom = (sub1 & sub2 & (s1 == s2)) | (sub1 & (s2 == C.HAP_ABSENT)) | (sub2 & (s1 == C.HAP_ABSENT))
        a1, a2 = np.maximum(_BASE4[s1], 0).astype(np.int64), np.maximum(_BASE4[s2], 0).astype(np.int64)
        sctx = ctx[site].astype(np.int64)
        flipped = self.flip[sctx]
        cc = self.canonical[sctx].astype(np.int64)
        alleles = [(sub1, np.where(flipped, 3 - a1, a1), hom), (second, np.where(flipped, 3 - a2, a2), hom & ~sub1)]
        base_ok = called[mid] & acgt
        for counts, where in ((self.inside, inside), (self.background, background)):
            w = keep & where
            counts.bases += np.bincount(canon[w], minlength=nctx)
            counts.add_sites(cc, w[site], alleles)
            b = where & base_ok
            counts.called += int(np.count_nonzero(b))
            counts.indels += int(np.count_nonzero(b & indel))
        if self.on_strands is not None:
            st = self.on_strands
            wp, wm = keep & plus, keep & minus
            st.bases += np.bincount(ctx[wp], minlength=nctx) + np.bincount(self.rc[ctx[wm]], minlength=nctx)
            raw = [(sub1, a1, hom), (second, a2, hom & ~sub1)]
            st.add_sites(sctx, wp[site], raw)
            st.add_sites(self.rc[sctx].astype(np.int64), wm[site], [(p, 3 - a, z) for p, a, z in raw])

    # ------------------------------------------------------------------ results

    def _name(self, c: int, a: int | None = None) -> str:
        """A context's letters, or a substitution in its context: 'A[C>T]G'."""
        s = "".join(ACGT[(c // 4 ** (self.k - 1 - t)) % 4] for t in range(self.k))
        if a is None:
            return s
        return f"{s[: self.h]}[{s[self.h]}>{ACGT[a]}]{s[self.h + 1 :]}"

    def types(self, stranded: bool = False) -> list[tuple[int, int]]:
        """The kinds of substitution, as (middle base, alternate base): 6 read on the centre's strand, or 12."""
        mids = range(4) if stranded else ((1, 3) if self.centre == "pyrimidine" else (0, 2))
        return [(m, a) for m in mids for a in range(4) if a != m]

    def expected(self) -> tuple[np.ndarray, np.ndarray | None]:
        """The regions' expected substitutions, by context and alternate base, from the background's rates: as
        counted (canonical contexts), and on the features' strands (with ``stranded``)."""
        bg = self.background
        rate = bg.subs / np.maximum(bg.bases, 1)[:, None]
        exp = self.inside.bases[:, None] * rate
        on = None
        if self.on_strands is not None:
            alt = np.arange(4)
            canon_alt = np.where(self.flip[:, None], 3 - alt[None, :], alt[None, :])
            on = self.on_strands.bases[:, None] * rate[self.canonical[:, None], canon_alt]
        return exp, on

    def results(self) -> dict:
        ins, bg = self.inside, self.background
        exp, on_exp = self.expected()
        out = {
            **self.notes,
            "context": self.k,
            "centre": self.centre,
            "isolated": self.isolated,
            "bases": {"regions": int(ins.bases.sum()), "background": int(bg.bases.sum())},
        }
        o, e, nb = int(ins.subs.sum()), float(exp.sum()), int(bg.subs.sum())
        out["substitutions"] = {
            "observed": o,
            "expected": round(e, 3),
            **_rounded(rate_ratio(o, nb, e / nb if nb else 0.0)),
            "background": nb,
            "per_kb": _per_kb(o, ins.bases.sum()),
            "background_per_kb": _per_kb(nb, bg.bases.sum()),
        }
        out["types"] = {}
        for m, a in self.types():
            sel = self.mid == m
            ot, et, nt = int(ins.subs[sel, a].sum()), float(exp[sel, a].sum()), int(bg.subs[sel, a].sum())
            out["types"][f"{ACGT[m]}>{ACGT[a]}"] = {
                "observed": ot,
                "expected": round(et, 3),
                **_rounded(rate_ratio(ot, nt, et / nt if nt else 0.0)),
                "background": nt,
            }
        if self.on_strands is not None:
            st = self.on_strands
            out["on_feature_strands"] = {"bases": int(st.bases.sum()), "types": {}}
            for m, a in self.types(stranded=True):
                sel = self.mid == m
                ot, et = int(st.subs[sel, a].sum()), float(on_exp[sel, a].sum())
                cm, ca = (3 - m, 3 - a) if self.flip[m * 4**self.h] else (m, a)  # (the background's kind)
                nt = int(bg.subs[self.mid == cm, ca].sum())
                out["on_feature_strands"]["types"][f"{ACGT[m]}>{ACGT[a]}"] = {
                    "observed": ot,
                    "expected": round(et, 3),
                    **_rounded(rate_ratio(ot, nt, et / nt if nt else 0.0)),
                }
        for name, cnt in (("regions", ins), ("background", bg)):
            ti = sum(int(cnt.subs[self.mid == m, a].sum()) for m, a in self.types() if (m, a) in _TRANSITION)
            tv = int(cnt.subs.sum()) - ti
            hom = int(cnt.hom.sum())
            out.setdefault("ti_tv", {})[name] = {"transitions": ti, "transversions": tv, "ratio": round(ti / tv, 4) if tv else None}
            out.setdefault("zygosity", {})[name] = {
                "heterozygous": int(cnt.subs.sum()) - hom,
                "homozygous": hom,
                "homozygous_share": round(hom / cnt.subs.sum(), 4) if cnt.subs.sum() else None,
            }
        out["indels"] = {
            "regions": ins.indels,
            "background": bg.indels,
            "called": {"regions": ins.called, "background": bg.called},
            "per_kb": _per_kb(ins.indels, ins.called),
            "background_per_kb": _per_kb(bg.indels, bg.called),
            **_rounded(rate_ratio(ins.indels, bg.indels, ins.called / bg.called if bg.called else 0.0)),
        }
        return out

    def table(self) -> list[dict]:
        """By context and substitution (canonical contexts): the regions' bases, observed and expected substitutions;
        the background's bases and substitutions."""
        exp, _ = self.expected()
        rows = []
        for m, a in self.types():
            for c in np.flatnonzero(self.mid == m).tolist():
                rows.append(
                    {
                        "substitution": f"{ACGT[m]}>{ACGT[a]}",
                        "context": self._name(c, a),
                        "region_bases": int(self.inside.bases[c]),
                        "region_observed": int(self.inside.subs[c, a]),
                        "region_expected": round(float(exp[c, a]), 4),
                        "background_bases": int(self.background.bases[c]),
                        "background_observed": int(self.background.subs[c, a]),
                    }
                )
        return rows


def _per_kb(n, bases) -> float | None:
    return round(1000.0 * float(n) / float(bases), 5) if bases else None


def _rounded(d: dict) -> dict:
    out = dict(d)
    if out.get("ratio") is not None:
        out["ratio"] = round(out["ratio"], 4)
    if out.get("interval"):
        out["interval"] = [None if x is None else round(x, 4) for x in out["interval"]]
    if out.get("p") is not None:
        out["p"] = float(f"{out['p']:.4g}")
    return out


def _islands_dropped(gfile, all_calls: bool) -> bool:
    s = gfile.metadata.get("sample")
    return not all_calls and bool(isinstance(s, dict) and s.get("callable_regions"))


def called_bases(view, a: int, b: int, gt: np.ndarray, flags: np.ndarray, drop: bool, block: int = 1 << 16) -> np.ndarray:
    """Whether each base of ``[a, b)`` of a view (whose gt and flags there are given) is called: known on both
    haplotypes; and, with ``drop``, not in an island, a run of called bases that calls cover entirely, outside the
    regions called with confidence (spec 6.1). A run is judged whole, read past ``a`` and ``b`` as far as it
    reaches (or until a base that no call covers shows it isn't an island), so where chunks end doesn't matter."""
    from .compare import _drop_islands

    def known(g, fl):
        g = _NORMAL_GT[g]
        return ((g >> 4) != C.HAP_UNKNOWN) & ((g & 0x0F) != C.HAP_UNKNOWN), (fl & C.FLAG_VARIANT_SITE) != 0

    called, site = known(gt, flags)
    if not drop or not called.size:
        return called
    has_flags = view.has("flags")

    def piece(x, y):
        fl = view.read("flags", x, y) if has_flags else np.zeros(y - x, dtype=np.uint8)
        return known(view.read("gt", x, y), fl)

    left, right = [], []
    if called[0]:  # (the run goes on before a: to its start, or to a base no call covers)
        x = a
        while x > 0:
            y, x = x, max(0, x - block)
            c, st = piece(x, y)
            stop = np.flatnonzero(~c | ~st)
            if stop.size:
                k = int(stop[-1])
                left.insert(0, (c[k:], st[k:]))
                break
            left.insert(0, (c, st))
    if called[-1]:  # (and after b)
        y = b
        while y < view.length:
            x, y = y, min(view.length, y + block)
            c, st = piece(x, y)
            stop = np.flatnonzero(~c | ~st)
            if stop.size:
                k = int(stop[0])
                right.append((c[: k + 1], st[: k + 1]))
                break
            right.append((c, st))
    if not left and not right:
        _drop_islands(called, site)
        return called
    n = sum(c.size for c, _ in left)
    every = np.concatenate([c for c, _ in left] + [called] + [c for c, _ in right])
    sites = np.concatenate([x for _, x in left] + [site] + [x for _, x in right])
    _drop_islands(every, sites)
    return every[n : n + called.size]


def count_spectrum(
    gfile,
    features,
    within=None,
    *,
    background=None,
    callable=None,
    k: int = 3,
    centre: str = "pyrimidine",
    isolated: int | None = None,
    stranded: bool = False,
    all_calls: bool = False,
    chunk: int = CHUNK,
) -> SpectrumCounter:
    """The sample's substitutions counted in the regions ``features`` (:class:`~gpxiltools.features.Feature`) and a
    background (see the module's notes): a :class:`SpectrumCounter`, whose ``results()`` compares them and whose
    ``table()`` gives them by context. ``within``: where to count (``(name, start, end)`` as
    ``GpxilFile.region_of`` gives them; default every sequence of the file); ``background``: features whose bases
    (outside the regions) are the background (default: every base outside the regions); ``callable``: features
    whose bases alone count."""
    if gfile.sample_id is None:
        raise ValueError(f"{gfile.path} has no sample: a spectrum is of a sample's substitutions")
    counter = SpectrumCounter(k, centre=centre, isolated=isolated, stranded=stranded)
    regions, unplaced = by_sequence(gfile, features)
    plus = minus = None
    if stranded:
        plus, _ = by_sequence(gfile, features, strand="+")
        minus, _ = by_sequence(gfile, features, strand="-")
    bg = by_sequence(gfile, background)[0] if background is not None else None
    ok = by_sequence(gfile, callable)[0] if callable is not None else None
    drop = _islands_dropped(gfile, all_calls)
    m = counter.margin
    none = Intervals()
    scanned = 0
    for view, s, e in targets(gfile, within):
        if not view.has("gt"):
            continue
        if not view.covers("ref", s, e):
            raise ValueError(f"{view.region(s, e)} has no reference: give it with --reference")
        seq, off = view.parent, view.start
        reg = regions.get(seq, none)
        for lo in range(s, e, chunk):
            hi = min(e, lo + chunk)
            a, b = max(0, lo - m), min(view.length, hi + m)
            pad_l, pad_r = m - (lo - a), m - (b - hi)

            def padded(x, fill, pad_l=pad_l, pad_r=pad_r):
                return np.concatenate((np.full(pad_l, fill, dtype=x.dtype), x, np.full(pad_r, fill, dtype=x.dtype)))

            gt0 = view.read("gt", a, b)
            fl0 = view.read("flags", a, b) if view.has("flags") else np.zeros(b - a, dtype=np.uint8)
            ref = padded(view.read("ref", a, b), 0)
            gt = padded(gt0, C.GT_UNKNOWN)
            flags = padded(fl0, 0)
            called = padded(called_bases(view, a, b, gt0, fl0, drop), False)
            if ok is not None:
                called &= ok.get(seq, none).mask(off + lo - m, off + hi + m)
            inside = reg.mask(off + lo, off + hi)
            back = ~inside if bg is None else bg.get(seq, none).mask(off + lo, off + hi) & ~inside
            pl = mi = None
            if stranded:
                pl = plus.get(seq, none).mask(off + lo, off + hi)
                mi = minus.get(seq, none).mask(off + lo, off + hi)
            # (a window's first base: whether a deletion begins there or before it, the file can't say)
            counter.add_chunk(ref, gt, flags, called, inside, back, pl, mi, unknown_before=lo == 0 and off > 0)
            scanned += hi - lo
    counter.notes = {
        "features": {"given": len(features), "on_no_sequence": unplaced, "region_bases": sum(r.bases for r in regions.values())},
        "scanned_bases": scanned,
        "calls_outside_confident_regions": "left out" if drop else "kept",
    }
    return counter


def spectrum(gfile, features, within=None, **kw) -> dict:
    """:func:`count_spectrum`'s results: observed and expected substitutions, by kind, and more (see the module's
    notes)."""
    return count_spectrum(gfile, features, within, **kw).results()


def format_results(d: dict) -> str:
    """The results as text."""
    s = d["substitutions"]
    centre = "pyrimidine (C or T)" if d["centre"] == "pyrimidine" else "purine (A or G)"
    lines = [
        f"bases counted: {d['bases']['regions']:,} in the regions, {d['bases']['background']:,} in the background"
        f" (called, with an A/C/G/T context of {d['context']}; read with a {centre} in the middle)",
    ]
    iso = f", isolated (no other variant within {d['isolated']} bases)" if d["isolated"] else ""
    lines.append(
        f"substitutions{iso}: {s['observed']:,} in the regions, {s['expected']:,.1f} expected from the background's rates"
        f" by context: {_ratio(s)}"
    )
    lines.append(f"  per kb: {s['per_kb']} in the regions, {s['background_per_kb']} in the background")
    lines.append("  by kind (observed / expected, ratio, 95 % interval, p):")
    for t, v in d["types"].items():
        lines.append(f"    {t}  {v['observed']:>9,} / {v['expected']:>11,.1f}  {_ratio(v)}")
    if "on_feature_strands" in d:
        lines.append("  on the features' strands:")
        for t, v in d["on_feature_strands"]["types"].items():
            lines.append(f"    {t}  {v['observed']:>9,} / {v['expected']:>11,.1f}  {_ratio(v)}")
    tt = d["ti_tv"]
    lines.append(
        "transitions / transversions: "
        + "; ".join(f"{k} {v['transitions']:,} / {v['transversions']:,} = {v['ratio']}" for k, v in tt.items())
    )
    z = d["zygosity"]
    lines.append("homozygous share: " + "; ".join(f"{k} {v['homozygous_share']}" for k, v in z.items()))
    i = d["indels"]
    lines.append(
        f"insertions and deletions: {i['regions']:,} in the regions ({i['per_kb']} per kb called), {i['background']:,} in the"
        f" background ({i['background_per_kb']} per kb): {_ratio(i)}"
    )
    return "\n".join(lines)


def _ratio(v: dict) -> str:
    if v.get("ratio") is None:
        return "ratio n/a"
    lo, hi = v["interval"]
    hi_t = "inf" if hi is None else f"{hi:.3g}"
    p = "p < 1e-300" if v["p"] == 0 else f"p = {v['p']:.3g}"  # (too small for a float)
    return f"{v['ratio']:.3g} times (95 % {lo:.3g}-{hi_t}; {p})"


def write_table(out, rows: list[dict]) -> None:
    keys = list(rows[0]) if rows else ["substitution", "context"]
    out.write("\t".join(keys) + "\n")
    for r in rows:
        out.write("\t".join(str(r[k]) for k in keys) + "\n")
