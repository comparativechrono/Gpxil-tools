"""0.8.0: gpxil spectrum, against counts made from the sites as text."""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest
from simulate import fasta_text

from gpxiltools.cli import main
from gpxiltools.container import GpxilFile
from gpxiltools.encode import encode
from gpxiltools.features import Feature, Intervals
from gpxiltools.spectrum import binom_interval, binom_test, count_spectrum, rate_ratio, spectrum

COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def _rc(s: str) -> str:
    return "".join(COMP[b] for b in reversed(s))


def _make(seed: int = 5, n: int = 9000):
    """A reference and a sample: SNVs (heterozygous, homozygous, two alternates), deletions, insertions and no calls,
    apart; and what each base is (the oracle)."""
    rng = random.Random(seed)
    ref = list("".join(rng.choice("ACGT") for _ in range(n)))
    for i in range(3000, 3040):
        ref[i] = "N"
    for i in range(500, 900):
        ref[i] = ref[i].lower()
    for i in range(1200, 1260, 2):  # (CpGs, which change more)
        ref[i], ref[i + 1] = "C", "G"
    seq = "".join(ref)
    up = seq.upper()
    sites = []  # (pos0, REF, ALT, GT, kind, haplotype alternates)
    pos = 30
    while pos < n - 40:
        if "N" in up[pos - 8 : pos + 12]:
            pos += 23
            continue
        kind = rng.choice(["snv"] * 6 + ["multi", "del", "ins", "nocall", "half"])
        r = up[pos]
        if kind in ("snv", "nocall", "half"):
            alt = rng.choice([b for b in "ACGT" if b != r])
            gt = {"snv": rng.choice(["0|1", "1|0", "1|1"]), "nocall": "./.", "half": "./1"}[kind]
            sites.append((pos, r, alt, gt, kind, None))
            pos += 1
        elif kind == "multi":
            a1, a2 = rng.sample([b for b in "ACGT" if b != r], 2)
            sites.append((pos, r, f"{a1},{a2}", "1|2", kind, (a1, a2)))
            pos += 1
        elif kind == "del":
            k = rng.randint(1, 5)
            sites.append((pos, up[pos : pos + k + 1], r, rng.choice(["0|1", "1|1"]), kind, k))
            pos += k + 1
        else:
            ins = "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 4)))
            sites.append((pos, r, r + ins, rng.choice(["1|0", "1|1"]), kind, ins))
            pos += 1
        pos += rng.randint(6, 40)
    return seq, sites


def _oracle(seq: str, sites, inside, *, k=3, centre="pyrimidine", isolated=None, called=None, stranded_minus=None):
    """Counts as text: bases by canonical context, substitution alleles by (context, alt), and indels; in and outside
    ``inside`` (a set of positions); ``called``: the positions that count (default: all but no calls)."""
    n, h = len(seq), k // 2
    up = seq.upper()
    nocall = {p for p, _, _, gt, kind, _ in sites if kind in ("nocall", "half")}
    if called is None:
        called = set(range(n)) - nocall
    variant = set()  # (bases with a variant: for isolation)
    alleles = {}  # pos -> [(alt, hom)]
    indel = set()
    for p, _r, alt, gt, kind, extra in sites:
        if kind == "snv":
            alleles[p] = [(alt, gt == "1|1")]
            variant.add(p)
        elif kind == "multi":
            alleles[p] = [(extra[0], False), (extra[1], False)]
            variant.add(p)
        elif kind == "half":
            variant.add(p)  # (a substitution on one haplotype, unknown on the other: not called, but a variant)
        elif kind == "del":
            variant.update(range(p + 1, p + 1 + extra))
            indel.add(p + 1)
        elif kind == "ins":
            variant.add(p)
            indel.add(p)
    flip_mids = "AG" if centre == "pyrimidine" else "CT"
    bases = {"in": Counter(), "bg": Counter()}
    subs = {"in": Counter(), "bg": Counter()}
    on_strand = {"bases": Counter(), "subs": Counter()}
    ind = {"in": 0, "bg": 0}
    callable_n = {"in": 0, "bg": 0}
    homs = {"in": 0, "bg": 0}
    for p in range(n):
        where = "in" if p in inside else "bg"
        if p not in called:
            continue
        if up[p] in "ACGT":
            callable_n[where] += 1
            ind[where] += p in indel
        if p - h < 0 or p + h >= n:
            continue
        ctx = up[p - h : p + h + 1]
        if any(b not in "ACGT" for b in ctx):
            continue
        flip = ctx[h] in flip_mids
        canon = _rc(ctx) if flip else ctx
        bases[where][canon] += 1
        alone = isolated is None or not any(q in variant for q in range(p - isolated, p + isolated + 1) if q != p)
        for alt, hom in alleles.get(p, []):
            if alone:
                subs[where][(canon, COMP[alt] if flip else alt)] += 1
                homs[where] += hom
        if stranded_minus is not None and where == "in":
            minus = p in stranded_minus
            c2 = _rc(ctx) if minus else ctx
            on_strand["bases"][c2] += 1
            for alt, _hom in alleles.get(p, []):
                if alone:
                    on_strand["subs"][(c2, COMP[alt] if minus else alt)] += 1
    _oracle.homs = homs
    return bases, subs, ind, callable_n, on_strand


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    d = tmp_path_factory.mktemp("spectrum")
    seq, sites = _make()
    fa = d / "ref.fa"
    fa.write_text(fasta_text({"chr1": seq}))
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=chr1,length={len(seq)}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
    ]
    lines += [f"chr1\t{p + 1}\t.\t{r}\t{alt}\t.\tPASS\t.\tGT\t{gt}" for p, r, alt, gt, _, _ in sites]
    vcf = d / "s.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    out = d / "s.gpxil.tif"
    encode(str(fa), str(out), vcf=str(vcf), width=200)
    return out, seq, sites, d


FEATS = [
    Feature("chr1", 100, 400, "a", None, "+"),
    Feature("chr1", 1150, 1300, "b", None, "-"),
    Feature("chr1", 5000, 7000, "c", None, "-"),
]


def _inside(feats) -> set[int]:
    return {p for f in feats for p in range(f.start, f.end)}


@pytest.mark.parametrize(
    "k,centre,isolated,chunk", [(3, "pyrimidine", None, 1 << 22), (1, "purine", None, 1000), (5, "pyrimidine", 6, 333)]
)
def test_counts_are_the_oracles(made, k, centre, isolated, chunk):
    path, seq, sites, _ = made
    inside = _inside(FEATS)
    bases, subs, ind, called_n, _ = _oracle(seq, sites, inside, k=k, centre=centre, isolated=isolated)
    with GpxilFile(path) as g:
        c = count_spectrum(g, FEATS, k=k, centre=centre, isolated=isolated, chunk=chunk)
    for where, cnt in (("in", c.inside), ("bg", c.background)):
        got_bases = {c._name(i): int(v) for i, v in enumerate(cnt.bases) if v}
        assert got_bases == dict(bases[where])
        got_subs = {(c._name(i), "ACGT"[a]): int(cnt.subs[i, a]) for i in range(cnt.bases.size) for a in range(4) if cnt.subs[i, a]}
        assert got_subs == dict(subs[where])
        assert cnt.indels == ind[where] and cnt.called == called_n[where] and int(cnt.hom.sum()) == _oracle.homs[where]
    d = c.results()
    assert d["substitutions"]["observed"] == sum(subs["in"].values()) and d["substitutions"]["background"] == sum(subs["bg"].values())
    # (the expected count, by hand)
    rate = {key: v / bases["bg"][key[0]] for key, v in subs["bg"].items()}
    want = sum(bases["in"][ctx] * r for (ctx, _), r in rate.items())
    assert d["substitutions"]["expected"] == pytest.approx(want, abs=1e-3)
    assert sum(t["observed"] for t in d["types"].values()) == d["substitutions"]["observed"]
    assert d["features"]["region_bases"] == len(inside) and d["features"]["on_no_sequence"] == 0


def test_strands_callable_and_within(made):
    path, seq, sites, _ = made
    inside = _inside(FEATS)
    minus = _inside([f for f in FEATS if f.strand == "-"])
    _, _, _, _, on = _oracle(seq, sites, inside, stranded_minus=minus)
    with GpxilFile(path) as g:
        c = count_spectrum(g, FEATS, stranded=True)
        st = c.on_strands
        assert {c._name(i): int(v) for i, v in enumerate(st.bases) if v} == dict(on["bases"])
        got = {(c._name(i), "ACGT"[a]): int(st.subs[i, a]) for i in range(st.bases.size) for a in range(4) if st.subs[i, a]}
        assert got == dict(on["subs"])
        d = c.results()
        assert len(d["on_feature_strands"]["types"]) == 12
        assert sum(t["observed"] for t in d["on_feature_strands"]["types"].values()) == d["substitutions"]["observed"]
        # (callable bases, and counting within a region)
        ok = [Feature("chr1", 0, 4500)]
        nocall = {p for p, _, _, gt, kind, _ in sites if kind in ("nocall", "half")}
        bases, subs, ind, _, _ = _oracle(seq, sites, inside, called=set(range(4500)) - nocall)
        c2 = count_spectrum(g, FEATS, callable=ok)
        assert int(c2.background.bases.sum()) == sum(bases["bg"].values()) and c2.background.indels == ind["bg"]
        c3 = count_spectrum(g, FEATS, [g.region_of("chr1:1-4,500")])  # (the same bases: contexts reach past it)
        for x, y in ((c3.inside, c2.inside), (c3.background, c2.background)):
            assert (x.bases == y.bases).all() and (x.subs == y.subs).all() and (x.indels, x.called) == (y.indels, y.called)
        # (a background of its own: the regions' flanks)
        flanks = [Feature("chr1", 0, 100), Feature("chr1", 400, 500)]
        bases_f, subs_f, _, _, _ = _oracle(seq, sites, inside, called=(set(range(0, 500)) - inside) - nocall)
        c4 = count_spectrum(g, FEATS, background=flanks)
        assert int(c4.background.bases.sum()) == sum(bases_f["bg"].values())
        assert int(c4.background.subs.sum()) == sum(subs_f["bg"].values())


def test_calls_outside_the_confident_regions(made, tmp_path):
    path, seq, sites, d = made
    bed = tmp_path / "callable.bed"
    bed.write_text("chr1\t0\t2500\nchr1\t6000\t9000\n")
    out = tmp_path / "c.gpxil.tif"
    encode(str(d / "ref.fa"), str(out), vcf=str(d / "s.vcf"), callable_bed=str(bed), width=200)
    inside = _inside(FEATS)
    nocall = {p for p, _, _, gt, kind, _ in sites if kind in ("nocall", "half")}
    conf = (set(range(0, 2500)) | set(range(6000, 9000))) - nocall
    bases, subs, ind, _, _ = _oracle(seq, sites, inside, called=conf)
    with GpxilFile(out) as g:
        c = count_spectrum(g, FEATS)
        assert c.notes["calls_outside_confident_regions"] == "left out"
        assert int(c.background.subs.sum()) == sum(subs["bg"].values()) and int(c.inside.subs.sum()) == sum(subs["in"].values())
        kept = count_spectrum(g, FEATS, all_calls=True)
        assert int(kept.background.subs.sum()) > int(c.background.subs.sum())  # (the islands' substitutions)
        # (where chunks end doesn't matter: an island, or a confident region, is judged whole)
        for chunk in (97, 101, 1000):
            small = count_spectrum(g, FEATS, chunk=chunk)
            for x, y in ((small.inside, c.inside), (small.background, c.background)):
                assert (x.bases == y.bases).all() and (x.subs == y.subs).all() and (x.called, x.indels) == (y.called, y.indels)


def test_a_confident_region_that_begins_with_a_call(tmp_path):
    """Calls cover the first bases of a region called with confidence: where a chunk ends among them, they are
    still the start of that region, not an island outside it."""
    rng = random.Random(3)
    seq = "".join(rng.choice("ACGT") for _ in range(600))
    other = {"A": "C", "C": "G", "G": "T", "T": "A"}
    fa = tmp_path / "ref.fa"
    fa.write_text(fasta_text({"chr1": seq}))
    lines = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=chr1,length=600>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
        f"chr1\t101\t.\t{seq[100:103]}\t{''.join(other[b] for b in seq[100:103])}\t.\tPASS\t.\tGT\t0|1",
        f"chr1\t301\t.\t{seq[300]}\t{other[seq[300]]}\t.\tPASS\t.\tGT\t0|1",
    ]
    vcf = tmp_path / "s.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    bed = tmp_path / "conf.bed"
    bed.write_text("chr1\t100\t600\n")
    out = tmp_path / "s.gpxil.tif"
    encode(str(fa), str(out), vcf=str(vcf), callable_bed=str(bed), width=64)
    with GpxilFile(out) as g:
        want = count_spectrum(g, [Feature("chr1", 250, 350)])
        assert int(want.background.subs.sum()) == 3 and want.background.called == 400
        for chunk in (98, 99, 100, 101, 102, 103, 104):
            got = count_spectrum(g, [Feature("chr1", 250, 350)], chunk=chunk)
            assert int(got.background.subs.sum()) == 3 and got.background.called == 400, chunk


def test_exact_tests():
    # (values from scipy.stats: beta.ppf for Clopper and Pearson's interval, and binomtest)
    lo, hi = binom_interval(3, 10)
    assert lo == pytest.approx(0.0667395, rel=1e-5) and hi == pytest.approx(0.652453, rel=1e-5)
    assert binom_interval(0, 10) == (0.0, pytest.approx(0.308497, rel=1e-5))
    lo, hi = binom_interval(1000, 1_000_000)
    assert lo == pytest.approx(0.000939001, rel=1e-5) and hi == pytest.approx(0.00106392, rel=1e-5)
    assert binom_test(500_000, 1_000_000, 0.505) == pytest.approx(1.53e-23, rel=1e-2)
    assert binom_test(3, 10, 0.3) == 1.0
    assert rate_ratio(120, 880, 0.05 / 0.95)["p"] == pytest.approx(5.30296e-18, rel=1e-4)  # (a small upper tail, kept)
    assert binom_test(990, 1000, 0.5) == pytest.approx(2 * 2.48334e-278, rel=1e-4)  # (scipy: binom.sf(989, 1000, 0.5))
    r = rate_ratio(30, 1000, 0.01)  # (30 where the background predicts 10)
    assert r["ratio"] == pytest.approx(3.0) and r["interval"][0] > 1.9 and r["interval"][1] < 4.6 and r["p"] < 1e-6
    assert rate_ratio(5, 0, 1.0)["ratio"] is None


def test_intervals():
    iv = Intervals([(10, 20), (15, 30), (30, 35), (50, 60), (5, 5)])
    assert iv.starts.tolist() == [10, 50] and iv.ends.tolist() == [35, 60] and iv.bases == 35
    m = iv.mask(0, 70)
    assert m.tolist() == [10 <= i < 35 or 50 <= i < 60 for i in range(70)]
    assert iv.mask(36, 50).sum() == 0 and iv.mask(55, 58).all()
    rng = random.Random(3)
    pairs = [(a, a + rng.randint(1, 30)) for a in (rng.randrange(200_000) for _ in range(10_000))]
    want = [False] * 200_100
    for a, b in pairs:
        want[a:b] = [True] * (b - a)
    many = Intervals(pairs)
    assert many.mask(0, 200_100).tolist() == want  # (cut by adding up, with many intervals)
    assert many.mask(150_000, 150_500).tolist() == want[150_000:150_500]  # (a slice each, with few)
    assert many.bases == sum(want)


def test_command_line(made, tmp_path, capsys):
    path, seq, sites, d = made
    bed = tmp_path / "f.bed"
    bed.write_text("".join(f"{f.contig}\t{f.start}\t{f.end}\t{f.name}\t0\t{f.strand}\n" for f in FEATS))
    tsv = tmp_path / "t.tsv"
    assert main(["spectrum", str(path), str(bed), "-o", str(tsv), "--isolated", "5"]) == 0
    out = capsys.readouterr().out
    assert "3 features: 2,450 bases" in out and "expected from the background's rates" in out and "C>T" in out
    rows = tsv.read_text().splitlines()
    assert rows[0].split("\t")[:3] == ["substitution", "context", "region_bases"] and len(rows) == 1 + 96
    assert main(["spectrum", str(path), str(bed), "--json", "--centre", "purine", "--stranded", "--context", "1"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert list(j["types"]) == ["A>C", "A>G", "A>T", "G>A", "G>C", "G>T"] and j["context"] == 1
    with GpxilFile(path) as g:
        assert j["substitutions"] == spectrum(g, FEATS, centre="purine", k=1, stranded=True)["substitutions"]
    assert main(["spectrum", str(path), str(bed), "--context", "4"]) == 1
    assert "odd number" in capsys.readouterr().err
