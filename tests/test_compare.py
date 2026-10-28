"""0.5.0: gpxil compare, two genomes base by base and a child with both parents."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
from simulate import fasta_text

from gpxiltools import codes as C
from gpxiltools.cli import main
from gpxiltools.compare import (
    PAIR_CLASSES,
    TRIO_CLASSES,
    _drop_islands,
    _haplotype_pairs,
    _Ids,
    _slide,
    compare_pair,
    compare_trio,
    format_compare,
    picture,
)
from gpxiltools.container import GpxilFile
from gpxiltools.encode import encode

L = 6000


def reference() -> str:
    """Random bases, with a run of ten As at 1000-1009, (CA)x6 at 2000-2011 and (TG)x15 at 3000-3029 (0-based)."""
    rng = random.Random(5)
    seq = [rng.choice("ACGT") for _ in range(L)]
    for at, s in ((999, "C" + "A" * 10 + "G"), (1999, "T" + "CA" * 6 + "G"), (2999, "C" + "TG" * 15 + "A")):
        seq[at : at + len(s)] = list(s)
    return "".join(seq)


REF = reference()


def snv(pos0: int, gt: str, *ps, alt: str | None = None):
    alt = alt or next(b for b in "ACGT" if b != REF[pos0])
    return (pos0, REF[pos0], alt, gt, *ps)


def write_vcf(path, sample: str, records) -> None:
    head = [
        "##fileformat=VCFv4.3",
        f"##contig=<ID=chr1,length={L}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set">',
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}",
    ]
    body = []
    for pos0, ref, alt, gt, *ps in sorted(records):
        fmt, val = ("GT:PS", f"{gt}:{ps[0]}") if ps else ("GT", gt)
        body.append(f"chr1\t{pos0 + 1}\t.\t{ref}\t{alt}\t50\tPASS\t.\t{fmt}\t{val}")
    path.write_text("\n".join(head + body) + "\n")


def make(tmp_path, name: str, records, *, bed: str | None = None, sample: str | None = None, **kw):
    fa = tmp_path / "ref.fa"
    if not fa.exists():
        fa.write_text(fasta_text({"chr1": REF}))
    vcf = tmp_path / f"{name}.vcf"
    write_vcf(vcf, sample or name, records)
    if bed is not None:
        (tmp_path / f"{name}.bed").write_text(bed)
        kw["callable_bed"] = tmp_path / f"{name}.bed"
    out = tmp_path / f"{name}.gpxil.tif"
    encode(fa, out, vcf=vcf, width=500, log=lambda m: None, **kw)
    return out


# A and B: the same, another genotype, only in one, and three variants written differently in repeats
PAIR_A = [
    snv(100, "0/1"),
    snv(200, "0/1"),
    snv(300, "0/1"),
    (999, "C", "CA", "0/1"),  # an A inserted before the run of As
    (1999, "TCA", "T", "1/1"),  # the first CA deleted
    (2999, "C", "CTG", "1|0", 3000),  # a TG inserted before the repeat, on the haplotype without...
    snv(3010, "0|1", 3000),  # ...this substitution in the repeat
    snv(4100, "0/1"),  # (B didn't call here)
    snv(4200, "0/1"),  # (B's call here is outside its regions called with confidence)
]
PAIR_B = [
    snv(100, "0/1"),
    snv(200, "1/1"),
    snv(400, "1/1"),
    (1009, "A", "AA", "0/1"),  # the A inserted after the run
    (2009, "ACA", "A", "1/1"),  # the last CA deleted
    snv(3010, "0/1"),
    (3029, "G", "GTG", "0/1"),  # the TG inserted after the repeat: the same, phasing not compared
    snv(4200, "0/1"),
]
B_BED = f"chr1\t0\t4000\nchr1\t4500\t{L}\n"


@pytest.fixture
def pair(tmp_path):
    return make(tmp_path, "A", PAIR_A), make(tmp_path, "B", PAIR_B, bed=B_BED)


def test_pair_counts(pair):
    with GpxilFile(pair[0]) as a, GpxilFile(pair[1]) as b:
        r = compare_pair(a, b)
    t = r["total"]
    assert (r["a"]["sample"], r["b"]["sample"], t["regions"], t["bases"]) == ("A", "B", 1, L)
    assert t["called"] == {"both": L - 500, "a_only": 500, "b_only": 0, "neither": 0}  # (nor is B's call at 4200)
    v = t["variant_bases"]
    assert (v["same"], v["written_differently"], v["only_a"], v["only_b"], v["other_genotype"]) == (2, 8, 1, 1, 1)
    assert t["stretches"] == {"written_differently": 3, "different": 3, "not_checked": 0}
    subs, indels = t["events"]["substitutions"], t["events"]["indels"]
    assert subs["a"] == {"total": 6, "found": 2, "missing": 1, "other_genotype": 1, "not_compared": 2}
    assert subs["b"] == {"total": 5, "found": 2, "missing": 1, "other_genotype": 1, "not_compared": 1}
    assert (subs["precision"], subs["recall"], subs["f1"]) == (0.5, 0.5, 0.5)
    assert indels["a"] == {"total": 3, "found": 3, "missing": 0, "other_genotype": 0, "not_compared": 0}
    assert indels["b"] == {"total": 3, "found": 3, "missing": 0, "other_genotype": 0, "not_compared": 0}
    assert indels["precision"] == indels["recall"] == 1.0
    text = format_compare(r)
    assert "written differently 8" in text and "precision 50.00 %" in text


def test_pair_all_calls_and_gap(pair):
    """--all-calls compares B's call outside its callable regions; a gap too small for a stretch splits it."""
    with GpxilFile(pair[0]) as a, GpxilFile(pair[1]) as b:
        r = compare_pair(a, b, all_calls=True)
        subs = r["total"]["events"]["substitutions"]
        assert subs["a"]["found"] == subs["b"]["found"] == 3 and subs["b"]["not_compared"] == 0
        assert r["total"]["called"]["both"] == L - 499
        # a smaller gap still finds the indels in repeats (the stretches widen over the repeat)...
        assert compare_pair(a, b, gap=0)["total"]["stretches"] == {"written_differently": 3, "different": 3, "not_checked": 0}
        # ...and a larger one puts the differences at 200, 300 and 400 in one stretch
        r = compare_pair(a, b, gap=150)
        assert r["total"]["stretches"] == {"written_differently": 3, "different": 1, "not_checked": 0}
        assert r["total"]["variant_bases"]["only_a"] == 1 and r["total"]["events"]["indels"]["a"]["found"] == 3


def test_pair_pieces_and_regions_add_up(pair):
    """Small pieces, and regions cutting through a stretch, count the same as the whole."""
    with GpxilFile(pair[0]) as a, GpxilFile(pair[1]) as b:
        whole = compare_pair(a, b)["total"]
        assert compare_pair(a, b, chunk=64)["total"] == whole
        parts = compare_pair(a, b, [("chr1", 0, 2005), ("chr1", 2005, 3015), ("chr1", 3015, None)])
        assert [x["bases"] for x in parts["regions"]] == [2005, 1010, L - 3015]
        t = parts["total"]
        assert t == {**whole, "regions": 3}


def test_pair_one_copy_equals_two(tmp_path):
    """A haploid call (haplotype 2 ABSENT) is the same as two copies of the allele."""
    a = make(tmp_path, "hap", [snv(100, "1"), (999, "C", "CA", "1")], haploid_contigs=["chr1"])
    b = make(tmp_path, "dip", [snv(100, "1/1"), (1009, "A", "AA", "1/1")])
    with GpxilFile(a) as x, GpxilFile(b) as y:
        t = compare_pair(x, y)["total"]
    assert (t["variant_bases"]["same"], t["variant_bases"]["written_differently"]) == (1, 2)
    assert t["events"]["substitutions"]["precision"] == t["events"]["indels"]["recall"] == 1.0


def test_pair_refuses(tmp_path, pair):
    fa = tmp_path / "other.fa"
    fa.write_text(fasta_text({"chr1": REF[:-1] + ("A" if REF[-1] != "A" else "C")}))
    other = tmp_path / "other.gpxil.tif"
    encode(fa, other, vcf=tmp_path / "A.vcf", log=lambda m: None)
    refonly = tmp_path / "ref.gpxil.tif"
    encode(tmp_path / "ref.fa", refonly, log=lambda m: None)
    with GpxilFile(pair[0]) as a, GpxilFile(other) as o, GpxilFile(refonly) as r:
        with pytest.raises(ValueError, match="not the same reference"):
            compare_pair(a, o)
        with pytest.raises(ValueError, match="no sample"):
            compare_pair(a, r)
        with pytest.raises(IndexError):
            compare_pair(a, a, [("chr1", 10, L + 1)])


def test_pair_picture(pair):
    with GpxilFile(pair[0]) as a, GpxilFile(pair[1]) as b:
        pic = picture([a, b], "chr1", 0, 4300, bases_per_row=100)
    idx = pic.index.reshape(-1)
    want = {100: 1, 300: 2, 400: 3, 200: 4, 999: 5, 2000: 5, 4100: 6, 4200: 6}
    assert {p: int(idx[p]) - 0xF0 for p in want} == want
    assert idx[50] < 0x20  # (the reference, faded)
    assert pic.shape == (43, 100) and [x for x, _ in pic.legend][1:] == [x for x, c in PAIR_CLASSES if c]


def test_drop_islands():
    called = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1], dtype=bool)
    site = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], dtype=bool)
    _drop_islands(called, site)
    assert called.astype(int).tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_slide():
    """How far an insertion or a deletion could be moved along a repeat of the reference."""
    ref4 = np.array([C.ASCII_TO_REF[ord(b)] for b in "GCAAAAGTCACACAT"]) & 0x0F
    assert _slide(ref4, 1, 2, "A") == (1, 6)  # (an A after the C: anywhere in the run of As)
    assert _slide(ref4, 2, 3, None) == (2, 6)  # (one A deleted)
    assert _slide(ref4, 7, 8, "CA") == (7, 14)  # (a CA after the T: anywhere in the CA repeat)
    assert _slide(ref4, 8, 10, None) == (8, 14)  # (a CA deleted)
    assert _slide(ref4, 6, 7, None) == (6, 7)  # (the G: nowhere else)


def test_haplotype_pairs_pieces():
    """Phasing isn't compared: pieces of a stretch are tried either way round, a deletion and the substitution
    next to it apart, and an insertion apart from the base it follows."""
    ids = _Ids()
    ref4 = np.array([1, 2, 4, 8])  # (ACGT)
    k1 = np.zeros(4, dtype=np.int32)
    k2 = np.array([0, C.HAP_DEL, C.HAP_DEL, C.BASE_A], dtype=np.int32)
    assert _haplotype_pairs(k1, k2, ref4, ids.items) == {("AA", "ACGT"), ("ACGA", "AT")}
    k2 = np.array([0, C.HAP_DEL | ids("TT", None) << 4, 0, 0], dtype=np.int32)  # (C deleted, and TT inserted after it)
    assert _haplotype_pairs(k1, k2, ref4, ids.items) == {("ACGT", "ATTGT"), ("ACTTGT", "AGT")}
    k2 = np.array([C.BASE_C, C.BASE_A, 0, 0], dtype=np.int32)  # (two substitutions next to each other: one piece)
    assert _haplotype_pairs(k1, k2, ref4, ids.items) == {("ACGT", "CAGT")}


# a child and both parents
TRIO = {
    "child": [
        snv(500, "0/1"),
        snv(600, "0/1"),
        snv(700, "1/1"),
        snv(800, "0/1"),
        snv(900, "1/1"),
        snv(1100, "0/1"),
        snv(1200, "0/1"),  # (two bases next to each other that neither parent has: one site)
        snv(1201, "0/1"),
        (1009, "A", "AA", "0/1"),  # (the mother's insertion, written after the run of As)
        snv(4200, "0/1"),  # (outside the child's regions called with confidence)
    ],
    "father": [snv(500, "0/1"), snv(700, "0/1"), snv(800, "0/1"), snv(900, "1/1")],
    "mother": [snv(600, "0/1"), snv(700, "0/1"), snv(800, "0/1"), (999, "C", "CA", "0/1")],
}


@pytest.fixture
def trio(tmp_path):
    return [make(tmp_path, who, recs, bed=B_BED if who == "child" else None) for who, recs in TRIO.items()]


def test_trio_counts(trio):
    with GpxilFile(trio[0]) as c, GpxilFile(trio[1]) as f, GpxilFile(trio[2]) as m:
        r = compare_trio(c, f, m)
        everything = compare_trio(c, f, m, all_calls=True)["total"]
    t = r["total"]
    assert (t["all_called"], t["child_variant_bases"]) == (L - 500, 9)
    assert (t["from_father"], t["from_mother"], t["from_both"], t["origin_unknown"]) == (1, 1, 1, 1)
    me = t["mendelian_errors"]
    assert (me["sites"], me["bases"], me["new_mutation_sites"], me["new_mutation_bases"]) == (3, 4, 2, 3)
    assert [(x["position"], x["end"], x["new"]) for x in me["examples"]] == [(901, 901, False), (1101, 1101, True), (1201, 1202, True)]
    assert me["examples"][0]["child"].count("/") == 1 and me["examples"][0]["mother"] == f"{REF[900]}/{REF[900]}"
    assert t["written_differently"] == {"stretches": 1, "bases": 1}
    assert t["haplotypes"] == {"father": None, "mother": None}
    assert everything["mendelian_errors"]["new_mutation_sites"] == 3  # (with the child's call outside its regions)
    text = format_compare(r)
    assert "3 sites (4 bases), 2 of them candidate new mutations" in text and "not phased" in text


def test_trio_picture(trio):
    with GpxilFile(trio[0]) as c, GpxilFile(trio[1]) as f, GpxilFile(trio[2]) as m:
        pic = picture([c, f, m], "chr1", 400, 1200)
    idx = pic.index.reshape(-1)
    want = {500: 1, 600: 3, 700: 5, 800: 6, 1009: 7, 900: 8, 1100: 9}
    assert {p: int(idx[p - 400]) - 0xF0 for p in want} == want
    labels = [x for x, _ in pic.legend]
    assert TRIO_CLASSES[2][0] not in labels and TRIO_CLASSES[1][0] in labels  # (no phased parent: no haplotype 2)


def test_trio_crossover(tmp_path):
    """A phased father: the child takes his haplotype 1, then haplotype 2; one site against it is an error, not two crossovers."""
    sites = [5000 + 20 * i for i in range(30)]
    father = [snv(p, "0|1", 5001) for p in sites]
    child = [snv(p, "0/1") for i, p in enumerate(sites) if i == 5 or i >= 15]
    c, f, m = (make(tmp_path, w, r) for w, r in (("kid", child), ("dad", father), ("mum", [snv(10, "0/1")])))
    with GpxilFile(c) as x, GpxilFile(f) as y, GpxilFile(m) as z:
        r = compare_trio(x, y, z)
        pic = picture([x, y, z], "chr1", 5000, 5600)
    t = r["total"]
    assert t["from_father"] == 16 and t["mendelian_errors"]["sites"] == 0
    dad = t["haplotypes"]["father"]
    assert (dad["sites"], dad["phase_sets"], dad["changes"], dad["crossovers"]) == (30, 1, 3, 1)
    assert dad["at"] == [["chr1", sites[14] + 1, sites[15] + 1]]
    assert t["haplotypes"]["mother"] is None
    idx = pic.index.reshape(-1)
    assert int(idx[sites[5] - 5000]) - 0xF0 == 2 and int(idx[sites[20] - 5000]) - 0xF0 == 2  # (his haplotype 2)
    assert TRIO_CLASSES[2][0] in [x for x, _ in pic.legend]
    text = format_compare(r)
    assert "1 crossover (between 5,281–5,301)" in text and "3 changes at 30 sites in 1 phase set" in text


def test_cli(pair, trio, tmp_path, capsys):
    assert main(["compare", str(pair[0]), str(pair[1]), "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["mode"] == "pair" and r["total"]["events"]["indels"]["a"]["found"] == 3
    png = tmp_path / "trio.png"
    args = ["compare", str(trio[0]), "--father", str(trio[1]), "--mother", str(trio[2]), "--region", "chr1:401-1200", "-o", str(png)]
    assert main([*args, "--width", "100", "--scale", "2"]) == 0
    out = capsys.readouterr()
    assert "Mendelian errors" in out.out and "wrote" in out.err and png.stat().st_size > 0
    from PIL import Image

    assert Image.open(png).size == (200, 16)
    # the whole of chr1 counted, and only a region of it drawn
    assert main([*args[:-4], "--draw", "chr1:4,001-4,400", "-o", str(png), "--width", "100"]) == 0
    assert "chr1:1-6,000" in capsys.readouterr().out and Image.open(png).size == (100, 4)
    for bad in (["compare", str(pair[0])], ["compare", str(pair[0]), str(pair[1]), "--father", str(trio[1])]):
        with pytest.raises(SystemExit):
            main(bad)
