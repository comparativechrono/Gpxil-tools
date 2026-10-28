"""0.4.0: gpxil stats, checked against what the simulations put in."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
from simulate import fasta_text, simulate

from gpxiltools import codes as C
from gpxiltools.cli import main
from gpxiltools.container import GpxilFile, GpxilWriter
from gpxiltools.encode import encode
from gpxiltools.stats import Tally, format_stats, region_tally, stats
from gpxiltools.subset import subset
from gpxiltools.vcf import SidecarEntry

ADDITIVE = (
    "bases", "called", "not_called", "partly_called", "one_copy", "filtered", "snv_het", "snv_hom", "snv_het_phased",
    "transitions", "transversions", "ins_het", "ins_hom", "ins_bases", "del_het", "del_hom", "del_bases",
    "complex_het", "complex_hom",
)  # fmt: skip


def _trim(r: str, x: str) -> tuple[int, str, str]:
    """The spec's allele trimming (section 6.2): the common suffix (keeping an anchor), then the common prefix."""
    while (len(r) > 1 and len(x) > 1 and r[-1] == x[-1]) if len(r) != len(x) else (r and x and r[-1] == x[-1]):
        r, x = r[:-1], x[:-1]
    k = 0
    while k < min(len(r), len(x)) and r[k] == x[k]:
        k += 1
    return k, r[k:], x[k:]


def expected(sim) -> Counter:
    """What gpxil stats should find, worked out from the simulated alleles (none of the sites overlap)."""
    n = Counter()
    for i, s in enumerate(sim.sites):
        n["gq_bases"] += len(s.ref)
        if i % 17 == 5:
            n["filtered"] += len(s.ref)
        if None in s.hap_alleles:
            n["partly_called"] += len(s.ref)
            continue
        subs: list[dict[int, str]] = [{}, {}]
        dels: list[tuple | None] = [None, None]
        ins: list[str | None] = [None, None]
        for h, allele in enumerate(s.hap_alleles):
            if allele == s.ref:
                continue
            k, r, x = _trim(s.ref, allele)
            for j in range(min(len(r), len(x))):
                if r[j] != x[j]:
                    subs[h][s.pos0 + k + j] = x[j]
            if len(x) > len(r):
                ins[h] = (s.pos0 + k + len(r) - 1, x[len(r) :])
            if len(r) > len(x):
                dels[h] = (s.pos0 + k + len(x), len(r) - len(x))
        for pos in set(subs[0]) | set(subs[1]):
            a, b = subs[0].get(pos), subs[1].get(pos)
            n["snv_hom" if a == b else "snv_het"] += 1
            refbase = sim.reference[pos].upper()
            for alt in {a, b} - {None}:
                n["transitions" if {refbase, alt} in ({"A", "G"}, {"C", "T"}) else "transversions"] += 1
        if dels[0] == dels[1] and dels[0]:
            n["del_hom"] += 1
            n["del_bases"] += dels[0][1]
        else:
            for d in dels:
                if d:
                    n["del_het"] += 1
                    n["del_bases"] += d[1]
        if ins[0] == ins[1] and ins[0]:
            n["ins_hom"] += 1
            n["ins_bases"] += len(ins[0][1])
        else:
            for x in ins:
                if x:
                    n["ins_het"] += 1
                    n["ins_bases"] += len(x[1])
    return n


@pytest.fixture
def encoded(tmp_path):
    sim = simulate(n=40000, n_sites=500, seed=31)
    fa = tmp_path / "ref.fa"
    fa.write_text(fasta_text({"chr1": sim.reference}))
    vcf = tmp_path / "s.vcf"
    vcf.write_text("\n".join(sim.vcf_lines("chr1")) + "\n")
    out = tmp_path / "s.gpxil.tif"
    encode(fa, out, vcf=vcf, width=1000, log=lambda m: None)
    return sim, out


def test_counts_are_what_was_put_in(encoded):
    sim, out = encoded
    want = expected(sim)
    with GpxilFile(out) as g:
        d = stats(g)["total"]
    assert d["bases"] == len(sim.reference)
    ref = d["reference"]
    up = sim.reference.upper()
    assert [ref[b] for b in "ACGT"] == [up.count(b) for b in "ACGT"] and ref["N"] == up.count("N")
    assert ref["soft_masked"] == sum(ch.islower() for ch in sim.reference)
    calls = d["calls"]
    assert calls["partly_called"] == want["partly_called"] and calls["not_called"] == 0
    assert calls["called"] == len(sim.reference) - want["partly_called"]
    assert calls["filtered"] == want["filtered"]
    s = d["substitutions"]
    assert (s["heterozygous"], s["homozygous"]) == (want["snv_het"], want["snv_hom"])
    assert (s["transitions"], s["transversions"]) == (want["transitions"], want["transversions"])
    assert s["ti_tv"] == round(want["transitions"] / want["transversions"], 3)
    for kind in ("ins", "del"):
        e = d["insertions" if kind == "ins" else "deletions"]
        assert (e["heterozygous"], e["homozygous"], e["bases"]) == (want[f"{kind}_het"], want[f"{kind}_hom"], want[f"{kind}_bases"])
    assert want["snv_het"] and want["snv_hom"] and want["del_het"] and want["del_hom"] and want["ins_het"] and want["ins_hom"]
    p = d["phasing"]
    assert p["fraction"] == 1.0 and p["phase_sets"] == 1  # simulated calls are phased, without PS: one phase set
    assert d["gq"]["bases"] == want["gq_bases"] and d["dp"] is None


def test_regions_add_up_to_the_whole(encoded):
    sim, out = encoded
    with GpxilFile(out) as g:
        c = g["chr1"]
        whole = region_tally(c)
        # cut inside deletions, so that runs cross the cuts
        gt = c.read("gt")
        dele = np.flatnonzero((C.hap1(gt) == C.HAP_DEL) | (C.hap2(gt) == C.HAP_DEL))
        runs = np.split(dele, np.flatnonzero(np.diff(dele) > 1) + 1)
        cuts = sorted({int(r[len(r) // 2]) for r in runs if len(r) > 2})[:6]
        assert len(cuts) >= 3
        edges = [0, *cuts, c.length]
        parts = [region_tally(c, a, b, chunk=997) for a, b in zip(edges, edges[1:])]  # small chunks: runs cross them too
        total = parts[0]
        for t in parts[1:]:
            total = total + t
        for name in ADDITIVE:
            assert getattr(total, name) == getattr(whole, name), name
        assert np.array_equal(total.ref_codes, whole.ref_codes) and np.array_equal(total.gq, whole.gq)


def test_one_copy_and_phase_sets(tmp_path):
    n = 3000
    ref = np.full(n, C.BASE_C, dtype=np.uint8)
    ref[:100] = C.BASE_A
    gt = np.full(n, (C.HAP_REF << 4) | C.HAP_ABSENT, dtype=np.uint8)  # one copy throughout
    gt[10] = (C.BASE_G << 4) | C.HAP_ABSENT  # A>G on the only copy: homozygous, a transition
    gt[200:205] = (C.HAP_DEL << 4) | C.HAP_ABSENT  # a deletion on the only copy
    gt[1000:1010] = 0x00  # two copies here
    gt[1001] = C.BASE_T  # heterozygous C>T
    gt[1003] = (C.BASE_A << 4) | C.BASE_G  # A on one, G on the other: heterozygous, two transversions? C>A tv, C>G tv
    flags = np.full(n, C.FLAG_VARIANT_SITE, dtype=np.uint8)
    flags[300] |= C.FLAG_INS1
    flags[[1001, 1003]] |= C.FLAG_PHASED
    ps = np.zeros(n, dtype=np.uint32)
    ps[1001], ps[1003] = 5, 9
    gq = np.full(n, C.QUAL_MISSING, dtype=np.uint8)
    gq[:4] = [10, 20, 30, 40]
    p = tmp_path / "one.gpxil.tif"
    with GpxilWriter(p, width=256, sample={"id": "S"}) as w:
        w.add_contig("chrX", {"ref": ref, "gt": gt, "flags": flags, "gq": gq, "ps": ps}, sidecar=[SidecarEntry(300, 1, "INS", "TTT")])
    with GpxilFile(p) as g:
        d = stats(g)["total"]
    assert d["calls"]["one_copy"] == n - 10
    s = d["substitutions"]
    assert (s["homozygous"], s["heterozygous"], s["transitions"], s["transversions"]) == (1, 2, 2, 2)
    assert d["deletions"] == {"heterozygous": 0, "homozygous": 1, "bases": 5}
    assert d["insertions"] == {"heterozygous": 0, "homozygous": 1, "bases": 3}
    assert d["phasing"] == {"heterozygous_substitutions_phased": 2, "fraction": 1.0, "phase_sets": 2, "n50": 1, "largest": 1}
    assert d["gq"] == {"bases": 4, "mean": 25.0, "min": 10, "q1": 10, "median": 20, "q3": 30, "max": 40}


def test_files_without_a_sample_or_without_the_reference(encoded, tmp_path):
    sim, out = encoded
    refonly = tmp_path / "r.gpxil.tif"
    subset(out, refonly, planes=["ref"], log=lambda m: None)
    with GpxilFile(refonly) as g:
        d = stats(g)["total"]
        assert d["calls"] is None and d["reference"]["A"] > 0
        assert "reference only" in format_stats(stats(g))
    noref = tmp_path / "n.gpxil.tif"
    subset(out, noref, include_ref=False, log=lambda m: None)
    with GpxilFile(noref) as g, GpxilFile(out) as full:
        d, whole = stats(g)["total"], stats(full)["total"]
        assert d["reference"] is None and d["substitutions"]["ti_tv"] is None
        assert d["deletions"] == whole["deletions"] and d["calls"] == whole["calls"]
    with GpxilFile(noref, reference=out) as g:
        assert stats(g)["total"] == whole


def test_stats_on_the_command_line(encoded, capsys):
    sim, out = encoded
    assert main(["stats", str(out), "chr1:1-20000", "1:20001-40000"]) == 0
    text = capsys.readouterr().out
    assert "s.gpxil.tif · sample SIM" in text and "region" in text and "all 2 regions · 40,000 bases" in text
    assert main(["stats", str(out), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["regions"][0]["contig"] == "chr1" and d["regions"][0]["start"] == 1 and d["total"]["regions"] == 1
    assert main(["stats", str(out), "chr1:39990-40100"]) == 1
    assert "outside chr1" in capsys.readouterr().err


def test_tallies_add(encoded):
    sim, out = encoded
    with GpxilFile(out) as g:
        t = region_tally(g["chr1"], 0, 100)
    assert (t + Tally()).bases == 100 and (Tally() + t).has_ref
