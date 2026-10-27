"""End-to-end: simulated haplotypes -> VCF -> gpxil -> haplotypes / VCF -> gpxil."""

import io

import numpy as np
import pytest
from simulate import fasta_text, simulate

from gpxiltools import codes as C
from gpxiltools.container import GpxilFile
from gpxiltools.encode import encode
from gpxiltools.export import haplotype_sequence, write_vcf


def _encode(tmp_path, sim, **kw):
    fa = tmp_path / "ref.fa"
    fa.write_text(fasta_text({"chr1": sim.reference}))
    vcf = tmp_path / "in.vcf"
    vcf.write_text("\n".join(sim.vcf_lines("chr1")) + "\n")
    out = tmp_path / "s.gpxil.tif"
    encode(fa, out, vcf=vcf, log=lambda m: None, **kw)
    return fa, vcf, out


@pytest.mark.parametrize("phased", [True, False])
def test_haplotypes_match_truth(tmp_path, phased):
    sim = simulate(n=15000, n_sites=250, seed=3 if phased else 4, phased=phased)
    _, _, out = _encode(tmp_path, sim)
    with GpxilFile(out) as g:
        c = g["chr1"]
        for h in (1, 2):
            got = haplotype_sequence(c, h, keep_case=False).decode()
            assert got == sim.haplotype(h)
        stats = g.metadata["contigs"][0]["encoding"]
        assert stats["conflicts"] == 0 and stats["ref_mismatches"] == 0
        fl = c.read("flags")
        assert bool(fl[sim.sites[0].pos0] & C.FLAG_PHASED) == phased


def test_vcf_export_round_trips_planes(tmp_path):
    sim = simulate(n=15000, n_sites=250, seed=5)
    fa, _, out = _encode(tmp_path, sim)
    buf = io.StringIO()
    with GpxilFile(out) as g:
        write_vcf(g, buf)
        gt1 = g["chr1"].read("gt")
        side1 = sorted((e.offset, e.hap, e.kind, e.data) for e in g["chr1"].sidecar())
        fl1 = g["chr1"].read("flags")
    vcf2 = tmp_path / "out.vcf"
    vcf2.write_text(buf.getvalue())
    out2 = tmp_path / "again.gpxil.tif"
    encode(fa, out2, vcf=vcf2, log=lambda m: None)
    with GpxilFile(out2) as g:
        assert np.array_equal(g["chr1"].read("gt"), gt1)
        assert sorted((e.offset, e.hap, e.kind, e.data) for e in g["chr1"].sidecar()) == side1
        fl2 = g["chr1"].read("flags")
        for bit in (C.FLAG_INS1, C.FLAG_INS2):
            assert np.array_equal(fl1 & bit, fl2 & bit)
        for h in (1, 2):
            assert haplotype_sequence(g["chr1"], h, keep_case=False).decode() == sim.haplotype(h)


def test_callable_bed_and_contig_aliases(tmp_path):
    sim = simulate(n=8000, n_sites=60, seed=9)
    fa = tmp_path / "ref.fa"
    fa.write_text(fasta_text({"1": sim.reference}))  # Ensembl-style name
    vcf = tmp_path / "in.vcf"
    vcf.write_text("\n".join(sim.vcf_lines("chr1")) + "\n")  # UCSC-style name
    bed = tmp_path / "callable.bed"
    bed.write_text("chr1\t0\t4000\n")
    out = tmp_path / "s.gpxil.tif"
    encode(fa, out, vcf=vcf, callable_bed=bed, log=lambda m: None)
    with GpxilFile(out) as g:
        c = g["chr1"]  # alias lookup
        assert c.name == "1"
        gt = c.read("gt")
        sites = {s.pos0 for s in sim.sites}
        outside = [i for i in range(4000, 8000) if not any(abs(i - p) < 15 for p in sites)]
        assert all(gt[i] == C.GT_UNKNOWN for i in outside[:200])
        assert g.metadata["contigs"][0]["callable_bases"] == 4000


def test_matches_bcftools_consensus(tmp_path):
    pysam = pytest.importorskip("pysam", reason="pysam not installed")  # skips this test only
    from pysam import bcftools

    sim = simulate(n=15000, n_sites=200, seed=21, nocalls=False)
    fa, vcf, out = _encode(tmp_path, sim)
    pysam.faidx(str(fa))
    pysam.tabix_compress(str(vcf), str(vcf) + ".gz", force=True)
    pysam.tabix_index(str(vcf) + ".gz", preset="vcf", force=True)
    with GpxilFile(out) as g:
        for h in (1, 2):
            res = bcftools.consensus("-f", str(fa), "-H", str(h), "-s", "SIM", str(vcf) + ".gz", catch_stdout=True)
            theirs = "".join(line for line in res.splitlines() if not line.startswith(">"))
            assert haplotype_sequence(g["chr1"], h).decode() == theirs
