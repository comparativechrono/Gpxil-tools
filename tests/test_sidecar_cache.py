"""A contig's sidecar is decoded once, and its insertions are looked up, not scanned, region by region."""

import numpy as np
from simulate import fasta_text, simulate

import gpxiltools.container as K
from gpxiltools.container import GpxilFile, Stretch
from gpxiltools.encode import encode
from gpxiltools.export import haplotype_sequence


def _file(tmp_path):
    sim = simulate(n=12000, n_sites=400, seed=21)
    fa = tmp_path / "ref.fa"
    fa.write_text(fasta_text({"chr1": sim.reference}))
    vcf = tmp_path / "in.vcf"
    vcf.write_text("\n".join(sim.vcf_lines("chr1")) + "\n")
    out = tmp_path / "s.gpxil.tif"
    encode(fa, out, vcf=vcf, log=lambda m: None)
    return out, sim


def _scan(entries, hap, start, end):
    """What _insertions did before: a pass over every entry."""
    out = {}
    for e in entries:
        if e.kind == "INS" and e.hap == hap and start <= e.offset < end:
            out[e.offset] = e.data
    return out


def test_sidecar_decoded_once(tmp_path, monkeypatch):
    out, _ = _file(tmp_path)
    calls = []
    real = K.decode_sidecar
    monkeypatch.setattr(K, "decode_sidecar", lambda blob: calls.append(1) or real(blob))
    with GpxilFile(out) as g:
        c = g["chr1"]
        for a in range(0, c.length - 50, 37):
            for h in (1, 2):
                haplotype_sequence(c, h, a, a + 50)
        c.sidecar()
        c.sidecar()
    assert len(calls) == 1


def test_insertions_match_a_scan(tmp_path):
    out, sim = _file(tmp_path)
    with GpxilFile(out) as g:
        c = g["chr1"]
        entries = c.sidecar()
        assert any(e.kind == "INS" for e in entries)  # (the simulation has insertions to look up)
        rng = np.random.default_rng(5)
        for _ in range(300):
            a = int(rng.integers(0, c.length))
            b = int(rng.integers(a, min(c.length, a + 400) + 1))
            for h in (1, 2):
                assert c.insertions(h, a, b) == _scan(entries, h, a, b)
        assert c.insertions(1) == _scan(entries, 1, 0, c.length)
        st = Stretch([c])
        for h in (1, 2):
            assert st.insertions(h, 100, 9000) == _scan(st.sidecar(), h, 100, 9000)
        # region by region, the haplotypes are the whole one's pieces where no insertion sits at a cut
        for h in (1, 2):
            whole = haplotype_sequence(c, h, keep_case=False).decode()
            assert whole == sim.haplotype(h)


def test_sidecar_list_is_a_copy(tmp_path):
    out, _ = _file(tmp_path)
    with GpxilFile(out) as g:
        c = g["chr1"]
        first = c.sidecar()
        n = len(first)
        first.clear()
        assert len(c.sidecar()) == n
