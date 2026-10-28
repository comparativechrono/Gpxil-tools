"""0.4.0: gpxil subset, working copies of some contigs and planes."""

from __future__ import annotations

import numpy as np
import pytest
from simulate import fasta_text, simulate
from test_web_and_browser import serve

from gpxiltools import codes as C
from gpxiltools.cli import main
from gpxiltools.container import GpxilFile, GpxilFormatError, GpxilWriter, codec_available
from gpxiltools.encode import encode
from gpxiltools.fasta import sequence_digests
from gpxiltools.subset import subset
from gpxiltools.validate import validate
from gpxiltools.vcf import SidecarEntry


@pytest.fixture
def genome(tmp_path):
    """An encoded sample on chr1 and chr3, with a reference-only chr2, as a 'whole genome'."""
    sims = [simulate(n=30000, n_sites=300, seed=s) for s in (3, 4)]
    fa = tmp_path / "ref.fa"
    fa.write_text(fasta_text({"chr1": sims[0].reference, "chr2": "ACGTTGCA" * 2000, "chr3": sims[1].reference}))
    vcf = tmp_path / "s.vcf"
    lines = sims[0].vcf_lines("chr1")
    lines[1:1] = [f"##contig=<ID=chr3,length={len(sims[1].reference)}>"]
    lines += sims[1].vcf_lines("chr3")[5:]
    vcf.write_text("\n".join(lines) + "\n")
    out = tmp_path / "genome.gpxil.tif"
    encode(fa, out, vcf=vcf, width=700, log=lambda m: None)
    return fa, out


def _same(a, b, planes=None):
    for p in planes or a.planes:
        assert np.array_equal(a.read(p), b.read(p)), p


def test_a_contig_copied_without_decoding(genome, tmp_path):
    _, src = genome
    out = tmp_path / "chr3.gpxil.tif"
    info = subset(src, out, ["3"], log=lambda m: None)  # by alias
    assert info["names"] == ["chr3"] and info["pages_encoded"] == 0 and info["pages_copied_as_stored"] == 4
    assert validate(out).ok
    with GpxilFile(src) as g, GpxilFile(out) as s:
        assert s.contigs == ["chr3"] and s.sample_id == g.sample_id and s.metadata["reference"] == g.metadata["reference"]
        _same(g["chr3"], s["chr3"])
        assert s["chr3"].sidecar() == g["chr3"].sidecar() and s["chr3"].sidecar()
        old = {k: v for k, v in g["chr3"].entry.items() if k != "pages"}
        assert {k: v for k, v in s["chr3"].entry.items() if k != "pages"} == old  # digests, encoding counts, aliases ...
        rec = s.metadata["subset"]
        assert rec["of"] == "genome.gpxil.tif" and rec["contigs"] == ["chr3"] and rec["its_summary"]["contigs"] == 3
        assert "summary" not in s.metadata


def test_contigs_in_file_order_and_planes_left_out(genome, tmp_path):
    _, src = genome
    out = tmp_path / "w.gpxil.tif"
    info = subset(src, out, ["chr3,chr1"], planes=["ref", "gt", "flags"], log=lambda m: None)
    assert info["names"] == ["chr1", "chr3"] and info["planes"] == ["ref", "gt", "flags"]
    assert validate(out).ok
    with GpxilFile(out) as s:
        assert s["chr1"].planes == ["ref", "gt", "flags"]
        assert set(s.metadata["planes"]) == {"ref", "gt", "flags"}
    # a reference-only contig gets the planes it has; --no-ref leaves the reference out
    out2 = tmp_path / "v.gpxil.tif"
    info = subset(src, out2, ["chr1", "chr2"], planes=["gt", "flags"], log=lambda m: None)
    assert info["names"] == ["chr1"]  # chr2 has neither plane
    out3 = tmp_path / "noref.gpxil.tif"
    subset(src, out3, include_ref=False, log=lambda m: None)
    with GpxilFile(out3) as s, GpxilFile(src) as g:
        assert s.contigs == ["chr1", "chr3"] and "ref" not in s["chr1"].pages  # chr2 had only its reference
        assert s["chr1"].entry["md5"] == g["chr1"].entry["md5"]
    assert validate(out3).ok


def test_the_reference_plane_put_back(genome, tmp_path):
    fa, src = genome
    noref = tmp_path / "noref.gpxil.tif"
    subset(src, noref, ["chr1"], include_ref=False, log=lambda m: None)
    for reference in (fa, src):  # a FASTA, or a gpxil file with the reference plane
        back = tmp_path / f"back-{reference.suffix}.gpxil.tif"
        info = subset(noref, back, planes=["ref", "gt", "flags", "gq"], reference=reference, log=lambda m: None)
        assert info["pages_encoded"] == 1  # the reference plane, read from the reference
        assert validate(back).ok
        with GpxilFile(back) as b, GpxilFile(src) as g:
            _same(g["chr1"], b["chr1"], ["ref", "gt", "flags", "gq"])
    with pytest.raises(KeyError, match="without its reference"):
        subset(noref, tmp_path / "x.gpxil.tif", planes=["ref", "gt", "flags"], log=lambda m: None)


@pytest.mark.parametrize("compression,level", [("lzma", None), ("none", None), ("deflate", 1), ("zstd", None)])
def test_strips_encoded_again(genome, tmp_path, compression, level):
    if not codec_available(compression):
        pytest.skip(f"{compression} needs imagecodecs")
    _, src = genome
    out = tmp_path / f"{compression}.gpxil.tif"
    info = subset(src, out, compression=compression, level=level, log=lambda m: None)
    assert info["pages_copied_as_stored"] == 0 and info["pages_encoded"] == 9
    assert validate(out).ok
    with GpxilFile(src) as g, GpxilFile(out) as s:
        for name in g.contigs:
            _same(g[name], s[name])
        expected = {"lzma": 34925, "none": 1, "deflate": 8, "zstd": 50000}[compression]
        assert int(s._tf.pages[0].compression) == expected


def _phased(tmp_path, ps_values):
    n = 5000
    ref = np.full(n, C.BASE_A, dtype=np.uint8)
    gt = np.zeros(n, dtype=np.uint8)
    flags = np.zeros(n, dtype=np.uint8)
    ps = np.zeros(n, dtype=np.uint32)
    for i, v in enumerate(ps_values):
        k = 100 + 1000 * i
        gt[k] = 0x04
        flags[k] = C.FLAG_PHASED | C.FLAG_VARIANT_SITE
        ps[k] = v
    flags[2000] |= C.FLAG_INS1
    p = tmp_path / "phased.gpxil.tif"
    with GpxilWriter(p, width=128, sample={"id": "S"}, reference={"name": "toy"}) as w:
        w.add_contig("c", {"ref": ref, "gt": gt, "flags": flags, "ps": ps, "x-depth": np.arange(n, dtype=np.uint16)},
                     sidecar=[SidecarEntry(2000, 1, "INS", "GG")], **sequence_digests(ref))  # fmt: skip
        w.plane_info = {"x-depth": {"description": "made-up depth", "units": "reads"}}
    return p


def test_phase_sets_and_extension_planes(tmp_path):
    one = _phased(tmp_path, [7, 7, 7])
    out = tmp_path / "o.gpxil.tif"
    subset(one, out, planes=["ref", "gt", "flags", "x-depth"], log=lambda m: None)  # one phase set: ps may go
    with GpxilFile(out) as s:
        assert s["c"].planes == ["ref", "gt", "flags", "x-depth"]
        assert s.metadata["planes"]["x-depth"] == {"description": "made-up depth", "units": "reads", "dtype": "uint16"}
        assert np.array_equal(s["c"].read("x-depth"), np.arange(5000, dtype=np.uint16))
    assert validate(out).ok
    several = _phased(tmp_path, [7, 9, 7])
    with pytest.raises(GpxilFormatError, match="more than one phase set"):
        subset(several, tmp_path / "p.gpxil.tif", planes=["ref", "gt", "flags"], log=lambda m: None)
    subset(several, tmp_path / "q.gpxil.tif", planes=["ref"], log=lambda m: None)  # no genotypes: no phasing to lose
    with GpxilFile(tmp_path / "q.gpxil.tif") as s:
        assert s.sample_id is None and s.metadata["sample"] is None


def test_what_subset_refuses(genome, tmp_path):
    _, src = genome
    out = tmp_path / "o.gpxil.tif"
    with pytest.raises(ValueError, match="is a region"):
        subset(src, out, ["chr1:1-1000"], log=lambda m: None)
    with pytest.raises(KeyError, match="no contig 'chr9'"):
        subset(src, out, ["chr9"], log=lambda m: None)
    with pytest.raises(ValueError, match="the input file"):
        subset(src, src, log=lambda m: None)
    with pytest.raises(GpxilFormatError, match="needs its flags"):
        subset(src, out, planes=["ref", "gt"], log=lambda m: None)
    with pytest.raises(ValueError, match="unknown plane 'depth'"):
        subset(src, out, planes=["depth"], log=lambda m: None)
    with pytest.raises(ValueError, match="no contig asked for has a dp plane"):
        subset(src, out, ["chr2"], planes=["ref", "dp"], log=lambda m: None)
    with pytest.raises(ValueError, match="--no-ref"):
        subset(src, out, planes=["ref"], include_ref=False, log=lambda m: None)
    assert not out.exists()


def test_a_working_copy_from_the_web(genome, tmp_path):
    _, src = genome
    out = tmp_path / "web.gpxil.tif"
    with serve(src.parent) as (url, log):
        info = subset(f"{url}/{src.name}", out, ["chr3"], log=lambda m: None)
        assert info["pages_copied_as_stored"] == 4
        assert len(log) < 40, "adjacent strips are read together"
    with GpxilFile(src) as g, GpxilFile(out) as s:
        _same(g["chr3"], s["chr3"])
        assert s.metadata["subset"]["of"] == src.name


def test_subset_on_the_command_line(genome, tmp_path, capsys):
    _, src = genome
    out = tmp_path / "cli.gpxil.tif"
    assert main(["subset", str(src), "chr1", "-o", str(out), "--no-ref"]) == 0
    err = capsys.readouterr().err
    assert "wrote" in err and "1 contig (chr1)" in err and "planes gt,flags,gq" in err and "strips copied as stored" in err
    assert main(["subset", str(src), "chr1:5-10", "-o", str(out)]) == 1
    assert "is a region" in capsys.readouterr().err
