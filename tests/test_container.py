import json

import numpy as np
import pytest
import tifffile

from gpxiltools import codes as C
from gpxiltools.container import GpxilFile, GpxilFormatError, GpxilWriter
from gpxiltools.validate import validate
from gpxiltools.vcf import SidecarEntry


def _codes(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([1, 2, 4, 8, 0x11, 0x12, 0x14, 0x18, 15], dtype=np.uint8), size=n)


@pytest.mark.parametrize("compression", ["deflate", "zstd", "lzw", "none"])
def test_write_read_region_access(tmp_path, compression):
    a, b = _codes(100_003, 1), _codes(5_000, 2)
    gt = np.zeros(100_003, dtype=np.uint8)
    gt[[10, 99_999]] = [0x04, 0x44]
    flags = np.zeros(100_003, dtype=np.uint8)
    flags[10] = C.FLAG_VARIANT_SITE | C.FLAG_INS2
    side = [SidecarEntry(10, 2, "INS", "ACGT")]
    p = tmp_path / "x.gpxil.tif"
    with GpxilWriter(p, width=1000, compression=compression, strip_bytes=4096, reference={"name": "test"}, sample={"id": "S"}) as w:
        w.add_contig("chrA", {"ref": a, "gt": gt, "flags": flags}, sidecar=side, aliases=["A"])
        w.add_contig("chrB", {"ref": b}, width=64)
    with GpxilFile(p) as g:
        assert g.contigs == ["chrA", "chrB"]
        ca = g["A"]
        assert ca.length == 100_003 and ca.width == 1000 and ca.rows == 101
        rng = np.random.default_rng(3)
        for s in rng.integers(0, 100_000, size=50):
            e = int(s) + int(rng.integers(0, 3000))
            e = min(e, ca.length)
            assert np.array_equal(ca.read("ref", int(s), e), a[s:e])
        assert np.array_equal(ca.read("gt"), gt)
        assert ca.sidecar() == side
        img = ca.image("ref")
        assert img.shape == (101, 1000) and not img.reshape(-1)[100_003:].any()
        assert g["chrB"].width == 64 and np.array_equal(g["chrB"].read("ref"), b)
        assert g.metadata["planes"]["gt"] == {"dtype": "uint8"}
    assert validate(p).ok


def test_page_descriptions_and_viewer_compatibility(tmp_path):
    from PIL import Image

    a = _codes(3000)
    p = tmp_path / "v.gpxil.tif"
    with GpxilWriter(p, width=100) as w:
        w.add_contig("c1", {"ref": a})
        w.add_contig("c2", {"ref": a[:500]})
    with tifffile.TiffFile(p) as tf:
        d1 = json.loads(tf.pages[1].description)
        assert d1 == {"gpxil": "0.1", "contig": "c2", "plane": "ref", "length": 500}
        assert tf.pages[0].tags["PageName"].value == "c1 ref"
        assert tf.pages[0].photometric == 3  # palette
        meta = json.loads(tf.pages[0].description)
        assert meta["contig"] == "c1" and meta["plane"] == "ref"
    im = Image.open(p)
    assert im.mode == "P" and im.size == (100, 30)
    assert np.array_equal(np.array(im).reshape(-1), a)


def test_extension_plane_and_errors(tmp_path):
    p = tmp_path / "e.gpxil.tif"
    cov = np.arange(1000, dtype=np.uint16)
    with GpxilWriter(p, width=100) as w:
        w.add_contig("c", {"ref": _codes(1000), "x-coverage": cov})
        with pytest.raises(GpxilFormatError, match="duplicate"):
            w.add_contig("c", {"ref": _codes(10)})
        with pytest.raises(GpxilFormatError, match="neither standard"):
            w.add_contig("d", {"coverage": cov})
        with pytest.raises(GpxilFormatError, match="differ"):
            w.add_contig("e", {"ref": _codes(10), "x-coverage": cov})
    with GpxilFile(p) as g:
        assert np.array_equal(g["c"].read("x-coverage", 5, 15), cov[5:15])
        assert g.metadata["planes"]["x-coverage"]["dtype"] == "uint16"


def test_not_a_gpxil_file(tmp_path):
    p = tmp_path / "plain.tif"
    tifffile.imwrite(p, np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(GpxilFormatError):
        GpxilFile(p)
    assert not validate(p).ok


def test_validator_catches_bad_codes(tmp_path):
    p = tmp_path / "bad.gpxil.tif"
    ref = _codes(500)
    gt = np.zeros(500, dtype=np.uint8)
    gt[3] = 0x70  # 0x7 is a reserved haplotype state
    gt[4] = ref[4] & 0x0F  # substitution equal to the reference base
    flags = np.zeros(500, dtype=np.uint8)
    flags[9] = C.FLAG_INS1  # flag without a sidecar entry
    with GpxilWriter(p, width=50, sample={"id": "S"}) as w:
        w.add_contig("c", {"ref": ref, "gt": gt, "flags": flags}, md5="0" * 32)
    rep = validate(p)
    text = "\n".join(rep.errors)
    assert "invalid hap1 states at 1 positions (first offset 3)" in text
    assert "equals the reference base" in text
    assert "INS1 flags" in text
    assert "md5" in text
