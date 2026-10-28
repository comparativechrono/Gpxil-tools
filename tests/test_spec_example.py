"""The worked example of the specification (appendix B) as a conformance test."""

import json
import os

import numpy as np
import pytest

from gpxiltools.container import GpxilFile
from gpxiltools.encode import encode
from gpxiltools.export import haplotype_sequence, iter_vcf_records
from gpxiltools.subset import subset
from gpxiltools.validate import validate

DATA = os.path.join(os.path.dirname(__file__), "data")
EXPECTED = json.load(open(os.path.join(DATA, "worked-example.expected.json")))
WINDOW = json.load(open(os.path.join(DATA, "worked-example-window.expected.json")))


def _check(path):
    with GpxilFile(path) as g:
        c = g[EXPECTED["contig"]]
        assert c.length == EXPECTED["length"] and c.width == EXPECTED["width"]
        for plane, values in EXPECTED["planes"].items():
            assert [f"0x{int(v):02x}" for v in c.read(plane)] == values, plane
            img = c.image(plane)
            assert [[f"0x{int(v):02x}" for v in row] for row in img] == EXPECTED["images_including_padding"][plane]
        side = [{"offset": e.offset, "hap": e.hap, "kind": e.kind, "data": e.data} for e in c.sidecar()]
        assert side == EXPECTED["sidecar"]
        assert haplotype_sequence(c, 1).decode() == EXPECTED["haplotype_1"]
        assert haplotype_sequence(c, 2).decode() == EXPECTED["haplotype_2"]
        assert haplotype_sequence(c, 1, keep_case=False).decode() == EXPECTED["haplotype_1_uppercase"]


def test_shipped_example_file_decodes_as_specified():
    path = os.path.join(DATA, "worked-example.gpxil.tif")
    _check(path)
    assert validate(path).ok


def test_encoding_the_example_reproduces_it(tmp_path):
    out = tmp_path / "again.gpxil.tif"
    encode(os.path.join(DATA, "worked-example.fa"), out, vcf=os.path.join(DATA, "worked-example.vcf"), width=4, log=lambda m: None)
    _check(out)


@pytest.mark.parametrize("compression", ["zstd", "lzw", "none"])
def test_example_independent_of_compression(tmp_path, compression):
    from gpxiltools.container import codec_available

    if not codec_available(compression):
        pytest.skip(f"{compression} needs imagecodecs, which this Python does not have")
    out = tmp_path / f"c-{compression}.gpxil.tif"
    encode(
        os.path.join(DATA, "worked-example.fa"),
        out,
        vcf=os.path.join(DATA, "worked-example.vcf"),
        width=4,
        compression=compression,
        log=lambda m: None,
    )
    _check(out)
    with GpxilFile(out) as g:
        assert np.array_equal(g["chrT"].read("gt"), [int(v, 16) for v in EXPECTED["planes"]["gt"]])


def _check_window(path):
    """The window of appendix B, chrT:5-10: what the whole contig holds at offsets 4-9, at the window's offsets."""
    with GpxilFile(path) as g:
        assert g.metadata["gpxil"] == "0.3" and g.contigs == [WINDOW["contig"]]
        c = g[WINDOW["contig"]]
        assert c.length == WINDOW["length"] and c.window == WINDOW["window"]
        for plane, values in WINDOW["planes"].items():
            assert [f"0x{int(v):02x}" for v in c.read(plane)] == values, plane
        side = [{"offset": e.offset, "hap": e.hap, "kind": e.kind, "data": e.data} for e in c.sidecar()]
        assert side == WINDOW["sidecar"]
        assert haplotype_sequence(c, 1).decode() == WINDOW["haplotype_1"]
        assert haplotype_sequence(c, 2).decode() == WINDOW["haplotype_2"]
        assert list(iter_vcf_records(c, nocall_blocks=False)) == WINDOW["vcf_records"]


def test_the_window_of_appendix_b(tmp_path):
    shipped = os.path.join(DATA, "worked-example-window.gpxil.tif")
    _check_window(shipped)
    assert validate(shipped).ok
    # cut out of the whole example, and converted on its own from the VCF: the same window
    cut = tmp_path / "cut.gpxil.tif"
    subset(os.path.join(DATA, "worked-example.gpxil.tif"), cut, ["chrT:5-10"], log=lambda m: None)
    _check_window(cut)
    made = tmp_path / "made.gpxil.tif"
    encode(
        os.path.join(DATA, "worked-example.fa"),
        made,
        vcf=os.path.join(DATA, "worked-example.vcf"),
        regions=[("chrT", 4, 10)],
        width=4,
        log=lambda m: None,
    )
    _check_window(made)
