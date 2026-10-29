"""0.8.0: gpxil annotate, against planes painted base by base in Python."""

from __future__ import annotations

import gzip
import math
import random

import numpy as np
import pytest
from simulate import fasta_text

from gpxiltools.annotate import annotate, segments
from gpxiltools.cli import main
from gpxiltools.container import GpxilFile
from gpxiltools.encode import encode
from gpxiltools.subset import subset
from gpxiltools.validate import validate


def _paint(n: int, feats, none):
    """Base by base: the shortest feature over a base wins; of two as long, the later."""
    out = [none] * n
    best = [None] * n
    for order, (s, e, v) in enumerate(feats):
        for p in range(max(0, s), min(n, e)):
            key = (e - s, -order)
            if best[p] is None or key < best[p]:
                best[p], out[p] = key, v
    return out


def test_segments():
    feats = [(0, 100, 0, 1.0), (10, 20, 1, 2.0), (15, 30, 2, 3.0), (40, 50, 3, 4.0), (40, 50, 4, 5.0), (60, 60, 5, 9.0)]
    s, e, v = segments(feats)
    got = [None] * 100
    for a, b, x in zip(s.tolist(), e.tolist(), v.tolist()):
        for p in range(a, b):
            got[p] = x
    assert got == _paint(100, [(a, b, x) for a, b, _, x in feats], None)
    assert segments([])[0].size == 0
    rng = random.Random(1)
    feats = [(a, a + rng.randint(1, 60), i, float(rng.randint(1, 5))) for i, a in enumerate(rng.randrange(500) for _ in range(200))]
    s, e, v = segments(feats)
    assert (s[1:] >= e[:-1]).all()
    got = [None] * 600
    for a, b, x in zip(s.tolist(), e.tolist(), v.tolist()):
        for p in range(a, b):
            got[p] = x
    assert got == _paint(600, [(a, b, x) for a, b, _, x in feats], None)


@pytest.fixture(scope="module")
def files(tmp_path_factory, sim):
    d = tmp_path_factory.mktemp("annotate")
    fa = d / "ref.fa"
    rng = random.Random(2)
    other = "".join(rng.choice("ACGT") for _ in range(3000))
    fa.write_text(fasta_text({"chr1": sim.reference, "chr2": other}))
    vcf = d / "s.vcf"
    vcf.write_text("\n".join(sim.vcf_lines("chr1")) + "\n")
    path = d / "s.gpxil.tif"
    encode(str(fa), str(path), vcf=str(vcf), width=300)
    return d, path, len(sim.reference), len(other)


def _rmsk(d, n1):
    """A UCSC rmsk table (with its bin column) of repeats, some nested."""
    rng = random.Random(4)
    kinds = [("AluSx", "SINE", "Alu"), ("AluY", "SINE", "Alu"), ("L1PA2", "LINE", "L1"), ("(GA)n", "Simple_repeat", "Simple_repeat")]
    rows = []
    for _ in range(120):
        a = rng.randrange(n1 - 400)
        b = a + rng.randint(5, 350)
        name, cls, fam = rng.choice(kinds)
        div = rng.randint(0, 300)
        rows.append((a, b, name, cls, fam, div, rng.choice("+-")))
    rows.append((50, 60, "AluY", "SINE", "Alu", 7, "+"))
    rows.append((40, 70, "L1PA2", "LINE", "L1", 120, "-"))  # (around the one before: it shows inside)
    path = d / "rmsk.txt.gz"
    with gzip.open(path, "wt") as fh:
        for a, b, name, cls, fam, div, strand in rows:
            fh.write(f"585\t1000\t{div}\t0\t0\tchr1\t{a}\t{b}\t-1\t{strand}\t{name}\t{cls}\t{fam}\t1\t100\t0\t1\n")
        fh.write("585\t1000\t5\t0\t0\tchrUn\t0\t10\t-1\t+\tAluY\tSINE\tAlu\t1\t100\t0\t1\n")
    return path, rows


@pytest.mark.parametrize("level", ["class", "family", "name"])
def test_repeats_as_categories(files, tmp_path, level):
    d, path, n1, n2 = files
    rm, rows = _rmsk(d, n1)
    out = tmp_path / "a.gpxil.tif"
    info = annotate(path, out, rm, plane="x-repeats", level=level, log=lambda m: None)
    assert info["kind"] == "category" and info["unplaced"] == 1 and info["features"] == len(rows) + 1
    label = {
        "class": lambda r: r[3],
        "family": lambda r: r[3] if r[4] == r[3] else f"{r[3]}/{r[4]}",
        "name": lambda r: f"{r[3]}/{r[2]}" if r[4] == r[3] else f"{r[3]}/{r[4]}/{r[2]}",
    }[level]
    names = sorted({label(r) for r in rows})
    number = {x: i for i, x in enumerate(names, 1)}
    want = _paint(n1, [(r[0], r[1], number[label(r)]) for r in rows], 0)
    with GpxilFile(out) as g, GpxilFile(path) as src:
        got = g["chr1"].read("x-repeats")
        assert got.dtype == np.uint8 and got.tolist() == want
        assert (g["chr2"].read("x-repeats") == 0).all()
        meta = g.metadata["planes"]["x-repeats"]
        assert meta["labels"] == {str(i): x for x, i in number.items()} and meta["kind"] == "category" and meta["level"] == level
        for p in ("ref", "gt", "flags"):
            assert (g["chr1"].read(p) == src["chr1"].read(p)).all()
        assert g["chr1"].sidecar() == src["chr1"].sidecar() and g.sample_id == src.sample_id
        assert g.metadata["annotations"][-1]["plane"] == "x-repeats"
    assert validate(str(out)).ok


def test_scores_values_presence_and_windows(files, tmp_path):
    d, path, n1, n2 = files
    rm, rows = _rmsk(d, n1)
    with GpxilFile(path) as g:
        std = list(g["chr1"].pages)  # (ref, gt, flags, gq: the standard planes, in their order)
    out = tmp_path / "s.gpxil.tif"
    annotate(path, out, rm, plane="x-divergence", kind="score", log=lambda m: None)
    want = _paint(n1, [(r[0], r[1], r[5] / 10) for r in rows], math.nan)
    with GpxilFile(out) as g:
        got = g["chr1"].read("x-divergence")
        assert got.dtype == np.float32
        assert np.array_equal(got, np.array(want, dtype=np.float32), equal_nan=True)
    bg = d / "rt.bedgraph"
    bg.write_text("track type=bedGraph\nchr1\t0\t1000\t0.5\nchr1\t1000\t2500\t-1.25\nchr2\t100\t200\t3\n")
    out2 = tmp_path / "v.gpxil.tif"
    info = annotate(out, out2, bg, plane="x-timing", units="log2 ratio", log=lambda m: None)
    assert info["kind"] == "value"
    with GpxilFile(out2) as g:
        v = g["chr1"].read("x-timing")
        assert v[0] == 0.5 and v[999] == 0.5 and v[1000] == -1.25 and math.isnan(v[2500])
        assert math.isnan(g["chr2"].read("x-timing")[99]) and g["chr2"].read("x-timing")[100] == 3
        assert g.metadata["planes"]["x-timing"]["units"] == "log2 ratio"
        assert list(g["chr1"].pages) == [*std, "x-divergence", "x-timing"]
        assert [a["plane"] for a in g.metadata["annotations"]] == ["x-divergence", "x-timing"]
    assert validate(str(out2)).ok
    # (an extension plane named before one the file has: the pages stay in order, spec 8.2)
    plain = d / "plain.bed"
    plain.write_text("chr1\t10\t20\nchr2\t0\t5\n")
    out3 = tmp_path / "p.gpxil.tif"
    annotate(out2, out3, plain, plane="x-a", log=lambda m: None)
    with GpxilFile(out3) as g:
        assert list(g["chr1"].pages) == [*std, "x-a", "x-divergence", "x-timing"]
        pres = g["chr1"].read("x-a")
        assert pres[9] == 0 and pres[10:20].all() and pres[20] == 0 and g.metadata["planes"]["x-a"]["kind"] == "presence"
    assert validate(str(out3)).ok
    with pytest.raises(ValueError, match="--replace"):
        annotate(out3, tmp_path / "again.gpxil.tif", plain, plane="x-a", log=lambda m: None)
    plain.write_text("chr1\t30\t40\n")
    annotate(out3, tmp_path / "again.gpxil.tif", plain, plane="x-a", replace=True, log=lambda m: None)
    with GpxilFile(tmp_path / "again.gpxil.tif") as g:
        assert np.flatnonzero(g["chr1"].read("x-a")).tolist() == list(range(30, 40))
        assert [a["plane"] for a in g.metadata["annotations"]] == ["x-divergence", "x-timing", "x-a"]
    # (windows that overlap hold the same float plane, NaN where there's no feature: the file is valid)
    over = tmp_path / "over.gpxil.tif"
    subset(str(path), str(over), ["chr1:1,001-3,000", "chr1:2,001-4,000"], log=lambda m: None)
    over_scored = tmp_path / "over_s.gpxil.tif"
    annotate(over, over_scored, rm, plane="x-divergence", kind="score", log=lambda m: None)
    assert validate(str(over_scored)).ok
    # (windows: positions are the sequence's)
    win = tmp_path / "w.gpxil.tif"
    subset(str(path), str(win), ["chr1:1,001-3,000", "chr1:5,001-6,000"], log=lambda m: None)
    outw = tmp_path / "wa.gpxil.tif"
    annotate(win, outw, rm, plane="x-repeats", log=lambda m: None)
    with GpxilFile(outw) as g:
        whole = _paint(n1, [(r[0], r[1], 1) for r in rows], 0)
        for w in g.windows:
            got = g[w.name].read("x-repeats")
            assert ((got > 0).tolist()) == [x > 0 for x in whole[w.start : w.end]]
    assert validate(str(outw)).ok


def test_command_line(files, tmp_path, capsys):
    d, path, n1, n2 = files
    rm, rows = _rmsk(d, n1)
    out = tmp_path / "c.gpxil.tif"
    assert main(["annotate", str(path), str(rm), "-o", str(out), "--plane", "x-alu", "--match", "/Alu/", "--level", "name", "-q"]) == 0
    err = capsys.readouterr().err
    assert "x-alu plane (category, uint8, 2 labels)" in err and "on no sequence of the file" in err
    with GpxilFile(out) as g:
        assert g.metadata["planes"]["x-alu"]["labels"] == {"1": "SINE/Alu/AluSx", "2": "SINE/Alu/AluY"}
    assert main(["annotate", str(path), str(rm), "-o", str(out), "--plane", "repeats"]) == 1
    assert "name the plane x-" in capsys.readouterr().err
    assert main(["annotate", str(path), str(rm), "-o", str(path), "--plane", "x-r"]) == 1
    assert "input file" in capsys.readouterr().err
