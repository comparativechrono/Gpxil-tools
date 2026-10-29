"""0.8.0: gpxil period: a stretch wrapped at its repeat period."""

from __future__ import annotations

import random

import numpy as np
import pytest
from simulate import fasta_text

from gpxiltools import codes as C
from gpxiltools.cli import main
from gpxiltools.container import GpxilFile
from gpxiltools.encode import encode
from gpxiltools.period import best_period, panels, picture, text


def _codes(seq: str) -> np.ndarray:
    return C.ASCII_TO_REF[np.frombuffer(seq.encode(), dtype=np.uint8)]


def test_best_period():
    assert best_period(_codes("AAAG" * 20))[:2] == (4, 1.0)
    assert best_period(_codes("A" * 30))[0] == 1
    assert best_period(_codes("AAAGAG" * 12))[0] == 6
    assert best_period(_codes("GGAA" * 9 + "GGAG" + "GGAA" * 9))[0] == 4
    rng = random.Random(1)
    period, share, shares = best_period(_codes("".join(rng.choice("ACGT") for _ in range(400))))
    assert share < 0.4 and set(shares) == set(range(1, 51))


LEFT = "CCTGTCCGTACCGTTACGCT"  # (no period)
REPEAT = "AAAG" * 15
RIGHT = "TTGCACGGTCAGGCTTACGA"


@pytest.fixture(scope="module")
def tr_file(tmp_path_factory):
    """A tandem repeat with, on haplotype 1, a substitution and a unit inserted; on haplotype 2 a unit deleted."""
    d = tmp_path_factory.mktemp("period")
    seq = "G" * 30 + LEFT + REPEAT + RIGHT + "C" * 30
    fa = d / "ref.fa"
    fa.write_text(fasta_text({"chr1": seq}))
    r0 = 30 + len(LEFT)  # (0-based start of the repeat)
    # VCF positions are 1-based: a substitution A>G at the repeat's 10th base (haplotype 1), a unit inserted after
    # the 20th (haplotype 1), a unit deleted after the 40th (haplotype 2)
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=chr1,length={len(seq)}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
        f"chr1\t{r0 + 10}\t.\t{seq[r0 + 9]}\tG\t.\tPASS\t.\tGT\t1|0",
        f"chr1\t{r0 + 20}\t.\t{seq[r0 + 19]}\t{seq[r0 + 19]}AAAG\t.\tPASS\t.\tGT\t1|0",
        f"chr1\t{r0 + 40}\t.\t{seq[r0 + 39 : r0 + 44]}\t{seq[r0 + 39]}\t.\tPASS\t.\tGT\t0|1",
    ]
    vcf = d / "s.vcf"
    vcf.write_text("\n".join(lines) + "\n")
    out = d / "s.gpxil.tif"
    encode(str(fa), str(out), vcf=str(vcf), width=64)
    return out, seq, r0


def test_panels_aligned_and_as_sequences(tr_file):
    path, seq, r0 = tr_file
    with GpxilFile(path) as g:
        c = g["chr1"]
        s, e = r0, r0 + len(REPEAT)
        ps = panels(c, s, e)
        assert [p.label for p in ps] == ["reference", "haplotype 1", "haplotype 2"]
        assert ps[0].codes == _codes(REPEAT).astype(int).tolist()
        assert ps[1].outline == {9} and ps[1].codes[9] == C.BASE_G and ps[1].ticks == {19: "AAAG"}
        assert ps[2].codes[40:44] == [-1] * 4 and not ps[2].outline
        sq = panels(c, s, e, layout="sequence")
        assert len(sq[1].codes) == len(REPEAT) + 4 and len(sq[2].codes) == len(REPEAT) - 4
        assert best_period(c.read("ref", s, e))[0] == 4
        t = text(ps, 4)
    lines = t.splitlines()
    assert lines[0].split() == ["reference", "haplotype", "1", "haplotype", "2"]
    assert lines[3].split() == ["AAAG", "AgAG", "AAAG"] and lines[5].split()[1] == "AAAG+"
    assert lines[11].split()[2] == "----" and "after base 20 of the stretch, AAAG" in t


def test_picture_and_command_line(tr_file, tmp_path, capsys):
    path, seq, r0 = tr_file
    with GpxilFile(path) as g:
        ps = panels(g["chr1"], r0, r0 + len(REPEAT))
    img = picture(ps, 4, scale=6, title="x")
    rows = len(REPEAT) // 4
    assert img.height == 18 + 14 + 16 + rows * 6 + 8 and img.width >= 3 * 4 * 6
    png = tmp_path / "p.png"
    region = f"chr1:{r0 + 1}-{r0 + len(REPEAT)}"
    assert main(["period", str(path), region, "-o", str(png), "--scale", "5"]) == 0
    assert "period 4" in capsys.readouterr().err and png.stat().st_size > 0
    assert main(["period", str(path), region, "--layout", "sequence"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].endswith("% of bases equal to the base a unit on") and "period 4" in out
    assert main(["period", str(path), f"chr1:1-{len(seq)}", "--period", "7"]) == 0
    assert "period 7" in capsys.readouterr().out
