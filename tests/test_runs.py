"""0.8.0: gpxil runs, against runs found in the sequence as text."""

from __future__ import annotations

import io
import json
import random
import re

import pytest
from simulate import fasta_text

from gpxiltools.cli import main
from gpxiltools.container import GpxilFile
from gpxiltools.encode import encode
from gpxiltools.features import base_set, reverse_complement_codes, set_name
from gpxiltools.runs import RunFinder, Summary, write_bed
from gpxiltools.subset import subset


def _sequence(seed: int = 3, n: int = 30000) -> str:
    """Random sequence with runs of purines, pyrimidines and single bases of many lengths, soft-masked stretches,
    IUPAC codes and N."""
    rng = random.Random(seed)
    parts = []
    while sum(map(len, parts)) < n:
        k = rng.random()
        if k < 0.3:
            parts.append("".join(rng.choice("AG") for _ in range(rng.randint(1, 120))))
        elif k < 0.45:
            parts.append("".join(rng.choice("CT") for _ in range(rng.randint(1, 90))))
        elif k < 0.5:
            parts.append(rng.choice("ACGT") * rng.randint(1, 70))
        elif k < 0.52:
            parts.append(rng.choice("RNYSW") * rng.randint(1, 5))
        else:
            parts.append("".join(rng.choice("ACGT") for _ in range(rng.randint(1, 40))))
    seq = "".join(parts)[:n]
    out = list(seq)
    for _ in range(40):  # soft-masked stretches
        a = rng.randrange(n - 50)
        for i in range(a, a + rng.randint(1, 50)):
            out[i] = out[i].lower()
    return "".join(out)


def _expected(seq: str, letters: str, min_length: int, offset: int = 0) -> list[tuple[int, int]]:
    """Runs of bases whose IUPAC set lies in ``letters``, found with a regular expression."""
    mask = base_set(letters)
    from gpxiltools.features import SET_OF

    ok = "".join(ch for ch, m in SET_OF.items() if m & ~mask & 0x0F == 0)
    pat = re.compile(f"[{ok}{ok.lower()}]{{{min_length},}}")
    return [(offset + m.start(), offset + m.end()) for m in pat.finditer(seq)]


@pytest.fixture(scope="module")
def ref_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("runs")
    seqs = {"chr1": _sequence(3), "chr2": _sequence(5, 9000)}
    fa = d / "ref.fa"
    fa.write_text(fasta_text(seqs))
    out = d / "ref.gpxil.tif"
    encode(str(fa), str(out), width=257)
    return out, seqs


@pytest.mark.parametrize("chunk", [7, 100, 1 << 22])
@pytest.mark.parametrize("letters,min_length", [("AG", 51), ("R", 20), ("CT", 5), ("A", 3), ("AT", 10), ("ACGT", 200), ("GA", 1)])
def test_runs_are_the_regular_expressions(ref_file, letters, min_length, chunk):
    path, seqs = ref_file
    with GpxilFile(path) as g:
        got = [(r.sequence, r.start, r.end, r.strand) for r in RunFinder(g, of=letters, min_length=min_length, chunk=chunk).runs()]
    want = [(name, a, b, "+") for name, seq in seqs.items() for a, b in _expected(seq, letters, min_length)]
    assert got == want


@pytest.mark.parametrize("chunk", [13, 1 << 22])
def test_both_strands(ref_file, chunk):
    path, seqs = ref_file
    with GpxilFile(path) as g:
        got = [(r.sequence, r.start, r.end, r.strand) for r in RunFinder(g, of="AG", min_length=40, both_strands=True, chunk=chunk).runs()]
        same = [r.strand for r in RunFinder(g, of="AT", min_length=8, both_strands=True).runs()]
    want = []
    for name, seq in seqs.items():
        want += [(name, a, b, "+") for a, b in _expected(seq, "AG", 40)]
        want += [(name, a, b, "-") for a, b in _expected(seq, "CT", 40)]
    assert got == sorted(want, key=lambda x: (list(seqs).index(x[0]), x[1], x[2], x[3]))
    assert set(same) == {"."}  # (AT is its own complement: one set of runs)


def test_regions_and_windows_keep_the_sequences_positions(ref_file, tmp_path):
    path, seqs = ref_file
    with GpxilFile(path) as g:
        got = [(r.start, r.end) for r in RunFinder(g, [g.region_of("chr1:10,001-20,000")], of="AG", min_length=30).runs()]
    want = _expected(seqs["chr1"][10000:20000], "AG", 30, 10000)
    assert got == want
    win = tmp_path / "win.gpxil.tif"
    subset(str(path), str(win), ["chr1:10,001-20,000"])
    with GpxilFile(win) as g:
        got = [(r.sequence, r.start, r.end) for r in RunFinder(g, of="AG", min_length=30).runs()]
    assert got == [("chr1", a, b) for a, b in want]
    # (a sequence that windows hold, named: all of them, each base once, overlapping and touching windows joined;
    # and regions that overlap, their union)
    wins = tmp_path / "wins.gpxil.tif"
    subset(str(path), str(wins), ["chr1:10,001-14,000", "chr1:12,001-16,000", "chr1:16,001-20,000"])
    with GpxilFile(wins) as g:
        named = [(r.start, r.end) for r in RunFinder(g, [g.region_of("chr1")], of="AG", min_length=30).runs()]
        regions = [g.region_of("chr1:10,001-15,000"), g.region_of("chr1:14,001-20,000")]
        both = [(r.start, r.end) for r in RunFinder(g, regions, of="AG", min_length=30).runs()]
    assert named == both == want


def test_bed_summary_and_command_line(ref_file, tmp_path, capsys):
    path, seqs = ref_file
    with GpxilFile(path) as g:
        f = RunFinder(g, of="AG", min_length=51)
        s = Summary()
        buf = io.StringIO()
        n = write_bed(buf, f.runs(), s, of="GA")
        d = s.as_dict("GA", 51, f.acgt)
    lines = buf.getvalue().splitlines()
    assert n == len(lines) == d["runs"]
    first = lines[0].split("\t")
    assert first[3] == "AG" and first[5] == "+" and int(first[4]) == min(1000, int(first[2]) - int(first[1]))
    acgt = sum(ch.upper() in "ACGT" for seq in seqs.values() for ch in seq)
    assert f.acgt == acgt and d["per_mb"] == round(1e6 * n / acgt, 3)
    bed = tmp_path / "runs.bed"
    assert main([*map(str, ["runs", path, "--of", "R", "-o", bed])]) == 0
    assert bed.read_text().splitlines() == lines
    assert "runs of AG of 51 bases or more" in capsys.readouterr().err
    assert main(["runs", str(path), "chr2", "--of", "AG", "--min", "51", "--json", "-o", str(bed)]) == 0
    d2 = json.loads(capsys.readouterr().out)
    assert d2["runs"] == len(_expected(seqs["chr2"], "AG", 51)) and d2["of"] == "AG"
    assert main(["runs", str(path), "--of", "AXG"]) == 1
    assert "IUPAC" in capsys.readouterr().err


def test_sets_and_strands():
    assert set_name(base_set("ga")) == "AG" and base_set("R") == base_set("AG") and base_set("u") == base_set("T")
    import numpy as np

    codes = np.array([1, 2, 4, 8, 5, 0x11, 15, 0], dtype=np.uint8)  # A C G T R a N -
    assert reverse_complement_codes(codes).tolist() == [0, 15, 0x18, 10, 1, 2, 4, 8]
