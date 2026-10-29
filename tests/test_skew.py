"""0.8.0: gpxil skew, against windows counted from the sequence as text."""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest
from simulate import fasta_text

from gpxiltools.cli import main
from gpxiltools.container import GpxilFile
from gpxiltools.encode import encode
from gpxiltools.features import Feature
from gpxiltools.skew import SkewCounter, iter_skew
from gpxiltools.subset import subset

COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def _rc(s: str) -> str:
    return "".join(COMP[b] for b in reversed(s))


def _sequence(rng: random.Random, n: int) -> str:
    s = []
    while len(s) < n:
        r = rng.random()
        if r < 0.1:
            s += list("AG" * rng.randint(1, 30))  # (purine runs, which skew)
        elif r < 0.12:
            s += [rng.choice("NRYSW")] * rng.randint(1, 4)
        else:
            s += [rng.choice("ACGT") for _ in range(rng.randint(1, 50))]
    s = s[:n]
    for a in range(0, n, 900):
        for i in range(a, min(n, a + rng.randint(0, 200))):
            s[i] = s[i].lower()
    return "".join(s)


def _oracle(seq: str, W: int, ks, *, lo=0, hi=None, unmasked=False, plus=None, minus=None):
    """Windows counted as text: positions [lo, hi) count; plus/minus: sets of positions in features on each strand
    (None: every base on the + strand)."""
    hi = len(seq) if hi is None else hi

    def ok(p, strand_set):
        if not (lo <= p < hi) or seq[p].upper() not in "ACGT":
            return False
        if unmasked and seq[p].islower():
            return False
        return strand_set is None or p in strand_set

    out = {}
    for p in range(lo, hi):
        w = p // W
        d = out.setdefault(w, {"acgt": Counter(), "kmers": {k: Counter() for k in ks}, "other": 0, "masked": 0})
        strands = [(plus, False)] if minus is None else [(plus, False), (minus, True)]
        counted = plus is None or p in plus or (minus is not None and p in minus)
        if counted:
            ch = seq[p]
            if ch.upper() not in "ACGT" and ch != "-":
                d["other"] += 1
            if ch.upper() in "ACGT" and ch.islower():
                d["masked"] += 1
        for sset, flip in strands:
            if not ok(p, sset):
                continue
            b = seq[p].upper()
            d["acgt"][COMP[b] if flip else b] += 1
            for k in ks:
                if all(ok(p + t, sset) for t in range(k)):
                    kmer = seq[p : p + k].upper()
                    d["kmers"][k][_rc(kmer) if flip else kmer] += 1
    rows = []
    for w in sorted(out):
        d = out[w]
        if not (sum(d["acgt"].values()) or d["other"] or d["masked"]):
            continue
        a, c, g, t = (d["acgt"][b] for b in "ACGT")
        row = {"start": w * W, "A": a, "C": c, "G": g, "T": t, "other": d["other"], "soft_masked": d["masked"]}
        row["gc_skew"] = round((g - c) / (g + c), 6) if g + c else None
        for k in ks:
            cnt = d["kmers"][k]
            # (over each pair of a k-mer and its reverse complement, palindromes left out)
            pairs = {min(x, _rc(x)) for x in cnt if x != _rc(x)}
            num = sum(abs(cnt[x] - cnt[_rc(x)]) for x in pairs)
            den = sum(cnt[x] + cnt[_rc(x)] for x in pairs)
            row[f"asymmetry_{k}"] = round(num / den, 6) if den else None
        rows.append(row)
    return rows


def _check(got: list[dict], want: list[dict]):
    assert [r["start"] for r in got] == [r["start"] for r in want]
    for g, w in zip(got, want):
        for key, v in w.items():
            assert g[key] == v, (g["start"], key, g[key], v)


@pytest.fixture(scope="module")
def ref_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("skew")
    rng = random.Random(6)
    seqs = {"chr1": _sequence(rng, 12_345), "chr2": _sequence(rng, 3_001)}
    fa = d / "ref.fa"
    fa.write_text(fasta_text(seqs))
    out = d / "ref.gpxil.tif"
    encode(str(fa), str(out), width=100)
    return out, seqs


@pytest.mark.parametrize("W,chunk,unmasked", [(1000, 1 << 22, False), (1000, 1500, True), (777, 100, False), (5000, 2000, False)])
def test_windows_are_the_oracles(ref_file, W, chunk, unmasked):
    path, seqs = ref_file
    ks = (1, 2, 3, 4)
    with GpxilFile(path) as g:
        rows = list(iter_skew(g, window=W, ks=ks, unmasked=unmasked, chunk=chunk))
    for name, seq in seqs.items():
        _check([r for r in rows if r["sequence"] == name], _oracle(seq, W, ks, unmasked=unmasked))
    # (the cumulative GC skew is the running sum)
    run = 0.0
    for r in [r for r in rows if r["sequence"] == "chr1"]:
        run += r["gc_skew"] or 0.0
        assert r["cumulative_gc_skew"] == pytest.approx(run, abs=1e-5)


def test_regions_windows_and_features(ref_file, tmp_path):
    path, seqs = ref_file
    seq = seqs["chr1"]
    ks = (1, 2, 3)
    with GpxilFile(path) as g:
        rows = list(iter_skew(g, [g.region_of("chr1:2,501-7,000")], window=1000, ks=ks, chunk=1700))
    _check(rows, _oracle(seq, 1000, ks, lo=2500, hi=7000))
    with GpxilFile(path) as g:  # (regions that overlap: their union, each base once)
        rows = list(iter_skew(g, [g.region_of("chr1:4,001-6,000"), g.region_of("chr1:1-5,000")], window=1000, ks=ks))
    _check(rows, _oracle(seq, 1000, ks, lo=0, hi=6000))
    win = tmp_path / "w.gpxil.tif"
    subset(str(path), str(win), ["chr1:1,001-2,500", "chr1:2,301-4,000", "chr1:6,001-6,500"], log=lambda m: None)
    held = "".join(ch if 1000 <= i < 4000 or 6000 <= i < 6500 else "-" for i, ch in enumerate(seq))
    with GpxilFile(win) as g:
        rows = list(iter_skew(g, window=1000, ks=ks, chunk=600))
    _check(rows, _oracle(held, 1000, ks))
    feats = [
        Feature("chr1", 100, 900, "", None, "+"),
        Feature("chr1", 500, 2000, "", None, "-"),
        Feature("chr1", 5000, 5100, "", None, "-"),
    ]
    plus = {p for f in feats if f.strand == "+" for p in range(f.start, f.end)}
    minus = {p for f in feats if f.strand == "-" for p in range(f.start, f.end)}
    with GpxilFile(path) as g:
        rows = list(iter_skew(g, [g.region_of("chr1")], window=1000, ks=ks, features=feats, stranded=True))
        flat = list(iter_skew(g, [g.region_of("chr1")], window=1000, ks=ks, features=feats))
    _check(rows, _oracle(seq, 1000, ks, plus=plus, minus=minus))
    _check(flat, _oracle(seq, 1000, ks, plus=plus | minus))


def test_totals_and_a_symmetric_sequence(tmp_path):
    rng = random.Random(8)
    half = "".join(rng.choice("ACGT") for _ in range(4000))
    seq = half + _rc(half)  # (every k-mer as common as its reverse complement, over the whole)
    fa = tmp_path / "s.fa"
    fa.write_text(fasta_text({"s": seq}))
    path = tmp_path / "s.gpxil.tif"
    encode(str(fa), str(path), width=100)
    counter = SkewCounter(window=8000, ks=(1, 2, 3))
    with GpxilFile(path) as g:
        rows = list(iter_skew(g, counter=counter))
    t = counter.totals()
    assert len(rows) == 1 and t["s"]["gc_skew"] == 0 and t["s"]["at_skew"] == 0 and t["s"]["asymmetry_1"] == 0
    assert t["all"]["bases"] == 8000 and t["s"]["asymmetry_2"] < 0.01
    with pytest.raises(ValueError, match="1 to 6"):
        SkewCounter(ks=(7,))


def test_command_line(ref_file, tmp_path, capsys):
    path, seqs = ref_file
    tsv, png = tmp_path / "w.tsv", tmp_path / "w.png"
    assert (
        main(["skew", str(path), "--window", "1000", "-o", str(tsv), "--picture", str(png), "--show", "gc_skew,at_skew,gc,asymmetry_2"])
        == 0
    )
    err = capsys.readouterr().err
    assert "17 windows of 1,000 bases" in err and "asymmetry k 1" in err  # (13 of chr1, 4 of chr2)
    lines = tsv.read_text().splitlines()
    assert lines[0].split("\t")[:4] == ["sequence", "start", "end", "bases"] and len(lines) == 18
    last = lines[-1].split("\t")
    assert last[:3] == ["chr2", "3000", "3001"]  # (the last window ends where its sequence does)
    from PIL import Image

    im = Image.open(png)
    assert im.width >= 13 and im.height > 2 * 4 * 12
    assert main(["skew", str(path), "chr2", "--json", "--k", "1,2", "--window", "1000"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["windows"] == 4 and set(j["totals"]) == {"chr2", "all"} and "asymmetry_2" in j["totals"]["all"]
    assert main(["skew", str(path), "--show", "nope"]) == 1
    assert "--show nope" in capsys.readouterr().err
