"""0.8.0: gpxil stack, against rows cut from the sequence as text."""

from __future__ import annotations

import gzip
import random

import numpy as np
import pytest
from simulate import fasta_text

from gpxiltools import codes as C
from gpxiltools.cli import main
from gpxiltools.container import GpxilFile
from gpxiltools.encode import encode
from gpxiltools.features import Feature, chosen, read_features
from gpxiltools.stack import GT_COMPLEMENT, consensus, profile, rendering, run_lengths, stack, write_rows
from gpxiltools.subset import subset

_COMP = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn-", "TGCAYRSWMKVHDBNtgcayrswmkvhdbn-")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _text(codes: np.ndarray) -> list[str]:
    return [C.REF_TO_ASCII[r].tobytes().decode() for r in codes]


def _expected_row(seq: str, f: Feature, anchor: str, up: int, down: int, stranded: bool = True) -> str:
    """The row as text: '-' where there is no base."""
    minus = stranded and f.strand == "-"
    off = {"start": 0, "end": f.length, "centre": f.length // 2}[anchor]
    if minus:
        b = f.end - off
        lo, hi = b - down, b + up
    else:
        b = f.start + off
        lo, hi = b - up, b + down
    row = "".join(seq[i] if 0 <= i < len(seq) else "-" for i in range(lo, hi))
    return _revcomp(row) if minus else row


def _sequence(rng: random.Random, n: int) -> str:
    s = list("".join(rng.choice("ACGT") for _ in range(n)))
    for a in range(0, n, 700):  # soft masking and IUPAC codes, which the strands must complement
        for i in range(a + 10, min(n, a + 60)):
            s[i] = s[i].lower()
        s[min(n - 1, a + 70)] = rng.choice("RYSWKMBDHVN")
    return "".join(s)


def _features(rng: random.Random, seqs: dict[str, str], k: int = 60) -> list[Feature]:
    out = []
    for i in range(k):
        name = rng.choice(list(seqs))
        n = len(seqs[name])
        a = rng.randrange(-5, n)  # (some near the ends, so that rows run off them)
        a = max(0, a)
        b = min(n, a + rng.choice([0, 1, 2, 7, 30, 101]))
        out.append(Feature(name, a, b, f"f{i}", float(rng.randrange(100)), rng.choice("+-.")))
    return out


@pytest.fixture(scope="module")
def ref_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("stack")
    rng = random.Random(4)
    seqs = {"chr1": _sequence(rng, 5000), "chr2": _sequence(rng, 1800)}
    fa = d / "ref.fa"
    fa.write_text(fasta_text(seqs))
    out = d / "ref.gpxil.tif"
    encode(str(fa), str(out), width=128)
    return out, seqs


@pytest.mark.parametrize("anchor", ["start", "end", "centre"])
@pytest.mark.parametrize("up,down,chunk", [(40, 25, 1 << 22), (0, 13, 50), (17, 0, 7), (300, 200, 1000)])
def test_rows_are_the_sequence_on_each_strand(ref_file, anchor, up, down, chunk):
    path, seqs = ref_file
    feats = _features(random.Random(up * 7 + down), seqs)
    with GpxilFile(path) as g:
        st = stack(g, feats, anchor=anchor, up=up, down=down, chunk=chunk)
        flat = stack(g, feats, anchor=anchor, up=up, down=down, chunk=chunk, stranded=False)
    assert [r.feature for r in st.rows] == feats and st.ref.shape == (len(feats), up + down)
    assert _text(st.ref) == [_expected_row(seqs[f.contig], f, anchor, up, down) for f in feats]
    assert _text(flat.ref) == [_expected_row(seqs[f.contig], f, anchor, up, down, stranded=False) for f in feats]
    assert st.gt is None and st.unplaced == 0


def test_windows_and_unknown_sequences(ref_file, tmp_path):
    path, seqs = ref_file
    win = tmp_path / "win.gpxil.tif"
    subset(str(path), str(win), ["chr1:1,001-2,000", "chr1:1,900-3,000", "chr1:4,001-4,500"])  # (two join; then a gap)
    feats = [
        Feature("chr1", 990, 1010, "a", None, "+"),  # (runs into the first window)
        Feature("1", 2995, 3010, "b", None, "-"),  # (out of the joined pair; the file finds 1 as chr1)
        Feature("chr1", 3500, 3600, "c", None, "+"),  # (in the gap)
        Feature("chr1", 3990, 4010, "d", None, "-"),
        Feature("chr2", 100, 200, "e", None, "+"),  # (no window of chr2)
        Feature("chrZ", 1, 2, "f", None, "+"),
    ]
    held = "".join(seqs["chr1"][i] if 1000 <= i < 3000 or 4000 <= i < 4500 else "-" for i in range(len(seqs["chr1"])))
    with GpxilFile(win) as g:
        st = stack(g, feats, up=30, down=30)
    assert st.unplaced == 2 and [r.feature.name for r in st.rows] == ["a", "b", "c", "d"]
    assert _text(st.ref) == [_expected_row(held, f, "start", 30, 30) for f in feats[:4]]
    assert _text(st.ref)[2] == "-" * 60


@pytest.fixture(scope="module")
def sample_file(tmp_path_factory, sim):
    d = tmp_path_factory.mktemp("stack_sample")
    fa = d / "ref.fa"
    fa.write_text(fasta_text({"chr1": sim.reference}))
    vcf = d / "s.vcf"
    vcf.write_text("\n".join(sim.vcf_lines("chr1")) + "\n")
    out = d / "s.gpxil.tif"
    encode(str(fa), str(out), vcf=str(vcf), width=256)
    return out, sim


def _complement_state(h: int) -> int:
    return {1: 8, 2: 4, 4: 2, 8: 1}.get(h, h)


@pytest.mark.parametrize("chunk", [1 << 22, 300])
def test_the_sample_on_each_strand(sample_file, chunk):
    path, sim = sample_file
    rng = random.Random(9)
    n = len(sim.reference)
    # features about the sites, on both strands, and one at each end
    feats = [Feature("chr1", s.pos0, s.pos0 + len(s.ref), "", None, rng.choice("+-")) for s in sim.sites[::3]]
    feats += [Feature("chr1", 0, 5, "", None, "-"), Feature("chr1", n - 5, n, "", None, "+")]
    up, down = 23, 31
    with GpxilFile(path) as g:
        c = g["chr1"]
        ref, gt, flags = c.read("ref"), c.read("gt"), c.read("flags")
        st = stack(g, feats, up=up, down=down, sample=True, chunk=chunk)
    ins = C.FLAG_INS1 | C.FLAG_INS2
    assert [r.feature for r in st.rows] == feats
    for r, rref, rgt, rfl in zip(st.rows, st.ref, st.gt, st.flags):
        lo, hi = r.lo, r.hi
        pos = range(lo, hi) if r.strand == "+" else range(hi - 1, lo - 1, -1)
        for t, p in enumerate(pos):
            if not 0 <= p < n:
                assert (rref[t], rgt[t], rfl[t]) == (0, 0, 0)
                continue
            if r.strand == "+":
                assert (rref[t], rgt[t], rfl[t]) == (ref[p], gt[p], flags[p])
            else:
                want_ref = int(C.ASCII_TO_REF[ord(_revcomp(chr(C.REF_TO_ASCII[ref[p]])))])
                want_gt = (_complement_state(gt[p] >> 4) << 4) | _complement_state(gt[p] & 15)
                before = int(flags[p - 1]) if p > 0 else 0  # (the insertion before it on the + strand follows it on the -)
                want_fl = (before & ins) | (int(flags[p]) & ~ins & 0xFF)
                assert (rref[t], rgt[t], rfl[t]) == (want_ref, want_gt, want_fl)
    assert (st.flags & ins).any() and any(r.strand == "-" for r in st.rows)
    assert GT_COMPLEMENT[0x48] == 0x21 and GT_COMPLEMENT[0xD2] == 0xD4 and GT_COMPLEMENT[0xFA] == 0xFA
    for mode in ("genotype", "haplotypes", "variants"):
        r = rendering(st, mode=mode)
        assert r.index.shape == (len(feats) * (2 if mode == "haplotypes" else 1), up + down)


def test_run_lengths():
    def codes(s):
        return C.ASCII_TO_REF[np.frombuffer(s.encode(), dtype=np.uint8)][None, :]

    assert run_lengths(codes("CCAAAAAAA"), 2)[0] == 7
    assert run_lengths(codes("CCAAAAGAAAAACC"), 2)[0] == 10  # (a G within, then two C at the end: trimmed)
    assert run_lengths(codes("AAAACCCAAAA"), 0)[0] == 4  # (the third C ends it, the two before are trimmed)
    assert run_lengths(codes("AAAACCAAAA"), 0)[0] == 10
    assert run_lengths(codes("AAAACCAAAA"), 0, tolerance=1)[0] == 4
    assert run_lengths(codes("aaaRAAN"), 0)[0] == 6 and run_lengths(codes("aaaRAAN"), 0, of="AG")[0] == 6
    assert run_lengths(codes("CAAAA"), 0)[0] == 5 and run_lengths(codes("CCCAAAA"), 0)[0] == 0
    assert run_lengths(codes("GAGGAAGT"), 0, of="R", tolerance=0)[0] == 7


@pytest.fixture(scope="module")
def run_file(tmp_path_factory):
    """Features with known runs of A after their starts (on their strands)."""
    d = tmp_path_factory.mktemp("stack_runs")
    rng = random.Random(2)
    s = list("".join(rng.choice("CG") for _ in range(2000)))
    plant = {100: "A" * 30, 300: "T" * 20, 500: "AAAAAGAAAAACC", 700: "AAAACCCAAAA"}
    for a, x in plant.items():
        s[a : a + len(x)] = x
    seq = "".join(s)
    fa = d / "ref.fa"
    fa.write_text(fasta_text({"chr1": seq}))
    out = d / "ref.gpxil.tif"
    encode(str(fa), str(out), width=100)
    feats = [
        Feature("chr1", 700, 710, "d", 5.0, "+"),
        Feature("chr1", 500, 540, "c", None, "+"),
        Feature("chr1", 300, 320, "b", 9.0, "-"),
        Feature("chr1", 100, 105, "a", 1.0, "+"),
    ]
    bed = d / "f.bed"
    bed.write_text(
        "".join(f"{f.contig}\t{f.start}\t{f.end}\t{f.name}\t{'.' if f.score is None else int(f.score)}\t{f.strand}\n" for f in feats)
    )
    return out, seq, feats, bed


def test_sorting(run_file):
    path, seq, feats, _ = run_file
    with GpxilFile(path) as g:

        def order(**kw):
            st = stack(g, feats, up=10, down=40, **kw)
            return "".join(r.feature.name for r in st.rows), [r.key for r in st.rows]

        assert order(sort="run") == ("abcd", [30, 20, 11, 4])
        assert order(sort="run", reverse=True)[0] == "dcba"
        assert order(sort="run", tolerance=0)[1] == [30, 20, 5, 4]
        assert order(sort="file") == ("dcba", [0, 1, 2, 3])
        assert order(sort="position")[0] == "abcd" and order(sort="position", reverse=True)[0] == "dcba"
        assert order(sort="length")[0] == "cbda"
        assert order(sort="score")[0] == "bdac" and order(sort="score", reverse=True)[0] == "adbc"  # (no score: last)
        assert order(sort="name")[0] == "abcd"
        st = stack(g, feats, up=10, down=40, sort="run", max_rows=2)
        assert len(st.rows) == 2 and st.dropped == 2 and st.ref.shape == (2, 50)
        assert consensus(st)[10:30] == "A" * 20


def test_profile_and_rows(run_file, tmp_path):
    path, seq, feats, _ = run_file
    with GpxilFile(path) as g:
        st = stack(g, feats, up=10, down=40, sort="run")
    p = profile(st)
    assert [d["offset"] for d in p] == list(range(-10, 40))
    rows = [_expected_row(seq, f, "start", 10, 40) for f in [feats[3], feats[2], feats[1], feats[0]]]
    for j, d in enumerate(p):
        col = [r[j] for r in rows]
        assert d["rows"] == sum(ch != "-" for ch in col)
        assert [d[b] for b in "ACGT"] == [col.count(b) for b in "ACGT"]
    assert p[10]["commonest"] == "A" and p[10]["commonest_share"] == 1.0 and "heterozygous" not in p[0]
    out = tmp_path / "rows.bed"
    with open(out, "w") as fh:
        write_rows(fh, st)
    lines = out.read_text().splitlines()
    assert lines[0] == "#sequence\tstart\tend\tname\tscore\tstrand\trow\trun_of_A"
    assert lines[1].split("\t") == ["chr1", "100", "105", "a", "1", "+", "1", "30"]
    assert lines[3].split("\t")[3:] == ["c", ".", "+", "3", "11"]


def test_reading_bed_and_repeatmasker(tmp_path):
    bed = tmp_path / "a.bed"
    bed.write_text("track name=x\n#c\nchr1\t10\t20\tn1\t5\t-\nchr2\t0\t5\n")
    f = read_features(bed)
    assert [(x.contig, x.start, x.end, x.name, x.score, x.strand) for x in f] == [
        ("chr1", 10, 20, "n1", 5.0, "-"),
        ("chr2", 0, 5, "", None, "."),
    ]
    ucsc = "585\t1504\t13\t4\t13\tchr1\t10000\t10468\t-248945954\t+\t(TAACCC)n\tSimple_repeat\tSimple_repeat\t1\t471\t0\t1\n"
    ucsc += "1892\t1910\t106\t13\t7\tchr1\t15797\t16080\t-248940342\t-\tAluSx\tSINE\tAlu\t-12\t300\t2\t5\n"
    rm = tmp_path / "rmsk.txt.gz"
    with gzip.open(rm, "wt") as fh:
        fh.write(ucsc)
    no_bin = tmp_path / "rmsk2.txt"
    no_bin.write_text("".join(line.split("\t", 1)[1] for line in ucsc.splitlines(True)))
    out = tmp_path / "x.fa.out"
    out.write_text(
        "   SW  perc perc perc  query      position in query           matching       repeat              position in  repeat\n"
        "score  div. del. ins.  sequence    begin     end    (left)    repeat         class/family         begin  end (left)   ID\n"
        "\n"
        "  463   1.3  0.6  1.7  chr1        10001    10468 (248945954) +  (CCCTAA)n      Simple_repeat            1  463    (0)      1\n"
        " 1892  10.6  1.3  0.7  chr1        15798    16080 (248940342) C  AluSx          SINE/Alu             (12)  300      2      5 *\n"
    )
    want = [("chr1", 10000, 10468, "Simple_repeat/(TAACCC)n", 1.3, "+"), ("chr1", 15797, 16080, "SINE/Alu/AluSx", 10.6, "-")]
    for p in (rm, no_bin):
        assert [(x.contig, x.start, x.end, x.name, x.score, x.strand) for x in read_features(p)] == want
    got = read_features(out)
    assert [(x.start, x.end, x.name, x.score, x.strand) for x in got] == [
        (10000, 10468, "Simple_repeat/(CCCTAA)n", 1.3, "+"),
        (15797, 16080, "SINE/Alu/AluSx", 10.6, "-"),
    ]
    assert [x.name for x in chosen(got, match="/Alu/")] == ["SINE/Alu/AluSx"]
    assert [x.name for x in read_features(rm, match="/Alu/")] == ["SINE/Alu/AluSx"]  # (chosen as the table is read)
    assert [x.name for x in read_features(bed, match="^n", min_length=10)] == ["n1"] and read_features(bed, min_length=11) == []
    assert len(chosen(got, min_length=300)) == 1
    with pytest.raises(ValueError, match="regular expression"):
        chosen(got, match="(")
    with pytest.raises(ValueError, match="RepeatMasker"):
        read_features(bed, "rmsk")


def test_command_line(run_file, sample_file, tmp_path, capsys):
    path, seq, feats, bed = run_file
    png = tmp_path / "s.png"
    assert main(["stack", str(path), str(bed), "-o", str(png), "--sort", "run", "--up", "10", "--down", "40"]) == 0
    err = capsys.readouterr().err
    assert "4 rows of 4 features" in err and "by the run of A from the anchor" in err and "50 x 4 px" in err
    from PIL import Image

    im = Image.open(png)
    assert im.size == (50, 4) and im.mode == "P"
    lab = tmp_path / "l.png"
    prof = tmp_path / "p.tsv"
    rows = tmp_path / "r.bed"
    args = ["stack", str(path), str(bed), "-o", str(lab), "--labelled", "--scale", "3", "--profile", str(prof), "--rows", str(rows)]
    assert main(args) == 0
    assert Image.open(lab).width > 150 and prof.read_text().startswith("offset\trows\tA\tC\tG\tT")
    assert len(rows.read_text().splitlines()) == 5
    tif = tmp_path / "b.tif"
    assert main(["stack", str(path), str(bed), "-o", str(tif), "--bin", "2", "--cmyk"]) == 0
    import tifffile

    assert tifffile.imread(tif).shape == (2, 600, 4)
    capsys.readouterr()
    assert main(["stack", str(path), str(bed), "--down", "5", "--up", "0"]) == 0  # (no picture: the profile)
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("offset") and len(out) == 6
    assert main(["stack", str(path), str(bed), "-o", str(png), "--mode", "variants"]) != 0
    assert main(["stack", str(path), str(bed), "--match", "^zz"]) != 0
    assert "nothing to stack" in capsys.readouterr().err
    spath, sim = sample_file
    sbed = tmp_path / "s.bed"
    sbed.write_text("".join(f"chr1\t{s.pos0}\t{s.pos0 + 1}\ts\t0\t{'+-'[k % 2]}\n" for k, s in enumerate(sim.sites[:50])))
    assert main(["stack", str(spath), str(sbed), "-o", str(png), "--mode", "variants", "--up", "20", "--down", "20"]) == 0
    sprof = tmp_path / "sp.tsv"
    assert main(["stack", str(spath), str(sbed), "--profile", str(sprof), "--up", "20", "--down", "20"]) == 0
    head = sprof.read_text().splitlines()[0].split("\t")
    assert head[-6:] == ["heterozygous", "homozygous_alternate", "deletion", "insertion_after_base", "no_call", "complex"]
