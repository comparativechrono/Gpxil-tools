"""gpxil 0.3: windows (spec 3.1), contigs that hold a stretch of a longer sequence and keep its positions.

A window holds what a contig of its whole parent would hold over its stretch. gpxiltools makes windows two ways,
by converting the calls around them (``gpxil encode --region``) and by cutting them out of a whole contig
(``gpxil subset`` with a region), and the two must agree, however the calls overlap the window's edges.
"""

from __future__ import annotations

import io
import json
import random

import numpy as np
import pytest
from simulate import fasta_text
from test_web_and_browser import serve

from gpxiltools import codes as C
from gpxiltools.cli import main
from gpxiltools.compare import compare_pair, compare_trio
from gpxiltools.container import GpxilFile, GpxilFormatError, GpxilWriter, check_window, parse_region
from gpxiltools.encode import encode
from gpxiltools.export import iter_vcf_records, write_vcf
from gpxiltools.fasta import encode_sequence, sequence_digests
from gpxiltools.stats import stats
from gpxiltools.subset import subset
from gpxiltools.validate import validate

BASES = "ACGT"
QUIET = {"log": lambda m: None}


def _other(b: str, rng: random.Random) -> str:
    return rng.choice([x for x in BASES if x != b.upper()])


def _seq(rng: random.Random, k: int) -> str:
    return "".join(rng.choice(BASES) for _ in range(k))


def dense_vcf(ref: str, chrom: str, rng: random.Random, n_records: int) -> list[str]:
    """Records of every kind the encoder reads, packed so that they overlap one another often: substitutions,
    indels, multi-allelic and spanning-deletion records, symbolic deletions and inversions, breakends, reference
    blocks; phased (with integer and other phase-set identifiers), unphased, haploid and missing genotypes."""
    n = len(ref)
    up = ref.upper()
    lines = [
        "##fileformat=VCFv4.3",
        f"##contig=<ID={chrom},length={n}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">',
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">',
        '##FORMAT=<ID=PS,Number=1,Type=String,Description="Phase set">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS",
    ]
    pos = sorted(rng.randint(2, n - 40) for _ in range(n_records))
    for p in pos:
        kind = rng.choice("snv snv snv mnp del del ins ins cplx multi star sdel inv bnd block".split())
        info, gts = ".", ["0/1", "1/0", "1/1", "0|1", "1|0", "1|1", "./1", "1/.", "1", "0/0", "./."]
        r = up[p - 1]
        if kind == "snv":
            alts = [_other(r, rng)]
        elif kind == "mnp":
            r = up[p - 1 : p + 1]
            alts = ["".join(_other(b, rng) for b in r)]
        elif kind == "del":
            r = up[p - 1 : p + rng.randint(1, 12)]
            alts = [r[0]]
        elif kind == "ins":
            alts = [r + _seq(rng, rng.randint(1, 8))]
        elif kind == "cplx":
            r = up[p - 1 : p + 2]
            alts = [_other(r[0], rng) + _seq(rng, rng.randint(0, 4))]
        elif kind == "multi":
            r = up[p - 1 : p + 3]
            alts = [r[0], r[:2] + _seq(rng, 3)]
            gts = ["1/2", "2/1", "1|2", "0/2", "2/2"]
        elif kind == "star":
            alts = [_other(r, rng), "*"]
            gts = ["1/2", "2/1", "2/0", "0/2"]
        elif kind == "sdel":
            k = rng.randint(2, 30)
            alts = ["<DEL>"]
            info = f"END={p + k}" if rng.random() < 0.7 else f"SVLEN=-{k}"
        elif kind == "inv":
            alts = [rng.choice(["<INV>", "<DUP>", "<INS:ME>"])]
            info = f"END={p + rng.randint(1, 40)}"
        elif kind == "bnd":
            alts = [rng.choice([f"{r}]{chrom}:{rng.randint(1, n)}]", f"{r}.", f".{r}"])]
        else:  # a reference block, called or not
            alts = ["<NON_REF>"]
            info = f"END={min(n, p + rng.randint(0, 25))}"
            gts = ["0/0", "./.", "0/0", "0|0"]
        if rng.random() < 0.03:
            r = _other(r[0], rng) + r[1:]  # (a REF that disagrees with the reference)
        gt = rng.choice(gts)
        fmt, vals = "GT:GQ:DP", [gt, str(rng.randint(0, 99)), str(rng.randint(0, 300))]
        if "|" in gt:
            fmt += ":PS"
            vals.append(rng.choice(["7", "9", "1500", "blockA", "blockB", "4294967295", "0"]))
        filt = rng.choice(["PASS", "PASS", "LowQ", "."])
        lines.append(f"{chrom}\t{p}\t.\t{r}\t{','.join(alts)}\t50\t{filt}\t{info}\t{fmt}\t{':'.join(vals)}")
    return lines


def _reference(rng: random.Random, n: int) -> str:
    seq = [rng.choice(BASES) for _ in range(n)]
    for i in range(n // 7, n // 7 + n // 9):
        seq[i] = seq[i].lower()
    for i in range(n // 2, n // 2 + 20):
        seq[i] = "N"
    return "".join(seq)


@pytest.fixture(scope="module", params=[1, 2, 3, 4])
def dense(request, tmp_path_factory):
    """A contig of 4,000 bases with 500 records that overlap often, its calls encoded whole, with callable and
    haploid regions; and 40 windows, from one base to most of it, whose edges cut calls."""
    seed = request.param
    rng = random.Random(seed)
    tmp = tmp_path_factory.mktemp(f"dense{seed}")
    n = 4000
    ref = _reference(rng, n)
    fa = tmp / "ref.fa"
    fa.write_text(fasta_text({"c1": ref, "c2": _reference(rng, 900)}))
    vcf = tmp / "s.vcf"
    vcf.write_text("\n".join(dense_vcf(ref, "c1", rng, 500)) + "\n")
    bed = tmp / "callable.bed"
    bed.write_text("c1\t0\t1800\nc1\t1850\t3900\n")
    hap = tmp / "haploid.bed"
    hap.write_text("c1\t3000\t3300\n")
    whole = tmp / "whole.gpxil.tif"
    common = dict(vcf=vcf, callable_bed=bed, haploid_bed=hap, width=64, **QUIET)
    encode(fa, whole, **common)
    windows = [(0, 1), (0, 777), (n - 1, n), (3333, n), (1799, 1851), (2999, 3301)]
    while len(windows) < 40:
        a = rng.randrange(n)
        windows.append((a, min(n, a + rng.choice([1, 2, 5, 40, 300, 1500]))))
    windows = sorted(set(windows))
    wins = tmp / "windows.gpxil.tif"
    encode(fa, wins, regions=[("c1", a, b) for a, b in windows], **common)
    return {"fa": fa, "vcf": vcf, "bed": bed, "hap": hap, "whole": whole, "windows": windows, "wins": wins, "n": n, "tmp": tmp}


def _phase_sets(c, s=0, e=None) -> list[str]:
    """A ps plane as the identifiers it stands for (spec 5.5), '' where there is none."""
    ids = c.entry.get("phase_set_ids", {})
    return ["" if v == 0 else ids.get(str(v), str(v)) for v in c.read("ps", s, e).tolist()]


def _sidecar(c, s=0, e=None, shift=0):
    e = c.length if e is None else e
    return [(x.offset - s, x.hap, x.kind, x.data) for x in c.sidecar() if s <= x.offset < e]


def test_windows_converted_on_their_own_hold_what_the_whole_contig_does(dense):
    """Converting just the calls around a window gives what converting the whole contig does, over its bases."""
    with GpxilFile(dense["whole"]) as g, GpxilFile(dense["wins"]) as w:
        assert w.metadata["gpxil"] == "0.3"
        c = g["c1"]
        assert len(w.windows) == len(dense["windows"]) == len(w)
        for (s, e), x in zip(dense["windows"], w):
            assert x.name == f"c1:{s + 1}-{e}" and x.window["of"] == "c1" and (x.start, x.end) == (s, e)
            assert x.window["length"] == c.length and x.window["md5"] == c.entry["md5"] and x.window["aliases"] == c.entry["aliases"]
            assert x.planes == c.planes
            for p in x.planes:
                if p != "ps":
                    assert np.array_equal(x.read(p), c.read(p, s, e)), (x.name, p)
            assert _phase_sets(x) == _phase_sets(c, s, e), x.name  # (the values may be numbered differently)
            assert _sidecar(x) == _sidecar(c, s, e), x.name
            assert x.entry["md5"] == sequence_digests(c.read("ref", s, e))["md5"]
            a2 = int(np.count_nonzero(C.hap2(x.read("gt")) == C.HAP_ABSENT))
            assert x.entry["ploidy"] == (1 if a2 == x.length else 2 if a2 == 0 else "mixed")
            assert x.entry["callable_bases"] == sum(max(0, min(e, b) - max(s, a)) for a, b in ((0, 1800), (1850, 3900)))
            assert x.entry["vcf_contig"] == "c1" and "records" in x.entry["encoding"]
    assert validate(dense["wins"]).ok


def test_windows_cut_out_of_the_whole_contig_are_the_same(dense):
    cut = dense["tmp"] / "cut.gpxil.tif"
    info = subset(dense["whole"], cut, [f"c1:{s + 1}-{e}" for s, e in dense["windows"]], **QUIET)
    assert info["windows"] == len(dense["windows"]) and info["gpxil"] == "0.3"
    rep = validate(cut)
    assert rep.ok, rep.errors
    with GpxilFile(cut) as k, GpxilFile(dense["wins"]) as w, GpxilFile(dense["whole"]) as g:
        assert k.contigs == w.contigs
        for a, b in zip(k, w):
            assert a.window == b.window and a.planes == b.planes
            for p in a.planes:
                if p != "ps":
                    assert np.array_equal(a.read(p), b.read(p)), (a.name, p)
            assert _phase_sets(a) == _phase_sets(b) and a.sidecar() == b.sidecar()
            for key in ("md5", "refget", "ploidy", "vcf_contig", "sidecar_entries"):
                assert a.entry.get(key) == b.entry.get(key), key
            assert "encoding" not in a.entry and "callable_bases" not in a.entry  # (a cut can't know them)
            assert a.entry.get("phase_sets", 0) == b.entry.get("phase_sets", 0)
        assert k.metadata["subset"]["gpxil"] == g.metadata["gpxil"] == "0.3"


def test_where_regions_are(tmp_path, dense):
    with GpxilFile(dense["wins"]) as w:
        x = w["c1:1800-1851"]
        assert repr(x) == "<Contig c1:1800-1851 length=52 window of c1 at 1,800-1,851 planes=['ref', 'gt', 'flags', 'gq', 'dp', 'ps']>"
        assert (x.parent, x.start, x.end, x.parent_length, x.region()) == ("c1", 1799, 1851, 4000, "c1:1,800-1,851")
        assert w.locate("c1:1800-1851") == (x, 0, 52)
        assert w.locate("c1", 1800, 1810) == (w.windows_of("c1")[_first_holding(w, 1800, 1810)], *_offsets(w, 1800, 1810))
        loc = w.locate("c1:1800-1851", 1805, 1806)
        assert loc == (x, 6, 7)
        with pytest.raises(IndexError, match="outside the window c1:1800-1851"):
            w.locate("c1:1800-1851", 0, 10)
        with pytest.raises(KeyError, match="windows of c1"):
            w.locate("c1")  # (several windows: which?)
        with pytest.raises(KeyError, match="no contig 'c9'"):
            w.locate("c9", 0, 10)
    assert parse_region("chr17:7,661,779-7,687,546") == ("chr17", 7661778, 7687546)
    assert parse_region("chr17:7561779-7787546:7,600,001-7,600,010") == ("chr17:7561779-7787546", 7600000, 7600010)
    assert parse_region(" chr17 ") == ("chr17", None, None) and parse_region("HLA-A*01:01") == ("HLA-A*01:01", None, None)
    with pytest.raises(ValueError, match="bad interval"):
        parse_region("chr1:10-9")


def _first_holding(w, s, e):
    return next(i for i, x in enumerate(w.windows_of("c1")) if x.start <= s and e <= x.end)


def _offsets(w, s, e):
    x = w.windows_of("c1")[_first_holding(w, s, e)]
    return s - x.start, e - x.start


def test_window_entries_are_checked(tmp_path):
    ok = {"of": "chr1", "start": 10, "length": 100}
    assert check_window("chr1:11-20", 10, ok) is ok
    for window, msg in [
        ({"of": "chr1"}, "'start' must be a whole number"),
        ({"of": "chr1", "start": -1}, "'start' must be a whole number"),
        ({"of": "chr1", "start": True}, "'start' must be a whole number"),
        ({"start": 3}, "'of' must name its parent"),
        ({"of": "chr 1", "start": 3}, "'of' must name its parent"),
        ({"of": "chr1", "start": 95, "length": 100}, "runs past the end of chr1"),
        ({"of": "chr1", "start": 0, "aliases": "1"}, "'aliases' must be a list"),
        ([1], "must be an object"),
    ]:
        with pytest.raises(GpxilFormatError, match=msg):
            check_window("w", 10, window)
    with pytest.raises(GpxilFormatError, match="its name is its parent's"):
        check_window("chr1", 10, {"of": "chr1", "start": 0})
    with pytest.raises(GpxilFormatError, match="its name is its parent's"):
        check_window("1", 10, {"of": "chr1", "start": 0, "aliases": ["1"]})
    with pytest.raises(GpxilFormatError, match="other names can't be its parent's"):
        check_window("w", 10, {"of": "chr1", "start": 0}, ["chr1"])
    ref = encode_sequence(b"ACGTACGTAC")
    with pytest.raises(GpxilFormatError, match="windows came with gpxil 0.3"):
        with GpxilWriter(tmp_path / "a.gpxil.tif", version="0.2") as w:
            w.add_contig("c:3-12", {"ref": ref}, window={"of": "c", "start": 2})
    with pytest.raises(GpxilFormatError, match="needs the md5 or refget"):
        with GpxilWriter(tmp_path / "b.gpxil.tif", sample={"id": "S"}) as w:
            w.add_contig("c:3-12", {"gt": ref * 0, "flags": ref * 0}, window={"of": "c", "start": 2})
    with GpxilWriter(tmp_path / "c.gpxil.tif", sample={"id": "S"}) as w:  # (the parent's digest identifies it as well)
        w.add_contig("c:3-12", {"gt": ref * 0, "flags": ref * 0}, window={"of": "c", "start": 2, "md5": "0" * 32})
    assert validate(tmp_path / "c.gpxil.tif").ok
    # a file that says 0.2 but holds a window: refused on reading
    bad = tmp_path / "d.gpxil.tif"
    with GpxilWriter(bad) as w:
        w.add_contig("c:3-12", {"ref": ref}, window={"of": "c", "start": 2})
        w.version = "0.2"
    rep = validate(bad)
    assert not rep.ok and any("windows, which came with gpxil 0.3" in e for e in rep.errors)


def test_copies_and_cuts_and_versions(tmp_path, dense):
    # a file of gpxil 0.2: a working copy stays 0.2; a window cut out of it is 0.3, and says its planes are 0.2's
    old = tmp_path / "old.gpxil.tif"
    with GpxilFile(dense["whole"]) as g, GpxilWriter(old, width=64, version="0.2", sample=g.metadata["sample"]) as w:
        w.copy_contig(g["c1"])
    a = subset(old, tmp_path / "a.gpxil.tif", ["c1"], **QUIET)
    b = subset(old, tmp_path / "b.gpxil.tif", ["c1:101-1100"], **QUIET)
    assert a["gpxil"] == "0.2" and b["gpxil"] == "0.3" and b["names"] == ["c1:101-1100"]
    with GpxilFile(tmp_path / "b.gpxil.tif") as x:
        assert x.metadata["gpxil"] == "0.3" and x.metadata["subset"]["gpxil"] == "0.2" and x.metadata["subset"]["of"] == "old.gpxil.tif"
    # a window cut out of a window names the sequence, and where in it
    c = subset(tmp_path / "b.gpxil.tif", tmp_path / "c.gpxil.tif", ["c1:201-300"], **QUIET)
    assert c["names"] == ["c1:201-300"]
    with GpxilFile(tmp_path / "c.gpxil.tif") as x, GpxilFile(dense["whole"]) as g:
        w = x["c1:201-300"]
        assert w.window["of"] == "c1" and w.start == 200 and w.window["length"] == 4000
        assert np.array_equal(w.read("gt"), g["c1"].read("gt", 200, 300))
        assert x.metadata["subset"]["its_subset"]["gpxil"] == "0.2"
    # a window copied as it is stored; the name of its sequence copies all its windows; a region that is a whole
    # window, or a whole contig, copies it
    d = subset(dense["wins"], tmp_path / "d.gpxil.tif", ["c1:1800-1851", "c1:3,000-3,301", "c1:3001-3300"], **QUIET)
    assert d["names"] == ["c1:1800-1851", "c1:3000-3301", "c1:3001-3300"]
    assert d["pages_copied_as_stored"] == 12 and d["pages_encoded"] == 6
    e = subset(dense["wins"], tmp_path / "e.gpxil.tif", ["c1"], **QUIET)
    assert e["contigs"] == len(dense["windows"]) and e["pages_encoded"] == 0
    f = subset(dense["whole"], tmp_path / "f.gpxil.tif", ["c1:1-4000", "c2"], **QUIET)
    assert f["names"] == ["c1", "c2"] and f["windows"] == 0 and f["gpxil"] == "0.3"
    for p in "abcdef":
        assert validate(tmp_path / f"{p}.gpxil.tif").ok, p


def test_windows_without_their_reference(tmp_path, dense):
    """A window without its ref plane is identified by its own digests and its parent's, and reads its bases from
    the parent in a reference: a FASTA, a gpxil file of the whole sequence, or one of a window that holds it."""
    noref = tmp_path / "noref.gpxil.tif"
    subset(dense["whole"], noref, ["c1:1001-2000", "c1:3501-3600"], include_ref=False, **QUIET)
    assert validate(noref).ok
    with GpxilFile(dense["whole"]) as g:
        want = g["c1"].read("ref", 1000, 2000)
    with GpxilFile(noref) as x:
        w = x["c1:1001-2000"]
        assert "ref" not in w.pages and w.entry["md5"] == sequence_digests(want)["md5"] and w.window["md5"]
    big = tmp_path / "big.gpxil.tif"
    subset(dense["whole"], big, ["c1:501-2500"], **QUIET)
    for reference in (dense["fa"], dense["whole"], big):
        with GpxilFile(noref, reference=reference) as x:
            assert np.array_equal(x["c1:1001-2000"].read("ref"), want), reference
            assert np.array_equal(x["c1:1001-2000"].read("ref", 10, 20), want[10:20])
    with GpxilFile(noref, reference=big) as x:
        with pytest.raises(GpxilFormatError, match="not found"):
            x["c1:3501-3600"].read("ref")  # (the window of the reference doesn't hold it)
    # a reference that differs in the window's bases is refused
    text = dense["fa"].read_text().splitlines()
    bad_fa = tmp_path / "bad.fa"
    seq = "".join(line for line in text[1:] if not line.startswith(">"))
    first = text[0]
    c1 = seq[:4000]
    c1 = c1[:1500] + ("A" if c1[1500].upper() != "A" else "C") + c1[1501:]
    bad_fa.write_text(fasta_text({first[1:].split()[0]: c1}))
    with GpxilFile(noref, reference=bad_fa) as x, pytest.raises(GpxilFormatError, match="does not match the md5"):
        x["c1:1001-2000"].read("ref")
    # the command line, with the reference
    out = tmp_path / "with-ref.gpxil.tif"
    assert main(["subset", str(noref), "c1:1001-2000", "--planes", "ref,gt,flags,ps", "--reference", str(dense["fa"]), "-o", str(out)]) == 0
    with GpxilFile(out) as x:
        assert np.array_equal(x["c1:1001-2000"].read("ref"), want)


def test_vcf_of_a_window_is_its_parents(tmp_path, dense):
    with GpxilFile(dense["whole"]) as g, GpxilFile(dense["wins"]) as w:
        x = w["c1:3334-4000"]
        lines = list(iter_vcf_records(x))
        whole = list(iter_vcf_records(g["c1"], start=3333, end=4000))
        assert lines and all(line.startswith("c1\t") for line in lines)
        pos = [int(line.split("\t")[1]) for line in lines]
        assert pos == sorted(pos) and pos[0] >= 3334
        # the same records, but at the window's first bases (a deletion there has no anchor in the window); phase
        # sets numbered as the files number them (spec 3.1, rule 1), so compared as the identifiers they stand for
        inner = [_ps_as_ids(line, g["c1"]) for line in whole if int(line.split("\t")[1]) > 3334 + 40]
        assert inner == [_ps_as_ids(line, x) for line in lines if int(line.split("\t")[1]) > 3334 + 40]
        out = io.StringIO()
        n = write_vcf(w, out, regions=[("c1", 3400, 3900)])
        body = [line for line in out.getvalue().splitlines() if not line.startswith("#")]
        assert len(body) == n and "##contig=<ID=c1,length=4000>" in out.getvalue()
        assert all(3400 - 40 <= int(line.split("\t")[1]) <= 3900 for line in body)
        # the file's overlapping windows of one sequence make one run of records, each once
        out = io.StringIO()
        write_vcf(w, out)
        pos = [int(line.split("\t")[1]) for line in out.getvalue().splitlines() if not line.startswith("#")]
        assert pos == sorted(pos) and len(pos) == len(set(pos))
    assert main(["vcf", str(dense["wins"]), "--region", "c1:3,401-3,900", "-o", str(tmp_path / "w.vcf")]) == 0
    assert (tmp_path / "w.vcf").read_text().count("\nc1\t") == n


def _ps_as_ids(line: str, contig) -> str:
    f = line.split("\t")
    keys, vals = f[8].split(":"), f[9].split(":")
    if "PS" in keys:
        i = keys.index("PS")
        vals[i] = contig.entry.get("phase_set_ids", {}).get(vals[i], vals[i])
    return "\t".join(f[:9] + [":".join(vals)])


def test_fasta_stats_render_and_info_give_the_parents_positions(tmp_path, dense, capsys):
    wins = str(dense["wins"])
    assert main(["fasta", wins, "--region", "c1:1,801-1,830"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(">c1:1801-1830\n")
    with GpxilFile(dense["whole"]) as g:
        assert out.split("\n")[1] == g["c1"].sequence(1800, 1830)
    assert main(["fasta", wins, "--region", "c1:1800-1851"]) == 0  # (a window, by its name)
    assert capsys.readouterr().out.startswith(">c1:1800-1851\n")
    with GpxilFile(dense["wins"]) as w:
        res = stats(w, [("c1", 3000, 3300)])
        r = res["regions"][0]
        assert (r["contig"], r["start"], r["end"]) == ("c1", 3001, 3300)
        assert w[r["window"]].start <= 3000 and 3300 <= w[r["window"]].end
    with GpxilFile(dense["whole"]) as g:
        whole = stats(g, [("c1", 3000, 3300)])["regions"][0]
    assert r["calls"] == whole["calls"] and r["substitutions"] == whole["substitutions"]
    assert main(["stats", wins, "c1:3001-3300"]) == 0
    assert "c1:3,001-3,300" in capsys.readouterr().out
    png = tmp_path / "w.png"
    assert main(["render", wins, "c1:3,001-3,300", "-o", str(png), "--mode", "variants", "--width", "50"]) == 0
    assert "c1:3,001-3,300" in capsys.readouterr().err
    assert main(["info", wins]) == 0
    out = capsys.readouterr().out
    assert "c1:1800-1851: a window of c1 (4,000 bp), its bases 1,800 to 1,851" in out
    pdf = tmp_path / "w.pdf"
    assert main(["book", wins, "-o", str(pdf), "--contigs", "c1:3000-3301", "--rgb", "--dpi", "72"]) == 0
    assert b"3,001 \x96 3,301" in _pdf_text(pdf) or b"3,000 \x96 3,301" in _pdf_text(pdf)


def _pdf_text(path) -> bytes:
    import zlib

    data = path.read_bytes()
    out = b""
    for chunk in data.split(b"stream\n")[1:]:
        try:
            out += zlib.decompress(chunk.split(b"\nendstream")[0])
        except zlib.error:
            pass
    return out


def test_windows_compare_with_whole_contigs_and_each_other(dense):
    with GpxilFile(dense["whole"]) as g, GpxilFile(dense["wins"]) as w:
        res = compare_pair(w, g)  # (every stretch both hold: the windows, against the whole contig)
        assert len(res["regions"]) == len(dense["windows"])
        for r in res["regions"]:
            assert r["contig"] == "c1" and r["windows"][1] is None and r["windows"][0]
            assert r["variant_bases"]["either"] == r["variant_bases"]["same"]
            assert all(r["events"][k]["a"]["missing"] == r["events"][k]["b"]["missing"] == 0 for k in r["events"])
        res = compare_pair(g, w, [("c1", 3333, 3900)])
        held = w[res["regions"][0]["windows"][1]]
        assert res["regions"][0]["windows"][0] is None and held.start <= 3333 and held.end >= 3900 and res["total"]["bases"] == 567
        trio = compare_trio(w, g, g, [("c1", 3333, 3900)])
        assert trio["regions"][0]["contig"] == "c1" and trio["total"]["bases"] == 567
        with pytest.raises(IndexError, match="isn't all in a window"):
            compare_pair(w, g, [("c1", 100, 3900)])


def test_encode_region_on_the_command_line(tmp_path, dense, capsys):
    out = tmp_path / "r.gpxil.tif"
    fa, vcf = str(dense["fa"]), str(dense["vcf"])
    assert main(["encode", fa, "-v", vcf, "--region", "c1:1,001-2,000", "--region", "c2", "-o", str(out)]) == 0
    err = capsys.readouterr().err
    assert "c1:1001-2000: 1,000 bp, a window of c1" in err and "c2: 900 bp, no calls (reference only)" in err
    with GpxilFile(out) as g:
        assert g.contigs == ["c1:1001-2000", "c2"] and g.metadata["summary"]["windows"] == 1
    assert main(["encode", fa, "-v", vcf, "--region", "c1:3,901-4,100", "-o", str(out)]) == 1
    assert "outside c1" in capsys.readouterr().err
    assert main(["encode", fa, "--region", "c7:1-10", "-o", str(out)]) == 1
    assert "the reference has no contig 'c7'" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["encode", fa, "--region", "c1:1-10", "--contigs", "c1", "-o", str(out)])
    # a reference alone: a window of the reference
    assert main(["encode", fa, "--region", "c1:11-20", "-o", str(out)]) == 0
    with GpxilFile(out) as g, GpxilFile(dense["whole"]) as whole:
        w = g["c1:11-20"]
        assert w.planes == ["ref"] and g.sample_id is None and w.window["of"] == "c1" and w.start == 10
        assert w.sequence() == whole["c1"].sequence(10, 20)
    assert main(["encode", fa, "--region", "c1:11-20", "-o", str(out), "--no-ref"]) == 1  # (nothing to store)
    assert "no contigs were written" in capsys.readouterr().err


def test_validate_checks_windows_against_a_parent_in_the_file(tmp_path, dense):
    bad = tmp_path / "bad.gpxil.tif"
    with GpxilFile(dense["whole"]) as g, GpxilWriter(bad, width=64, sample=g.metadata["sample"]) as w:
        w.copy_contig(g["c1"])
        w.cut_contig(g["c1"], 100, 200)
        planes = {p: g["c1"].read(p, 300, 400) for p in ("ref", "gt", "flags")}  # (not the bases it says)
        w.add_contig("c1:201-300", planes, window={"of": "c1", "start": 200, "length": 4000}, **sequence_digests(planes["ref"]))
    rep = validate(bad)
    assert any("c1:201-300: its ref plane differs from that of c1" in e for e in rep.errors), rep.errors
    assert not any("c1:101-200" in e for e in rep.errors)


def test_a_window_from_a_genome_on_the_web(tmp_path, dense):
    out = tmp_path / "web.gpxil.tif"
    with serve(dense["whole"].parent) as (url, log):
        info = subset(f"{url}/{dense['whole'].name}", out, ["c1:2,001-2,100"], **QUIET)
        assert info["names"] == ["c1:2001-2100"] and len(log) < 30
    with GpxilFile(out) as x, GpxilFile(dense["whole"]) as g:
        assert np.array_equal(x["c1:2001-2100"].read("gt"), g["c1"].read("gt", 2000, 2100))
        assert x.metadata["subset"]["of"] == dense["whole"].name


def test_the_metadata_is_the_schemas(dense):
    import importlib.resources

    import jsonschema

    schema = json.loads(importlib.resources.files("gpxiltools").joinpath("schema", "gpxil-0.3.schema.json").read_text())
    with GpxilFile(dense["wins"]) as w:
        jsonschema.validate(w.metadata, schema)
        broken = json.loads(json.dumps(w.metadata))
        broken["contigs"][0]["window"]["start"] = -5
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)


def test_a_fasta_is_read_once_for_all_windows_of_a_chromosome(tmp_path, dense, monkeypatch):
    import gpxiltools.fasta as F

    noref = tmp_path / "noref.gpxil.tif"
    subset(dense["whole"], noref, ["c1:1-1000", "c1:2001-3000", "c1:3001-3500"], include_ref=False, **QUIET)
    reads = []
    real = F.iter_fasta
    monkeypatch.setattr(F, "iter_fasta", lambda *a, **k: (reads.append(a), real(*a, **k))[1])
    with GpxilFile(noref, reference=dense["fa"]) as x, GpxilFile(dense["whole"]) as g:
        for w in x:
            for s in range(0, w.length, 250):
                assert np.array_equal(w.read("ref", s, s + 250), g["c1"].read("ref", w.start + s, w.start + s + 250))
    assert len(reads) == 1


def test_a_window_with_one_phase_set_needs_no_ps_plane(tmp_path, dense):
    with GpxilFile(dense["whole"]) as g:
        ps = g["c1"].read("ps")
        if len({int(v) for v in np.unique(ps) if v}) < 2:
            pytest.skip("this contig has fewer than two phase sets")
        # a stretch without phased calls, and one with several phase sets
        quiet = next(s for s in range(0, len(ps) - 50, 10) if not ps[s : s + 50].any())
        busy = next(s for s in range(0, len(ps) - 400, 10) if len({int(v) for v in np.unique(ps[s : s + 400]) if v}) > 1)
    out = tmp_path / "q.gpxil.tif"
    subset(dense["whole"], out, [f"c1:{quiet + 1}-{quiet + 50}"], planes=["ref", "gt", "flags"], **QUIET)
    with pytest.raises(GpxilFormatError, match="more than one phase set"):
        subset(dense["whole"], out, [f"c1:{busy + 1}-{busy + 400}"], planes=["ref", "gt", "flags"], **QUIET)
