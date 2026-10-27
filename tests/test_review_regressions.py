"""Regression tests for problems found in the code review of gpxiltools 0.1.0.

Each test names the review case it covers. They use tiny hand-written
inputs so the expected planes can be checked base by base.
"""

import io
import json

import numpy as np
import pytest
import tifffile

from gpxiltools import codes as C
from gpxiltools.cli import main
from gpxiltools.container import GpxilFile, GpxilFormatError, GpxilWriter
from gpxiltools.encode import encode
from gpxiltools.export import genotype_iupac_sequence, haplotype_sequence, iter_haplotype, write_fasta_stream, write_vcf
from gpxiltools.fasta import encode_sequence, iter_fasta
from gpxiltools.validate import validate
from gpxiltools.vcf import SidecarEntry

HEADER = (
    "##fileformat=VCFv4.3\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
)
FA = ">c1\nACGTACGTACGTACGTACGT\n"


def run(tmp_path, fasta, rows=None, *, width=20, name="t", **kw):
    """Encode ``fasta`` (text) with VCF ``rows`` (tuples or raw lines); returns the output path."""
    fa = tmp_path / f"{name}.fa"
    fa.write_text(fasta)
    if rows is not None:
        lines = [r if isinstance(r, str) else "\t".join(str(x) for x in r) for r in rows]
        vcf = tmp_path / f"{name}.vcf"
        vcf.write_text(HEADER + "".join(line + "\n" for line in lines))
        kw["vcf"] = vcf
    out = tmp_path / f"{name}.gpxil.tif"
    encode(fa, out, width=width, log=lambda m: None, **kw)
    return out


def rec(pos, ref, alt, gt, *, chrom="c1", filt="PASS", info=".", fmt="GT"):
    return (chrom, pos, ".", ref, alt, ".", filt, info, fmt, gt)


def planes(out, contig="c1"):
    with GpxilFile(out) as g:
        c = g[contig]
        return {p: c.read(p) for p in c.planes}, c.sidecar(), dict(c.entry)


def test_A_missing_genotype_keeps_called_variant(tmp_path):
    out = run(tmp_path, FA, [rec(2, "C", "T", "0|1"), rec(1, "ACG", "A", ".")])
    p, _, entry = planes(out)
    assert list(p["gt"][:3]) == [C.GT_UNKNOWN, 0x08, C.GT_UNKNOWN]
    assert entry["encoding"]["missing_genotypes"] == 1
    with GpxilFile(out) as g:
        assert haplotype_sequence(g["c1"], 2, keep_case=False).decode() == "NTN" + FA.split()[1][3:]


def test_B_blank_lines_in_vcf_body(tmp_path):
    out = run(tmp_path, FA, [rec(2, "C", "T", "0|1"), "", "   "])
    assert planes(out)[0]["gt"][1] == 0x08 and validate(out).ok


def test_C_format_keys_seen_late_still_get_planes(tmp_path):
    rng = np.random.default_rng(1)
    seq = "".join(rng.choice(list("ACGT"), size=30000))
    rows = [rec(i + 1, seq[i], "A" if seq[i] != "A" else "C", "0|1") for i in range(20001)]
    rows.append(rec(5, seq[4], "A" if seq[4] != "A" else "C", "0|1:50:30:77", chrom="c2", fmt="GT:GQ:DP:PS"))
    out = run(tmp_path, ">c1\n" + seq + "\n>c2\n" + seq[:100] + "\n", rows, width=1000)
    with GpxilFile(out) as g:
        assert g["c2"].planes == ["ref", "gt", "flags", "gq", "dp", "ps"]
        assert g["c2"].read("gq", 4, 5)[0] == 50 and g["c2"].read("ps", 4, 5)[0] == 77
        assert (g["c1"].read("gq") == C.QUAL_MISSING).all()


def test_E_non_ascii_contig_name_fails_cleanly(tmp_path):
    with pytest.raises(GpxilFormatError, match="printable ASCII"):
        run(tmp_path, ">chré\nACGT\n", width=4)
    assert not (tmp_path / "t.gpxil.tif").exists()


def test_F_fasta_name_ends_at_tab(tmp_path):
    fa = tmp_path / "x.fa"
    fa.write_text(">c1\tsome description\nACGT\n>c2  two spaces\nAC\n")
    recs = list(iter_fasta(fa))
    assert [(r.name, r.description) for r in recs] == [("c1", "some description"), ("c2", "two spaces")]
    assert [r.name for r in iter_fasta(fa, only={"c2"})] == ["c2"]


def test_G_empty_fasta_record_is_skipped(tmp_path):
    out = run(tmp_path, ">e\n>c1\nACGT\n", width=4)
    with GpxilFile(out) as g:
        assert g.contigs == ["c1"] and g.metadata["summary"]["skipped_empty_contigs"] == ["e"]
    assert validate(out).ok


def test_H_space_delimited_bed(tmp_path):
    bed = tmp_path / "callable.bed"
    bed.write_text("c1 0 10\n")
    out = run(tmp_path, FA, [rec(2, "C", "T", "0|1")], callable_bed=bed)
    p, _, entry = planes(out)
    assert entry["callable_bases"] == 10
    assert list(p["gt"][:2]) == [0x00, 0x08] and (p["gt"][10:] == C.GT_UNKNOWN).all()


def test_I_insertions_and_breakends(tmp_path):
    out = run(
        tmp_path,
        FA,
        [rec(2, "C", "CRY", "0|1"), rec(5, "A", "A.", "0|1"), rec(10, "C", "C]chr17:198982]", "0|1", info="SVTYPE=BND")],
    )
    _, side, entry = planes(out)
    assert [(e.offset, e.hap, e.kind, e.data) for e in side] == [
        (1, 2, "INS", "NN"),
        (4, 2, "COMPLEX", "REF=A;ALT=A."),
        (9, 2, "COMPLEX", "REF=C;ALT=C]chr17:198982]"),  # mate contig name keeps its case
    ]
    assert entry["encoding"]["nonacgt_inserted_bases"] == 2
    assert validate(out).ok


def test_J_symbolic_insertion_svlen_is_not_a_span(tmp_path):
    out = run(tmp_path, FA, [rec(3, "G", "<INS>", "0/1", info="SVTYPE=INS;SVLEN=8"), rec(17, "A", "<INS>", "1/1", info="SVLEN=300")])
    p, side, _ = planes(out)
    assert list(np.flatnonzero(p["flags"] & C.FLAG_VARIANT_SITE)) == [2, 16]
    assert not np.isin(C.hap1(p["gt"]), [C.HAP_DEL]).any() and not np.isin(C.hap2(p["gt"]), [C.HAP_DEL]).any()
    assert [(e.offset, e.hap) for e in side] == [(2, 2), (16, 1), (16, 2)]


def test_J_symbolic_deletion_uses_its_own_svlen(tmp_path):
    out = run(tmp_path, FA, [rec(3, "G", "<DEL>,<DUP>", "1/2", info="SVLEN=-4,10")])
    p, side, _ = planes(out)
    assert list(C.hap1(p["gt"])[2:8]) == [0, C.HAP_DEL, C.HAP_DEL, C.HAP_DEL, C.HAP_DEL, 0]
    assert [(e.offset, e.hap, e.data) for e in side] == [(2, 2, "REF=G;ALT=<DUP>")]


def test_K_vcf44_leading_phase_indicator(tmp_path):
    out = run(tmp_path, FA, [rec(2, "C", "T", "|1")])
    p, _, entry = planes(out)
    assert p["gt"][1] == 0x8A and entry["ploidy"] == "mixed"


@pytest.mark.parametrize("extra", [[], [rec(5, "A", "AGG", "0/1")]])
def test_M_exported_phase_needs_every_site_phased(tmp_path, extra):
    out = run(tmp_path, FA, [rec(5, "ACG", "A", "0/1"), rec(6, "C", "T", "1|0")] + extra)
    buf = io.StringIO()
    with GpxilFile(out) as g:
        write_vcf(g, buf)
    rows = [line.split("\t") for line in buf.getvalue().splitlines() if line.startswith("c1\t5\t")]
    assert len(rows) == 1 and rows[0][9] == "1/2"


def test_O_vcf_export_uses_canonical_names(datadir):
    buf = io.StringIO()
    with GpxilFile(datadir / "worked-example.gpxil.tif") as g:
        write_vcf(g, buf, ["T"])  # an alias
    text = buf.getvalue()
    assert "##contig=<ID=chrT,length=10>" in text and "\nchrT\t2\t" in text and "\nT\t" not in text


def test_P_no_ref_files_with_reference(tmp_path, capsys):
    out = run(tmp_path, FA, [rec(2, "C", "T", "0|1")], include_ref=False)
    assert validate(out).ok
    assert main(["vcf", str(out)]) == 1
    assert "stored without its reference sequence" in capsys.readouterr().err
    fa = tmp_path / "t.fa"
    vcf = tmp_path / "back.vcf"
    assert main(["vcf", str(out), "--reference", str(fa), "-o", str(vcf)]) == 0
    assert "c1\t2\t.\tC\tT\t" in vcf.read_text()
    h1 = tmp_path / "h2.fa"
    assert main(["fasta", str(out), "--which", "2", "--reference", str(fa), "-o", str(h1)]) == 0
    assert h1.read_text().split("\n")[1] == "ATGTACGTACGTACGTACGT"
    wrong = tmp_path / "wrong.fa"
    wrong.write_text(">c1\nACGTACGTACGTACGTACGA\n")
    assert main(["vcf", str(out), "--reference", str(wrong)]) == 1
    assert "does not match the md5" in capsys.readouterr().err


def test_R_haploid_contig_with_diploid_calls(tmp_path):
    out = run(tmp_path, FA, [rec(2, "C", "T", "0|1"), rec(6, "C", "A", "1|0")], haploid_contigs=["c1"])
    p, _, entry = planes(out)
    assert list(p["gt"][:7]) == [0x0A, 0x08, 0x0A, 0x0A, 0x0A, 0x10, 0x0A]  # the VCF's diploid calls win
    assert entry["encoding"]["ploidy_mismatches"] == 2 and entry["ploidy"] == "mixed"
    with GpxilFile(out) as g:
        assert g.metadata["sample"]["haploid_regions"] == "contigs c1"


def test_S_phase_set_ids_do_not_collide(tmp_path):
    rows = [rec(2, "C", "T", "0|1:4294967295", fmt="GT:PS"), rec(6, "C", "A", "1|0:blockB", fmt="GT:PS")]
    out = run(tmp_path, FA, rows)
    p, _, entry = planes(out)
    assert int(p["ps"][1]) == 4294967295 and int(p["ps"][5]) == 4294967294
    assert entry["phase_set_ids"] == {"4294967294": "blockB"} and entry["phase_sets"] == 2


def test_T_unphased_spanning_deletion_goes_with_the_deletion(tmp_path):
    out = run(tmp_path, FA, [rec(2, "CGTA", "C", "0/1"), rec(4, "T", "*,A", "1/2")])
    p, _, entry = planes(out)
    assert list(p["gt"][1:5]) == [0x00, 0x0D, 0x1D, 0x0D]
    assert entry["encoding"]["spanning_deletions_reassigned"] == 1 and entry["encoding"]["conflicts"] == 0


def test_U_iupac_ref_alleles(tmp_path):
    out = run(tmp_path, FA, [rec(2, "R", "T", "0|1"), rec(5, "R", "T", "0|1")])
    assert planes(out)[2]["encoding"]["ref_mismatches"] == 1  # R = A/G matches A (pos 5) but not C (pos 2)


def test_BY_reserved_states_read_as_unknown(tmp_path):
    ref = encode_sequence(b"ACGTACGTAC")
    gt = np.zeros(10, np.uint8)
    gt[2] = 0x30
    p = tmp_path / "by.gpxil.tif"
    with GpxilWriter(p, width=5, sample={"id": "S"}) as w:
        w.add_contig("c", {"ref": ref, "gt": gt, "flags": np.zeros(10, np.uint8)})
    with GpxilFile(p) as g:
        assert haplotype_sequence(g["c"], 1).decode() == "ACNTACGTAC"
        assert genotype_iupac_sequence(g["c"]).decode() == "ACNTACGTAC"


def _rewrite_metadata(path, fn):
    with tifffile.TiffFile(path) as tf:
        meta = json.loads(tf.pages[0].description)
    fn(meta)
    tifffile.tiffcomment(path, json.dumps(meta))


def test_V_validator_finds_structural_problems(tmp_path):
    ref = encode_sequence(b"ACGTACGTAC")
    fl = np.zeros(10, np.uint8)
    fl[4] = C.FLAG_INS1
    fl[1] = 0  # no PHASED flag ...
    gt = np.zeros(10, np.uint8)
    gt[7] = 0xC0  # COMPLEX without a sidecar entry
    ps = np.zeros(10, np.uint32)
    ps[1] = 9  # ... but a phase set
    p = tmp_path / "v.gpxil.tif"
    with GpxilWriter(p, width=5, sample={"id": "S"}) as w:
        w.add_contig(
            "c",
            {"ref": ref, "gt": gt, "flags": fl, "ps": ps},
            sidecar=[
                SidecarEntry(4, 1, "INS", "CC"),
                SidecarEntry(4, 1, "INS", "AA"),
                SidecarEntry(2, 1, "INS", "acg"),
                SidecarEntry(6, 2, "COMPLEX", "REF=G;ALT=<INV>"),
                SidecarEntry(6, 2, "COMPLEX", "REF=G;ALT=<DUP>"),
            ],
            refget="SQ." + "A" * 32,
        )
        w.add_contig("d", {"ref": ref, "gt": gt * 0, "flags": fl * 0, "gq": np.full(10, 255, np.uint8), "x-cov": np.ones(10, np.uint16)})
    text = "\n".join(validate(p).errors)
    for expected in (
        "refget",
        "not sorted",
        "more than one INS entry",
        "must be uppercase",
        "COMPLEX runs have no COMPLEX sidecar entry",
        "ps: non-zero phase set where the PHASED flag is clear",
        "more than one COMPLEX entry",
    ):
        assert expected in text, expected

    def damage(meta):
        meta["sample"] = None
        del meta["contigs"][1]["pages"]["gt"], meta["contigs"][1]["pages"]["flags"]
        del meta["planes"]["x-cov"]
        meta["contigs"].reverse()

    _rewrite_metadata(p, damage)
    text = "\n".join(validate(p).errors)
    for expected in ("'sample' is not an object", "gq without a gt plane", "x-cov is not described", "page order", "page 0"):
        assert expected in text, expected


def test_W_small_uncompressed_file_stays_classic_tiff(tmp_path):
    p = tmp_path / "w.gpxil.tif"
    with GpxilWriter(p, width=5, compression="none") as w:
        w.add_contig("c", {"ref": encode_sequence(b"ACGTACGTAC")})
    with tifffile.TiffFile(p) as tf:
        assert not tf.is_bigtiff


def test_writer_requires_sample_and_gt_for_genotype_planes(tmp_path):
    ref = encode_sequence(b"ACGT")
    z = np.zeros(4, np.uint8)
    with GpxilWriter(tmp_path / "a.gpxil.tif", width=4) as w:
        with pytest.raises(GpxilFormatError, match="sample"):
            w.add_contig("c", {"ref": ref, "gt": z, "flags": z})
        with pytest.raises(GpxilFormatError, match="need a gt plane"):
            w.add_contig("c", {"ref": ref, "gq": z})
        w.add_contig("c", {"ref": ref})


def test_overwritten_complex_allele_leaves_no_sidecar_entry(tmp_path):
    out = run(tmp_path, FA, [rec(5, "A", "<INV>", "1|0", info="END=9"), rec(5, "A", "G", "1|0")])
    p, side, entry = planes(out)
    assert p["gt"][4] == 0x40 and side == [] and entry["encoding"]["conflicts"] == 1
    assert validate(out).ok


def test_two_structural_alleles_on_one_base_conflict(tmp_path):
    out = run(tmp_path, FA, [rec(5, "A", "<INV>", "1|0", info="END=9"), rec(5, "A", "<DUP>", "1|0", info="END=9")])
    p, side, entry = planes(out)
    assert entry["encoding"]["conflicts"] == 1
    assert [(e.offset, e.hap, e.data) for e in side] == [(4, 1, "REF=A;ALT=<DUP>;END=9")]


def test_streamed_haplotype_fasta_equals_whole_sequence(tmp_path, sim_files):
    fa, vcf = sim_files
    out = tmp_path / "s.gpxil.tif"
    encode(fa, out, vcf=vcf, width=333, log=lambda m: None)
    with GpxilFile(out) as g:
        c = g["1"]
        for h in (1, 2):
            whole = haplotype_sequence(c, h)
            assert b"".join(iter_haplotype(c, h, chunk=777)) == whole
            buf = io.BytesIO()
            write_fasta_stream(buf, "x", iter_haplotype(c, h, chunk=1001), line_width=61)
            lines = buf.getvalue().decode().split("\n")
            assert lines[0] == ">x" and "".join(lines[1:]).encode() == whole
            assert all(len(x) == 61 for x in lines[1:-2])


def test_annotation_precedence_is_per_field(tmp_path):
    rows = [
        rec(5, "A", "G", "0|1:50", fmt="GT:GQ"),  # changes base 4, has GQ but no DP
        rec(3, "G", "<NON_REF>", "0/0:10:20", fmt="GT:GQ:DP", info="END=8"),  # reference block over 2..7
    ]
    p, _, _ = planes(run(tmp_path, FA, rows))
    assert list(p["gq"][1:9]) == [255, 10, 10, 50, 10, 10, 10, 255]  # the variant's GQ wins at its base
    assert list(p["dp"][1:9]) == [255, 20, 20, 20, 20, 20, 20, 255]  # DP only comes from the block
    assert p["gt"][4] == 0x04
