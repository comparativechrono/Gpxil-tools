"""Unphased calls that overlap another call (gpxil 0.2, spec section 6.5).

Two heterozygous calls that overlap can't both be on one haplotype. Applied as written, both alleles of ``0/1`` go
to haplotype 2, where they run together. In 0.2, an allele that doesn't fit the haplotype the written order gives it
(it would write REF or a variant where that haplotype already has another call's variant) goes to the other
haplotype, if both alleles fit the other way round.
"""

import json

import tifffile

from gpxiltools import codes as C
from gpxiltools.container import FORMAT_VERSION, GpxilFile, GpxilWriter
from gpxiltools.export import haplotype_sequence
from gpxiltools.fasta import encode_sequence
from gpxiltools.subset import subset
from gpxiltools.validate import validate
from gpxiltools.vcf import Call, encode_calls

SEQ = "ACGTACGTACGTACGTACGT"
REF = encode_sequence(SEQ.encode())


def encode(calls):
    planes, side, stats, _ = encode_calls(REF.copy(), calls, contig="t", planes=("gt", "flags"))
    return [C.describe_gt_byte(int(v)) for v in planes["gt"]], side, stats


def call(pos, ref, alt, gt, **kw):
    return Call(pos=pos, ref=ref, alts=alt.split(","), filter="PASS", gt=gt, **kw)


def test_two_overlapping_deletions_go_one_to_each_haplotype():
    # the second deletion's anchor (offset 3) is a base the first deletes, so the two can't be on one haplotype
    # (as GIAB's HG002 v4.2.1 writes chr17:72,809,199)
    gt, _, stats = encode([call(2, "CGT", "C", "0/1"), call(4, "TA", "T", "0/1")])
    assert gt[1:6] == ["REF|REF", "REF|DEL", "REF|DEL", "DEL|REF", "REF|REF"]
    assert stats.overlapping_calls_reassigned == 1 and stats.conflicts == 0
    # spec 0.1 put both on haplotype 2, as one deletion of three bases: phased calls still are applied as written
    gt, _, stats = encode([call(2, "CGT", "C", "0|1"), call(4, "TA", "T", "0|1")])
    assert gt[1:6] == ["REF|REF", "REF|DEL", "REF|DEL", "REF|DEL", "REF|REF"]
    assert stats.overlapping_calls_reassigned == 0


def test_a_substitution_inside_a_deletion_goes_to_the_other_haplotype():
    gt, _, stats = encode([call(2, "CGTA", "C", "0/1"), call(4, "T", "G", "0/1")])
    assert gt[2:5] == ["REF|DEL", "G|DEL", "REF|DEL"]
    assert stats.overlapping_calls_reassigned == 1 and stats.conflicts == 0
    # written the right way round already: nothing to do
    gt, _, stats = encode([call(2, "CGTA", "C", "0/1"), call(4, "T", "G", "1/0")])
    assert gt[3] == "G|DEL" and stats.overlapping_calls_reassigned == 0


def test_the_ref_allele_always_fits():
    # the deletion's REF allele over the substitution's base is no reason to move anything
    gt, _, stats = encode([call(3, "G", "C", "0/1"), call(2, "CGT", "C", "1/0")])
    assert gt[2] == "DEL|C" and stats.overlapping_calls_reassigned == 0 and stats.conflicts == 0


def test_neither_way_fits_so_as_written():
    # haplotype 1 has A and haplotype 2 C: a T fits neither, so it is applied as written, a conflict (spec 6.4)
    gt, _, stats = encode([call(3, "G", "A,C", "1|2"), call(3, "G", "T", "0/1")])
    assert gt[2] == "A|T" and stats.conflicts == 1 and stats.overlapping_calls_reassigned == 0


def test_two_different_insertions_at_one_base():
    gt, side, stats = encode([call(5, "A", "AC", "0/1"), call(5, "A", "AGG", "0/1")])
    assert [(e.offset, e.hap, e.data) for e in side] == [(4, 1, "GG"), (4, 2, "C")]
    assert stats.overlapping_calls_reassigned == 1 and stats.conflicts == 0
    # the same insertion twice fits where it is
    gt, side, stats = encode([call(5, "A", "AC", "0/1"), call(5, "A", "AC", "0/1")])
    assert [(e.offset, e.hap) for e in side] == [(4, 2)] and stats.overlapping_calls_reassigned == 0


def test_two_structural_alleles_at_one_base():
    gt, side, stats = encode([call(5, "A", "<INV>", "0/1", end=9), call(5, "A", "<DUP>", "0/1", end=9)])
    assert [(e.offset, e.hap, e.data) for e in side] == [(4, 1, "REF=A;ALT=<DUP>;END=9"), (4, 2, "REF=A;ALT=<INV>;END=9")]
    assert stats.overlapping_calls_reassigned == 1 and stats.conflicts == 0


def test_a_symbolic_deletion_and_a_substitution_inside_it():
    gt, _, stats = encode([call(2, "C", "<DEL>", "0/1", end=6), call(4, "T", "A", "0/1")])
    assert gt[3] == "A|DEL" and stats.overlapping_calls_reassigned == 1


def test_spanning_deletion_rule_comes_first():
    # a '*' allele is placed by its own rule (spec 6.5), not this one
    gt, _, stats = encode([call(2, "CGTA", "C", "0/1"), call(4, "T", "*,A", "1/2")])
    assert gt[3] == "A|DEL" and stats.spanning_deletions_reassigned == 1 and stats.overlapping_calls_reassigned == 0


VCF_HEADER = (
    "##fileformat=VCFv4.3\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
)


def test_files_are_gpxil_0_2_and_say_what_was_reassigned(tmp_path, capsys):
    fa = tmp_path / "t.fa"
    fa.write_text(f">c1\n{SEQ}\n")
    vcf = tmp_path / "t.vcf"
    vcf.write_text(VCF_HEADER + "c1\t2\t.\tCGT\tC\t.\tPASS\t.\tGT\t0/1\nc1\t4\t.\tTA\tT\t.\tPASS\t.\tGT\t0/1\n")
    out = tmp_path / "t.gpxil.tif"
    from gpxiltools.cli import main

    assert main(["encode", str(fa), "-v", str(vcf), "-o", str(out), "--width", "20"]) == 0
    assert "1 unphased call overlaps another, so its alleles were put the other way round (gpxil 0.2" in capsys.readouterr().err
    with GpxilFile(out) as g:
        assert FORMAT_VERSION == "0.2" and g.metadata["gpxil"] == "0.2"
        assert g["c1"].entry["encoding"]["overlapping_calls_reassigned"] == 1
        assert g.metadata["summary"]["overlapping_calls_reassigned"] == 1
        assert haplotype_sequence(g["c1"], 1) == b"ACGTCGTACGTACGTACGT" and haplotype_sequence(g["c1"], 2) == b"ACACGTACGTACGTACGT"
    with tifffile.TiffFile(out) as t:
        assert all(json.loads(p.description)["gpxil"] == "0.2" for p in t.pages)
    assert validate(out).ok and not validate(out).warnings


def test_a_copy_keeps_the_version_of_its_source(tmp_path):
    # a file of gpxil 0.1, as gpxiltools 0.5 wrote it: its copy is still 0.1, its planes made by 0.1's rules
    old = tmp_path / "old.gpxil.tif"
    with GpxilWriter(old, width=20, version="0.1") as w:
        w.add_contig("c1", {"ref": REF.copy()})
        w.add_contig("c2", {"ref": REF.copy()})
    with GpxilFile(old) as g:
        assert g.metadata["gpxil"] == "0.1"
    assert validate(old).ok and not any("declares" in m for m in validate(old).warnings)
    subset(old, tmp_path / "copy.gpxil.tif", ["c2"], log=lambda m: None)
    with GpxilFile(tmp_path / "copy.gpxil.tif") as g:
        assert g.metadata["gpxil"] == "0.1" and g.contigs == ["c2"]
    # a file of a newer 0.x is read, and checked as 0.2
    new = tmp_path / "new.gpxil.tif"
    with GpxilWriter(new, width=20, version="0.3") as w:
        w.add_contig("c1", {"ref": REF.copy()})
    assert any("declares gpxil 0.3, newer than this validator" in m for m in validate(new).warnings)
