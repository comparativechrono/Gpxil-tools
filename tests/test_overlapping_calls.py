"""Unphased calls that overlap another call (gpxil 0.2, spec section 6.5).

Heterozygous calls that overlap are alternatives at one place: two of them can't both be on one haplotype. Applied
as written, both alleles of ``0/1`` go to haplotype 2, where they run together. In 0.2, an allele that doesn't fit
the haplotype the written order gives it (an earlier call's ALT allele occupies there an offset it occupies: the
bases an allele changes, and the anchor of a plain insertion or deletion) goes to the other haplotype, if both
alleles fit the other way round.
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


def test_split_multiallelic_records_go_one_to_each_haplotype_in_either_order():
    # REF=CGTA ALT=C,CA (1/2) split into two 0/1 records: deletions of three bases and of two, at one anchor
    for calls in ([call(2, "CGTA", "C", "0/1"), call(2, "CGT", "C", "0/1")], [call(2, "CGT", "C", "0/1"), call(2, "CGTA", "C", "0/1")]):
        gt, _, stats = encode(calls)
        assert sorted(gt[2:5]) == ["DEL|DEL", "DEL|DEL", "DEL|REF"] or sorted(gt[2:5]) == ["DEL|DEL", "DEL|DEL", "REF|DEL"]
        assert stats.overlapping_calls_reassigned == 1 and stats.conflicts == 0
    # phased, as written, both on haplotype 2: the shorter deletion lies inside the longer, which stays (no conflict)
    gt, _, stats = encode([call(2, "CGTA", "C", "0|1"), call(2, "CGT", "C", "0|1")])
    assert gt[2:5] == ["REF|DEL"] * 3 and stats.conflicts == 0


def test_a_deletion_and_a_substitution_at_its_anchor_either_order():
    # REF=CG ALT=C,AG (1/2) split: the substitution is at the deletion's anchor, so the two go apart, in either order
    for calls in ([call(2, "CG", "C", "0/1"), call(2, "C", "A", "0/1")], [call(2, "C", "A", "0/1"), call(2, "CG", "C", "0/1")]):
        gt, _, stats = encode(calls)
        assert gt[1:3] in (["A|REF", "REF|DEL"], ["REF|A", "DEL|REF"]) and stats.overlapping_calls_reassigned == 1


def test_a_deletion_and_an_insertion_at_one_anchor_either_order():
    # REF=CG ALT=C,CTG (1/2) split
    for calls in ([call(2, "CG", "C", "0/1"), call(2, "C", "CT", "0/1")], [call(2, "C", "CT", "0/1"), call(2, "CG", "C", "0/1")]):
        gt, side, stats = encode(calls)
        deleted = 1 if gt[2] == "DEL|REF" else 2
        assert gt[2] in ("DEL|REF", "REF|DEL") and [(e.offset, e.hap) for e in side] == [(1, 3 - deleted)]
        assert stats.overlapping_calls_reassigned == 1


def test_a_record_written_twice_stays_heterozygous():
    gt, _, stats = encode([call(3, "G", "A", "0/1"), call(3, "G", "A", "0/1")])
    assert gt[2] == "REF|A" and stats.overlapping_calls_reassigned == 0
    # in either order: the same allele goes to the haplotype that has it, not the other
    gt, _, stats = encode([call(3, "G", "A", "0/1"), call(3, "G", "A", "1/0")])
    assert gt[2] == "REF|A" and stats.overlapping_calls_reassigned == 1
    gt, _, stats = encode([call(3, "G", "A", "1|0"), call(3, "G", "C,A", "1/2")])  # (phased first, then C/A)
    assert gt[2] == "A|C" and stats.overlapping_calls_reassigned == 1
    # homozygous before: the same allele fits either haplotype
    gt, _, stats = encode([call(3, "G", "A", "1/1"), call(3, "G", "A", "0/1")])
    assert gt[2] == "A|A" and stats.overlapping_calls_reassigned == 0
    gt, _, stats = encode([call(2, "CGT", "C", "0/1"), call(2, "CGT", "C", "0/1")])
    assert gt[2:4] == ["REF|DEL", "REF|DEL"] and stats.overlapping_calls_reassigned == 0
    # (the same allele written with more of the reference around it is the same allele)
    gt, _, stats = encode([call(2, "CG", "C", "0/1"), call(2, "CGT", "CT", "0/1")])
    assert gt[2] == "REF|DEL" and stats.overlapping_calls_reassigned == 0


def test_an_allele_padded_to_a_longer_ref_occupies_only_what_it_changes():
    # GIAB HG002 v4.2.1, chr1:237,542,058: CTTTA>CTA,C 1/2, then ATTTATT>A 0/1 at its last base, which only haplotype 2
    # deletes; CTA (TT deleted) leaves that base, so the deletion after it goes to haplotype 1
    seq = "ACCTTTATTTATTGACGTACGT"
    ref = encode_sequence(seq.encode())
    planes, _, stats, _ = encode_calls(ref, [call(3, "CTTTA", "CTA,C", "1/2"), call(7, "ATTTATT", "A", "0/1")], contig="t", planes=("gt",))
    gt = [C.describe_gt_byte(int(v)) for v in planes["gt"]]
    assert gt[2:14] == ["REF|REF", "DEL|DEL", "DEL|DEL", "REF|DEL", "REF|DEL"] + ["DEL|REF"] * 6 + ["REF|REF"]
    assert stats.overlapping_calls_reassigned == 1 and stats.conflicts == 0


def test_alleles_that_change_different_bases_of_one_ref_are_not_alternatives():
    # REF=CG ALT=TG,CA (1/2) split into two 0/1 records: once trimmed, a substitution at each base; they occupy no
    # offset in common, so they are applied as written, both on haplotype 2 (spec 6.5 says so)
    gt, _, stats = encode([call(2, "CG", "TG", "0/1"), call(2, "CG", "CA", "0/1")])
    assert gt[1:3] == ["REF|T", "REF|A"] and stats.overlapping_calls_reassigned == 0


def test_allele_numbers_are_digits():
    # "²" is a digit to Python, but no allele number: counted, and nothing written (it used to raise an error)
    gt, _, stats = encode([call(2, "CGTA", "C", "0/1"), call(4, "T", "*,A", "0/²"), call(6, "C", "T", "0/+1"), call(8, "T", "A", "0/ 1")])
    assert stats.bad_allele_index == 3 and gt[5] == "REF|REF" and gt[7] == "REF|REF"


def test_a_record_without_ref_is_skipped():
    # (an insertion with no anchor would go before the record, or before the contig: an invalid sidecar)
    gt, side, stats = encode([call(1, "", "G", "0/1"), call(5, "", "T", "0/1")])
    assert stats.skipped_out_of_range == 2 and not side and set(gt) == {"REF|REF"}


def test_alleles_given_in_lowercase_read_as_uppercase():
    gt, side, _ = encode([call(3, "g", "a", "0|1"), call(5, "a", "act", "1|0"), call(9, "a", "<del>", "0|1")])
    assert gt[2] == "REF|A" and [(e.offset, e.hap, e.data) for e in side] == [(4, 1, "CT"), (8, 2, "REF=A;ALT=<del>")]


def test_an_unknown_base_at_an_n_is_unknown():
    # the one-base fast path: ALT N (or R) at a reference N is UNKNOWN, not REF (N and UNKNOWN share the code 15)
    ref = encode_sequence(b"ACGTNACGT")
    planes, _, _, _ = encode_calls(ref, [call(5, "N", "R", "0|1"), call(5, "N", "A", "1|0")], contig="t", planes=("gt",))
    assert C.describe_gt_byte(int(planes["gt"][4])) == "A|UNKNOWN"


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
    assert "1 unphased call overlaps another, so its alleles were put the other way round (gpxil 0.3" in capsys.readouterr().err
    with GpxilFile(out) as g:
        assert FORMAT_VERSION == "0.3" and g.metadata["gpxil"] == "0.3"
        assert g["c1"].entry["encoding"]["overlapping_calls_reassigned"] == 1
        assert g.metadata["summary"]["overlapping_calls_reassigned"] == 1
        assert haplotype_sequence(g["c1"], 1) == b"ACGTCGTACGTACGTACGT" and haplotype_sequence(g["c1"], 2) == b"ACACGTACGTACGTACGT"
    with tifffile.TiffFile(out) as t:
        assert all(json.loads(p.description)["gpxil"] == "0.3" for p in t.pages)
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
    # a file of a newer 0.x is read, and checked as 0.3; its copy is still of that version
    new = tmp_path / "new.gpxil.tif"
    with GpxilWriter(new, width=20, version="0.4") as w:
        w.add_contig("c1", {"ref": REF.copy()})
    assert any("declares gpxil 0.4, newer than this validator" in m for m in validate(new).warnings)
    subset(new, tmp_path / "copy4.gpxil.tif", ["c1"], log=lambda m: None)
    with GpxilFile(tmp_path / "copy4.gpxil.tif") as g:
        assert g.metadata["gpxil"] == "0.4"
