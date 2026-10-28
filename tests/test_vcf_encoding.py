import numpy as np

from gpxiltools import codes as C
from gpxiltools.fasta import encode_sequence
from gpxiltools.vcf import Call, allele_events, encode_calls, iter_calls


def ev(ref, alt):
    e = allele_events(ref, alt)
    return e.subs, e.dels, e.ins


def test_allele_events_basic():
    assert ev("A", "G") == ([(0, C.BASE_G)], [], [])
    assert ev("AC", "GT") == ([(0, C.BASE_G), (1, C.BASE_T)], [], [])
    assert ev("ACGT", "A") == ([], [(1, 3)], [])
    assert ev("A", "ATTG") == ([], [], [(0, "TTG")])


def test_deletion_in_repeat_is_left_aligned():
    # TACAC -> TAC: the two deleted bases are placed right after the anchor
    assert ev("TACAC", "TAC") == ([], [(1, 2)], [])
    # an insertion of A after A keeps its anchor inside the record
    assert ev("A", "AA") == ([], [], [(0, "A")])
    assert ev("TAC", "TACAC") == ([], [], [(0, "AC")])


def test_complex_allele_is_subs_then_indel():
    s, d, i = ev("A", "GTT")
    assert s == [(0, C.BASE_G)] and i == [(0, "TT")] and d == []
    s, d, i = ev("AC", "G")
    assert s == [(0, C.BASE_G)] and d == [(1, 1)]


REF = encode_sequence(b"ACGTACGTACGTACGTACGT")


def encode(calls, **kw):
    planes, side, stats, ps = encode_calls(REF.copy(), calls, contig="t", planes=kw.pop("planes", ("gt", "flags")), **kw)
    return planes, side, stats


def test_snv_genotypes_and_phase():
    planes, _, _ = encode(
        [
            Call(pos=2, ref="C", alts=["T"], filter="PASS", gt="0|1"),
            Call(pos=3, ref="G", alts=["A"], filter="PASS", gt="1/0"),
            Call(pos=4, ref="T", alts=["C", "G"], filter="PASS", gt="2|1"),
            Call(pos=5, ref="A", alts=["G"], filter="q10", gt="1/1"),
        ]
    )
    gt, fl = planes["gt"], planes["flags"]
    assert C.describe_gt_byte(int(gt[1])) == "REF|T"
    assert C.describe_gt_byte(int(gt[2])) == "A|REF"  # unphased order follows the GT string
    assert C.describe_gt_byte(int(gt[3])) == "G|C"
    assert C.describe_gt_byte(int(gt[4])) == "G|G"
    assert fl[1] & C.FLAG_PHASED and not fl[2] & C.FLAG_PHASED
    assert fl[4] & C.FLAG_FILTERED and not fl[1] & C.FLAG_FILTERED
    assert all(fl[i] & C.FLAG_VARIANT_SITE for i in (1, 2, 3, 4)) and not fl[0]


def test_nocall_haploid_and_ref_mismatch():
    planes, _, stats = encode(
        [
            Call(pos=1, ref="A", alts=["G"], filter=".", gt=".|1"),
            Call(pos=6, ref="C", alts=["T"], filter=".", gt="1"),
            Call(pos=8, ref="A", alts=["C"], filter=".", gt="0/1"),  # reference has T here
        ]
    )
    gt = planes["gt"]
    assert C.describe_gt_byte(int(gt[0])) == "UNKNOWN|G"
    assert C.describe_gt_byte(int(gt[5])) == "T|ABSENT"
    assert stats.ref_mismatches == 1


def test_deletion_insertion_and_spanning_star():
    planes, side, stats = encode(
        [
            Call(pos=2, ref="CGTA", alts=["C"], filter=".", gt="0|1"),  # deletes 3-5 (1-based) on hap2
            Call(pos=4, ref="T", alts=["A", "*"], filter=".", gt="1|2"),  # SNV on hap1, hap2 deleted upstream
            Call(pos=9, ref="A", alts=["ATT"], filter=".", gt="1|1"),
        ]
    )
    gt, fl = planes["gt"], planes["flags"]
    assert [C.describe_gt_byte(int(v)) for v in gt[1:6]] == ["REF|REF", "REF|DEL", "A|DEL", "REF|DEL", "REF|REF"]
    assert fl[8] & C.FLAG_INS1 and fl[8] & C.FLAG_INS2
    assert [(e.offset, e.hap, e.kind, e.data) for e in side] == [(8, 1, "INS", "TT"), (8, 2, "INS", "TT")]
    assert stats.conflicts == 0 and stats.spanning_deletion_alleles == 1


def test_ref_allele_never_overwrites_a_variant():
    # an overlapping record that says REF must not undo an SNV called by another record
    planes, _, stats = encode(
        [
            Call(pos=3, ref="G", alts=["C"], filter=".", gt="1|0"),
            Call(pos=2, ref="CGT", alts=["C"], filter=".", gt="0|1"),
        ]
    )
    assert C.describe_gt_byte(int(planes["gt"][2])) == "C|DEL"
    assert stats.conflicts == 0


def test_conflicting_variants_are_counted_last_wins():
    planes, _, stats = encode(
        [
            Call(pos=3, ref="G", alts=["C"], filter=".", gt="0|1"),
            Call(pos=3, ref="G", alts=["T"], filter=".", gt="0|1"),  # split multi-allelic, phased: as written
        ]
    )
    assert C.describe_gt_byte(int(planes["gt"][2])) == "REF|T"
    assert stats.conflicts == 1


def test_unphased_split_multiallelic_records_go_one_to_each_haplotype():
    # (spec 0.2, section 6.5: the second record's T doesn't fit haplotype 2, which has the first's C)
    planes, _, stats = encode(
        [
            Call(pos=3, ref="G", alts=["C"], filter=".", gt="0/1"),
            Call(pos=3, ref="G", alts=["T"], filter=".", gt="0/1"),
        ]
    )
    assert C.describe_gt_byte(int(planes["gt"][2])) == "T|C"
    assert stats.conflicts == 0 and stats.overlapping_calls_reassigned == 1


def test_gvcf_block_gq_dp_and_callable_mask():
    callable_mask = np.zeros(20, dtype=bool)
    callable_mask[0:10] = True
    planes, _, _ = encode(
        [
            Call(pos=11, ref="G", alts=["<NON_REF>"], filter=".", gt="0/0", end=15, gq=40, dp=300),
            Call(pos=3, ref="G", alts=["A", "<NON_REF>"], filter=".", gt="0/1", gq=99, dp=20),
        ],
        planes=("gt", "flags", "gq", "dp"),
        callable_mask=callable_mask,
    )
    gt, gq, dp = planes["gt"], planes["gq"], planes["dp"]
    assert gt[0] == 0 and gt[19] == C.GT_UNKNOWN  # outside callable: unknown
    assert (gt[10:15] == 0).all()  # the gVCF block asserts REF over POS..END
    assert gq[2] == 99 and (gq[10:15] == 40).all() and gq[0] == C.QUAL_MISSING
    assert (dp[10:15] == C.QUAL_MAX).all()  # depth 300 capped at 254


def test_symbolic_deletion_and_other_sv():
    planes, side, stats = encode(
        [
            Call(pos=2, ref="C", alts=["<DEL>"], filter=".", gt="0/1", end=6),
            Call(pos=13, ref="A", alts=["<INV>"], filter=".", gt="1/1", end=16),
        ]
    )
    h2 = C.hap2(planes["gt"])
    assert (h2[2:6] == C.HAP_DEL).all() and h2[1] == C.HAP_REF
    assert C.hap1(planes["gt"])[12] == C.HAP_COMPLEX
    assert stats.ref_mismatches == 0
    assert any(e.kind == "COMPLEX" and "ALT=<INV>" in e.data and "END=16" in e.data for e in side)
    assert stats.symbolic_alleles == 3


def test_haploid_mask_default_absent_and_ps_plane():
    hm = np.zeros(20, dtype=bool)
    hm[10:] = True
    planes, _, _ = encode(
        [Call(pos=2, ref="C", alts=["G"], filter=".", gt="0|1", ps="1234"), Call(pos=14, ref="C", alts=["G"], filter=".", gt="1")],
        planes=("gt", "flags", "ps"),
        haploid_mask=hm,
    )
    h2 = C.hap2(planes["gt"])
    assert (h2[10:] == C.HAP_ABSENT).all() and h2[0] == C.HAP_REF
    assert planes["ps"][1] == 1234 and planes["ps"][0] == 0


def test_iter_calls_parses_sample_column():
    lines = [
        "chr1\t10\t.\tA\tG,T\t30\tPASS\t.\tGT:GQ:DP:PS\t1|2:50:12:10",
        "chr1\t20\t.\tN\t<DEL>\t.\t.\tSVTYPE=DEL;END=40\tGT\t0/1",
    ]
    calls = [c for _, c in iter_calls(lines, 0)]
    assert calls[0].gt == "1|2" and calls[0].gq == 50 and calls[0].dp == 12 and calls[0].ps == "10"
    assert calls[1].end == 40 and calls[1].alts == ["<DEL>"]
