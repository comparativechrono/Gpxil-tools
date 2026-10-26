import gzip

import numpy as np
import pytest

from gpxiltools import codes as C
from gpxiltools.fasta import FastaError, encode_sequence, iter_fasta, sequence_digests


def test_iupac_round_trip_all_symbols():
    text = b"ACGTRYSWKMBDHVN-acgtryswkmbdhvn"
    codes = encode_sequence(text)
    assert C.ref_to_ascii(codes) == text
    # the 4-bit set: R = A|G, N = all four
    assert codes[4] == C.BASE_A | C.BASE_G
    assert codes[14] == 15
    assert codes[15] == 0  # gap
    assert all(c & C.SOFTMASK for c in codes[16:])


def test_u_and_dot():
    assert list(encode_sequence(b"Uu.")) == [C.BASE_T, C.BASE_T | C.SOFTMASK, 0]


def test_invalid_character_reports_position():
    with pytest.raises(FastaError, match=r"'\*' at offset 3"):
        encode_sequence(b"ACG*T", name="x")
    assert list(encode_sequence(b"AC*", invalid="N")) == [1, 2, 15]


def test_genotype_iupac_and_dosage():
    ref = encode_sequence(b"AACT")
    # pos0 hom-ref, pos1 het A/G, pos2 hom T, pos3 deleted on hap1
    gt = np.array([0x00, 0x04, 0x88, (C.HAP_DEL << 4) | 0], dtype=np.uint8)
    iu = C.genotype_iupac(ref, gt)
    assert C.IUPAC[iu[1]] == "R"
    assert C.IUPAC[iu[2]] == "T"
    assert C.IUPAC[iu[3]] == "T"
    d = C.dosage(ref, gt)
    assert d.tolist() == [[2, 0, 0, 0], [1, 0, 1, 0], [0, 0, 0, 2], [0, 0, 0, 1]]


def test_describe_gt_byte():
    assert C.describe_gt_byte(0x04) == "REF|G"
    assert C.describe_gt_byte(0xDA) == "DEL|ABSENT"


def test_fasta_parsing_edge_cases(tmp_path):
    p = tmp_path / "x.fa.gz"
    content = b">one first record\r\nACGT\r\nac\r\n\r\n>two\nNNNN\nAC GT\n>three\nA"
    with gzip.open(p, "wb") as fh:
        fh.write(content)
    recs = list(iter_fasta(p))
    assert [r.name for r in recs] == ["one", "two", "three"]
    assert recs[0].description == "first record"
    assert C.ref_to_ascii(recs[0].codes) == b"ACGTac"
    assert C.ref_to_ascii(recs[1].codes) == b"NNNNACGT"
    assert recs[2].length == 1


def test_fasta_long_record_crosses_blocks(tmp_path, monkeypatch):
    import gpxiltools.fasta as F

    monkeypatch.setattr(F, "_BLOCK", 7)  # force many tiny blocks
    seq = "ACGTN" * 50
    p = tmp_path / "y.fa"
    p.write_text(">a\n" + "\n".join(seq[i : i + 13] for i in range(0, len(seq), 13)) + "\n>b\nTT\n")
    a, b = F.read_fasta(p)
    assert C.ref_to_ascii(a.codes).decode() == seq
    assert C.ref_to_ascii(b.codes) == b"TT"


def test_digests_match_refget_example():
    d = sequence_digests(encode_sequence(b"acgt"))
    assert d["refget"] == "SQ.aKF498dAxcJAqme6QYQ7EZ07-fiw8Kw2"  # GA4GH refget spec example for ACGT
    assert d["md5"] == "f1f8f4bf413b16ad135722aa4591043e"
