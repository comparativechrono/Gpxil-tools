"""Code tables for the standard gpxil planes (spec section 5).

Everything in a gpxil file is an integer code, never a colour. Colours are
applied at render time from a palette (see :mod:`gpxiltools.palettes`).

``ref`` plane (uint8)
    Low nibble: IUPAC nucleotide as a 4-bit set over {A, C, G, T}
    (A=1, C=2, G=4, T=8, so R = A|G = 5, N = 15). This is the same 4-bit
    code BAM uses for read bases. 0 means "no base" (padding or a gap).
    Bit 4 (0x10): soft-masked (lowercase in the source FASTA).
    Bits 5-7: reserved, MUST be 0.

``gt`` plane (uint8)
    High nibble: state of haplotype 1; low nibble: state of haplotype 2.
    States are relative to the reference base at the same position.

``flags`` plane (uint8)
    Bit field; see the ``FLAG_*`` constants.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- ref plane
BASE_A = 0x1
BASE_C = 0x2
BASE_G = 0x4
BASE_T = 0x8
BASE_N = 0xF
NO_BASE = 0x0
SOFTMASK = 0x10
REF_RESERVED_MASK = 0xE0

#: IUPAC symbol for each 4-bit set (index = code). Index 0 is the gap symbol.
IUPAC = "-ACMGRSVTWYHKDBN"

#: ASCII -> ref code. 0xFF marks a byte that is not a valid sequence symbol.
INVALID = 0xFF
ASCII_TO_REF = np.full(256, INVALID, dtype=np.uint8)
for _code, _sym in enumerate(IUPAC):
    ASCII_TO_REF[ord(_sym)] = _code
    if _sym.isalpha():
        ASCII_TO_REF[ord(_sym.lower())] = _code | SOFTMASK
ASCII_TO_REF[ord("U")] = BASE_T
ASCII_TO_REF[ord("u")] = BASE_T | SOFTMASK
ASCII_TO_REF[ord(".")] = NO_BASE

#: ref code -> ASCII (soft-masked codes decode to lowercase).
REF_TO_ASCII = np.full(256, ord("?"), dtype=np.uint8)
for _code, _sym in enumerate(IUPAC):
    REF_TO_ASCII[_code] = ord(_sym)
    REF_TO_ASCII[_code | SOFTMASK] = ord(_sym.lower()) if _sym.isalpha() else ord(_sym)

# ---------------------------------------------------------------- gt plane
#: Haplotype matches the reference base.
HAP_REF = 0x0
#: Single-bit states 0x1/0x2/0x4/0x8 = haplotype carries that (non-reference) base.
HAP_SUB_CODES = (BASE_A, BASE_C, BASE_G, BASE_T)
#: Haplotype not present here (haploid region, e.g. male chrX outside the PARs).
HAP_ABSENT = 0xA
#: Allele too complex for per-base states (symbolic SV etc.); details in the sidecar.
HAP_COMPLEX = 0xC
#: Reference base deleted on this haplotype.
HAP_DEL = 0xD
#: Allele unknown (no call, ".").
HAP_UNKNOWN = 0xF

HAP_STATE_NAMES = {
    HAP_REF: "REF",
    BASE_A: "A",
    BASE_C: "C",
    BASE_G: "G",
    BASE_T: "T",
    HAP_ABSENT: "ABSENT",
    HAP_COMPLEX: "COMPLEX",
    HAP_DEL: "DEL",
    HAP_UNKNOWN: "UNKNOWN",
}
VALID_HAP_STATES = frozenset(HAP_STATE_NAMES)

#: gt byte for "both haplotypes unknown" (used outside callable regions).
GT_UNKNOWN = (HAP_UNKNOWN << 4) | HAP_UNKNOWN

BASE_TO_SUB = {"A": BASE_A, "C": BASE_C, "G": BASE_G, "T": BASE_T, "U": BASE_T}

# ---------------------------------------------------------------- flags plane
FLAG_PHASED = 0x01  #: genotype here was phased ("|"); hap1/hap2 order is meaningful
FLAG_INS1 = 0x02  #: insertion follows this base on haplotype 1 (sequence in sidecar)
FLAG_INS2 = 0x04  #: insertion follows this base on haplotype 2 (sequence in sidecar)
FLAG_FILTERED = 0x08  #: the VCF record covering this base did not PASS
FLAG_VARIANT_SITE = 0x10  #: covered by a VCF record for this sample (any genotype)
FLAG_RESERVED_MASK = 0xE0

FLAG_NAMES = {
    FLAG_PHASED: "PHASED",
    FLAG_INS1: "INS1",
    FLAG_INS2: "INS2",
    FLAG_FILTERED: "FILTERED",
    FLAG_VARIANT_SITE: "VARIANT_SITE",
}

# ---------------------------------------------------------------- gq / dp planes
QUAL_MISSING = 255  #: gq/dp value for "not available"
QUAL_MAX = 254  #: gq/dp values are capped here


def hap1(gt: np.ndarray) -> np.ndarray:
    """Haplotype-1 state nibble of each gt byte."""
    return np.right_shift(gt, 4)


def hap2(gt: np.ndarray) -> np.ndarray:
    """Haplotype-2 state nibble of each gt byte."""
    return np.bitwise_and(gt, 0x0F)


def pack_gt(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    return (np.left_shift(h1.astype(np.uint8), 4) | h2.astype(np.uint8)).astype(np.uint8)


def ref_base_nibble(ref: np.ndarray) -> np.ndarray:
    """IUPAC 4-bit set of each ref byte (drops the soft-mask bit)."""
    return np.bitwise_and(ref, 0x0F)


_VALID_STATE_LUT = np.zeros(16, dtype=bool)
_VALID_STATE_LUT[list(VALID_HAP_STATES)] = True


def normalize_states(state: np.ndarray) -> np.ndarray:
    """Reserved haplotype states read as UNKNOWN (spec 10, reader rule 4)."""
    state = np.asarray(state, dtype=np.uint8)
    valid = _VALID_STATE_LUT[state]
    if valid.all():
        return state
    return np.where(valid, state, HAP_UNKNOWN).astype(np.uint8)


def haplotype_bases(ref: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Absolute base (IUPAC 4-bit set) carried by one haplotype at each position.

    Returns 0 where the haplotype has no base (DEL, ABSENT) and 15 (N) where
    the base is unknown (UNKNOWN, COMPLEX, or a reserved state).
    """
    state = normalize_states(state)
    base = ref_base_nibble(ref).copy()
    sub = np.isin(state, HAP_SUB_CODES)
    base[sub] = state[sub]
    base[(state == HAP_DEL) | (state == HAP_ABSENT)] = NO_BASE
    base[(state == HAP_UNKNOWN) | (state == HAP_COMPLEX)] = BASE_N
    return base


def genotype_iupac(ref: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Union of the bases on both haplotypes as one IUPAC 4-bit set.

    A heterozygous A/G site becomes R (A|G = 5): the "both channels on" idea,
    stored as a single code rather than as colour channels.
    """
    return haplotype_bases(ref, hap1(gt)) | haplotype_bases(ref, hap2(gt))


def dosage(ref: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Copies of A, C, G and T at each position, shape (n, 4), uint8.

    This is the 4-channel (one-hot x ploidy) array used by sequence models.
    Unknown or ambiguous bases contribute nothing.
    """
    out = np.zeros(ref.shape + (4,), dtype=np.uint8)
    for h in (hap1(gt), hap2(gt)):
        b = haplotype_bases(ref, h)
        for i, bit in enumerate(HAP_SUB_CODES):
            out[..., i] += b == bit
    return out


def ref_to_ascii(ref: np.ndarray) -> bytes:
    return REF_TO_ASCII[ref].tobytes()


def describe_gt_byte(value: int) -> str:
    """Human-readable form of one gt byte, e.g. ``0x04 -> 'REF|G'``."""
    a, b = value >> 4, value & 0xF
    return f"{HAP_STATE_NAMES.get(a, '?')}|{HAP_STATE_NAMES.get(b, '?')}"
