"""Turning gpxil planes back into sequences and VCF records."""

from __future__ import annotations

import io
from typing import Iterator

import numpy as np

from . import codes as C
from .container import Contig

_SUB_ASCII = np.full(16, ord("N"), dtype=np.uint8)
for _b, _ch in zip(C.HAP_SUB_CODES, b"ACGT"):
    _SUB_ASCII[_b] = _ch
_SUBS = np.array(C.HAP_SUB_CODES, dtype=np.uint8)


def _hap_states(contig: Contig, h: int, start: int, end: int) -> np.ndarray:
    if not contig.has("gt"):
        return np.zeros(end - start, dtype=np.uint8)
    gt = contig.read("gt", start, end)
    return C.hap1(gt) if h == 1 else C.hap2(gt)


def _insertions(contig: Contig, h: int, start: int, end: int) -> dict[int, str]:
    out = {}
    for e in contig.sidecar():
        if e.kind == "INS" and e.hap == h and start <= e.offset < end:
            out[e.offset] = e.data
    return out


def haplotype_sequence(contig: Contig, h: int, start: int = 0, end: int | None = None, *, keep_case: bool = True) -> bytes:
    """ASCII sequence of haplotype ``h`` (1 or 2) over reference interval [start, end).

    UNKNOWN and COMPLEX positions become N; DEL and ABSENT positions are
    dropped; insertions anchored inside the interval follow their anchor.
    With ``keep_case`` (default) the soft-mask case of the reference carries
    over: a substituted base is lowercase where the reference base was, and
    an insertion takes the case of its anchor base (the same convention as
    ``bcftools consensus``). Otherwise everything is uppercase.
    """
    if h not in (1, 2):
        raise ValueError("haplotype must be 1 or 2")
    end = contig.length if end is None else end
    ref = contig.read("ref", start, end)
    st = _hap_states(contig, h, start, end)
    masked = (ref & C.SOFTMASK) != 0 if keep_case else np.zeros(ref.shape, dtype=bool)
    chars = C.REF_TO_ASCII[ref if keep_case else np.bitwise_and(ref, 0x0F)].copy()
    sub = np.isin(st, _SUBS)
    chars[sub] = _SUB_ASCII[st[sub]]
    chars[(st == C.HAP_UNKNOWN) | (st == C.HAP_COMPLEX)] = ord("N")
    lower = (sub | (st == C.HAP_UNKNOWN) | (st == C.HAP_COMPLEX)) & masked
    chars[lower] |= 0x20  # ASCII lowercase
    keep = (st != C.HAP_DEL) & (st != C.HAP_ABSENT)
    ins = _insertions(contig, h, start, end) if contig.has("gt") else {}
    seq = chars[keep]
    if not ins:
        return seq.tobytes()
    kept_through = np.cumsum(keep)  # number of kept positions in [start, i]
    pieces, prev = [], 0
    for anchor in sorted(ins):
        k = int(kept_through[anchor - start])
        pieces.append(seq[prev:k].tobytes())
        s = ins[anchor]
        pieces.append((s.lower() if masked[anchor - start] else s.upper()).encode("ascii"))
        prev = k
    pieces.append(seq[prev:].tobytes())
    return b"".join(pieces)


def genotype_iupac_sequence(contig: Contig, start: int = 0, end: int | None = None) -> bytes:
    """Per-reference-position consensus as IUPAC codes (het A/G -> R).

    Indels are not applied: a base deleted on one haplotype shows the other
    haplotype's base, and a base deleted on both becomes '-'.
    """
    end = contig.length if end is None else end
    ref = contig.read("ref", start, end)
    gt = contig.read("gt", start, end) if contig.has("gt") else np.zeros_like(ref)
    code = C.genotype_iupac(ref, gt)
    out = C.REF_TO_ASCII[code].copy()
    same = gt == 0
    out[same] = C.REF_TO_ASCII[ref[same]]  # keep soft-mask case where nothing changed
    return out.tobytes()


# ------------------------------------------------------------------ VCF export

_VCF_HEADER = """##fileformat=VCFv4.3
##source={source}
##reference={reference}
{contigs}##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=gpxil_filtered,Description="An input record covering this site did not pass its filters">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality (capped at 254 in gpxil)">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth (capped at 254 in gpxil)">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set (gpxil ordinal)">
##INFO=<ID=GPXIL_COMPLEX,Number=.,Type=String,Description="Sidecar data for alleles gpxil could not express per base">
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of a no-call block">
##ALT=<ID=*,Description="Placeholder allele of a no-call block">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}
"""


def event_windows(h1: np.ndarray, h2: np.ndarray, ins_any: np.ndarray) -> list[tuple[int, int]]:
    """Reference windows ``[start, end)`` that need one VCF record each.

    Substitutions are single-base windows; a deletion run takes the anchor
    base before it; an insertion takes its anchor base. Windows that share
    a base are merged, so records never overlap. Adjacent substitutions stay
    separate records (merging them would claim they are on one haplotype).
    """
    n = h1.shape[0]
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    special = (C.HAP_COMPLEX,)
    sub = np.isin(h1, _SUBS) | np.isin(h2, _SUBS) | np.isin(h1, special) | np.isin(h2, special)
    pos = np.flatnonzero(sub)
    starts.append(pos)
    ends.append(pos + 1)
    dele = (h1 == C.HAP_DEL) | (h2 == C.HAP_DEL)
    if dele.any():
        d = np.diff(np.concatenate(([0], dele.view(np.int8), [0])))
        rs, re_ = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        starts.append(np.maximum(rs - 1, 0))
        ends.append(re_)
    ipos = np.flatnonzero(ins_any)
    starts.append(ipos)
    ends.append(ipos + 1)
    s = np.concatenate(starts)
    e = np.concatenate(ends)
    if s.size == 0:
        return []
    order = np.argsort(s, kind="stable")
    s, e = s[order], e[order]
    out = []
    cs, ce = int(s[0]), int(e[0])
    for a, b in zip(s[1:].tolist(), e[1:].tolist()):
        if a < ce:  # shares a base with the current window
            ce = max(ce, b)
        else:
            out.append((cs, ce))
            cs, ce = a, b
    out.append((cs, ce))
    return [(a, min(b, n)) for a, b in out]


def iter_vcf_records(contig: Contig, chrom: str | None = None, *, nocall_blocks: bool = True) -> Iterator[str]:
    """Yield VCF body lines (no newline) describing the sample on this contig.

    One record per event window (see :func:`event_windows`): REF is the
    reference over the window and each haplotype contributes its sequence
    over the same window as an allele. Records are equivalent to the input
    calls in haplotype content, not a copy of the input records.

    With ``nocall_blocks`` (the default), runs of UNKNOWN outside event
    windows are written as gVCF-style blocks (``ALT=<*>``, ``INFO/END``,
    ``GT=./.``) so that re-encoding the VCF restores them.
    """
    import heapq

    chrom = chrom or contig.name
    if not contig.has("gt"):
        return
    events = _iter_event_records(contig, chrom)
    if not nocall_blocks:
        yield from (line for _, line in events)
        return
    blocks = _iter_nocall_blocks(contig, chrom)
    for _, line in heapq.merge(events, blocks, key=lambda t: t[0]):
        yield line


def _iter_nocall_blocks(contig: Contig, chrom: str) -> Iterator[tuple[int, str]]:
    gt = contig.read("gt")
    flags = contig.read("flags") if contig.has("flags") else np.zeros_like(gt)
    h1, h2 = C.hap1(gt), C.hap2(gt)
    ins_any = (flags & (C.FLAG_INS1 | C.FLAG_INS2)) != 0
    covered = np.zeros(gt.shape[0], dtype=bool)
    for a, b in event_windows(h1, h2, ins_any):
        covered[a:b] = True
    u1 = (h1 == C.HAP_UNKNOWN) & ~covered
    u2 = (h2 == C.HAP_UNKNOWN) & ~covered
    absent2 = h2 == C.HAP_ABSENT
    any_u = (u1 | u2).astype(np.int16)
    code = (
        u1.astype(np.int16)
        | (u2.astype(np.int16) << 1)
        | (absent2.astype(np.int16) << 2)
        | (((flags & C.FLAG_FILTERED) != 0).astype(np.int16) << 3)
        | (((flags & C.FLAG_PHASED) != 0).astype(np.int16) << 4)
    ) * any_u
    if not code.any():
        return
    change = np.flatnonzero(np.diff(np.concatenate(([0], code, [0]))) != 0)
    ref = contig.read("ref")
    ref_ascii = C.REF_TO_ASCII[np.bitwise_and(ref, 0x0F)]
    for a, b in zip(change[:-1].tolist(), change[1:].tolist()):
        c = int(code[a])
        if c == 0:
            continue
        haploid = bool(c & 4)
        g1 = "." if c & 1 else "0"
        g2 = "." if c & 2 else "0"
        sep = "|" if c & 16 else "/"
        gt_str = g1 if haploid else f"{g1}{sep}{g2}"
        filt = "gpxil_filtered" if c & 8 else "."
        yield a, "\t".join([chrom, str(a + 1), ".", chr(ref_ascii[a]), "<*>", ".", filt, f"END={b}", "GT", gt_str])


def _iter_event_records(contig: Contig, chrom: str) -> Iterator[tuple[int, str]]:
    ref = contig.read("ref")
    gt = contig.read("gt")
    flags = contig.read("flags") if contig.has("flags") else np.zeros_like(gt)
    gq = contig.read("gq") if contig.has("gq") else None
    dp = contig.read("dp") if contig.has("dp") else None
    ps = contig.read("ps") if contig.has("ps") else None
    h = [C.hap1(gt), C.hap2(gt)]
    ins: list[dict[int, str]] = [{}, {}]
    complex_data: dict[tuple[int, int], str] = {}
    for ent in contig.sidecar():
        if ent.kind == "INS":
            ins[ent.hap - 1][ent.offset] = ent.data
        elif ent.kind == "COMPLEX":
            complex_data[(ent.offset, ent.hap)] = ent.data
    ins_any = (flags & (C.FLAG_INS1 | C.FLAG_INS2)) != 0
    ref_ascii = C.REF_TO_ASCII[np.bitwise_and(ref, 0x0F)]
    n = contig.length
    for a0, b0 in event_windows(h[0], h[1], ins_any):
        if a0 == 0 and ((h[0][0] == C.HAP_DEL) or (h[1][0] == C.HAP_DEL)) and b0 < n:
            b0 += 1  # deletion at the contig start: pad with the base after (VCF rule)
        refseq = ref_ascii[a0:b0].tobytes().decode()
        alleles: list[str | None] = []
        for k in (0, 1):
            states = h[k][a0:b0]
            if (states == C.HAP_ABSENT).all():
                continue  # haploid here
            has_ins = any(p in ins[k] for p in range(a0, b0))
            if ((states == C.HAP_UNKNOWN).all() and not has_ins) or (states == C.HAP_COMPLEX).any():
                alleles.append(None)
                continue
            parts = []
            for i, p in enumerate(range(a0, b0)):
                st = int(states[i])
                if st == C.HAP_REF:
                    parts.append(refseq[i])
                elif st in C.HAP_SUB_CODES:
                    parts.append(chr(_SUB_ASCII[st]))
                elif st == C.HAP_UNKNOWN:
                    parts.append("N")  # unknown base inside an otherwise known allele
                if p in ins[k]:
                    parts.append(ins[k][p])
            alleles.append("".join(parts))
        if not alleles:
            continue
        alts: list[str] = []
        gt_idx = []
        for allele in alleles:
            if allele is None:
                gt_idx.append(".")
            elif allele == refseq:
                gt_idx.append("0")
            else:
                if not allele:
                    allele = "<DEL>"  # nothing left to anchor on
                if allele not in alts:
                    alts.append(allele)
                gt_idx.append(str(alts.index(allele) + 1))
        if not alts and not complex_data_in(complex_data, a0, b0):
            continue
        phased = bool(flags[a0:b0].max() & C.FLAG_PHASED) and len(gt_idx) == 2
        gt_str = ("|" if phased else "/").join(gt_idx)
        filt = "gpxil_filtered" if (flags[a0:b0] & C.FLAG_FILTERED).any() else "PASS"
        cdata = [complex_data[(p, k)] for p in range(a0, b0) for k in (1, 2) if (p, k) in complex_data]
        info = "GPXIL_COMPLEX=" + ",".join(d.replace(";", "|").replace(",", "/") for d in cdata) if cdata else "."
        fmt, vals = ["GT"], [gt_str]
        site = a0 if b0 - a0 == 1 else a0 + 1
        site = min(site, n - 1)
        if gq is not None and gq[site] != C.QUAL_MISSING:
            fmt.append("GQ")
            vals.append(str(int(gq[site])))
        if dp is not None and dp[site] != C.QUAL_MISSING:
            fmt.append("DP")
            vals.append(str(int(dp[site])))
        if ps is not None and phased and ps[site]:
            fmt.append("PS")
            vals.append(str(int(ps[site])))
        yield a0, "\t".join([chrom, str(a0 + 1), ".", refseq, ",".join(alts) or ".", ".", filt, info, ":".join(fmt), ":".join(vals)])


def complex_data_in(complex_data: dict, a0: int, b0: int) -> bool:
    return any((p, k) in complex_data for p in range(a0, b0) for k in (1, 2))


def write_vcf(gfile, out: io.TextIOBase, contigs: list[str] | None = None, *, nocall_blocks: bool = True) -> int:
    """Write the sample in a gpxil file as VCF. Returns the number of records."""
    meta = gfile.metadata
    names = contigs or gfile.contigs
    contig_lines = "".join(f"##contig=<ID={n},length={gfile[n].length}>\n" for n in names)
    sample = gfile.sample_id or "SAMPLE"
    ref = meta.get("reference", {}) or {}
    out.write(
        _VCF_HEADER.format(
            source=meta.get("generator", "gpxiltools"),
            reference=ref.get("name") or ref.get("source") or "unknown",
            contigs=contig_lines,
            sample=sample,
        )
    )
    count = 0
    for name in names:
        for line in iter_vcf_records(gfile[name], nocall_blocks=nocall_blocks):
            out.write(line + "\n")
            count += 1
    return count
