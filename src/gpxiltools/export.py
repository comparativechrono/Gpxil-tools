"""Turning gpxil planes back into sequences and VCF records.

Everything here works chunk by chunk, so memory stays bounded on the
longest chromosome.
"""

from __future__ import annotations

import heapq
import io
from collections.abc import Iterator
from typing import BinaryIO

import numpy as np

from . import codes as C
from .container import Contig

CHUNK = 1 << 23  #: bases processed at a time

_SUB_ASCII = np.full(16, ord("N"), dtype=np.uint8)
for _b, _ch in zip(C.HAP_SUB_CODES, b"ACGT"):
    _SUB_ASCII[_b] = _ch
_SUBS = np.array(C.HAP_SUB_CODES, dtype=np.uint8)


def _hap_states(contig: Contig, h: int, start: int, end: int) -> np.ndarray:
    if not contig.has("gt"):
        return np.zeros(end - start, dtype=np.uint8)
    gt = contig.read("gt", start, end)
    return C.normalize_states(C.hap1(gt) if h == 1 else C.hap2(gt))


def _insertions(contig: Contig, h: int, start: int, end: int) -> dict[int, str]:
    out = {}
    for e in contig.sidecar():
        if e.kind == "INS" and e.hap == h and start <= e.offset < end:
            out[e.offset] = e.data
    return out


def haplotype_sequence(contig: Contig, h: int, start: int = 0, end: int | None = None, *, keep_case: bool = True) -> bytes:
    """ASCII sequence of haplotype ``h`` (1 or 2) over reference interval [start, end).

    UNKNOWN, COMPLEX and reserved states become N; DEL and ABSENT positions
    are dropped; insertions anchored inside the interval follow their anchor.
    With ``keep_case`` (default) the soft-mask case of the reference carries
    over: a substituted base is lowercase where the reference base was, and
    an insertion takes the case of its anchor base. Otherwise everything is
    uppercase. For whole chromosomes prefer :func:`iter_haplotype` or
    :func:`write_fasta_stream`, which work in chunks.
    """
    if h not in (1, 2):
        raise ValueError("haplotype must be 1 or 2")
    end = contig.length if end is None else end
    ins = _insertions(contig, h, start, end) if contig.has("gt") else {}
    return _haplotype_chunk(contig, h, start, end, ins, keep_case)


def _haplotype_chunk(contig: Contig, h: int, start: int, end: int, ins: dict[int, str], keep_case: bool) -> bytes:
    ref = contig.read("ref", start, end)
    st = _hap_states(contig, h, start, end)
    masked = (ref & C.SOFTMASK) != 0 if keep_case else np.zeros(ref.shape, dtype=bool)
    chars = C.REF_TO_ASCII[ref if keep_case else np.bitwise_and(ref, 0x0F)].copy()
    sub = np.isin(st, _SUBS)
    chars[sub] = _SUB_ASCII[st[sub]]
    unknown = (st == C.HAP_UNKNOWN) | (st == C.HAP_COMPLEX)
    chars[unknown] = ord("N")
    chars[(sub | unknown) & masked] |= 0x20  # ASCII lowercase
    keep = (st != C.HAP_DEL) & (st != C.HAP_ABSENT)
    seq = chars[keep]
    anchors = sorted(a for a in ins if start <= a < end)
    if not anchors:
        return seq.tobytes()
    kept_through = np.cumsum(keep)  # number of kept positions in [start, i]
    pieces, prev = [], 0
    for anchor in anchors:
        k = int(kept_through[anchor - start])
        pieces.append(seq[prev:k].tobytes())
        s = ins[anchor]
        pieces.append((s.lower() if masked[anchor - start] else s.upper()).encode("ascii"))
        prev = k
    pieces.append(seq[prev:].tobytes())
    return b"".join(pieces)


def iter_haplotype(
    contig: Contig, h: int, start: int = 0, end: int | None = None, *, keep_case: bool = True, chunk: int = CHUNK
) -> Iterator[bytes]:
    """:func:`haplotype_sequence` in pieces of about ``chunk`` reference bases."""
    end = contig.length if end is None else end
    ins = _insertions(contig, h, start, end) if contig.has("gt") else {}
    for s in range(start, end, chunk):
        yield _haplotype_chunk(contig, h, s, min(end, s + chunk), ins, keep_case)


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


def iter_genotype_iupac(contig: Contig, start: int = 0, end: int | None = None, chunk: int = CHUNK) -> Iterator[bytes]:
    end = contig.length if end is None else end
    for s in range(start, end, chunk):
        yield genotype_iupac_sequence(contig, s, min(end, s + chunk))


def iter_reference(contig: Contig, start: int = 0, end: int | None = None, chunk: int = CHUNK) -> Iterator[bytes]:
    end = contig.length if end is None else end
    for s in range(start, end, chunk):
        yield C.ref_to_ascii(contig.read("ref", s, min(end, s + chunk)))


def write_fasta_stream(out: BinaryIO, name: str, pieces, *, line_width: int = 60) -> None:
    """Write one FASTA record from an iterator of sequence pieces, wrapping lines across pieces."""
    out.write(f">{name}\n".encode())
    carry = b""
    for piece in pieces:
        buf = carry + piece
        cut = len(buf) - len(buf) % line_width
        for i in range(0, cut, line_width):
            out.write(buf[i : i + line_width] + b"\n")
        carry = buf[cut:]
    if carry:
        out.write(carry + b"\n")


# ------------------------------------------------------------------ VCF export

_VCF_HEADER = """##fileformat=VCFv4.3
##source={source}
##reference={reference}
{contigs}##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=gpxil_filtered,Description="An input record covering this site did not pass its filters">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality (capped at 254 in gpxil)">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth (capped at 254 in gpxil)">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set">
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
    special = (C.HAP_COMPLEX,)
    sub = np.isin(h1, _SUBS) | np.isin(h2, _SUBS) | np.isin(h1, special) | np.isin(h2, special)
    pos = np.flatnonzero(sub)
    starts = [pos]
    ends = [pos + 1]
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


class _Planes:
    """The planes of one chunk ``[lo, hi)`` of a contig."""

    def __init__(self, contig: Contig, lo: int, hi: int):
        self.lo, self.hi = lo, hi
        self.ref = contig.read("ref", lo, hi)
        gt = contig.read("gt", lo, hi)
        self.h = [C.normalize_states(C.hap1(gt)), C.normalize_states(C.hap2(gt))]
        self.flags = contig.read("flags", lo, hi) if contig.has("flags") else np.zeros(hi - lo, dtype=np.uint8)
        self.gq = contig.read("gq", lo, hi) if contig.has("gq") else None
        self.dp = contig.read("dp", lo, hi) if contig.has("dp") else None
        self.ps = contig.read("ps", lo, hi) if contig.has("ps") else None
        self.ins_any = (self.flags & (C.FLAG_INS1 | C.FLAG_INS2)) != 0


def _chunked_windows(contig: Contig, chunk: int) -> Iterator[tuple[_Planes, list[tuple[int, int]]]]:
    """Event windows chunk by chunk, each window wholly inside one loaded chunk.

    A chunk starts one base early (the anchor of a deletion at its first
    base) and grows while its last window reaches its last base, so no
    window is ever cut.
    """
    n = contig.length
    start = 0
    while start < n:
        end = min(n, start + chunk)
        while True:
            lo = max(0, start - 1)
            p = _Planes(contig, lo, end)
            wins = event_windows(p.h[0], p.h[1], p.ins_any)
            if end < n and wins and wins[-1][1] + lo >= end:
                end = min(n, end + chunk)
                continue
            break
        yield p, [(a + lo, b + lo) for a, b in wins if a + lo >= start - 1]
        start = end


def _record(contig: Contig, chrom: str, p: _Planes, a0: int, b0: int, ins: list[dict[int, str]], complex_data: dict) -> str | None:
    n = contig.length
    lo = p.lo
    if a0 == 0 and ((p.h[0][0] == C.HAP_DEL) or (p.h[1][0] == C.HAP_DEL)) and b0 < p.hi:
        b0 += 1  # deletion at the contig start: pad with the base after (VCF rule)
    ref_ascii = C.REF_TO_ASCII[np.bitwise_and(p.ref[a0 - lo : b0 - lo], 0x0F)]
    refseq = ref_ascii.tobytes().decode()
    alleles: list[str | None] = []
    for k in (0, 1):
        states = p.h[k][a0 - lo : b0 - lo]
        if (states == C.HAP_ABSENT).all():
            continue  # haploid here
        has_ins = any(q in ins[k] for q in range(a0, b0))
        if ((states == C.HAP_UNKNOWN).all() and not has_ins) or (states == C.HAP_COMPLEX).any():
            alleles.append(None)
            continue
        parts = []
        for i, q in enumerate(range(a0, b0)):
            st = int(states[i])
            if st == C.HAP_REF:
                parts.append(refseq[i])
            elif st in C.HAP_SUB_CODES:
                parts.append(chr(_SUB_ASCII[st]))
            elif st == C.HAP_UNKNOWN:
                parts.append("N")  # unknown base inside an otherwise known allele
            if q in ins[k]:
                parts.append(ins[k][q])
        alleles.append("".join(parts))
    if not alleles:
        return None
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
    cdata = [complex_data[(q, k)] for q in range(a0, b0) for k in (1, 2) if (q, k) in complex_data]
    if not alts and not cdata:
        return None
    fl = p.flags[a0 - lo : b0 - lo]
    called = (fl & C.FLAG_VARIANT_SITE) != 0
    look = fl[called] if called.any() else fl
    phased = len(gt_idx) == 2 and bool(((look & C.FLAG_PHASED) != 0).all())
    gt_str = ("|" if phased else "/").join(gt_idx)
    filt = "gpxil_filtered" if (fl & C.FLAG_FILTERED).any() else "PASS"
    info = "GPXIL_COMPLEX=" + ",".join(d.replace(";", "|").replace(",", "/") for d in cdata) if cdata else "."
    fmt, vals = ["GT"], [gt_str]
    site = min((a0 if b0 - a0 == 1 else a0 + 1), n - 1) - lo
    for key, plane, cond in (("GQ", p.gq, True), ("DP", p.dp, True), ("PS", p.ps, phased)):
        if plane is None or not cond:
            continue
        v = int(plane[site])
        if (key == "PS" and v) or (key != "PS" and v != C.QUAL_MISSING):
            fmt.append(key)
            vals.append(str(v))
    return "\t".join([chrom, str(a0 + 1), ".", refseq, ",".join(alts) or ".", ".", filt, info, ":".join(fmt), ":".join(vals)])


def _nocall_blocks(contig: Contig, chrom: str, windows: np.ndarray, chunk: int) -> Iterator[tuple[int, str]]:
    """Runs of UNKNOWN outside event windows, as gVCF-style blocks."""
    n = contig.length
    ws, we = windows[:, 0], windows[:, 1]
    pending = None  # [code, start, ref base] of a run that may continue into the next chunk

    def emit(code, a, b, base):
        haploid = bool(code & 4)
        g1 = "." if code & 1 else "0"
        g2 = "." if code & 2 else "0"
        sep = "|" if code & 16 else "/"
        gt_str = g1 if haploid else f"{g1}{sep}{g2}"
        filt = "gpxil_filtered" if code & 8 else "."
        return a, "\t".join([chrom, str(a + 1), ".", base, "<*>", ".", filt, f"END={b}", "GT", gt_str])

    for cs in range(0, n, chunk):
        ce = min(n, cs + chunk)
        ref_ascii = C.REF_TO_ASCII[np.bitwise_and(contig.read("ref", cs, ce), 0x0F)]
        gt = contig.read("gt", cs, ce)
        flags = contig.read("flags", cs, ce) if contig.has("flags") else np.zeros(ce - cs, dtype=np.uint8)
        h1, h2 = C.normalize_states(C.hap1(gt)), C.normalize_states(C.hap2(gt))
        covered = np.zeros(ce - cs, dtype=bool)
        i0 = int(np.searchsorted(we, cs, side="right"))
        i1 = int(np.searchsorted(ws, ce, side="left"))
        for a, b in zip(ws[i0:i1].tolist(), we[i0:i1].tolist()):
            covered[max(a, cs) - cs : min(b, ce) - cs] = True
        u1 = (h1 == C.HAP_UNKNOWN) & ~covered
        u2 = (h2 == C.HAP_UNKNOWN) & ~covered
        any_u = u1 | u2
        code = (
            u1.astype(np.int16)
            | (u2.astype(np.int16) << 1)
            | ((h2 == C.HAP_ABSENT).astype(np.int16) << 2)
            | (((flags & C.FLAG_FILTERED) != 0).astype(np.int16) << 3)
            | (((flags & C.FLAG_PHASED) != 0).astype(np.int16) << 4)
        ) * any_u
        change = np.flatnonzero(np.diff(code)) + 1
        bounds = np.concatenate(([0], change, [ce - cs]))
        for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
            c = int(code[a])
            ga, gb = a + cs, b + cs
            base = chr(ref_ascii[a])
            if pending is not None:
                if c == pending[0] and ga == cs:
                    ga, base = pending[1], pending[2]  # the run continues from the last chunk
                else:
                    yield emit(pending[0], pending[1], cs, pending[2])
                pending = None
            if c == 0:
                continue
            if gb == ce and ce < n:
                pending = [c, ga, base]  # may continue in the next chunk
            else:
                yield emit(c, ga, gb, base)
    if pending is not None:
        yield emit(pending[0], pending[1], n, pending[2])


def iter_vcf_records(contig: Contig, chrom: str | None = None, *, nocall_blocks: bool = True, chunk: int = CHUNK) -> Iterator[str]:
    """Yield VCF body lines (no newline) describing the sample on this contig.

    One record per event window (see :func:`event_windows`): REF is the
    reference over the window and each haplotype contributes its sequence
    over the same window as an allele. A record is phased only if every
    called base in it was phased. Records are equivalent to the input calls
    in haplotype content, not a copy of the input records.

    With ``nocall_blocks`` (the default), runs of UNKNOWN outside event
    windows are written as gVCF-style blocks (``ALT=<*>``, ``INFO/END``,
    ``GT=./.``) so that re-encoding the VCF restores them.
    """
    chrom = chrom or contig.name
    if not contig.has("gt"):
        return
    ins: list[dict[int, str]] = [{}, {}]
    complex_data: dict[tuple[int, int], str] = {}
    for ent in contig.sidecar():
        if ent.kind == "INS":
            ins[ent.hap - 1][ent.offset] = ent.data
        elif ent.kind == "COMPLEX":
            complex_data[(ent.offset, ent.hap)] = ent.data
    events: list[tuple[int, str]] = []
    windows: list[tuple[int, int]] = []
    for p, wins in _chunked_windows(contig, chunk):
        for a0, b0 in wins:
            windows.append((a0, b0))
            line = _record(contig, chrom, p, a0, b0, ins, complex_data)
            if line is not None:
                events.append((a0, line))
        del p
    if not nocall_blocks:
        for _, line in events:
            yield line
        return
    warr = np.array(windows, dtype=np.int64).reshape(-1, 2)
    for _, line in heapq.merge(events, _nocall_blocks(contig, chrom, warr, chunk), key=lambda t: t[0]):
        yield line


def write_vcf(gfile, out: io.TextIOBase, contigs: list[str] | None = None, *, nocall_blocks: bool = True) -> int:
    """Write the sample in a gpxil file as VCF. Returns the number of records."""
    meta = gfile.metadata
    names = list(dict.fromkeys(gfile[n].name for n in (contigs or gfile.contigs)))  # canonical names, also for aliases
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
