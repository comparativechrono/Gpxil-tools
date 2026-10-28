"""VCF reading and conversion of one sample's genotypes into gpxil planes.

The conversion turns each VCF record into per-base haplotype states (spec
section 6). It preserves the content of both haplotypes: every substitution,
deletion and insertion, the phase, haploid calls and no-calls. It does not
preserve VCF record syntax (how alleles were split into records); exporting
back to VCF gives an equivalent, normalised call set.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from . import codes as C
from .fasta import open_maybe_gzip

# Working arrays: one byte per base and haplotype.
_STATE_MASK = 0x0F  # bits 0-3: state (spec 5.2)
_ASSERTED = 0x10  # bit 4: a call has asserted this base (spec 6.4)
_INS = 0x20  # bit 5: an insertion follows this base
# bits 6-7: the value of an annotation plane at this base came from a call that
# wrote a strong state here (spec 6.5): hap-1 array 0x40 gq, 0x80 dp; hap-2 array 0x40 ps
_Q_GQ, _Q_DP, _Q_PS = (0, 0x40), (0, 0x80), (1, 0x40)
_QBITS = 0xC0

_WEAK = (C.HAP_REF, C.HAP_UNKNOWN, C.HAP_ABSENT)
_REFBLOCK_ALTS = {"<NON_REF>", "<*>", "<X>"}
_ACGTN = re.compile(r"[^ACGTN]")


class VcfError(ValueError):
    pass


def is_symbolic(alt: str) -> bool:
    """Symbolic (``<DEL>``), breakend (``G]chr2:5]``) or single-breakend (``G.``) allele."""
    return alt.startswith("<") or "[" in alt or "]" in alt or (alt != "*" and "." in alt)


# ----------------------------------------------------------------- reading


@dataclass
class VcfHeader:
    samples: list[str]
    contigs: dict[str, int | None]
    formats: set[str]
    lines: list[str]


def read_header(path) -> VcfHeader:
    lines: list[str] = []
    samples: list[str] = []
    contigs: dict[str, int | None] = {}
    formats: set[str] = set()
    with open_maybe_gzip(path) as fh:
        for raw in fh:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("##"):
                lines.append(line)
                if line.startswith("##contig=<"):
                    m_id = re.search(r"[<,]ID=([^,>]+)", line)
                    m_len = re.search(r"[<,]length=([0-9]+)", line)
                    if m_id:
                        contigs[m_id.group(1)] = int(m_len.group(1)) if m_len else None
                elif line.startswith("##FORMAT=<"):
                    m_id = re.search(r"[<,]ID=([^,>]+)", line)
                    if m_id:
                        formats.add(m_id.group(1))
                continue
            if line.startswith("#CHROM"):
                lines.append(line)
                cols = line.split("\t")
                samples = cols[9:]
            break
    return VcfHeader(samples=samples, contigs=contigs, formats=formats, lines=lines)


@dataclass
class Call:
    """One VCF record reduced to what a single-sample gpxil plane needs."""

    pos: int  # 1-based POS as in the VCF
    ref: str
    alts: list[str]
    filter: str
    gt: str
    end: int | None = None  # INFO END (1-based, inclusive)
    svlen: list[int] | None = None  # INFO SVLEN, one value per ALT
    gq: int | None = None
    dp: int | None = None
    ps: str | None = None


def _info_values(info: str, key: str) -> list[str] | None:
    if info == "." or (key + "=") not in info:
        return None
    for item in info.split(";"):
        if item.startswith(key + "="):
            return item[len(key) + 1 :].split(",")
    return None


def _to_int(value: str | None) -> int | None:
    if value is None or value in (".", ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def iter_calls(lines: Iterable[str], sample_index: int) -> Iterator[tuple[str, Call]]:
    """Parse VCF body lines into ``(chrom, Call)`` for one sample."""
    col = 9 + sample_index
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        f = line.rstrip("\r\n").split("\t")
        if len(f) <= col:
            raise VcfError(f"record at {f[0]}:{f[1] if len(f) > 1 else '?'} has no column for sample {sample_index}")
        keys = f[8].split(":")
        vals = f[col].split(":")
        sample = dict(zip(keys, vals))
        gt = sample.get("GT", ".")
        alts = [] if f[4] == "." else f[4].split(",")
        info = f[7]
        sym = [is_symbolic(a) for a in alts]
        need_end = not alts or any(sym)
        end_v = _info_values(info, "END") if need_end else None
        svlen_v = _info_values(info, "SVLEN") if need_end else None
        yield (
            f[0],
            Call(
                pos=int(f[1]),
                ref=f[3].upper(),
                alts=[a if s else a.upper() for a, s in zip(alts, sym)],
                filter=f[6],
                gt=gt,
                end=_to_int(end_v[0]) if end_v else None,
                svlen=[v if v is not None else 0 for v in (_to_int(x) for x in svlen_v)] if svlen_v else None,
                gq=_to_int(sample.get("GQ")),
                dp=_to_int(sample.get("DP")),
                ps=sample.get("PS"),
            ),
        )


def iter_body_lines(path) -> Iterator[str]:
    with open_maybe_gzip(path) as fh:
        for raw in fh:
            if raw.startswith(b"#") or not raw.strip():
                continue
            yield raw.decode("utf-8", "replace")


@dataclass
class SplitResult:
    index: dict[str, tuple[str, int]]  # contig -> (temp path, records)
    format_keys: set[str]  # FORMAT keys used by any record


def split_by_contig(path, sample_index: int, workdir: str, buffer_bytes: int = 32 << 20, keep=None) -> SplitResult:
    """Stream a VCF once, writing each contig's calls for one sample to a temp file.

    This lets the encoder handle one chromosome at a time whatever the record
    order of the VCF. Lines are buffered in memory and appended to the
    per-contig files in batches, so only one file is open at a time however
    many contigs there are. The FORMAT keys of every record are collected on
    the way, so the choice of optional planes sees the whole file. With
    ``keep``, only the contigs it accepts (by name) are written.
    """
    buffers: dict[str, list[str]] = {}
    counts: Counter[str] = Counter()
    paths: dict[str, str] = {}
    formats_seen: set[str] = set()
    format_keys: set[str] = set()
    rejected: set[str] = set()
    buffered = 0
    col = 9 + sample_index

    def flush() -> None:
        nonlocal buffered
        for chrom, lines in buffers.items():
            if lines:
                with open(paths[chrom], "a", encoding="utf-8") as fh:
                    fh.writelines(lines)
                lines.clear()
        buffered = 0

    for line in iter_body_lines(path):
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) <= col:
            raise VcfError(f"record at {parts[0]}:{parts[1] if len(parts) > 1 else '?'} has no column for sample {sample_index}")
        chrom = parts[0]
        if len(parts) > 8 and parts[8] not in formats_seen:
            formats_seen.add(parts[8])
            format_keys.update(parts[8].split(":"))
        if keep is not None and chrom not in paths:
            if chrom in rejected:
                continue
            if not keep(chrom):
                rejected.add(chrom)
                continue
        if chrom not in paths:
            paths[chrom] = os.path.join(workdir, f"contig{len(paths):05d}.vcf")
            buffers[chrom] = []
        out = "\t".join(parts[:9] + [parts[col]]) + "\n"  # fixed columns plus the chosen sample
        buffers[chrom].append(out)
        buffered += len(out)
        counts[chrom] += 1
        if buffered >= buffer_bytes:
            flush()
    flush()
    return SplitResult({c: (paths[c], counts[c]) for c in paths}, format_keys)


# ----------------------------------------------------------------- alleles


def parse_gt(gt: str) -> tuple[list[str], bool]:
    """'0|1' -> (['0', '1'], True); '1/2' -> (['1', '2'], False); '1' -> (['1'], False).

    A missing genotype ('.' or empty) gives an empty list: the number of
    alleles is unknown. A leading phase indicator (VCF 4.4, e.g. '|0|1') is
    dropped.
    """
    g = gt.strip()
    if g in ("", "."):
        return [], False
    if g[0] in "|/":
        g = g[1:]
    if "|" in g and "/" not in g:
        return g.split("|"), True
    return g.replace("|", "/").split("/"), False


def allele_number(a: str) -> int | None:
    """The number a GT allele is (0 the REF allele, 1 the first ALT allele, ...), or None for a missing allele (``.``)
    and for anything not written in the digits 0-9 (spec 6.2: nothing is written for it)."""
    return int(a) if a.isascii() and a.isdigit() else None


def trim_alleles(ref: str, alt: str) -> tuple[int, str, str]:
    """Steps 1 and 2 of spec 6.2: the common suffix trimmed (but never all of the shorter allele, so that an indel
    keeps its anchor), then the common prefix. Returns ``(k, r, a)``: the length ``k`` of the prefix and the cores
    left of REF and ALT, which start at offset ``k`` of the record."""
    m, n = len(ref), len(alt)
    qmax = min(m, n) - (1 if m != n else 0)
    q = 0
    while q < qmax and ref[m - 1 - q] == alt[n - 1 - q]:
        q += 1
    p = 0
    while p < m - q and p < n - q and ref[p] == alt[p]:
        p += 1
    return p, ref[p : m - q], alt[p : n - q]


@dataclass
class AlleleEvents:
    """Per-base effect of one ALT allele relative to its record's REF span."""

    subs: list[tuple[int, int]] = field(default_factory=list)  # (offset, state)
    dels: list[tuple[int, int]] = field(default_factory=list)  # (offset, length)
    ins: list[tuple[int, str]] = field(default_factory=list)  # (anchor offset, sequence)


def allele_events(ref: str, alt: str) -> AlleleEvents:
    """Decompose REF->ALT into substitutions, a deletion and/or an insertion.

    The common suffix is trimmed first, then the common prefix, which places
    an indel as far left as the record allows (the VCF left-alignment
    convention). For indels the suffix trim stops before consuming the
    shorter allele, so the anchor base stays inside the record. The remaining
    core is aligned left to right: shared length as substitutions, then the
    surplus as an insertion (ALT longer) or a deletion (REF longer). Any
    alignment that yields the same haplotype is equally valid; fixing one
    makes encoders agree, and left placement matches how callers write
    companion records inside repeats.
    """
    p, rc, ac = trim_alleles(ref, alt)
    ev = AlleleEvents()
    k = min(len(rc), len(ac))
    for i in range(k):
        if rc[i] != ac[i]:
            ev.subs.append((p + i, C.BASE_TO_SUB.get(ac[i], C.HAP_UNKNOWN)))
    if len(ac) > k:
        # p + k >= 1 here: with ALT longer, the suffix trim leaves REF's first base
        ev.ins.append((p + k - 1, ac[k:]))
    elif len(rc) > k:
        ev.dels.append((p + k, len(rc) - k))
    return ev


# ----------------------------------------------------------------- planes


@dataclass
class EncodeStats:
    records: int = 0
    skipped_out_of_range: int = 0
    clipped_to_contig_end: int = 0
    ref_mismatches: int = 0
    conflicts: int = 0
    ploidy_mismatches: int = 0
    missing_genotypes: int = 0
    spanning_deletion_alleles: int = 0
    spanning_deletions_reassigned: int = 0
    overlapping_calls_reassigned: int = 0
    symbolic_alleles: int = 0
    nonacgt_inserted_bases: int = 0
    polyploid_skipped: int = 0
    bad_allele_index: int = 0
    ploidy: int | str = 2
    gt_counts: Counter = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        if len(self.examples) < 20:
            self.examples.append(msg)

    def as_dict(self) -> dict:
        keys = (
            "records",
            "skipped_out_of_range",
            "clipped_to_contig_end",
            "ref_mismatches",
            "conflicts",
            "ploidy_mismatches",
            "missing_genotypes",
            "spanning_deletion_alleles",
            "spanning_deletions_reassigned",
            "overlapping_calls_reassigned",
            "symbolic_alleles",
            "nonacgt_inserted_bases",
            "polyploid_skipped",
            "bad_allele_index",
        )
        return {k: getattr(self, k) for k in keys}


@dataclass
class SidecarEntry:
    offset: int  # 0-based position (pixel index) on the contig
    hap: int  # 1 or 2
    kind: str  # "INS" or "COMPLEX"
    data: str

    def to_tsv(self) -> str:
        return f"{self.offset}\t{self.hap}\t{self.kind}\t{self.data}"


def _strong(states: np.ndarray) -> np.ndarray:
    return (states != C.HAP_REF) & (states != C.HAP_UNKNOWN) & (states != C.HAP_ABSENT)


# One bit per base (bit i % 8 of byte i // 8): the bases ALT alleles occupy on a haplotype (spec 6.5)
def _bits(n: int) -> bytearray:
    return bytearray((n + 7) >> 3)


def _bits_set(bits: bytearray, a: int, b: int) -> None:
    """Set the bits of [a, b)."""
    i, j = a >> 3, (b - 1) >> 3
    lo, hi = (0xFF << (a & 7)) & 0xFF, 0xFF >> (7 - ((b - 1) & 7))
    if i == j:
        bits[i] |= lo & hi
    else:
        bits[i] |= lo
        bits[i + 1 : j] = b"\xff" * (j - i - 1)
        bits[j] |= hi


def _bits_any(bits: bytearray, a: int, b: int) -> bool:
    """Whether any bit of [a, b) is set."""
    i, j = a >> 3, (b - 1) >> 3
    lo, hi = (0xFF << (a & 7)) & 0xFF, 0xFF >> (7 - ((b - 1) & 7))
    if i == j:
        return bool(bits[i] & lo & hi)
    return bool(bits[i] & lo) or bool(bits[j] & hi) or bits.count(0, i + 1, j) != j - i - 1


def record_span(call: Call, n: int) -> tuple[int, int, bool] | None:
    """The bases a record covers on a contig of ``n`` bases (spec 6.2, 6.3): ``(start, end, clipped)``, 0-based with
    the end excluded, ``clipped`` if END took it past the contig's end (where it is cut). None for a record that is
    skipped: no REF allele, or outside the contig."""
    p0 = call.pos - 1
    if not call.ref or p0 < 0 or p0 >= n or p0 + len(call.ref) > n:
        return None
    alts = call.alts
    only_refblock = not alts or all(a in _REFBLOCK_ALTS for a in alts)
    symbolic = any(is_symbolic(a) for a in alts)
    span_end = p0 + max(len(call.ref), 1)
    if call.end is not None and (only_refblock or symbolic):
        span_end = max(span_end, call.end)
    elif symbolic and call.svlen:
        for i, a in enumerate(alts):
            if a.startswith("<DEL") and i < len(call.svlen) and call.svlen[i]:
                span_end = max(span_end, call.pos + abs(call.svlen[i]))
    return p0, min(span_end, n), span_end > n


def ps_key(ps: str) -> str:
    """A phase-set identifier as it is compared (spec 5.5): one written in the digits 0-9 is an integer, compared as
    one ('077' is 77, '00' is 0); any other is compared as it is written."""
    return str(int(ps)) if ps.isascii() and ps.isdigit() else ps


def phased_ps(call: Call, n: int) -> str | None:
    """The phase-set identifier a record gives its bases (spec 5.5, 6.5): that of a phased call of two alleles with a
    ``PS``, not skipped (section 6.2); else None."""
    if call.ps in (None, ".", "") or record_span(call, n) is None:
        return None
    alleles, phased = parse_gt(call.gt)
    return ps_key(call.ps) if phased and len(alleles) == 2 else None


def ps_numbering(keys: Iterable[str]) -> dict[str, int]:
    """The values of a contig's phase-set identifiers (spec 5.5), given those of its phased calls in file order
    (:func:`phased_ps`): an integer from 1 to 2^32-1 is itself, and the others take the largest values that none of
    those integers takes, downwards from 2^32-1, in order of first appearance. So a window's values are those of a
    contig of its whole parent, whichever calls it converts."""
    keys = list(dict.fromkeys(ps_key(k) for k in keys))
    out: dict[str, int] = {}
    for k in keys:
        if k.isascii() and k.isdigit() and 1 <= int(k) <= 0xFFFFFFFF:
            out[k] = int(k)
    taken = set(out.values())
    nxt = 0xFFFFFFFF
    for k in keys:
        if k not in out:
            while nxt in taken:
                nxt -= 1
            out[k] = nxt
            taken.add(nxt)
    return out


class PlaneBuilder:
    """Accumulates one sample's calls on one contig into gt/flags/... planes.

    ``ref`` is the contig's reference, or a stretch of it that starts at ``origin``, for a window (spec 3.1): then
    ``contig_length`` is the whole contig's, positions in calls are the contig's, and the planes are the stretch's.
    The calls given must lie inside the stretch (see :func:`window_calls`).
    """

    def __init__(
        self,
        ref: np.ndarray,
        *,
        callable_mask: np.ndarray | None = None,
        haploid_mask: np.ndarray | None = None,
        want_gq: bool = False,
        want_dp: bool = False,
        want_ps: bool = False,
        contig: str = "",
        origin: int = 0,
        contig_length: int | None = None,
        ps_values: dict[str, int] | None = None,
    ):
        n = ref.shape[0]
        self.ref = ref
        #: the values of the contig's phase-set identifiers (:func:`ps_numbering`), for a window; else worked out from the
        #: calls given, which are then all the contig's
        self.ps_values = ps_values
        self.contig = contig
        self.length = n
        self.origin = origin
        self.contig_length = origin + n if contig_length is None else contig_length
        self.w = [np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)]
        # initial states (spec 6.1): callable regions first, then haploid regions
        if callable_mask is not None:
            np.copyto(self.w[0], C.HAP_UNKNOWN, where=~callable_mask)
            np.copyto(self.w[1], C.HAP_UNKNOWN, where=~callable_mask)
        if haploid_mask is not None:
            np.copyto(self.w[1], C.HAP_ABSENT, where=haploid_mask)
        self.flags = np.zeros(n, dtype=np.uint8)
        self.gq = np.full(n, C.QUAL_MISSING, dtype=np.uint8) if want_gq else None
        self.dp = np.full(n, C.QUAL_MISSING, dtype=np.uint8) if want_dp else None
        # phase sets are kept as ordinals (1, 2, ... in order of first use) while
        # building and mapped to their final values in finish()
        self.ps_ord = np.zeros(n, dtype=np.uint16) if want_ps else None
        self.ps_keys: list[str] = []
        self._ps_index: dict[str, int] = {}
        self.ps_ids: dict[str, int] = {}
        self.ins: dict[tuple[int, int], str] = {}  # (anchor offset, hap) -> inserted bases
        self.complex: dict[tuple[int, int], str] = {}  # (offset, hap) -> COMPLEX sidecar data
        # what ALT alleles occupy on each haplotype, and which alleles they are (spec 6.5)
        self.occupied = [_bits(n), _bits(n)]
        self.placed: list[set[int]] = [set(), set()]
        self.stats = EncodeStats()

    # -- helpers ---------------------------------------------------------
    def _ref_ok(self, start: int, allele: str) -> bool:
        """Each REF base must be an IUPAC letter whose set meets the reference base's (spec 6.7).

        So N matches any base but not a gap, and '-', '.' or '*' never match.
        """
        seg = np.bitwise_and(self.ref[start : start + len(allele)], 0x0F)
        raw = np.frombuffer(allele.encode("ascii", "replace"), dtype=np.uint8)
        a = C.ASCII_TO_REF[raw]
        ok = (a != C.INVALID) & ((np.bitwise_and(a, 0x0F) & seg) != 0)
        return bool(ok.all())

    def _write_states(self, h: int, start: int, states: np.ndarray, where: np.ndarray | None = None) -> np.ndarray:
        """Write states for haplotype ``h`` with the precedence rules of spec 6.4.

        Strong states (bases, DEL, COMPLEX) always win; a different strong
        state already asserted is a conflict and the later call wins. Weak
        states (REF, UNKNOWN, ABSENT) only fill bases no call has asserted;
        ABSENT meeting a base asserted with a strong state is also a
        conflict (a haploid call against a diploid one). With ``where``,
        only those bases are touched. Returns the mask of strong states
        written.
        """
        w = self.w[h]
        end = start + states.shape[0]
        seg = w[start:end]
        asserted = (seg & _ASSERTED) != 0
        cur = seg & _STATE_MASK
        strong = _strong(states)
        if where is not None:
            strong &= where
        cur_strong = _strong(cur)
        conflict = asserted & cur_strong & ((strong & (cur != states)) | (states == C.HAP_ABSENT))
        if where is not None:
            conflict &= where
        if conflict.any():
            self.stats.conflicts += int(conflict.sum())
            self.stats.note(f"{self.contig}:{self.origin + start + int(np.flatnonzero(conflict)[0]) + 1} conflicting states on hap{h + 1}")
        if h == 1:
            mism = (cur == C.HAP_ABSENT) & ~asserted & (states != C.HAP_ABSENT)
            if where is not None:
                mism &= where
            if mism.any():
                self.stats.ploidy_mismatches += int(mism.sum())
        new = np.where(strong, states, np.where(asserted, cur, states)).astype(np.uint8) | _ASSERTED | (seg & (_INS | _QBITS))
        w[start:end] = new if where is None else np.where(where, new, seg)
        return strong

    def _write_scalar(self, h: int, pos: int, state: int) -> bool:
        w = self.w[h]
        v = int(w[pos])
        asserted = v & _ASSERTED
        cur = v & _STATE_MASK
        keep = v & (_INS | _QBITS)
        if h == 1 and cur == C.HAP_ABSENT and not asserted and state != C.HAP_ABSENT:
            self.stats.ploidy_mismatches += 1
        if state in _WEAK:
            if asserted:
                if state == C.HAP_ABSENT and cur not in _WEAK:
                    self.stats.conflicts += 1
                    self.stats.note(f"{self.contig}:{self.origin + pos + 1} haploid call meets a variant on hap2")
                return False
            w[pos] = state | _ASSERTED | keep
            return False
        if asserted and cur not in _WEAK and cur != state:
            self.stats.conflicts += 1
            self.stats.note(f"{self.contig}:{self.origin + pos + 1} conflicting states on hap{h + 1}")
        w[pos] = state | _ASSERTED | keep
        return True

    @staticmethod
    def _inserted(seq: str) -> tuple[str, int]:
        """Inserted bases as the sidecar keeps them (spec 7): uppercase, U as T, others as N; and how many were others."""
        seq = seq.upper().replace("U", "T")
        bad = len(_ACGTN.findall(seq))
        return (_ACGTN.sub("N", seq) if bad else seq), bad

    def _add_insertion(self, h: int, anchor: int, seq: str) -> None:
        seq, bad = self._inserted(seq)
        self.stats.nonacgt_inserted_bases += bad
        key = (anchor, h + 1)
        if key in self.ins and self.ins[key] != seq:
            self.stats.conflicts += 1
            self.stats.note(f"{self.contig}:{self.origin + anchor + 1} two insertions on hap{h + 1}")
        self.ins[key] = seq
        self.w[h][anchor] |= _INS

    def _ps_ordinal(self, ps: str) -> int:
        ps = ps_key(ps)  # ('077' and '77' are the same VCF integer)
        i = self._ps_index.get(ps)
        if i is None:
            self.ps_keys.append(ps)
            i = self._ps_index[ps] = len(self.ps_keys)
            if i > 0xFFFF and self.ps_ord.dtype == np.uint16:
                self.ps_ord = self.ps_ord.astype(np.uint32)
        return i

    # -- main entry ------------------------------------------------------
    def add(self, call: Call) -> None:
        st = self.stats
        st.records += 1
        text = call.ref + "".join(call.alts)
        if text != text.upper():  # (REF and sequence ALT alleles are read case-insensitively, spec 6.2: iter_calls does)
            call = dataclasses.replace(call, ref=call.ref.upper(), alts=[a if is_symbolic(a) else a.upper() for a in call.alts])
        n = self.contig_length
        span = record_span(call, n)
        if span is None:
            st.skipped_out_of_range += 1
            st.note(f"{self.contig}:{call.pos} " + (f"outside the contig (length {n})" if call.ref else "has no REF allele"))
            return
        p0, span_end, clipped = span
        if clipped:
            st.clipped_to_contig_end += 1
            st.note(f"{self.contig}:{call.pos} span clipped at the contig end")
        p0, span_end = p0 - self.origin, span_end - self.origin  # (the stretch's offsets)
        if p0 < 0 or span_end > self.length:
            raise ValueError(f"{self.contig}:{call.pos}: the record isn't inside the stretch being encoded")
        if not self._ref_ok(p0, call.ref):
            st.ref_mismatches += 1
            st.note(f"{self.contig}:{call.pos} REF {call.ref} does not match the reference")

        alleles, phased = parse_gt(call.gt)
        st.gt_counts[call.gt] += 1
        if len(alleles) > 2:
            st.polyploid_skipped += 1
            st.note(f"{self.contig}:{call.pos} ploidy {len(alleles)} not supported by gpxil")
            return
        span = span_end - p0
        strong = np.zeros(span, dtype=bool)
        if not alleles:
            # missing genotype: both haplotypes unknown, but an absent haplotype 2 is
            # left as it is (and not asserted: the call says nothing about ploidy)
            st.missing_genotypes += 1
            unknown = np.full(span, C.HAP_UNKNOWN, dtype=np.uint8)
            self._write_states(0, p0, unknown)
            self._write_states(1, p0, unknown, where=(self.w[1][p0:span_end] & _STATE_MASK) != C.HAP_ABSENT)
        else:
            if len(alleles) == 2 and not phased:
                alleles = self._place_spanning_deletion(alleles, call, p0, span_end)
                alleles = self._place_overlapping(alleles, call, p0, span_end)
            for h, a in enumerate(alleles):
                s = self._apply_allele(h, a, call, p0, len(call.ref), span_end)
                if s is not None:
                    strong |= s
                c = None if a == "0" or a == "." else self._occupies(a, call, p0)
                if c is not None:
                    _bits_set(self.occupied[h], c[0], c[1])
                    self.placed[h].add(c[2])
            if len(alleles) == 1:  # haploid call
                if span == 1:
                    self._write_scalar(1, p0, C.HAP_ABSENT)
                else:
                    self._write_states(1, p0, np.full(span, C.HAP_ABSENT, dtype=np.uint8))

        f = C.FLAG_VARIANT_SITE
        if phased and len(alleles) == 2:
            f |= C.FLAG_PHASED
        if call.filter not in ("PASS", "."):
            f |= C.FLAG_FILTERED
        self.flags[p0:span_end] |= f
        self._annotate(call, p0, span_end, strong, phased and len(alleles) == 2)

    def _annotate(self, call: Call, p0: int, span_end: int, strong: np.ndarray, phased: bool) -> None:
        """gq, dp and ps over the call's span (spec 6.5).

        For each field separately, a call that wrote a strong state to a base
        (on either haplotype) takes precedence there over calls that did not;
        among equals the later call wins.
        """
        jobs = []
        if self.gq is not None and call.gq is not None:
            jobs.append((self.gq, _Q_GQ, lambda: min(max(call.gq, 0), C.QUAL_MAX)))
        if self.dp is not None and call.dp is not None:
            jobs.append((self.dp, _Q_DP, lambda: min(max(call.dp, 0), C.QUAL_MAX)))
        if self.ps_ord is not None and phased and call.ps not in (None, ".", ""):
            ordinal = self._ps_ordinal(call.ps)  # numbered at first appearance (spec 5.5)
            jobs.append((None, _Q_PS, lambda: ordinal))
        any_strong = bool(strong.any())
        for arr, (h, bit), value in jobs:
            w = self.w[h][p0:span_end]
            write = strong | ((w & bit) == 0)
            if write.any():
                v = value()  # for ps this may widen self.ps_ord, so look the array up afterwards
                target = self.ps_ord if arr is None else arr
                target[p0:span_end][write] = v
            if any_strong:
                w[strong] |= bit

    def _place_spanning_deletion(self, alleles: list[str], call: Call, p0: int, span_end: int) -> list[str]:
        """Unphased genotype with a '*' allele: put '*' on the haplotype with the deletion.

        In an unphased genotype the allele order means nothing, so when exactly
        one haplotype already carries an asserted deletion over this record,
        '*' (the overlapping deletion) belongs to that haplotype (spec 6.5).
        """
        if "*" not in call.alts:
            return alleles
        star = []
        for i, a in enumerate(alleles):
            idx = self._alt_index(a, call)
            if idx is not None and call.alts[idx - 1] == "*":
                star.append(i)
        if len(star) != 1:
            return alleles
        deleted = []
        for h in (0, 1):
            seg = self.w[h][p0:span_end]
            if (((seg & _STATE_MASK) == C.HAP_DEL) & ((seg & _ASSERTED) != 0)).any():
                deleted.append(h)
        if len(deleted) == 1 and deleted[0] != star[0]:
            self.stats.spanning_deletions_reassigned += 1
            return [alleles[1], alleles[0]]
        return alleles

    def _alt_index(self, a: str, call: Call) -> int | None:
        """The ALT allele an allele number names (1-based), or None for REF, a missing allele or a number that isn't one."""
        idx = allele_number(a)
        return idx if idx is not None and 0 < idx <= len(call.alts) else None

    @staticmethod
    def _symbolic_end(alt: str, alt_index: int, call: Call) -> int | None:
        """END of a symbolic allele: INFO/END, or for a deletion without it POS + |SVLEN| (spec 6.3)."""
        end = call.end
        if end is None and alt.startswith("<DEL") and call.svlen and alt_index < len(call.svlen) and call.svlen[alt_index]:
            end = call.pos + abs(call.svlen[alt_index])
        return end

    def _occupies(self, a: str, call: Call, p0: int) -> tuple[int, int, int] | None:
        """The offsets ``[lo, hi)`` that allele ``a`` of ``call`` occupies, and a number standing for the allele (spec
        6.5): a sequence allele occupies its cores (spec 6.2), and the anchor before them if one is empty (a plain
        insertion or deletion); ``<DEL>``, POS to END; another symbolic allele, POS. None for an allele that occupies
        nothing: the REF allele, a missing allele, ``*``, a placeholder, a number with no ALT allele, ALT equal to REF."""
        idx = self._alt_index(a, call)
        if idx is None:
            return None
        alt = call.alts[idx - 1]
        if alt == "*" or alt in _REFBLOCK_ALTS:
            return None
        if is_symbolic(alt):
            end = self._symbolic_end(alt, idx - 1, call)
            lo, hi = (
                p0,
                (min(end, self.contig_length) - self.origin if alt.startswith("<DEL") and end is not None and end > call.pos else p0 + 1),
            )
            what = f"{p0}\t{alt}\t{end}"
        else:
            k, r, x = trim_alleles(call.ref, alt)
            if not r and not x:
                return None
            lo = max(p0, p0 + k - (1 if (not r) != (not x) else 0))
            hi = p0 + k + len(r)
            what = f"{p0 + k}\t{r}\t{x}".upper().replace("U", "T")  # (as the planes and the sidecar read them)
        return lo, max(hi, lo + 1), int.from_bytes(hashlib.blake2b(what.encode(), digest_size=8).digest(), "little")

    def _fits(self, h: int, c: tuple[int, int, int] | None) -> bool:
        """Whether an allele (as :meth:`_occupies` gives it) fits haplotype ``h`` (spec 6.5): it occupies nothing; or
        the same allele was put on ``h`` before; or it wasn't put on the other haplotype, and no ALT allele of an
        earlier call occupies there an offset it occupies."""
        if c is None or c[2] in self.placed[h]:
            return True
        return c[2] not in self.placed[1 - h] and not _bits_any(self.occupied[h], c[0], c[1])

    def _place_overlapping(self, alleles: list[str], call: Call, p0: int, span_end: int) -> list[str]:
        """Unphased genotype of two different alleles: each on a haplotype it fits, if the written order can't.

        In an unphased genotype the allele order means nothing. Heterozygous calls that overlap are alternatives at one
        place, so two of them can't be on one haplotype, yet applied as written they would both go to haplotype 2 and
        run together (two deletions becoming one). So when an allele doesn't fit the haplotype the written order gives
        it (an earlier call's ALT allele occupies there an offset it occupies), and the alleles swapped both fit, they
        are swapped (spec 6.5).
        """
        a, b = alleles
        if a == b or not (_bits_any(self.occupied[0], p0, span_end) or _bits_any(self.occupied[1], p0, span_end)):
            return alleles  # (homozygous; or no ALT allele occupies anything of the record's span yet)
        ia, ib = self._alt_index(a, call), self._alt_index(b, call)
        if any(i and call.alts[i - 1] == "*" for i in (ia, ib)):
            return alleles  # (a '*' allele, which _place_spanning_deletion has placed)
        ca, cb = self._occupies(a, call, p0), self._occupies(b, call, p0)
        if self._fits(0, ca) and self._fits(1, cb):
            return alleles
        if self._fits(0, cb) and self._fits(1, ca):
            self.stats.overlapping_calls_reassigned += 1
            self.stats.note(f"{self.contig}:{call.pos} {call.gt} put the other way round: it overlaps another call")
            return [b, a]
        return alleles

    def _apply_allele(self, h: int, a: str, call: Call, p0: int, m: int, span_end: int) -> np.ndarray | None:
        """Apply one haplotype's allele; returns the mask of strong states written (or None)."""
        st = self.stats
        span = span_end - p0
        if a == ".":
            if span == 1:
                self._write_scalar(h, p0, C.HAP_UNKNOWN)
            else:
                self._write_states(h, p0, np.full(span, C.HAP_UNKNOWN, dtype=np.uint8))
            return None
        idx = allele_number(a)
        if idx is None:
            st.bad_allele_index += 1
            st.note(f"{self.contig}:{call.pos} GT allele {a!r} is not a number")
            return None
        if idx == 0:
            if span == 1:
                self._write_scalar(h, p0, C.HAP_REF)
            else:
                self._write_states(h, p0, np.zeros(span, dtype=np.uint8))
            return None
        if idx > len(call.alts):
            st.bad_allele_index += 1
            st.note(f"{self.contig}:{call.pos} GT allele {idx} but {len(call.alts)} ALT alleles")
            return None
        alt = call.alts[idx - 1]
        if alt == "*":
            # spanning deletion: the record that starts the deletion set DEL
            st.spanning_deletion_alleles += 1
            return None
        if alt in _REFBLOCK_ALTS:
            # "any other allele" placeholder from gVCF: the bases are unknown
            self._write_states(h, p0, np.full(span, C.HAP_UNKNOWN, dtype=np.uint8))
            return None
        if is_symbolic(alt):
            st.symbolic_alleles += 1
            return self._apply_symbolic(h, alt, idx - 1, call, p0, span_end)
        ev = allele_events(call.ref, alt)
        out = np.zeros(span, dtype=bool)
        if span == 1 and not ev.dels and not ev.ins and len(ev.subs) == 1:
            state = ev.subs[0][1]
            if state in C.HAP_SUB_CODES and state == self.ref[p0] & 0x0F:  # (only when REF disagrees with the reference)
                state = C.HAP_REF
            out[0] = self._write_scalar(h, p0, state)  # plain SNV, the common case
            return out
        # states over the record's whole span: REF wherever the allele does not differ
        states = np.zeros(span, dtype=np.uint8)
        for off, s in ev.subs:
            states[off] = s
        for off, length in ev.dels:
            states[off : off + length] = C.HAP_DEL
        same = np.isin(states, C.HAP_SUB_CODES) & (states == np.bitwise_and(self.ref[p0:span_end], 0x0F))
        states[same] = C.HAP_REF  # a "substitution" to the reference base itself is REF (spec 5.2)
        out[:] = self._write_states(h, p0, states)
        for anchor, seq in ev.ins:
            self._add_insertion(h, p0 + anchor, seq)
            out[anchor] = True  # an insertion changes the haplotype at its anchor (spec 6.5)
        return out

    def _apply_symbolic(self, h: int, alt: str, alt_index: int, call: Call, p0: int, span_end: int) -> np.ndarray:
        out = np.zeros(span_end - p0, dtype=bool)
        end = self._symbolic_end(alt, alt_index, call)
        if alt.startswith("<DEL") and end is not None and end > call.pos:
            # SV deletion: POS is the padding base, kept; bases POS+1..END are deleted
            self._write_scalar(h, p0, C.HAP_REF)
            del_start, del_end = p0 + 1, min(end, self.contig_length) - self.origin
            if del_end > del_start:
                s = self._write_states(h, del_start, np.full(del_end - del_start, C.HAP_DEL, dtype=np.uint8))
                out[1 : 1 + s.shape[0]] = s
            return out
        data = f"REF={call.ref};ALT={alt}" + (f";END={end}" if end is not None else "")
        self._write_states(h, p0, np.array([C.HAP_COMPLEX], dtype=np.uint8))
        key = (p0, h + 1)
        if key in self.complex and self.complex[key] != data:
            self.stats.conflicts += 1  # two different structural alleles on one haplotype base
            self.stats.note(f"{self.contig}:{call.pos} two COMPLEX alleles on hap{h + 1}")
        self.complex[key] = data
        out[0] = True
        return out

    # -- result ----------------------------------------------------------
    def finish(self, chunk: int = 1 << 24) -> tuple[dict[str, np.ndarray], list[SidecarEntry]]:
        n = self.length
        # a COMPLEX allele later replaced by another call's variant leaves no COMPLEX state: drop its entry
        complex_entries = [
            SidecarEntry(off, hap, "COMPLEX", data)
            for (off, hap), data in self.complex.items()
            if (self.w[hap - 1][off] & _STATE_MASK) == C.HAP_COMPLEX
        ]
        gt = np.empty(n, dtype=np.uint8)
        flags = self.flags
        absent2 = 0
        for s in range(0, n, chunk):  # chunked, so no full-length temporaries
            e = min(n, s + chunk)
            rb = np.bitwise_and(self.ref[s:e], 0x0F)
            w1, w2 = self.w[0][s:e], self.w[1][s:e]
            for w in (w1, w2):
                # a "substitution" to the reference base itself (possible when the
                # VCF REF disagrees with the reference) is just REF (spec 5.2 rule 1)
                st = w & _STATE_MASK
                same = (st == rb) & np.isin(st, C.HAP_SUB_CODES)
                if same.any():
                    w[same] &= ~np.uint8(_STATE_MASK)
            gt[s:e] = ((w1 & _STATE_MASK) << 4) | (w2 & _STATE_MASK)
            absent2 += int(((w2 & _STATE_MASK) == C.HAP_ABSENT).sum())
            fl = flags[s:e]
            fl[(w1 & _INS) != 0] |= C.FLAG_INS1
            fl[(w2 & _INS) != 0] |= C.FLAG_INS2
        self.w = [None, None]  # release the working arrays
        self.stats.ploidy = 1 if absent2 == n else (2 if absent2 == 0 else "mixed")
        planes = {"gt": gt, "flags": flags}
        if self.gq is not None:
            planes["gq"] = self.gq
        if self.dp is not None:
            planes["dp"] = self.dp
        if self.ps_ord is not None:
            planes["ps"] = self._finish_ps(chunk)
        side = [SidecarEntry(a, h, "INS", s) for (a, h), s in self.ins.items()] + complex_entries
        side.sort(key=lambda e: (e.offset, e.hap, e.kind))
        return planes, side

    def _finish_ps(self, chunk: int) -> np.ndarray:
        """Map phase-set ordinals to stored values (spec 5.5): those of :func:`ps_numbering`, over every identifier
        of the contig's phased calls. ``ps_ids`` gets those that end up on bases (an identifier all of whose bases
        other calls took is on none)."""
        nkeys = len(self.ps_keys)
        counts = np.zeros(nkeys + 1, dtype=np.int64)
        for s in range(0, self.length, chunk):
            counts += np.bincount(self.ps_ord[s : s + chunk], minlength=nkeys + 1)
        values = self.ps_values if self.ps_values is not None else ps_numbering(self.ps_keys)
        lut = np.zeros(nkeys + 1, dtype=np.uint32)
        for i in range(1, nkeys + 1):
            key = self.ps_keys[i - 1]
            if key not in values:
                raise ValueError(f"{self.contig}: the phase set {key!r} isn't among the contig's (ps_values)")
            lut[i] = values[key]
            if counts[i]:
                self.ps_ids[key] = int(values[key])
        ps = np.empty(self.length, dtype=np.uint32)
        for s in range(0, self.length, chunk):
            ps[s : s + chunk] = lut[self.ps_ord[s : s + chunk]]
        self.ps_ord = None
        return ps


def load_bed(path, contig: str, length: int, aliases: Iterable[str] = ()) -> np.ndarray | None:
    """Boolean mask of positions covered by BED intervals on ``contig``.

    Columns may be separated by tabs or spaces. Returns None if the BED has
    no intervals for this contig under any alias.
    """
    names = {contig, *aliases}
    mask = np.zeros(length, dtype=bool)
    seen = False
    with open_maybe_gzip(path) as fh:
        for raw in fh:
            if raw.startswith((b"#", b"track", b"browser")) or not raw.strip():
                continue
            f = raw.decode().split()
            if f[0] in names:
                seen = True
                s, e = int(f[1]), int(f[2])
                mask[max(s, 0) : min(e, length)] = True
    return mask if seen else None


def encode_calls(
    ref: np.ndarray,
    calls: Iterable[Call],
    *,
    contig: str = "",
    callable_mask: np.ndarray | None = None,
    haploid_mask: np.ndarray | None = None,
    planes: Iterable[str] = ("gt", "flags"),
    origin: int = 0,
    contig_length: int | None = None,
    ps_values: dict[str, int] | None = None,
) -> tuple[dict[str, np.ndarray], list[SidecarEntry], EncodeStats, dict[str, int]]:
    """Build variant planes for one contig from an iterable of :class:`Call`; or for a stretch of it that starts
    at ``origin`` (``ref`` and the masks being the stretch's, ``contig_length`` the contig's), from calls that lie
    inside the stretch (see :func:`window_stretch`), with ``ps_values`` the values of the whole contig's phase-set
    identifiers (:func:`ps_numbering`)."""
    wanted = set(planes)
    builder = PlaneBuilder(
        ref,
        callable_mask=callable_mask,
        haploid_mask=haploid_mask,
        want_gq="gq" in wanted,
        want_dp="dp" in wanted,
        want_ps="ps" in wanted,
        contig=contig,
        origin=origin,
        contig_length=contig_length,
        ps_values=ps_values,
    )
    for call in calls:
        builder.add(call)
    out, side = builder.finish()
    return out, side, builder.stats, builder.ps_ids


def window_stretch(spans: Iterable[tuple[int, int, bool] | None], start: int, end: int) -> tuple[int, int, np.ndarray]:
    """The calls to convert for a window ``[start, end)`` of a contig (spec 3.1), from the spans of all its calls in
    file order (:func:`record_span`, None for a record skipped).

    A base depends only on the calls whose spans cover it and, through the placement of unphased calls (spec 6.5), on
    the calls that overlap those, and so on. So the calls whose spans join up with the window, through spans that
    share a base, converted alone over the stretch they cover, give the window the values that converting the whole
    contig would. Returns that stretch ``(a, b)`` and which calls are in it (a boolean per call)."""
    spans = list(spans)
    got = [(i, s[0], s[1]) for i, s in enumerate(spans) if s is not None]
    idx = np.array([g[0] for g in got], dtype=np.int64)
    lo = np.array([g[1] for g in got] + [start], dtype=np.int64)
    hi = np.array([g[2] for g in got] + [end], dtype=np.int64)
    order = np.argsort(lo, kind="stable")
    los, his = lo[order], hi[order]
    reach = np.maximum.accumulate(his)  # (the furthest end of the spans so far, in order of start)
    # a span begins a new run of spans that share bases where it starts at or after the end of every one before it
    begins = np.concatenate(([True], los[1:] >= reach[:-1]))
    run = np.cumsum(begins) - 1
    mine = run[int(np.flatnonzero(order == len(got))[0])]  # (the run with the window in it)
    a, b = int(los[run == mine].min()), int(his[run == mine].max())
    keep = np.zeros(len(spans), dtype=bool)
    keep[idx[(lo[:-1] < b) & (hi[:-1] > a)]] = True
    return a, b, keep


def window_entries(entries: list[SidecarEntry], start: int, end: int, first_gt: int | None = None) -> list[SidecarEntry]:
    """A window's sidecar (spec 3.1): the entries at ``[start, end)``, at the window's offsets; and where the window's
    first base is COMPLEX on a haplotype (``first_gt``, its gt byte) without an entry there (a run of COMPLEX bases
    that began before the window), a copy of the entry nearest before it, which starts the run."""
    out = [SidecarEntry(e.offset - start, e.hap, e.kind, e.data) for e in entries if start <= e.offset < end]
    if start > 0 and first_gt is not None:
        for hap, state in ((1, first_gt >> 4), (2, first_gt & 0x0F)):
            if state != C.HAP_COMPLEX or any(e.offset == 0 and e.hap == hap and e.kind == "COMPLEX" for e in out):
                continue
            before = [e for e in entries if e.kind == "COMPLEX" and e.hap == hap and e.offset < start]
            if before:
                out.append(SidecarEntry(0, hap, "COMPLEX", max(before, key=lambda e: e.offset).data))
    out.sort(key=lambda e: (e.offset, e.hap, e.kind))
    return out


class TempSplit:
    """Context manager around :func:`split_by_contig` with automatic cleanup."""

    def __init__(self, path, sample_index: int, keep=None):
        self.path = path
        self.sample_index = sample_index
        self.keep = keep
        self._tmp = None
        self.index: dict[str, tuple[str, int]] = {}
        self.format_keys: set[str] = set()

    def __enter__(self) -> TempSplit:
        self._tmp = tempfile.TemporaryDirectory(prefix="gpxil-vcf-")
        res = split_by_contig(self.path, self.sample_index, self._tmp.name, keep=self.keep)
        self.index, self.format_keys = res.index, res.format_keys
        return self

    def calls(self, chrom: str, keep=None) -> Iterator[Call]:
        """The calls on ``chrom``, in file order; with ``keep`` (a boolean per call), only those it keeps, the others
        not even read."""
        entry = self.index.get(chrom)
        if entry is None:
            return iter(())
        path, _ = entry

        def gen():
            with open(path, encoding="utf-8") as fh:
                lines = fh if keep is None else (line for line, k in zip(fh, keep) if k)
                for _, call in iter_calls(lines, 0):
                    yield call

        return gen()

    def __exit__(self, *exc) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
