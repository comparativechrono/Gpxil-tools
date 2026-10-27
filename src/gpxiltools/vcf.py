"""VCF reading and conversion of one sample's genotypes into gpxil planes.

The conversion turns each VCF record into per-base haplotype states (spec
section 6). It preserves the content of both haplotypes: every substitution,
deletion and insertion, the phase, haploid calls and no-calls. It does not
preserve VCF record syntax (how alleles were split into records); exporting
back to VCF gives an equivalent, normalised call set.
"""

from __future__ import annotations

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
                    m_len = re.search(r"[<,]length=(\d+)", line)
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


def split_by_contig(path, sample_index: int, workdir: str, buffer_bytes: int = 32 << 20) -> SplitResult:
    """Stream a VCF once, writing each contig's calls for one sample to a temp file.

    This lets the encoder handle one chromosome at a time whatever the record
    order of the VCF. Lines are buffered in memory and appended to the
    per-contig files in batches, so only one file is open at a time however
    many contigs there are. The FORMAT keys of every record are collected on
    the way, so the choice of optional planes sees the whole file.
    """
    buffers: dict[str, list[str]] = {}
    counts: Counter[str] = Counter()
    paths: dict[str, str] = {}
    formats_seen: set[str] = set()
    format_keys: set[str] = set()
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
        if chrom not in paths:
            paths[chrom] = os.path.join(workdir, f"contig{len(paths):05d}.vcf")
            buffers[chrom] = []
        if len(parts) > 8 and parts[8] not in formats_seen:
            formats_seen.add(parts[8])
            format_keys.update(parts[8].split(":"))
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
    m, n = len(ref), len(alt)
    qmax = min(m, n) - (1 if m != n else 0)
    q = 0
    while q < qmax and ref[m - 1 - q] == alt[n - 1 - q]:
        q += 1
    p = 0
    while p < m - q and p < n - q and ref[p] == alt[p]:
        p += 1
    rc, ac = ref[p : m - q], alt[p : n - q]
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


class PlaneBuilder:
    """Accumulates one sample's calls on one contig into gt/flags/... planes."""

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
    ):
        n = ref.shape[0]
        self.ref = ref
        self.contig = contig
        self.length = n
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
            self.stats.note(f"{self.contig}:{start + int(np.flatnonzero(conflict)[0]) + 1} conflicting states on hap{h + 1}")
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
                    self.stats.note(f"{self.contig}:{pos + 1} haploid call meets a variant on hap2")
                return False
            w[pos] = state | _ASSERTED | keep
            return False
        if asserted and cur not in _WEAK and cur != state:
            self.stats.conflicts += 1
            self.stats.note(f"{self.contig}:{pos + 1} conflicting states on hap{h + 1}")
        w[pos] = state | _ASSERTED | keep
        return True

    def _add_insertion(self, h: int, anchor: int, seq: str) -> None:
        seq = seq.upper().replace("U", "T")
        bad = len(_ACGTN.findall(seq))
        if bad:
            self.stats.nonacgt_inserted_bases += bad
            seq = _ACGTN.sub("N", seq)
        key = (anchor, h + 1)
        if key in self.ins and self.ins[key] != seq:
            self.stats.conflicts += 1
            self.stats.note(f"{self.contig}:{anchor + 1} two insertions on hap{h + 1}")
        self.ins[key] = seq
        self.w[h][anchor] |= _INS

    def _ps_ordinal(self, ps: str) -> int:
        try:
            v = int(ps)
            if 1 <= v <= 0xFFFFFFFF:
                ps = str(v)  # '077' and '77' are the same VCF integer
        except ValueError:
            pass
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
        p0 = call.pos - 1
        alts = call.alts
        n = self.length
        if p0 < 0 or p0 >= n or p0 + len(call.ref) > n:
            st.skipped_out_of_range += 1
            st.note(f"{self.contig}:{call.pos} outside the contig (length {n})")
            return
        only_refblock = not alts or all(a in _REFBLOCK_ALTS for a in alts)
        symbolic = any(is_symbolic(a) for a in alts)
        span_end = p0 + max(len(call.ref), 1)
        if call.end is not None and (only_refblock or symbolic):
            span_end = max(span_end, call.end)
        elif symbolic and call.svlen:
            for i, a in enumerate(alts):
                if a.startswith("<DEL") and i < len(call.svlen) and call.svlen[i]:
                    span_end = max(span_end, call.pos + abs(call.svlen[i]))
        if span_end > n:
            st.clipped_to_contig_end += 1
            st.note(f"{self.contig}:{call.pos} span clipped at the contig end")
            span_end = n
        if not self._ref_ok(p0, call.ref):
            st.ref_mismatches += 1
            st.note(f"{self.contig}:{call.pos} REF {call.ref} does not match the reference")

        alleles, phased = parse_gt(call.gt)
        st.gt_counts[call.gt] += 1
        if len(alleles) > 2:
            st.polyploid_skipped += 1
            st.note(f"{self.contig}:{call.pos} ploidy {len(alleles)} not supported in gpxil 0.1")
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
            for h, a in enumerate(alleles):
                s = self._apply_allele(h, a, call, p0, len(call.ref), span_end)
                if s is not None:
                    strong |= s
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
        star = []
        for i, a in enumerate(alleles):
            if a.isdigit() and 0 < int(a) <= len(call.alts) and call.alts[int(a) - 1] == "*":
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
        try:
            idx = int(a)
        except ValueError:
            st.bad_allele_index += 1
            st.note(f"{self.contig}:{call.pos} GT allele {a!r} is not a number")
            return None
        if idx == 0:
            if span == 1:
                self._write_scalar(h, p0, C.HAP_REF)
            else:
                self._write_states(h, p0, np.zeros(span, dtype=np.uint8))
            return None
        if idx < 0 or idx > len(call.alts):
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
            if state == self.ref[p0] & 0x0F:  # possible only when REF disagrees with the reference
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
        end = call.end
        if end is None and alt.startswith("<DEL") and call.svlen and alt_index < len(call.svlen) and call.svlen[alt_index]:
            end = call.pos + abs(call.svlen[alt_index])
        if alt.startswith("<DEL") and end is not None and end > call.pos:
            # SV deletion: POS is the padding base, kept; bases POS+1..END are deleted
            self._write_scalar(h, p0, C.HAP_REF)
            del_start, del_end = p0 + 1, min(end, self.length)
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
        """Map phase-set ordinals to stored values (spec 5.5).

        An integer PS in 1..2^32-1 is stored as is. Other identifiers get the
        largest values not used by an integer PS on this contig, in order of
        first appearance. Identifiers that end up on no base (every base of
        their calls taken by other calls) are dropped.
        """
        nkeys = len(self.ps_keys)
        counts = np.zeros(nkeys + 1, dtype=np.int64)
        for s in range(0, self.length, chunk):
            counts += np.bincount(self.ps_ord[s : s + chunk], minlength=nkeys + 1)
        used = [i for i in range(1, nkeys + 1) if counts[i]]
        ints: dict[int, int] = {}
        for i in used:
            try:
                v = int(self.ps_keys[i - 1])
            except ValueError:
                continue
            if 1 <= v <= 0xFFFFFFFF:
                ints[i] = v
        taken = set(ints.values())
        lut = np.zeros(nkeys + 1, dtype=np.uint32)
        nxt = 0xFFFFFFFF
        for i in used:
            v = ints.get(i)
            if v is None:
                while nxt in taken:
                    nxt -= 1
                v = nxt
                taken.add(v)
            lut[i] = v
            self.ps_ids[self.ps_keys[i - 1]] = int(v)
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
) -> tuple[dict[str, np.ndarray], list[SidecarEntry], EncodeStats, dict[str, int]]:
    """Build variant planes for one contig from an iterable of :class:`Call`."""
    wanted = set(planes)
    builder = PlaneBuilder(
        ref,
        callable_mask=callable_mask,
        haploid_mask=haploid_mask,
        want_gq="gq" in wanted,
        want_dp="dp" in wanted,
        want_ps="ps" in wanted,
        contig=contig,
    )
    for call in calls:
        builder.add(call)
    out, side = builder.finish()
    return out, side, builder.stats, builder.ps_ids


class TempSplit:
    """Context manager around :func:`split_by_contig` with automatic cleanup."""

    def __init__(self, path, sample_index: int):
        self.path = path
        self.sample_index = sample_index
        self._tmp = None
        self.index: dict[str, tuple[str, int]] = {}
        self.format_keys: set[str] = set()

    def __enter__(self) -> TempSplit:
        self._tmp = tempfile.TemporaryDirectory(prefix="gpxil-vcf-")
        res = split_by_contig(self.path, self.sample_index, self._tmp.name)
        self.index, self.format_keys = res.index, res.format_keys
        return self

    def calls(self, chrom: str) -> Iterator[Call]:
        entry = self.index.get(chrom)
        if entry is None:
            return iter(())
        path, _ = entry

        def gen():
            with open(path, encoding="utf-8") as fh:
                for _, call in iter_calls(fh, 0):
                    yield call

        return gen()

    def __exit__(self, *exc) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
