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
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np

from . import codes as C
from .fasta import open_maybe_gzip

_STATE_MASK = 0x0F
_ASSERTED = 0x10
_INS = 0x20

_REFBLOCK_ALTS = {"<NON_REF>", "<*>", "<X>"}
_SYMBOLIC = re.compile(r"^<.*>$")


class VcfError(ValueError):
    pass


# ----------------------------------------------------------------- reading


@dataclass
class VcfHeader:
    samples: list[str]
    contigs: dict[str, int | None]
    lines: list[str]


def read_header(path) -> VcfHeader:
    lines: list[str] = []
    samples: list[str] = []
    contigs: dict[str, int | None] = {}
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
                continue
            if line.startswith("#CHROM"):
                lines.append(line)
                cols = line.split("\t")
                samples = cols[9:]
            break
    return VcfHeader(samples=samples, contigs=contigs, lines=lines)


@dataclass
class Call:
    """One VCF record reduced to what a single-sample gpxil plane needs."""

    pos: int  # 1-based POS as in the VCF
    ref: str
    alts: list[str]
    filter: str
    gt: str
    end: int | None = None  # INFO END (1-based, inclusive)
    svlen: int | None = None
    gq: int | None = None
    dp: int | None = None
    ps: str | None = None


def _info_int(info: str, key: str) -> int | None:
    if info == "." or (key + "=") not in info:
        return None
    for item in info.split(";"):
        if item.startswith(key + "="):
            value = item[len(key) + 1 :].split(",")[0]
            try:
                return int(value)
            except ValueError:
                return None
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
        if not line or line.startswith("#"):
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
        need_end = not alts or any(a.startswith("<") or "[" in a or "]" in a for a in alts)
        yield f[0], Call(
            pos=int(f[1]),
            ref=f[3].upper(),
            alts=[a if a.startswith("<") else a.upper() for a in alts],
            filter=f[6],
            gt=gt,
            end=_info_int(info, "END") if need_end else None,
            svlen=_info_int(info, "SVLEN") if need_end else None,
            gq=_to_int(sample.get("GQ")),
            dp=_to_int(sample.get("DP")),
            ps=sample.get("PS"),
        )


def iter_body_lines(path) -> Iterator[str]:
    with open_maybe_gzip(path) as fh:
        for raw in fh:
            if raw.startswith(b"#"):
                continue
            yield raw.decode("utf-8", "replace")


def split_by_contig(path, sample_index: int, workdir: str) -> dict[str, tuple[str, int]]:
    """Stream a VCF once, writing each contig's calls for one sample to a temp file.

    Returns ``{contig: (temp_path, n_records)}``. This lets the encoder handle
    one chromosome at a time whatever the record order of the VCF.
    """
    handles: dict[str, object] = {}
    counts: Counter[str] = Counter()
    paths: dict[str, str] = {}
    col = 9 + sample_index
    try:
        for line in iter_body_lines(path):
            tab = line.find("\t")
            chrom = line[:tab]
            fh = handles.get(chrom)
            if fh is None:
                p = os.path.join(workdir, f"contig{len(handles):05d}.vcf")
                paths[chrom] = p
                fh = handles[chrom] = open(p, "w", encoding="utf-8")
            # keep the fixed columns plus the chosen sample only
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= col:
                raise VcfError(f"record at {parts[0]}:{parts[1]} has no column for sample {sample_index}")
            fh.write("\t".join(parts[:9] + [parts[col]]) + "\n")
            counts[chrom] += 1
    finally:
        for fh in handles.values():
            fh.close()
    return {c: (paths[c], counts[c]) for c in paths}


# ----------------------------------------------------------------- alleles


def parse_gt(gt: str) -> tuple[list[str], bool]:
    """'0|1' -> (['0', '1'], True); '1/2' -> (['1', '2'], False); '1' -> (['1'], False)."""
    if "|" in gt and "/" not in gt:
        return gt.split("|"), True
    return gt.replace("|", "/").split("/"), False


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
        ev.ins.append((p + k - 1, ac[k:]))
    elif len(rc) > k:
        ev.dels.append((p + k, len(rc) - k))
    return ev


# ----------------------------------------------------------------- planes


@dataclass
class EncodeStats:
    records: int = 0
    skipped_out_of_range: int = 0
    ref_mismatches: int = 0
    conflicts: int = 0
    spanning_deletion_alleles: int = 0
    symbolic_alleles: int = 0
    polyploid_skipped: int = 0
    bad_allele_index: int = 0
    gt_counts: Counter = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        if len(self.examples) < 20:
            self.examples.append(msg)

    def as_dict(self) -> dict:
        return {
            "records": self.records,
            "skipped_out_of_range": self.skipped_out_of_range,
            "ref_mismatches": self.ref_mismatches,
            "conflicts": self.conflicts,
            "spanning_deletion_alleles": self.spanning_deletion_alleles,
            "symbolic_alleles": self.symbolic_alleles,
            "polyploid_skipped": self.polyploid_skipped,
            "bad_allele_index": self.bad_allele_index,
        }


@dataclass
class SidecarEntry:
    offset: int  # 0-based position (pixel index) on the contig
    hap: int  # 1 or 2
    kind: str  # "INS" or "COMPLEX"
    data: str

    def to_tsv(self) -> str:
        return f"{self.offset}\t{self.hap}\t{self.kind}\t{self.data}"


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
        if callable_mask is not None:
            outside = ~callable_mask
            self.w[0][outside] = C.HAP_UNKNOWN
            self.w[1][outside] = C.HAP_UNKNOWN
        if haploid_mask is not None:
            self.w[1][haploid_mask] = C.HAP_ABSENT
        self.flags = np.zeros(n, dtype=np.uint8)
        self.gq = np.full(n, C.QUAL_MISSING, dtype=np.uint8) if want_gq else None
        self.dp = np.full(n, C.QUAL_MISSING, dtype=np.uint8) if want_dp else None
        self.ps = np.zeros(n, dtype=np.uint32) if want_ps else None
        self.ps_ids: dict[str, int] = {}
        self.ps_synthetic: dict[str, str] = {}
        self.ins: dict[tuple[int, int], str] = {}
        self.complex: list[SidecarEntry] = []
        self.stats = EncodeStats()
        self._ref_upper = None

    # -- helpers ---------------------------------------------------------
    def _ref_ok(self, start: int, allele: str) -> bool:
        """REF allele bases must be compatible with the reference (IUPAC-aware)."""
        seg = np.bitwise_and(self.ref[start : start + len(allele)], 0x0F)
        for i, ch in enumerate(allele):
            bit = C.BASE_TO_SUB.get(ch)
            if bit is None:
                continue  # N or IUPAC in REF: accept
            code = int(seg[i])
            if code != C.BASE_N and not (code & bit):
                return False
        return True

    def _write_states(self, h: int, start: int, states: np.ndarray) -> None:
        """Write per-base states for haplotype ``h`` with the precedence rules.

        Non-REF states always win (a different non-REF state already asserted
        counts as a conflict; the later record wins). REF and UNKNOWN only
        fill positions no record has asserted yet.
        """
        w = self.w[h]
        end = start + states.shape[0]
        seg = w[start:end]
        asserted = (seg & _ASSERTED) != 0
        cur = seg & _STATE_MASK
        weak = (states == C.HAP_REF) | (states == C.HAP_UNKNOWN)
        strong = ~weak
        conflict = asserted & strong & (cur != C.HAP_REF) & (cur != states) & (cur != C.HAP_UNKNOWN)
        if conflict.any():
            self.stats.conflicts += int(conflict.sum())
            self.stats.note(f"{self.contig}:{start + int(np.flatnonzero(conflict)[0]) + 1} conflicting states on hap{h + 1}")
        new = np.where(strong, states, np.where(asserted, cur, states))
        w[start:end] = new.astype(np.uint8) | _ASSERTED | (seg & _INS)

    def _write_scalar(self, h: int, pos: int, state: int) -> None:
        w = self.w[h]
        v = int(w[pos])
        asserted = v & _ASSERTED
        cur = v & _STATE_MASK
        if state in (C.HAP_REF, C.HAP_UNKNOWN):
            if asserted:
                return
            w[pos] = state | _ASSERTED | (v & _INS)
            return
        if asserted and cur not in (C.HAP_REF, C.HAP_UNKNOWN, state):
            self.stats.conflicts += 1
            self.stats.note(f"{self.contig}:{pos + 1} conflicting states on hap{h + 1}")
        w[pos] = state | _ASSERTED | (v & _INS)

    def _add_insertion(self, h: int, anchor: int, seq: str) -> None:
        key = (anchor, h + 1)
        if key in self.ins and self.ins[key] != seq:
            self.stats.conflicts += 1
            self.stats.note(f"{self.contig}:{anchor + 1} two insertions on hap{h + 1}")
        self.ins[key] = seq
        self.w[h][anchor] |= _INS

    def _ps_value(self, ps: str) -> int:
        """VCF PS -> ps-plane value: the PS integer itself when it fits in 1..2^32-1.

        Other identifiers (0, negative, non-numeric) get synthetic values
        counting down from 2^32-1; the mapping is kept in ``ps_synthetic``.
        """
        v = self.ps_ids.get(ps)
        if v is not None:
            return v
        try:
            v = int(ps)
        except ValueError:
            v = -1
        if not (1 <= v <= 0xFFFFFFFF):
            v = 0xFFFFFFFF - len(self.ps_synthetic)
            self.ps_synthetic[str(v)] = ps
        self.ps_ids[ps] = v
        return v

    # -- main entry ------------------------------------------------------
    def add(self, call: Call) -> None:
        st = self.stats
        st.records += 1
        p0 = call.pos - 1
        alts = call.alts
        only_refblock = not alts or all(a in _REFBLOCK_ALTS for a in alts)
        symbolic = any(_SYMBOLIC.match(a) for a in alts)
        span_end = p0 + max(len(call.ref), 1)
        end = call.end
        if end is None and symbolic and call.svlen is not None:
            end = call.pos + abs(call.svlen)
        if end is not None and (only_refblock or symbolic):
            span_end = max(span_end, end)
        if p0 < 0 or span_end > self.length:
            st.skipped_out_of_range += 1
            st.note(f"{self.contig}:{call.pos} outside contig (length {self.length})")
            return
        if not self._ref_ok(p0, call.ref):
            st.ref_mismatches += 1
            st.note(f"{self.contig}:{call.pos} REF {call.ref} does not match the reference")

        alleles, phased = parse_gt(call.gt)
        st.gt_counts[call.gt] += 1
        if len(alleles) > 2:
            st.polyploid_skipped += 1
            st.note(f"{self.contig}:{call.pos} ploidy {len(alleles)} not supported in gpxil 0.1")
            return
        haploid = len(alleles) == 1

        m = len(call.ref)
        for h, a in enumerate(alleles):
            self._apply_allele(h, a, call, p0, m, span_end)
        if haploid:
            span = span_end - p0
            if span == 1:
                self._write_scalar(1, p0, C.HAP_ABSENT)
            else:
                self._write_states(1, p0, np.full(span, C.HAP_ABSENT, dtype=np.uint8))

        f = C.FLAG_VARIANT_SITE
        if phased and not haploid:
            f |= C.FLAG_PHASED
        if call.filter not in ("PASS", "."):
            f |= C.FLAG_FILTERED
        self.flags[p0:span_end] |= f
        if self.gq is not None and call.gq is not None:
            self.gq[p0:span_end] = min(max(call.gq, 0), C.QUAL_MAX)
        if self.dp is not None and call.dp is not None:
            self.dp[p0:span_end] = min(max(call.dp, 0), C.QUAL_MAX)
        if self.ps is not None and phased and call.ps not in (None, ".", ""):
            self.ps[p0:span_end] = self._ps_value(call.ps)

    def _apply_allele(self, h: int, a: str, call: Call, p0: int, m: int, span_end: int) -> None:
        st = self.stats
        if a == ".":
            span = span_end - p0
            if span == 1:
                self._write_scalar(h, p0, C.HAP_UNKNOWN)
            else:
                self._write_states(h, p0, np.full(span, C.HAP_UNKNOWN, dtype=np.uint8))
            return
        try:
            idx = int(a)
        except ValueError:
            st.bad_allele_index += 1
            return
        if idx == 0:
            span = span_end - p0
            if span == 1:
                self._write_scalar(h, p0, C.HAP_REF)
            else:
                self._write_states(h, p0, np.zeros(span, dtype=np.uint8))
            return
        if idx > len(call.alts):
            st.bad_allele_index += 1
            st.note(f"{self.contig}:{call.pos} GT allele {idx} but only {len(call.alts)} ALT alleles")
            return
        alt = call.alts[idx - 1]
        if alt == "*":
            # Spanning deletion: the deleting record upstream already set DEL.
            st.spanning_deletion_alleles += 1
            return
        if alt in _REFBLOCK_ALTS:
            # "some other allele" placeholder from gVCF: base unknown
            self._write_states(h, p0, np.full(span_end - p0, C.HAP_UNKNOWN, dtype=np.uint8))
            return
        if alt.startswith("<") or "[" in alt or "]" in alt:
            st.symbolic_alleles += 1
            self._apply_symbolic(h, alt, call, p0, span_end)
            return
        ev = allele_events(call.ref, alt)
        if m == 1 and not ev.dels and not ev.ins and len(ev.subs) == 1:
            self._write_scalar(h, p0, ev.subs[0][1])  # plain SNV, the common case
            return
        states = np.zeros(m, dtype=np.uint8)
        for off, s in ev.subs:
            states[off] = s
        for off, length in ev.dels:
            states[off : off + length] = C.HAP_DEL
        self._write_states(h, p0, states)
        for anchor, seq in ev.ins:
            pos = p0 + anchor
            if pos < 0:
                self._mark_complex(h, p0, p0 + m, f"REF={call.ref};ALT={alt}")
                continue
            self._add_insertion(h, pos, seq)

    def _apply_symbolic(self, h: int, alt: str, call: Call, p0: int, span_end: int) -> None:
        end = call.end
        if end is None and call.svlen is not None:
            end = call.pos + abs(call.svlen)
        if alt.startswith("<DEL") and end is not None and end > call.pos:
            # SV deletion: POS is the padding base; bases POS+1..END are deleted
            del_start, del_end = p0 + 1, min(end, self.length)
            self._write_states(h, del_start, np.full(del_end - del_start, C.HAP_DEL, dtype=np.uint8))
            return
        data = f"REF={call.ref};ALT={alt}" + (f";END={end}" if end is not None else "")
        self._mark_complex(h, p0, p0 + 1, data)

    def _mark_complex(self, h: int, start: int, end: int, data: str) -> None:
        self._write_states(h, start, np.full(end - start, C.HAP_COMPLEX, dtype=np.uint8))
        self.complex.append(SidecarEntry(start, h + 1, "COMPLEX", data))

    # -- result ----------------------------------------------------------
    def finish(self, chunk: int = 1 << 24) -> tuple[dict[str, np.ndarray], list[SidecarEntry]]:
        n = self.length
        gt = np.empty(n, dtype=np.uint8)
        flags = self.flags
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
            fl = flags[s:e]
            fl[(w1 & _INS) != 0] |= C.FLAG_INS1
            fl[(w2 & _INS) != 0] |= C.FLAG_INS2
        self.w = [None, None]  # release the working arrays
        planes = {"gt": gt, "flags": flags}
        if self.gq is not None:
            planes["gq"] = self.gq
        if self.dp is not None:
            planes["dp"] = self.dp
        if self.ps is not None:
            planes["ps"] = self.ps
        side = [SidecarEntry(a, h, "INS", s) for (a, h), s in self.ins.items()] + self.complex
        side.sort(key=lambda e: (e.offset, e.hap, e.kind))
        return planes, side


def load_bed(path, contig: str, length: int, aliases: Iterable[str] = ()) -> np.ndarray | None:
    """Boolean mask of positions covered by BED intervals on ``contig``.

    Returns None if the BED has no intervals for this contig under any alias.
    """
    names = {contig, *aliases}
    mask = np.zeros(length, dtype=bool)
    seen = False
    with open_maybe_gzip(path) as fh:
        for raw in fh:
            if raw.startswith((b"#", b"track", b"browser")) or not raw.strip():
                continue
            f = raw.decode().split("\t")
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


def scan_formats(path, sample_index: int, max_records: int = 20000) -> set[str]:
    """FORMAT keys present (with a value) for the sample in the first records."""
    keys: set[str] = set()
    col = 9 + sample_index
    for i, line in enumerate(iter_body_lines(path)):
        if i >= max_records:
            break
        f = line.rstrip("\r\n").split("\t")
        if len(f) > col:
            for k, v in zip(f[8].split(":"), f[col].split(":")):
                if v not in (".", ""):
                    keys.add(k)
    return keys


class TempSplit:
    """Context manager around :func:`split_by_contig` with automatic cleanup."""

    def __init__(self, path, sample_index: int):
        self.path = path
        self.sample_index = sample_index
        self._tmp = None
        self.index: dict[str, tuple[str, int]] = {}

    def __enter__(self) -> "TempSplit":
        self._tmp = tempfile.TemporaryDirectory(prefix="gpxil-vcf-")
        self.index = split_by_contig(self.path, self.sample_index, self._tmp.name)
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
