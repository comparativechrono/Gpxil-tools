"""Conformance checks for gpxil files (spec section 10).

Planes are checked chunk by chunk, so validating a whole genome needs about
as much memory as one chunk of every plane of a contig.
"""

from __future__ import annotations

import json
import zlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import codes as C
from .container import (
    EXTENSION_DTYPES,
    FORMAT_VERSION,
    SIDECAR_TAG,
    STANDARD_PLANES,
    GpxilFile,
    GpxilFormatError,
    _asarray,
    _decode,
    decode_sidecar,
    rows_for,
    sort_planes,
)
from .fasta import SequenceDigester

CHUNK = 1 << 24  #: bases checked at a time

_VALID_REF = np.zeros(256, dtype=bool)
_VALID_REF[0:16] = True
_VALID_REF[0x11:0x20] = True  # soft-masked bases (0x10 alone = masked gap is not valid)
_VALID_STATE = np.zeros(16, dtype=bool)
_VALID_STATE[list(C.VALID_HAP_STATES)] = True
_ALLOWED_COMPRESSION = {1, 5, 8, 32946, 32773, 50000, 34925}  # none, LZW, Deflate, old Deflate, PackBits, ZSTD, LZMA
_PHOTOMETRIC = {"palette": 3, "minisblack": 1}
_INS_BASES = set("ACGTN")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


class _Tally:
    """Counts problems found chunk by chunk and remembers the first offset of each."""

    def __init__(self) -> None:
        self.count: Counter[str] = Counter()
        self.first: dict[str, int] = {}

    def add(self, key: str, mask: np.ndarray, base: int) -> np.ndarray | None:
        if not mask.any():
            return None
        idx = np.flatnonzero(mask)
        self.count[key] += int(idx.size)
        self.first.setdefault(key, base + int(idx[0]))
        return idx


def _check_schema(meta: dict, rep: Report) -> None:
    """Validate the metadata against the bundled JSON Schema, if jsonschema is installed."""
    try:
        import jsonschema
    except ImportError:
        rep.warn("jsonschema not installed: metadata schema not checked")
        return
    from importlib.resources import files

    schema = json.loads(files("gpxiltools").joinpath("schema/gpxil-0.1.schema.json").read_text())
    for err in jsonschema.Draft202012Validator(schema).iter_errors(meta):
        where = "/".join(str(p) for p in err.absolute_path) or "(top level)"
        rep.error(f"metadata {where}: {err.message}")


def validate(path, *, check_digests: bool = True, chunk: int = CHUNK) -> Report:
    """Check a file against the specification; returns a :class:`Report`."""
    rep = Report()
    try:
        g = GpxilFile(path)
    except GpxilFormatError as exc:
        rep.error(str(exc))
        return rep
    except Exception as exc:  # not a TIFF at all, truncated, ...
        rep.error(f"cannot open as TIFF: {exc}")
        return rep
    with g:
        meta = g.metadata
        _check_schema(meta, rep)
        if str(meta.get("gpxil")) != FORMAT_VERSION:
            rep.warn(f"file declares gpxil {meta.get('gpxil')}, validator implements {FORMAT_VERSION}")
        layout = meta.get("layout") or {}
        if layout.get("type") != "raster":
            rep.error(f"layout.type must be 'raster', got {layout.get('type')!r}")
        entries = [e for e in meta.get("contigs", []) if isinstance(e, dict)]
        _check_file(meta, entries, rep)
        n_pages = len(g._tf.pages)
        seen_pages: dict[int, tuple[str, str]] = {}
        for entry in entries:
            _check_contig(g, entry, rep, seen_pages, n_pages, check_digests, chunk)
        unclaimed = [i for i in range(n_pages) if i not in seen_pages]
        if unclaimed:
            rep.warn(f"pages not listed in metadata (ignored by readers): {unclaimed[:10]}")
    return rep


def _check_file(meta: dict, entries: list[dict], rep: Report) -> None:
    """Rules that span contigs: sample, plane descriptions, names and page order."""
    has_gt = any("gt" in (e.get("pages") or {}) for e in entries)
    sample = meta.get("sample")
    if has_gt and not (isinstance(sample, dict) and sample.get("id")):
        rep.error("the file has gt planes but metadata 'sample' is not an object with an 'id' (sections 9.1, 9.4)")
    if not has_gt and sample is not None:
        rep.error("metadata 'sample' must be null in a file without gt planes (section 9.1)")
    plane_meta = meta.get("planes") or {}
    used = {p for e in entries for p in (e.get("pages") or {})}
    for p in sorted(used):
        if p.startswith("x-") and p not in plane_meta:
            rep.error(f"extension plane {p} is not described in the 'planes' metadata (section 9.2)")
        want = STANDARD_PLANES.get(p, {}).get("dtype")
        have = (plane_meta.get(p) or {}).get("dtype")
        if want and have and have != want:
            rep.error(f"'planes' metadata gives {p} dtype {have}; the standard plane is {want}")
    names = Counter(e.get("name") for e in entries)
    dup = [n for n, k in names.items() if k > 1]
    if dup:
        rep.error(f"duplicate contig names: {dup[:5]}")
    # page order (section 8.2): contigs in metadata order, planes in standard order
    order = []
    for e in entries:
        pages = e.get("pages") or {}
        for p in sort_planes(pages):
            if isinstance(pages[p], int):
                order.append((e.get("name"), p, pages[p]))
    if order:
        if order[0][2] != 0:
            rep.error("page 0 must hold the first plane of the first contig (section 8.2)")
        for prev, cur in zip(order, order[1:]):
            if cur[2] <= prev[2]:
                rep.error(
                    f"page order differs from section 8.2: {cur[0]}/{cur[1]} (page {cur[2]}) comes after "
                    f"{prev[0]}/{prev[1]} (page {prev[2]})"
                )
                break


def _padding_ok(page, length: int, width: int) -> bool:
    """True if the pixels after ``length`` in the last row are 0."""
    rows = rows_for(length, width)
    tail = rows * width - length
    if tail == 0:
        return True
    if page.is_tiled or page.rowsperstrip <= 0:
        return not _asarray(page).reshape(-1)[length:].any()
    rps = page.rowsperstrip
    s = (rows - 1) // rps
    fh = page.parent.filehandle
    fh.seek(page.dataoffsets[s])
    seg, _, _ = _decode(page, fh.read(page.databytecounts[s]), s)
    flat = np.asarray(seg).reshape(-1)
    last = (rows - 1 - s * rps) * width
    return not flat[last + width - tail : last + width].any()


def _check_contig(g: GpxilFile, entry: dict, rep: Report, seen_pages: dict, n_pages: int, check_digests: bool, chunk: int) -> None:
    name = entry.get("name")
    if not name or "length" not in entry or "pages" not in entry:
        rep.error(f"contig entry missing name/length/pages: {entry}")
        return
    contig = g[name]
    L, W = contig.length, contig.width
    rows = rows_for(L, W)
    plane_meta = g.metadata.get("planes") or {}
    pages = g._tf.pages
    ok: dict[str, object] = {}
    for plane, idx in contig.pages.items():
        where = f"{name}/{plane}"
        if plane not in STANDARD_PLANES and not plane.startswith("x-"):
            rep.error(f"{where}: unknown plane name")
        if not (0 <= idx < n_pages):
            rep.error(f"{where}: page index {idx} out of range (file has {n_pages} pages)")
            continue
        if idx in seen_pages:
            rep.error(f"{where}: page {idx} already used by {seen_pages[idx]}")
        seen_pages[idx] = (name, plane)
        page = pages[idx]
        if page.shape[:2] != (rows, W) or page.samplesperpixel != 1:
            rep.error(f"{where}: page shape {page.shape} / {page.samplesperpixel} samples, expected ({rows}, {W}) x 1")
            continue
        want = STANDARD_PLANES.get(plane, {}).get("dtype") or (plane_meta.get(plane) or {}).get("dtype")
        if want and page.dtype != np.dtype(want):
            rep.error(f"{where}: dtype {page.dtype}, expected {want}")
        elif plane.startswith("x-") and page.dtype.name not in EXTENSION_DTYPES:
            rep.error(f"{where}: dtype {page.dtype} is not allowed for an extension plane")
        photometric = _PHOTOMETRIC[STANDARD_PLANES.get(plane, {}).get("photometric", "minisblack")]
        if int(page.photometric) != photometric:
            rep.error(f"{where}: PhotometricInterpretation {int(page.photometric)}, expected {photometric} (section 8.3)")
        if plane in ("ref", "gt", "flags") and int(page.predictor) not in (0, 1):
            rep.error(f"{where}: predictor {page.predictor!r} not allowed for a categorical plane")
        if int(page.compression) not in _ALLOWED_COMPRESSION:
            rep.warn(f"{where}: compression {page.compression!r} is outside the recommended set")
        try:
            d = json.loads(page.description) if idx != 0 else g.metadata
            if not isinstance(d.get("gpxil"), str):
                rep.error(f"{where}: page description has no 'gpxil' version (section 8.4)")
            if d.get("contig") != name or d.get("plane") != plane or d.get("length") != L:
                rep.error(f"{where}: page description names {d.get('contig')}/{d.get('plane')} of length {d.get('length')}")
        except Exception:
            rep.error(f"{where}: page description is not gpxil JSON")
        if not _padding_ok(page, L, W):
            rep.error(f"{where}: padding pixels after position {L} are not 0")
        ok[plane] = page

    if "gt" in contig.pages and "flags" not in contig.pages:
        rep.error(f"{name}: gt plane without flags plane")
    extra = [p for p in ("gq", "dp", "ps") if p in contig.pages]
    if extra and "gt" not in contig.pages:
        rep.error(f"{name}: {', '.join(extra)} without a gt plane (section 3)")
    if "ref" not in contig.pages and not entry.get("md5") and not entry.get("refget"):
        rep.error(f"{name}: no ref plane and no md5/refget digest identifying the reference")

    # sidecar
    side = []
    if "gt" in ok:
        tag = ok["gt"].tags.get(SIDECAR_TAG)
        if tag is not None:
            try:
                blob = bytes(tag.value)
                lines = zlib.decompress(blob).decode("utf-8").split("\n", 2)
                if len(lines) < 2 or not lines[0].startswith("#gpxil-sidecar\t") or lines[1] != "offset\thap\tkind\tdata":
                    rep.error(f"{name}: sidecar does not start with its two header lines (section 7)")
                side = decode_sidecar(blob)
            except Exception as exc:
                rep.error(f"{name}: sidecar cannot be decoded: {exc}")
    complex_entries = sorted((e.offset, e.hap) for e in side if e.kind == "COMPLEX")

    # per-base checks, chunk by chunk
    t = _Tally()
    digester = SequenceDigester() if (check_digests and "ref" in ok and (entry.get("md5") or entry.get("refget"))) else None
    ins_flagged: dict[int, set[int]] = {1: set(), 2: set()}
    complex_starts: set[tuple[int, int]] = set()
    stale_complex: list[tuple[int, int]] = []
    prev_complex = {1: False, 2: False}
    ce = 0  # next COMPLEX entry to check
    for s in range(0, L, chunk):
        e = min(L, s + chunk)
        ref = contig.read("ref", s, e) if "ref" in ok else None
        if ref is not None:
            t.add("ref", ~_VALID_REF[ref], s)
            if digester is not None:
                digester.update(ref)
        if "gt" in ok:
            gt = contig.read("gt", s, e)
            nibs = {1: C.hap1(gt), 2: C.hap2(gt)}
            del gt
            for hap, nib in nibs.items():
                t.add(f"state{hap}", ~_VALID_STATE[nib], s)
                if ref is not None:
                    t.add(f"same{hap}", np.isin(nib, C.HAP_SUB_CODES) & (nib == np.bitwise_and(ref, 0x0F)), s)
                comp = nib == C.HAP_COMPLEX
                if comp.any():
                    start = comp.copy()
                    start[1:] &= ~comp[:-1]
                    start[0] &= not prev_complex[hap]
                    complex_starts.update((s + int(i), hap) for i in np.flatnonzero(start))
                prev_complex[hap] = bool(comp[-1])
            while ce < len(complex_entries) and complex_entries[ce][0] < e:
                off, hap = complex_entries[ce]
                if off >= s and hap in nibs and nibs[hap][off - s] != C.HAP_COMPLEX:
                    stale_complex.append((off, hap))
                ce += 1
        if "flags" in ok:
            flags = contig.read("flags", s, e)
            t.add("flags", (flags & C.FLAG_RESERVED_MASK) != 0, s)
            for hap, bit in ((1, C.FLAG_INS1), (2, C.FLAG_INS2)):
                ins_flagged[hap].update(s + int(i) for i in np.flatnonzero(flags & bit))
            if "ps" in ok:
                ps = contig.read("ps", s, e)
                t.add("ps", (ps != 0) & ((flags & C.FLAG_PHASED) == 0), s)

    messages = {
        "ref": "ref: invalid codes",
        "state1": "gt: invalid hap1 states",
        "state2": "gt: invalid hap2 states",
        "same1": "gt: hap1 substitution equals the reference base (REF must be coded 0)",
        "same2": "gt: hap2 substitution equals the reference base (REF must be coded 0)",
        "flags": "flags: reserved bits set",
        "ps": "ps: non-zero phase set where the PHASED flag is clear (section 5.5)",
    }
    for key, msg in messages.items():
        if t.count[key]:
            rep.error(f"{name}/{msg} at {t.count[key]:,} positions (first offset {t.first[key]})")

    if digester is not None:
        dig = digester.result()
        for k in ("md5", "refget"):
            if entry.get(k) and entry[k] != dig[k]:
                rep.error(f"{name}/ref: {k} {dig[k]} does not match metadata {entry[k]}")

    # sidecar rules (section 7)
    keys = [(e.offset, e.hap, e.kind) for e in side]
    if keys != sorted(keys):
        rep.error(f"{name}: sidecar entries are not sorted by offset, hap and kind")
    ins_count = Counter((e.offset, e.hap) for e in side if e.kind == "INS")
    dups = [k for k, n in ins_count.items() if n > 1]
    if dups:
        rep.error(f"{name}: {len(dups)} bases have more than one INS entry on the same haplotype, e.g. {sorted(dups)[:3]}")
    cdups = sorted(k for k, n in Counter(complex_entries).items() if n > 1)
    if cdups:
        rep.error(f"{name}: {len(cdups)} bases have more than one COMPLEX entry on the same haplotype, e.g. {cdups[:3]}")
    for e in side:
        if not (0 <= e.offset < L):
            rep.error(f"{name}: sidecar offset {e.offset} outside contig")
        if e.hap not in (1, 2):
            rep.error(f"{name}: sidecar entry at {e.offset} has hap {e.hap}")
        if e.kind == "INS":
            if not e.data or not set(e.data) <= _INS_BASES:
                rep.error(f"{name}: insertion at {e.offset} must be uppercase A, C, G, T or N, got {e.data[:20]!r}")
        elif e.kind == "COMPLEX":
            items = {kv.split("=", 1)[0] for kv in e.data.split(";")}
            if not {"REF", "ALT"} <= items:
                rep.error(f"{name}: COMPLEX entry at {e.offset} lacks REF= or ALT=")
        else:
            rep.warn(f"{name}: unknown sidecar kind {e.kind!r}")
    if "flags" in ok:
        for hap in (1, 2):
            mine = {k[0] for k in ins_count if k[1] == hap}
            if ins_flagged[hap] != mine:
                diff = sorted(ins_flagged[hap] ^ mine)
                rep.error(
                    f"{name}: INS{hap} flags ({len(ins_flagged[hap])}) and sidecar insertions ({len(mine)}) disagree; e.g. offsets {diff[:3]}"
                )
    missing = sorted(complex_starts - set(complex_entries))
    if missing:
        rep.error(f"{name}: {len(missing)} COMPLEX runs have no COMPLEX sidecar entry (section 6.6), e.g. (offset, hap) {missing[:3]}")
    if stale_complex:
        rep.warn(f"{name}: {len(stale_complex)} COMPLEX sidecar entries are on bases that are not COMPLEX, e.g. {stale_complex[:3]}")
    rep.checked.append(name)
