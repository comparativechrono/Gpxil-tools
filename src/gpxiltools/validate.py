"""Conformance checks for gpxil files (spec section 9)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import codes as C
from .container import (
    FORMAT_VERSION,
    SIDECAR_TAG,
    STANDARD_PLANES,
    GpxilFile,
    GpxilFormatError,
    decode_sidecar,
    rows_for,
)
from .fasta import sequence_digests

_VALID_REF = np.zeros(256, dtype=bool)
_VALID_REF[0:16] = True
_VALID_REF[0x11:0x20] = True  # soft-masked bases (0x10 alone = masked gap is not valid)
_VALID_STATE = np.zeros(16, dtype=bool)
_VALID_STATE[list(C.VALID_HAP_STATES)] = True
_ALLOWED_COMPRESSION = {1, 5, 8, 32946, 32773, 50000, 34925}  # none, LZW, Deflate, old Deflate, PackBits, ZSTD, LZMA


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


def validate(path, *, check_digests: bool = True) -> Report:
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
        layout = meta.get("layout", {})
        if layout.get("type") != "raster":
            rep.error(f"layout.type must be 'raster', got {layout.get('type')!r}")
        pages = g._tf.pages
        n_pages = len(pages)
        seen_pages: dict[int, tuple[str, str]] = {}
        for entry in meta.get("contigs", []):
            name = entry.get("name")
            if not name or "length" not in entry or "pages" not in entry:
                rep.error(f"contig entry missing name/length/pages: {entry}")
                continue
            contig = g[name]
            L, W = contig.length, contig.width
            rows = rows_for(L, W)
            planes = {}
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
                want = STANDARD_PLANES.get(plane, {}).get("dtype")
                if want and page.dtype != np.dtype(want):
                    rep.error(f"{where}: dtype {page.dtype}, expected {want}")
                if plane in ("ref", "gt", "flags") and int(page.predictor) not in (0, 1):
                    rep.error(f"{where}: predictor {page.predictor!r} not allowed for a categorical plane")
                if int(page.compression) not in _ALLOWED_COMPRESSION:
                    rep.warn(f"{where}: compression {page.compression!r} is outside the recommended set")
                try:
                    d = json.loads(page.description) if idx != 0 else meta
                    if d.get("contig") != name or d.get("plane") != plane:
                        rep.error(f"{where}: page description names {d.get('contig')}/{d.get('plane')}")
                except Exception:
                    rep.error(f"{where}: page description is not gpxil JSON")
                img = page.asarray().reshape(-1)
                if img[L:].any():
                    rep.error(f"{where}: padding pixels after position {L} are not 0")
                planes[plane] = img[:L]
            # plane-specific code checks
            ref = planes.get("ref")
            if ref is not None:
                bad = ~_VALID_REF[ref]
                if bad.any():
                    i = int(np.flatnonzero(bad)[0])
                    rep.error(f"{name}/ref: invalid code 0x{int(ref[i]):02x} at offset {i} ({int(bad.sum())} invalid)")
                if check_digests and "md5" in entry:
                    dig = sequence_digests(ref)
                    for k in ("md5", "refget"):
                        if k in entry and entry[k] != dig[k]:
                            rep.error(f"{name}/ref: {k} {dig[k]} does not match metadata {entry[k]}")
            elif not entry.get("md5") and not entry.get("refget"):
                rep.error(f"{name}: no ref plane and no md5/refget digest identifying the reference")
            gt = planes.get("gt")
            flags = planes.get("flags")
            if gt is not None:
                if flags is None:
                    rep.error(f"{name}: gt plane without flags plane")
                for label, nib in (("hap1", C.hap1(gt)), ("hap2", C.hap2(gt))):
                    bad = ~_VALID_STATE[nib]
                    if bad.any():
                        i = int(np.flatnonzero(bad)[0])
                        rep.error(f"{name}/gt: invalid {label} state 0x{int(nib[i]):x} at offset {i}")
                if ref is not None:
                    rb = np.bitwise_and(ref, 0x0F)
                    for label, nib in (("hap1", C.hap1(gt)), ("hap2", C.hap2(gt))):
                        same = np.isin(nib, C.HAP_SUB_CODES) & (nib == rb)
                        if same.any():
                            rep.error(
                                f"{name}/gt: {label} substitution equals the reference base at {int(same.sum())} positions "
                                f"(first offset {int(np.flatnonzero(same)[0])}); REF must be coded 0"
                            )
            if flags is not None:
                if (flags & C.FLAG_RESERVED_MASK).any():
                    rep.error(f"{name}/flags: reserved bits set")
                side = []
                tag = pages[contig.pages["gt"]].tags.get(SIDECAR_TAG) if "gt" in contig.pages else None
                if tag is not None:
                    try:
                        side = decode_sidecar(bytes(tag.value))
                    except Exception as exc:
                        rep.error(f"{name}: sidecar cannot be decoded: {exc}")
                ins_side = {(e.offset, e.hap) for e in side if e.kind == "INS"}
                for hap, bit in ((1, C.FLAG_INS1), (2, C.FLAG_INS2)):
                    flagged = {(int(i), hap) for i in np.flatnonzero(flags & bit)}
                    mine = {k for k in ins_side if k[1] == hap}
                    if flagged != mine:
                        rep.error(
                            f"{name}: INS{hap} flags ({len(flagged)}) and sidecar insertions ({len(mine)}) disagree; "
                            f"e.g. {sorted(flagged ^ mine)[:3]}"
                        )
                for e in side:
                    if not (0 <= e.offset < L):
                        rep.error(f"{name}: sidecar offset {e.offset} outside contig")
                    if e.kind not in ("INS", "COMPLEX"):
                        rep.warn(f"{name}: unknown sidecar kind {e.kind!r}")
                    if e.kind == "INS" and not set(e.data.upper()) <= set("ACGTN"):
                        rep.error(f"{name}: insertion at {e.offset} has non-ACGTN bases")
            rep.checked.append(name)
        unclaimed = [i for i in range(n_pages) if i not in seen_pages]
        if unclaimed:
            rep.warn(f"pages not listed in metadata (ignored by readers): {unclaimed[:10]}")
    return rep
