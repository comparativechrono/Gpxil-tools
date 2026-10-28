"""Working copies: some contigs, and some planes, of a gpxil file in a new file.

A contig is self-contained in a gpxil file (its pages and its metadata entry,
spec sections 3 and 9.3), so copying some contigs gives a valid file without
encoding anything again. The strips are copied as they are stored, so a
chromosome comes out of a whole genome, or out of a file on the web, in
about the time it takes to read it, and with little memory.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Callable, Iterable

from .container import (
    COMPRESSIONS,
    DEFAULT_COMPRESSION,
    STANDARD_PLANES,
    BigTiffNeeded,
    GpxilFile,
    GpxilWriter,
    compression_of,
    sort_planes,
)
from .encode import contig_aliases
from .palettes import DEFAULT_PALETTE, PALETTES
from .remote import is_url

#: top-level metadata keys that the writer sets, or that describe the whole input file
_OWN_KEYS = {"gpxil", "generator", "created", "layout", "palette", "reference", "sample", "planes", "contigs", "contig", "plane", "length"}
_REGION = re.compile(r"^(.+):[\d,]+(-[\d,]+)?$")


def contig_names(g: GpxilFile, contigs: Iterable[str] | None) -> list[str]:
    """The file's names of the contigs asked for (by name or alias; commas separate several), in file order."""
    if not contigs:
        return list(g.contigs)
    out = []
    for item in contigs:
        for n in (x.strip() for x in str(item).split(",")):
            if not n:
                continue
            found = n if n in g else next((a for a in contig_aliases(n) if a in g), None)
            if found is not None:
                out.append(g[found].name)
                continue
            if _REGION.match(n):
                raise ValueError(
                    f"{n} is a region: gpxil subset copies whole contigs (gpxil 0.1 has no files of part of a contig);"
                    " gpxil fasta, vcf and render take --region"
                )
            have = ", ".join(g.contigs[:8]) + (", …" if len(g) > 8 else "")
            raise KeyError(f"no contig {n!r} in {g.path} (it has {have})")
    order = {name: i for i, name in enumerate(g.contigs)}
    return sorted(dict.fromkeys(out), key=order.__getitem__)


def subset(
    path,
    out,
    contigs: Iterable[str] | None = None,
    *,
    planes: Iterable[str] | None = None,
    include_ref: bool = True,
    reference=None,
    compression: str | None = None,
    level: int | None = None,
    bigtiff: bool | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Copy contigs (by default all) of the gpxil file ``path`` into a new gpxil file ``out``.

    ``planes`` limits the planes copied (by default each contig's own);
    ``include_ref=False`` leaves out the reference plane, which the contigs'
    digests then identify, as with ``gpxil encode --no-ref``. A ``reference``
    (FASTA or gpxil file, path or URL) supplies the reference plane of
    contigs stored without one, when ``planes`` asks for ``ref``.
    ``compression`` defaults to the input's, and then the strips are copied
    as they are stored, without being decoded; another compression, or a
    ``level``, encodes them again. ``path`` can be a URL. Returns a summary.
    """
    log = log or (lambda msg: print(msg, file=sys.stderr))
    if compression is not None and compression not in COMPRESSIONS:
        raise ValueError(f"unknown compression {compression!r}; choose from {', '.join(COMPRESSIONS)}")
    if not is_url(path) and os.path.exists(out) and os.path.samefile(path, out):
        raise ValueError(f"{out} is the input file: write the copy to another file")
    want = None if planes is None else list(dict.fromkeys(p.strip() for p in planes if p.strip()))
    if want is not None:
        bad = [p for p in want if p not in STANDARD_PLANES and not p.startswith("x-")]
        if bad:
            raise ValueError(f"unknown plane {bad[0]!r}: the planes are {', '.join(STANDARD_PLANES)} and extensions x-…")
        if not want:
            raise ValueError("no planes asked for")
        if "ref" in want and not include_ref:
            raise ValueError("the planes asked for include ref, which include_ref=False (--no-ref) leaves out")
    t0 = time.time()
    with GpxilFile(path, reference=reference) as g:
        names = contig_names(g, contigs)
        jobs: list[tuple[str, list[str]]] = []
        for n in names:
            c = g[n]
            keep = [p for p in (want if want is not None else c.planes) if p in c.pages or (p == "ref" and want is not None)]
            if not include_ref:
                keep = [p for p in keep if p != "ref"]
            if not keep:
                log(f"  {n}: none of the planes asked for: skipped")
                continue
            jobs.append((n, sort_planes(keep)))
        if want is not None:
            missing = [p for p in want if p != "ref" and not any(p in g[n].pages for n in names)]
            if missing:
                raise ValueError(f"no contig asked for has a {missing[0]} plane")
        if not jobs:
            raise ValueError("nothing to copy")
        first = g[jobs[0][0]]
        stored = compression_of(first._page(first.planes[0]))
        comp = compression or stored or DEFAULT_COMPRESSION
        try:
            info = _write(g, out, jobs, comp, level, bigtiff, log, t0)
        except BigTiffNeeded:
            log("  the file will be larger than 4 GiB: starting again as BigTIFF")
            info = _write(g, out, jobs, comp, level, True, log, t0)
    info["seconds"] = round(time.time() - t0, 1)
    info["bytes"] = os.path.getsize(out)
    return info


def _write(g: GpxilFile, out, jobs, compression: str, level: int | None, bigtiff: bool | None, log, t0: float) -> dict:
    meta = g.metadata
    palette = meta.get("palette") if meta.get("palette") in PALETTES else DEFAULT_PALETTE
    writer = GpxilWriter(
        out,
        width=int((meta.get("layout") or {}).get("width") or g[jobs[0][0]].width),
        compression=compression,
        level=level,
        bigtiff=bigtiff,
        palette=palette,
        reference=meta.get("reference") or {},
        sample=meta.get("sample"),
        metadata={k: v for k, v in meta.items() if k not in _OWN_KEYS and k not in ("summary", "subset")},
        require_codec=False,  # (copy_contig checks the codec where strips must be encoded again)
    )
    writer.plane_info = {p: {k: v for k, v in (d or {}).items() if k != "dtype"} for p, d in (meta.get("planes") or {}).items()}
    try:
        stored = encoded = 0
        for name, keep in jobs:
            c = g[name]
            info = writer.copy_contig(c, keep, recompress=level is not None)
            stored += writer.last_copy["stored"]
            encoded += writer.last_copy["encoded"]
            log(f"  {name}: {c.length:,} bp, {','.join(info.pages)}  [{time.time() - t0:.0f}s]")
        record = {
            "of": str(g.path).rstrip("/").rsplit("/", 1)[-1] if is_url(g.path) else os.path.basename(str(g.path)),
            "generator": meta.get("generator"),
            "created": meta.get("created"),
            "contigs": [c.name for c in writer.contigs],
        }
        for key in ("summary", "subset"):
            if key in meta:
                record["its_" + key] = meta[key]
        writer.extra_metadata["subset"] = record
        written = sort_planes({p for c in writer.contigs for p in c.pages})
        writer.close()
    except BaseException:
        writer.abort()
        raise
    return {
        "contigs": len(writer.contigs),
        "names": [c.name for c in writer.contigs],
        "planes": written,
        "bases": sum(c.length for c in writer.contigs),
        "compression": compression,
        "pages_copied_as_stored": stored,
        "pages_encoded": encoded,
    }
