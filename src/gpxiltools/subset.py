"""Working copies: some contigs, and some planes, of a gpxil file in a new file; and windows cut out of them.

A contig is self-contained in a gpxil file (its pages and its metadata entry,
spec sections 3 and 9.3), so copying some contigs gives a valid file without
encoding anything again. The strips are copied as they are stored, so a
chromosome comes out of a whole genome, or out of a file on the web, in
about the time it takes to read it, and with little memory.

A region instead of a contig cuts a window (spec 3.1): a contig that holds
that stretch of the sequence and says where it lies in it, so that positions
stay the chromosome's. Its strips are read and encoded again a strip at a
time, so a gene with its surroundings comes out of a whole genome on the web
in a second or two, as a file of a few kilobytes.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterable

from .container import (
    COMPRESSIONS,
    DEFAULT_COMPRESSION,
    STANDARD_PLANES,
    WINDOW_VERSION,
    BigTiffNeeded,
    GpxilFile,
    GpxilWriter,
    compression_of,
    later_version,
    parse_region,
    path_name,
    sort_planes,
)
from .encode import contig_aliases
from .palettes import DEFAULT_PALETTE, PALETTES
from .remote import is_url

#: top-level metadata keys that the writer sets, or that describe the whole input file
_OWN_KEYS = {"gpxil", "generator", "created", "layout", "palette", "reference", "sample", "planes", "contigs", "contig", "plane", "length"}


def _items(contigs: Iterable[str]) -> list[str]:
    """The contigs and regions asked for: commas separate names, but not in a region, where they group digits."""
    out = []
    for item in contigs:
        item = str(item).strip()
        if ":" in item:  # (a region, 'chr17:7,661,779-7,687,546', or a window's name)
            out.append(item)
        else:
            out += [x.strip() for x in item.split(",") if x.strip()]
    return out


def _locate(g: GpxilFile, text: str):
    """A region asked for: (contig, start, end) in the contig's offsets, trying the other spellings of its name."""
    name, s, e = parse_region(text)
    if s is None:
        have = ", ".join(g.contigs[:8]) + (", …" if len(g) > 8 else "")
        raise KeyError(f"no contig {text!r} in {g.path} (it has {have})")
    for n in (name, *contig_aliases(name)):
        try:
            return g.locate(n, s, e)
        except KeyError:
            continue
    return g.locate(name, s, e)  # (raises, saying what the file has)


def plan(g: GpxilFile, contigs: Iterable[str] | None) -> list[tuple[str, int, int]]:
    """What to write: ``(contig, start, end)`` for each contig or region asked for (by name, other name, or a region
    of a sequence the file holds; default every contig), in the order of the file's sequences, and along each by
    position. A whole contig is copied; a stretch of one is cut out as a window."""
    if not contigs:
        return [(n, 0, g[n].length) for n in g.contigs]
    jobs = []
    for n in _items(contigs):
        found = n if n in g else next((a for a in contig_aliases(n) if a in g), None)
        if found is not None:
            c = g[found]
            jobs.append((c.name, 0, c.length))
        elif g.windows_of(n):  # (a sequence of which the file holds windows: all of them)
            jobs += [(w.name, 0, w.length) for w in g.windows_of(n)]
        else:
            c, s, e = _locate(g, n)
            jobs.append((c.name, s, e))
    # (in the order of the file's sequences: each, if it is copied whole, then its windows by where they lie in it)
    seqs: dict[str, int] = {}
    for c in g:
        seqs.setdefault(c.parent, len(seqs))
    order = {name: i for i, name in enumerate(g.contigs)}

    def key(j):
        c = g[j[0]]
        window = c.window is not None or (j[1], j[2]) != (0, c.length)
        return seqs[c.parent], window, c.start + j[1], c.start + j[2], order[j[0]]

    return sorted(dict.fromkeys(jobs), key=key)


def contig_names(g: GpxilFile, contigs: Iterable[str] | None) -> list[str]:
    """The file's names of the contigs asked for (by name or alias; commas separate several), in file order."""
    jobs = plan(g, contigs)
    cut = [j for j in jobs if (j[1], j[2]) != (0, g[j[0]].length)]
    if cut:
        name, s, e = cut[0]
        raise ValueError(f"{g[name].region(s, e)} is a region, not a contig")
    return list(dict.fromkeys(j[0] for j in jobs))


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
    """Copy contigs (by default all) of the gpxil file ``path`` into a new gpxil file ``out``, or cut windows of them.

    ``contigs`` are contigs, by name or alias, or regions (``chr17:7,561,779-7,787,546``): a region of a contig
    is cut out as a window (spec 3.1), a contig named for the stretch it holds that keeps its positions in the
    chromosome. ``planes`` limits the planes copied (by default each contig's own); ``include_ref=False`` leaves
    out the reference plane, which the contigs' digests then identify, as with ``gpxil encode --no-ref``. A
    ``reference`` (FASTA or gpxil file, path or URL) supplies the reference plane of contigs stored without one,
    when ``planes`` asks for ``ref``. ``compression`` defaults to the input's, and then the strips of whole
    contigs are copied as they are stored, without being decoded; another compression, or a ``level``, encodes
    them again (a window's are always encoded). ``path`` can be a URL. Returns a summary.
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
        jobs: list[tuple[str, int, int, list[str]]] = []
        for n, s, e in plan(g, contigs):
            c = g[n]
            keep = [p for p in (want if want is not None else c.planes) if p in c.pages or (p == "ref" and want is not None)]
            if not include_ref:
                keep = [p for p in keep if p != "ref"]
            if not keep:
                log(f"  {c.region(s, e) if (s, e) != (0, c.length) else n}: none of the planes asked for: skipped")
                continue
            jobs.append((n, s, e, sort_planes(keep)))
        if want is not None:
            missing = [p for p in want if p != "ref" and not any(p in g[n].pages for n, *_ in jobs)]
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
    cuts = [j for j in jobs if (j[1], j[2]) != (0, g[j[0]].length)]
    source_version = str(meta["gpxil"])
    # (the version whose rules for calls, section 6, the planes follow: the source's, or, for a copy of a copy, the
    # version its own subset record gives)
    rec = meta.get("subset")
    rules = str(rec["gpxil"]) if isinstance(rec, dict) and rec.get("gpxil") else source_version
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
        # a copy keeps the version of its source, any 0.x: its planes were made by that version's rules; but a file
        # with windows is gpxil 0.3 at least, and then its subset record says which version its planes follow
        version=later_version(source_version, WINDOW_VERSION) if cuts else source_version,
    )
    writer.plane_info = {p: {k: v for k, v in (d or {}).items() if k != "dtype"} for p, d in (meta.get("planes") or {}).items()}
    try:
        stored = encoded = 0
        for name, s, e, keep in jobs:
            c = g[name]
            if (s, e) == (0, c.length):
                info = writer.copy_contig(c, keep, recompress=level is not None)
                what = f"{c.length:,} bp" + (f", a window of {c.parent}" if c.window else "")
            else:
                info = writer.cut_contig(c, s, e, keep)
                what = f"{e - s:,} bp, a window of {c.parent} cut out of {c.name}"
            stored += writer.last_copy["stored"]
            encoded += writer.last_copy["encoded"]
            log(f"  {info.name}: {what}, {','.join(info.pages)}  [{time.time() - t0:.0f}s]")
        record = {
            "of": path_name(g.path),
            "gpxil": rules,
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
        "windows": sum(1 for c in writer.contigs if "window" in c.extra),
        "planes": written,
        "bases": sum(c.length for c in writer.contigs),
        "compression": compression,
        "pages_copied_as_stored": stored,
        "pages_encoded": encoded,
        "gpxil": writer.version,
    }
