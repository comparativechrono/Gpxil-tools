"""High-level FASTA (+ VCF) -> gpxil conversion, one chromosome at a time."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterable

import numpy as np

from .container import DEFAULT_COMPRESSION, DEFAULT_WIDTH, BigTiffNeeded, GpxilWriter, check_contig_name, region_text
from .fasta import FastaRecord, iter_fasta, sequence_digests
from .palettes import DEFAULT_PALETTE
from .vcf import TempSplit, encode_calls, load_bed, phased_ps, ps_numbering, read_header, record_span, window_entries, window_stretch

# GRCh38 pseudoautosomal regions, 0-based half-open (GRC definitions).
GRCH38_PAR = {
    "X": (156_040_895, [(10_000, 2_781_479), (155_701_382, 156_030_895)]),
    "Y": (57_227_415, [(10_000, 2_781_479), (56_887_902, 57_217_415)]),
}
OPTIONAL_PLANES = (("gq", "GQ"), ("dp", "DP"), ("ps", "PS"))


def contig_aliases(name: str) -> list[str]:
    """Other common spellings of a contig name ('21' <-> 'chr21', 'MT' <-> 'chrM')."""
    out = []
    if name.startswith("chr"):
        out.append(name[3:])
    else:
        out.append("chr" + name)
    if name in ("MT", "M", "chrM", "chrMT"):
        out += ["MT", "M", "chrM", "chrMT"]
    return [a for a in dict.fromkeys(out) if a != name]


_PRIMARY = {str(i) for i in range(1, 23)} | {"X", "Y"}


def ucsc_name(name: str) -> str:
    """Ensembl/NCBI chromosome names to UCSC style; other names are kept."""
    if name in ("MT", "M"):
        return "chrM"
    return "chr" + name if name in _PRIMARY else name


def _match(name: str, available: Iterable[str]) -> str | None:
    avail = set(available)
    if name in avail:
        return name
    for a in contig_aliases(name):
        if a in avail:
            return a
    return None


def male_haploid_mask(name: str, length: int) -> np.ndarray | None:
    """Haploid (non-PAR) mask for chrX/chrY of a male sample on GRCh38."""
    key = name[3:] if name.startswith("chr") else name
    if key not in GRCH38_PAR:
        return None
    expected, pars = GRCH38_PAR[key]
    if length != expected:
        raise ValueError(f"{name} has length {length}, not the GRCh38 length {expected}; pass a haploid-regions BED instead")
    mask = np.ones(length, dtype=bool)
    for s, e in pars:
        mask[s:e] = False
    return mask


def reference_records(path, *, invalid: str = "error", wanted: Callable[[str], bool] | None = None):
    """The reference's sequences, one :class:`FastaRecord` per contig.

    ``path`` is a FASTA file, or a gpxil file (a path or URL) whose reference
    plane is read contig by contig; with ``wanted``, a gpxil file's contigs
    that it rejects (by name or alias) are skipped without being read.
    """
    from .container import GpxilFile, GpxilFormatError, is_tiff

    if not is_tiff(path):
        yield from iter_fasta(path, invalid=invalid)
        return
    with GpxilFile(path) as g:
        for c in g:
            if c.window is not None:
                # (a window holds a stretch of its sequence: a VCF's calls on the sequence can't all be placed on it)
                if wanted is None or any(wanted(n) for n in (c.parent, *c.parent_aliases)):
                    raise GpxilFormatError(
                        f"{path}: {c.name} is a window of {c.parent} (a stretch of it), so it can't be the reference:"
                        f" give all of {c.parent} (a FASTA file, or a gpxil file of it), with --region for a stretch"
                    )
                continue
            names = [c.name, *c.entry.get("aliases", [])]
            if wanted is not None and not any(wanted(n) for n in names):
                continue
            if "ref" not in c.pages:
                raise GpxilFormatError(f"{path}: {c.name} has no reference plane, so it cannot be a reference")
            yield FastaRecord(name=c.name, description="", codes=c.read("ref"))


def _reference_meta(path, reference_name: str | None) -> dict:
    from .container import GpxilFile, is_tiff
    from .remote import is_url

    base = path.rstrip("/").rsplit("/", 1)[-1] if is_url(path) else os.path.basename(str(path))
    if is_tiff(path):
        with GpxilFile(path) as g:
            ref = g.metadata.get("reference") or {}
        return {"name": reference_name or ref.get("name") or base.split(".")[0], "source": ref.get("source") or base, "via": base}
    return {"name": reference_name or base.split(".fa")[0], "source": base}


def encode(
    fasta: str | os.PathLike,
    out: str | os.PathLike,
    *,
    vcf: str | os.PathLike | None = None,
    sample: str | None = None,
    callable_bed: str | os.PathLike | None = None,
    haploid_bed: str | os.PathLike | None = None,
    haploid_contigs: Iterable[str] = (),
    sex: str | None = None,
    contigs: Iterable[str] | None = None,
    regions: Iterable[tuple[str, int | None, int | None]] | None = None,
    ucsc_names: bool = False,
    include_ref: bool = True,
    planes: Iterable[str] | None = None,
    width: int = DEFAULT_WIDTH,
    compression: str = DEFAULT_COMPRESSION,
    level: int | None = None,
    palette: str = DEFAULT_PALETTE,
    reference_name: str | None = None,
    invalid: str = "error",
    bigtiff: bool | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Encode a reference FASTA, optionally with one sample's VCF, into a gpxil file.

    The reference can also be a gpxil file (a path or URL) with the
    reference plane, such as the Gpxil browser's GRCh38 files.

    ``regions``, ``(contig, start, end)`` with 0-based positions and the end
    excluded (None and None: the whole contig), replace ``contigs``: each
    stretch becomes a window (spec 3.1), a contig named for it
    (``chr17:7561779-7787546``) that keeps the chromosome's positions and
    holds what the whole chromosome's planes would hold there. Only the calls
    that join up with it through overlapping spans are converted.

    Returns a summary dict (also stored in the file's metadata). With
    ``bigtiff=None`` the file is classic TIFF unless it would pass 4 GiB, in
    which case the encode restarts as BigTIFF. On any error the partial
    output is removed.
    """
    log = log or (lambda msg: print(msg, file=sys.stderr))
    haploid_contigs = set(haploid_contigs)
    if sex not in (None, "XX", "XY"):
        raise ValueError("sex must be 'XX' or 'XY'")
    if contigs is not None and regions is not None:
        raise ValueError("give contigs or regions, not both")
    split = None
    sample_index = None
    if vcf is not None:
        hdr = read_header(vcf)
        if not hdr.samples:
            raise ValueError(f"{vcf}: VCF has no sample columns")
        if sample is None:
            sample_index = 0
        elif sample in hdr.samples:
            sample_index = hdr.samples.index(sample)
        else:
            raise ValueError(f"sample {sample!r} not in {vcf} (samples: {', '.join(hdr.samples[:10])})")
        sample = hdr.samples[sample_index]
        log(f"splitting {os.path.basename(str(vcf))} by contig for sample {sample} ...")
        asked = None if contigs is None and regions is None else {r[0] for r in regions} if regions is not None else set(contigs)
        if asked is not None:
            # (only the contigs asked for are kept, under any of their names)
            asked = {x for n in asked for x in (n, ucsc_name(n), *contig_aliases(n), *contig_aliases(ucsc_name(n)))}

        def keep(chrom: str) -> bool:
            return asked is None or bool({chrom, ucsc_name(chrom), *contig_aliases(chrom)} & asked)

        split = TempSplit(vcf, sample_index, keep=keep).__enter__()
    try:
        args = dict(
            vcf=vcf,
            sample=sample,
            split=split,
            callable_bed=callable_bed,
            haploid_bed=haploid_bed,
            haploid_contigs=haploid_contigs,
            sex=sex,
            contigs=None if contigs is None else set(contigs),
            regions=None if regions is None else [tuple(r) for r in regions],
            ucsc_names=ucsc_names,
            include_ref=include_ref,
            planes=None if planes is None else list(planes),
            width=width,
            compression=compression,
            level=level,
            palette=palette,
            reference_name=reference_name,
            invalid=invalid,
            log=log,
        )
        try:
            return _encode_once(fasta, out, bigtiff=bigtiff, **args)
        except BigTiffNeeded:
            log("  the file will be larger than 4 GiB: starting again as BigTIFF")
            return _encode_once(fasta, out, bigtiff=True, **args)
    finally:
        if split is not None:
            split.__exit__(None, None, None)


def _haploid_description(haploid_bed, haploid_contigs, sex) -> str | None:
    parts = []
    if haploid_bed:
        parts.append(os.path.basename(str(haploid_bed)))
    if haploid_contigs:
        parts.append("contigs " + ",".join(sorted(haploid_contigs)))
    if sex == "XY":
        parts.append("GRCh38 chrX and chrY outside the pseudoautosomal regions (sex XY)")
    return "; ".join(parts) or None


def _encode_once(
    fasta,
    out,
    *,
    bigtiff,
    vcf,
    sample,
    split,
    callable_bed,
    haploid_bed,
    haploid_contigs,
    sex,
    contigs,
    regions,
    ucsc_names,
    include_ref,
    planes,
    width,
    compression,
    level,
    palette,
    reference_name,
    invalid,
    log,
) -> dict:
    plane_set = ["ref"] if include_ref else []
    sample_meta = None
    vcf_contigs: list[str] = []
    if split is not None:
        if planes is None:
            plane_set += ["gt", "flags"] + [p for p, key in OPTIONAL_PLANES if key in split.format_keys]
        else:
            plane_set += [p for p in planes if p not in plane_set]
        vcf_contigs = list(split.index)
        sample_meta = {
            "id": sample,
            "source": os.path.basename(str(vcf)),
            "callable_regions": os.path.basename(str(callable_bed)) if callable_bed else None,
            "haploid_regions": _haploid_description(haploid_bed, haploid_contigs, sex),
            "sex": sex,
        }
    ref_meta = _reference_meta(fasta, reference_name)
    totals: dict = {
        "contigs": 0,
        "bases": 0,
        "records": 0,
        "conflicts": 0,
        "ref_mismatches": 0,
        "skipped": 0,
        "overlapping_calls_reassigned": 0,
    }
    if regions is not None:
        totals["windows"] = 0
    used_vcf: set[str] = set()
    found: set[int] = set()  # (the regions whose contig the reference has)
    t0 = time.time()
    writer = GpxilWriter(
        out, width=width, compression=compression, level=level, palette=palette, reference=ref_meta, sample=sample_meta, bigtiff=bigtiff
    )
    try:

        def names_of(n: str) -> set[str]:
            u = ucsc_name(n) if ucsc_names else n
            return {n, u, *contig_aliases(u)}

        def wanted(n: str) -> bool:
            if regions is not None:
                return any(r[0] in names_of(n) for r in regions)
            return contigs is None or bool(names_of(n) & contigs)

        for rec in reference_records(fasta, invalid=invalid, wanted=wanted):
            name = ucsc_name(rec.name) if ucsc_names else rec.name
            known_as = names_of(rec.name) | {name}
            if regions is not None:
                # (whole contigs, and windows: their [start, end), in order)
                targets = []
                for k, (n, s, e) in enumerate(regions):
                    if n in known_as:
                        found.add(k)
                        targets.append(None if e is None and not s else (s or 0, rec.length if e is None else e))
                targets = list(dict.fromkeys(targets))
            elif contigs is None or known_as & contigs:
                targets = [None]
            else:
                targets = []
            if not targets:
                continue
            if rec.length == 0:
                totals.setdefault("skipped_empty_contigs", []).append(name)
                log(f"  {name}: empty sequence, skipped")
                continue
            check_contig_name(name)
            aliases = [a for a in dict.fromkeys([rec.name, *contig_aliases(name)]) if a != name]
            digests = sequence_digests(rec.codes)
            for target in targets:
                if target is not None and not (0 <= target[0] < target[1] <= rec.length):
                    raise ValueError(f"{name}:{target[0] + 1:,}-{target[1]:,} is outside {name} ({rec.length:,} bases)")
            if any(target in (None, (0, rec.length)) for target in targets):
                targets = [None, *(x for x in targets if x not in (None, (0, rec.length)))]
            # what the VCF and the regions files say of this contig, for the contig and its windows alike
            calls = None
            if split is not None:
                vname = _match(name, vcf_contigs) or _match(rec.name, vcf_contigs)
                callable_mask = load_bed(callable_bed, name, rec.length, [rec.name, *aliases]) if callable_bed else None
                if vname is None and (callable_mask is None or not callable_mask.any()):
                    # no calls and no callable interval: nothing is known about the sample here (spec 6.1)
                    calls = False
                else:
                    if callable_bed and callable_mask is None:
                        log(f"  note: {os.path.basename(str(callable_bed))} has no intervals on {name}; bases without calls are UNKNOWN")
                        callable_mask = np.zeros(rec.length, dtype=bool)
                    haploid_mask = load_bed(haploid_bed, name, rec.length, [rec.name, *aliases]) if haploid_bed else None
                    if name in haploid_contigs or rec.name in haploid_contigs or any(a in haploid_contigs for a in aliases):
                        haploid_mask = np.ones(rec.length, dtype=bool)
                    if sex == "XY" and haploid_mask is None:
                        haploid_mask = male_haploid_mask(name, rec.length)
                    if vname:
                        used_vcf.add(vname)
                    calls = (vname, callable_mask, haploid_mask)
                    got: list = []

                    def scan(vname=vname, n=rec.length, got=got):
                        """the spans of the contig's calls, and the values of its phase-set identifiers (spec 5.5):
                        worked out once, for all its windows"""
                        if not got:
                            spans, keys = [], []
                            for c in split.calls(vname):
                                spans.append(record_span(c, n))
                                k = phased_ps(c, n)
                                if k is not None:
                                    keys.append(k)
                            got.append((spans, ps_numbering(keys)))
                        return got[0]

            for target in targets:
                if target is None:
                    label, length = name, rec.length
                    entry: dict = {"aliases": aliases, **digests}
                    data = {"ref": rec.codes} if include_ref else {}
                else:
                    s, e = target
                    label, length = region_text(name, s, e), e - s
                    window = {"of": name, "start": s, "length": rec.length, **digests}
                    if aliases:
                        window["aliases"] = aliases
                    entry = {"window": window, **sequence_digests(rec.codes[s:e])}
                    data = {"ref": rec.codes[s:e]} if include_ref else {}
                if rec.converted:
                    entry["invalid_characters_converted_to_N"] = rec.converted
                side = None
                if calls is False and not data:
                    totals.setdefault("contigs_without_data", []).append(label)
                    log(f"  {label}: no calls, and --no-ref leaves nothing to store: skipped")
                    continue
                if calls:
                    vname, callable_mask, haploid_mask = calls
                    var_planes, side, stats, ps_ids = _convert(
                        split, vname, rec, name, target, callable_mask, haploid_mask, [p for p in plane_set if p != "ref"], scan
                    )
                    data.update(var_planes)
                    entry["vcf_contig"] = vname
                    gt = var_planes["gt"]
                    absent2 = int(np.count_nonzero((gt & 0x0F) == 0x0A))
                    entry["ploidy"] = 1 if absent2 == length else 2 if absent2 == 0 else "mixed"
                    entry["encoding"] = stats.as_dict()
                    if callable_mask is not None:
                        a, b = (0, rec.length) if target is None else target
                        entry["callable_bases"] = int(np.count_nonzero(callable_mask[a:b]))
                    if "ps" in var_planes:
                        # (the phase sets on the bases: of a contig, every one used; of a window, those in it)
                        here = set(ps_ids.values()) if target is None else {int(v) for v in np.unique(var_planes["ps"]) if v}
                        if here:
                            entry["phase_sets"] = len(here)
                        synthetic = {str(v): k for k, v in ps_ids.items() if str(v) != k and v in here}
                        if synthetic:
                            entry["phase_set_ids"] = synthetic  # ps value -> original VCF PS
                    totals["records"] += stats.records
                    totals["conflicts"] += stats.conflicts
                    totals["overlapping_calls_reassigned"] += stats.overlapping_calls_reassigned
                    totals["ref_mismatches"] += stats.ref_mismatches
                    totals["skipped"] += stats.skipped_out_of_range + stats.polyploid_skipped
                    if stats.examples:
                        entry["encoding"]["examples"] = stats.examples[:5]
                if not data:
                    continue  # --no-ref and nothing else for this contig
                writer.add_contig(label, data, length=length, sidecar=side, **entry)
                totals["contigs"] += 1
                totals["bases"] += length
                if target is not None:
                    totals["windows"] += 1
                log(
                    f"  {label}: {length:,} bp"
                    + (f", a window of {name}" if target is not None else "")
                    + (
                        f", {entry['encoding']['records']:,} records"
                        if "encoding" in entry
                        else ", no calls (reference only)"
                        if calls is False
                        else ""
                    )
                    + f"  [{time.time() - t0:.0f}s]"
                )
            calls = None  # (free the masks before the next contig)
        if regions is not None:
            missing = [regions[k][0] for k in range(len(regions)) if k not in found]
            if missing:
                raise ValueError(
                    f"the reference has no contig {missing[0]!r}" + (f" (nor {', '.join(missing[1:4])})" if len(missing) > 1 else "")
                )
        if split is not None:
            unused = [c for c in vcf_contigs if c not in used_vcf and wanted(c)]  # contigs left out by --contigs are not news
            if unused:
                totals["vcf_contigs_without_reference"] = unused
                log(f"  note: VCF contigs not in the reference (skipped): {', '.join(unused[:10])}")
            if not any("gt" in c.pages for c in writer.contigs):
                # a file without genotype planes describes no sample (spec 9.1)
                writer.sample = None
                log("  note: no contig received genotype planes, so the file is reference only")
        writer.extra_metadata["summary"] = totals
        writer.close()
    except BaseException:
        writer.abort()
        raise
    totals["seconds"] = round(time.time() - t0, 1)
    totals["bytes"] = os.path.getsize(out)
    return totals


def _convert(split, vname, rec, name, target, callable_mask, haploid_mask, planes, scan):
    """The sample's planes, sidecar, counts and phase sets of a contig, or of a window ``target`` = (start, end) of it
    (spec 3.1): converted from the calls that join up with the window through overlapping spans, alone, over the
    stretch they cover, and cut to the window, which gives what converting the whole contig would. ``scan()`` gives
    the spans of the contig's calls (:func:`record_span`) and the values of its phase-set identifiers
    (:func:`ps_numbering`), worked out once for all its windows."""
    n = rec.length
    if target is None:
        calls = split.calls(vname) if vname else iter(())
        return encode_calls(rec.codes, calls, contig=name, callable_mask=callable_mask, haploid_mask=haploid_mask, planes=planes)
    s, e = target
    spans, ps_values = scan() if vname else ((), {})
    a, b, keep = window_stretch(spans, s, e)
    calls = split.calls(vname, keep) if vname else iter(())
    var_planes, side, stats, ps_ids = encode_calls(
        rec.codes[a:b],
        calls,
        contig=name,
        callable_mask=None if callable_mask is None else callable_mask[a:b],
        haploid_mask=None if haploid_mask is None else haploid_mask[a:b],
        planes=planes,
        origin=a,
        contig_length=n,
        ps_values=ps_values,
    )
    cut = {p: np.ascontiguousarray(v[s - a : e - a]) for p, v in var_planes.items()}
    side = window_entries(side, s - a, e - a, int(var_planes["gt"][s - a])) if side else side
    return cut, side, stats, ps_ids
