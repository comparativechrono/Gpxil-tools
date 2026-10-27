"""High-level FASTA (+ VCF) -> gpxil conversion, one chromosome at a time."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterable

import numpy as np

from .container import DEFAULT_COMPRESSION, DEFAULT_WIDTH, BigTiffNeeded, GpxilWriter, check_contig_name
from .fasta import iter_fasta, sequence_digests
from .palettes import DEFAULT_PALETTE
from .vcf import TempSplit, encode_calls, load_bed, read_header

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

    Returns a summary dict (also stored in the file's metadata). With
    ``bigtiff=None`` the file is classic TIFF unless it would pass 4 GiB, in
    which case the encode restarts as BigTIFF. On any error the partial
    output is removed.
    """
    log = log or (lambda msg: print(msg, file=sys.stderr))
    haploid_contigs = set(haploid_contigs)
    if sex not in (None, "XX", "XY"):
        raise ValueError("sex must be 'XX' or 'XY'")
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
        split = TempSplit(vcf, sample_index).__enter__()
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
    ref_meta = {
        "name": reference_name or os.path.basename(str(fasta)).split(".fa")[0],
        "source": os.path.basename(str(fasta)),
    }
    totals: dict = {"contigs": 0, "bases": 0, "records": 0, "conflicts": 0, "ref_mismatches": 0, "skipped": 0}
    used_vcf: set[str] = set()
    t0 = time.time()
    writer = GpxilWriter(
        out, width=width, compression=compression, level=level, palette=palette, reference=ref_meta, sample=sample_meta, bigtiff=bigtiff
    )
    try:
        for rec in iter_fasta(fasta, invalid=invalid):
            name = ucsc_name(rec.name) if ucsc_names else rec.name
            if contigs is not None and not (name in contigs or rec.name in contigs or any(a in contigs for a in contig_aliases(name))):
                continue
            if rec.length == 0:
                totals.setdefault("skipped_empty_contigs", []).append(name)
                log(f"  {name}: empty sequence, skipped")
                continue
            check_contig_name(name)
            aliases = [a for a in dict.fromkeys([rec.name, *contig_aliases(name)]) if a != name]
            entry: dict = {"aliases": aliases, **sequence_digests(rec.codes)}
            if rec.converted:
                entry["invalid_characters_converted_to_N"] = rec.converted
            data = {"ref": rec.codes} if include_ref else {}
            side = None
            if split is not None:
                vname = _match(name, vcf_contigs) or _match(rec.name, vcf_contigs)
                callable_mask = load_bed(callable_bed, name, rec.length, [rec.name, *aliases]) if callable_bed else None
                if vname is None and callable_mask is None:
                    # no calls and no callable regions: nothing is known about the sample here (spec 6.1)
                    if data:
                        writer.add_contig(name, data, length=rec.length, **entry)
                        totals["contigs"] += 1
                        totals["bases"] += rec.length
                        log(f"  {name}: {rec.length:,} bp, no calls (reference only)  [{time.time() - t0:.0f}s]")
                    continue
                if callable_bed and callable_mask is None:
                    log(f"  note: {os.path.basename(str(callable_bed))} has no intervals on {name}; bases without calls are UNKNOWN")
                    callable_mask = np.zeros(rec.length, dtype=bool)
                haploid_mask = load_bed(haploid_bed, name, rec.length, [rec.name, *aliases]) if haploid_bed else None
                if name in haploid_contigs or rec.name in haploid_contigs or any(a in haploid_contigs for a in aliases):
                    haploid_mask = np.ones(rec.length, dtype=bool)
                if sex == "XY" and haploid_mask is None:
                    haploid_mask = male_haploid_mask(name, rec.length)
                calls = split.calls(vname) if vname else iter(())
                if vname:
                    used_vcf.add(vname)
                var_planes, side, stats, ps_ids = encode_calls(
                    rec.codes,
                    calls,
                    contig=name,
                    callable_mask=callable_mask,
                    haploid_mask=haploid_mask,
                    planes=[p for p in plane_set if p != "ref"],
                )
                n_callable = None if callable_mask is None else int(callable_mask.sum())
                del callable_mask, haploid_mask  # free before the planes are written
                data.update(var_planes)
                entry["vcf_contig"] = vname
                entry["ploidy"] = stats.ploidy
                entry["encoding"] = stats.as_dict()
                if n_callable is not None:
                    entry["callable_bases"] = n_callable
                if ps_ids:
                    entry["phase_sets"] = len(ps_ids)
                    synthetic = {str(v): k for k, v in ps_ids.items() if str(v) != k}
                    if synthetic:
                        entry["phase_set_ids"] = synthetic  # ps value -> original VCF PS
                totals["records"] += stats.records
                totals["conflicts"] += stats.conflicts
                totals["ref_mismatches"] += stats.ref_mismatches
                totals["skipped"] += stats.skipped_out_of_range + stats.polyploid_skipped
                if stats.examples:
                    entry["encoding"]["examples"] = stats.examples[:5]
            if not data:
                continue  # --no-ref and nothing else for this contig
            writer.add_contig(name, data, length=rec.length, sidecar=side, **entry)
            totals["contigs"] += 1
            totals["bases"] += rec.length
            log(
                f"  {name}: {rec.length:,} bp"
                + (f", {entry['encoding']['records']:,} records" if "encoding" in entry else "")
                + f"  [{time.time() - t0:.0f}s]"
            )
        if split is not None:
            unused = [c for c in vcf_contigs if c not in used_vcf]
            if unused:
                totals["vcf_contigs_without_reference"] = unused
                log(f"  note: VCF contigs not in the FASTA (skipped): {', '.join(unused[:10])}")
        writer.extra_metadata["summary"] = totals
        writer.close()
    except BaseException:
        writer.abort()
        raise
    totals["seconds"] = round(time.time() - t0, 1)
    totals["bytes"] = os.path.getsize(out)
    return totals
