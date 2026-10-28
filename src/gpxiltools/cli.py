"""Command-line interface: ``gpxil <command> ...``."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from . import __version__


def _region(text: str | None):
    """'chr21', 'chr21:5,000,001-5,100,000' -> (contig, start0, end) (end may be None)."""
    if text is None:
        return None
    m = re.fullmatch(r"([^:]+)(?::([\d,]+)-([\d,]+))?", text.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad region {text!r}; use CONTIG or CONTIG:START-END (1-based)")
    name, s, e = m.groups()
    if s is None:
        return name, 0, None
    s, e = int(s.replace(",", "")), int(e.replace(",", ""))
    if s < 1 or e < s:
        raise argparse.ArgumentTypeError(f"bad interval in {text!r}")
    return name, s - 1, e


class _TextAsBytes:
    """Bytes written to a text stream: for consoles whose sys.stdout has no .buffer (notebooks, the browser)."""

    def __init__(self, text):
        self._text = text

    def write(self, data: bytes) -> int:
        self._text.write(data.decode("utf-8"))
        return len(data)

    def flush(self) -> None:
        self._text.flush()


def _stdout_bytes():
    return getattr(sys.stdout, "buffer", None) or _TextAsBytes(sys.stdout)


def _open(a):
    """Open the input file, with the reference (FASTA or gpxil) for files written with --no-ref."""
    from .container import GpxilFile

    return GpxilFile(a.file, reference=getattr(a, "reference", None))


def cmd_encode(a) -> int:
    from .encode import encode

    out = a.output or (os.path.basename(a.vcf or a.fasta).split(".")[0] + ".gpxil.tif")
    summary = encode(
        a.fasta,
        out,
        vcf=a.vcf,
        sample=a.sample,
        callable_bed=a.callable,
        haploid_bed=a.haploid,
        haploid_contigs=a.haploid_contig or (),
        sex=a.sex,
        contigs=a.contigs.split(",") if a.contigs else None,
        ucsc_names=a.ucsc_names,
        include_ref=not a.no_ref,
        planes=a.planes.split(",") if a.planes else None,
        width=a.width,
        compression=a.compression,
        level=a.level,
        palette=a.palette,
        reference_name=a.reference_name,
        invalid="N" if a.invalid_to_n else "error",
        bigtiff=True if a.bigtiff else None,
    )
    mb = summary["bytes"] / 1e6
    print(f"wrote {out}: {summary['contigs']} contigs, {summary['bases']:,} bases, {mb:,.1f} MB in {summary['seconds']}s", file=sys.stderr)
    if summary.get("conflicts"):
        print(f"note: {summary['conflicts']} conflicting overlapping calls (later record kept); see `gpxil info --json`", file=sys.stderr)
    n = summary.get("ref_mismatches") or 0
    if n:
        # (the usual cause: a VCF called against another reference, such as GRCh37 on GRCh38)
        what = "record's REF allele doesn't" if n == 1 else "records' REF alleles don't"
        print(
            f"note: {n:,} {what} match the reference: was the VCF called against another one?"
            " `gpxil info --json` lists the first for each contig",
            file=sys.stderr,
        )
    return 0


def cmd_info(a) -> int:
    from .container import GpxilFile

    with GpxilFile(a.file) as g:
        if a.json:
            print(json.dumps(g.metadata, indent=2))
            return 0
        m = g.metadata
        print(f"{a.file}: gpxil {m.get('gpxil')}, written by {m.get('generator')}, {m.get('created')}")
        ref = m.get("reference") or {}
        print(f"reference: {ref.get('name')} ({ref.get('source')})")
        if g.sample_id:
            s = m["sample"]
            print(
                f"sample: {s.get('id')} from {s.get('source')}"
                + (f", callable regions {s['callable_regions']}" if s.get("callable_regions") else "")
            )
        print(f"layout: raster, width {m['layout']['width']}; palette {m.get('palette')}")
        print(f"{'contig':<16}{'length':>14}  {'planes':<24}{'records':>10}{'sidecar':>9}")
        for c in g:
            e = c.entry
            rec = e.get("encoding", {}).get("records", "")
            print(
                f"{c.name:<16}{c.length:>14,}  {','.join(c.planes):<24}{rec if rec == '' else f'{rec:,}':>10}{e.get('sidecar_entries', 0):>9,}"
            )
    return 0


def cmd_validate(a) -> int:
    from .validate import validate

    rep = validate(a.file, check_digests=not a.no_digests)
    for w in rep.warnings:
        print(f"warning: {w}")
    for e in rep.errors:
        print(f"error: {e}")
    print(
        f"{a.file}: {'valid' if rep.ok else 'INVALID'} ({len(rep.checked)} contigs checked, {len(rep.errors)} errors, {len(rep.warnings)} warnings)"
    )
    return 0 if rep.ok else 1


def cmd_subset(a) -> int:
    from .subset import subset

    info = subset(
        a.file,
        a.output,
        a.contigs,
        planes=a.planes.split(",") if a.planes else None,
        include_ref=not a.no_ref,
        reference=a.reference,
        compression=a.compression,
        level=a.level,
    )
    how = "strips copied as stored" if not info["pages_encoded"] else f"{info['compression']}, {info['pages_encoded']} pages encoded again"
    print(
        f"wrote {a.output}: {info['contigs']} contig{'s' if info['contigs'] != 1 else ''} ({', '.join(info['names'][:6])}"
        f"{', …' if info['contigs'] > 6 else ''}), {info['bases']:,} bases, planes {','.join(info['planes'])};"
        f" {info['bytes'] / 1e6:,.1f} MB in {info['seconds']}s ({how})",
        file=sys.stderr,
    )
    return 0


def cmd_stats(a) -> int:
    from .stats import format_stats, stats

    with _open(a) as g:
        res = stats(g, a.regions or None)
    print(json.dumps(res, indent=2) if a.json else format_stats(res))
    return 0


def cmd_compare(a) -> int:
    from .compare import compare_pair, compare_trio, format_compare, picture
    from .container import GpxilFile

    trio = bool(a.father or a.mother)
    if trio and not (a.father and a.mother):
        raise SystemExit("gpxil: error: a trio needs both --father and --mother")
    if trio and len(a.files) != 1:
        raise SystemExit("gpxil: error: with --father and --mother, give one file: the child's")
    if not trio and len(a.files) != 2:
        raise SystemExit("gpxil: error: give two files to compare, or a child's with --father and --mother")
    files = []
    try:
        for path in [a.files[0], a.father, a.mother] if trio else a.files:
            files.append(GpxilFile(path, reference=a.reference))
        kw = {"gap": a.gap, "all_calls": a.all_calls}
        res = compare_trio(*files, a.region, **kw) if trio else compare_pair(*files, a.region, **kw)
        if a.output:
            first = res["regions"][0]
            name, start, end = a.draw or (first["contig"], first["start"] - 1, first["end"])
            pic = picture(files, name, start, end, bases_per_row=a.width, palette=a.palette, **kw)
            pic = pic.scaled(a.scale)
            if a.output.lower().endswith((".tif", ".tiff")):
                pic.save_tiff(a.output, colorspace="cmyk" if a.cmyk else "rgb")
            elif a.cmyk:
                raise SystemExit("--cmyk needs a .tif output (PNG has no CMYK)")
            else:
                pic.save_png(a.output)
            h, w = pic.shape
            print(f"wrote {a.output}: {w} x {h} px, {pic.title}", file=sys.stderr)
    finally:
        for g in files:
            g.close()
    print(json.dumps(res, indent=2) if a.json else format_compare(res))
    return 0


def cmd_render(a) -> int:
    from .layout import hilbert_image
    from .render import render

    name, start, end = a.region
    with _open(a) as g:
        c = g[name]
        r = render(c, mode=a.mode, palette=a.palette, start=start, end=end, bases_per_row=a.width, mark_insertions=a.mark_insertions)
        if a.layout == "hilbert":
            import numpy as np

            if r.rows_per_line != 1:
                raise SystemExit("--layout hilbert works with the reference, genotype and variants modes")
            flat = r.index.reshape(-1)[: (end or c.length) - start]
            img, order = hilbert_image(flat.astype(np.uint8))
            r.index = img
        r = r.scaled(a.scale)
        out = a.output
        if out.lower().endswith((".tif", ".tiff")):
            r.save_tiff(out, colorspace="cmyk" if a.cmyk else "rgb", dpi=a.dpi)
        else:
            if a.cmyk:
                raise SystemExit("--cmyk needs a .tif output (PNG has no CMYK)")
            r.save_png(out, dpi=a.dpi)
        h, w = r.shape
        print(f"wrote {out}: {w} x {h} px, {r.title}", file=sys.stderr)
    return 0


def cmd_book(a) -> int:
    from .book import BookLayout, make_book

    layout = BookLayout(page=a.page, dpi=a.dpi, margin_mm=a.margin, scale=a.scale)
    with _open(a) as g:
        info = make_book(
            g,
            a.output,
            contigs=a.contigs.split(",") if a.contigs else None,
            mode=a.mode,
            palette=a.palette,
            colorspace="rgb" if a.rgb else "cmyk",
            layout=layout,
            title=a.title,
            max_pages=a.max_pages,
            log=lambda m: print(m, file=sys.stderr),
        )
    print(f"wrote {a.output}: {info['pages']} pages, {info['bases_per_page']:,} bases per full page", file=sys.stderr)
    return 0


def cmd_fasta(a) -> int:
    from .export import iter_genotype_iupac, iter_haplotype, iter_reference, write_fasta_stream

    out = open(a.output, "wb") if a.output else _stdout_bytes()
    try:
        with _open(a) as g:
            if a.which != "ref" and not g.sample_id:
                raise SystemExit(f"gpxil: error: {a.file} has no sample, so only --which ref can be exported")
            regions = [a.region] if a.region else [(n, 0, None) for n in g.contigs]
            for name, s, e in regions:
                c = g[name]
                e = c.length if e is None else e
                if not (0 <= s < e <= c.length):
                    raise SystemExit(f"gpxil: error: {name}:{s + 1}-{e} is outside {c.name} (length {c.length:,})")
                if a.which != "ref" and not c.has("gt"):
                    print(f"note: {c.name} has no genotype plane (no calls); skipped", file=sys.stderr)
                    continue
                if a.which == "ref":
                    pieces = iter_reference(c, s, e)
                elif a.which == "iupac":
                    pieces = iter_genotype_iupac(c, s, e)
                else:
                    pieces = iter_haplotype(c, int(a.which), s, e)
                label = c.name if (s, e) == (0, c.length) else f"{c.name}:{s + 1}-{e}"
                suffix = "" if a.which == "ref" else f"_{'iupac' if a.which == 'iupac' else 'hap' + a.which}"
                write_fasta_stream(out, label + suffix, pieces)
    finally:
        if a.output:
            out.close()
    return 0


def cmd_vcf(a) -> int:
    from .export import write_vcf

    if a.contigs and a.region:
        raise SystemExit("gpxil: error: give --contigs or --region, not both")
    out = open(a.output, "w") if a.output else sys.stdout
    try:
        with _open(a) as g:
            n = write_vcf(g, out, a.contigs.split(",") if a.contigs else None, nocall_blocks=not a.no_nocall_blocks, regions=a.region)
    finally:
        if a.output:
            out.close()
    print(f"wrote {n:,} records", file=sys.stderr)
    return 0


def cmd_palettes(a) -> int:
    from .palettes import describe

    for p in describe():
        bases = ", ".join(f"{b} {tuple(p['srgb'][b])} / CMYK {tuple(p['cmyk_percent'][b])}" for b in "ACGTN")
        print(f"{p['name']}: {p['description']}\n    {bases}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .container import COMPRESSIONS, DEFAULT_COMPRESSION, DEFAULT_WIDTH
    from .palettes import PALETTES

    p = argparse.ArgumentParser(prog="gpxil", description="Genome pixel images: encode, inspect, render and export gpxil files.")
    p.add_argument("--version", action="version", version=f"gpxiltools {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="FASTA (+ one sample's VCF) -> .gpxil.tif")
    e.add_argument("fasta", help="reference FASTA (plain, gzip or bgzip), or a gpxil file (path or URL) with the reference plane")
    e.add_argument("-v", "--vcf", help="VCF/VCF.gz with the sample's calls")
    e.add_argument("-s", "--sample", help="sample column to use (default: first)")
    e.add_argument("-o", "--output", help="output path (default: <name>.gpxil.tif)")
    e.add_argument("--callable", metavar="BED", help="confidently called regions; positions outside become UNKNOWN")
    e.add_argument("--haploid", metavar="BED", help="regions where the sample has one copy (haplotype 2 ABSENT)")
    e.add_argument("--haploid-contig", action="append", metavar="NAME", help="whole contig haploid (repeatable)")
    e.add_argument("--sex", choices=["XX", "XY"], help="XY: treat GRCh38 chrX/chrY outside the PARs as haploid")
    e.add_argument("--contigs", help="comma-separated subset of contigs")
    e.add_argument("--ucsc-names", action="store_true", help="rename 1..22,X,Y,MT to chr1..chr22,chrX,chrY,chrM")
    e.add_argument("--no-ref", action="store_true", help="omit the ref plane (reference identified by digests)")
    e.add_argument("--planes", help="variant planes to write (default: gt,flags + gq/dp/ps when present)")
    e.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"bases per image row (default {DEFAULT_WIDTH})")
    e.add_argument("--compression", choices=list(COMPRESSIONS), default=DEFAULT_COMPRESSION)
    e.add_argument("--level", type=int, help="compression level")
    e.add_argument("--palette", choices=list(PALETTES), default="classic", help="colours stored for image viewers")
    e.add_argument("--reference-name", help="e.g. GRCh38")
    e.add_argument("--invalid-to-n", action="store_true", help="turn unknown FASTA characters into N instead of failing")
    e.add_argument("--bigtiff", action="store_true", help="force BigTIFF (needed above 4 GiB)")
    e.set_defaults(fn=cmd_encode)

    i = sub.add_parser("info", help="summarise a gpxil file")
    i.add_argument("file", help="a gpxil file: a path or a URL")
    i.add_argument("--json", action="store_true", help="print the full metadata")
    i.set_defaults(fn=cmd_info)

    v = sub.add_parser("validate", help="check a file against the gpxil specification")
    v.add_argument("file")
    v.add_argument("--no-digests", action="store_true", help="skip recomputing sequence digests")
    v.set_defaults(fn=cmd_validate)

    s = sub.add_parser("subset", help="copy some contigs (and planes) into a new gpxil file: a working copy, without encoding again")
    s.add_argument("file", help="a gpxil file: a path or a URL")
    s.add_argument("contigs", nargs="*", metavar="CONTIG", help="contigs to copy, by name or alias (default: all)")
    s.add_argument("-o", "--output", required=True, help="the new gpxil file")
    s.add_argument("--planes", help="planes to copy, comma-separated, e.g. ref,gt,flags (default: all each contig has)")
    s.add_argument("--no-ref", action="store_true", help="leave out the ref plane (the contigs keep their reference's digests)")
    s.add_argument(
        "--reference", metavar="FASTA", help="reference FASTA or gpxil file, to write the ref plane of contigs stored without one"
    )
    s.add_argument("--compression", choices=list(COMPRESSIONS), help="default: the input's, copying its strips without decoding them")
    s.add_argument("--level", type=int, help="compression level (encodes the strips again)")
    s.set_defaults(fn=cmd_subset)

    st = sub.add_parser("stats", help="count what a file holds: calls, variants, phasing, quality")
    st.add_argument("file", help="a gpxil file: a path or a URL")
    st.add_argument(
        "regions", nargs="*", type=_region, metavar="REGION", help="CONTIG or CONTIG:START-END (1-based, inclusive); default: every contig"
    )
    st.add_argument("--json", action="store_true", help="the counts as JSON")
    st.add_argument("--reference", metavar="FASTA", help="reference FASTA or gpxil file, for files written with --no-ref")
    st.set_defaults(fn=cmd_stats)

    cp = sub.add_parser("compare", help="compare two genomes base by base, or a child's with both parents' (a trio)")
    cp.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="two gpxil files, A and B (B taken as the truth); or the child's, with --father and --mother",
    )
    cp.add_argument("--father", metavar="FILE", help="the father's gpxil file, for a trio")
    cp.add_argument("--mother", metavar="FILE", help="the mother's gpxil file, for a trio")
    cp.add_argument(
        "--region",
        type=_region,
        action="append",
        help="CONTIG or CONTIG:START-END (1-based, inclusive); repeatable; default: every contig all have",
    )
    cp.add_argument("-o", "--output", help="a picture, one pixel a base: .png, or .tif")
    cp.add_argument("--draw", type=_region, metavar="REGION", help="the region to draw with -o (default: the first region compared)")
    cp.add_argument("--width", type=int, help="bases per row of the picture (default: the files' layout width)")
    cp.add_argument("--scale", type=int, default=1, help="pixels per base in the picture, each way")
    cp.add_argument("--palette", choices=list(PALETTES), default="classic", help="colours of the reference, faded, in the picture")
    cp.add_argument("--cmyk", action="store_true", help="the picture as a CMYK TIFF for print")
    cp.add_argument(
        "--gap", type=int, default=50, help="differences this close together are one stretch, compared as sequences (default 50)"
    )
    cp.add_argument("--all-calls", action="store_true", help="also compare the calls outside the regions called with confidence")
    cp.add_argument("--json", action="store_true", help="the counts as JSON")
    cp.add_argument("--reference", metavar="FASTA", help="reference FASTA or gpxil file, for files written with --no-ref")
    cp.set_defaults(fn=cmd_compare)

    r = sub.add_parser("render", help="draw a contig or interval as PNG (screen) or TIFF (RGB or CMYK)")
    r.add_argument("file")
    r.add_argument("region", type=_region, help="CONTIG or CONTIG:START-END (1-based, inclusive)")
    r.add_argument("-o", "--output", required=True, help=".png or .tif")
    r.add_argument("--mode", choices=["reference", "genotype", "haplotypes", "variants"], default="genotype")
    r.add_argument("--palette", choices=list(PALETTES), default="classic")
    r.add_argument("--width", type=int, help="bases per row (default: the file's layout width)")
    r.add_argument("--layout", choices=["raster", "hilbert"], default="raster")
    r.add_argument("--scale", type=int, default=1, help="pixels per base, each way")
    r.add_argument("--cmyk", action="store_true", help="CMYK TIFF for print")
    r.add_argument("--dpi", type=int, help="resolution to record in the image")
    r.add_argument("--mark-insertions", action="store_true", help="haplotypes mode: mark insertion anchors")
    r.add_argument("--reference", metavar="FASTA", help="reference FASTA or gpxil file (path or URL), for files written with --no-ref")
    r.set_defaults(fn=cmd_render)

    b = sub.add_parser("book", help="print-ready PDF with one dot per base")
    b.add_argument("file")
    b.add_argument("-o", "--output", required=True)
    b.add_argument("--contigs", help="comma-separated subset of contigs")
    b.add_argument("--mode", choices=["reference", "genotype", "haplotypes", "variants"], default="genotype")
    b.add_argument("--palette", choices=list(PALETTES), default="classic")
    b.add_argument("--rgb", action="store_true", help="RGB colours for screen instead of CMYK for print")
    b.add_argument("--page", choices=["A3", "A4", "A5", "letter"], default="A4")
    b.add_argument("--dpi", type=int, default=300, help="printer dots per inch (one base per dot)")
    b.add_argument("--scale", type=int, default=1, help="dots per base, each way")
    b.add_argument("--margin", type=float, default=15.0, help="page margin in mm")
    b.add_argument("--title")
    b.add_argument("--max-pages", type=int, help="stop after this many pages (for proofs)")
    b.add_argument("--reference", metavar="FASTA", help="reference FASTA or gpxil file (path or URL), for files written with --no-ref")
    b.set_defaults(fn=cmd_book)

    f = sub.add_parser("fasta", help="export reference, haplotype or IUPAC-genotype sequence")
    f.add_argument("file")
    f.add_argument("--which", choices=["ref", "1", "2", "iupac"], default="ref", help="ref, haplotype 1, haplotype 2 or IUPAC genotype")
    f.add_argument("--region", type=_region)
    f.add_argument("-o", "--output")
    f.add_argument("--reference", metavar="FASTA", help="reference FASTA or gpxil file (path or URL), for files written with --no-ref")
    f.set_defaults(fn=cmd_fasta)

    x = sub.add_parser("vcf", help="export the sample's genotypes as VCF")
    x.add_argument("file")
    x.add_argument("-o", "--output")
    x.add_argument("--contigs", help="comma-separated contigs (default: all)")
    x.add_argument("--region", type=_region, action="append", help="CONTIG or CONTIG:START-END (1-based, inclusive); repeatable")
    x.add_argument("--no-nocall-blocks", action="store_true", help="omit no-call blocks (ALT=<*>)")
    x.add_argument("--reference", metavar="FASTA", help="reference FASTA or gpxil file (path or URL), for files written with --no-ref")
    x.set_defaults(fn=cmd_vcf)

    pl = sub.add_parser("palettes", help="list the built-in palettes")
    pl.set_defaults(fn=cmd_palettes)
    return p


def main(argv=None) -> int:
    from .container import GpxilFormatError

    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except BrokenPipeError:  # e.g. `gpxil vcf file | head`
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())  # so the final flush at exit does not complain
        return 0
    except (GpxilFormatError, KeyError, IndexError, OSError, ValueError) as exc:
        # expected problems (bad input, missing contig or plane) get a message, not a traceback
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"gpxil: error: {msg}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
