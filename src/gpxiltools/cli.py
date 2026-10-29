"""Command-line interface: ``gpxil <command> ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def _names(text: str) -> list[str]:
    """Contig names given as text: commas separate them, but not in a region, where they group digits."""
    from .container import split_names

    return split_names(text)


def _region(text: str | None):
    """A region given as text, kept as text: what it is depends on the file (a contig may be named 'HLA-A*01:14', or
    even 'HLA:0-5'), so each command reads it with the file's ``region_of`` (:func:`_in`), which says what is wrong
    with a text that is neither a contig's name nor a region."""
    if text is None:
        return None
    if not str(text).strip(" \t\r\n"):
        raise argparse.ArgumentTypeError("an empty region: use CONTIG or CONTIG:START-END (1-based, inclusive)")
    return text


def _in(g, texts):
    """Regions given as text, as the file reads them: ``(name, start, end)``, 0-based, the end excluded; 'chr21'
    the whole contig, 'chr21:5,000,001-5,100,000' a region, 'chr21:5,000,001' one base, and a contig's name always
    that contig (spec 3.1: a window's name is its region)."""
    if texts is None:
        return None
    return [g.region_of(x) for x in texts]


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
    from .container import FORMAT_VERSION
    from .encode import encode

    if a.contigs and a.region:
        raise SystemExit("gpxil: error: give --contigs or --region, not both")

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
        regions=a.region,
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
    size = f"{summary['bytes'] / 1e6:,.1f} MB" if summary["bytes"] >= 1e5 else f"{summary['bytes'] / 1e3:,.1f} kB"
    nc = summary["contigs"]
    print(f"wrote {out}: {nc} contig{'s' if nc != 1 else ''}, {summary['bases']:,} bases, {size} in {summary['seconds']}s", file=sys.stderr)
    if summary.get("conflicts"):
        print(f"note: {summary['conflicts']} conflicting overlapping calls (later record kept); see `gpxil info --json`", file=sys.stderr)
    n = summary.get("overlapping_calls_reassigned") or 0
    if n:
        what = "call overlaps another, so its alleles were" if n == 1 else "calls overlap others, so their alleles were"
        print(f"note: {n:,} unphased {what} put the other way round (gpxil {FORMAT_VERSION}, section 6.5)", file=sys.stderr)
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
        sub = m.get("subset")
        if isinstance(sub, dict) and sub.get("of"):
            print(f"copied from {sub['of']}" + (f" (gpxil {sub['gpxil']})" if sub.get("gpxil") else ""))
        wide = max([16, *(len(c.name) + 2 for c in g)])
        print(f"{'contig':<{wide}}{'length':>14}  {'planes':<24}{'records':>10}{'sidecar':>9}")
        for c in g:
            e = c.entry
            rec = e.get("encoding", {}).get("records", "")
            print(
                f"{c.name:<{wide}}{c.length:>14,}  {','.join(c.planes):<24}{rec if rec == '' else f'{rec:,}':>10}{e.get('sidecar_entries', 0):>9,}"
            )
        for c in g.windows:
            n = c.parent_length
            print(f"{c.name}: a window of {c.parent}" + (f" ({n:,} bp)" if n else "") + f", its bases {c.start + 1:,} to {c.end:,}")
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
    size = f"{info['bytes'] / 1e6:,.1f} MB" if info["bytes"] >= 1e5 else f"{info['bytes'] / 1e3:,.1f} kB"
    wins = info["windows"]
    print(
        f"wrote {a.output}: {info['contigs']} contig{'s' if info['contigs'] != 1 else ''} ({', '.join(info['names'][:6])}"
        f"{', …' if info['contigs'] > 6 else ''}){f', {wins} of them windows' if wins > 1 else ', a window' if wins else ''},"
        f" {info['bases']:,} bases, planes {','.join(info['planes'])}; {size} in {info['seconds']}s ({how})",
        file=sys.stderr,
    )
    return 0


def cmd_stats(a) -> int:
    from .stats import format_stats, stats

    with _open(a) as g:
        res = stats(g, _in(g, a.regions) or None)
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
        regions = _in(files[0], a.region)
        res = compare_trio(*files, regions, **kw) if trio else compare_pair(*files, regions, **kw)
        if a.output:
            first = res["regions"][0]
            name, start, end = files[0].region_of(a.draw) if a.draw else (first["contig"], first["start"] - 1, first["end"])
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

    with _open(a) as g:
        # (read through the windows that overlap or touch the one that holds it, as every command reads a region)
        c, start, end = g.view(*g.locate(*g.region_of(a.region), join=True))
        r = render(c, mode=a.mode, palette=a.palette, start=start, end=end, bases_per_row=a.width, mark_insertions=a.mark_insertions)
        if a.layout == "hilbert":
            import numpy as np

            if r.rows_per_line != 1:
                raise SystemExit("--layout hilbert works with the reference, genotype and variants modes")
            flat = r.index.reshape(-1)[: end - start]
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
            contigs=_names(a.contigs) if a.contigs else None,
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
            regions = [g.region_of(a.region)] if a.region else [(n, None, None) for n in g.contigs]
            for name, s, e in regions:
                # (read through the windows that overlap or touch the one that holds it: an event that one cuts is
                # whole, and a window without the sample's planes, or the reference, reads them from another)
                c, s, e = g.view(*g.locate(name, s, e, join=True))
                if a.which != "ref" and not c.has("gt"):
                    print(f"note: {c.region(s, e)} has no genotype plane (no calls); skipped", file=sys.stderr)
                    continue
                if a.which == "ref":
                    pieces = iter_reference(c, s, e)
                elif a.which == "iupac":
                    pieces = iter_genotype_iupac(c, s, e)
                else:
                    pieces = iter_haplotype(c, int(a.which), s, e)
                # (a stretch, or a window, named by the region it is of its sequence: spec 3.1)
                label = c.name if (s, e) == (0, c.length) and c.window is None else c.region(s, e, commas=False)
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
            n = write_vcf(g, out, _names(a.contigs) if a.contigs else None, nocall_blocks=not a.no_nocall_blocks, regions=_in(g, a.region))
    finally:
        if a.output:
            out.close()
    print(f"wrote {n:,} records", file=sys.stderr)
    return 0


def cmd_runs(a) -> int:
    from .runs import RunFinder, Summary, format_summary, write_bed

    with _open(a) as g:
        finder = RunFinder(g, _in(g, a.regions) or None, of=a.of, min_length=a.min, both_strands=a.both_strands)
        summary = Summary()
        if a.output:
            with open(a.output, "w") as out:
                write_bed(out, finder.runs(), summary, of=a.of)
        elif a.json:
            for r in finder.runs():
                summary.add(r)
        else:
            write_bed(sys.stdout, finder.runs(), summary, of=a.of)
        d = summary.as_dict(a.of, a.min, finder.acgt)
    if a.json:
        print(json.dumps(d, indent=2))
    else:
        print(format_summary(d) + (f"; wrote {a.output}" if a.output else ""), file=sys.stderr)
    return 0


def cmd_period(a) -> int:
    from .period import best_period, panels, picture, text

    with _open(a) as g:
        c, s, e = g.view(*g.locate(*g.region_of(a.region), join=True))
        if not c.covers("ref", s, e):
            raise ValueError(f"{c.region(s, e)} has no reference: give it with --reference")
        found, share, shares = best_period(c.read("ref", s, e), a.max_period)
        period = a.period or found
        if a.period:
            share = shares.get(period)
        elif share < 0.6:
            print(f"note: no clear period (the best, {period}, has {share:.2f} of bases equal a unit on)", file=sys.stderr)
        ps = panels(c, s, e, layout=a.layout)
        title = f"{c.region(s, e)}: period {period}" + (f", {100 * share:.0f} % of bases equal to the base a unit on" if share else "")
        if a.output:
            img = picture(ps, period, scale=a.scale, palette=a.palette, title=title)
            img.save(a.output)
            print(f"wrote {a.output}: {img.width} x {img.height} px; {title}", file=sys.stderr)
        else:
            print(title)
            print(text(ps, period))
    return 0


def _sorted_by(st, layout: str = "bed") -> str:
    return {
        "file": "as the features come",
        "position": "by position",
        "length": "by length, longest first",
        "score": "by divergence from the repeat's consensus, highest first" if layout == "rmsk" else "by score, highest first",
        "name": "by name",
        "run": f"by the run of {st.run_of} from the anchor, longest first",
    }[st.sort] + (" (reversed)" if st.reversed else "")


def cmd_stack(a) -> int:
    from .container import path_name
    from .features import layout_of, read_features
    from .stack import binned, labelled, legend_of, rendering, save_pixels, stack, write_profile, write_rows

    layout = layout_of(a.features) if a.layout == "auto" else a.layout
    feats = read_features(a.features, layout, match=a.match, min_length=a.min_length)
    with _open(a) as g:
        if a.mode != "reference" and g.sample_id is None:
            raise ValueError(f"{a.file} has no sample: draw the reference (--mode reference)")
        st = stack(
            g,
            feats,
            anchor=a.anchor,
            up=a.up,
            down=a.down,
            stranded=not a.ignore_strand,
            sort=a.sort,
            run_of=a.run_of,
            tolerance=a.tolerance,
            reverse=a.reverse,
            max_rows=a.max_rows,
            sample=a.mode != "reference" or bool(a.profile and g.sample_id),
        )
    left_out = [
        f"{st.unplaced:,} on no sequence of the file" if st.unplaced else "",
        f"{st.dropped:,} past --max-rows" if st.dropped else "",
    ]
    note = f"{len(st.rows):,} rows of {st.features:,} features" + (f" ({'; '.join(x for x in left_out if x)})" if any(left_out) else "")
    if not st.rows:
        raise ValueError(f"nothing to stack: {note}")
    note += f"; {st.up:,} bases before each one's {st.anchor} and {st.down:,} from it; sorted {_sorted_by(st, layout)}"
    if a.profile:
        with open(a.profile, "w") as out:
            write_profile(out, st)
    if a.rows:
        with open(a.rows, "w") as out:
            write_rows(out, st)
    if a.output:
        tif = a.output.lower().endswith((".tif", ".tiff"))
        if a.cmyk and (not tif or a.labelled):
            raise ValueError("--cmyk needs a .tif output, without --labelled")
        r = rendering(st, mode=a.mode, palette=a.palette)
        if a.bin > 1 or a.labelled:
            space = "cmyk" if a.cmyk else "rgb"
            pixels = binned(r, a.bin, space) if a.bin > 1 else (r.to_cmyk() if a.cmyk else r.to_rgb())
            if a.labelled:
                title = f"{os.path.basename(a.features)} on {path_name(a.file)}: {st.anchor}s lined up ({a.mode})"
                rows_note = f"{len(st.rows):,} rows, sorted {_sorted_by(st, layout)}" + (f"; {a.bin} a pixel row" if a.bin > 1 else "")
                img = labelled(pixels, st, scale=a.scale, title=title, legend=legend_of(r), rows_note=rows_note)
                img.save(a.output)
                w, h = img.size
            else:
                w, h = save_pixels(pixels, a.output, scale=a.scale)
        else:
            r = r.scaled(a.scale)
            if tif:
                r.save_tiff(a.output, colorspace="cmyk" if a.cmyk else "rgb")
            else:
                r.save_png(a.output)
            h, w = r.shape
        note += f"; wrote {a.output}: {w} x {h} px"
    elif not a.profile:
        write_profile(sys.stdout, st)
    print(note, file=sys.stderr)
    return 0


def cmd_spectrum(a) -> int:
    from .features import read_features
    from .spectrum import count_spectrum, format_results, write_table

    feats = read_features(a.features, a.layout, match=a.match, min_length=a.min_length)
    back = read_features(a.background) if a.background else None
    called = read_features(a.callable) if a.callable else None
    with _open(a) as g:
        counter = count_spectrum(
            g,
            feats,
            _in(g, a.regions) or None,
            background=back,
            callable=called,
            k=a.context,
            centre=a.centre,
            isolated=a.isolated,
            stranded=a.stranded,
            all_calls=a.all_calls,
        )
    d = counter.results()
    if a.output:
        with open(a.output, "w") as out:
            write_table(out, counter.table())
    if a.json:
        print(json.dumps(d, indent=2))
    else:
        f = d["features"]
        head = f"{f['given']:,} features" + (f" ({f['on_no_sequence']:,} on no sequence of the file)" if f["on_no_sequence"] else "")
        print(
            f"{head}: {f['region_bases']:,} bases; calls outside the regions called with confidence {d['calls_outside_confident_regions']}"
        )
        print(format_results(d))
    if a.output:
        print(f"wrote {a.output}: by context", file=sys.stderr)
    return 0


def cmd_annotate(a) -> int:
    from .annotate import annotate

    info = annotate(
        a.file,
        a.output,
        a.features,
        plane=a.plane,
        kind=getattr(a, "as"),
        level=a.level,
        layout=a.layout,
        match=a.match,
        min_length=a.min_length,
        description=a.description,
        units=a.units,
        replace=a.replace,
        log=(lambda m: None) if a.quiet else (lambda m: print(m, file=sys.stderr)),
    )
    labels = f", {info['labels']:,} labels" if info["kind"] == "category" else ""
    placed = f" ({info['unplaced']:,} on no sequence of the file)" if info["unplaced"] else ""
    print(
        f"wrote {a.output}: {info['contigs']} contigs with a {info['plane']} plane ({info['kind']}, {info['dtype']}{labels}) from"
        f" {info['features']:,} features{placed}, over {info['bases_with_a_feature']:,} bases; {info['bytes'] / 1e6:.1f} MB in {info['seconds']} s",
        file=sys.stderr,
    )
    return 0


def cmd_skew(a) -> int:
    from .features import read_features
    from .skew import SKEWS, SkewCounter, iter_skew, picture

    ks = sorted({int(x) for x in a.k.split(",") if x.strip()}) if a.k else [1, 2, 3, 4]
    show = [x.strip() for x in a.show.split(",") if x.strip()] if a.show else ["gc_skew", "at_skew"]
    known = [*SKEWS, "cumulative_gc_skew", *(f"asymmetry_{k}" for k in ks)]
    bad = [x for x in show if x not in known]
    if bad:
        raise ValueError(f"--show {bad[0]}: choose from {', '.join(known)}")
    feats = read_features(a.features, a.layout, match=a.match, min_length=a.min_length) if a.features else None
    if a.stranded and feats is None:
        raise ValueError("--stranded reads features on their strands: give --features")
    counter = SkewCounter(a.window, ks, unmasked=a.unmasked)
    out = open(a.output, "w") if a.output else (None if a.json else sys.stdout)
    kept, n = [], 0
    try:
        with _open(a) as g:
            for row in iter_skew(g, _in(g, a.regions) or None, features=feats, stranded=a.stranded, counter=counter):
                if out is not None:
                    if n == 0:
                        out.write("\t".join(row) + "\n")
                    out.write("\t".join("" if v is None else str(v) for v in row.values()) + "\n")
                if a.picture:
                    kept.append({k: row[k] for k in ("sequence", "start", "end", *show)})
                n += 1
    finally:
        if a.output and out is not None:
            out.close()
    totals = counter.totals()
    if a.picture:
        if not kept:
            raise ValueError("no windows with bases to draw")
        cols = max(r["end"] // a.window for r in kept)
        if cols > 20000:
            raise ValueError(f"the picture would be {cols:,} windows wide: give a larger --window")
        img = picture(kept, show, window=a.window, title=f"{os.path.basename(str(a.file))}: windows of {a.window:,} bases")
        img.save(a.picture)
    if a.json:
        print(json.dumps({"window": a.window, "k": ks, "windows": n, "totals": totals}, indent=2))
    t = totals.get("all") or {}
    if t:
        asym = ", ".join(f"k {k} {t[f'asymmetry_{k}']}" for k in ks)
        print(
            f"{n:,} windows of {a.window:,} bases; over all {t['bases']:,} bases: GC {t['gc']}, GC skew {t['gc_skew']},"
            f" AT skew {t['at_skew']}; asymmetry {asym}"
            + (f"; wrote {a.output}" if a.output else "")
            + (f"; drew {a.picture}" if a.picture else ""),
            file=sys.stderr,
        )
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

    p = argparse.ArgumentParser(prog="gpxil", description="Genome pixel images: encode, inspect, compare, render and export gpxil files.")
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
    e.add_argument(
        "--region",
        type=_region,
        action="append",
        metavar="REGION",
        help="CONTIG:START-END (1-based, inclusive): a window of the contig, which keeps its positions (repeatable)",
    )
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

    s = sub.add_parser(
        "subset", help="copy some contigs (and planes) into a new gpxil file, or windows of them: a working copy, without encoding again"
    )
    s.add_argument("file", help="a gpxil file: a path or a URL")
    s.add_argument(
        "contigs",
        nargs="*",
        metavar="CONTIG",
        help="contigs to copy, by name or alias, or regions (CONTIG:START-END, 1-based) to cut out as windows (default: all)",
    )
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

    ru = sub.add_parser("runs", help="runs of a composition: the longest stretches whose bases all belong to a set (AG: GA-sequences)")
    ru.add_argument("file", help="a gpxil file: a path or a URL")
    ru.add_argument(
        "regions", nargs="*", type=_region, metavar="REGION", help="CONTIG or CONTIG:START-END (1-based, inclusive); default: every contig"
    )
    ru.add_argument("--of", required=True, metavar="BASES", help="the set, in IUPAC letters: AG (or R) for purines, CT, A, AT, GC ...")
    ru.add_argument("--min", type=int, default=51, metavar="N", help="the shortest run to give (default 51: more than 50 bases)")
    ru.add_argument("--both-strands", action="store_true", help="the set's runs on the other strand too (strand -)")
    ru.add_argument("-o", "--output", help="the runs as BED (default: to standard output)")
    ru.add_argument("--json", action="store_true", help="a summary as JSON (give -o for the runs)")
    ru.add_argument("--reference", metavar="FASTA", help="reference FASTA or gpxil file, for files written with --no-ref")
    ru.set_defaults(fn=cmd_runs)

    pe = sub.add_parser("period", help="a stretch wrapped at its repeat period, a unit a row: reference and haplotypes")
    pe.add_argument("file", help="a gpxil file: a path or a URL")
    pe.add_argument("region", type=_region, help="CONTIG:START-END (1-based, inclusive)")
    pe.add_argument("--period", type=int, help="bases a unit (default: found, the shortest that fits)")
    pe.add_argument("--max-period", type=int, default=50, help="the longest period to look for (default 50)")
    pe.add_argument(
        "--layout", choices=["aligned", "sequence"], default="aligned", help="haplotypes at the reference's positions, or as sequences"
    )
    pe.add_argument("-o", "--output", help="a picture (.png); default: the units as text")
    pe.add_argument("--scale", type=int, default=8, help="pixels a base, each way (default 8)")
    pe.add_argument("--palette", choices=list(PALETTES), default="classic")
    pe.add_argument("--reference", metavar="FASTA", help="reference FASTA or gpxil file, for files written with --no-ref")
    pe.set_defaults(fn=cmd_period)

    sk = sub.add_parser("stack", help="features stacked, one a row, lined up at their starts, ends or centres (the GA-complex image)")
    sk.add_argument("file", help="a gpxil file: a path or a URL")
    sk.add_argument("features", help="a BED file, or a RepeatMasker table (UCSC's rmsk, or a .out file); plain or gzipped")
    sk.add_argument("-o", "--output", help="the picture: .png or .tif (default: none, and the profile to standard output)")
    sk.add_argument(
        "--anchor", choices=["start", "end", "centre"], default="start", help="line up each feature's start, end or centre, on its strand"
    )
    sk.add_argument("--up", type=int, default=400, metavar="N", help="bases before the anchor (default 400)")
    sk.add_argument("--down", type=int, default=200, metavar="N", help="bases from the anchor on (default 200)")
    sk.add_argument("--ignore-strand", action="store_true", help="read every feature on the + strand")
    sk.add_argument(
        "--sort",
        choices=["file", "position", "length", "score", "name", "run"],
        default="file",
        help="the rows' order (default: as the features come); run: the run of --run-of bases from the anchor",
    )
    sk.add_argument("--run-of", default="A", metavar="BASES", help="--sort run: the run's bases, in IUPAC letters (default A: poly(A))")
    sk.add_argument("--tolerance", type=int, default=2, metavar="N", help="--sort run: other bases a run may hold (default 2)")
    sk.add_argument("--reverse", action="store_true", help="sort the other way")
    sk.add_argument("--max-rows", type=int, metavar="N", help="the first N rows, once sorted")
    sk.add_argument("--match", metavar="REGEX", help="features whose name matches ('/Alu/'; a repeat's name is class/family/name)")
    sk.add_argument("--min-length", type=int, default=0, metavar="N", help="features of N bases or more")
    sk.add_argument("--layout", choices=["auto", "bed", "rmsk"], default="auto", help="the features' file (default: told by its lines)")
    sk.add_argument(
        "--mode",
        choices=["reference", "genotype", "haplotypes", "variants"],
        default="reference",
        help="what to draw (default the reference)",
    )
    sk.add_argument("--palette", choices=list(PALETTES), default="classic")
    sk.add_argument("--scale", type=int, default=1, help="pixels per base, each way")
    sk.add_argument("--bin", type=int, default=1, metavar="K", help="each pixel row the mean colour of K rows (for many features)")
    sk.add_argument("--labelled", action="store_true", help="a title, the anchor, the bases from it, and a legend")
    sk.add_argument("--cmyk", action="store_true", help="CMYK TIFF for print")
    sk.add_argument("--profile", metavar="TSV", help="each column's bases (and the sample's variants), as a table")
    sk.add_argument("--rows", metavar="BED", help="the features in the order drawn, with each one's row and sort key")
    sk.add_argument("--reference", metavar="FASTA", help="reference FASTA or gpxil file, for files written with --no-ref")
    sk.set_defaults(fn=cmd_stack)

    sp = sub.add_parser("spectrum", help="the sample's substitutions in regions against the rest of the genome, context by context")
    sp.add_argument("file", help="a gpxil file with a sample: a path or a URL")
    sp.add_argument("features", help="the regions: a BED file, or a RepeatMasker table (UCSC rmsk or .out); plain or gzipped")
    sp.add_argument(
        "regions", nargs="*", type=_region, metavar="REGION", help="count only here: CONTIG or CONTIG:START-END (default: every contig)"
    )
    sp.add_argument("--background", metavar="BED", help="the background: these bases, outside the regions (default: all outside them)")
    sp.add_argument("--callable", metavar="BED", help="count only these bases (as well as the bases the file calls)")
    sp.add_argument(
        "--all-calls", action="store_true", help="keep the calls outside the regions the file was written to call with confidence"
    )
    sp.add_argument("--context", type=int, default=3, metavar="K", help="bases of context: 1, 3 (default), 5 ...")
    sp.add_argument(
        "--centre",
        choices=["pyrimidine", "purine"],
        default="pyrimidine",
        help="read contexts with this in the middle (default pyrimidine)",
    )
    sp.add_argument("--stranded", action="store_true", help="also the regions' substitutions on each feature's own strand (12 kinds)")
    sp.add_argument("--isolated", type=int, metavar="N", help="count only substitutions with no other variant within N bases")
    sp.add_argument("--match", metavar="REGEX", help="features whose name matches ('/Alu/'; a repeat's name is class/family/name)")
    sp.add_argument("--min-length", type=int, default=0, metavar="N", help="features of N bases or more")
    sp.add_argument("--layout", choices=["auto", "bed", "rmsk"], default="auto", help="the features' file (default: told by its lines)")
    sp.add_argument("-o", "--output", metavar="TSV", help="the counts by context and substitution, as a table")
    sp.add_argument("--json", action="store_true", help="the results as JSON")
    sp.add_argument("--reference", metavar="FASTA", help="reference FASTA or gpxil file, for files written with --no-ref")
    sp.set_defaults(fn=cmd_spectrum)

    an = sub.add_parser("annotate", help="a copy of a file with an annotation plane (x-NAME) from a BED file or a RepeatMasker table")
    an.add_argument("file", help="a gpxil file: a path or a URL")
    an.add_argument("features", help="a BED or bedGraph file, or a RepeatMasker table (UCSC rmsk or .out); plain or gzipped")
    an.add_argument("-o", "--output", required=True, help="the copy, with the new plane")
    an.add_argument("--plane", required=True, metavar="x-NAME", help="the new plane's name: x- and letters, digits, - or _")
    an.add_argument(
        "--as",
        choices=["auto", "category", "presence", "score", "value"],
        default="auto",
        help="labels as numbers, 1 where a feature is, scores, or a bedGraph's values (default: told by the file)",
    )
    an.add_argument("--level", choices=["class", "family", "name"], default="family", help="a repeat's label (default family: SINE/Alu)")
    an.add_argument("--layout", choices=["auto", "bed", "rmsk"], default="auto", help="the features' file (default: told by its lines)")
    an.add_argument("--match", metavar="REGEX", help="features whose name matches ('/Alu/'; a repeat's name is class/family/name)")
    an.add_argument("--min-length", type=int, default=0, metavar="N", help="features of N bases or more")
    an.add_argument("--description", help="what the plane holds, for its metadata (default: said from the features)")
    an.add_argument("--units", help="the values' units, for the plane's metadata")
    an.add_argument("--replace", action="store_true", help="make the plane again if the file has it")
    an.add_argument("-q", "--quiet", action="store_true", help="no progress")
    an.set_defaults(fn=cmd_annotate)

    sw = sub.add_parser("skew", help="strand symmetry in windows: composition, GC and AT skews, k-mer asymmetry (Chargaff's second rule)")
    sw.add_argument("file", help="a gpxil file: a path or a URL")
    sw.add_argument(
        "regions", nargs="*", type=_region, metavar="REGION", help="CONTIG or CONTIG:START-END (1-based, inclusive); default: every contig"
    )
    sw.add_argument(
        "--window", type=int, default=100_000, metavar="N", help="bases a window, at multiples of N along each sequence (default 100,000)"
    )
    sw.add_argument("--k", metavar="K,...", help="k-mer lengths for the asymmetries, 1 to 6 (default 1,2,3,4)")
    sw.add_argument("--unmasked", action="store_true", help="leave out soft-masked bases (repeats)")
    sw.add_argument("--features", metavar="BED", help="count only bases in these features (BED, or a RepeatMasker table)")
    sw.add_argument("--stranded", action="store_true", help="read each feature on its own strand (genes: their coding strands)")
    sw.add_argument("--match", metavar="REGEX", help="--features: those whose name matches")
    sw.add_argument("--min-length", type=int, default=0, metavar="N", help="--features: those of N bases or more")
    sw.add_argument("--layout", choices=["auto", "bed", "rmsk"], default="auto", help="--features: the file (default: told by its lines)")
    sw.add_argument("-o", "--output", metavar="TSV", help="the windows as a table (default: to standard output, unless --json)")
    sw.add_argument("--picture", metavar="PNG", help="a map of the windows: a track a measure for each sequence, a pixel a window")
    sw.add_argument(
        "--show", metavar="M,...", help="the picture's measures (default gc_skew,at_skew; also gc, ry_skew, km_skew, asymmetry_K ...)"
    )
    sw.add_argument("--json", action="store_true", help="the totals, by sequence and over all, as JSON")
    sw.add_argument("--reference", metavar="FASTA", help="reference FASTA or gpxil file, for files written with --no-ref")
    sw.set_defaults(fn=cmd_skew)

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
