"""Figures for the gpxil specification and README.

    python make_figures.py --chr21 HG002_chr21.gpxil.tif --bench bench_chr21.json \
        [--genome HG002_GRCh38.gpxil.tif] [--book book_proof.pdf] --out docs/img
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from gpxiltools import codes as C  # noqa: E402
from gpxiltools.container import GpxilFile  # noqa: E402
from gpxiltools.layout import hilbert_image  # noqa: E402
from gpxiltools.palettes import PALETTES  # noqa: E402
from gpxiltools.render import class_counts, render  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
BLUE = "#2a78d6"
SURFACE = "#fcfcfb"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def save_index(r, path, scale=1):
    r.scaled(scale).save_png(path)


def fig_region(c, out):
    r = render(c, mode="genotype", start=30_000_000, end=30_250_000, bases_per_row=500)
    save_index(r, os.path.join(out, "region-genotype.png"))


def fig_modes(c, out, start=36_496_384, bpr=64, rows=16, scale=9):
    end = start + bpr * rows
    panels = []
    for mode in ("genotype", "haplotypes", "variants"):
        r = render(c, mode=mode, start=start, end=end, bases_per_row=bpr, mark_insertions=True)
        panels.append((mode, r.scaled(scale).to_rgb()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1, 1, 1]})
    titles = {
        "genotype": "genotype: het sites show the mix of both alleles",
        "haplotypes": "haplotypes: copy 1 above copy 2 (magenta = insertion)",
        "variants": "variants: orange het, red hom, black deletion, green insertion",
    }
    for ax, (mode, img) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(titles[mode], fontsize=9.5, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"HG002, chr21:{start + 1:,}-{end:,} ({bpr} bases per row, one square per base)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "render-modes.png"), dpi=110)
    plt.close(fig)


def fig_palettes(c, out, start=30_100_000, bpr=160, rows=60, scale=3):
    end = start + bpr * rows
    fig, axes = plt.subplots(1, len(PALETTES), figsize=(4.2 * len(PALETTES), 2.2))
    for ax, name in zip(axes, PALETTES):
        r = render(c, mode="genotype", palette=name, start=start, end=end, bases_per_row=bpr)
        ax.imshow(r.scaled(scale).to_rgb(), interpolation="nearest")
        ax.set_title(f"{name}", fontsize=10, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "palettes.png"), dpi=100)
    plt.close(fig)


def fig_hilbert(c, out):
    r = render(c, mode="reference", bases_per_row=c.length)
    img, order = hilbert_image(r.index.reshape(-1)[: c.length])
    rgb = r.rgb[img]
    small = Image.fromarray(rgb).resize((1024, 1024), Image.BOX)
    small.save(os.path.join(out, "hilbert-chr21.png"))


def fig_image_analysis(c, out, bin_rows=8, width=4096):
    """GC content and variant density computed with plain array/image operations."""
    from skimage.measure import block_reduce

    ref = c.read("ref")
    gt = c.read("gt")
    n = ref.size
    rows = -(-n // width)
    pad = rows * width - n

    def img(a):
        return np.concatenate([a, np.zeros(pad, a.dtype)]).reshape(rows, width)

    base = ref & 0x0F
    acgt = img(np.isin(base, C.HAP_SUB_CODES).astype(np.float32))
    gc = img(np.isin(base, (C.BASE_C, C.BASE_G)).astype(np.float32))
    var = img((gt != 0) & (gt != C.GT_UNKNOWN)).astype(np.float32)
    blk = (bin_rows, 64)
    n_acgt = block_reduce(acgt, blk, np.sum)
    gc_frac = np.where(n_acgt > blk[0] * blk[1] * 0.5, block_reduce(gc, blk, np.sum) / np.maximum(n_acgt, 1), np.nan)
    dens = np.where(n_acgt > blk[0] * blk[1] * 0.5, block_reduce(var, blk, np.sum) / np.maximum(n_acgt, 1) * 1000, np.nan)
    from matplotlib.colors import LinearSegmentedColormap

    blues = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    blues.set_bad("#e9e8e4")
    oranges = LinearSegmentedColormap.from_list("seqorange", ["#fde3d6", "#f7b08f", "#eb6834", "#b84517", "#7a2c0d"])
    oranges.set_bad("#e9e8e4")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    kb = bin_rows * 64
    for ax, data, cmap, label in (
        (axes[0], gc_frac * 100, blues, "GC content, %"),
        (axes[1], dens, oranges, "variant bases per kb (HG002)"),
    ):
        vmin, vmax = np.nanpercentile(data, [2, 98])
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(label, fontsize=10, loc="left")
    fig.suptitle(
        f"chr21 as {rows:,} rows of {width:,} bases, block-averaged over {bin_rows} x 64 pixels ({kb:,} bases per block); grey: N gaps",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out, "image-analysis.png"), dpi=110)
    plt.close(fig)


def fig_karyotype(g, out, names=None, bins_longest=600):
    names = names or [n for n in g.contigs if g[n].length > 1_000_000]
    longest = max(g[n].length for n in names)
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(names) + 0.6))
    for i, name in enumerate(names):
        cobj = g[name]
        bins = max(4, int(round(bins_longest * cobj.length / longest)))
        counts, rgb, _ = class_counts(cobj, bins=bins, mode="genotype")
        counts = counts.astype(float)
        stripe = counts @ rgb.astype(float) / np.maximum(counts.sum(1, keepdims=True), 1)
        ax.imshow(stripe[None].astype(np.uint8), extent=(0, cobj.length / 1e6, i + 0.85, i + 0.15), aspect="auto", interpolation="nearest")
        ax.text(-2, i + 0.5, name, ha="right", va="center", fontsize=8, color=INK2)
    ax.set_ylim(len(names), 0)
    ax.set_xlim(-0.5, longest / 1e6 + 1)
    ax.set_yticks([])
    ax.set_xlabel("position (Mb)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        "HG002 on GRCh38: mean colour of each slice of every chromosome\n"
        "dark grey: assembly gaps (N); light grey: soft-masked or not confidently called",
        loc="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out, "karyotype.png"), dpi=110)
    plt.close(fig)


def fig_bench(bench, out):
    rows = bench["reference"]
    pick = [
        ("FASTA.gz as distributed (Ensembl)", "FASTA.gz (Ensembl)", False),
        ("UCSC .2bit (exact size)", "UCSC .2bit", False),
        ("PNG, 24-bit RGB colours (notebook approach)", "PNG, RGB colours (notebooks)", False),
        ("PNG, 8-bit palette, level 9", "PNG, 8-bit codes", False),
        ("JPEG XL lossless, effort 7 (grey codes)", "JPEG XL lossless", False),
        ("WebP lossless (palette)", "WebP lossless", False),
        ("TIFF, Hilbert layout (order 13), deflate 9, 256 tiles", "TIFF, Hilbert layout", False),
        ("TIFF, deflate 9, 512x512 tiles", "TIFF, tiles", False),
        ("gpxil TIFF, deflate 9, 256 KiB strips", "gpxil, Deflate (default)", True),
        ("gpxil TIFF, zstd 19, 256 KiB strips", "gpxil, Zstandard 19", True),
        ("gpxil TIFF, lzma, 256 KiB strips", "gpxil, LZMA", True),
    ]
    by = {r["name"]: r for r in rows}
    items = [(label, by[k], g) for k, label, g in pick if k in by]
    items.sort(key=lambda t: t[1]["bits_per_base"])
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = np.arange(len(items))
    vals = [t[1]["bits_per_base"] for t in items]
    cols = [BLUE if t[2] else "#b9b8b3" for t in items]
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, v, t in zip(y, vals, items):
        ax.text(v + 0.03, yi, f"{v:.2f}  ({t[1]['bytes'] / 1e6:.1f} MB)", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [t[0] for t in items], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("bits per base, chr21 reference incl. soft-masking (46.7 Mbp)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, max(vals) * 1.32)
    ax.set_title("Size: gpxil (blue) vs other ways to store the same sequence", loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bench-size.png"), dpi=110)
    plt.close(fig)

    pick2 = [
        ("gpxil TIFF, deflate 9, 64 KiB strips", "gpxil, 64 KiB strips", True),
        ("gpxil TIFF, deflate 9, 256 KiB strips", "gpxil, 256 KiB strips (default)", True),
        ("gpxil TIFF, deflate 9, 1 MiB strips", "gpxil, 1 MiB strips", True),
        ("TIFF, deflate 9, 512x512 tiles", "TIFF, 512 x 512 tiles", False),
        ("PNG, 8-bit palette, level 9", "PNG (decode all)", False),
        ("WebP lossless (palette)", "WebP (decode all)", False),
        ("JPEG XL lossless, effort 7 (grey codes)", "JPEG XL (decode all)", False),
    ]
    items = [(label, by[k], g) for k, label, g in pick2 if k in by and by[k]["random_access_ms"]]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    y = np.arange(len(items))
    vals = [t[1]["random_access_ms"] for t in items]
    ax.barh(y, vals, color=[BLUE if t[2] else "#b9b8b3" for t in items], height=0.62)
    ax.set_xscale("log")
    for yi, v in zip(y, vals):
        ax.text(v * 1.12, yi, f"{v:,.2f} ms" if v < 10 else f"{v:,.0f} ms", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [t[0] for t in items], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("time to read one random 1 kb interval (log scale)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, which="major")
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(min(vals) / 2, max(vals) * 8)
    ax.set_title("Random access: strips of whole rows map genomic intervals to few bytes", loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bench-access.png"), dpi=110)
    plt.close(fig)


def fig_book(pdf, out):
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", "40", "-png", "-l", "5", pdf, os.path.join(td, "p")], check=True)
        files = sorted(f for f in os.listdir(td) if f.endswith(".png"))
        ims = [Image.open(os.path.join(td, f)).convert("RGB") for f in files]
    gap = 10
    w = sum(i.size[0] for i in ims) + gap * (len(ims) + 1)
    h = max(i.size[1] for i in ims) + 2 * gap
    canvas = Image.new("RGB", (w, h), (233, 232, 228))
    x = gap
    for im in ims:
        canvas.paste(im, (x, gap))
        x += im.size[0] + gap
    canvas.save(os.path.join(out, "book-pages.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chr21", required=True)
    ap.add_argument("--bench")
    ap.add_argument("--genome")
    ap.add_argument("--book")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    want = set(a.only or ["region", "modes", "palettes", "hilbert", "analysis", "bench", "karyotype", "book"])
    with GpxilFile(a.chr21) as g:
        c = g[g.contigs[0]]
        if "region" in want:
            fig_region(c, a.out)
        if "modes" in want:
            fig_modes(c, a.out)
        if "palettes" in want:
            fig_palettes(c, a.out)
        if "hilbert" in want:
            fig_hilbert(c, a.out)
        if "analysis" in want:
            fig_image_analysis(c, a.out)
    if a.bench and "bench" in want:
        fig_bench(json.load(open(a.bench)), a.out)
    if a.genome and "karyotype" in want:
        with GpxilFile(a.genome) as g:
            primary = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
            order = [n for n in primary if n in g]
            fig_karyotype(g, a.out, order)
    if a.book and "book" in want:
        fig_book(a.book, a.out)


if __name__ == "__main__":
    main()
