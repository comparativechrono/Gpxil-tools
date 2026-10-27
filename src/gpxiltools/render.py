"""Rendering gpxil planes as pictures for screen (sRGB) or print (CMYK).

Rendering never changes the data: it maps codes to a small set of colour
classes and keeps the result as an indexed image plus two lookup tables,
one sRGB and one CMYK. The same rendering can therefore be written as a
palette PNG for the screen or as a CMYK TIFF/PDF for a printer, with no
colour conversion of the pixel data in between.

Modes:

``reference``
    The reference sequence, one pixel per base.
``genotype``
    The sample's genome: homozygous bases in their base colour, a
    heterozygous site in the mix of its two alleles (an A/G site uses the
    IUPAC code R = A|G and the palette's mixing rule), bases deleted on both
    haplotypes as paper.
``haplotypes``
    Two pixel rows per row of bases: haplotype 1 above haplotype 2.
    Lossless for substitutions and deletions; insertion anchors can be marked.
``variants``
    Faded reference with variant sites highlighted by zygosity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import codes as C
from .palettes import DEFAULT_PALETTE, Palette, get_palette, ref_colours

MODES = ("reference", "genotype", "haplotypes", "variants")

# variant-mode classes (indices into the class table)
_VAR_CLASSES = [
    ("reference (faded)", None),
    ("heterozygous", ((255, 140, 0), (0, 0.45, 1.0, 0.0))),
    ("homozygous alternate", ((200, 0, 0), (0.0, 1.0, 1.0, 0.1))),
    ("deletion", ((20, 20, 20), (0.0, 0.0, 0.0, 1.0))),
    ("insertion after base", ((0, 160, 80), (0.85, 0.0, 1.0, 0.0))),
    ("no call", ((160, 160, 160), (0.0, 0.0, 0.0, 0.4))),
    ("complex", ((140, 60, 170), (0.5, 0.8, 0.0, 0.0))),
]
INSERTION_MARK = ((255, 0, 255), (0.0, 1.0, 0.0, 0.0))


@dataclass
class Rendering:
    """An indexed image with sRGB and CMYK lookup tables."""

    index: np.ndarray  # (H, W) uint8
    rgb: np.ndarray  # (256, 3) uint8
    cmyk: np.ndarray  # (256, 4) uint8, ink 0-255
    legend: list[tuple[str, int]] = field(default_factory=list)
    bases_per_row: int = 0
    rows_per_line: int = 1  # pixel rows per row of bases (2 for haplotypes)
    start: int = 0
    end: int = 0
    title: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.index.shape

    def to_rgb(self) -> np.ndarray:
        return self.rgb[self.index]

    def to_cmyk(self) -> np.ndarray:
        return self.cmyk[self.index]

    def scaled(self, factor: int) -> Rendering:
        """Nearest-neighbour enlargement (each pixel becomes factor x factor)."""
        if factor == 1:
            return self
        idx = np.repeat(np.repeat(self.index, factor, axis=0), factor, axis=1)
        return Rendering(idx, self.rgb, self.cmyk, self.legend, self.bases_per_row, self.rows_per_line, self.start, self.end, self.title)

    def save_png(self, path, dpi: int | None = None) -> None:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        im = Image.fromarray(self.index, mode="P")
        im.putpalette(self.rgb.reshape(-1).tolist())
        kw = {"optimize": False, "compress_level": 9}
        if dpi:
            kw["dpi"] = (dpi, dpi)
        im.save(path, format="PNG", **kw)

    def save_tiff(self, path, colorspace: str = "rgb", dpi: int | None = None, icc_profile: bytes | None = None) -> None:
        """Write an uncompressed-palette-free TIFF: RGB, or CMYK for print."""
        import tifffile

        kw = {"compression": "zlib", "compressionargs": {"level": 9}, "metadata": None}
        if dpi:
            kw.update(resolution=(dpi, dpi), resolutionunit="INCH")
        if icc_profile:
            kw["iccprofile"] = icc_profile
        if colorspace == "rgb":
            tifffile.imwrite(path, self.to_rgb(), photometric="rgb", **kw)
        elif colorspace == "cmyk":
            tifffile.imwrite(path, self.to_cmyk(), photometric="separated", **kw)
        else:
            raise ValueError("colorspace must be 'rgb' or 'cmyk'")


def _lut_from_classes(classes: list[tuple[tuple, tuple]]) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.full((256, 3), 255, dtype=np.uint8)
    cmyk = np.zeros((256, 4), dtype=np.uint8)
    for i, (r, k) in enumerate(classes):
        rgb[i] = r
        cmyk[i] = np.round(np.array(k) * 255)
    return rgb, cmyk


def _wrap(values: np.ndarray, width: int, fill: int) -> np.ndarray:
    n = values.shape[0]
    rows = max(1, math.ceil(n / width))
    out = np.full(rows * width, fill, dtype=np.uint8)
    out[:n] = values
    return out.reshape(rows, width)


def _base_legend(pal: Palette) -> list[tuple[str, int]]:
    items = [("A", C.BASE_A), ("C", C.BASE_C), ("G", C.BASE_G), ("T", C.BASE_T)]
    mixes = [("A/C", 3), ("A/G", 5), ("A/T", 9), ("C/G", 6), ("C/T", 10), ("G/T", 12)]
    rest = [("N (unknown)", C.BASE_N), ("no base", C.NO_BASE)]
    soft = [("A, soft-masked repeat", C.BASE_A | C.SOFTMASK)] if pal.softmask_fade else []
    return items + mixes + soft + rest


def render_arrays(
    ref: np.ndarray,
    gt: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    *,
    mode: str = "genotype",
    palette: str | Palette = DEFAULT_PALETTE,
    bases_per_row: int = 8192,
    mark_insertions: bool = False,
    start: int = 0,
    title: str = "",
) -> Rendering:
    """Render per-base arrays (already cut to the interval of interest)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    pal = get_palette(palette) if isinstance(palette, str) else palette
    rgb_lut, cmyk_lut = ref_colours(pal)
    n = ref.shape[0]
    if gt is None:
        gt = np.zeros(n, dtype=np.uint8)
        if mode != "reference":
            mode = "reference"
    soft = np.bitwise_and(ref, C.SOFTMASK)

    if mode == "reference":
        idx = _wrap(ref, bases_per_row, C.NO_BASE)
        return Rendering(idx, rgb_lut, cmyk_lut, _base_legend(pal), bases_per_row, 1, start, start + n, title)

    if mode == "genotype":
        code = C.genotype_iupac(ref, gt) | soft
        code[code == C.SOFTMASK] = C.NO_BASE  # deleted on both haplotypes: paper
        idx = _wrap(code.astype(np.uint8), bases_per_row, C.NO_BASE)
        return Rendering(idx, rgb_lut, cmyk_lut, _base_legend(pal), bases_per_row, 1, start, start + n, title)

    if mode == "haplotypes":
        rgb = rgb_lut.copy()
        cmyk = cmyk_lut.copy()
        mark = 0xFE
        rgb[mark] = INSERTION_MARK[0]
        cmyk[mark] = np.round(np.array(INSERTION_MARK[1]) * 255)
        rows = []
        for h, bit in ((C.hap1(gt), C.FLAG_INS1), (C.hap2(gt), C.FLAG_INS2)):
            b = C.haplotype_bases(ref, h)
            keep_soft = (b != C.NO_BASE) & (b != C.BASE_N) | (h == C.HAP_REF)
            b = (b | np.where(keep_soft, soft, 0)).astype(np.uint8)
            if mark_insertions and flags is not None:
                b[(flags & bit) != 0] = mark
            rows.append(_wrap(b, bases_per_row, C.NO_BASE))
        idx = np.empty((rows[0].shape[0] * 2, bases_per_row), dtype=np.uint8)
        idx[0::2] = rows[0]
        idx[1::2] = rows[1]
        legend = _base_legend(pal) + ([("insertion follows", mark)] if mark_insertions else [])
        return Rendering(idx, rgb, cmyk, legend, bases_per_row, 2, start, start + n, title)

    # variants
    faded_rgb = rgb_lut.astype(float)
    faded_rgb = (faded_rgb + (255 - faded_rgb) * 0.8).astype(np.uint8)
    faded_cmyk = (cmyk_lut.astype(float) * 0.2).astype(np.uint8)
    h1, h2 = C.hap1(gt), C.hap2(gt)
    sub1, sub2 = np.isin(h1, C.HAP_SUB_CODES), np.isin(h2, C.HAP_SUB_CODES)
    cls = np.zeros(n, dtype=np.uint8)  # 0 = faded reference, encoded separately below
    het = (sub1 & (h2 == C.HAP_REF)) | (sub2 & (h1 == C.HAP_REF)) | (sub1 & sub2 & (h1 != h2))
    hom = (sub1 & sub2 & (h1 == h2)) | (sub1 & (h2 == C.HAP_ABSENT)) | (sub2 & (h1 == C.HAP_ABSENT))  # incl. haploid
    cls[het] = 1
    cls[hom] = 2
    cls[(h1 == C.HAP_DEL) | (h2 == C.HAP_DEL)] = 3
    if flags is not None:
        cls[(flags & (C.FLAG_INS1 | C.FLAG_INS2)) != 0] = 4
    cls[((h1 == C.HAP_UNKNOWN) | (h2 == C.HAP_UNKNOWN)) & (cls == 0)] = 5
    cls[(h1 == C.HAP_COMPLEX) | (h2 == C.HAP_COMPLEX)] = 6
    # indices: 0..31 faded reference codes, 0xF0+k variant classes
    out = np.where(cls == 0, ref, 0xF0 + cls).astype(np.uint8)
    rgb = faded_rgb.copy()
    cmyk = faded_cmyk.copy()
    legend = [("reference (faded)", C.BASE_A)]
    for k, (label, colours) in enumerate(_VAR_CLASSES):
        if colours is None:
            continue
        rgb[0xF0 + k] = colours[0]
        cmyk[0xF0 + k] = np.round(np.array(colours[1]) * 255)
        legend.append((label, 0xF0 + k))
    idx = _wrap(out, bases_per_row, C.NO_BASE)
    return Rendering(idx, rgb, cmyk, legend, bases_per_row, 1, start, start + n, title)


def render(
    contig,
    *,
    mode: str = "genotype",
    palette: str | Palette = DEFAULT_PALETTE,
    start: int = 0,
    end: int | None = None,
    bases_per_row: int | None = None,
    mark_insertions: bool = False,
) -> Rendering:
    """Render a :class:`~gpxiltools.container.Contig` (or an interval of it)."""
    end = contig.length if end is None else end
    if not contig.has("ref"):
        raise ValueError(f"{contig.name} has no ref plane; rendering needs the reference sequence")
    ref = contig.read("ref", start, end)
    gt = contig.read("gt", start, end) if contig.has("gt") else None
    flags = contig.read("flags", start, end) if contig.has("flags") else None
    bpr = bases_per_row or contig.width
    title = f"{contig.name}:{start + 1:,}-{end:,}"
    return render_arrays(
        ref, gt, flags, mode=mode, palette=palette, bases_per_row=bpr, mark_insertions=mark_insertions, start=start, title=title
    )


def class_counts(contig, *, bins: int, mode: str = "genotype", palette: str | Palette = DEFAULT_PALETTE, chunk: int = 1 << 23):
    """How many bases of each colour class fall in each of ``bins`` slices.

    Returns ``(counts, rgb_lut, cmyk_lut)`` with counts of shape (bins, 256);
    works chunk by chunk, so memory stays small even for chromosome 1.
    """
    n = contig.length
    edges = np.linspace(0, n, bins + 1).astype(np.int64)
    counts = np.zeros(bins * 256, dtype=np.int64)
    rgb = cmyk = None
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        r = render(contig, mode=mode, palette=palette, start=s, end=e, bases_per_row=e - s)
        rgb, cmyk = r.rgb, r.cmyk
        idx = r.index.reshape(-1).astype(np.int64)
        pos = np.arange(s, e)
        if r.rows_per_line == 2:  # haplotypes: both rows belong to the same positions
            pos = np.concatenate([pos, pos])
        b = np.searchsorted(edges, pos, side="right") - 1
        counts += np.bincount(b * 256 + idx[: pos.size], minlength=bins * 256)
    return counts.reshape(bins, 256), rgb, cmyk


def overview_rgb(contig, *, bins: int = 1024, palette: str | Palette = DEFAULT_PALETTE, mode: str = "genotype") -> np.ndarray:
    """Mean colour of each of ``bins`` equal slices of a contig, shape (bins, 3)."""
    counts, rgb, _ = class_counts(contig, bins=bins, mode=mode, palette=palette)
    counts = counts.astype(np.float64)
    tot = counts.sum(axis=1, keepdims=True)
    return np.round(counts @ rgb.astype(np.float64) / np.maximum(tot, 1)).astype(np.uint8)
