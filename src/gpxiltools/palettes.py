"""Display palettes: integer codes -> sRGB (screen) and CMYK (print).

Palettes are rendering choices, not data. A gpxil file stores codes; the
same file can be shown with any palette, on screen in sRGB or printed in
CMYK, without converting pixel data between colour spaces.

Each palette defines the four bases in both colour spaces. Other IUPAC codes,
soft-masked bases and haplotype mixtures are derived by fixed rules
(:func:`ref_colours`), so a palette is fully described by a handful of values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import codes as C

RGB = tuple[int, int, int]
CMYK = tuple[float, float, float, float]  # ink coverage 0..1


@dataclass(frozen=True)
class Palette:
    name: str
    description: str
    rgb: dict[str, RGB]  # keys: A C G T N gap
    cmyk: dict[str, CMYK]
    #: how two- and three-base IUPAC codes (and heterozygous sites) are shown:
    #: "mean" averages the component colours; "overprint" lays down every
    #: component ink at full strength (subtractive mixing on paper).
    mix: str = "mean"
    #: soft-masked (repeat) bases are blended this far towards paper white
    softmask_fade: float = 0.45
    notes: str = ""
    extra: dict = field(default_factory=dict)


_PAPER_RGB: RGB = (255, 255, 255)
_PAPER_CMYK: CMYK = (0.0, 0.0, 0.0, 0.0)

PALETTES: dict[str, Palette] = {
    "classic": Palette(
        name="classic",
        description="A red, C blue, G green, T yellow: the colours of the original gpxil notebooks.",
        rgb={"A": (255, 0, 0), "C": (0, 0, 255), "G": (0, 255, 0), "T": (255, 255, 0), "N": (128, 128, 128), "gap": _PAPER_RGB},
        cmyk={
            "A": (0.0, 1.0, 1.0, 0.0),
            "C": (1.0, 0.8, 0.0, 0.0),
            "G": (0.8, 0.0, 1.0, 0.0),
            "T": (0.0, 0.0, 1.0, 0.0),
            "N": (0.0, 0.0, 0.0, 0.5),
            "gap": _PAPER_CMYK,
        },
        notes="Pure RGB green and blue are outside every print gamut; the CMYK values are the nearest process mixes.",
    ),
    "gpxi": Palette(
        name="gpxi",
        description="Albrecht-Buehler (2009) GPxI grey levels: A black, C dark grey, T light grey, G white.",
        rgb={"A": (0, 0, 0), "C": (85, 85, 85), "G": (255, 255, 255), "T": (170, 170, 170), "N": (170, 200, 235), "gap": (255, 250, 230)},
        cmyk={
            "A": (0.0, 0.0, 0.0, 1.0),
            "C": (0.0, 0.0, 0.0, 0.67),
            "G": (0.0, 0.0, 0.0, 0.0),
            "T": (0.0, 0.0, 0.0, 0.33),
            "N": (0.3, 0.1, 0.0, 0.0),
            "gap": (0.0, 0.02, 0.1, 0.0),
        },
        softmask_fade=0.0,
        notes="Grey levels follow PLoS ONE 4(3):e4701. N and padding get pale tints so they are not mistaken for G.",
    ),
    "okabe-ito": Palette(
        name="okabe-ito",
        description="Colour-blind-safe Okabe-Ito colours in the IGV convention: A green, C blue, G orange, T vermillion.",
        rgb={"A": (0, 158, 115), "C": (0, 114, 178), "G": (230, 159, 0), "T": (213, 94, 0), "N": (153, 153, 153), "gap": _PAPER_RGB},
        cmyk={
            "A": (0.97, 0.0, 0.75, 0.0),
            "C": (1.0, 0.5, 0.0, 0.0),
            "G": (0.0, 0.5, 1.0, 0.0),
            "T": (0.0, 0.8, 1.0, 0.0),
            "N": (0.0, 0.0, 0.0, 0.4),
            "gap": _PAPER_CMYK,
        },
    ),
    "process": Palette(
        name="process",
        description="One process ink per base: A cyan, C magenta, G yellow, T black. A heterozygous site prints both inks.",
        rgb={"A": (0, 174, 239), "C": (236, 0, 140), "G": (255, 242, 0), "T": (35, 31, 32), "N": (160, 160, 160), "gap": _PAPER_RGB},
        cmyk={
            "A": (1.0, 0.0, 0.0, 0.0),
            "C": (0.0, 1.0, 0.0, 0.0),
            "G": (0.0, 0.0, 1.0, 0.0),
            "T": (0.0, 0.0, 0.0, 1.0),
            "N": (0.0, 0.0, 0.0, 0.35),
            "gap": _PAPER_CMYK,
        },
        mix="overprint",
        notes=(
            "Here the 4-bit IUPAC code is literally a 1-bit-per-ink CMYK pixel: bit A -> cyan, C -> magenta, "
            "G -> yellow, T -> black. sRGB values are approximate screen previews of the inks."
        ),
    ),
}

DEFAULT_PALETTE = "classic"
_BASES = ("A", "C", "G", "T")


def get_palette(name: str) -> Palette:
    try:
        return PALETTES[name]
    except KeyError:
        raise ValueError(f"unknown palette {name!r}; choose from {', '.join(PALETTES)}") from None


def _mix_rgb(cols: list[RGB], how: str) -> RGB:
    arr = np.array(cols, dtype=float)
    if how == "overprint":
        # multiply transmittances, a crude model of stacked inks
        out = np.prod(arr / 255.0, axis=0) * 255.0
    else:
        out = arr.mean(axis=0)
    return tuple(int(round(v)) for v in out)


def _mix_cmyk(cols: list[CMYK], how: str) -> CMYK:
    arr = np.array(cols, dtype=float)
    out = arr.max(axis=0) if how == "overprint" else arr.mean(axis=0)
    return tuple(float(v) for v in out)


def _fade_rgb(c: RGB, t: float) -> RGB:
    return tuple(int(round(v + (255 - v) * t)) for v in c)


def _fade_cmyk(c: CMYK, t: float) -> CMYK:
    return tuple(v * (1 - t) for v in c)


def ref_colours(palette: str | Palette = DEFAULT_PALETTE) -> tuple[np.ndarray, np.ndarray]:
    """Lookup tables for every ref-plane code.

    Returns ``(rgb, cmyk)``: arrays of shape (256, 3) uint8 and (256, 4) uint8
    (ink percentage scaled to 0-255). Codes outside the defined range map to
    the gap colour.
    """
    pal = get_palette(palette) if isinstance(palette, str) else palette
    rgb = np.zeros((256, 3), dtype=np.uint8)
    cmyk = np.zeros((256, 4), dtype=np.uint8)
    rgb[:] = pal.rgb["gap"]
    cmyk[:] = np.round(np.array(pal.cmyk["gap"]) * 255)
    for code in range(16):
        if code == C.NO_BASE:
            c_rgb, c_cmyk = pal.rgb["gap"], pal.cmyk["gap"]
        elif code == C.BASE_N:
            c_rgb, c_cmyk = pal.rgb["N"], pal.cmyk["N"]
        else:
            members = [b for i, b in enumerate(_BASES) if code & (1 << i)]
            c_rgb = _mix_rgb([pal.rgb[b] for b in members], pal.mix)
            c_cmyk = _mix_cmyk([pal.cmyk[b] for b in members], pal.mix)
        for masked in (False, True):
            idx = code | (C.SOFTMASK if masked else 0)
            r, k = c_rgb, c_cmyk
            if masked and code not in (C.NO_BASE,):
                r, k = _fade_rgb(r, pal.softmask_fade), _fade_cmyk(k, pal.softmask_fade)
            rgb[idx] = r
            cmyk[idx] = np.round(np.array(k) * 255)
    return rgb, cmyk


def tiff_colormap(rgb_lut: np.ndarray) -> np.ndarray:
    """(256, 3) uint8 LUT -> TIFF ColorMap array (3, 256) uint16."""
    return (rgb_lut.astype(np.uint16).T * 257).astype(np.uint16)


# Fixed viewing colours for the raw gt and flags pages embedded in a file.
# They only make the pages readable in an ordinary image viewer; renderers
# use richer modes (see gpxiltools.render).
_GT_VIEW = {
    "refref": (245, 245, 245),
    "het": (255, 140, 0),
    "hom": (200, 0, 0),
    "del": (20, 20, 20),
    "unknown": (150, 150, 150),
    "complex": (140, 60, 170),
    "haploid_ref": (230, 236, 245),
    "haploid_alt": (200, 0, 0),
}


def gt_view_colours() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for v in range(256):
        a, b = v >> 4, v & 0xF
        states = {a, b}
        if C.HAP_DEL in states:
            col = _GT_VIEW["del"]
        elif C.HAP_COMPLEX in states:
            col = _GT_VIEW["complex"]
        elif C.HAP_UNKNOWN in states:
            col = _GT_VIEW["unknown"]
        elif C.HAP_ABSENT in states:
            other = a if b == C.HAP_ABSENT else b
            col = _GT_VIEW["haploid_ref"] if other in (C.HAP_REF, C.HAP_ABSENT) else _GT_VIEW["haploid_alt"]
        elif a == C.HAP_REF and b == C.HAP_REF:
            col = _GT_VIEW["refref"]
        elif a == C.HAP_REF or b == C.HAP_REF or a != b:
            col = _GT_VIEW["het"]
        else:
            col = _GT_VIEW["hom"]
        lut[v] = col
    return lut


def flags_view_colours() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for v in range(256):
        if v & C.FLAG_FILTERED:
            col = (220, 30, 30)
        elif v & (C.FLAG_INS1 | C.FLAG_INS2):
            col = (0, 160, 80)
        elif v & C.FLAG_VARIANT_SITE:
            col = (60, 110, 220) if v & C.FLAG_PHASED else (120, 150, 200)
        elif v & C.FLAG_PHASED:
            col = (200, 210, 235)
        else:
            col = (250, 250, 250)
        lut[v] = col
    return lut


def describe() -> list[dict]:
    """Palette definitions as plain data (for docs and metadata)."""
    out = []
    for p in PALETTES.values():
        out.append(
            {
                "name": p.name,
                "description": p.description,
                "mix": p.mix,
                "softmask_fade": p.softmask_fade,
                "srgb": {k: list(v) for k, v in p.rgb.items()},
                "cmyk_percent": {k: [round(x * 100) for x in v] for k, v in p.cmyk.items()},
                "notes": p.notes,
            }
        )
    return out
