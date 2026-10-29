"""A stretch wrapped at its repeat period (``gpxil period``): each unit of the repeat a row, so that units line up.

The period is given, or found: for each period ``k`` up to a limit, the
share of the stretch's bases that equal the base ``k`` further on; the
period is the shortest whose share is within 0.05 of the best (so a repeat
of ``AAAG`` is period 4, not 8). A stretch with no clear period (a best
share under 0.6) still gets the best one, with a warning.

The reference is drawn, then, when the file has a sample, its two
haplotypes, side by side, a square a base, in the palette's colours:

- ``aligned`` (default): at the reference's positions, so a haplotype's row
  is the reference's row; a substituted base is outlined (in black or
  white, whichever stands out on its colour), a deleted base is a dash, an
  unknown one is in the palette's colour for N, a haplotype that is absent
  (haploid) is left blank, and an insertion after a base is a bar on that
  base's right edge (the inserted bases are listed in the text);
- ``sequence``: each haplotype's own sequence, insertions in and deletions
  out, wrapped at the same period, so a unit gained or lost shifts the
  units after it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import codes as C
from .palettes import ref_colours

PAPER, DASH, GRID = (255, 255, 255), (90, 90, 90), (215, 215, 215)


def best_period(codes: np.ndarray, max_period: int = 50) -> tuple[int, float, dict[int, float]]:
    """The period of a stretch of reference codes: ``(period, share, shares by period)``, where a period's share is
    the fraction of A/C/G/T bases equal to the base ``period`` further on."""
    x = np.asarray(codes, dtype=np.uint8) & 0x0F
    good = np.isin(x, C.HAP_SUB_CODES)
    shares: dict[int, float] = {}
    for k in range(1, min(max_period, x.size - 1) + 1):
        both = good[:-k] & good[k:]
        n = int(both.sum())
        shares[k] = float((x[:-k] == x[k:])[both].sum() / n) if n else 0.0
    if not shares:
        return 1, 0.0, shares
    best = max(shares.values())
    period = min(k for k, v in shares.items() if v >= best - 0.05)
    return period, shares[period], shares


@dataclass
class Panel:
    """One column of the picture: a label, and the cells, row by row (``codes``: reference codes, or one of the
    special colours below as negative numbers; ``outline``: cells to outline; ``ticks``: cells an insertion follows)."""

    label: str
    codes: list[int]
    outline: set[int] = field(default_factory=set)
    ticks: dict[int, str] = field(default_factory=dict)


def _contrast(rgb) -> tuple[int, int, int]:
    """Black or white, whichever stands out on a colour."""
    r, g, b = rgb
    return (0, 0, 0) if 0.2126 * r + 0.7152 * g + 0.0722 * b > 110 else (255, 255, 255)


def panels(view, start: int, end: int, *, layout: str = "aligned") -> list[Panel]:
    """The reference and, when there is a sample, its haplotypes over ``[start, end)`` of a contig or view."""
    if layout not in ("aligned", "sequence"):
        raise ValueError("layout must be 'aligned' or 'sequence'")
    if not view.covers("ref", start, end):
        raise ValueError(f"{view.region(start, end)} has no reference: give it with --reference")
    ref = view.read("ref", start, end)
    out = [Panel("reference", (ref & 0x0F).astype(int).tolist())]
    if not view.has("gt"):
        return out
    if layout == "sequence":
        from .export import haplotype_sequence

        for h in (1, 2):
            seq = haplotype_sequence(view, h, start, end, keep_case=False)
            out.append(Panel(f"haplotype {h}", (C.ASCII_TO_REF[np.frombuffer(seq, dtype=np.uint8)] & 0x0F).astype(int).tolist()))
        return out
    gt = view.read("gt", start, end)
    flags = view.read("flags", start, end) if view.has("flags") else np.zeros(end - start, dtype=np.uint8)
    ins = {}
    for e in view.sidecar():
        if e.kind == "INS" and start <= e.offset < end:
            ins[(e.hap, e.offset - start)] = e.data
    for h, states, bit in ((1, C.hap1(gt), C.FLAG_INS1), (2, C.hap2(gt), C.FLAG_INS2)):
        st = C.normalize_states(states)
        codes = (ref & 0x0F).astype(int)
        sub = np.isin(st, C.HAP_SUB_CODES)
        codes[sub] = st[sub]
        codes[st == C.HAP_DEL] = -1
        codes[(st == C.HAP_UNKNOWN) | (st == C.HAP_COMPLEX)] = -2
        codes[st == C.HAP_ABSENT] = -3
        ticks = {int(i): ins.get((h, int(i)), "") for i in np.flatnonzero((flags & bit) != 0)}
        out.append(Panel(f"haplotype {h}", codes.tolist(), set(np.flatnonzero(sub).tolist()), ticks))
    return out


def picture(panels: list[Panel], period: int, *, scale: int = 8, palette: str = "classic", title: str = ""):
    """The panels side by side, each wrapped at ``period``: an RGB :class:`PIL.Image.Image`."""
    from PIL import Image, ImageDraw

    rgb, _ = ref_colours(palette)
    px, gap, top, margin = max(2, scale), max(12, 2 * scale), 18, 8
    rows = max(-(-len(p.codes) // period) for p in panels)
    widths = [max(period * px, 6 * len(p.label) + 4) for p in panels]
    width = margin * 2 + sum(widths) + (len(panels) - 1) * gap
    key = "outlined: substituted; dash: deleted; bar at the right: an insertion follows"
    height = top + 16 + rows * px + margin + (14 if title else 0) + 14
    width = max(width, 6 * len(title) + 2 * margin, 6 * len(key) + 2 * margin)
    img = Image.new("RGB", (width, height), PAPER)
    d = ImageDraw.Draw(img)
    if title:
        d.text((margin, 2), title, fill=(60, 60, 60))
    y0 = top + (14 if title else 0)
    unknown = tuple(int(v) for v in rgb[C.BASE_N])
    x0 = margin
    for p, pw in zip(panels, widths):
        d.text((x0, y0 - 14), p.label, fill=(0, 0, 0))
        for i, code in enumerate(p.codes):
            x, y = x0 + (i % period) * px, y0 + (i // period) * px
            box = [x, y, x + px - 1, y + px - 1]
            if code == -3:  # (absent: blank)
                continue
            col = PAPER if code == -1 else unknown if code == -2 else tuple(int(v) for v in rgb[code])
            d.rectangle(box, fill=col, outline=GRID if px >= 6 else None)
            if code == -1:
                h = max(1, px // 5)
                d.rectangle([x + px // 5, y + (px - h) // 2, x + px - 1 - px // 5, y + (px - h) // 2 + h - 1], fill=DASH)
            if i in p.outline:
                d.rectangle(box, outline=_contrast(col), width=max(1, px // 5))
            if i in p.ticks:
                d.rectangle([x + px - max(2, px // 4), y, x + px - 1, y + px - 1], fill=_contrast(col))
        x0 += pw + gap
    d.text((margin, height - 14), key, fill=(90, 90, 90))
    return img


def text(panels: list[Panel], period: int) -> str:
    """The panels as text, side by side, a unit a line; ``-`` a deleted base, ``n`` an unknown one, ``.`` absent;
    a substituted base in lower case; ``+`` after a base an insertion follows."""

    def cells(p: Panel) -> list[str]:
        out = []
        for i, code in enumerate(p.codes):
            ch = {-1: "-", -2: "n", -3: "."}.get(code) or C.IUPAC[code]
            out.append((ch.lower() if i in p.outline else ch) + ("+" if i in p.ticks else ""))
        return out

    cols = [cells(p) for p in panels]
    lines = [[] for _ in range(max(-(-len(c) // period) for c in cols))]
    widths = []
    for c in cols:
        rows = ["".join(c[r * period : (r + 1) * period]) for r in range(len(lines))]
        w = max([len(x) for x in rows] + [0])
        widths.append(w)
        for r, x in enumerate(rows):
            lines[r].append(x)
    head = "  ".join(p.label.ljust(max(w, len(p.label))) for p, w in zip(panels, widths))
    body = ["  ".join(x.ljust(max(w, len(p.label))) for x, p, w in zip(row, panels, widths)).rstrip() for row in lines]
    inserted = [f"{p.label}: after base {i + 1} of the stretch, {s}" for p in panels for i, s in sorted(p.ticks.items()) if s]
    return "\n".join([head.rstrip(), *body] + (["", "insertions:", *inserted] if inserted else []))
