"""A printable "genome book": every base of a genome as one printed dot.

The PDF is written directly (no PDF library needed). Each data page embeds
one indexed image whose colour table is either DeviceCMYK (for a printer)
or DeviceRGB (for the screen). Pixels are drawn with interpolation off, so
each base stays a crisp square dot.
"""

from __future__ import annotations

import datetime as _dt
import math
import zlib
from dataclasses import dataclass

import numpy as np

from . import __version__
from .palettes import DEFAULT_PALETTE, get_palette
from .render import class_counts, render

PAGE_SIZES_PT = {
    "A3": (841.89, 1190.55),
    "A4": (595.28, 841.89),
    "A5": (419.53, 595.28),
    "letter": (612.0, 792.0),
}
MM = 72 / 25.4

# Helvetica advance widths (1/1000 em) for ASCII 32..126, from the standard AFM.
_HELV = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
]  # fmt: skip


def text_width(s: str, size: float, bold: bool = False) -> float:
    w = sum(_HELV[ord(ch) - 32] if 32 <= ord(ch) <= 126 else 556 for ch in s)
    return w * size / 1000 * (1.06 if bold else 1.0)


def _pdf_str(s: str) -> bytes:
    b = s.encode("cp1252", "replace")
    return b"(" + b.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


class _Pdf:
    """Just enough PDF 1.4 to place text and images on pages."""

    def __init__(self):
        self.objs: list[bytes | None] = []
        self.pages: list[int] = []
        self.pages_id = self.reserve()
        self.fonts = {
            "F1": self.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"),
            "F2": self.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"),
        }

    def reserve(self) -> int:
        self.objs.append(None)
        return len(self.objs)

    def add(self, body: bytes, oid: int | None = None) -> int:
        if oid is None:
            self.objs.append(body)
            return len(self.objs)
        self.objs[oid - 1] = body
        return oid

    def stream(self, dictionary: bytes, data: bytes, compress: bool = True) -> int:
        if compress:
            data = zlib.compress(data, 6)
            dictionary = dictionary + b" /Filter /FlateDecode"
        return self.add(b"<< " + dictionary + b" /Length %d >>\nstream\n" % len(data) + data + b"\nendstream")

    def image(self, data: np.ndarray, colorspace: bytes) -> int:
        h, w = data.shape[:2]
        return self.stream(
            b"/Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace %s /BitsPerComponent 8 /Interpolate false"
            % (w, h, colorspace),
            np.ascontiguousarray(data).tobytes(),
        )

    def page(self, size: tuple[float, float], content: bytes, images: dict[str, int]) -> None:
        xobj = b" ".join(b"/%s %d 0 R" % (k.encode(), v) for k, v in images.items())
        fonts = b" ".join(b"/%s %d 0 R" % (k.encode(), v) for k, v in self.fonts.items())
        cid = self.stream(b"", content)
        pid = self.add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.2f %.2f] /Resources << /Font << %s >> /XObject << %s >> >> /Contents %d 0 R >>"
            % (self.pages_id, size[0], size[1], fonts, xobj, cid)
        )
        self.pages.append(pid)

    def save(self, path, info: dict[str, str]) -> None:
        kids = b" ".join(b"%d 0 R" % p for p in self.pages)
        self.add(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(self.pages)), self.pages_id)
        catalog = self.add(b"<< /Type /Catalog /Pages %d 0 R >>" % self.pages_id)
        info_id = self.add(b"<< " + b" ".join(b"/%s %s" % (k.encode(), _pdf_str(v)) for k, v in info.items()) + b" >>")
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, body in enumerate(self.objs, start=1):
            offsets.append(len(out))
            out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(self.objs) + 1)
        for off in offsets:
            out += b"%010d 00000 n \n" % off
        out += b"trailer\n<< /Size %d /Root %d 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(self.objs) + 1, catalog, info_id, xref)
        with open(path, "wb") as fh:
            fh.write(out)


class _Content:
    def __init__(self):
        self.ops: list[bytes] = []

    def text(self, x: float, y: float, s: str, size: float = 10, bold: bool = False, grey: float = 0.0) -> None:
        font = b"F2" if bold else b"F1"
        self.ops.append(b"BT %.3f g /%s %.2f Tf %.2f %.2f Td %s Tj ET" % (grey, font, size, x, y, _pdf_str(s)))

    def text_centre(self, cx: float, y: float, s: str, size: float = 10, bold: bool = False, grey: float = 0.0) -> None:
        self.text(cx - text_width(s, size, bold) / 2, y, s, size, bold, grey)

    def image(self, name: str, x: float, y: float, w: float, h: float) -> None:
        self.ops.append(b"q %.4f 0 0 %.4f %.4f %.4f cm /%s Do Q" % (w, h, x, y, name.encode()))

    def rect(self, x, y, w, h, fill_rgb=None, fill_cmyk=None, stroke: float | None = None) -> None:
        op = b"q "
        if fill_cmyk is not None:
            op += b"%.3f %.3f %.3f %.3f k " % tuple(fill_cmyk)
        elif fill_rgb is not None:
            op += b"%.3f %.3f %.3f rg " % tuple(fill_rgb)
        if stroke is not None:
            op += b"%.2f w 0.6 G " % stroke
        paint = b"B" if (stroke is not None and (fill_rgb is not None or fill_cmyk is not None)) else (b"S" if stroke is not None else b"f")
        op += b"%.2f %.2f %.2f %.2f re %s Q" % (x, y, w, h, paint)
        self.ops.append(op)

    def bytes(self) -> bytes:
        return b"\n".join(self.ops)


@dataclass
class BookLayout:
    page: str = "A4"
    dpi: int = 300
    margin_mm: float = 15.0
    scale: int = 1  # printer dots per base, each way
    header_pt: float = 26.0
    footer_pt: float = 18.0

    @property
    def size(self) -> tuple[float, float]:
        return PAGE_SIZES_PT[self.page]

    def grid(self, rows_per_line: int = 1) -> tuple[int, int]:
        """(bases per row, rows of bases) that fit in the page's image area."""
        w, h = self.size
        m = self.margin_mm * MM
        dots_w = (w - 2 * m) * self.dpi / 72
        dots_h = (h - 2 * m - self.header_pt - self.footer_pt) * self.dpi / 72
        return int(dots_w // self.scale), int(dots_h // (self.scale * rows_per_line))

    def bases_per_page(self, rows_per_line: int = 1) -> int:
        a, b = self.grid(rows_per_line)
        return a * b


def estimate_pages(total_bases: int, n_contigs: int, layout: BookLayout | None = None, rows_per_line: int = 1) -> int:
    layout = layout or BookLayout()
    per = layout.bases_per_page(rows_per_line)
    return 2 + n_contigs + math.ceil(total_bases / per) + n_contigs // 2


def _palette_cs(rgb: np.ndarray, cmyk: np.ndarray, colorspace: str) -> bytes:
    if colorspace == "cmyk":
        return b"[/Indexed /DeviceCMYK 255 <" + cmyk.astype(np.uint8).tobytes().hex().encode() + b">]"
    return b"[/Indexed /DeviceRGB 255 <" + rgb.astype(np.uint8).tobytes().hex().encode() + b">]"


def make_book(
    gfile,
    out_path,
    *,
    contigs: list[str] | None = None,
    mode: str = "genotype",
    palette: str = DEFAULT_PALETTE,
    colorspace: str = "cmyk",
    layout: BookLayout | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    max_pages: int | None = None,
    log=None,
) -> dict:
    """Write a PDF book of a gpxil file, one printed dot per base."""
    layout = layout or BookLayout()
    log = log or (lambda m: None)
    if colorspace not in ("cmyk", "rgb"):
        raise ValueError("colorspace must be 'cmyk' or 'rgb'")
    names = contigs or [c for c in gfile.contigs if gfile[c].has("ref")]
    pal = get_palette(palette)
    sample = gfile.sample_id
    ref_name = (gfile.metadata.get("reference") or {}).get("name", "")
    title = title or (f"The genome of {sample}" if sample else f"The {ref_name or 'reference'} genome")
    subtitle = subtitle or (f"Reference {ref_name}" if ref_name else "")
    pdf = _Pdf()
    W, H = layout.size
    m = layout.margin_mm * MM
    cs_name = b"/DeviceCMYK" if colorspace == "cmyk" else b"/DeviceRGB"
    page_no = 0
    total_bases = sum(gfile[n].length for n in names)
    rows_per_line = 2 if mode == "haplotypes" else 1
    bpr, nrows = layout.grid(rows_per_line)
    per_page = bpr * nrows

    def footer(c: _Content):
        nonlocal page_no
        page_no += 1
        c.text_centre(W / 2, m * 0.6, str(page_no), 8, grey=0.35)

    # ------------------------------------------------------------- cover
    log("cover")
    c = _Content()
    c.text_centre(W / 2, H - m - 90, title, 26, bold=True)
    if subtitle:
        c.text_centre(W / 2, H - m - 118, subtitle, 12, grey=0.3)
    c.text_centre(W / 2, H - m - 138, f"{total_bases:,} bases, one printed dot each", 10, grey=0.3)
    longest = max(gfile[n].length for n in names)
    bar_w_max = W - 2 * m - 50
    top = H - m - 175
    bar_h = min(14.0, (top - m - 40) / max(1, len(names)) * 0.7)
    gap = bar_h * 0.45
    images = {}
    for i, name in enumerate(names):
        contig = gfile[name]
        bins = max(8, int(round(256 * contig.length / longest)))
        counts, rgb_lut, cmyk_lut = class_counts(contig, bins=bins, mode=mode if mode != "haplotypes" else "genotype", palette=pal)
        counts = counts.astype(np.float64)
        tot = np.maximum(counts.sum(axis=1, keepdims=True), 1)
        lut = cmyk_lut if colorspace == "cmyk" else rgb_lut
        stripe = np.round(counts @ lut.astype(np.float64) / tot).astype(np.uint8)[None, :, :]
        key = f"S{i}"
        images[key] = pdf.image(stripe, cs_name)
        y = top - i * (bar_h + gap)
        bw = bar_w_max * contig.length / longest
        c.text(m, y + bar_h * 0.2, name[:12], min(8, bar_h * 0.8), grey=0.25)
        c.image(key, m + 50, y, bw, bar_h)
    c.text_centre(W / 2, m + 12, f"gpxil {gfile.metadata.get('gpxil')} · gpxiltools {__version__}", 7, grey=0.5)
    footer(c)
    pdf.page((W, H), c.bytes(), images)

    # ------------------------------------------------------------- how to read
    c = _Content()
    y = H - m - 30
    c.text(m, y, "How to read this book", 16, bold=True)
    y -= 26
    lines = [
        f"Every base of DNA is one square dot, {25.4 * layout.scale / layout.dpi:.3f} mm wide ({layout.dpi} dots per inch).",
        f"Dots run left to right in rows of {bpr:,} bases, rows run top to bottom, like text.",
        f"A full page holds {per_page:,} bases. The page header gives the first and last position.",
        "Pale dots are soft-masked repeats in the reference." if pal.softmask_fade else "",
        "Where the two inherited copies differ (heterozygous), the dot mixes both colours." if mode == "genotype" else "",
        "Each row of bases is drawn twice: the copy from one parent above the other." if mode == "haplotypes" else "",
        "Grey dots are unknown: gaps in the reference (N), or bases not confidently called in this genome.",
        "White space is where a base is missing (deleted on both copies, or past the end of a chromosome).",
    ]
    for line in [t for t in lines if t]:
        c.text(m, y, line, 10)
        y -= 15
    y -= 12
    sample_r = render(gfile[names[0]], mode=mode, palette=pal, start=0, end=1, bases_per_row=1)
    legend = sample_r.legend
    sw = 16
    col_x = [m, m + (W - 2 * m) / 2]
    for i, (label, idx) in enumerate(legend):
        col = i % 2
        row = i // 2
        yy = y - row * (sw + 8)
        if colorspace == "cmyk":
            c.rect(col_x[col], yy, sw, sw, fill_cmyk=sample_r.cmyk[idx] / 255, stroke=0.3)
        else:
            c.rect(col_x[col], yy, sw, sw, fill_rgb=sample_r.rgb[idx] / 255, stroke=0.3)
        c.text(col_x[col] + sw + 8, yy + 4, label, 10)
    yb = y - ((len(legend) + 1) // 2) * (sw + 8) - 20
    c.text(m, yb, f"Palette: {pal.name} — {pal.description}", 8, grey=0.35)
    if colorspace == "cmyk":
        c.text(m, yb - 12, "Colours are device CMYK ink values; proof with your printer's ICC profile.", 8, grey=0.35)
    footer(c)
    pdf.page((W, H), c.bytes(), {})

    # ------------------------------------------------------------- chapters
    pages_written = 2
    stop = False
    for name in names:
        contig = gfile[name]
        c = _Content()
        c.text_centre(W / 2, H / 2 + 20, name, 40, bold=True)
        c.text_centre(W / 2, H / 2 - 10, f"{contig.length:,} bases", 12, grey=0.3)
        n_pages = math.ceil(contig.length / per_page)
        c.text_centre(W / 2, H / 2 - 28, f"{n_pages} page{'s' if n_pages != 1 else ''}", 10, grey=0.4)
        footer(c)
        pdf.page((W, H), c.bytes(), {})
        pages_written += 1
        for p in range(n_pages):
            if max_pages is not None and pages_written >= max_pages:
                stop = True
                break
            s = p * per_page
            e = min(contig.length, s + per_page)
            r = render(contig, mode=mode, palette=pal, start=s, end=e, bases_per_row=bpr)
            cs = _palette_cs(r.rgb, r.cmyk, colorspace)
            img_id = pdf.image(r.index, cs)
            rows_px = r.index.shape[0]
            dot = 72 / layout.dpi * layout.scale
            iw = bpr * dot
            ih = rows_px * dot
            c = _Content()
            c.text(m, H - m - 10, f"{name}", 9, bold=True)
            label = f"{s + 1:,} – {e:,}"
            c.text(W - m - text_width(label, 9), H - m - 10, label, 9, grey=0.25)
            c.image("Im", m, H - m - layout.header_pt - ih, iw, ih)
            footer(c)
            pdf.page((W, H), c.bytes(), {"Im": img_id})
            pages_written += 1
            if pages_written % 20 == 0:
                log(f"  {pages_written} pages")
        if stop:
            break
    pdf.save(
        out_path,
        {
            "Title": title,
            "Creator": f"gpxiltools {__version__}",
            "Producer": "gpxiltools book writer",
            "CreationDate": _dt.datetime.now().strftime("D:%Y%m%d%H%M%S"),
        },
    )
    return {"pages": len(pdf.pages), "bases_per_page": per_page, "bases_per_row": bpr, "rows_per_page": nrows, "truncated": stop}
