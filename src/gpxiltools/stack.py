"""Features stacked, one a row, lined up at an anchor (``gpxil stack``).

Albrecht-Buehler saw Alu elements line up upstream of the pure GA-sequences
that begin with poly(A) by stacking a chromosome's GA-sequences, one a row,
lined up at their starts, and sorting the rows by the poly(A) (PLoS ONE
2009, Figs 1, 3 and 4). ``gpxil stack`` makes that picture of any features:
a BED file's intervals, or a RepeatMasker table's repeats.

Each feature is a row of ``up`` bases before its anchor and ``down`` from it,
the anchor its start, its end or its centre, read on the feature's own strand:
a feature on the - strand is read as the reverse complement (its sample's
substitutions complemented too, an insertion after the base it follows on
that strand), so that every row runs 5' to 3' and column ``up`` is the first
base after the anchor. A feature without a strand is read on the + strand.
Where a row runs off its sequence, or out of the windows that hold it, it
has no bases (paper, in the picture).

Rows are sorted as the features come (``file``), by ``position``,
``length``, ``score`` (a RepeatMasker repeat's: its divergence from its
consensus, which grows with its age), ``name``, or ``run``: the length of
the run of a set of bases (A: a poly(A)) that starts at the anchor, allowing
a few other bases in it (``tolerance``), as his Fig. 4 sorts them. Lengths,
scores and runs go longest first.

The rows are drawn as ``gpxil render`` draws bases (:func:`picture`), and
:func:`profile` counts each column's bases (and, for a sample, its
variants): the picture as numbers.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from . import codes as C
from .features import CHUNK, COMPLEMENT_SET, Feature, base_set, sequence_name, set_name

ANCHORS = ("start", "end", "centre")
SORTS = ("file", "position", "length", "score", "name", "run")
_LONGEST_FIRST = ("length", "score", "run")
_INS = C.FLAG_INS1 | C.FLAG_INS2


def _state_complement() -> np.ndarray:
    """gt byte -> the gt byte of the other strand: each haplotype's substitution complemented."""
    nib = np.arange(16)
    comp = np.where(np.isin(nib, C.HAP_SUB_CODES), COMPLEMENT_SET[nib], nib)
    v = np.arange(256)
    return ((comp[v >> 4] << 4) | comp[v & 0x0F]).astype(np.uint8)


GT_COMPLEMENT = _state_complement()


@dataclass
class Row:
    """A feature's row: its bases ``[lo, hi)`` of the sequence (on the + strand), read on ``strand``."""

    feature: Feature
    index: int  #: the feature's place among those given (0-based)
    sequence: str  #: the sequence's name, as the file has it
    lo: int
    hi: int
    strand: str  #: '+' or '-'
    key: object = None  #: what the rows were sorted by


@dataclass
class Stack:
    """The rows, in the order drawn, and their bases: ``ref`` (rows, up + down) reference codes, each row 5' to 3' on
    its strand (0: no base); for a sample, ``gt`` and ``flags`` likewise (else None)."""

    rows: list[Row]
    ref: np.ndarray
    gt: np.ndarray | None
    flags: np.ndarray | None
    up: int
    down: int
    anchor: str
    sort: str
    run_of: str
    features: int  #: how many features were given
    unplaced: int  #: how many are on no sequence of the file
    dropped: int  #: how many were left out by ``max_rows``
    reversed: bool = False  #: sorted the other way

    @property
    def width(self) -> int:
        return self.up + self.down


def _interval(f: Feature, strand: str, anchor: str, up: int, down: int) -> tuple[int, int]:
    off = {"start": 0, "end": f.length, "centre": f.length // 2}[anchor]
    if strand == "-":
        b = f.end - off
        return b - down, b + up
    b = f.start + off
    return b - up, b + down


def _fill(gfile, rows: list[Row], width: int, sample: bool, chunk: int):
    """The rows' bases on the + strand: ref and gt (rows, width), flags (rows, width + 1), whose first column is the
    base before the row (where an insertion before the row's first base would be flagged)."""
    n = len(rows)
    ref = np.zeros((n, width), dtype=np.uint8)
    gt = np.zeros((n, width), dtype=np.uint8) if sample else None
    fl = np.zeros((n, width + 1), dtype=np.uint8) if sample else None
    by_seq: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_seq.setdefault(r.sequence, []).append(i)
    for seq, idx in by_seq.items():
        idx.sort(key=lambda i: rows[i].lo)
        los = [rows[i].lo for i in idx]
        for v in gfile.stretches(seq):
            # (the rows that overlap it, or begin just after it: a row's flags begin with the base before it)
            k, k1 = bisect.bisect_right(los, v.start - width), bisect.bisect_left(los, v.end + 1)
            while k < k1:
                lo0 = los[k]
                j = bisect.bisect_left(los, lo0 + max(chunk, 1), k, k1)  # (rows read together: those near the first)
                a, b = max(v.start, lo0 - 1), min(v.end, rows[idx[j - 1]].hi)
                if a < b:
                    sa, sb = a - v.start, b - v.start
                    if not v.covers("ref", sa, sb):
                        raise ValueError(f"{v.region(sa, sb)} has no reference: give it with --reference")
                    R = v.read("ref", sa, sb)
                    G = F = None
                    if sample and v.has("gt"):
                        G = v.read("gt", sa, sb)
                        F = v.read("flags", sa, sb) if v.has("flags") else np.zeros(b - a, dtype=np.uint8)
                    for i in idx[k:j]:
                        r = rows[i]
                        x, y = max(r.lo, a), min(r.hi, b)
                        if x < y:
                            ref[i, x - r.lo : y - r.lo] = R[x - a : y - a]
                            if sample:
                                gt[i, x - r.lo : y - r.lo] = C.GT_UNKNOWN if G is None else G[x - a : y - a]
                        x = max(r.lo - 1, a)
                        if F is not None and x < y:
                            fl[i, x - r.lo + 1 : y - r.lo + 1] = F[x - a : y - a]
                k = j
    return ref, gt, fl


def _to_strands(rows: list[Row], ref, gt, fl):
    """Each - strand row reverse-complemented; the flags cut to the row (an insertion's flag moved to the base it
    follows on the - strand)."""
    minus = np.fromiter((r.strand == "-" for r in rows), dtype=bool, count=len(rows))
    flags = None if fl is None else fl[:, 1:].copy()
    if minus.any():
        m = ref[minus]
        ref[minus] = (COMPLEMENT_SET[m & 0x0F] | (m & C.SOFTMASK))[:, ::-1]
        if gt is not None:
            gt[minus] = GT_COMPLEMENT[gt[minus]][:, ::-1]
            f = fl[minus]
            flags[minus] = ((f[:, :-1] & _INS) | (f[:, 1:] & np.uint8(~_INS & 0xFF)))[:, ::-1]
    return ref, gt, flags


def run_lengths(ref: np.ndarray, at: int, of: str = "A", tolerance: int = 2) -> np.ndarray:
    """For each row of ``ref``, the length of the run of the set ``of`` that starts at column ``at``: its bases
    belong to the set, but for up to ``tolerance`` others (the run ends at the next), and it ends with one that
    belongs (soft masking ignored; the row's end ends it)."""
    mask = base_set(of)
    nib = np.arange(256) & 0x0F
    member = (nib != 0) & ((nib & ~mask & 0x0F) == 0)
    m = member[ref[:, at:]]
    if m.shape[1] == 0:
        return np.zeros(ref.shape[0], dtype=np.int64)
    within = np.cumsum(~m, axis=1) <= tolerance
    m &= within
    last = m.shape[1] - np.argmax(m[:, ::-1], axis=1)
    return np.where(m.any(axis=1), last, 0).astype(np.int64)


def stack(
    gfile,
    features: list[Feature],
    *,
    anchor: str = "start",
    up: int = 400,
    down: int = 200,
    stranded: bool = True,
    sort: str = "file",
    run_of: str = "A",
    tolerance: int = 2,
    reverse: bool = False,
    max_rows: int | None = None,
    sample: bool = False,
    chunk: int = CHUNK,
) -> Stack:
    """The features of a gpxil file stacked (see the module's notes): ``sample`` reads the sample's planes too (if
    the file has a sample); ``stranded=False`` reads every feature on the + strand; ``reverse`` sorts the other
    way; ``max_rows`` keeps the first rows, once sorted."""
    if anchor not in ANCHORS:
        raise ValueError(f"the anchor must be one of {', '.join(ANCHORS)}")
    if sort not in SORTS:
        raise ValueError(f"sort by one of {', '.join(SORTS)}")
    if up < 0 or down < 0 or up + down < 1:
        raise ValueError("a row needs bases: give up and down, 0 or more, not both 0")
    if sort == "run" and down < 1:
        raise ValueError("--sort run measures the run from the anchor: give --down 1 or more")
    base_set(run_of)  # (says what's wrong with the letters)
    width = up + down
    names: dict[str, str | None] = {}
    rows = []
    for i, f in enumerate(features):
        if f.contig not in names:
            names[f.contig] = sequence_name(gfile, f.contig)
        seq = names[f.contig]
        if seq is None:
            continue
        strand = "-" if stranded and f.strand == "-" else "+"
        lo, hi = _interval(f, strand, anchor, up, down)
        rows.append(Row(f, i, seq, lo, hi, strand))
    unplaced = len(features) - len(rows)
    sample = sample and gfile.sample_id is not None
    ref, gt, fl = _fill(gfile, rows, width, sample, chunk)
    ref, gt, flags = _to_strands(rows, ref, gt, fl)

    order = list(range(len(rows)))
    descending = (sort in _LONGEST_FIRST) != reverse
    if sort == "file":
        keys = [r.index for r in rows]
    elif sort == "position":
        seq_order = {}
        for v in gfile.stretches():
            seq_order.setdefault(v.parent, len(seq_order))
        keys = [(seq_order.get(r.sequence, len(seq_order)), r.feature.start, r.feature.end) for r in rows]
    elif sort == "length":
        keys = [r.feature.length for r in rows]
    elif sort == "score":
        keys = [r.feature.score for r in rows]
    elif sort == "name":
        keys = [r.feature.name for r in rows]
    else:
        keys = run_lengths(ref, up, run_of, tolerance).tolist()
    for r, k in zip(rows, keys):
        r.key = k
    if sort == "score":  # (features without a score go last, as they come)
        have = [i for i in order if keys[i] is not None]
        order = sorted(have, key=lambda i: keys[i], reverse=descending) + [i for i in order if keys[i] is None]
    else:
        order = sorted(order, key=lambda i: keys[i], reverse=descending)  # (stable either way)
    dropped = 0
    if max_rows is not None and len(order) > max_rows:
        dropped = len(order) - max_rows
        order = order[:max_rows]
    pick = np.array(order, dtype=np.int64)
    return Stack(
        [rows[i] for i in order],
        ref[pick],
        None if gt is None else gt[pick],
        None if flags is None else flags[pick],
        up,
        down,
        anchor,
        sort,
        set_name(base_set(run_of)),
        len(features),
        unplaced,
        dropped,
        reverse,
    )


# ------------------------------------------------------------------ pictures


def rendering(st: Stack, *, mode: str = "reference", palette="classic", mark_insertions: bool = True):
    """The stack drawn as ``gpxil render`` draws bases (``mode``: reference, genotype, haplotypes or variants; the
    sample's modes need a stack read with ``sample``): a :class:`~gpxiltools.render.Rendering`, one pixel a base,
    a row a feature (two, one a haplotype, in the haplotypes mode)."""
    from .render import render_arrays

    if not st.rows:
        raise ValueError("no rows to draw")
    if mode != "reference" and st.gt is None:
        raise ValueError(f"the {mode} mode draws the sample's genome: stack a file with a sample, read with its planes")
    return render_arrays(
        st.ref.reshape(-1),
        None if st.gt is None else st.gt.reshape(-1),
        None if st.flags is None else st.flags.reshape(-1),
        mode=mode,
        palette=palette,
        bases_per_row=st.width,
        mark_insertions=mark_insertions,
        title=f"{len(st.rows):,} features, lined up at their {st.anchor}s",
    )


def binned(r, k: int, colorspace: str = "rgb") -> np.ndarray:
    """The mean colour of each ``k`` features' rows (the last bin may hold fewer): (bins x rows a feature, width,
    3 or 4) uint8, RGB or CMYK; a haplotypes rendering keeps a row a haplotype."""
    if k < 1:
        raise ValueError("bin by 1 or more")
    lut = (r.rgb if colorspace == "rgb" else r.cmyk).astype(np.float64)
    rpl = r.rows_per_line
    h, w = r.index.shape
    feats = h // rpl
    idx = r.index.reshape(feats, rpl, w)
    out = []
    for a in range(0, feats, k):
        out.append(np.round(lut[idx[a : a + k]].mean(axis=0)).astype(np.uint8))
    return np.concatenate(out, axis=0)


def _step(width: int) -> int:
    for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000):
        if width / s <= 12:
            return s
    return 100000


def labelled(pixels: np.ndarray, st: Stack, *, scale: int = 1, title: str = "", legend=(), rows_note: str = ""):
    """An RGB picture of the stack's pixels (h, w, 3) with a title, the anchor marked, the bases from it along the
    bottom, and a legend: a :class:`PIL.Image.Image`."""
    from PIL import Image, ImageDraw

    Image.MAX_IMAGE_PIXELS = None
    body = Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8), "RGB")
    if scale > 1:
        body = body.resize((body.width * scale, body.height * scale), Image.NEAREST)
    left, top, right, bottom = 24, 46, 210, 40
    width = max(left + body.width + right, 7 * len(title) + 2 * left, 7 * len(rows_note) + 2 * left)
    img = Image.new("RGB", (width, top + body.height + bottom), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((left, 4), title, fill=(0, 0, 0))
    if rows_note:
        d.text((left, 18), rows_note, fill=(90, 90, 90))
    img.paste(body, (left, top))
    ax = left + st.up * scale
    d.line([(ax, top - 8), (ax, top - 2)], fill=(200, 0, 0), width=2)
    d.line([(ax, top + body.height + 1), (ax, top + body.height + 7)], fill=(200, 0, 0), width=2)
    label = f"{st.anchor}"
    d.text((max(left, ax - 3 * len(label)), top - 20), label, fill=(200, 0, 0))
    step = _step(st.width)
    y = top + body.height + 2
    for off in range(-(st.up // step) * step, st.down + 1, step):
        x = ax + off * scale
        if left <= x <= left + body.width:
            d.line([(x, y), (x, y + 4)], fill=(0, 0, 0))
            t = f"{off:+,}" if off else "0"
            d.text((x - 3 * len(t), y + 7), t, fill=(0, 0, 0))
    d.text((left, y + 22), "bases from the anchor, 5' to 3' on each feature's strand", fill=(90, 90, 90))
    lx, ly = left + body.width + 16, top
    for name, colour in legend:
        d.rectangle([lx, ly, lx + 11, ly + 11], fill=tuple(int(v) for v in colour), outline=(160, 160, 160))
        d.text((lx + 18, ly), name, fill=(0, 0, 0))
        ly += 16
    return img


def save_pixels(pixels: np.ndarray, path, *, scale: int = 1, dpi: int | None = None) -> tuple[int, int]:
    """Pixels (h, w, 3 RGB or 4 CMYK) to a .png (RGB only) or .tif, each ``scale`` x ``scale``: the size written."""
    if scale > 1:
        pixels = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)
    h, w = pixels.shape[:2]
    if str(path).lower().endswith((".tif", ".tiff")):
        import tifffile

        kw = {"compression": "zlib", "compressionargs": {"level": 9}, "metadata": None}
        if dpi:
            kw.update(resolution=(dpi, dpi), resolutionunit="INCH")
        tifffile.imwrite(path, pixels, photometric="rgb" if pixels.shape[2] == 3 else "separated", **kw)
    else:
        from PIL import Image

        if pixels.shape[2] != 3:
            raise ValueError("CMYK needs a .tif output (PNG has no CMYK)")
        Image.MAX_IMAGE_PIXELS = None
        Image.fromarray(np.ascontiguousarray(pixels), "RGB").save(path, dpi=(dpi, dpi) if dpi else None)
    return w, h


def legend_of(r, used: np.ndarray | None = None) -> list[tuple[str, tuple]]:
    """A rendering's legend, as ``(label, RGB)``: only the colours its picture uses."""
    present = set(np.unique(r.index).tolist()) if used is None else set(used.tolist())
    return [(label, tuple(r.rgb[i])) for label, i in r.legend if i in present]


# ------------------------------------------------------------------ numbers


def profile(st: Stack) -> list[dict]:
    """Each column's counts: of rows with a base, of each base (A, C, G, T; ``other``: IUPAC codes for several bases,
    N), of soft-masked bases, the commonest of A, C, G and T and its share of them; and, for a sample, of each kind
    of variant (as the variants mode draws them: :data:`gpxiltools.render.VARIANT_CLASSES`). ``offset``: the bases
    from the anchor (0 the first after it)."""
    from .render import VARIANT_CLASSES, variant_classes

    nib = st.ref & 0x0F
    acgt = {b: (nib == code).sum(axis=0) for b, code in zip("ACGT", (C.BASE_A, C.BASE_C, C.BASE_G, C.BASE_T))}
    have = (nib != 0).sum(axis=0)
    soft = ((st.ref & C.SOFTMASK) != 0).sum(axis=0)
    counts = np.stack([acgt[b] for b in "ACGT"])
    top = counts.argmax(axis=0)
    total = counts.sum(axis=0)
    cls = None if st.gt is None else variant_classes(st.gt, st.flags)
    out = []
    for j in range(st.width):
        d = {"offset": j - st.up, "rows": int(have[j])}
        d.update({b: int(acgt[b][j]) for b in "ACGT"})
        d["other"] = int(have[j] - total[j])
        d["soft_masked"] = int(soft[j])
        d["commonest"] = "ACGT"[top[j]] if total[j] else ""
        d["commonest_share"] = round(float(counts[top[j], j] / total[j]), 4) if total[j] else None
        if cls is not None:
            col = cls[:, j]
            for k, name in enumerate(VARIANT_CLASSES[1:], 1):
                d[name.replace(" ", "_")] = int((col == k).sum())
        out.append(d)
    return out


def write_profile(out, st: Stack) -> None:
    """:func:`profile` as a table, tab-separated, with a heading."""
    rows = profile(st)
    keys = list(rows[0]) if rows else ["offset"]
    out.write("\t".join(keys) + "\n")
    for d in rows:
        out.write("\t".join("" if d[k] is None else str(d[k]) for k in keys) + "\n")


def write_rows(out, st: Stack) -> None:
    """The features in the order drawn, as BED (6 columns: the feature's own interval, name, score and strand), then
    the row (1 the top) and the key the rows were sorted by."""
    key = {"run": f"run_of_{st.run_of}"}.get(st.sort, st.sort)
    out.write(f"#sequence\tstart\tend\tname\tscore\tstrand\trow\t{key}\n")
    for n, r in enumerate(st.rows, 1):
        f = r.feature
        score = "." if f.score is None else f"{f.score:g}"
        out.write(f"{r.sequence}\t{f.start}\t{f.end}\t{f.name or '.'}\t{score}\t{f.strand}\t{n}\t{'' if r.key is None else r.key}\n")


def consensus(st: Stack) -> str:
    """The commonest base of each column (A, C, G or T; ``-`` where no row has one of them)."""
    counts = np.stack([((st.ref & 0x0F) == code).sum(axis=0) for code in (C.BASE_A, C.BASE_C, C.BASE_G, C.BASE_T)])
    letters = np.array(list("ACGT"))[counts.argmax(axis=0)]
    letters[counts.sum(axis=0) == 0] = "-"
    return "".join(letters.tolist())


__all__ = [
    "ANCHORS",
    "SORTS",
    "Row",
    "Stack",
    "stack",
    "run_lengths",
    "rendering",
    "binned",
    "labelled",
    "save_pixels",
    "profile",
    "consensus",
]
