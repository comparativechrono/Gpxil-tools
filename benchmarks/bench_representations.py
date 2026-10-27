"""Compare ways of storing a genome as pixels.

Measures size, encode/decode time, random-access time and losslessness for
image containers (TIFF, PNG, WebP, JPEG XL), pixel encodings (RGB, 8-bit
codes, 4-bit packed, CMYK dosage) and layouts (raster, Hilbert), against
FASTA/2bit baselines.

Usage:
    python bench_representations.py --fasta chr21.fa.gz --vcf HG002_chr21.vcf.gz \
        [--bed confident.bed] [--out results.json]

Requires: numpy, tifffile, imagecodecs, Pillow, gpxiltools.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import lzma
import os
import platform
import sys
import tempfile
import time
import zlib

import imagecodecs
import numpy as np
import tifffile
from PIL import Image

import gpxiltools
from gpxiltools import codes as C
from gpxiltools.container import GpxilFile, GpxilWriter, to_image
from gpxiltools.fasta import iter_fasta
from gpxiltools.layout import hilbert_image
from gpxiltools.palettes import ref_colours
from gpxiltools.vcf import encode_calls, iter_body_lines, iter_calls, load_bed

Image.MAX_IMAGE_PIXELS = None


def timed(fn, *a, repeat=1, **k):
    best, out = None, None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn(*a, **k)
        dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    return out, best


def twobit_size(codes: np.ndarray) -> int:
    """Exact size of a single-sequence UCSC .2bit file (v0, 32-bit offsets)."""
    upper = np.bitwise_and(codes, 0x0F)
    n = codes.shape[0]

    def runs(mask):
        d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
        return int((d == 1).sum())

    n_blocks = runs(upper == C.BASE_N)
    m_blocks = runs((codes & C.SOFTMASK) != 0)
    header = 16 + 1 + len("chr") + 4  # file header + one index entry (name length byte, name, offset)
    record = 4 + 4 + 8 * n_blocks + 4 + 8 * m_blocks + 4 + (n + 3) // 4
    return header + record


def tiff_bytes(img: np.ndarray, **kw) -> bytes:
    buf = io.BytesIO()
    tifffile.imwrite(buf, img, metadata=None, **kw)
    return buf.getvalue()


def tiff_random_access(path: str, regions: list[tuple[int, int]]) -> float:
    with GpxilFile(path) as g:
        c = g[g.contigs[0]]
        t = time.perf_counter()
        for s, e in regions:
            c.read("ref", s, e)
        return time.perf_counter() - t


def fmt_row(name, size, n, enc=None, dec=None, ra=None, lossless=True, note=""):
    return {
        "name": name,
        "bytes": int(size),
        "bits_per_base": round(8 * size / n, 3),
        "encode_s": None if enc is None else round(enc, 2),
        "decode_s": None if dec is None else round(dec, 2),
        "random_access_ms": None if ra is None else round(ra, 3),
        "lossless": lossless,
        "note": note,
    }


def bench_reference(codes: np.ndarray, width: int, fasta_gz_size: int, workdir: str, log) -> list[dict]:
    n = codes.shape[0]
    rng = np.random.default_rng(42)
    regions = [(int(s), int(s) + 1000) for s in rng.integers(0, n - 1000, size=1000)]
    rows = []
    ascii_seq = C.REF_TO_ASCII[codes].tobytes()
    fasta_text = b">chr\n" + b"\n".join(ascii_seq[i : i + 60] for i in range(0, n, 60)) + b"\n"

    log("baselines")
    rows.append(fmt_row("FASTA, uncompressed (60 bp lines)", len(fasta_text), n))
    rows.append(fmt_row("FASTA.gz as distributed (Ensembl)", fasta_gz_size, n, note="gzip, includes soft-masking"))
    rows.append(fmt_row("UCSC .2bit (exact size)", twobit_size(codes), n, note="2 bits/base + N and mask blocks"))
    z, t = timed(lzma.compress, fasta_text, preset=9)
    rows.append(fmt_row("FASTA.xz -9", len(z), n, enc=t))
    zs, t = timed(imagecodecs.zstd_encode, fasta_text, level=19)
    rows.append(fmt_row("FASTA.zst -19", len(zs), n, enc=t))

    img = to_image(codes, width)
    rgb_lut, _ = ref_colours("classic")
    cmap = (rgb_lut.astype(np.uint16).T * 257).astype(np.uint16)

    log("gpxil TIFF variants")
    variants = [
        ("gpxil TIFF, deflate 6, 256 KiB strips", dict(compression="deflate", level=6)),
        ("gpxil TIFF, deflate 9, 256 KiB strips", dict(compression="deflate", level=9)),
        ("gpxil TIFF, deflate 9, 64 KiB strips", dict(compression="deflate", level=9, strip_bytes=1 << 16)),
        ("gpxil TIFF, deflate 9, 1 MiB strips", dict(compression="deflate", level=9, strip_bytes=1 << 20)),
        ("gpxil TIFF, zstd 3, 256 KiB strips", dict(compression="zstd", level=3)),
        ("gpxil TIFF, zstd 9, 256 KiB strips", dict(compression="zstd", level=9)),
        ("gpxil TIFF, zstd 19, 256 KiB strips", dict(compression="zstd", level=19)),
        ("gpxil TIFF, zstd 19, 1 MiB strips", dict(compression="zstd", level=19, strip_bytes=1 << 20)),
        ("gpxil TIFF, lzma, 256 KiB strips", dict(compression="lzma")),
        ("gpxil TIFF, lzw, 256 KiB strips", dict(compression="lzw")),
        ("gpxil TIFF, packbits", dict(compression="packbits")),
        ("gpxil TIFF, uncompressed", dict(compression="none")),
    ]
    for name, kw in variants:
        path = os.path.join(workdir, "v.gpxil.tif")

        def write(path=path, kw=kw):
            with GpxilWriter(path, width=width, **kw) as w:
                w.add_contig("chr", {"ref": codes})

        _, t_enc = timed(write)
        size = os.path.getsize(path)
        with GpxilFile(path) as g:
            back, t_dec = timed(g["chr"].read, "ref")
        ok = bool((back == codes).all())
        ra = tiff_random_access(path, regions) / len(regions) * 1000
        rows.append(fmt_row(name, size, n, t_enc, t_dec, ra, ok))
        log(f"  {name}: {size:,} B")

    # tiles instead of strips (2-D access pattern)
    def tiled():
        return tiff_bytes(img, tile=(512, 512), compression="zlib", compressionargs={"level": 9}, photometric="palette", colormap=cmap)

    b, t_enc = timed(tiled)
    path = os.path.join(workdir, "tiled.tif")
    open(path, "wb").write(b)

    def read_tiled_regions():
        with tifffile.TiffFile(path) as tf:
            z = tf.pages[0].aszarr()
            import zarr

            arr = zarr.open(z, mode="r")
            t = time.perf_counter()
            for s, e in regions[:200]:
                r0, r1 = s // width, (e - 1) // width + 1
                arr[r0:r1, :]
            return (time.perf_counter() - t) / 200 * 1000

    try:
        ra_tiled = read_tiled_regions()
    except Exception:
        ra_tiled = None
    back, t_dec = timed(tifffile.imread, path)
    rows.append(fmt_row("TIFF, deflate 9, 512x512 tiles", len(b), n, t_enc, t_dec, ra_tiled, bool((back.reshape(-1)[:n] == codes).all())))

    log("4-bit packing and a separate mask plane")
    nib = np.bitwise_and(codes, 0x0F)
    pad = nib if n % 2 == 0 else np.concatenate([nib, [0]]).astype(np.uint8)
    packed = (pad[0::2] << 4 | pad[1::2]).astype(np.uint8)
    mask = ((codes & C.SOFTMASK) != 0).astype(np.uint8)
    mask_packed = np.packbits(mask)
    strip = 1 << 18
    for codec, enc in (("deflate 9", lambda b: zlib.compress(b, 9)), ("zstd 19", lambda b: imagecodecs.zstd_encode(b, level=19))):
        size4 = sum(len(enc(packed[i : i + strip].tobytes())) for i in range(0, packed.size, strip))
        sizem = sum(len(enc(mask_packed[i : i + strip].tobytes())) for i in range(0, mask_packed.size, strip))
        rows.append(fmt_row(f"4-bit packed IUPAC + 1-bit mask, {codec}", size4 + sizem, n, note=f"payload only: {size4:,} + {sizem:,} B"))

    log("PNG / WebP / JPEG XL")
    pal_img = Image.fromarray(img, mode="P")
    pal_img.putpalette(rgb_lut.reshape(-1).tolist())

    def png8():
        buf = io.BytesIO()
        pal_img.save(buf, format="PNG", compress_level=9)
        return buf.getvalue()

    b, t_enc = timed(png8)
    back, t_dec = timed(lambda: np.array(Image.open(io.BytesIO(b))))
    rows.append(
        fmt_row(
            "PNG, 8-bit palette, level 9",
            len(b),
            n,
            t_enc,
            t_dec,
            t_dec * 1000,
            bool((back == img).all()),
            "any read decodes the whole image",
        )
    )

    nib_img = to_image(nib, width)
    nib_pil = Image.fromarray(nib_img, mode="P")
    nib_pil.putpalette(rgb_lut[:16].reshape(-1).tolist())

    def png4():
        buf = io.BytesIO()
        nib_pil.save(buf, format="PNG", compress_level=9, bits=4)
        return buf.getvalue()

    b, t_enc = timed(png4)
    back = np.array(Image.open(io.BytesIO(b)))
    rows.append(
        fmt_row("PNG, 4-bit palette (mask dropped)", len(b), n, t_enc, None, None, bool((back == nib_img).all()), "loses soft-masking")
    )

    rgb = rgb_lut[img]

    def png_rgb():
        buf = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", compress_level=9)
        return buf.getvalue()

    b, t_enc = timed(png_rgb)
    rows.append(
        fmt_row(
            "PNG, 24-bit RGB colours (notebook approach)", len(b), n, t_enc, note="needs a colour->code reverse lookup to recover bases"
        )
    )

    if img.shape[0] <= 16383 and img.shape[1] <= 16383:

        def webp():
            buf = io.BytesIO()
            pal_img.save(buf, format="WEBP", lossless=True, quality=100, method=6)
            return buf.getvalue()

        b, t_enc = timed(webp)
        back_rgb, t_dec = timed(lambda: np.array(Image.open(io.BytesIO(b)).convert("RGB")))
        lossless = bool((back_rgb == rgb).all())
        rows.append(fmt_row("WebP lossless (palette)", len(b), n, t_enc, t_dec, t_dec * 1000, lossless, "max 16383 px per side; RGB only"))

    for effort in (3, 7):
        try:
            b, t_enc = timed(imagecodecs.jpegxl_encode, img, lossless=True, effort=effort)
            back, t_dec = timed(imagecodecs.jpegxl_decode, b)
            rows.append(
                fmt_row(f"JPEG XL lossless, effort {effort} (grey codes)", len(b), n, t_enc, t_dec, t_dec * 1000, bool((back == img).all()))
            )
        except Exception as exc:  # pragma: no cover - depends on codec build
            rows.append(fmt_row(f"JPEG XL lossless, effort {effort}", 0, n, note=f"failed: {exc}"))

    log("Hilbert layout")
    himg, order = hilbert_image(codes)
    b, t_enc = timed(tiff_bytes, himg, tile=(256, 256), compression="zlib", compressionargs={"level": 9})
    rows.append(
        fmt_row(
            f"TIFF, Hilbert layout (order {order}), deflate 9, 256 tiles",
            len(b),
            n,
            t_enc,
            note="2-D locality; each tile is 65,536 consecutive bases",
        )
    )
    b, t_enc = timed(tiff_bytes, himg, tile=(256, 256), compression="zstd", compressionargs={"level": 19})
    rows.append(fmt_row(f"TIFF, Hilbert layout (order {order}), zstd 19, 256 tiles", len(b), n, t_enc))
    return rows


def bench_variants(codes: np.ndarray, vcf: str, bed: str | None, width: int, workdir: str, log) -> tuple[list[dict], dict]:
    n = codes.shape[0]
    rows: list[dict] = []
    info: dict = {}
    vcf_bytes = open(vcf, "rb").read()
    vcf_text = gzip.decompress(vcf_bytes) if vcf_bytes[:2] == b"\x1f\x8b" else vcf_bytes
    body = [ln for ln in vcf_text.split(b"\n") if ln and not ln.startswith(b"#")]
    gt_only = b"\n".join(b"\t".join(ln.split(b"\t")[:5] + [ln.split(b"\t")[9].split(b":")[0]]) for ln in body)
    rows.append(
        fmt_row(
            "VCF.gz (bgzip-level gzip of the input)", len(gzip.compress(vcf_text, 6)), n, note=f"{len(body):,} records, all INFO/FORMAT"
        )
    )
    rows.append(fmt_row("VCF.gz, CHROM-ALT + GT only", len(gzip.compress(gt_only, 9)), n, note="the content the gt plane holds"))

    log("encoding genotype planes")
    calls = (c for _, c in iter_calls(iter_body_lines(vcf), 0))
    (planes, side, stats, _), t_enc = timed(encode_calls, codes, calls, contig="chr", planes=("gt", "flags"))
    info["records"] = stats.records
    info["sidecar_entries"] = len(side)
    info["gt_nonzero"] = int(np.count_nonzero(planes["gt"]))
    for comp, lvl in (("deflate", 9), ("zstd", 19)):
        path = os.path.join(workdir, f"gt_{comp}.tif")
        with GpxilWriter(path, width=width, compression=comp, level=lvl, sample={"id": "sample"}) as w:
            w.add_contig("chr", {"gt": planes["gt"], "flags": planes["flags"]}, sidecar=side)
        rows.append(fmt_row(f"gpxil gt + flags planes + sidecar, {comp} {lvl}", os.path.getsize(path), n, t_enc, note="no ref plane"))
    if bed:
        mask = load_bed(bed, "chr21", n, ["21", "chr"])
        if mask is not None:
            calls = (c for _, c in iter_calls(iter_body_lines(vcf), 0))
            planes_b, side_b, _, _ = encode_calls(codes, calls, contig="chr", planes=("gt", "flags"), callable_mask=mask)
            path = os.path.join(workdir, "gt_bed.tif")
            with GpxilWriter(path, width=width, compression="deflate", level=9, sample={"id": "sample"}) as w:
                w.add_contig("chr", {"gt": planes_b["gt"], "flags": planes_b["flags"]}, sidecar=side_b)
            rows.append(
                fmt_row(
                    "gpxil gt + flags, deflate 9, outside-benchmark = UNKNOWN",
                    os.path.getsize(path),
                    n,
                    note=f"{int(mask.sum()):,} callable bases",
                )
            )
            info["callable_bases"] = int(mask.sum())

    log("CMYK dosage and RGB 'both channels' encodings")
    dose = C.dosage(codes, planes["gt"])  # (n, 4) copies of A, C, G, T
    rows_n = -(-n // width)
    full = np.zeros((rows_n * width, 4), dtype=np.uint8)
    full[:n] = np.minimum(dose * 127, 255)
    cmyk_img = full.reshape(rows_n, width, 4)
    b = tiff_bytes(cmyk_img, photometric="separated", compression="zlib", compressionargs={"level": 9}, rowsperstrip=32)
    rows.append(
        fmt_row(
            "CMYK TIFF, one ink per base = allele dosage (0/1/2 copies)",
            len(b),
            n,
            lossless=False,
            note="loses phase, indel content, no-call vs deletion",
        )
    )

    # what the dosage image cannot distinguish
    h1, h2 = C.hap1(planes["gt"]), C.hap2(planes["gt"])
    phased = (planes["flags"] & C.FLAG_PHASED) != 0
    info["dosage_losses"] = {
        "positions_with_insertions": int(((planes["flags"] & (C.FLAG_INS1 | C.FLAG_INS2)) != 0).sum()),
        "deleted_vs_unknown_ambiguous": int(((h1 == C.HAP_DEL) | (h2 == C.HAP_DEL) | (h1 == C.HAP_UNKNOWN) | (h2 == C.HAP_UNKNOWN)).sum()),
        "phased_positions": int(phased.sum()),
        # a dosage image cannot say which haplotype carries what wherever the two differ
        "phased_positions_haplotypes_differ": int((phased & (h1 != h2)).sum()),
    }
    # RGB with base colours added channel-wise: A=R, G=G, C=B, T=R+G (notebook colours)
    chan = {C.BASE_A: (1, 0, 0), C.BASE_C: (0, 0, 1), C.BASE_G: (0, 1, 0), C.BASE_T: (1, 1, 0)}
    lut = np.zeros((16, 3), dtype=np.uint8)
    for code in range(16):
        v = np.zeros(3, dtype=np.uint8)
        for bit, c in chan.items():
            if code & bit:
                v |= np.array(c, dtype=np.uint8)
        lut[code] = v
    iupac = C.genotype_iupac(codes, planes["gt"])
    collide = {}
    for code in range(1, 16):
        key = tuple(lut[code])
        collide.setdefault(key, []).append(C.IUPAC[code])
    info["rgb_both_channels_collisions"] = {str(k): v for k, v in collide.items() if len(v) > 1}
    info["rgb_positions_A/G_het_vs_T"] = int((np.isin(iupac, [C.BASE_A | C.BASE_G])).sum())
    return rows, info


def bench_matplotlib(codes: np.ndarray, width: int = 500, rows_n: int = 500, dpi: int = 150) -> dict:
    """How much of a sequence survives the notebook's plt.imshow + savefig route.

    Uses the notebooks' figure size rule, figsize=(10, 10 * rows / width).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq = codes[30_000_000 : 30_000_000 + width * rows_n] & 0x0F
    rgb_lut, _ = ref_colours("classic")
    img = rgb_lut[seq.reshape(rows_n, width)].astype(np.float64) / 255
    fig = plt.figure(figsize=(10, 10 * rows_n / width))
    plt.imshow(img, aspect="auto")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, dpi=dpi)
    plt.close(fig)
    out = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    # The image sits inside the axes area of the figure. Find its bounding box
    # (non-white pixels) and sample each base's cell at its centre: the most
    # favourable way to read bases back.
    nonwhite = (out < 250).any(axis=2)
    ys_, xs_ = np.flatnonzero(nonwhite.any(axis=1)), np.flatnonzero(nonwhite.any(axis=0))
    y0, y1, x0, x1 = ys_[0], ys_[-1] + 1, xs_[0], xs_[-1] + 1
    box = out[y0:y1, x0:x1]
    ys = ((np.arange(rows_n) + 0.5) * box.shape[0] / rows_n).astype(int)
    xs = ((np.arange(width) + 0.5) * box.shape[1] / width).astype(int)
    sampled = box[np.ix_(ys, xs)].astype(int)
    truth = (img * 255).round().astype(int)
    exact = (sampled == truth).all(axis=2)
    # nearest palette colour, as a reader would have to guess
    pal = rgb_lut[[1, 2, 4, 8]].astype(int)
    guess = np.argmin(((sampled[..., None, :] - pal[None, None]) ** 2).sum(-1), axis=-1)
    true_idx = np.searchsorted([1, 2, 4, 8], seq.reshape(rows_n, width).clip(1, 8))
    ok_base = seq.reshape(rows_n, width) <= 8
    nearest_ok = (guess == true_idx) & ok_base
    return {
        "bases": int(seq.size),
        "dpi": dpi,
        "grid": [rows_n, width],
        "figure_pixels": list(out.shape[:2]),
        "image_box_pixels": [int(y1 - y0), int(x1 - x0)],
        "pixels_per_base": [round(float(y1 - y0) / rows_n, 3), round(float(x1 - x0) / width, 3)],
        "bases_exact_colour_at_cell_centre": round(float(exact.mean()), 4),
        "bases_recovered_by_nearest_colour": round(float(nearest_ok.sum() / ok_base.sum()), 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--vcf")
    ap.add_argument("--bed")
    ap.add_argument("--width", type=int, default=8192)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args(argv)

    def log(m):
        print(m, file=sys.stderr, flush=True)

    rec = next(iter_fasta(args.fasta))
    codes = rec.codes
    n = codes.shape[0]
    log(f"{rec.name}: {n:,} bases")
    result = {
        "sequence": {
            "name": rec.name,
            "length": n,
            "N": int(((codes & 0x0F) == C.BASE_N).sum()),
            "softmasked": int(((codes & 0x10) != 0).sum()),
        },
        "width": args.width,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tifffile": tifffile.__version__,
            "imagecodecs": imagecodecs.__version__,
            "pillow": Image.__version__,
            "gpxiltools": gpxiltools.__version__,
            "cpu": platform.processor() or platform.machine(),
            "cores": os.cpu_count(),
        },
    }
    with tempfile.TemporaryDirectory() as wd:
        result["reference"] = bench_reference(codes, args.width, os.path.getsize(args.fasta), wd, log)
        if args.vcf:
            result["variants"], result["variant_info"] = bench_variants(codes, args.vcf, args.bed, args.width, wd, log)
    log("matplotlib route")
    result["matplotlib_route"] = [bench_matplotlib(codes, 500, 500, 150), bench_matplotlib(codes, 5000, 200, 600)]
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
