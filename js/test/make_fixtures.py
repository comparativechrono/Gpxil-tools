"""Files for the JavaScript reader's tests (js/test/read.test.mjs), written by gpxiltools, and what gpxiltools reads
back from each: its metadata, every plane (as little-endian bytes in a .bin file) and every sidecar.

    python3 js/test/make_fixtures.py OUTDIR      -> OUTDIR/expected.json and the files

The files differ in what a reader must cope with: Deflate or none; many strips of a few rows, in classic TIFF,
BigTIFF or big-endian TIFF; 8-, 16- and 32-bit planes, signed and float; a sidecar with insertions and a COMPLEX
allele; a contig without a sample; the specification's worked example. Others must be refused, each with the code
the reader gives: another major version, a page compressed with LZW, tiles, a TIFF that isn't gpxil, not a TIFF.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))  # (the working tree's gpxiltools)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from simulate import fasta_text, simulate  # noqa: E402

from gpxiltools.container import GpxilFile, GpxilWriter  # noqa: E402
from gpxiltools.encode import encode  # noqa: E402


def base_file(out_dir: str) -> str:
    """A genome encoded from a VCF: chr1 simulated (SNVs, MNPs, indels, multi-allelic sites, no-calls, phased),
    chr2 with a substitution and a structural allele (COMPLEX), chrM without calls (reference only)."""
    sim = simulate(n=60_000, n_sites=500, seed=3, phased=True)
    chr2 = "ACGTTGCA" * 2_500
    rng = random.Random(4)
    chr_m = "".join(rng.choice("ACGT") for _ in range(997))
    fa = os.path.join(out_dir, "src.fa")
    with open(fa, "w") as fh:
        fh.write(fasta_text({"chr1": sim.reference, "chr2": chr2, "chrM": chr_m}))
    lines = sim.vcf_lines("chr1")
    lines += [
        "chr2\t101\t.\tA\tG\t50\tPASS\t.\tGT:GQ\t0/1:40",
        "chr2\t501\t.\tC\t<DUP>\t50\tPASS\tEND=900\tGT:GQ\t0|1:30",
    ]
    vcf = os.path.join(out_dir, "src.vcf")
    with open(vcf, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    out = os.path.join(out_dir, "base.gpxil.tif")
    encode(fa, out, vcf=vcf, width=97, log=lambda m: None)
    return out


def planes_of(path: str) -> list[tuple[str, dict[str, np.ndarray], list, dict]]:
    """Every contig of a file, with extra planes for the readers' sake: dp, ps (32-bit, large values), and two
    extension planes, x-score (int16, negative too) and x-frac (float32)."""
    out = []
    with GpxilFile(path) as g:
        for c in g:
            planes = {p: c.read(p) for p in c.planes}
            if "gt" in planes:
                n = c.length
                i = np.arange(n)
                planes["dp"] = ((i * 7) % 255).astype(np.uint8)
                planes["ps"] = np.where(planes["flags"] & 1, 4_294_967_000 + (i // 1000), 0).astype(np.uint32)
                planes["x-score"] = ((i % 2001) - 1000).astype(np.int16)
                planes["x-frac"] = (i / 7.0).astype(np.float32)
            entry = {k: v for k, v in c.entry.items() if k not in ("name", "length", "width", "pages", "sidecar_entries")}
            out.append((c.name, planes, c.sidecar(), entry))
        meta = g.metadata
    return out, meta


def rewrite(src: str, out: str, **kw) -> None:
    contigs, meta = planes_of(src)
    with GpxilWriter(out, reference=meta.get("reference"), sample=meta.get("sample"), **kw) as w:
        w.plane_info = {"x-score": {"description": "a test plane, signed"}, "x-frac": {"description": "a test plane, float"}}
        for name, planes, side, entry in contigs:
            w.add_contig(name, planes, sidecar=side or None, **entry)


def big_endian(src: str, out: str) -> None:
    """The same file, page for page, written big-endian by tifffile (the writer of gpxiltools writes little-endian)."""
    with tifffile.TiffFile(src) as t:
        pages = [(p.asarray(), p.description, p.rowsperstrip, p.tags[65400].value if 65400 in p.tags else None) for p in t.pages]
    with tifffile.TiffWriter(out, byteorder=">") as tw:
        for img, desc, rps, side in pages:
            extra = [(65400, 7, len(side), side, False)] if side is not None else []
            tw.write(img, compression="zlib", rowsperstrip=rps, description=desc, metadata=None, photometric="minisblack", extratags=extra)


def expected_of(path: str, name: str, out_dir: str) -> dict:
    with GpxilFile(path) as g, tifffile.TiffFile(path) as t:
        offsets = [p.offset for p in t.pages]
        contigs = []
        for c in g:
            planes = {}
            for p in c.planes:
                a = c.read(p)
                le = a.astype(a.dtype.newbyteorder("<"), copy=False)
                fn = f"{name}.{c.name}.{p}.bin"
                with open(os.path.join(out_dir, fn), "wb") as fh:
                    fh.write(le.tobytes())
                planes[p] = {"dtype": a.dtype.name, "bin": fn, "page": c.pages[p], "ifd": offsets[c.pages[p]]}
            side = [{"offset": e.offset, "hap": e.hap, "kind": e.kind, "data": e.data} for e in c.sidecar()]
            contigs.append(
                {
                    "name": c.name,
                    "length": c.length,
                    "width": c.width,
                    "aliases": list(c.entry.get("aliases", [])),
                    "planes": planes,
                    "sidecar": side,
                }
            )
        return {"file": os.path.basename(path), "version": str(g.metadata["gpxil"]), "meta": g.metadata, "contigs": contigs}


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, "work")
    os.makedirs(work, exist_ok=True)
    base = base_file(work)
    good = {}
    files = {
        "strips": dict(compression="deflate", width=97, strip_bytes=1500),
        "plain": dict(compression="none", width=97, strip_bytes=1500),
        "bigtiff": dict(compression="deflate", width=128, strip_bytes=4096, bigtiff=True),
        "wide": dict(compression="deflate", width=8192),
    }
    for name, kw in files.items():
        path = os.path.join(out_dir, f"{name}.gpxil.tif")
        rewrite(base, path, **kw)
        good[name] = path
    good["bigendian"] = os.path.join(out_dir, "bigendian.gpxil.tif")
    big_endian(good["strips"], good["bigendian"])
    good["example"] = os.path.join(out_dir, "example.gpxil.tif")
    shutil.copy(os.path.join(ROOT, "tests", "data", "worked-example.gpxil.tif"), good["example"])
    newer = os.path.join(out_dir, "newer.gpxil.tif")  # (a later 0.x: read as it is)
    ref = np.frombuffer(b"\x01\x02\x04\x08" * 50, dtype=np.uint8)
    with GpxilWriter(newer, width=16, version="0.3") as w:
        w.add_contig("c1", {"ref": ref.copy()})
    good["newer"] = newer

    bad = {}
    path = os.path.join(out_dir, "major1.gpxil.tif")
    with GpxilWriter(path, width=16, version="1.0") as w:
        w.add_contig("c1", {"ref": ref.copy()})
    bad["major1"] = {"file": os.path.basename(path), "code": "version", "at": "open"}
    path = os.path.join(out_dir, "lzw.gpxil.tif")  # (gpxil's Deflate page, its Compression tag said to be LZW)
    shutil.copy(good["plain"], path)
    with tifffile.TiffFile(path) as t:
        tag = t.pages[1].tags["Compression"]
        at, little = tag.valueoffset, t.byteorder == "<"
    with open(path, "r+b") as fh:
        fh.seek(at)
        fh.write((5).to_bytes(2, "little" if little else "big"))
    bad["lzw"] = {
        "file": os.path.basename(path),
        "code": "compression",
        "at": "read",
        "contig": "chr1",
        "plane": "gt",
        "compression": "LZW",
    }
    path = os.path.join(out_dir, "tiled.gpxil.tif")
    desc = json.dumps(
        {
            "gpxil": "0.2",
            "layout": {"type": "raster", "width": 32},
            "contigs": [{"name": "c1", "length": 1024, "pages": {"ref": 0}}],
            "contig": "c1",
            "plane": "ref",
            "length": 1024,
        }
    )
    tifffile.imwrite(path, np.ones((32, 32), np.uint8), tile=(16, 16), description=desc, metadata=None)
    bad["tiled"] = {"file": os.path.basename(path), "code": "tiles", "at": "open"}
    path = os.path.join(out_dir, "picture.tif")
    tifffile.imwrite(path, np.zeros((8, 8), np.uint8))
    bad["picture"] = {"file": os.path.basename(path), "code": "not-gpxil", "at": "open"}
    path = os.path.join(out_dir, "noise.bin")
    with open(path, "wb") as fh:
        fh.write(zlib.compress(b"not a tiff at all" * 20))
    bad["noise"] = {"file": os.path.basename(path), "code": "not-tiff", "at": "open"}

    expected = {"good": {name: expected_of(path, name, out_dir) for name, path in good.items()}, "bad": bad}
    with open(os.path.join(out_dir, "expected.json"), "w") as fh:
        json.dump(expected, fh, indent=1)
    shutil.rmtree(work)


if __name__ == "__main__":
    main(sys.argv[1])
