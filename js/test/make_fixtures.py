"""Files for the JavaScript reader's tests (js/test/read.test.mjs), written by gpxiltools, and what gpxiltools reads
back from each: its metadata, every plane (as little-endian bytes in a .bin file) and every sidecar.

    python3 js/test/make_fixtures.py OUTDIR      -> OUTDIR/expected.json and the files

The files differ in what a reader must cope with: Deflate or none; many strips of a few rows, or tiles, in classic
TIFF, BigTIFF or big-endian TIFF; 8-, 16- and 32-bit planes, signed and float, with the predictors extension planes
may have; a sidecar with insertions and a COMPLEX allele; a contig without a sample; the specification's worked
example; names that differ only in case; windows (section 3.1), with and without their reference plane, and where
gpxiltools finds regions in them. Others must be refused, each with the code the reader gives: another major version,
a page compressed with LZW, a TIFF that isn't gpxil, not a TIFF, windows that say wrong things of where they are, and
files made to hurt a reader (section 13): directories that loop, a strip that inflates to 200 MB, strips shorter than
their rows, a strip of no bytes, a tag or a strip larger than a reader takes.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import struct
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
from gpxiltools.subset import subset  # noqa: E402


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


def retiff(src: str, out: str, byteorder: str = "<", tile: tuple[int, int] | None = None, predictors: bool = False) -> None:
    """The same file, page for page, written by tifffile (the writer of gpxiltools writes little-endian strips, with no
    predictor): big-endian, or in tiles (length, width), and with a predictor on each extension plane (2, horizontal
    differencing, for integers; 3, floating point, for floats), which section 8.3 allows."""
    with tifffile.TiffFile(src) as t:
        pages = [(p.asarray(), p.description, p.rowsperstrip, p.tags[65400].value if 65400 in p.tags else None) for p in t.pages]
    with tifffile.TiffWriter(out, byteorder=byteorder) as tw:
        for img, desc, rps, side in pages:
            extra = [(65400, 7, len(side), side, False)] if side is not None else []
            plane = json.loads(desc)["plane"]
            predictor = (3 if img.dtype.kind == "f" else 2) if predictors and plane.startswith("x-") else None
            layout = dict(tile=tile) if tile else dict(rowsperstrip=rps)
            tw.write(
                img,
                compression="zlib",
                predictor=predictor,
                description=desc,
                metadata=None,
                photometric="minisblack",
                extratags=extra,
                **layout,
            )


def hand_tiff(pages: list[dict], loop_to: int | None = None) -> bytes:
    """A little-endian classic TIFF built by hand, so that it can be wrong: pages of dict(W, H, rps, strips (bytes
    each), desc, comp=1, bits=8, fmt=None (SampleFormat), pred=None (Predictor), counts=None (strip byte counts other
    than the strips' sizes), patch=None ({tag: (type, count, value or offset)} for entries to write as given)); the
    last directory pointing back to page loop_to."""
    buf = bytearray(b"II*\x00\x00\x00\x00\x00")

    def array(typ: int, vals: list[int]) -> tuple[int, bytes]:
        raw = b"".join(struct.pack("<H" if typ == 3 else "<I", v) for v in vals)
        if len(raw) <= 4:
            return len(vals), raw.ljust(4, b"\x00")
        at = len(buf)
        buf.extend(raw)
        return len(vals), struct.pack("<I", at)

    ifds, link = [], 4
    for pg in pages:
        offs = []
        for st in pg["strips"]:
            offs.append(len(buf))
            buf.extend(st + (b"\x00" if len(st) % 2 else b""))
        desc = pg["desc"].encode() + b"\x00"
        desc_at = len(buf)
        buf.extend(desc + (b"\x00" if len(desc) % 2 else b""))
        entries = {
            256: (4, *array(4, [pg["W"]])),
            257: (4, *array(4, [pg["H"]])),
            258: (3, *array(3, [pg.get("bits", 8)])),
            259: (3, *array(3, [pg.get("comp", 1)])),
            262: (3, *array(3, [1])),
            270: (2, len(desc), struct.pack("<I", desc_at)),
            273: (4, *array(4, offs)),
            277: (3, *array(3, [1])),
            278: (4, *array(4, [pg["rps"]])),
            279: (4, *array(4, pg.get("counts") or [len(st) for st in pg["strips"]])),
        }
        if pg.get("fmt"):
            entries[339] = (3, *array(3, [pg["fmt"]]))
        if pg.get("pred"):
            entries[317] = (3, *array(3, [pg["pred"]]))
        for tag, (typ, n, value) in (pg.get("patch") or {}).items():
            entries[tag] = (typ, n, struct.pack("<I", value))
        ifds.append(len(buf))
        struct.pack_into("<I", buf, link, len(buf))
        buf.extend(struct.pack("<H", len(entries)))
        for tag in sorted(entries):
            typ, n, value = entries[tag]
            buf.extend(struct.pack("<HHI", tag, typ, n) + value)
        link = len(buf)
        buf.extend(b"\x00\x00\x00\x00")
    if loop_to is not None:
        struct.pack_into("<I", buf, link, ifds[loop_to])
    return bytes(buf)


def hand_meta(contigs: list[dict], width: int) -> str:
    c = contigs[0]
    meta = {
        "gpxil": "0.2",
        "layout": {"type": "raster", "width": width},
        "contigs": contigs,
        "contig": c["name"],
        "plane": "ref",
        "length": c["length"],
    }
    return json.dumps(meta)


def hand_files(out_dir: str) -> dict:
    """Files made to hurt a reader, each with the code it must give (at "open", or "read" of a contig's plane)."""
    acgt = bytes([1, 2, 4, 8]) * 5
    one = [{"name": "c", "length": 20, "pages": {"ref": 0}}]
    bad = {}

    def put(name: str, data: bytes, **want) -> None:
        with open(os.path.join(out_dir, name + ".tif"), "wb") as fh:
            fh.write(data)
        bad[name] = {"file": name + ".tif", **want}

    # a chain of directories that loops back, with the gt plane on page 1; on page 1e400 (Infinity); on page 3,000,000
    looped = [{"name": "c", "length": 20, "pages": {"ref": 0, "gt": 1}}]
    put(
        "loop",
        hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [acgt], "desc": hand_meta(looped, 10)}], loop_to=0),
        code="format",
        at="read",
        contig="c",
        plane="gt",
    )
    desc = hand_meta(looped, 10).replace('"gt": 1', '"gt": 1e400')
    put(
        "loop-infinity",
        hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [acgt], "desc": desc}], loop_to=0),
        code="format",
        at="read",
        contig="c",
        plane="gt",
    )
    desc = hand_meta(looped, 10).replace('"gt": 1', '"gt": 3000000')
    put(
        "loop-far",
        hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [acgt], "desc": desc}], loop_to=0),
        code="format",
        at="read",
        contig="c",
        plane="gt",
    )
    # a page of one strip of 256 KiB, stored in 200 KiB (as such a strip could be), that inflates to 200 MB
    wide = [{"name": "c", "length": 1 << 18, "pages": {"ref": 0}}]
    bomb = zlib.compress(b"\x00" * (200 << 20), 9)
    put(
        "bomb",
        hand_tiff([{"W": 1 << 18, "H": 1, "rps": 1, "strips": [bomb], "comp": 8, "desc": hand_meta(wide, 1 << 18)}]),
        code="format",
        at="read",
        contig="c",
        plane="ref",
    )
    # a strip said to be stored in 4 GiB: refused before anything is fetched
    put(
        "huge-count",
        hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [acgt], "counts": [0xFFFFFFF0], "desc": hand_meta(one, 10)}]),
        code="format",
        at="read",
        contig="c",
        plane="ref",
    )
    # 100,000 empty directories one after another, the gt plane said to be on the last: walked from the bytes read
    far = [{"name": "c", "length": 20, "pages": {"ref": 0, "gt": 100_000}}]
    data = bytearray(hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [acgt], "desc": hand_meta(far, 10)}]))
    ifd = struct.unpack_from("<I", data, 4)[0]
    start = len(data)
    struct.pack_into("<I", data, ifd + 2 + 12 * struct.unpack_from("<H", data, ifd)[0], start)
    for i in range(100_000):
        data += struct.pack("<HI", 0, start + 6 * (i + 1) if i < 99_999 else 0)
    put("chain", bytes(data), code="format", at="read", contig="c", plane="gt")
    # page numbers that aren't (null, ""), and contig entries that aren't
    nulls = [{"name": "c", "length": 20, "pages": {"ref": 0, "gt": None, "flags": ""}}]
    put(
        "page-null",
        hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [acgt], "desc": hand_meta(nulls, 10)}]),
        code="format",
        at="read",
        contig="c",
        plane="gt",
    )
    bad["page-empty"] = dict(bad["page-null"], plane="flags")
    for name, contigs in (
        ("contig-null", [None]),
        ("aliases-text", [{"name": "c", "length": 20, "pages": {"ref": 0}, "aliases": "chrC"}]),
        ("pages-text", [{"name": "c", "length": 20, "pages": "ref"}]),
    ):
        desc = json.dumps(
            {"gpxil": "0.2", "layout": {"type": "raster", "width": 10}, "contigs": contigs, "contig": "c", "plane": "ref", "length": 20}
        )
        put(name, hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [acgt], "desc": desc}]), code="format", at="open")
    # windows that say wrong things of where they are (section 3.1)
    for name, window in (
        ("window-named-as-parent", {"of": "c", "start": 0}),
        ("window-negative", {"of": "x", "start": -3}),
        ("window-past-end", {"of": "x", "start": 90, "length": 100}),
        ("window-text", {"of": "x", "start": "10"}),
        ("window-list", [1, 2]),
        ("window-no-parent", {"start": 3}),
        ("window-length-null", {"of": "x", "start": 3, "length": None}),
        ("window-md5-null", {"of": "x", "start": 3, "md5": None}),
        ("window-refget-number", {"of": "x", "start": 3, "refget": 7}),
    ):
        contigs = [{"name": "c", "length": 20, "pages": {"ref": 0}, "window": window}]
        desc = json.dumps(
            {"gpxil": "0.3", "layout": {"type": "raster", "width": 10}, "contigs": contigs, "contig": "c", "plane": "ref", "length": 20}
        )
        put(name, hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [acgt], "desc": desc}]), code="format", at="open")
    # TIFFs that aren't gpxil, with pages this reader couldn't read either: told by their metadata, before their layout
    for name, kw in (("rgb", dict(data=np.zeros((8, 8, 3), np.uint8), photometric="rgb")), ("bilevel", dict(data=np.zeros((8, 8), bool)))):
        tifffile.imwrite(os.path.join(out_dir, name + ".tif"), metadata=None, **kw)
        bad[name] = {"file": name + ".tif", "code": "not-gpxil", "at": "open"}
    # strips shorter than their rows: 20 pixels in one strip of 10 bytes, stored and compressed
    put(
        "short",
        hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [acgt[:10]], "desc": hand_meta(one, 10)}]),
        code="format",
        at="read",
        contig="c",
        plane="ref",
    )
    put(
        "short-deflate",
        hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [zlib.compress(acgt[:10])], "comp": 8, "desc": hand_meta(one, 10)}]),
        code="format",
        at="read",
        contig="c",
        plane="ref",
    )
    # a strip of no bytes (row 2 of 2): no range of the file is asked for that isn't one (see the test)
    put(
        "zero",
        hand_tiff([{"W": 10, "H": 2, "rps": 1, "strips": [acgt[:10], b""], "desc": hand_meta(one, 10)}]),
        code="format",
        at="read",
        contig="c",
        plane="ref",
    )
    # an ImageDescription said to be 128 MiB; a gt page in strips of 300 MiB (one row 300 Mi bases wide)
    put(
        "huge-tag",
        hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [acgt], "desc": hand_meta(one, 10), "patch": {270: (2, 128 << 20, 8)}}]),
        code="limit",
        at="open",
    )
    two = [{"name": "c", "length": 20, "pages": {"ref": 0, "gt": 1}}]
    gt_desc = json.dumps({"gpxil": "0.2", "contig": "c", "plane": "gt", "length": 20})
    put(
        "huge-strip",
        hand_tiff(
            [
                {"W": 10, "H": 2, "rps": 2, "strips": [acgt], "desc": hand_meta(two, 10)},
                {"W": 300 << 20, "H": 1, "rps": 1, "strips": [acgt], "desc": gt_desc},
            ]
        ),
        code="limit",
        at="read",
        contig="c",
        plane="gt",
    )
    return bad


def odd_files(out_dir: str) -> dict:
    """Files that are odd but must be read: an ImageWidth said to hold 2^30 values (only the first is read); two
    contigs whose names differ only in case, and an alias that is another contig's name; a float plane with
    predictor 2; a zlib stream with padding after it."""
    acgt = bytes([1, 2, 4, 8]) * 5
    files = {}
    one = [{"name": "c", "length": 20, "pages": {"ref": 0}}]
    data = bytearray(hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [acgt], "desc": hand_meta(one, 10)}]))
    ifd = struct.unpack_from("<I", data, 4)[0]
    for i in range(struct.unpack_from("<H", data, ifd)[0]):
        if struct.unpack_from("<H", data, ifd + 2 + 12 * i)[0] == 256:  # ImageWidth: 2^30 values, the first 10, at the end
            struct.pack_into("<HHII", data, ifd + 2 + 12 * i, 256, 4, 1 << 30, len(data))
    data += struct.pack("<I", 10)
    with open(os.path.join(out_dir, "many-values.tif"), "wb") as fh:
        fh.write(data)
    files["many-values"] = "many-values.tif"
    two = [
        {"name": "a", "length": 20, "pages": {"ref": 0}, "aliases": ["b"]},
        {"name": "A", "length": 20, "pages": {"ref": 1}},
        {"name": "b", "length": 20, "pages": {"ref": 2}},
    ]
    pages = [{"W": 10, "H": 2, "rps": 2, "strips": [bytes([code]) * 20], "desc": hand_meta(two, 10)} for code in (1, 2, 4)]
    for k, pg in enumerate(pages[1:], 1):
        pg["desc"] = json.dumps({"gpxil": "0.2", "contig": two[k]["name"], "plane": "ref", "length": 20})
    with open(os.path.join(out_dir, "case.tif"), "wb") as fh:
        fh.write(hand_tiff(pages))
    files["case"] = "case.tif"
    # a float32 extension plane with predictor 2 (its 32-bit words differenced, as libtiff writes it; tifffile reads it)
    x = (np.arange(64, dtype=np.float32) / 7).reshape(4, 16)
    words = x.view(np.uint32).astype(np.int64)
    words[:, 1:] = (words[:, 1:] - words[:, :-1]) % 2**32
    contigs = [{"name": "c", "length": 64, "pages": {"x-f": 0}}]
    desc = json.dumps(
        {"gpxil": "0.2", "layout": {"type": "raster", "width": 16}, "contigs": contigs, "contig": "c", "plane": "x-f", "length": 64}
    )
    strip = zlib.compress(words.astype("<u4").tobytes())
    with open(os.path.join(out_dir, "float-pred2.tif"), "wb") as fh:
        fh.write(hand_tiff([{"W": 16, "H": 4, "rps": 4, "strips": [strip], "comp": 8, "bits": 32, "fmt": 3, "pred": 2, "desc": desc}]))
    with open(os.path.join(out_dir, "float-pred2.bin"), "wb") as fh:
        fh.write(x.astype("<f4").tobytes())
    with tifffile.TiffFile(os.path.join(out_dir, "float-pred2.tif")) as t:
        assert np.array_equal(t.pages[0].asarray(), x)  # (as tifffile reads it)
    files["float-pred2"] = "float-pred2.tif"
    # a strip whose zlib stream is followed, inside its byte count, by padding (zlib ignores it)
    with open(os.path.join(out_dir, "padded.tif"), "wb") as fh:
        fh.write(
            hand_tiff([{"W": 10, "H": 2, "rps": 2, "strips": [zlib.compress(acgt) + bytes(7)], "comp": 8, "desc": hand_meta(one, 10)}])
        )
    files["padded"] = "padded.tif"
    return files


def locate_cases(path: str) -> list[dict]:
    """Regions, and where gpxiltools finds them in a file of windows: the contig and its offsets, or the kind of
    error ("contig": no such contig; "region": not one the file holds, or not which)."""
    asked = [
        ["chr1", 30000, 30010],
        ["chr1", 33500, 33600],  # (in both windows that overlap: the first)
        ["chr1", 33000, 40000],  # (just the second)
        ["chr1", 100, 200],
        ["chr1", 600, 700],
        ["chr1", None, None],  # (three windows: which?)
        ["chr2", None, None],
        ["2", 600, 610],  # (by another name of the sequence)
        ["chrM", None, None],
        ["chrM", 10, 20],
        ["chr1:30001-34000", None, None],
        ["chr1:30001-34000", 30005, 30010],
        ["chr1:30001-34000", 10, 20],
        ["chr9", None, None],
        ["chr1", 39990, 40010],  # (past the end of the window)
        ["CHR1", 33500, 33600],  # (names ignoring case, with or without "chr", as contig() takes them)
        ["1", 30000, 30010],
        ["Chr2", 600, 610],
        ["chr1", 33600, 33500],  # (a stretch of no bases, or less)
        ["chr1", 33500, 33500],
        ["chr1", 0, None],  # (from position 0 on: not "the whole contig")
        ["chr1", 30005, None],
        ["chrM", 0, None],
    ]
    out = []
    with GpxilFile(path) as g:
        for name, s, e in asked:
            try:
                c, a, b = g.locate(name, s, e)
                out.append({"ask": [name, s, e], "contig": c.name, "start": a, "end": b})
            except (KeyError, IndexError) as exc:
                msg = exc.args[0] if exc.args else ""
                out.append({"ask": [name, s, e], "error": "contig" if str(msg).startswith("no contig") else "region"})
    return out


def locate_text_cases(path: str) -> list[dict]:
    """Regions given as text, and where gpxiltools finds them (``GpxilFile.locate(text)``, as gpxil-read.js's
    ``locate(text)``): a contig's name is that contig, else the text is read as a region."""
    asked = [
        "chr1:33,501-33,600",
        "chr1:30001-34000",  # (a window's name)
        "CHR1:30001-34000",  # (not a name as written: a region, found in the window of just that stretch)
        "chr1:30001-34000:30,006-30,010",
        "chr2:601",
        "chr2: 601 - 610",
        "2:601-610",
        "chrM",
        "chrm",
        "chr1:10-5",
        "chr1:0-5",
        "chr9:1-10",
        "chr1:01-5",
        " chrM ",  # (a name with spaces around it: the name)
        " chr2:601-610 ",
    ]
    out = []
    with GpxilFile(path) as g:
        for text in asked:
            try:
                c, a, b = g.locate(text)
                out.append({"ask": text, "contig": c.name, "start": a, "end": b})
            except (KeyError, IndexError, ValueError) as exc:
                msg = exc.args[0] if exc.args else ""
                out.append({"ask": text, "error": "contig" if str(msg).startswith("no contig") else "region"})
    return out


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
                    "window": c.window,
                    "parent": c.parent,
                    "start": c.start,
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
    retiff(good["strips"], good["bigendian"], byteorder=">")
    good["tiled"] = os.path.join(out_dir, "tiled.gpxil.tif")  # (tiles 32 wide across rows of 97, 16 rows long)
    retiff(good["strips"], good["tiled"], tile=(16, 32), predictors=True)
    good["predictors"] = os.path.join(out_dir, "predictors.gpxil.tif")
    retiff(good["strips"], good["predictors"], byteorder=">", predictors=True)
    good["example"] = os.path.join(out_dir, "example.gpxil.tif")
    shutil.copy(os.path.join(ROOT, "tests", "data", "worked-example.gpxil.tif"), good["example"])
    newer = os.path.join(out_dir, "newer.gpxil.tif")  # (a later 0.x: read as it is)
    ref = np.frombuffer(b"\x01\x02\x04\x08" * 50, dtype=np.uint8)
    with GpxilWriter(newer, width=16, version="0.4") as w:
        w.add_contig("c1", {"ref": ref.copy()})
    good["newer"] = newer
    # windows (section 3.1), cut out of the whole contigs: two that overlap, one that starts at a COMPLEX allele
    # (chr2's, at 501), a whole contig beside them; and windows without their reference plane
    good["windows"] = os.path.join(out_dir, "windows.gpxil.tif")
    cuts = ["chr1:1-500", "chr1:30,001-34,000", "chr1:33,001-40,000", "chr2:501-2,000", "chrM"]
    subset(good["strips"], good["windows"], cuts, log=lambda m: None)
    good["windows-noref"] = os.path.join(out_dir, "windows-noref.gpxil.tif")
    subset(good["strips"], good["windows-noref"], ["chr1:30,001-34,000", "chr2:401-600"], include_ref=False, log=lambda m: None)

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
    path = os.path.join(out_dir, "picture.tif")
    tifffile.imwrite(path, np.zeros((8, 8), np.uint8))
    bad["picture"] = {"file": os.path.basename(path), "code": "not-gpxil", "at": "open"}
    path = os.path.join(out_dir, "noise.bin")
    with open(path, "wb") as fh:
        fh.write(zlib.compress(b"not a tiff at all" * 20))
    bad["noise"] = {"file": os.path.basename(path), "code": "not-tiff", "at": "open"}

    bad.update(hand_files(out_dir))
    expected = {
        "good": {name: expected_of(path, name, out_dir) for name, path in good.items()},
        "bad": bad,
        "odd": odd_files(out_dir),
        "locate": locate_cases(good["windows"]),
        "locate_text": locate_text_cases(good["windows"]),
    }
    with open(os.path.join(out_dir, "expected.json"), "w") as fh:
        json.dump(expected, fh, indent=1)
    shutil.rmtree(work)


if __name__ == "__main__":
    main(sys.argv[1])
