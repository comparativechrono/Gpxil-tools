"""Reading and writing the GPXIL-TIFF container (spec section 8).

A gpxil file is an ordinary multi-page TIFF: one page per (contig, plane),
8-bit palette pages for the categorical planes so any image viewer shows a
picture, strips of whole rows so a genomic interval is read by decoding only
the strips it touches, and JSON metadata in the first page's
ImageDescription.
"""

from __future__ import annotations

import datetime as _dt
import functools
import io
import json
import math
import os
import re
import zlib
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import tifffile

from . import __version__
from . import codes as C
from .palettes import DEFAULT_PALETTE, flags_view_colours, gt_view_colours, ref_colours, tiff_colormap
from .remote import RangeFile, is_url
from .vcf import SidecarEntry

FORMAT_VERSION = "0.1"
SIDECAR_TAG = 65400  #: private TIFF tag (reusable range) holding the zlib-compressed sidecar TSV
SIDECAR_HEADER = "#gpxil-sidecar\t0.1\noffset\thap\tkind\tdata\n"

STANDARD_PLANES: dict[str, dict] = {
    "ref": {"dtype": "uint8", "photometric": "palette", "missing": None},
    "gt": {"dtype": "uint8", "photometric": "palette", "missing": None},
    "flags": {"dtype": "uint8", "photometric": "palette", "missing": None},
    "gq": {"dtype": "uint8", "photometric": "minisblack", "missing": C.QUAL_MISSING},
    "dp": {"dtype": "uint8", "photometric": "minisblack", "missing": C.QUAL_MISSING},
    "ps": {"dtype": "uint32", "photometric": "minisblack", "missing": 0},
}
PLANE_ORDER = list(STANDARD_PLANES)
EXTENSION_DTYPES = {"uint8", "uint16", "uint32", "int8", "int16", "int32", "float32"}

COMPRESSIONS = {
    # name: (tifffile compression, default level)
    "deflate": ("zlib", 9),
    "zstd": ("zstd", 19),
    "lzw": ("lzw", None),
    "lzma": ("lzma", None),
    "packbits": ("packbits", None),
    "none": (None, None),
}
DEFAULT_COMPRESSION = "deflate"
DEFAULT_WIDTH = 8192
DEFAULT_STRIP_BYTES = 1 << 18
#: TIFF Compression tag values of each compression (32946 is the obsolete code for Deflate)
COMPRESSION_TAGS = {"deflate": (8, 32946), "zstd": (50000,), "lzw": (5,), "lzma": (34925,), "packbits": (32773,), "none": (1,)}
#: contig metadata keys that describe the gt plane, or the ps plane, and go when those planes do
GT_ENTRY_KEYS = ("ploidy", "vcf_contig", "encoding", "callable_bases", "phase_sets", "phase_set_ids")
PS_ENTRY_KEYS = ("phase_sets", "phase_set_ids")
COPY_READ_BYTES = 1 << 23  #: copied strips are read up to this many bytes at a time (one request for a file on the web)


def compression_of(page) -> str | None:
    """The name (as in COMPRESSIONS) of a page's compression, or None for one gpxil files do not use."""
    code = int(page.compression)
    return next((name for name, tags in COMPRESSION_TAGS.items() if code in tags), None)


TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


@functools.cache
def codec_available(compression: str) -> bool:
    """True if this Python can write and read the compression.

    Deflate (the default), LZMA and no compression need only Python itself;
    Zstandard, LZW and PackBits need the imagecodecs package (Zstandard also
    works on Python 3.14 and later), which is not available everywhere, for
    example in Pyodide in a web browser.
    """
    codec, _ = COMPRESSIONS[compression]
    if codec is None:
        return True
    buf = io.BytesIO()
    try:
        tifffile.imwrite(buf, np.zeros((1, 1), dtype=np.uint8), compression=codec)
        buf.seek(0)
        tifffile.imread(buf)
    except Exception:
        return False
    return True


def _codec_error(page, exc: Exception) -> GpxilFormatError:
    """A missing codec, said plainly (tifffile raises KeyError or ImportError)."""
    comp = getattr(getattr(page, "compression", None), "name", "this compression")
    return GpxilFormatError(
        f"the file is compressed with {comp}, which this Python cannot decode: install imagecodecs (pip install imagecodecs) ({exc})"
    )


def _decode(page, data: bytes, index: int):
    try:
        return page.decode(data, index)
    except (KeyError, ImportError) as exc:
        raise _codec_error(page, exc) from None


def _asarray(page) -> np.ndarray:
    try:
        return page.asarray()
    except (KeyError, ImportError) as exc:
        raise _codec_error(page, exc) from None


def is_tiff(path) -> bool:
    """True for a URL ending in .tif/.tiff, or a local file that starts with a TIFF header (a gpxil file, not FASTA)."""
    if is_url(path):
        return path.lower().split("?", 1)[0].endswith((".tif", ".tiff"))
    try:
        with open(path, "rb") as fh:
            return fh.read(4) in TIFF_MAGIC
    except OSError:
        return False


CLASSIC_TIFF_LIMIT = (1 << 32) - (1 << 26)  #: stay 64 MiB clear of the 4 GiB classic-TIFF offset limit


def _no_codec_message(compression: str) -> str:
    return (
        f"{compression} compression needs the imagecodecs package, which this Python does not have "
        "(pip install imagecodecs), or use deflate, the default"
    )


class GpxilFormatError(ValueError):
    pass


class BigTiffNeeded(GpxilFormatError):
    """Raised by a writer in automatic mode when the file would pass 4 GiB as classic TIFF."""


class MissingPlaneError(KeyError):
    """A contig does not have the requested plane."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


_EXT_NAME = re.compile(r"^x-[A-Za-z0-9_-]+$")
_CONTIG_NAME = re.compile(r"^[\x21-\x7e]+$")


def check_contig_name(name: str) -> None:
    """Contig names are printable ASCII without spaces (spec 9.3)."""
    if not isinstance(name, str) or not _CONTIG_NAME.match(name):
        raise GpxilFormatError(f"contig name {name!r} must be non-empty printable ASCII without whitespace")


def plane_dtype(name: str, arr: np.ndarray | None = None) -> np.dtype:
    if name in STANDARD_PLANES:
        return np.dtype(STANDARD_PLANES[name]["dtype"])
    if not _EXT_NAME.match(name):
        raise GpxilFormatError(f"plane name {name!r} is neither standard ({', '.join(PLANE_ORDER)}) nor an extension name x-[A-Za-z0-9_-]+")
    if arr is None:
        raise GpxilFormatError(f"extension plane {name!r} needs data to infer its dtype")
    if arr.dtype.name not in EXTENSION_DTYPES:
        raise GpxilFormatError(f"extension plane {name!r} has unsupported dtype {arr.dtype}")
    return arr.dtype


def sort_planes(names: Iterable[str]) -> list[str]:
    names = list(names)
    std = [p for p in PLANE_ORDER if p in names]
    ext = sorted(p for p in names if p not in STANDARD_PLANES)
    return std + ext


def rows_for(length: int, width: int) -> int:
    return max(1, math.ceil(length / width))


def to_image(values: np.ndarray, width: int) -> np.ndarray:
    """1-D per-base array -> 2-D raster (row-major), padded with 0."""
    n = values.shape[0]
    rows = rows_for(n, width)
    if rows * width == n:
        return values.reshape(rows, width)
    out = np.zeros(rows * width, dtype=values.dtype)
    out[:n] = values
    return out.reshape(rows, width)


def _stored_strip_bytes(page):
    """A page's strips as they are stored, reading runs of adjacent strips at once (one request on the web)."""
    fh = page.parent.filehandle
    offsets, counts = page.dataoffsets, page.databytecounts
    i, n = 0, len(offsets)
    while i < n:
        j, total = i + 1, counts[i]
        while j < n and offsets[j] == offsets[j - 1] + counts[j - 1] and total + counts[j] <= COPY_READ_BYTES:
            total += counts[j]
            j += 1
        fh.seek(offsets[i])
        data = fh.read(total)
        if len(data) != total:
            raise GpxilFormatError(f"the file ends inside strip {i} of a page: is it complete?")
        pos = 0
        for k in range(i, j):
            yield data[pos : pos + counts[k]]
            pos += counts[k]
        i = j


def _several_phase_sets(contig) -> bool:
    """True if a contig's ps plane holds more than one phase set (from its metadata, or by reading the plane)."""
    known = contig.entry.get("phase_sets")
    if isinstance(known, int):
        return known > 1
    seen: set[int] = set()
    for s in range(0, contig.length, 1 << 24):
        seen.update(int(v) for v in np.unique(contig.read("ps", s, min(contig.length, s + (1 << 24)))) if v)
        if len(seen) > 1:
            return True
    return False


def encode_sidecar(entries: list[SidecarEntry]) -> bytes:
    text = SIDECAR_HEADER + "".join(e.to_tsv() + "\n" for e in entries)
    return zlib.compress(text.encode("utf-8"), 9)


def decode_sidecar(blob: bytes) -> list[SidecarEntry]:
    text = zlib.decompress(blob).decode("utf-8")
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith("offset\t"):
            continue
        off, hap, kind, data = line.split("\t", 3)
        out.append(SidecarEntry(int(off), int(hap), kind, data))
    return out


# --------------------------------------------------------------------- writer


@dataclass
class ContigInfo:
    name: str
    length: int
    width: int
    pages: dict[str, int]
    extra: dict

    def as_dict(self) -> dict:
        d = {"name": self.name, "length": self.length, "width": self.width, "pages": self.pages}
        d.update(self.extra)
        return d


class GpxilWriter:
    """Stream contigs into a gpxil file, one contig at a time.

    >>> with GpxilWriter("out.gpxil.tif", reference={"name": "GRCh38"}) as w:
    ...     w.add_contig("chr21", {"ref": ref_codes, "gt": gt, "flags": flags})
    """

    def __init__(
        self,
        path,
        *,
        width: int = DEFAULT_WIDTH,
        compression: str = DEFAULT_COMPRESSION,
        level: int | None = None,
        strip_bytes: int = DEFAULT_STRIP_BYTES,
        bigtiff: bool | None = None,
        palette: str = DEFAULT_PALETTE,
        reference: dict | None = None,
        sample: dict | None = None,
        metadata: dict | None = None,
        require_codec: bool = True,
    ):
        if compression not in COMPRESSIONS:
            raise ValueError(f"unknown compression {compression!r}; choose from {', '.join(COMPRESSIONS)}")
        # (require_codec=False: a copy whose strips are all copied as stored needs no codec)
        if require_codec and not codec_available(compression):
            raise GpxilFormatError(_no_codec_message(compression))
        if width < 1:
            raise ValueError("width must be positive")
        self.path = os.fspath(path)
        self.width = int(width)
        self.compression = compression
        codec, default_level = COMPRESSIONS[compression]
        self._codec = codec
        self._level = default_level if level is None else level
        self.strip_bytes = strip_bytes
        self.palette = palette
        self.reference = reference or {}
        self.sample = sample
        self.extra_metadata = metadata or {}
        # bigtiff=None: classic TIFF, raising BigTiffNeeded before the file
        # could pass 4 GiB (gpxiltools.encode then starts again as BigTIFF)
        self._auto_bigtiff = bigtiff is None
        self.bigtiff = bool(bigtiff)
        self._tw = tifffile.TiffWriter(self.path, bigtiff=self.bigtiff)
        self._page = 0
        self.contigs: list[ContigInfo] = []
        self._names: set[str] = set()
        ref_rgb, _ = ref_colours(palette)
        self._colormaps = {
            "ref": tiff_colormap(ref_rgb),
            "gt": tiff_colormap(gt_view_colours()),
            "flags": tiff_colormap(flags_view_colours()),
        }
        self._plane_dtypes: dict[str, str] = {}
        #: extra keys for the 'planes' metadata of a plane (description, units, ...), as copied from another file
        self.plane_info: dict[str, dict] = {}
        #: how many pages the last copy_contig copied as stored and encoded again
        self.last_copy: dict[str, int] = {"stored": 0, "encoded": 0}
        self._closed = False

    def __enter__(self) -> GpxilWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def abort(self) -> None:
        """Close and delete a file that could not be completed."""
        if not self._closed:
            try:
                self._tw.close()
            except Exception:
                pass
            self._closed = True
        try:
            os.remove(self.path)
        except OSError:
            pass

    def add_contig(
        self,
        name: str,
        planes: dict[str, np.ndarray],
        *,
        length: int | None = None,
        width: int | None = None,
        sidecar: list[SidecarEntry] | None = None,
        **extra,
    ) -> ContigInfo:
        """Write all planes of one contig. ``extra`` goes into its metadata entry."""
        if self._closed:
            raise RuntimeError("writer is closed")
        check_contig_name(name)
        if name in self._names:
            raise GpxilFormatError(f"duplicate contig name {name!r}")
        if not planes:
            raise GpxilFormatError(f"contig {name!r} has no planes")
        lengths = {p: int(a.shape[0]) for p, a in planes.items()}
        if length is None:
            length = next(iter(lengths.values()))
        bad = {p: n for p, n in lengths.items() if n != length}
        if bad:
            raise GpxilFormatError(f"contig {name!r}: planes {bad} differ from length {length}")
        if length < 1:
            raise GpxilFormatError(f"contig {name!r} is empty; contigs must have at least one base")
        if sidecar and "gt" not in planes:
            raise GpxilFormatError("a sidecar is stored with the gt plane, which is missing")
        if "gt" in planes and "flags" not in planes:
            raise GpxilFormatError(f"contig {name!r}: a gt plane needs a flags plane")
        if "gt" not in planes and any(p in planes for p in ("gq", "dp", "ps")):
            raise GpxilFormatError(f"contig {name!r}: gq, dp and ps planes need a gt plane")
        if "gt" in planes and not (isinstance(self.sample, dict) and self.sample.get("id")):
            raise GpxilFormatError("a gt plane describes a sample: give GpxilWriter(sample={'id': ...})")
        w = int(width or self.width)
        rows = rows_for(length, w)
        if not self.bigtiff:
            raw = sum(rows * w * plane_dtype(p, a).itemsize for p, a in planes.items())
            if self._tw.filehandle.tell() + raw > CLASSIC_TIFF_LIMIT:
                if self._auto_bigtiff:
                    raise BigTiffNeeded(f"{self.path} may pass 4 GiB")
                raise GpxilFormatError(f"{self.path} may pass 4 GiB: write it with bigtiff=True")
        pages: dict[str, int] = {}
        for plane in sort_planes(planes):
            arr = np.ascontiguousarray(planes[plane])
            dt = plane_dtype(plane, arr)
            if arr.dtype != dt:
                arr = arr.astype(dt)
            prev = self._plane_dtypes.setdefault(plane, dt.name)
            if prev != dt.name:
                raise GpxilFormatError(f"plane {plane!r} has dtype {dt.name} here but {prev} in an earlier contig")
            img = to_image(arr, w)
            blob = encode_sidecar(sidecar) if plane == "gt" and sidecar else None
            self._tw.write(img, **self._page_kwargs(name, plane, length, self._rows_per_strip(rows, w, dt), blob))
            pages[plane] = self._page
            self._page += 1
        if sidecar:
            extra = dict(extra)
            extra["sidecar_entries"] = len(sidecar)
        info = ContigInfo(name=name, length=length, width=w, pages=pages, extra=extra)
        self.contigs.append(info)
        self._names.add(name)
        return info

    def _rows_per_strip(self, rows: int, width: int, dt: np.dtype) -> int:
        return max(1, min(rows, self.strip_bytes // max(1, width * dt.itemsize)))

    def _page_kwargs(self, name: str, plane: str, length: int, rps: int, sidecar_blob: bytes | None) -> dict:
        """tifffile.TiffWriter.write arguments for one page: tags, page metadata and colour map (section 8)."""
        desc = {"gpxil": FORMAT_VERSION, "contig": name, "plane": plane, "length": length}
        kwargs = dict(
            compression=self._codec,
            rowsperstrip=rps,
            description=json.dumps(desc, ensure_ascii=True),
            metadata=None,
            software=f"gpxiltools {__version__}",
            datetime=_dt.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
        )
        if self._codec is not None and self._level is not None:
            kwargs["compressionargs"] = {"level": self._level}
        if plane in self._colormaps:
            kwargs.update(photometric="palette", colormap=self._colormaps[plane])
        else:
            kwargs.update(photometric="minisblack")
        extratags = [(285, "s", 0, f"{name} {plane}", True)]  # PageName
        if sidecar_blob:
            extratags.append((SIDECAR_TAG, 7, len(sidecar_blob), sidecar_blob, True))
        kwargs["extratags"] = extratags
        return kwargs

    def _check_dtype(self, plane: str, dt: np.dtype) -> None:
        prev = self._plane_dtypes.setdefault(plane, dt.name)
        if prev != dt.name:
            raise GpxilFormatError(f"plane {plane!r} has dtype {dt.name} here but {prev} in an earlier contig")

    def _stored_strips(self, page, dt: np.dtype, level_given: bool) -> bool:
        """True if the page's strips can be copied as they are stored: the same compression, strips, no predictor."""
        if level_given or page.is_tiled or page.rowsperstrip <= 0 or self.compression == "none":
            return False
        if compression_of(page) != self.compression or int(page.predictor) not in (0, 1) or int(page.fillorder) != 1:
            return False
        if page.dtype != dt or (dt.itemsize > 1 and page.parent.byteorder != self._tw.tiff.byteorder):
            return False
        return len(page.dataoffsets) == math.ceil(page.shape[0] / page.rowsperstrip)

    def copy_contig(self, src: Contig, planes: Iterable[str] | None = None, *, recompress: bool = False) -> ContigInfo:
        """Copy a contig of an open gpxil file, with all or some of its planes, strip by strip.

        Strips are copied as they are stored when the compression is the
        same (and ``recompress`` is false), without being decoded; otherwise
        they are decoded and encoded again a strip at a time. Either way the
        memory used is about one strip, whatever the contig's size, and the
        source can be a file on the web. ``planes`` defaults to the
        contig's own planes; ``ref`` can also come from a reference attached
        to the source file. The contig keeps its metadata entry, without the
        keys that describe planes left out.
        """
        if self._closed:
            raise RuntimeError("writer is closed")
        name, length, w, rows = src.name, src.length, src.width, src.rows
        check_contig_name(name)
        if name in self._names:
            raise GpxilFormatError(f"duplicate contig name {name!r}")
        want = sort_planes(src.planes if planes is None else dict.fromkeys(planes))
        if not want:
            raise GpxilFormatError(f"contig {name!r}: no planes to copy")
        for p in want:
            if not src.has(p):
                src._page(p)  # raises MissingPlaneError, saying how to give a missing reference
        if "gt" in want and "flags" not in want:
            raise GpxilFormatError(f"contig {name!r}: a gt plane needs its flags plane (section 3)")
        if "gt" not in want and any(p in want for p in ("gq", "dp", "ps")):
            raise GpxilFormatError(f"contig {name!r}: gq, dp and ps planes need the gt plane (section 3)")
        if "gt" in want and not (isinstance(self.sample, dict) and self.sample.get("id")):
            raise GpxilFormatError("a gt plane describes a sample: give GpxilWriter(sample={'id': ...})")
        if "ref" not in want and not (src.entry.get("md5") or src.entry.get("refget")):
            raise GpxilFormatError(
                f"contig {name!r}: without its ref plane a contig needs the md5 or refget digest of its reference (section 3)"
            )
        if "gt" in want and "ps" in src.pages and "ps" not in want and _several_phase_sets(src):
            raise GpxilFormatError(
                f"contig {name!r} has more than one phase set: without the ps plane its phased calls would read as one"
                " phase set (section 5.5), so keep ps"
            )
        side = src.sidecar() if "gt" in want else []
        plan: dict[str, tuple[np.dtype, object | None]] = {}
        size = 0
        for p in want:
            stored = p in src.pages
            page = src._page(p) if stored else None
            dt = plane_dtype(p, np.zeros(0, dtype=page.dtype) if page is not None else None) if p.startswith("x-") else plane_dtype(p)
            raw = page if (page is not None and self._stored_strips(page, dt, recompress)) else None
            if raw is None and not codec_available(self.compression):
                raise GpxilFormatError(_no_codec_message(self.compression))
            plan[p] = (dt, raw)
            size += sum(raw.databytecounts) if raw is not None else rows * w * dt.itemsize
        if not self.bigtiff and self._tw.filehandle.tell() + size > CLASSIC_TIFF_LIMIT:
            if self._auto_bigtiff:
                raise BigTiffNeeded(f"{self.path} may pass 4 GiB")
            raise GpxilFormatError(f"{self.path} may pass 4 GiB: write it with bigtiff=True")
        pages: dict[str, int] = {}
        self.last_copy = {
            "stored": sum(raw is not None for _, raw in plan.values()),
            "encoded": sum(raw is None for _, raw in plan.values()),
        }
        for p in want:
            dt, raw = plan[p]
            self._check_dtype(p, dt)
            if raw is not None:
                rps, strips = raw.rowsperstrip, _stored_strip_bytes(raw)
            else:
                rps = self._rows_per_strip(rows, w, dt)
                strips = self._encoded_strips(src, p, dt, rps)
            blob = encode_sidecar(side) if p == "gt" and side else None
            kwargs = self._page_kwargs(name, p, length, rps, blob)
            kwargs.pop("compressionargs", None)  # (the strips arrive encoded)
            self._tw.write(strips, shape=(rows, w), dtype=dt, **kwargs)
            pages[p] = self._page
            self._page += 1
        extra = {k: v for k, v in src.entry.items() if k not in ("name", "length", "width", "pages", "sidecar_entries")}
        for k in GT_ENTRY_KEYS if "gt" not in pages else PS_ENTRY_KEYS if "ps" not in pages and "ps" in src.pages else ():
            extra.pop(k, None)
        if side:
            extra["sidecar_entries"] = len(side)
        info = ContigInfo(name=name, length=length, width=w, pages=pages, extra=extra)
        self.contigs.append(info)
        self._names.add(name)
        return info

    def _encoded_strips(self, src: Contig, plane: str, dt: np.dtype, rps: int):
        """The plane's strips, read from the source a strip's worth of rows at a time and encoded."""
        from tifffile import TIFF

        w, rows, n = src.width, src.rows, src.length
        order = self._tw.tiff.byteorder
        code = COMPRESSION_TAGS[self.compression][0]
        args = {} if self._level is None else {"level": self._level}
        for r0 in range(0, rows, rps):
            r1 = min(rows, r0 + rps)
            a, b = r0 * w, min(n, r1 * w)
            buf = np.zeros((r1 - r0) * w, dtype=dt.newbyteorder(order))
            buf[: b - a] = src.read(plane, a, b)
            if code == 1:
                yield buf.tobytes()
            else:
                yield bytes(TIFF.COMPRESSORS[code](buf.reshape(r1 - r0, w), **args))

    def metadata(self) -> dict:
        planes = {}
        for p, dt in self._plane_dtypes.items():
            spec = {**self.plane_info.get(p, {}), "dtype": dt}
            if p in STANDARD_PLANES and STANDARD_PLANES[p]["missing"] is not None:
                spec["missing"] = STANDARD_PLANES[p]["missing"]
            planes[p] = spec
        meta = {
            "gpxil": FORMAT_VERSION,
            "generator": f"gpxiltools {__version__}",
            "created": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
            "layout": {"type": "raster", "width": self.width},
            "palette": self.palette,
            "reference": self.reference,
            # a file without genotype planes describes no sample (spec 9.1)
            "sample": self.sample if any("gt" in c.pages for c in self.contigs) else None,
            "planes": planes,
            "contigs": [c.as_dict() for c in self.contigs],
        }
        if self.contigs:
            first = self.contigs[0]
            p0 = next(p for p, i in first.pages.items() if i == 0)
            meta.update({"contig": first.name, "plane": p0, "length": first.length})
        meta.update(self.extra_metadata)
        return meta

    def close(self) -> None:
        if self._closed:
            return
        self._tw.close()
        self._closed = True
        if not self.contigs:
            raise GpxilFormatError("no contigs were written")
        tifffile.tiffcomment(self.path, json.dumps(self.metadata(), ensure_ascii=True, separators=(",", ":")))


# --------------------------------------------------------------------- reader


class Contig:
    """One contig of an open gpxil file. Reads planes lazily, strip by strip."""

    def __init__(self, gfile: GpxilFile, entry: dict):
        self._g = gfile
        self.entry = entry
        self.name: str = entry["name"]
        self.length: int = int(entry["length"])
        self.width: int = int(entry.get("width") or gfile.metadata["layout"]["width"])
        self.rows: int = rows_for(self.length, self.width)
        self.pages: dict[str, int] = {k: int(v) for k, v in entry["pages"].items()}

    def __repr__(self) -> str:
        return f"<Contig {self.name} length={self.length} planes={list(self.pages)}>"

    @property
    def planes(self) -> list[str]:
        return list(self.pages)

    def has(self, plane: str) -> bool:
        """True if the plane is stored, or is ``ref`` and a reference (FASTA or gpxil) is attached."""
        return plane in self.pages or (plane == "ref" and self._g._reference_path is not None)

    def _reference_contig(self) -> Contig:
        """The matching contig of an attached gpxil reference, checked against this contig's digests."""
        cached = self._g._ref_contigs.get(self.name)
        if cached is not None:
            return cached
        ref = self._g._reference_gpxil()
        for n in (self.name, *self.entry.get("aliases", [])):
            for cand in (n, "chr" + n, n.removeprefix("chr")):
                if cand in ref:
                    rc = ref[cand]
                    if rc.length != self.length:
                        raise GpxilFormatError(f"{rc.name} in {ref.path} has length {rc.length}, not {self.length}")
                    if "ref" not in rc.pages:
                        raise GpxilFormatError(f"{rc.name} in {ref.path} has no reference plane either")
                    for k in ("md5", "refget"):
                        if k in self.entry and k in rc.entry and self.entry[k] != rc.entry[k]:
                            raise GpxilFormatError(f"{rc.name} in {ref.path} does not match the {k} of {self.name}")
                    self._g._ref_contigs[self.name] = rc
                    return rc
        raise GpxilFormatError(f"{self.name} not found in {ref.path}")

    def _external_ref(self) -> np.ndarray:
        """The reference of a contig stored without a ref plane, from the attached FASTA."""
        if self._g._reference_is_gpxil:
            return self._reference_contig().read("ref")
        cached = self._g._ref_cache
        if cached is not None and cached[0] == self.name:
            return cached[1]
        from .fasta import iter_fasta, sequence_digests

        names = set()
        for n in (self.name, *self.entry.get("aliases", [])):
            names |= {n, "chr" + n, n.removeprefix("chr")}
        for rec in iter_fasta(self._g._reference_path, invalid="N", only=names):
            if rec.length != self.length:
                raise GpxilFormatError(f"{rec.name} in {self._g._reference_path} has length {rec.length}, not {self.length}")
            dig = sequence_digests(rec.codes)
            for k in ("md5", "refget"):
                if k in self.entry and self.entry[k] != dig[k]:
                    raise GpxilFormatError(f"{rec.name} in {self._g._reference_path} does not match the {k} of {self.name}")
            self._g._ref_cache = (self.name, rec.codes)
            return rec.codes
        raise GpxilFormatError(f"{self.name} not found in {self._g._reference_path}")

    def _page(self, plane: str):
        if plane not in self.pages:
            if plane == "ref":
                raise MissingPlaneError(
                    f"contig {self.name} is stored without its reference sequence; give the reference FASTA "
                    "(GpxilFile(path, reference=...), or --reference on the command line)"
                )
            raise MissingPlaneError(f"contig {self.name} has no {plane!r} plane (has {', '.join(self.pages)})")
        return self._g._tf.pages[self.pages[plane]]

    def image(self, plane: str) -> np.ndarray:
        """The plane as stored: a 2-D (rows, width) raster including padding."""
        return _asarray(self._page(plane))

    def read(self, plane: str, start: int = 0, end: int | None = None) -> np.ndarray:
        """Per-base values of ``plane`` for positions ``[start, end)`` (0-based)."""
        end = self.length if end is None else end
        if not (0 <= start <= end <= self.length):
            raise IndexError(f"{self.name}:{start}-{end} outside 0-{self.length}")
        if plane == "ref" and "ref" not in self.pages and self._g._reference_path is not None:
            if self._g._reference_is_gpxil:  # read just the region from the reference file
                return self._reference_contig().read("ref", start, end)
            return self._external_ref()[start:end].copy()
        page = self._page(plane)
        if start == end:
            return np.zeros(0, dtype=page.dtype)
        w = self.width
        r0, r1 = start // w, (end - 1) // w + 1
        if page.is_tiled or page.rowsperstrip <= 0:
            flat = _asarray(page).reshape(-1)
            return flat[start:end].copy()
        rps = page.rowsperstrip
        s0, s1 = r0 // rps, (r1 - 1) // rps + 1
        fh = self._g._tf.filehandle
        chunks = []
        for s in range(s0, s1):
            fh.seek(page.dataoffsets[s])
            data = fh.read(page.databytecounts[s])
            seg, _, _ = _decode(page, data, s)
            nrows = min(rps, self.rows - s * rps)
            chunks.append(np.asarray(seg).reshape(-1)[: nrows * w])
        flat = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        base = s0 * rps * w
        return flat[start - base : end - base].copy()

    def sidecar(self) -> list[SidecarEntry]:
        if "gt" not in self.pages:
            return []
        tag = self._page("gt").tags.get(SIDECAR_TAG)
        if tag is None:
            return []
        return decode_sidecar(bytes(tag.value))

    # convenience views --------------------------------------------------
    def sequence(self, start: int = 0, end: int | None = None) -> str:
        """Reference sequence (soft-masked bases in lowercase)."""
        return C.ref_to_ascii(self.read("ref", start, end)).decode("ascii")


class GpxilFile:
    """An open gpxil file: a path, a URL (read with HTTP range requests) or a binary file object.

    >>> with GpxilFile("HG002.gpxil.tif") as g:
    ...     chr21 = g["chr21"]
    ...     gt = chr21.read("gt", 5_000_000, 5_010_000)

    ``reference`` is a FASTA file, or a gpxil file (path or URL) with the
    reference plane, for files written without one (``--no-ref``).
    """

    def __init__(self, path, reference=None):
        self._fh = None
        if hasattr(path, "read") and hasattr(path, "seek"):
            self.path = str(getattr(path, "name", None) or "<file object>")
            source = path
        elif is_url(path):
            self.path = path
            source = self._fh = RangeFile(path)
        else:
            self.path = os.fspath(path)
            source = self.path
        self._reference_path = None
        self._reference_is_gpxil = False
        self._ref_file: GpxilFile | None = None
        self._ref_contigs: dict[str, Contig] = {}
        self._ref_cache: tuple[str, np.ndarray] | None = None
        if reference is not None:
            self.attach_reference(reference)
        try:
            self._tf = tifffile.TiffFile(source)
        except Exception:
            if self._fh is not None:
                self._fh.close()
            raise
        try:
            desc = self._tf.pages[0].description
            meta = json.loads(desc)
        except Exception as exc:
            self.close()
            raise GpxilFormatError(f"{self.path}: first page has no gpxil JSON metadata") from exc
        if not isinstance(meta, dict) or "gpxil" not in meta or "contigs" not in meta:
            self.close()
            raise GpxilFormatError(f"{self.path}: not a gpxil file (no 'gpxil'/'contigs' keys in metadata)")
        major = str(meta["gpxil"]).split(".")[0]
        if major != FORMAT_VERSION.split(".")[0]:
            self.close()
            raise GpxilFormatError(f"{self.path}: gpxil version {meta['gpxil']} not supported by this reader")
        self.metadata: dict = meta
        self._contigs = {e["name"]: Contig(self, e) for e in meta["contigs"]}
        self._aliases: dict[str, str] = {}
        for e in meta["contigs"]:
            for a in e.get("aliases", []):
                self._aliases.setdefault(a, e["name"])

    def __enter__(self) -> GpxilFile:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._tf.close()
        if self._fh is not None:
            self._fh.close()
        if self._ref_file is not None:
            self._ref_file.close()
            self._ref_file = None

    def __repr__(self) -> str:
        return f"<GpxilFile {self.path} contigs={len(self._contigs)} sample={self.sample_id}>"

    @property
    def contigs(self) -> list[str]:
        return list(self._contigs)

    def attach_reference(self, reference) -> None:
        """Use a reference for contigs stored without a ref plane, checked against their digests.

        ``reference`` is a FASTA file, or a gpxil file (path or URL) whose
        reference plane is then read region by region.
        """
        self._reference_path = reference if is_url(reference) else os.fspath(reference)
        self._reference_is_gpxil = is_tiff(self._reference_path)
        if self._ref_file is not None:
            self._ref_file.close()
        self._ref_file = None
        self._ref_contigs = {}
        self._ref_cache = None

    def _reference_gpxil(self) -> GpxilFile:
        if self._ref_file is None:
            self._ref_file = GpxilFile(self._reference_path)
        return self._ref_file

    @property
    def sample_id(self) -> str | None:
        s = self.metadata.get("sample")
        return s.get("id") if isinstance(s, dict) else None

    def __iter__(self):
        return iter(self._contigs.values())

    def __len__(self) -> int:
        return len(self._contigs)

    def __contains__(self, name: str) -> bool:
        return name in self._contigs or name in self._aliases

    def __getitem__(self, name: str) -> Contig:
        if name in self._contigs:
            return self._contigs[name]
        if name in self._aliases:
            return self._contigs[self._aliases[name]]
        raise KeyError(f"no contig {name!r} in {self.path}")


def open_gpxil(path, reference=None) -> GpxilFile:
    return GpxilFile(path, reference=reference)
