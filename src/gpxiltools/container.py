"""Reading and writing the GPXIL-TIFF container (spec section 8).

A gpxil file is an ordinary multi-page TIFF: one page per (contig, plane),
8-bit palette pages for the categorical planes so any image viewer shows a
picture, strips of whole rows so a genomic interval is read by decoding only
the strips it touches, and JSON metadata in the first page's
ImageDescription.
"""

from __future__ import annotations

import bisect
import datetime as _dt
import functools
import heapq
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
from .vcf import SidecarEntry, window_entries

FORMAT_VERSION = "0.3"  #: the version of the specification this writes (and its rules for VCF calls, section 6)
KNOWN_VERSIONS = ("0.1", "0.2", "0.3")  #: versions this reads and checks as they are (any 0.x is read, spec 14)
WINDOW_VERSION = "0.3"  #: the first version with windows (spec 3.1): a file that holds one has this version or a later one
SIDECAR_TAG = 65400  #: private TIFF tag (reusable range) holding the zlib-compressed sidecar TSV
SIDECAR_HEADER = "#gpxil-sidecar\t0.1\noffset\thap\tkind\tdata\n"  # (the sidecar's own version: unchanged in gpxil 0.2 and 0.3)


def version_key(v) -> tuple[int, ...]:
    """'0.10' -> (0, 10): versions compared as numbers. Anything else sorts first."""
    try:
        return tuple(int(x) for x in str(v).split("."))
    except ValueError:
        return (-1,)


def later_version(a, b) -> str:
    """The later of two versions."""
    return str(a) if version_key(a) >= version_key(b) else str(b)


#: NAME:START-END or NAME:POS, 1-based; numbers may have commas and spaces, and don't begin with 0, so that a name with
#: colons in it (HLA-A*01:01:01:01) is a name. The name ends at the last colon (gpxil-read.js's parseRegion is the same)
_REGION = re.compile(r"^(.+):[ ]*([1-9][0-9, ]*?)[ ]*(?:-[ ]*([1-9][0-9, ]*?))?$")
_ZERO = re.compile(r"^.+:[ ]*0[ ]*(?:-[ ]*[0-9, ]+)?$")  # (a region from position 0: an error, not a name)
#: commas that group a region's digits, after the piece of a region before them (split_names)
_DIGIT_GROUP = re.compile(r"^[0-9]{3}(?:[ ]*-.*)?$")


def split_names(text: str) -> list[str]:
    """Names and regions given as text, separated by commas: 'chr1,chr2' -> ['chr1', 'chr2']; but commas that group
    a region's digits stay in it, 'chr17:7,661,779-7,687,546,chr2' -> ['chr17:7,661,779-7,687,546', 'chr2'], and a
    name with colons in it is a name, 'HLA-A*01:14,HLA-A*01:20' -> both."""
    out: list[str] = []
    for piece in str(text).split(","):
        if out and ":" in out[-1] and _DIGIT_GROUP.match(piece.strip(" ")):
            out[-1] += "," + piece
        else:
            out.append(piece)
    return [x.strip(" ") for x in out if x.strip(" ")]


def parse_region(text: str) -> tuple[str, int | None, int | None]:
    """'chr17:7,661,779-7,687,546' -> ('chr17', 7661778, 7687546): 1-based and inclusive in, 0-based with the end
    excluded out; 'chr17:7,676,154' -> one base, ('chr17', 7676153, 7676154); 'chr17' -> ('chr17', None, None). The
    name ends at the last colon, so a window's name (spec 3.1), 'chr17:7561779-7787546', can be followed by a region
    as well; numbers don't begin with 0, so 'HLA-A*01:01:01:01' is a name."""
    text = str(text).strip(" \t\r\n")
    m = _REGION.match(text)
    if not m:
        if not text:
            raise ValueError("an empty region")
        if _ZERO.match(text):
            raise ValueError(f"bad interval in {text!r}: positions count from 1")
        return text, None, None
    num = lambda x: int(re.sub(r"[, ]", "", x))  # noqa: E731
    s = num(m.group(2))
    e = s if m.group(3) is None else num(m.group(3))
    if e < s:
        raise ValueError(f"bad interval in {text!r}: give START-END, 1-based, START no more than END")
    return m.group(1), s - 1, e


def region_text(name: str, start: int, end: int, *, commas: bool = False) -> str:
    """('chr17', 7661778, 7687546) -> 'chr17:7661779-7687546' (1-based, inclusive), or with commas for people."""
    return f"{name}:{start + 1:,}-{end:,}" if commas else f"{name}:{start + 1}-{end}"


def path_name(path) -> str:
    """The name of a file, from its path or URL."""
    s = str(path).rstrip("/")
    return s.rsplit("/", 1)[-1] if is_url(s) else os.path.basename(s)


def _listed(contigs, k: int = 4) -> str:
    return ", ".join(c.region() for c in contigs[:k]) + (", …" if len(contigs) > k else "")


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


def _whole_number(x) -> bool:
    """An integer, also as JSON may write one (2.0), as JSON Schema's integer is; and one that JSON readers hold
    exactly (at most 2^53 - 1, as gpxil-read.js reads it)."""
    if isinstance(x, bool) or not (isinstance(x, int) or (isinstance(x, float) and x.is_integer())):
        return False
    return abs(x) <= 2**53 - 1


def check_window(name: str, length: int, window, aliases: Iterable[str] = ()) -> dict:
    """A window's ``window`` key (spec 3.1, 9.3), checked: its parent named, its start a whole number, the window
    inside the parent, and none of its names the parent's. Returns it; raises GpxilFormatError."""
    if not isinstance(window, dict):
        raise GpxilFormatError(f"contig {name}: 'window' must be an object, with 'of' and 'start'")
    of, start, plen = window.get("of"), window.get("start"), window.get("length")
    if not isinstance(of, str) or not _CONTIG_NAME.match(of):
        raise GpxilFormatError(f"window {name}: 'of' must name its parent, in printable ASCII without whitespace")
    if not _whole_number(start) or start < 0:
        raise GpxilFormatError(f"window {name}: 'start' must be a whole number, 0 or more (got {start!r})")
    if "length" in window and (not _whole_number(plen) or plen < 1):
        raise GpxilFormatError(f"window {name}: the parent's 'length' must be a whole number, 1 or more (got {plen!r})")
    window = {**window, "start": int(start), **({"length": int(plen)} if plen is not None else {})}
    start = window["start"]
    plen = window.get("length")
    if plen is not None and start + length > plen:
        raise GpxilFormatError(f"window {name} runs past the end of {of}: {start:,} + {length:,} bases > {plen:,}")
    others = window.get("aliases", [])
    if not isinstance(others, list) or not all(isinstance(a, str) for a in others):
        raise GpxilFormatError(f"window {name}: the parent's 'aliases' must be a list of names")
    for k in ("md5", "refget"):
        if k in window and not isinstance(window[k], str):
            raise GpxilFormatError(f"window {name}: the parent's {k!r} must be a string")
    parent = {of, *others}
    if name in parent:
        raise GpxilFormatError(f"window {name}: its name is its parent's; a window needs a name of its own, such as {of}:FIRST-LAST")
    clash = sorted(parent & set(aliases))
    if clash:
        raise GpxilFormatError(f"window {name}: its other names can't be its parent's ({', '.join(clash)})")
    return window


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


def _identified(entry: dict) -> bool:
    """Whether a contig entry identifies its reference sequence without a ref plane: its own digest, or (a window)
    its parent's (spec 3, rule 1)."""
    win = entry.get("window") if isinstance(entry.get("window"), dict) else {}
    return bool(entry.get("md5") or entry.get("refget") or win.get("md5") or win.get("refget"))


def _several_phase_sets(contig, start: int = 0, end: int | None = None) -> bool:
    """True if a contig's ps plane holds more than one phase set (from its metadata, or by reading the plane), over
    all of it or over ``[start, end)``."""
    end = contig.length if end is None else end
    known = contig.entry.get("phase_sets")
    if isinstance(known, int) and (known <= 1 or (start, end) == (0, contig.length)):
        return known > 1
    seen: set[int] = set()
    for s in range(start, end, 1 << 24):
        seen.update(int(v) for v in np.unique(contig.read("ps", s, min(end, s + (1 << 24)))) if v)
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


def _window_sidecar(src, start: int, end: int) -> list[SidecarEntry]:
    """The sidecar of a window cut out of ``src`` at ``[start, end)`` (spec 3.1)."""
    first = int(src.read("gt", start, start + 1)[0]) if start > 0 and "gt" in src.pages else None
    return window_entries(src.sidecar(), start, end, first)


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
        version: str = FORMAT_VERSION,
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
        # the specification's version the file follows: this one's, or (gpxil subset) its source's, whose planes it copies
        self.version = str(version)
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
        if "window" in extra:
            self._check_window(name, length, extra["window"], extra.get("aliases", ()))
            if "ref" not in planes and not _identified(extra):
                raise GpxilFormatError(
                    f"window {name!r}: without its ref plane a window needs the md5 or refget digest of its own"
                    " sequence, or of its parent's (sections 3 and 3.1)"
                )
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

    def _check_window(self, name: str, length: int, window, aliases=()) -> None:
        if version_key(self.version) < version_key(WINDOW_VERSION):
            raise GpxilFormatError(
                f"window {name}: windows came with gpxil {WINDOW_VERSION}, and this file is being written as gpxil {self.version}"
            )
        check_window(name, length, window, aliases)

    def _rows_per_strip(self, rows: int, width: int, dt: np.dtype) -> int:
        return max(1, min(rows, self.strip_bytes // max(1, width * dt.itemsize)))

    def _page_kwargs(self, name: str, plane: str, length: int, rps: int, sidecar_blob: bytes | None) -> dict:
        """tifffile.TiffWriter.write arguments for one page: tags, page metadata and colour map (section 8)."""
        desc = {"gpxil": self.version, "contig": name, "plane": plane, "length": length}
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
        if src.window is not None:
            self._check_window(name, length, src.window, src.entry.get("aliases", ()))
        want = self._planes_wanted(src, planes, name)
        if "ref" not in want and not _identified(src.entry):
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

    def _planes_wanted(self, src: Contig, planes: Iterable[str] | None, name: str) -> list[str]:
        """The planes to write of a contig copied or cut, checked against the rules of section 3."""
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
        return want

    def cut_contig(self, src: Contig, start: int, end: int, planes: Iterable[str] | None = None, *, name: str | None = None) -> ContigInfo:
        """Write a window (spec 3.1): offsets ``[start, end)`` of a contig of an open gpxil file, whole or itself a window.

        The planes are read and encoded a strip at a time, so the source can be a whole genome on the web. The
        sidecar entries at ``[start, end)`` are kept, at the window's offsets. The window names its parent (the
        source's parent, if the source is a window) and where in it it starts, and is named, unless ``name`` is
        given, after the region it holds in its parent: ``chr17:7561779-7787546``. Its own digests are worked out
        when its bases can be read (a ref plane, or a reference attached to the source); its parent's are copied.
        Of the source's metadata, it keeps what holds for a stretch too: what the counts of a conversion and the
        callable bases were, it can't tell.
        """
        if self._closed:
            raise RuntimeError("writer is closed")
        if not (0 <= start < end <= src.length):
            raise IndexError(f"{src.name}: {start:,}-{end:,} is outside 0-{src.length:,}")
        if src.window is None and (start, end) == (0, src.length):
            raise GpxilFormatError(f"a window of all of {src.name} would be {src.name} itself: copy the contig instead")
        n = end - start
        name = name or region_text(src.parent, src.start + start, src.start + end)
        check_contig_name(name)
        if name in self._names:
            raise GpxilFormatError(f"duplicate contig name {name!r}")
        win: dict = {"of": src.parent, "start": src.start + start}
        if src.parent_length is not None:
            win["length"] = src.parent_length
        win.update(src.parent_digests())
        if src.parent_aliases:
            win["aliases"] = list(src.parent_aliases)
        self._check_window(name, n, win)
        want = self._planes_wanted(src, planes, name)
        if "gt" in want and "ps" in src.pages and "ps" not in want and _several_phase_sets(src, start, end):
            raise GpxilFormatError(
                f"{src.region(start, end)} has more than one phase set: without the ps plane its phased calls would read"
                " as one phase set (section 5.5), so keep ps"
            )
        readable = src.has("ref")
        if not readable and not _identified({"window": win}):
            raise GpxilFormatError(
                f"{src.name} has no reference plane and no digest of its sequence, so a window of it couldn't be"
                " identified: give the reference (--reference)"
            )
        w = self.width
        rows = rows_for(n, w)
        dts = {p: (plane_dtype(p, np.zeros(0, dtype=src._page(p).dtype)) if p.startswith("x-") else plane_dtype(p)) for p in want}
        if not codec_available(self.compression):
            raise GpxilFormatError(_no_codec_message(self.compression))
        from .fasta import SequenceDigester

        size = sum(rows * w * dt.itemsize for dt in dts.values())
        if not self.bigtiff and self._tw.filehandle.tell() + size > CLASSIC_TIFF_LIMIT:
            if self._auto_bigtiff:
                raise BigTiffNeeded(f"{self.path} may pass 4 GiB")
            raise GpxilFormatError(f"{self.path} may pass 4 GiB: write it with bigtiff=True")
        # what the window's entry says of its own bases, worked out as its planes are written
        digester = SequenceDigester() if readable else None
        seen = {"absent2": 0, "ps": set()}

        def watch(plane):
            if plane == "ref":
                return digester.update
            if plane == "gt":
                return lambda v: seen.__setitem__("absent2", seen["absent2"] + int(np.count_nonzero((v & 0x0F) == C.HAP_ABSENT)))
            if plane == "ps":
                return lambda v: seen["ps"].update(np.unique(v[v != 0]).tolist())
            return None

        side = _window_sidecar(src, start, end) if "gt" in want else []
        pages: dict[str, int] = {}
        self.last_copy = {"stored": 0, "encoded": len(want)}
        for p in want:
            dt = dts[p]
            self._check_dtype(p, dt)
            rps = self._rows_per_strip(rows, w, dt)
            strips = self._window_strips(src, p, dt, rps, start, end, w, watch(p))
            blob = encode_sidecar(side) if p == "gt" and side else None
            kwargs = self._page_kwargs(name, p, n, rps, blob)
            kwargs.pop("compressionargs", None)  # (the strips arrive encoded)
            self._tw.write(strips, shape=(rows, w), dtype=dt, **kwargs)
            pages[p] = self._page
            self._page += 1
        if digester is not None and "ref" not in pages:  # (not written, but its bases can be read: for the digests)
            for s in range(start, end, 1 << 24):
                digester.update(src.read("ref", s, min(end, s + (1 << 24))))
        # (the VCF contig, with the sample's planes only, as copy_contig keeps it)
        keys = ("vcf_contig", "invalid_characters_converted_to_N") if "gt" in pages else ("invalid_characters_converted_to_N",)
        extra: dict = {k: src.entry[k] for k in keys if k in src.entry}
        if digester is not None:
            extra.update(digester.result())
        extra["window"] = win
        if "gt" in pages:
            extra["ploidy"] = 1 if seen["absent2"] == n else 2 if seen["absent2"] == 0 else "mixed"
        if "ps" in pages and seen["ps"]:  # (as encode writes it: with no phase set on its bases, none)
            extra["phase_sets"] = len(seen["ps"])
            ids = src.entry.get("phase_set_ids")
            if isinstance(ids, dict):
                kept = {k: v for k, v in ids.items() if str(k).isdigit() and int(k) in seen["ps"]}
                if kept:
                    extra["phase_set_ids"] = kept
        if side:
            extra["sidecar_entries"] = len(side)
        info = ContigInfo(name=name, length=n, width=w, pages=pages, extra=extra)
        self.contigs.append(info)
        self._names.add(name)
        return info

    def _window_strips(self, src: Contig, plane: str, dt: np.dtype, rps: int, start: int, end: int, w: int, watch=None):
        """The strips of a plane over ``[start, end)`` of a contig, in rows of ``w``, read a strip's worth at a time
        and encoded; ``watch`` sees each piece of values."""
        from tifffile import TIFF

        n = end - start
        rows = rows_for(n, w)
        order = self._tw.tiff.byteorder
        code = COMPRESSION_TAGS[self.compression][0]
        args = {} if self._level is None else {"level": self._level}
        for r0 in range(0, rows, rps):
            r1 = min(rows, r0 + rps)
            a, b = r0 * w, min(n, r1 * w)
            vals = src.read(plane, start + a, start + b)
            if watch is not None:
                watch(vals)
            buf = np.zeros((r1 - r0) * w, dtype=dt.newbyteorder(order))
            buf[: b - a] = vals
            if code == 1:
                yield buf.tobytes()
            else:
                yield bytes(TIFF.COMPRESSORS[code](buf.reshape(r1 - r0, w), **args))

    def _encoded_strips(self, src: Contig, plane: str, dt: np.dtype, rps: int):
        """The plane's strips, read from the source a strip's worth of rows at a time and encoded."""
        return self._window_strips(src, plane, dt, rps, 0, src.length, src.width)

    def metadata(self) -> dict:
        planes = {}
        for p, dt in self._plane_dtypes.items():
            spec = {**self.plane_info.get(p, {}), "dtype": dt}
            if p in STANDARD_PLANES and STANDARD_PLANES[p]["missing"] is not None:
                spec["missing"] = STANDARD_PLANES[p]["missing"]
            planes[p] = spec
        meta = {
            "gpxil": self.version,
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
        #: for a window (spec 3.1), its ``window`` key: the parent, and where the window starts in it; else None
        self.window: dict | None = (
            check_window(self.name, self.length, entry["window"], entry.get("aliases", ())) if "window" in entry else None
        )

    def __repr__(self) -> str:
        where = f" window of {self.parent} at {self.start + 1:,}-{self.end:,}" if self.window else ""
        return f"<Contig {self.name} length={self.length}{where} planes={list(self.pages)}>"

    # where the contig lies in its sequence: for a window, its parent (spec 3.1); for a whole contig, itself
    @property
    def parent(self) -> str:
        """The name of the sequence this contig holds bases of: a window's parent, or the contig itself."""
        return self.window["of"] if self.window else self.name

    @property
    def start(self) -> int:
        """The offset in :attr:`parent` of this contig's offset 0 (0 for a whole contig)."""
        return int(self.window["start"]) if self.window else 0

    @property
    def end(self) -> int:
        """The offset in :attr:`parent` just past this contig's last base."""
        return self.start + self.length

    @property
    def parent_length(self) -> int | None:
        """The length of :attr:`parent`, if known (a window may not say)."""
        if self.window is None:
            return self.length
        n = self.window.get("length")
        return int(n) if n is not None else None

    @property
    def parent_aliases(self) -> list[str]:
        """The other names of :attr:`parent`."""
        return list((self.window or self.entry).get("aliases", []))

    def parent_digests(self) -> dict:
        """The md5 and refget of :attr:`parent`, those known."""
        src = self.window or self.entry
        return {k: src[k] for k in ("md5", "refget") if isinstance(src.get(k), str)}

    def region(self, start: int = 0, end: int | None = None, *, commas: bool = True) -> str:
        """Offsets ``[start, end)`` of this contig as a region of :attr:`parent`, 1-based: 'chr17:7,661,779-7,687,546'."""
        end = self.length if end is None else end
        return region_text(self.parent, self.start + start, self.start + end, commas=commas)

    @property
    def planes(self) -> list[str]:
        return list(self.pages)

    def has(self, plane: str) -> bool:
        """True if the plane is stored, or is ``ref`` and a reference (FASTA or gpxil) is attached."""
        return plane in self.pages or (plane == "ref" and self._g._reference_path is not None)

    def _names_in_reference(self) -> list[str]:
        """The names the sequence this contig holds bases of may have in a reference: its own and its other names,
        each with and without 'chr'."""
        out: list[str] = []
        for n in (self.parent, *self.parent_aliases):
            out += [n, "chr" + n, n.removeprefix("chr")]
        return list(dict.fromkeys(out))

    def _reference_contig(self) -> tuple[Contig, int]:
        """The contig of an attached gpxil reference that holds this contig's bases, checked against its digests,
        and the offset there of this contig's offset 0. For a window, the reference can hold its parent, or a window
        of the parent that holds all of this one."""
        cached = self._g._ref_contigs.get(self.name)
        if cached is not None:
            return cached
        ref = self._g._reference_gpxil()
        for cand in self._names_in_reference():
            try:
                rc, s, e = ref.locate(cand, self.start, self.end) if self.window else (ref[cand], 0, None)
            except (KeyError, IndexError):
                continue
            where = f"{rc.name} in {ref.path}"
            if self.window is None and rc.length != self.length:
                raise GpxilFormatError(f"{where} has length {rc.length}, not {self.length}")
            if "ref" not in rc.pages:
                raise GpxilFormatError(f"{where} has no reference plane either")
            n, theirs = self.parent_length, rc.parent_length
            if n is not None and theirs is not None and n != theirs:
                raise GpxilFormatError(f"{rc.parent} in {ref.path} has length {theirs:,}, not the {n:,} of {self.parent}")
            mine, theirs = self.parent_digests(), rc.parent_digests()
            for k in ("md5", "refget"):
                if k in mine and k in theirs and mine[k] != theirs[k]:
                    raise GpxilFormatError(f"{where} does not match the {k} of {self.parent}")
            if self.window and not (set(mine) & set(theirs)):
                # (no digest of the parent to compare: the window's own bases are checked against its own digests)
                self._check_own_digests(rc.read("ref", s, s + self.length), where)
            self._g._ref_contigs[self.name] = (rc, s)
            return rc, s
        raise GpxilFormatError(f"{self.region() if self.window else self.name} not found in {ref.path}")

    def _check_own_digests(self, codes: np.ndarray, where: str) -> None:
        from .fasta import sequence_digests

        own = {k: self.entry[k] for k in ("md5", "refget") if isinstance(self.entry.get(k), str)}
        if not own:
            return
        dig = sequence_digests(codes)
        for k, v in own.items():
            if dig[k] != v:
                raise GpxilFormatError(f"{where} does not match the {k} of {self.name}")

    def _external_ref(self) -> np.ndarray:
        """The reference of a contig stored without a ref plane, from the attached FASTA (for a window, the stretch
        of its parent it holds). The last sequence read is kept, so a file's windows of one chromosome read it once;
        each contig is checked against its digests the first time."""
        if self._g._reference_is_gpxil:
            rc, s = self._reference_contig()
            return rc.read("ref", s, s + self.length)
        from .fasta import iter_fasta, sequence_digests

        names = self._names_in_reference()
        path = self._g._reference_path
        cached = self._g._ref_cache
        if cached is not None and cached[0] in names:
            rec_name, codes = cached
        else:
            records = iter_fasta(path, invalid="N", only=set(names))
            try:
                rec = next(records, None)
            finally:
                records.close()
            if rec is None:
                raise GpxilFormatError(f"{self.parent} not found in {path}")
            rec_name, codes = rec.name, rec.codes
            self._g._ref_cache = (rec_name, codes)
        mine = codes if self.window is None else codes[self.start : self.end]
        if self.name not in self._g._ref_checked:
            n = self.parent_length
            if (n is not None and len(codes) != n) or len(codes) < self.end:
                raise GpxilFormatError(f"{rec_name} in {path} has length {len(codes):,}, not {n or self.length:,}")
            if self.window is None or not any(isinstance(self.entry.get(k), str) for k in ("md5", "refget")):
                # (a whole contig; or a window that gives only its parent's digests: worked out once for the record)
                got = self._g._ref_digests
                if got is None or got[0] != rec_name or got[1] is not codes:
                    got = self._g._ref_digests = (rec_name, codes, sequence_digests(codes))
                dig = got[2]
                for k, v in self.parent_digests().items():
                    if dig[k] != v:
                        raise GpxilFormatError(f"{rec_name} in {path} does not match the {k} of {self.parent}")
            else:
                self._check_own_digests(mine, f"{rec_name} in {path}")
            self._g._ref_checked.add(self.name)
        return mine

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
                rc, s = self._reference_contig()
                return rc.read("ref", s + start, s + end)
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


#: what a stretch reads where none of its contigs that have the sample's planes holds a base: nothing known there
#: (spec 5.2-5.5). The reference, and extension planes, can't be made up: a stretch has them only if all its contigs do
_STRETCH_FILL = {
    "gt": ((C.HAP_UNKNOWN << 4) | C.HAP_UNKNOWN, np.uint8),
    "flags": (0, np.uint8),
    "gq": (C.QUAL_MISSING, np.uint8),
    "dp": (C.QUAL_MISSING, np.uint8),
    "ps": (0, np.uint32),
}


class Stretch:
    """Contigs of one sequence that overlap or touch, read as one contig over the stretch of it they cover together.

    Each holds what its parent holds over its bases (spec 3.1), so where they overlap they agree, and a stretch holds
    what a window of all of it would: an event cut by the end of one window is whole in the stretch, if another holds
    the rest of it. A base of a plane is read from a contig that has the plane and holds the base (the one that
    reaches furthest). Where no contig with the sample's planes holds a base, the stretch says nothing is known there
    (gt UNKNOWN, no flags, quality, depth or phase set), so contigs with different planes join too; it has the
    reference, and extension planes, only if all its contigs do. The sidecars are merged, each entry once (a
    window's copy, at its first base, of the COMPLEX entry of the allele that base belongs to, spec 3.1, is left out
    where the entry itself is there). Offsets are the stretch's own: offset ``i`` is position ``start + i`` of the
    sequence. It reads as a :class:`Contig` does (``read``, ``has``, ``sidecar``, ``region`` …).
    """

    def __init__(self, contigs: Iterable[Contig]):
        self.parts = sorted(contigs, key=lambda c: (c.start, -c.end))
        if not self.parts or len({c.parent for c in self.parts}) != 1:
            raise ValueError("a stretch is of contigs of one sequence")
        first = self.parts[0]
        self.start = first.start
        self.end = max(c.end for c in self.parts)
        self.length = self.end - self.start
        self.parent = first.parent
        self.parent_length = next((c.parent_length for c in self.parts if c.parent_length is not None), None)
        self.parent_aliases = list(dict.fromkeys(a for c in self.parts for a in c.parent_aliases))
        self.width = first.width
        self.name = "+".join(c.name for c in self.parts)
        #: (where it lies, as a window's key says)
        self.window = {"of": self.parent, "start": self.start}
        self.planes = sort_planes({p for c in self.parts for p in c.planes if self.has(p)})
        self._segments_of: dict[str, tuple[list, list]] = {}
        self._sidecar: list[SidecarEntry] | None = None

    def __repr__(self) -> str:
        return f"<Stretch {self.region()} of {', '.join(c.name for c in self.parts)}>"

    def parent_digests(self) -> dict:
        return next((d for d in (c.parent_digests() for c in self.parts) if d), {})

    def region(self, start: int = 0, end: int | None = None, *, commas: bool = True) -> str:
        end = self.length if end is None else end
        return region_text(self.parent, self.start + start, self.start + end, commas=commas)

    def has(self, plane: str) -> bool:
        if plane in _STRETCH_FILL:
            return any(c.has(plane) for c in self.parts)
        return all(c.has(plane) for c in self.parts)

    def _segments(self, plane: str) -> tuple[list, list]:
        """The stretch in pieces ``(a, b, contig)`` at its offsets, each read from the contig with the plane that holds
        it and reaches furthest (None: no contig with the plane holds it); and the pieces' starts, to look them up."""
        if plane not in self._segments_of:
            parts = sorted((c for c in self.parts if c.has(plane)), key=lambda c: c.start)
            bounds = sorted({self.start, self.end, *(c.start for c in parts), *(c.end for c in parts)})
            heap: list = []
            i = 0
            segs: list[tuple[int, int, Contig | None]] = []
            for a, b in zip(bounds, bounds[1:]):
                while i < len(parts) and parts[i].start <= a:
                    heapq.heappush(heap, (-parts[i].end, i, parts[i]))
                    i += 1
                while heap and -heap[0][0] <= a:
                    heapq.heappop(heap)
                owner = heap[0][2] if heap else None
                if segs and segs[-1][2] is owner:
                    segs[-1] = (segs[-1][0], b - self.start, owner)
                else:
                    segs.append((a - self.start, b - self.start, owner))
            self._segments_of[plane] = (segs, [s[0] for s in segs])
        return self._segments_of[plane]

    def read(self, plane: str, start: int = 0, end: int | None = None) -> np.ndarray:
        end = self.length if end is None else end
        if not (0 <= start <= end <= self.length):
            raise IndexError(f"{self.region()}: {start}-{end} outside 0-{self.length}")
        if not self.has(plane):
            raise MissingPlaneError(f"{self.name} has no {plane!r} plane in all its contigs")
        fill = _STRETCH_FILL.get(plane)
        segs, starts = self._segments(plane)
        pieces = []
        k = max(0, bisect.bisect_right(starts, start) - 1)
        pos = start
        while pos < end:
            a, b, c = segs[k]
            y = min(end, b)
            if c is None:
                pieces.append(np.full(y - pos, fill[0], dtype=fill[1]))
            else:
                pieces.append(c.read(plane, self.start + pos - c.start, self.start + y - c.start))
            pos = y
            k += 1
        if not pieces:
            if fill is not None:
                return np.zeros(0, dtype=fill[1])
            return next(c for c in self.parts if c.has(plane)).read(plane, 0, 0)
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    def sidecar(self) -> list[SidecarEntry]:
        if self._sidecar is None:
            seen: set[tuple] = set()
            out: list[SidecarEntry] = []
            for c in sorted((c for c in self.parts if c.has("gt")), key=lambda c: c.start):
                shift = c.start - self.start
                for e in c.sidecar():
                    at = e.offset + shift
                    key = (at, e.hap, e.kind, e.data)
                    if key in seen:
                        continue
                    if e.kind == "COMPLEX" and e.offset == 0 and shift > 0 and self._copied(out, at, e):
                        continue  # (a window's copy of the entry that begins the run, which another window holds)
                    seen.add(key)
                    out.append(SidecarEntry(at, e.hap, e.kind, e.data))
            out.sort(key=lambda e: (e.offset, e.hap, e.kind))
            self._sidecar = out
        return self._sidecar

    def _copied(self, entries: list[SidecarEntry], at: int, e: SidecarEntry) -> bool:
        """Whether a COMPLEX entry at a window's first base, stretch offset ``at``, is the window's copy (spec 3.1):
        the base before it is COMPLEX on that haplotype, in a run whose nearest entry before is the same."""
        gt = int(self.read("gt", at - 1, at)[0])
        if ((gt >> 4) if e.hap == 1 else (gt & 0x0F)) != C.HAP_COMPLEX:
            return False
        before = [x for x in entries if x.kind == "COMPLEX" and x.hap == e.hap and x.offset < at]
        return bool(before) and max(before, key=lambda x: x.offset).data == e.data


def join_contigs(contigs: Iterable[Contig], need: str | None = None) -> list[Contig | Stretch]:
    """Contigs of one sequence joined where they overlap or touch, in order along it: each join as the contig of it
    that holds all the others, or as a :class:`Stretch` of them. With ``need``, only the contigs that have that
    plane."""
    out: list[Contig | Stretch] = []
    group: list[Contig] = []
    hi = -1
    for c in sorted((c for c in contigs if need is None or c.has(need)), key=lambda c: (c.start, -c.end)):
        if group and c.start > hi:
            out.append(_joined(group))
            group = []
        group.append(c)
        hi = max(hi, c.end) if len(group) > 1 else c.end
    if group:
        out.append(_joined(group))
    return out


def _joined(group: list[Contig]) -> Contig | Stretch:
    lo, hi = min(c.start for c in group), max(c.end for c in group)
    whole = next((c for c in group if c.start == lo and c.end == hi), None)
    # (one contig that holds all the others, with every plane they have: itself)
    if whole is not None and all(set(c.planes) <= set(whole.planes) for c in group):
        return whole
    return Stretch(group)


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
        self._ref_contigs: dict[str, tuple[Contig, int]] = {}
        self._ref_cache: tuple[str, np.ndarray] | None = None  # (the FASTA record read last: its name and codes)
        self._ref_checked: set[str] = set()  # (the contigs checked against the FASTA's digests)
        self._ref_digests: tuple | None = None  # (the digests of the FASTA record read last: its name, codes, digests)
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
        # names, then other names, ignoring case, of whole contigs (a window is found by its name as written, so that
        # a name loosely matched is never a window's: spec 3.1); where two match, the first (as gpxil-read.js's)
        self._folded: dict[str, str] = {}
        whole = [e for e in meta["contigs"] if "window" not in e]
        for e in whole:
            self._folded.setdefault(e["name"].lower(), e["name"])
        for e in whole:
            for a in e.get("aliases", []):
                self._folded.setdefault(a.lower(), e["name"])

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
        self._ref_checked = set()
        self._ref_digests = None

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

    @property
    def windows(self) -> list[Contig]:
        """The file's windows (spec 3.1), in file order."""
        return [c for c in self._contigs.values() if c.window is not None]

    def find(self, name: str) -> Contig | None:
        """A contig by name or other name: as written, else a whole contig ignoring case, with or without 'chr' (as
        gpxil-read.js's ``contig()``); None if there is none. ``g[name]`` takes names only as written."""
        if name in self:
            return self[name]
        n = str(name).lower()
        for k in (n, "chr" + n, n[3:] if n.startswith("chr") else n):
            if k in self._folded:
                return self._contigs[self._folded[k]]
        return None

    def windows_of(self, name: str) -> list[Contig]:
        """The windows of the sequence ``name`` (their parent's name, or another name of it; ignoring case, with or
        without 'chr', as gpxil-read.js's ``windowsOf()``), in file order."""
        n = str(name).lower()
        want = {n, "chr" + n, n[3:] if n.startswith("chr") else n}
        return [c for c in self.windows if any(x.lower() in want for x in (c.parent, *c.parent_aliases))]

    def sequence_contigs(self, name: str) -> list[Contig]:
        """The contigs that hold bases of the sequence ``name``: the sequence itself, if the file holds it whole, and
        its windows (spec 3.1), in file order."""
        whole = self.find(name)
        if whole is not None and whole.window is not None:
            whole = None  # (a window's own name: not the name of a sequence)
        wins = self.windows_of(whole.name if whole is not None else name)
        return ([whole] if whole is not None else []) + [w for w in wins if w is not whole]

    def stretches(self, name: str | None = None, need: str | None = None) -> list[Contig | Stretch]:
        """The contigs of a sequence (by default, of every sequence, each in the order the file first holds it),
        joined where they overlap or touch: along each sequence in order, each join as the contig of it that holds
        all the others, or as a :class:`Stretch` of them. With ``need``, only the contigs that have that plane."""
        if name is not None:
            return join_contigs(self.sequence_contigs(name), need)
        seqs: dict[str, list[Contig]] = {}
        for c in self:
            seqs.setdefault(c.parent, []).append(c)
        return [s for cs in seqs.values() for s in join_contigs(cs, need)]

    def stretch_of(self, contig: Contig | Stretch, need: str | None = None) -> Contig | Stretch:
        """The join of the file's contigs of a contig's sequence that holds it (see :meth:`stretches`); the contig
        itself if it has no ``need`` plane, or is a stretch already."""
        if isinstance(contig, Stretch):
            return contig
        for s in join_contigs([c for c in self if c.parent == contig.parent], need):
            if s.start <= contig.start and contig.end <= s.end:
                return s
        return contig

    def region_of(self, text: str) -> tuple[str, int | None, int | None]:
        """A region given as text, as gpxil-read.js's ``locate(text)`` reads it: the whole of a contig named so (as
        :meth:`find` finds it), else :func:`parse_region` (``chr17:7,661,779-7,687,546``, ``chr17:7,676,154``)."""
        c = self.find(str(text).strip(" \t\r\n"))
        return (c.name, None, None) if c is not None else parse_region(text)

    def locate(self, name: str, start: int | None = None, end: int | None = None, *, join: bool = False) -> tuple:
        """Where a region is: the contig that holds it, and the offsets ``[s, e)`` of the region there.

        ``name`` names a contig, whole or a window (as :meth:`find` finds it), or a sequence that windows of the file
        hold (spec 3.1). ``start`` and ``end`` (0-based, the end excluded) are positions in the sequence, so on a
        window they are positions in its parent, and a region of the parent is found in a window that holds all of
        it. Without them the region is the whole contig: for a sequence that only windows hold, its window, if it
        has one only; and a ``name`` that no contig has is read as a region, ``chr17:7,661,779-7,687,546``, as
        gpxil-read.js's ``locate(text)`` reads it. With ``join``, a region that no window holds all of, but windows
        that overlap or touch hold together, is found in their :class:`Stretch`.
        """
        if start is None and end is None and self.find(name) is None:
            n, s, e = parse_region(name)  # (a region given as text: chr17:7,661,779-7,687,546)
            if s is not None:
                return self.locate(n, s, e, join=join)
        whole = start is None and end is None
        c = self.find(name)
        if c is not None:
            if whole:
                return c, 0, c.length
            s = c.start if start is None else start
            e = c.end if end is None else end
            if not (c.start <= s < e <= c.end):
                if c.window is None:
                    raise IndexError(f"{region_text(name, s, e, commas=True)} is outside {c.name} (length {c.length:,})")
                raise IndexError(
                    f"{region_text(c.parent, s, e, commas=True)} is outside the window {c.name}, which holds {c.region()}"
                    " (a window's positions are its parent's)"
                )
            return c, s - c.start, e - c.start
        wins = self.windows_of(name)
        if wins:
            if whole:
                if len(wins) == 1:
                    return wins[0], 0, wins[0].length
                raise KeyError(f"{path_name(self.path)} holds {len(wins)} windows of {name} ({_listed(wins)}): give one, or a region")
            s = start or 0
            # (a window of just that stretch, or else the first that holds it all)
            exact = [w for w in wins if (w.start, w.end) == (s, end)]
            for w in exact or wins:
                if w.start <= s < w.end:
                    e = w.end if end is None else end
                    if s < e <= w.end:
                        return w, s - w.start, e - w.start
            e = s + 1 if end is None else end
            if join and s < e:
                for v in self.stretches(name):
                    if v.start <= s and e <= v.end:
                        return v, s - v.start, e - v.start
            raise IndexError(
                f"{region_text(name, s, e, commas=True)} isn't all in a window of {path_name(self.path)}: it holds {_listed(wins)}"
            )
        have = ", ".join(self.contigs[:8]) + (", …" if len(self) > 8 else "")
        raise KeyError(f"no contig {name!r} in {self.path} (it has {have})")


def open_gpxil(path, reference=None) -> GpxilFile:
    return GpxilFile(path, reference=reference)
