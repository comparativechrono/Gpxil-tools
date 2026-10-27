"""Reading gpxil files on web servers, with HTTP range requests.

A gpxil file keeps each contig's planes in strips of whole rows, so a region
needs only the strips it touches. :class:`RangeFile` is a read-only file
object over a URL that fetches those bytes, a block at a time, and keeps the
recent blocks, so that

    >>> g = GpxilFile("https://comparativechrono.github.io/Gpxil-browser/data/grch38/gpxil/chr17.gpxil.tif")
    >>> g["chr17"].sequence(7_668_401, 7_668_461)

downloads a few hundred kilobytes, not the 23 MB file. The server must
answer range requests (GitHub Pages, S3, and most web servers do); a server
that does not is refused unless the file is small.
"""

from __future__ import annotations

import io
import re
import urllib.request
from collections import OrderedDict

from . import __version__

BLOCK = 1 << 18  #: bytes fetched at a time (a gpxil strip is at most this before compression)
CACHE_BLOCKS = 64  #: blocks kept (16 MB)
WHOLE_FILE_LIMIT = 32 << 20  #: a server without range requests is accepted for files up to this size

_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)")


def is_url(path) -> bool:
    return isinstance(path, str) and path.lower().startswith(("http://", "https://"))


class RangeFile(io.RawIOBase):
    """A read-only, seekable file object over a URL, read with HTTP range requests."""

    def __init__(self, url: str, *, block: int = BLOCK, cache_blocks: int = CACHE_BLOCKS, timeout: float = 60, headers: dict | None = None):
        super().__init__()
        self.url = url
        self.name = url
        self.block = int(block)
        self.cache_blocks = int(cache_blocks)
        self.timeout = timeout
        self.headers = {"User-Agent": f"gpxiltools/{__version__}", **(headers or {})}
        self.requests = 0  #: HTTP requests made
        self.fetched = 0  #: bytes downloaded
        self._pos = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._whole: bytes | None = None
        self.size = self._probe()

    # -- HTTP ----------------------------------------------------------------
    def _get(self, start: int, end: int) -> tuple[int, bytes, str | None]:
        """GET bytes [start, end] (inclusive): (status, body, Content-Range)."""
        req = urllib.request.Request(self.url, headers={**self.headers, "Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = r.read()
            status = r.status
            content_range = r.headers.get("Content-Range")
        self.requests += 1
        self.fetched += len(body)
        return status, body, content_range

    def _head_size(self) -> int | None:
        """The file's size from a HEAD request's Content-Length, or None."""
        req = urllib.request.Request(self.url, headers=self.headers, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                n = r.headers.get("Content-Length")
        except OSError:
            return None
        finally:
            self.requests += 1
        return int(n) if n and n.strip().isdigit() else None

    def _probe(self) -> int:
        status, body, content_range = self._get(0, self.block - 1)
        if status == 206:
            m = _CONTENT_RANGE.match(content_range or "")
            if m and m.group(3) != "*":
                size = int(m.group(3))
            else:
                # a range answered without the file's size, as a browser shows a range from another web site
                # (Content-Range is hidden unless the server exposes it): the size from a HEAD request instead
                size = self._head_size()
                if size is None:
                    raise OSError(f"{self.url}: the server answered a range request but didn't say how big the file is")
            self._cache[0] = body
            return size
        if status == 200:  # the server sent the whole file: fine if it is small
            if len(body) > WHOLE_FILE_LIMIT:
                raise OSError(f"{self.url}: the server does not answer range requests, and the file is too big to download whole")
            self._whole = body
            return len(body)
        raise OSError(f"{self.url}: unexpected answer to a range request (HTTP {status}, Content-Range {content_range!r})")

    def _blocks(self, first: int, last: int) -> None:
        """Make sure blocks first..last are cached, fetching each missing run in one request."""
        keep = max(self.cache_blocks, last - first + 1)  # a long read keeps all its blocks until it is copied out
        i = first
        while i <= last:
            if i in self._cache:
                self._cache.move_to_end(i)
                i += 1
                continue
            j = i
            while j + 1 <= last and (j + 1) not in self._cache:
                j += 1
            start, end = i * self.block, min(self.size, (j + 1) * self.block) - 1
            status, body, _ = self._get(start, end)
            if status != 206 or len(body) != end - start + 1:
                raise OSError(f"{self.url}: asked for bytes {start}-{end}, got HTTP {status} with {len(body)} bytes")
            for k in range(i, j + 1):
                a = (k - i) * self.block
                self._cache[k] = body[a : a + self.block]
            while len(self._cache) > keep:
                self._cache.popitem(last=False)
            i = j + 1

    # -- file object ---------------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self.size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def readinto(self, b) -> int:
        n = min(len(b), max(0, self.size - self._pos))
        if n == 0:
            return 0
        if self._whole is not None:
            b[:n] = self._whole[self._pos : self._pos + n]
        else:
            first, last = self._pos // self.block, (self._pos + n - 1) // self.block
            self._blocks(first, last)
            out = memoryview(b)
            done = 0
            for k in range(first, last + 1):
                blk = self._cache[k]
                a = self._pos + done - k * self.block
                take = min(len(blk) - a, n - done)
                out[done : done + take] = blk[a : a + take]
                done += take
        self._pos += n
        return n

    def readall(self) -> bytes:
        return self.read(self.size - self._pos)

    def __repr__(self) -> str:
        return f"<RangeFile {self.url} size={self.size} requests={self.requests} fetched={self.fetched}>"
