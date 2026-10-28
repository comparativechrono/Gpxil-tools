# gpxil-read.js: gpxil files in web pages

`gpxil-read.js` is gpxiltools' reader for JavaScript. It reads a gpxil file's metadata, and any interval of any
plane, reading only the strips (or tiles) that hold it: from a Blob (a file the page was given), from bytes, or from
the web with HTTP range requests, which it makes as few as it can (strips next to each other in one request) and
caches (decoded strips, shared by every file). It reads every version 0.x of the
[specification](https://github.com/comparativechrono/gpxil) (it implements 0.2): pages in strips or tiles,
uncompressed or Deflate, with or without a predictor (horizontal differencing, or the floating-point predictor),
classic TIFF or BigTIFF, in either byte order, with planes of 8, 16 or 32 bits, and the sidecar. It needs nothing but
what browsers have, and runs in Node 18 or later.

Files are untrusted input (specification section 13), so it refuses what would make it loop or take all the memory:
directories that loop back, more than 2^20 pages, a tag of more than 64 MiB, a strip or tile of more than 256 MiB
decoded (a whole human chromosome 1 in one strip of 8-bit values still fits), a sidecar of more than 256 MiB; and it
never inflates a strip past the size its rows need, nor reads one that holds less.

The [Gpxil browser](https://github.com/comparativechrono/Gpxil-browser) and
[GpxilStudio](https://github.com/comparativechrono/gpxilstudio) each keep a copy. It is not published on npm.

```html
<script src="gpxil-read.js"></script>   <!-- or, in a module: import "./gpxil-read.js"; in Node: require("./gpxil-read.js") -->
<script type="module">
  const g = await GpxilRead.open(GpxilRead.fromUrl("HG002_chr17.gpxil.tif"));
  g.version;                                               // "0.2"
  g.contigs;                                               // [{name, length, width, pages, aliases, entry}]
  const gt = await g.read("chr17", "gt", 7661778, 7687546); // Uint8Array; 0-based, the end excluded
  const ps = await g.read("chr17", "ps", 7661778, 7687546); // Uint32Array
  const side = await g.sidecar("chr17");                   // [{offset, hap, kind, data}]
  GpxilRead.state(gt[0] >> 4);                             // "REF": what the codes mean
</script>
```

| | |
|---|---|
| `open(source)` | a gpxil file: `meta` (its JSON metadata), `version`, `contigs`, `contig(name)` (by name or alias, ignoring case and `chr`), `read(contig, plane, start, end)`, `sidecar(contig)`, `page(contig, plane)`, `file` (its `TiffFile`) |
| `fromUrl(url, {size, key, fetchRange})`, `fromBlob(blob, key)`, `fromBytes(bytes, key)` | sources. A source is `{key, size, read(start, end, what)}`; `what` says what is read (`"header"`, `"directory"`, `"tags"` or `"strips"`), so a page can show what it is loading through its own `fetchRange`. Decoded strips are cached by the key (the URL, by default), so a key must change when its file does: GpxilStudio keys a file by its path and the time it was last written |
| `new TiffFile(source)` | the TIFF below: `page(k)`, `pageAt(offset)` (a page by the offset of its directory), `readPage(page, start, end)` |
| `setCacheBytes(n)`, `cacheStats()` | how much decoded data to keep (192 MB by default) |
| `base(code)`, `state(nibble)`, `flags(byte)`, `IUPAC`, `STATE`, `FLAGS` | what the codes mean (specification section 5) |
| `parseSidecar(text)`, `inflate(bytes, limit)`, `TAG`, `SPEC` | the rest |

Errors carry a `code`, so that a page can say what went wrong in its own words: `http`, `not-tiff`, `not-gpxil`,
`version` (another major version), `format` (the file contradicts itself or TIFF), `limit` (it claims more than the
reader takes, above), `layout`, `compression` (with `compression`, its name), `predictor`, `contig`, `plane`.

## Tests

`node --test js/test/read.test.mjs` writes files with gpxiltools (with this repository's `src`, so it needs Python
with gpxiltools' requirements) and reads every plane of every one of them, whole and in windows, from bytes, a Blob
and a local web server, comparing what it reads with what gpxiltools reads: Deflate and uncompressed pages, many
strips of a few rows, tiles, BigTIFF, big-endian TIFF, planes of 8, 16 and 32 bits, signed and float, with and
without predictors, sidecars with insertions and a COMPLEX allele, the specification's worked example and a file of a
later 0.x. It checks that files it can't read are refused with the right code, at once, without the whole file
being fetched: another major version, LZW, not gpxil, not TIFF, and files made to hurt a reader (directories that
loop, a strip that inflates to 200 MB, strips shorter than their rows, a strip of no bytes, a tag or strips too
large). It checks that strips are read in as few requests as they can be, that contigs whose names differ only in
case are each found, and that the file loads as a classic script, as an ES module and with `require()`. CI runs it.

## The copies

A site keeps `gpxil-read.js` as it is here at a commit of main, unchanged, with `gpxil-read.source.json` beside it
naming the commit and the file's SHA-256. The site's tests check the checksum, and its CI that the file at that
commit is the same. To update a site, on main here (pushed):

```sh
python3 js/copy_to.py ../gpxil-browser/site/js/vendor
python3 js/copy_to.py ../gpxilstudio/assets/vendor/gpxil-read
```

Change the reader here, never in a copy.
