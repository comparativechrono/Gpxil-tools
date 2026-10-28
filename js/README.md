# gpxil-read.js: gpxil files in web pages

`gpxil-read.js` is gpxiltools' reader for JavaScript. It reads a gpxil file's metadata, and any interval of any
plane, reading only the strips (or tiles) that hold it: from a Blob (a file the page was given), from bytes, or from
the web with HTTP range requests, which it makes as few as it can (strips next to each other in one request) and
caches (decoded strips, shared by every file). It reads every version 0.x of the
[specification](https://github.com/comparativechrono/gpxil) (it implements 0.3): pages in strips or tiles,
uncompressed or Deflate, with or without a predictor (horizontal differencing, or the floating-point predictor),
classic TIFF or BigTIFF, in either byte order, with planes of 8, 16 or 32 bits, and the sidecar. It knows windows
(section 3.1), contigs that hold a stretch of a longer sequence: each says where it lies in its sequence, and a region
of the sequence is found in a window that holds it. It needs nothing but what browsers have, and runs in Node 18 or
later.

Files are untrusted input (specification section 13), so it refuses what would make it loop or take all the memory:
directories that loop back, more than 2^20 pages, a tag of more than 64 MiB, a strip or tile of more than 256 MiB
decoded (a whole human chromosome 1 in one strip of 8-bit values still fits), a strip stored in more bytes than it
could need, a sidecar of more than 256 MiB; it never inflates a strip past the size its rows need, nor reads one that
holds less. Directories one after another are walked from the bytes already read. A TIFF that isn't gpxil is told by
its metadata (`not-gpxil`) before its pages are looked at.

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

  // a window: TP53 and 100 kb either side, out of HG002's genome (gpxil subset ... chr17:7,561,779-7,787,546)
  const w = await GpxilRead.open(GpxilRead.fromUrl("HG002_TP53.gpxil.tif"));
  const c = w.contigs[0];                                  // {name: "chr17:7561779-7787546", window: {of: "chr17", start: 7561778, ...},
                                                           //  parent: "chr17", start: 7561778, end: 7787546, ...}
  const at = w.locate("chr17:7,661,779-7,687,546");        // {contig: c, start: 100000, end: 125768}: the window's offsets
  const tp53 = await w.read(at.contig, "gt", at.start, at.end);
  GpxilRead.regionText(c, at.start, at.end);               // "chr17:7,661,779-7,687,546"
</script>
```

| | |
|---|---|
| `open(source)` | a gpxil file: `meta` (its JSON metadata), `version`, `contigs`, `contig(name)` (by name or alias; a whole contig also ignoring case and `chr`, a window only as written), `read(contig, plane, start, end)` (the contig's own offsets), `sidecar(contig)`, `page(contig, plane)`, `file` (its `TiffFile`) |
| a contig | `{name, length, width, pages, aliases, entry, window, parent, start, end, parentLength, parentAliases}`: `window` is its `window` key (`{of, start, length, md5, refget, aliases}`), or null for a whole contig; `parent`, `start` and `end` say where it lies in its sequence (a whole contig's is itself, from 0), so offset `i` is position `start + i` there |
| `g.locate(name, start, end)`, `g.locate(text)` | where a region is: `{contig, start, end}` with the contig's offsets. Positions are the sequence's: a window's parent's, so `locate("chr17", 7661778, 7687546)` finds the window that holds them (the one of just that stretch, or else the first that holds it all). Names are taken as `contig()` takes them. A text, `"chr17:7,661,779-7,687,546"`, is a region unless it is a contig's name (spaces around it aside); without positions (`undefined` or `null`), the whole contig, or a sequence's only window. gpxiltools' `GpxilFile.locate` answers the same |
| `g.windows`, `g.windowsOf(name)` | the file's windows; those of a sequence |
| `parseRegion(text)`, `regionText(contig, start, end)` | `"chr17:7,661,779-7,687,546"` to `{name, start, end}` (0-based, the end excluded), `"chr17:7,676,154"` to one base, `"chr17"` to `{name}`, and an interval that isn't one (from 0, or ending before it starts) to null. The name ends at the last colon, so a window's name is one; numbers are the digits 0 to 9, with commas or spaces, and don't begin with 0, so a name with colons in it (`HLA-A*01:01:01:01`) is a name. gpxiltools' `parse_region` reads them the same way. `regionText` gives offsets of a contig as a region of its sequence |
| `fromUrl(url, {size, key, fetchRange})`, `fromBlob(blob, key)`, `fromBytes(bytes, key)` | sources. A source is `{key, size, read(start, end, what)}`; `what` says what is read (`"header"`, `"directory"`, `"tags"` or `"strips"`), so a page can show what it is loading through its own `fetchRange`. Decoded strips are cached by the key (the URL, by default), so a key must change when its file does: GpxilStudio keys a file by its path and the time it was last written |
| `new TiffFile(source)` | the TIFF below: `page(k)`, `pageAt(offset)` (a page by the offset of its directory), `readPage(page, start, end)` |
| `setCacheBytes(n)`, `cacheStats()` | how much decoded data to keep (192 MB by default) |
| `base(code)`, `state(nibble)`, `flags(byte)`, `IUPAC`, `STATE`, `FLAGS` | what the codes mean (specification section 5) |
| `parseSidecar(text)`, `inflate(bytes, limit)`, `TAG`, `SPEC` | the rest |

Errors carry a `code`, so that a page can say what went wrong in its own words: `http`, `not-tiff`, `not-gpxil`,
`version` (another major version), `format` (the file contradicts itself or TIFF, or a window's metadata says
wrong things of where it is), `limit` (it claims more than the reader takes, above), `layout`, `compression` (with
`compression`, its name), `predictor`, `contig`, `plane`, `region` (a region the file doesn't hold, or a sequence
with several windows asked for whole).

## Tests

`node --test js/test/read.test.mjs` writes files with gpxiltools (with this repository's `src`, so it needs Python
with gpxiltools' requirements) and reads every plane of every one of them, whole and in windows, from bytes, a Blob
and a local web server, comparing what it reads with what gpxiltools reads: Deflate and uncompressed pages, many
strips of a few rows, tiles, BigTIFF, big-endian TIFF, planes of 8, 16 and 32 bits, signed and float, with and
without predictors, sidecars with insertions and a COMPLEX allele, the specification's worked example, a file of a
later 0.x, and windows with and without their reference plane, where it finds regions just where gpxiltools finds
them. It checks that files it can't read are refused with the right code, at once, without the whole file being
fetched: another major version, LZW, not gpxil, not TIFF, windows that say wrong things of where they are, and files
made to hurt a reader (directories that loop, a strip that inflates to 200 MB, strips shorter than their rows, a
strip of no bytes, a tag or strips too large). It checks that strips are read in as few requests as they can be, that contigs whose names differ only in
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
