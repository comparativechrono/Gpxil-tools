/* gpxil-read.js: a reader of gpxil files for web pages, and for Node.
 *
 * Part of gpxiltools (https://github.com/comparativechrono/Gpxil-tools), under the MIT licence. The Gpxil browser
 * and GpxilStudio each keep a copy of this file as it is here at a recorded commit; it is not published on npm.
 * js/README.md says how to update the copies.
 *
 * A gpxil file (https://github.com/comparativechrono/gpxil) is a TIFF with one page per plane of each contig,
 * stored in strips of whole rows or in tiles. This reads its metadata, then any interval of a plane by reading only
 * the strips or tiles that hold it: from a Blob, from bytes, or from the web with HTTP range requests. Data are
 * inflated with the DecompressionStream that browsers and Node 18 or later have. It reads files of any version 0.x
 * of the specification (it implements 0.3), uncompressed or Deflate, with or without a predictor, classic TIFF or
 * BigTIFF, in either byte order. A window (section 3.1), a contig that holds a stretch of a longer sequence, says
 * where it lies in it, and regions are found in it by the sequence's positions.
 *
 * Files are untrusted input (specification section 13): a file that claims more than this reader decodes at once
 * (a strip or tile of more than 256 MiB, a tag of more than 64 MiB, a sidecar of more than 256 MiB, more than 2^20
 * pages), that loops, or whose sizes disagree, is refused with an error, and no strip inflates past its size.
 *
 * Load it as a classic script (<script src="gpxil-read.js">), import it for its side effect (import
 * "./gpxil-read.js"), or require() it in Node: it defines globalThis.GpxilRead, and module.exports in Node.
 *
 *   const g = await GpxilRead.open(GpxilRead.fromUrl("HG002_chr17.gpxil.tif"));
 *   g.version;                                                  // "0.2"
 *   const gt = await g.read("chr17", "gt", 7661778, 7687546);   // Uint8Array; 0-based, the end excluded
 *   const side = await g.sidecar("chr17");                      // [{offset, hap, kind, data}]
 *
 *   // a window: the contig chr17:7561779-7787546 holds bases 7,561,779 to 7,787,546 of chr17
 *   const w = await GpxilRead.open(GpxilRead.fromUrl("TP53.gpxil.tif"));
 *   const { contig, start, end } = w.locate("chr17:7,661,779-7,687,546"); // the window, at its own offsets
 *   contig.parent, contig.start;                                // "chr17", 7561778: offset i is chr17's start + i
 *   const tp53 = await w.read(contig, "gt", start, end);
 *
 * A source is {key, size, read(start, end, what) -> Promise<Uint8Array>}, the end excluded; `what` says what is
 * read ("header", "directory", "tags" or "strips"), for a page that shows what it is loading. Decoded strips are
 * cached by the source's key, so a key must change when its file does. Lower down, new GpxilRead.TiffFile(source)
 * reads pages by index, or by the offset of their directory, and readPage(page, start, end) the values at 1-D
 * positions [start, end) of a page.
 */
(function (root) {
  "use strict";

  const SPEC = "0.3"; // the version of the gpxil specification this reader implements
  const MAJOR = "0"; // it reads any version 0.x (specification section 14)

  const TAG = {
    ImageWidth: 256,
    ImageLength: 257,
    BitsPerSample: 258,
    Compression: 259,
    ImageDescription: 270,
    StripOffsets: 273,
    SamplesPerPixel: 277,
    RowsPerStrip: 278,
    StripByteCounts: 279,
    PageName: 285,
    Predictor: 317,
    TileWidth: 322,
    TileLength: 323,
    TileOffsets: 324,
    TileByteCounts: 325,
    SampleFormat: 339,
    Sidecar: 65400, // GPXIL_SIDECAR: the zlib-compressed sidecar, on a contig's gt page (section 8.3)
  };
  const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8 };
  const COMPRESSION = { 1: "none", 5: "LZW", 7: "JPEG", 8: "Deflate", 32773: "PackBits", 32946: "Deflate", 34925: "LZMA", 50000: "Zstandard" };
  const PROBE = 16384; // bytes read at a directory: its tags and, as tifffile lays files out, their arrays
  const GROUP = 8 << 20; // strips next to each other in the file are read together, up to this many bytes
  // what a file may claim before it is refused (section 13: inputs are untrusted)
  const MAX_TAGS = 4096; // tags in a directory
  const MAX_TAG_BYTES = 64 << 20; // bytes of one tag's value (the sidecar's, compressed, included)
  const MAX_SEGMENT = 256 << 20; // bytes of one strip or tile, decoded (a whole human chromosome 1 in one strip fits)
  const MAX_SEGMENTS = 1 << 24; // strips or tiles in a page
  const MAX_SIDECAR = 256 << 20; // bytes of a sidecar, inflated
  const MAX_PAGES = 1 << 20; // pages in a file

  /** an Error with a code, so that a page can say it in its own words: http, not-tiff, not-gpxil, version, format,
      limit, layout, compression, predictor, contig, plane, region */
  function fail(code, message) {
    const e = new Error(message);
    e.code = code;
    return e;
  }
  const MiB = (n) => `${Math.round(n / 1048576)} MiB`;

  /* ---------------- sources ---------------- */
  let unnamed = 0;
  /** bytes held in memory (a Uint8Array, an ArrayBuffer or a Node Buffer) */
  function fromBytes(bytes, key) {
    const u = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    return { key: key || "bytes:" + ++unnamed, size: u.length, read: async (a, b) => u.subarray(a, Math.min(b, u.length)) };
  }
  /** a Blob or File, read a slice at a time */
  function fromBlob(blob, key) {
    return { key: key || "blob:" + ++unnamed, size: blob.size, read: async (a, b) => new Uint8Array(await blob.slice(a, b).arrayBuffer()) };
  }
  /** a file on the web, read with HTTP range requests. opts: size (if known), key (for the cache; the URL by
      default), fetchRange(url, start, end, what) -> Promise<Uint8Array> to fetch some other way (to show progress,
      say). A server that ignores ranges sends the whole file: it is then read once, and kept. */
  function fromUrl(url, opts) {
    const o = opts || {};
    let whole = null;
    const get =
      o.fetchRange ||
      (async (u, a, b) => {
        if (whole) return (await whole).slice(a, b);
        const r = await fetch(u, { headers: { Range: `bytes=${a}-${b - 1}` } });
        if (r.status === 206) return new Uint8Array(await r.arrayBuffer());
        if (r.status === 200) {
          if (typeof console !== "undefined") console.warn(`${u}: the server ignores range requests, so the whole file is read`);
          whole = r.arrayBuffer().then((x) => new Uint8Array(x));
          return (await whole).slice(a, b);
        }
        throw fail("http", `${u}: HTTP ${r.status}`);
      });
    // (an empty range is never asked for: "bytes=18-17" is invalid, and a server answers it with the whole file)
    return { key: o.key || url, size: o.size || 0, read: async (a, b, what) => (b > a ? get(url, a, b, what) : new Uint8Array(0)) };
  }

  /** zlib-wrapped Deflate (TIFF compression 8, and the sidecar). With a limit, an error (code "limit") as soon as
      more than that many bytes come out, so that a small file can't inflate to fill the memory. Bytes after the end
      of the stream (padding) are ignored, as zlib ignores them; so is anything after `enough` bytes have come out */
  async function inflate(bytes, limit, enough) {
    const max = limit === undefined ? Infinity : limit;
    const reader = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate")).getReader();
    const parts = [];
    let n = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        n += value.length;
        if (n > max) throw fail("limit", `compressed data that inflate to more than ${max} bytes`);
        parts.push(value);
      }
    } catch (e) {
      reader.cancel().catch(() => {});
      if (e && e.code === "limit") throw e;
      const junk = e && (e.code === "ERR_TRAILING_JUNK_AFTER_STREAM_END" || /junk/i.test(String(e.message)));
      if (!(junk || (enough !== undefined && n >= enough))) throw fail("format", `compressed data that don't inflate (${(e && e.message) || e})`);
    }
    if (parts.length === 1) return parts[0];
    const out = new Uint8Array(n);
    let o = 0;
    for (const p of parts) {
      out.set(p, o);
      o += p.length;
    }
    return out;
  }

  /* ---------------- decoded strips, shared by every file: the least recently used go first ---------------- */
  const cache = new Map();
  let cached = 0;
  let budget = 192 * 1024 * 1024;
  const pending = new Map();
  function recall(key) {
    const v = cache.get(key);
    if (v !== undefined) {
      cache.delete(key);
      cache.set(key, v);
    }
    return v;
  }
  function evict(keep) {
    while (cached > budget && cache.size > keep) {
      const [k, v] = cache.entries().next().value;
      cache.delete(k);
      cached -= v.byteLength;
    }
  }
  function remember(key, data) {
    if (cache.has(key)) cached -= cache.get(key).byteLength;
    cache.delete(key);
    cache.set(key, data);
    cached += data.byteLength;
    evict(1); // (the strip just read stays, whatever its size)
  }
  /** how many bytes of decoded strips to keep (192 MB by default) */
  function setCacheBytes(n) {
    budget = Math.max(0, Number(n) || 0);
    evict(0);
  }
  const cacheStats = () => ({ strips: cache.size, bytes: cached, budget });

  /* ---------------- TIFF ---------------- */
  class View {
    constructor(bytes, base, little) {
      this.v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      this.base = base; // the file offset of bytes[0]
      this.little = little;
    }
    has(off, n) {
      return off >= this.base && off + n <= this.base + this.v.byteLength;
    }
    u8(off) {
      return this.v.getUint8(off - this.base);
    }
    u16(off) {
      return this.v.getUint16(off - this.base, this.little);
    }
    u32(off) {
      return this.v.getUint32(off - this.base, this.little);
    }
    u64(off) {
      const v = this.v.getBigUint64(off - this.base, this.little);
      if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw fail("format", "an offset too large to read");
      return Number(v);
    }
    /** a value of a TIFF type that holds integers; NaN for the others */
    num(type, off) {
      if (type === 1 || type === 2 || type === 6 || type === 7) return this.u8(off);
      if (type === 3 || type === 8) return this.u16(off);
      if (type === 4 || type === 9 || type === 13) return this.u32(off);
      if (type === 16 || type === 17 || type === 18) return this.u64(off);
      return NaN;
    }
  }
  const count = (n) => Number.isSafeInteger(n) && n >= 0;

  class TiffFile {
    constructor(source) {
      this.source = source;
      this.h = null;
      this.offsets = []; // directory offsets, page by page, as far as walked
      this.seen = new Set(); // the same, to find a chain of directories that loops back
      this.byOffset = new Map(); // directory offset -> page promise
      this.byIndex = new Map(); // page index -> page promise
      this._walking = Promise.resolve();
    }
    /** byte order, classic or BigTIFF, and the first directory */
    async header() {
      if (this.h) return this.h;
      if (!this._h)
        this._h = (async () => {
          const b = await this.source.read(0, 16, "header");
          if (b.length < 8) throw fail("not-tiff", "not a TIFF file: it is too short");
          const little = b[0] === 0x49;
          if (!(little || b[0] === 0x4d) || b[0] !== b[1]) throw fail("not-tiff", "not a TIFF file");
          const r = new View(b, 0, little);
          const magic = r.u16(2);
          let h;
          if (magic === 42) h = { little, big: false, first: r.u32(4) };
          else if (magic === 43 && b.length >= 16) h = { little, big: true, first: r.u64(8) };
          else throw fail("not-tiff", `not a TIFF file (magic number ${magic})`);
          this.offsets = [h.first];
          this.seen = new Set([h.first]);
          this.h = h;
          return h;
        })();
      return this._h;
    }
    async _dir(at) {
      const h = await this.header();
      const size = this.source.size;
      if (!count(at) || at < (h.big ? 16 : 8) || (size && at >= size)) throw fail("format", `a directory at ${at}, outside the file`);
      // (the bytes read for the last directory often hold this one: directories one after another are walked from them)
      let r = this._last && this._last.has(at, h.big ? 8 : 2) ? this._last : null;
      if (!r) r = new View(await this.source.read(at, size ? Math.min(size, at + PROBE) : at + PROBE, "directory"), at, h.little);
      if (!r.has(at, h.big ? 8 : 2)) throw fail("format", `the file ends before the directory at ${at}`);
      const n = h.big ? r.u64(at) : r.u16(at);
      if (n > MAX_TAGS) throw fail("format", `a directory of ${n} tags: not a gpxil page`);
      const entry = h.big ? 20 : 12;
      const first = at + (h.big ? 8 : 2);
      const end = first + n * entry + (h.big ? 8 : 4);
      if (!r.has(first, end - first)) {
        r = new View(await this.source.read(at, Math.max(end, size ? Math.min(size, at + PROBE) : at + PROBE), "directory"), at, h.little);
        if (!r.has(first, end - first)) throw fail("format", `the file ends inside the directory at ${at}`);
      }
      this._last = r;
      const tags = new Map();
      for (let i = 0; i < n; i++) {
        const e = first + i * entry;
        const tag = r.u16(e);
        const type = r.u16(e + 2);
        const k = h.big ? r.u64(e + 4) : r.u32(e + 4);
        const bytesOf = (TYPE_SIZE[type] || 1) * k;
        const inline = bytesOf <= (h.big ? 8 : 4);
        if (!tags.has(tag)) tags.set(tag, { type, count: k, at: inline ? e + (h.big ? 12 : 8) : h.big ? r.u64(e + 12) : r.u32(e + 8), bytes: bytesOf });
      }
      return { r, tags, next: h.big ? r.u64(first + n * entry) : r.u32(first + n * entry) };
    }
    /** the directory offset of page k, walking the chain of directories as far as it. One walk at a time: two at
        once would both add the directory after the last one known, and every later page would be off by one */
    ifd(k) {
      if (!count(k) || k >= MAX_PAGES) return Promise.reject(fail("format", `the file has no page ${k}`));
      const walk = this._walking.then(async () => {
        await this.header();
        while (this.offsets.length <= k) {
          const at = this.offsets[this.offsets.length - 1];
          if (!at) throw fail("format", `the file has no page ${k}`);
          const next = (await this._dir(at)).next;
          if (next) {
            if (this.seen.has(next))
              throw fail("format", `the file's directories loop back (page ${this.offsets.length} is page ${this.offsets.indexOf(next)} again)`);
            this.seen.add(next);
          }
          this.offsets.push(next);
        }
        if (!this.offsets[k]) throw fail("format", `the file has no page ${k}`);
        return this.offsets[k];
      });
      this._walking = walk.catch(() => {});
      return walk;
    }
    /** page k (0 is the first) */
    page(k) {
      if (!this.byIndex.has(k)) {
        const p = this.ifd(k).then((at) => this.pageAt(at));
        p.catch(() => this.byIndex.delete(k));
        this.byIndex.set(k, p);
      }
      return this.byIndex.get(k);
    }
    /** the page whose directory is at a file offset (as the Gpxil browser's manifest gives it) */
    pageAt(at) {
      if (!this.byOffset.has(at)) {
        const p = this._page(at);
        p.catch(() => this.byOffset.delete(at));
        this.byOffset.set(at, p);
      }
      return this.byOffset.get(at);
    }
    /** what reads the tags of a directory: values(tag, n) (up to n values, all without n; null if the page hasn't
        the tag), one(tag, fallback) (the first value) and raw(tag) (the value's bytes, for ASCII and UNDEFINED) */
    _tags(r, tags) {
      const h = this.h;
      /** a View holding the first n values of a tag, read from the file if the directory's bytes don't */
      const view = async (t, n) => {
        const bytes = n * (TYPE_SIZE[t.type] || 1);
        if (bytes > MAX_TAG_BYTES) throw fail("limit", `a tag of ${MiB(bytes)}, more than this reader reads (${MiB(MAX_TAG_BYTES)})`);
        if (r.has(t.at, bytes)) return r;
        const v = new View(await this.source.read(t.at, t.at + bytes, "tags"), t.at, h.little);
        if (!v.has(t.at, bytes)) throw fail("format", "the file ends inside a tag");
        return v;
      };
      const values = async (tag, n) => {
        const t = tags.get(tag);
        if (!t) return null;
        const k = n === undefined ? t.count : Math.min(n, t.count);
        const v = await view(t, k);
        const step = TYPE_SIZE[t.type] || 1;
        const out = new Array(k);
        for (let i = 0; i < k; i++) out[i] = v.num(t.type, t.at + i * step);
        return out;
      };
      const one = async (tag, fallback) => {
        const v = await values(tag, 1);
        return v && v.length ? v[0] : fallback;
      };
      const raw = async (tag) => {
        const t = tags.get(tag);
        if (!t) return null;
        const v = await view(t, t.count);
        return new Uint8Array(v.v.buffer, v.v.byteOffset + (t.at - v.base), t.bytes).slice();
      };
      /** the ImageDescription (a page's JSON metadata) as text, or null */
      const description = async () => {
        const v = await raw(TAG.ImageDescription);
        if (!v) return null;
        let n = v.length;
        while (n && v[n - 1] === 0) n--;
        return new TextDecoder().decode(v.subarray(0, n));
      };
      return { values, one, raw, description };
    }
    /** the ImageDescription of page k, read without checking the page's layout (so that a TIFF that isn't gpxil is
        told as one by its metadata) */
    async description(k) {
      const { r, tags } = await this._dir(await this.ifd(k));
      return this._tags(r, tags).description();
    }
    async _page(at) {
      const { r, tags } = await this._dir(at);
      const { values, one, raw, description } = this._tags(r, tags);
      const tiled = tags.has(TAG.TileWidth) || tags.has(TAG.TileOffsets);
      const page = {
        offset: at,
        width: await one(TAG.ImageWidth, 0),
        height: await one(TAG.ImageLength, 0),
        bits: await one(TAG.BitsPerSample, 1),
        samples: await one(TAG.SamplesPerPixel, 1),
        format: await one(TAG.SampleFormat, 1),
        compression: await one(TAG.Compression, 1),
        predictor: await one(TAG.Predictor, 1),
        tiled,
        /** the page's ImageDescription (its JSON metadata), or null */
        description,
        /** the raw bytes of a tag's value (the sidecar's, say), or null */
        tagBytes: raw,
      };
      const { width: W, height: H, bits } = page;
      if (page.samples !== 1 || ![8, 16, 32].includes(bits)) throw fail("format", `a page of ${page.samples} samples of ${bits} bits: not a gpxil plane`);
      if (![1, 2, 3].includes(page.format) || (page.format === 3 && bits !== 32))
        throw fail("format", `a page of ${bits}-bit samples of format ${page.format}: not a gpxil plane`);
      if (!(count(W) && count(H) && W >= 1 && H >= 1 && count(W * H))) throw fail("format", "a page without a width and height");
      if (![1, 8, 32946].includes(page.compression)) {
        const name = COMPRESSION[page.compression] || "compression " + page.compression;
        const e = fail(
          "compression",
          `compressed with ${name}, which this reader doesn't decode (it reads uncompressed and Deflate pages, as every gpxil reader must; gpxiltools reads the others)`,
        );
        e.compression = name;
        throw e;
      }
      if (!(page.predictor === 1 || page.predictor === 2 || (page.predictor === 3 && page.format === 3)))
        throw fail("predictor", `a page of format ${page.format} with predictor ${page.predictor}, which this reader doesn't undo`);
      const bps = bits / 8;
      let n;
      if (tiled) {
        page.tileWidth = await one(TAG.TileWidth, 0);
        page.tileLength = await one(TAG.TileLength, 0);
        if (!(count(page.tileWidth) && count(page.tileLength) && page.tileWidth >= 1 && page.tileLength >= 1))
          throw fail("format", "a tiled page without a tile size");
        page.across = Math.ceil(W / page.tileWidth);
        n = page.across * Math.ceil(H / page.tileLength);
        page.segmentBytes = page.tileWidth * page.tileLength * bps;
      } else {
        const rps = await one(TAG.RowsPerStrip, 2 ** 32 - 1);
        page.rowsPerStrip = Math.max(1, Math.min(count(rps) ? rps : H, H));
        n = Math.ceil(H / page.rowsPerStrip);
        page.segmentBytes = page.rowsPerStrip * W * bps;
      }
      const what = tiled ? "tiles" : "strips";
      if (page.segmentBytes > MAX_SEGMENT)
        throw fail("limit", `${what} of ${MiB(page.segmentBytes)}, more than this reader decodes at once (${MiB(MAX_SEGMENT)})`);
      if (n > MAX_SEGMENTS) throw fail("limit", `a page of ${n} ${what}, more than this reader reads (${MAX_SEGMENTS})`);
      page.offsets = (await values(tiled ? TAG.TileOffsets : TAG.StripOffsets, n)) || [];
      page.counts = (await values(tiled ? TAG.TileByteCounts : TAG.StripByteCounts, n)) || [];
      if (page.offsets.length < n || page.counts.length < n)
        throw fail("format", `a page of ${n} ${what} lists ${Math.min(page.offsets.length, page.counts.length)}`);
      // stored, a strip or tile needs no more bytes than it holds decoded (Deflate: plus what a stored block adds)
      const full = page.segmentBytes;
      const most = page.compression === 1 ? full + 16 : full + (full >> 3) + (full >> 6) + 64;
      for (let s = 0; s < n; s++) {
        if (!(count(page.offsets[s]) && count(page.counts[s]))) throw fail("format", `${what} at offsets that aren't`);
        if (page.counts[s] > most) throw fail("format", `${what.slice(0, -1)} ${s} is stored in ${page.counts[s]} bytes, and it holds ${full}`);
      }
      return page;
    }
    _key(page, s) {
      return `${this.source.key}#${page.offset}#${s}#${page.offsets[s]}+${page.counts[s]}`;
    }
    /** strip or tile s of a page, decoded: its stored bytes inflated (no further than its size) and checked */
    async _decode(page, s, stored) {
      const bps = page.bits / 8;
      const full = page.segmentBytes;
      const need = page.tiled ? full : Math.min(page.rowsPerStrip, page.height - s * page.rowsPerStrip) * page.width * bps;
      const what = page.tiled ? "tile" : "strip";
      let data;
      if (page.compression === 1) data = stored.slice(0, Math.min(stored.length, full));
      else {
        try {
          data = await inflate(stored, full, need);
        } catch (e) {
          throw e.code === "limit" ? fail("format", `${what} ${s} inflates to more than the ${full} bytes it holds`) : e;
        }
      }
      if (data.length < need) throw fail("format", `${what} ${s} holds ${data.length} bytes, and its pixels need ${need}`);
      if (page.predictor !== 1) unpredict(page, data, page.tiled ? page.tileWidth : page.width, this.h.little);
      return data;
    }
    /** strips or tiles of a page (indices in increasing order), decoded; those next to each other in the file read
        together, those read before taken from the cache, those being read awaited */
    _segments(page, idx) {
      const out = new Array(idx.length);
      const missing = [];
      idx.forEach((s, i) => {
        const key = this._key(page, s);
        const hit = recall(key);
        if (hit) out[i] = hit;
        else if (pending.has(key)) out[i] = pending.get(key);
        else missing.push([s, i]);
      });
      const groups = [];
      for (const [s, i] of missing) {
        const g = groups[groups.length - 1];
        const off = page.offsets[s];
        const n = page.counts[s];
        if (g && off === g.end && g.end - g.start + n <= GROUP) {
          g.items.push([s, i]);
          g.end = off + n;
        } else groups.push({ start: off, end: off + n, items: [[s, i]] });
      }
      for (const g of groups) {
        const fetched = this.source.read(g.start, g.end, "strips");
        for (const [s, i] of g.items) {
          const key = this._key(page, s);
          const p = fetched
            .then(async (bytes) => {
              const o = page.offsets[s] - g.start;
              const stored = bytes.subarray(o, o + page.counts[s]);
              if (stored.length < page.counts[s]) throw fail("format", "the file ends inside a strip");
              const data = await this._decode(page, s, stored);
              remember(key, data);
              return data;
            })
            .finally(() => pending.delete(key));
          pending.set(key, p);
          out[i] = p;
        }
      }
      return Promise.all(out);
    }
    /** the values at 1-D positions [start, end) of a page whose rows are page.width wide: a Uint8Array, or an
        Int8Array, (U)Int16Array, (U)Int32Array or Float32Array as the page's type is */
    async readPage(page, start, end) {
      const h = await this.header();
      const W = page.width;
      const bps = page.bits / 8;
      start = Math.max(0, Math.floor(start));
      end = Math.min(Math.floor(end), W * page.height);
      const n = Math.max(0, end - start);
      const out = new Uint8Array(n * bps);
      if (n) {
        const r0 = Math.floor(start / W);
        const r1 = Math.floor((end - 1) / W);
        if (!page.tiled) {
          const rps = page.rowsPerStrip;
          const idx = [];
          for (let s = Math.floor(r0 / rps); s <= Math.floor(r1 / rps); s++) idx.push(s);
          const data = await this._segments(page, idx);
          idx.forEach((s, i) => {
            const first = s * rps * W; // the 1-D position of the strip's first pixel
            const a = Math.max(start, first) - first;
            const b = Math.min(end, first + rps * W) - first;
            if (b > a) out.set(data[i].subarray(a * bps, b * bps), (first + a - start) * bps);
          });
        } else {
          // tiles: rows r0..r1 of the image, the first from start's column, the last to end's
          const TW = page.tileWidth;
          const TL = page.tileLength;
          const lo = (y) => (y === r0 ? start - r0 * W : 0);
          const hi = (y) => (y === r1 ? end - r1 * W : W);
          const wanted = new Set();
          const mark = (y, xa, xb) => {
            for (let tc = Math.floor(xa / TW); tc <= Math.floor((xb - 1) / TW); tc++) wanted.add(Math.floor(y / TL) * page.across + tc);
          };
          mark(r0, lo(r0), hi(r0));
          if (r1 > r0) mark(r1, lo(r1), hi(r1));
          for (let y = r0 + 1; y < r1; y = (Math.floor(y / TL) + 1) * TL) mark(y, 0, W); // (one row of each tile row between)
          const idx = [...wanted].sort((a, b) => a - b);
          const data = await this._segments(page, idx);
          idx.forEach((t, i) => {
            const tr = Math.floor(t / page.across);
            const tc = t % page.across;
            for (let y = Math.max(r0, tr * TL); y <= Math.min(r1, tr * TL + TL - 1); y++) {
              const xa = Math.max(lo(y), tc * TW);
              const xb = Math.min(hi(y), tc * TW + TW);
              if (xb > xa)
                out.set(data[i].subarray(((y - tr * TL) * TW + xa - tc * TW) * bps, ((y - tr * TL) * TW + xb - tc * TW) * bps), (y * W + xa - start) * bps);
            }
          });
        }
      }
      return typed(page, out, h.little);
    }
  }
  /** undo a predictor, row by row of a decoded strip or tile whose rows are `width` samples wide: 2, horizontal
      differencing of integers; 3, the floating-point predictor (bytes of each row in planes, most significant
      first, then differenced) of Adobe's TIFF Technical Note 3 */
  function unpredict(page, data, width, little) {
    const bps = page.bits / 8;
    const rowBytes = width * bps;
    const rows = Math.floor(data.length / rowBytes);
    if (page.predictor === 2) {
      const v = new DataView(data.buffer, data.byteOffset, data.byteLength);
      for (let y = 0; y < rows; y++) {
        const o = y * rowBytes;
        if (bps === 1) for (let x = 1; x < width; x++) data[o + x] += data[o + x - 1];
        else if (bps === 2) for (let x = 1; x < width; x++) v.setUint16(o + 2 * x, v.getUint16(o + 2 * x, little) + v.getUint16(o + 2 * x - 2, little), little);
        else for (let x = 1; x < width; x++) v.setUint32(o + 4 * x, v.getUint32(o + 4 * x, little) + v.getUint32(o + 4 * x - 4, little), little);
      }
      return;
    }
    const tmp = new Uint8Array(rowBytes);
    for (let y = 0; y < rows; y++) {
      const row = data.subarray(y * rowBytes, (y + 1) * rowBytes);
      for (let i = 1; i < rowBytes; i++) row[i] += row[i - 1];
      tmp.set(row);
      for (let x = 0; x < width; x++) for (let b = 0; b < bps; b++) row[x * bps + (little ? bps - 1 - b : b)] = tmp[b * width + x];
    }
  }
  function typed(page, bytes, little) {
    if (page.bits === 8) return page.format === 2 ? new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length) : bytes;
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    // (TIFF data are in the file's byte order, typed arrays in the machine's, which is little-endian everywhere)
    if (!little) {
      const u = new Uint8Array(buf);
      const w = page.bits / 8;
      for (let i = 0; i < u.length; i += w) u.subarray(i, i + w).reverse();
    }
    if (page.bits === 16) return page.format === 2 ? new Int16Array(buf) : new Uint16Array(buf);
    return page.format === 3 ? new Float32Array(buf) : page.format === 2 ? new Int32Array(buf) : new Uint32Array(buf);
  }

  /* ---------------- gpxil ---------------- */
  /** a contig's sidecar text (section 7) as entries [{offset, hap, kind, data}], in the order stored */
  function parseSidecar(text) {
    const out = [];
    for (const line of String(text).split("\n")) {
      const l = line.replace(/\r$/, "");
      if (!l || l[0] === "#" || l.startsWith("offset\t")) continue;
      const [off, hap, kind, ...rest] = l.split("\t");
      out.push({ offset: Number(off), hap: Number(hap), kind, data: rest.join("\t") });
    }
    return out;
  }

  /** open a gpxil file: its metadata (the JSON of the first page) and its contigs */
  async function open(source) {
    const file = new TiffFile(source);
    const desc = await file.description(0); // (page 0's layout is checked when it is read: a TIFF that isn't gpxil is told by this)
    let meta;
    try {
      meta = JSON.parse(desc);
    } catch (e) {
      throw fail("not-gpxil", "a TIFF file, but not gpxil: its first page has no gpxil metadata");
    }
    if (!meta || typeof meta !== "object" || meta.gpxil == null || !Array.isArray(meta.contigs))
      throw fail("not-gpxil", "a TIFF file, but not gpxil: its metadata has no gpxil version or contigs");
    const version = String(meta.gpxil);
    if (version.split(".")[0] !== MAJOR) throw fail("version", `a file of gpxil ${version}, a major version this reader (of gpxil ${SPEC}) can't read`);
    return new Gpxil(file, meta);
  }

  /** a window's "window" key (section 3.1, 9.3), checked: its parent named, where it starts in it, inside it, and
      none of its names its parent's */
  function checkWindow(c, w) {
    const bad = (why) => fail("format", `the metadata's window ${c.name}: ${why}`);
    if (!w || typeof w !== "object" || Array.isArray(w)) throw bad("its window key must be an object, with of and start");
    if (typeof w.of !== "string" || !/^[\x21-\x7e]+$/.test(w.of)) throw bad("of must name its parent");
    if (!(Number.isSafeInteger(w.start) && w.start >= 0)) throw bad(`start must be a whole number, 0 or more (it is ${JSON.stringify(w.start)})`);
    if (w.length !== undefined && !(Number.isSafeInteger(w.length) && w.length >= 1)) throw bad("its parent's length must be a whole number");
    if (w.length !== undefined && w.start + c.length > w.length) throw bad(`it runs past the end of ${w.of}`);
    if (w.aliases !== undefined && !(Array.isArray(w.aliases) && w.aliases.every((a) => typeof a === "string")))
      throw bad("its parent's aliases must be a list of names");
    const parent = new Set([w.of, ...(w.aliases || [])]);
    if (parent.has(c.name) || c.aliases.some((a) => parent.has(a))) throw bad("its names must differ from its parent's");
    for (const k of ["md5", "refget"]) if (w[k] !== undefined && typeof w[k] !== "string") throw bad(`its parent's ${k} must be a string`);
    const str = (k) => (typeof w[k] === "string" ? w[k] : null);
    return {
      of: w.of,
      start: w.start,
      length: w.length === undefined ? null : w.length,
      md5: str("md5"),
      refget: str("refget"),
      aliases: (w.aliases || []).slice(),
    };
  }

  /** "chr17:7,661,779-7,687,546" -> {name: "chr17", start: 7661778, end: 7687546} (0-based, the end excluded);
      "chr17:7,661,779" -> one base; "chr17" -> {name: "chr17"}. The name ends at the last colon, so a window's name
      (chr17:7561779-7787546) is a name too, and can be followed by a region; numbers may have commas and spaces, and
      don't begin with 0, so a name with colons in it (HLA-A*01:01:01:01) is a name. null for an interval that isn't
      one (from position 0, or ending before it starts). gpxiltools' parse_region is the same */
  function parseRegion(text) {
    const t = String(text || "").replace(/^[ \t\r\n]+|[ \t\r\n]+$/g, "");
    if (!t) return null;
    const m = t.match(/^(.+):[ ]*([1-9][0-9, ]*?)[ ]*(?:-[ ]*([1-9][0-9, ]*?))?$/);
    if (!m) return /^.+:[ ]*0[ ]*(?:-[ ]*[0-9, ]+)?$/.test(t) ? null : { name: t };
    const s = Number(m[2].replace(/[, ]/g, ""));
    const e = m[3] === undefined ? s : Number(m[3].replace(/[, ]/g, ""));
    if (!(Number.isSafeInteger(s) && Number.isSafeInteger(e) && s >= 1 && e >= s)) return null;
    return { name: m[1], start: s - 1, end: e };
  }
  const fmt = (n) => n.toLocaleString("en-GB");
  /** [start, end) of a contig as a region of its sequence: "chr17:7,661,779-7,687,546" */
  function regionText(contig, start, end) {
    const s = start === undefined ? 0 : start;
    const e = end === undefined ? contig.length : end;
    return `${contig.parent}:${fmt(contig.start + s + 1)}-${fmt(contig.start + e)}`;
  }

  class Gpxil {
    constructor(file, meta) {
      this.file = file;
      this.meta = meta;
      this.version = String(meta.gpxil);
      const width = Number((meta.layout || {}).width) || 0;
      const object = (v) => v && typeof v === "object" && !Array.isArray(v);
      this.contigs = meta.contigs.map((c, i) => {
        if (!object(c) || !["string", "number"].includes(typeof c.name) || c.name === "" || !count(Number(c.length)))
          throw fail("format", `the metadata's contig ${i + 1} has no name or length`);
        if ((c.pages !== undefined && !object(c.pages)) || (c.aliases !== undefined && !Array.isArray(c.aliases)))
          throw fail("format", `the metadata's contig ${c.name}: pages must be an object, aliases a list`);
        const out = {
          name: String(c.name),
          length: Number(c.length),
          width: Number(c.width || width),
          pages: c.pages || {},
          aliases: (c.aliases || []).map(String),
          entry: c,
          window: null,
        };
        if (c.window !== undefined) out.window = checkWindow(out, c.window);
        // where the contig lies in its sequence: a window's parent (section 3.1), or itself
        out.parent = out.window ? out.window.of : out.name;
        out.start = out.window ? out.window.start : 0;
        out.end = out.start + out.length;
        out.parentLength = out.window ? out.window.length : out.length;
        out.parentAliases = out.window ? out.window.aliases : out.aliases;
        return out;
      });
      // names first, then aliases; exactly as written, then (whole contigs only: a window is found by its name as
      // written, section 3.1) ignoring case; where two match, the first contig
      this._exact = new Map();
      this._folded = new Map();
      const add = (m, k, c) => m.has(k) || m.set(k, c);
      for (const c of this.contigs) add(this._exact, c.name, c);
      for (const c of this.contigs) for (const a of c.aliases) add(this._exact, a, c);
      const whole = this.contigs.filter((c) => !c.window);
      for (const c of whole) add(this._folded, c.name.toLowerCase(), c);
      for (const c of whole) for (const a of c.aliases) add(this._folded, a.toLowerCase(), c);
      this._sidecars = new Map();
    }
    /** a contig by name or alias: as written, else a whole contig ignoring case, with or without "chr" (null if there is
        none) */
    contig(name) {
      if (name && typeof name === "object") return name;
      const s = String(name);
      const n = s.toLowerCase();
      return this._exact.get(s) || this._folded.get(n) || this._folded.get("chr" + n) || this._folded.get(n.replace(/^chr/, "")) || null;
    }
    _contig(contig) {
      const c = this.contig(contig);
      if (!c) throw fail("contig", `no contig ${contig}`);
      return c;
    }
    /** the file's windows (section 3.1) */
    get windows() {
      return this.contigs.filter((c) => c.window);
    }
    /** the windows of a sequence (by its name, or another name of it; as written, else ignoring case, with or
        without "chr"), in file order */
    windowsOf(name) {
      const n = String(name).toLowerCase();
      const want = new Set([n, "chr" + n, n.replace(/^chr/, "")]);
      return this.windows.filter((c) => [c.parent, ...c.parentAliases].some((x) => want.has(x.toLowerCase())));
    }
    /** where a region is: {contig, start, end}, the contig that holds it and its offsets there (0-based, the end
        excluded). Give a contig's name (a window's too), or a sequence that windows hold, with positions in the
        sequence: on a window, positions in its parent, so a region of the parent is found in a window that holds
        it all. Without positions, the whole contig (for a sequence that only windows hold, its window, if it has
        one only). A text, "chr17:7,661,779-7,687,546", is read as a region, unless it is a contig's name */
    locate(name, start, end) {
      if (start === null) start = undefined; // (null as in JSON: not given)
      if (end === null) end = undefined;
      if (start === undefined && end === undefined && !(name && typeof name === "object")) {
        const text = String(name).replace(/^[ \t\r\n]+|[ \t\r\n]+$/g, "");
        if (!this.contig(name) && this.contig(text)) {
          name = text; // (a contig's name, with spaces around it)
        } else if (!this.contig(name)) {
          const r = parseRegion(name);
          if (!r) throw fail("region", `${name} isn't a region: give CONTIG or CONTIG:START-END, 1-based`);
          if (r.start !== undefined) return this.locate(r.name, r.start, r.end);
        }
      }
      const whole = start === undefined && end === undefined;
      const c = this.contig(name);
      if (c) {
        if (whole) return { contig: c, start: 0, end: c.length };
        const s = start === undefined ? c.start : start;
        const e = end === undefined ? c.end : end;
        if (!(c.start <= s && s < e && e <= c.end))
          throw fail(
            "region",
            c.window
              ? `${c.parent}:${fmt(s + 1)}-${fmt(e)} is outside the window ${c.name}, which holds ${regionText(c)}`
              : `${c.name}:${fmt(s + 1)}-${fmt(e)} is outside ${c.name} (${fmt(c.length)} bases)`,
          );
        return { contig: c, start: s - c.start, end: e - c.start };
      }
      const wins = this.windowsOf(name);
      if (wins.length) {
        if (whole) {
          if (wins.length === 1) return { contig: wins[0], start: 0, end: wins[0].length };
          throw fail(
            "region",
            `${wins.length} windows of ${name}: give one, or a region (${wins
              .slice(0, 4)
              .map((w) => regionText(w))
              .join(", ")})`,
          );
        }
        const s = start || 0;
        const exact = wins.filter((w) => w.start === s && w.end === end);
        for (const w of exact.length ? exact : wins) {
          const e = end === undefined ? w.end : end;
          if (w.start <= s && s < w.end && e <= w.end && s < e) return { contig: w, start: s - w.start, end: e - w.start };
        }
        const e = end === undefined ? s + 1 : end;
        throw fail(
          "region",
          `${name}:${fmt(s + 1)}-${fmt(e)} isn't all in a window of this file (${wins
            .slice(0, 4)
            .map((w) => regionText(w))
            .join(", ")})`,
        );
      }
      throw fail("contig", `no contig ${name}`);
    }
    /** the page of a contig's plane */
    page(contig, plane) {
      const c = this._contig(contig);
      if (!Object.prototype.hasOwnProperty.call(c.pages, plane)) return Promise.reject(fail("plane", `${c.name} has no ${plane} plane`));
      const k = c.pages[plane];
      if (typeof k !== "number" || !count(k))
        return Promise.reject(fail("format", `${c.name}'s ${plane} plane is on page ${JSON.stringify(k)}, which isn't a page number`));
      return this.file.page(k);
    }
    /** a plane's values over [start, end) of a contig (0-based, the end excluded; clipped to the contig) */
    async read(contig, plane, start, end) {
      const c = this._contig(contig);
      const page = await this.page(c, plane);
      if (page.width * page.height < c.length) throw fail("layout", `${c.name}'s ${plane} page holds fewer pixels than the contig has bases`);
      return this.file.readPage(page, Math.max(0, start), Math.min(c.length, end));
    }
    /** a contig's sidecar (section 7): [{offset, hap, kind, data}] as stored; [] if it has none */
    sidecar(contig) {
      const c = this._contig(contig);
      if (!("gt" in c.pages)) return Promise.resolve([]);
      if (!this._sidecars.has(c.name)) {
        const p = (async () => {
          const page = await this.page(c, "gt");
          const raw = await page.tagBytes(TAG.Sidecar);
          if (!raw) return [];
          let text;
          try {
            text = await inflate(raw, MAX_SIDECAR);
          } catch (e) {
            throw e.code === "limit" ? fail("limit", `${c.name}'s sidecar inflates to more than this reader reads (${MiB(MAX_SIDECAR)})`) : e;
          }
          return parseSidecar(new TextDecoder().decode(text));
        })();
        p.catch(() => this._sidecars.delete(c.name));
        this._sidecars.set(c.name, p);
      }
      return this._sidecars.get(c.name);
    }
  }

  /* ---------------- what the codes mean (section 5, appendix A) ---------------- */
  const IUPAC = "-ACMGRSVTWYHKDBN"; // ref, low nibble
  const STATE = { 0: "REF", 1: "A", 2: "C", 4: "G", 8: "T", 10: "ABSENT", 12: "COMPLEX", 13: "DEL", 15: "UNKNOWN" }; // gt nibbles
  const FLAGS = [
    [1, "PHASED"],
    [2, "INS1"],
    [4, "INS2"],
    [8, "FILTERED"],
    [16, "VARIANT_SITE"],
  ];
  /** a ref code as its IUPAC letter, lowercase if soft-masked */
  const base = (code) => {
    const ch = IUPAC[code & 15];
    return code & 16 ? ch.toLowerCase() : ch;
  };
  /** a gt nibble's name ("reserved N" for a value the specification doesn't use, which readers treat as UNKNOWN) */
  const state = (s) => STATE[s] || "reserved " + s;
  /** the names of the flags set in a flags byte */
  const flags = (f) => FLAGS.filter(([b]) => f & b).map(([, n]) => n);

  const api = {
    SPEC,
    open,
    parseRegion,
    regionText,
    TiffFile,
    fromUrl,
    fromBlob,
    fromBytes,
    inflate,
    parseSidecar,
    setCacheBytes,
    cacheStats,
    TAG,
    IUPAC,
    STATE,
    FLAGS,
    base,
    state,
    flags,
  };
  root.GpxilRead = api;
  if (typeof module === "object" && module && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
