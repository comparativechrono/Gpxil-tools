/* gpxil-read.js: a reader of gpxil files for web pages, and for Node.
 *
 * Part of gpxiltools (https://github.com/comparativechrono/Gpxil-tools), under the MIT licence. The Gpxil browser
 * and GpxilStudio each keep a copy of this file as it is here at a recorded commit; it is not published on npm.
 * js/README.md says how to update the copies.
 *
 * A gpxil file (https://github.com/comparativechrono/gpxil) is a TIFF with one page per plane of each contig,
 * stored in strips of whole rows. This reads its metadata, then any interval of a plane by reading only the strips
 * that hold it: from a Blob, from bytes, or from the web with HTTP range requests. Strips are inflated with the
 * DecompressionStream that browsers and Node 18 or later have. It reads files of any version 0.x of the
 * specification (it implements 0.2), uncompressed or Deflate, classic TIFF or BigTIFF, in either byte order.
 *
 * Load it as a classic script (<script src="gpxil-read.js">), import it for its side effect (import
 * "./gpxil-read.js"), or require() it in Node: it defines globalThis.GpxilRead, and module.exports in Node.
 *
 *   const g = await GpxilRead.open(GpxilRead.fromUrl("HG002_chr17.gpxil.tif"));
 *   g.version;                                                  // "0.2"
 *   const gt = await g.read("chr17", "gt", 7661778, 7687546);   // Uint8Array; 0-based, the end excluded
 *   const side = await g.sidecar("chr17");                      // [{offset, hap, kind, data}]
 *
 * A source is {key, size, read(start, end, what) -> Promise<Uint8Array>}, the end excluded; `what` says what is
 * read ("header", "directory", "tags" or "strips"), for a page that shows what it is loading. Lower down,
 * new GpxilRead.TiffFile(source) reads pages by index, or by the offset of their directory, and readPage(page,
 * start, end) the values at 1-D positions [start, end) of a page.
 */
(function (root) {
  "use strict";

  const SPEC = "0.2"; // the version of the gpxil specification this reader implements
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
    SampleFormat: 339,
    Sidecar: 65400, // GPXIL_SIDECAR: the zlib-compressed sidecar, on a contig's gt page (section 8.3)
  };
  const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8 };
  const COMPRESSION = { 1: "none", 5: "LZW", 7: "JPEG", 8: "Deflate", 32773: "PackBits", 32946: "Deflate", 34925: "LZMA", 50000: "Zstandard" };
  const PROBE = 16384; // bytes read at a directory: its tags and, as tifffile lays files out, their arrays
  const GROUP = 8 << 20; // strips next to each other in the file are read together, up to this many bytes
  const MAX_TAGS = 4096; // a directory claiming more is not a gpxil page (section 13: inputs are untrusted)

  /** an Error with a code, so that a page can say it in its own words: http, not-tiff, not-gpxil, version, format,
      tiles, layout, compression, predictor, contig, plane */
  function fail(code, message) {
    const e = new Error(message);
    e.code = code;
    return e;
  }

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
    return { key: o.key || url, size: o.size || 0, read: (a, b, what) => get(url, a, b, what) };
  }

  /** zlib-wrapped Deflate (TIFF compression 8, and the sidecar) */
  async function inflate(bytes) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
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
  }

  class TiffFile {
    constructor(source) {
      this.source = source;
      this.h = null;
      this.offsets = []; // directory offsets, page by page, as far as walked
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
          this.h = h;
          return h;
        })();
      return this._h;
    }
    async _dir(at) {
      const h = await this.header();
      let bytes = await this.source.read(at, this.source.size ? Math.min(this.source.size, at + PROBE) : at + PROBE, "directory");
      let r = new View(bytes, at, h.little);
      if (!r.has(at, h.big ? 8 : 2)) throw fail("format", `the file ends before the directory at ${at}`);
      const n = h.big ? r.u64(at) : r.u16(at);
      if (n > MAX_TAGS) throw fail("format", `a directory of ${n} tags: not a gpxil page`);
      const size = h.big ? 20 : 12;
      const first = at + (h.big ? 8 : 2);
      const end = first + n * size + (h.big ? 8 : 4);
      if (!r.has(first, end - first)) {
        bytes = await this.source.read(at, end, "directory");
        r = new View(bytes, at, h.little);
        if (!r.has(first, end - first)) throw fail("format", `the file ends inside the directory at ${at}`);
      }
      const tags = new Map();
      for (let i = 0; i < n; i++) {
        const e = first + i * size;
        const tag = r.u16(e);
        const type = r.u16(e + 2);
        const count = h.big ? r.u64(e + 4) : r.u32(e + 4);
        const bytesOf = (TYPE_SIZE[type] || 1) * count;
        const inline = bytesOf <= (h.big ? 8 : 4);
        tags.set(tag, { type, count, at: inline ? e + (h.big ? 12 : 8) : h.big ? r.u64(e + 12) : r.u32(e + 8), bytes: bytesOf });
      }
      return { r, tags, next: h.big ? r.u64(first + n * size) : r.u32(first + n * size) };
    }
    /** the directory offset of page k, walking the chain of directories as far as it. One walk at a time: two at
        once would both add the directory after the last one known, and every later page would be off by one */
    ifd(k) {
      const walk = this._walking.then(async () => {
        await this.header();
        while (this.offsets.length <= k) {
          const at = this.offsets[this.offsets.length - 1];
          if (!at) throw fail("format", `the file has no page ${k}`);
          this.offsets.push((await this._dir(at)).next);
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
    async _page(at) {
      const { r, tags } = await this._dir(at);
      const h = this.h;
      const values = async (tag) => {
        const t = tags.get(tag);
        if (!t) return null;
        let rr = r;
        if (!rr.has(t.at, t.bytes)) {
          rr = new View(await this.source.read(t.at, t.at + t.bytes, "tags"), t.at, h.little);
          if (!rr.has(t.at, t.bytes)) throw fail("format", `the file ends inside tag ${tag}`);
        }
        const step = TYPE_SIZE[t.type] || 1;
        const out = new Array(t.count);
        for (let i = 0; i < t.count; i++) {
          const o = t.at + i * step;
          out[i] =
            t.type === 3
              ? rr.u16(o)
              : t.type === 4 || t.type === 9 || t.type === 13
                ? rr.u32(o)
                : t.type === 16 || t.type === 17 || t.type === 18
                  ? rr.u64(o)
                  : rr.u8(o);
        }
        return out;
      };
      const one = async (tag, fallback) => {
        const v = await values(tag);
        return v && v.length ? v[0] : fallback;
      };
      if (tags.has(TAG.TileWidth)) throw fail("tiles", "a tiled page: this reader reads pages stored in strips, as gpxil files are");
      const page = {
        offset: at,
        width: await one(TAG.ImageWidth, 0),
        height: await one(TAG.ImageLength, 0),
        bits: await one(TAG.BitsPerSample, 1),
        samples: await one(TAG.SamplesPerPixel, 1),
        format: await one(TAG.SampleFormat, 1),
        compression: await one(TAG.Compression, 1),
        predictor: await one(TAG.Predictor, 1),
        rowsPerStrip: await one(TAG.RowsPerStrip, 2 ** 32 - 1),
        offsets: (await values(TAG.StripOffsets)) || [],
        counts: (await values(TAG.StripByteCounts)) || [],
        /** the page's ImageDescription (its JSON metadata), or null */
        description: async () => {
          const v = await values(TAG.ImageDescription);
          if (!v) return null;
          let n = v.length;
          while (n && v[n - 1] === 0) n--;
          return new TextDecoder().decode(Uint8Array.from(v.slice(0, n)));
        },
        /** the raw bytes of a tag's value (the sidecar's, say), or null */
        tagBytes: async (tag) => {
          const t = tags.get(tag);
          if (!t) return null;
          return r.has(t.at, t.bytes)
            ? new Uint8Array(r.v.buffer, r.v.byteOffset + (t.at - r.base), t.bytes).slice()
            : this.source.read(t.at, t.at + t.bytes, "tags");
        },
      };
      if (page.samples !== 1 || ![8, 16, 32].includes(page.bits))
        throw fail("format", `a page of ${page.samples} samples of ${page.bits} bits: not a gpxil plane`);
      if (!(page.width >= 1 && page.height >= 1)) throw fail("format", "a page without a width or height");
      if (![1, 8, 32946].includes(page.compression)) {
        const name = COMPRESSION[page.compression] || "compression " + page.compression;
        const e = fail(
          "compression",
          `compressed with ${name}, which this reader doesn't decode (it reads uncompressed and Deflate pages, as every gpxil reader must; gpxiltools reads the others)`,
        );
        e.compression = name;
        throw e;
      }
      if (page.predictor !== 1) throw fail("predictor", `a page with a predictor (${page.predictor}), which this reader doesn't undo`);
      page.rowsPerStrip = Math.max(1, Math.min(page.rowsPerStrip, page.height));
      const strips = Math.ceil(page.height / page.rowsPerStrip);
      if (page.offsets.length < strips || page.counts.length < strips)
        throw fail("format", `a page of ${strips} strips lists ${Math.min(page.offsets.length, page.counts.length)}`);
      return page;
    }
    _key(page, s) {
      return `${this.source.key}#${page.offset}#${s}`;
    }
    /** decoded strips s0..s1 of a page, those next to each other in the file read together */
    _strips(page, s0, s1) {
      const out = new Array(s1 - s0 + 1);
      const missing = [];
      for (let s = s0; s <= s1; s++) {
        const key = this._key(page, s);
        const hit = recall(key);
        if (hit) out[s - s0] = hit;
        else if (pending.has(key)) out[s - s0] = pending.get(key);
        else missing.push(s);
      }
      const groups = [];
      for (const s of missing) {
        const g = groups[groups.length - 1];
        const off = page.offsets[s];
        const n = page.counts[s];
        if (g && off === g.end && g.end - g.start + n <= GROUP) {
          g.strips.push(s);
          g.end = off + n;
        } else groups.push({ start: off, end: off + n, strips: [s] });
      }
      for (const g of groups) {
        const fetched = this.source.read(g.start, g.end, "strips");
        for (const s of g.strips) {
          const key = this._key(page, s);
          const p = fetched
            .then(async (bytes) => {
              const o = page.offsets[s] - g.start;
              const raw = bytes.subarray(o, o + page.counts[s]);
              if (raw.length < page.counts[s]) throw fail("format", "the file ends inside a strip");
              const data = page.compression === 1 ? raw.slice() : await inflate(raw);
              remember(key, data);
              return data;
            })
            .finally(() => pending.delete(key));
          pending.set(key, p);
          out[s - s0] = p;
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
        const rps = page.rowsPerStrip;
        const s0 = Math.floor(Math.floor(start / W) / rps);
        const s1 = Math.floor(Math.floor((end - 1) / W) / rps);
        const strips = await this._strips(page, s0, s1);
        for (let s = s0; s <= s1; s++) {
          const data = strips[s - s0];
          const first = s * rps * W; // the 1-D position of the strip's first pixel
          const a = Math.max(start, first) - first;
          const b = Math.min(end, first + Math.floor(data.length / bps)) - first;
          if (b > a) out.set(data.subarray(a * bps, b * bps), (first + a - start) * bps);
        }
      }
      return typed(page, out, h.little);
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
    const p0 = await file.page(0);
    const desc = await p0.description();
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

  class Gpxil {
    constructor(file, meta) {
      this.file = file;
      this.meta = meta;
      this.version = String(meta.gpxil);
      const width = Number((meta.layout || {}).width) || 0;
      this.contigs = meta.contigs.map((c) => ({
        name: String(c.name),
        length: Number(c.length),
        width: Number(c.width || width),
        pages: c.pages || {},
        aliases: (c.aliases || []).map(String),
        entry: c,
      }));
      this._byName = new Map();
      for (const c of this.contigs) this._byName.set(c.name.toLowerCase(), c);
      for (const c of this.contigs) for (const a of c.aliases) if (!this._byName.has(a.toLowerCase())) this._byName.set(a.toLowerCase(), c);
      this._sidecars = new Map();
    }
    /** a contig by name or alias, ignoring case, with or without "chr" (null if there is none) */
    contig(name) {
      if (name && typeof name === "object") return name;
      const n = String(name).toLowerCase();
      return this._byName.get(n) || this._byName.get("chr" + n) || this._byName.get(n.replace(/^chr/, "")) || null;
    }
    _contig(contig) {
      const c = this.contig(contig);
      if (!c) throw fail("contig", `no contig ${contig}`);
      return c;
    }
    /** the page of a contig's plane */
    page(contig, plane) {
      const c = this._contig(contig);
      if (!(plane in c.pages)) return Promise.reject(fail("plane", `${c.name} has no ${plane} plane`));
      return this.file.page(Number(c.pages[plane]));
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
          return raw ? parseSidecar(new TextDecoder().decode(await inflate(raw))) : [];
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
