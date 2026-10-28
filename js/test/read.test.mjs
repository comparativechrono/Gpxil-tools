// The JavaScript reader (js/gpxil-read.js) against files that gpxiltools wrote: every plane of every contig, whole
// and in windows, read from bytes, from a Blob and over HTTP range requests, compared with what gpxiltools reads
// from the same files; their sidecars and metadata; the files it must refuse, each with its code; and how it loads.
//
//   node --test js/test/read.test.mjs     (runs python3 js/test/make_fixtures.py first; $PYTHON names another Python)
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";
import zlib from "node:zlib";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const READER = path.join(HERE, "..", "gpxil-read.js");
const R = createRequire(import.meta.url)(READER);
const OUT = fs.mkdtempSync(path.join(os.tmpdir(), "gpxil-read-"));
let expected;

let server;
let base;
let requests = 0;
let wholeAnswers = 0; // (requests answered with the whole file: an invalid range, or none)
before(async () => {
  execFileSync(process.env.PYTHON || "python3", [path.join(HERE, "make_fixtures.py"), OUT], { stdio: "inherit" });
  expected = JSON.parse(fs.readFileSync(path.join(OUT, "expected.json"), "utf8"));
  // a static server that answers range requests, as GitHub Pages does
  server = http.createServer((req, res) => {
    requests++;
    const file = path.join(OUT, decodeURIComponent(new URL(req.url, "http://x").pathname));
    if (!file.startsWith(OUT) || !fs.existsSync(file)) {
      res.writeHead(404);
      return res.end();
    }
    const size = fs.statSync(file).size;
    const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || "");
    if (m && m[2] && +m[2] < +m[1]) {
      wholeAnswers++; // an invalid range is ignored (RFC 9110), as by GitHub Pages: the whole file
      res.writeHead(200, { "Content-Length": size });
      return fs.createReadStream(file).pipe(res);
    }
    if (m) {
      const a = +m[1];
      const b = Math.min(m[2] ? +m[2] : size - 1, size - 1);
      if (a >= size) {
        res.writeHead(416, { "Content-Range": `bytes */${size}` });
        return res.end();
      }
      res.writeHead(206, { "Content-Range": `bytes ${a}-${b}/${size}`, "Content-Length": b - a + 1 });
      fs.createReadStream(file, { start: a, end: b }).pipe(res);
    } else {
      wholeAnswers++;
      res.writeHead(200, { "Content-Length": size });
      fs.createReadStream(file).pipe(res);
    }
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${server.address().port}/`;
});
after(() => {
  server.close();
  fs.rmSync(OUT, { recursive: true, force: true });
});

const TYPED = { uint8: Uint8Array, int8: Int8Array, uint16: Uint16Array, int16: Int16Array, uint32: Uint32Array, int32: Int32Array, float32: Float32Array };
const bytesOf = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
const bin = (fn) => fs.readFileSync(path.join(OUT, fn));
const SOURCES = {
  bytes: (f) => R.fromBytes(fs.readFileSync(path.join(OUT, f))),
  blob: (f) => R.fromBlob(new Blob([fs.readFileSync(path.join(OUT, f))]), "blob:" + f),
  url: (f) => R.fromUrl(base + f),
};
function rng(seed) {
  let s = seed >>> 0;
  return () => (s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32;
}

test("every plane of every file reads as gpxiltools reads it, from bytes, a Blob or the web", async () => {
  for (const [name, want] of Object.entries(expected.good)) {
    for (const [kind, source] of Object.entries(SOURCES)) {
      const g = await R.open(source(want.file));
      const where = `${name} (${kind})`;
      assert.equal(g.version, want.version, where);
      assert.deepEqual(g.meta, want.meta, where);
      assert.deepEqual(
        g.contigs.map((c) => [c.name, c.length, c.width]),
        want.contigs.map((c) => [c.name, c.length, c.width]),
        where,
      );
      for (const c of want.contigs) {
        for (const [plane, p] of Object.entries(c.planes)) {
          const got = await g.read(c.name, plane, 0, c.length);
          assert.ok(got instanceof TYPED[p.dtype], `${where} ${c.name} ${plane}: a ${got.constructor.name}, not ${p.dtype}`);
          assert.ok(bytesOf(got).equals(bin(p.bin)), `${where} ${c.name} ${plane}`);
        }
        assert.deepEqual(await g.sidecar(c.name), c.sidecar, `${where} ${c.name} sidecar`);
      }
    }
  }
});

test("intervals: across rows and strips, clipped at either end, by alias", async () => {
  const r = rng(11);
  for (const [name, want] of Object.entries(expected.good)) {
    const g = await R.open(SOURCES.url(want.file));
    for (const c of want.contigs) {
      for (const [plane, p] of Object.entries(c.planes)) {
        const all = bin(p.bin);
        const size = TYPED[p.dtype].BYTES_PER_ELEMENT;
        const windows = [
          [0, 1],
          [c.length - 1, c.length],
          [-5, 7],
          [c.length - 3, c.length + 50],
          [c.width - 2, c.width + 2],
        ];
        for (let i = 0; i < 12; i++) {
          const a = Math.floor(r() * c.length);
          windows.push([a, a + 1 + Math.floor(r() * Math.min(5000, c.length))]);
        }
        for (const [a, b] of windows) {
          const got = await g.read(c.name, plane, a, b);
          const lo = Math.max(0, a);
          const hi = Math.min(c.length, b);
          assert.ok(bytesOf(got).equals(all.subarray(lo * size, hi * size)), `${name} ${c.name} ${plane} [${a}, ${b})`);
        }
      }
    }
  }
  const g = await R.open(SOURCES.bytes(expected.good.strips.file));
  assert.equal(g.contig("1")?.name, "chr1");
  assert.equal(g.contig("CHR2")?.name, "chr2");
  assert.equal(g.contig("MT")?.name, "chrM"); // (an alias gpxil encode records)
  assert.equal(g.contig("chr7"), null);
  await assert.rejects(g.read("chr9", "gt", 0, 10), { code: "contig" });
  await assert.rejects(g.read("chrM", "gt", 0, 10), { code: "plane" });
});

test("windows (section 3.1): where they lie in their sequence, and regions found in them as gpxiltools finds them", async () => {
  const where = (x) => [x.contig.name, x.start, x.end];
  for (const name of ["windows", "windows-noref"]) {
    const want = expected.good[name];
    const g = await R.open(SOURCES.url(want.file));
    for (const c of want.contigs) {
      const got = g.contig(c.name);
      assert.deepEqual([got.parent, got.start, got.end], [c.parent, c.start, c.start + c.length], `${name} ${c.name}`);
      if (!c.window) assert.equal(got.window, null);
      else {
        const w = c.window;
        assert.deepEqual(got.window, { of: w.of, start: w.start, length: w.length, md5: w.md5, refget: w.refget, aliases: w.aliases });
        assert.equal(got.parentLength, w.length);
        assert.deepEqual(got.parentAliases, w.aliases);
      }
    }
    assert.equal(g.version, "0.3");
  }
  const g = await R.open(SOURCES.bytes(expected.good.windows.file));
  assert.deepEqual(
    g.windows.map((c) => c.name),
    ["chr1:1-500", "chr1:30001-34000", "chr1:33001-40000", "chr2:501-2000"],
  );
  assert.deepEqual(
    g.windowsOf("1").map((c) => c.name),
    ["chr1:1-500", "chr1:30001-34000", "chr1:33001-40000"],
  );
  for (const k of expected.locate) {
    const [name, s, e] = k.ask;
    const args = s === null && e === null ? [name] : [name, s, e]; // (null for a position not given, as gpxiltools' None)
    if (k.error) assert.throws(() => g.locate(...args), { code: k.error }, JSON.stringify(k.ask));
    else assert.deepEqual(where(g.locate(...args)), [k.contig, k.start, k.end], JSON.stringify(k.ask));
  }
  // the same, written as text: a region; a window's name; a window's name and a region; one base
  assert.deepEqual(where(g.locate("chr1:33,501-33,600")), ["chr1:30001-34000", 3500, 3600]);
  assert.deepEqual(where(g.locate("chr1:30001-34000")), ["chr1:30001-34000", 0, 4000]);
  assert.deepEqual(where(g.locate("chr1:30001-34000:30,006-30,010")), ["chr1:30001-34000", 5, 10]);
  assert.deepEqual(where(g.locate("chr2:601")), ["chr2:501-2000", 100, 101]);
  assert.throws(() => g.locate("chr1:10-5"), { code: "region" });
  // the bases a window holds are those of the whole contig there
  const whole = await R.open(SOURCES.bytes(expected.good.strips.file));
  const at = g.locate("chr1", 33500, 33600);
  for (const plane of ["ref", "gt", "flags", "ps", "x-frac"])
    assert.deepEqual(await g.read(at.contig, plane, at.start, at.end), await whole.read("chr1", plane, 33500, 33600), plane);
  // its sidecar is the whole contig's there: chr2's COMPLEX allele at 501 is the window's first base
  const side = await g.sidecar("chr2:501-2000");
  assert.deepEqual(
    side.filter((x) => x.kind === "COMPLEX"),
    (await whole.sidecar("chr2")).filter((x) => x.kind === "COMPLEX").map((x) => ({ ...x, offset: x.offset - 500 })),
  );
  assert.equal(side[0].offset, 0);
  assert.deepEqual(R.parseRegion("chr17:7,661,779-7,687,546"), { name: "chr17", start: 7661778, end: 7687546 });
  assert.deepEqual(R.parseRegion(" chr17 "), { name: "chr17" });
  assert.deepEqual(R.parseRegion("chr17:7561779-7787546:7,600,001"), { name: "chr17:7561779-7787546", start: 7600000, end: 7600001 });
  assert.equal(R.parseRegion("chr1:0-5"), null);
  assert.equal(R.parseRegion("chr1:0"), null);
  assert.equal(R.parseRegion("chr1:10-9"), null);
  assert.equal(R.parseRegion(""), null);
  // numbers may have spaces; they don't begin with 0, so a name with colons in it is a name (as gpxiltools reads it)
  assert.deepEqual(R.parseRegion("chr17: 7 661 779 - 7 687 546"), { name: "chr17", start: 7661778, end: 7687546 });
  assert.deepEqual(R.parseRegion("HLA-A*01:01:01:01"), { name: "HLA-A*01:01:01:01" });
  assert.deepEqual(R.parseRegion("HLA-A*01:01:01:01:1-100"), { name: "HLA-A*01:01:01:01", start: 0, end: 100 });
  assert.deepEqual(R.parseRegion("chr17:,-5"), { name: "chr17:,-5" });
  assert.equal(R.regionText(g.contig("chr1:30001-34000"), 5, 10), "chr1:30,006-30,010");
  assert.equal(R.regionText(g.contig("chrM")), "chrM:1-997");
});

test("pages by the offset of their directory, as the Gpxil browser finds them", async () => {
  const want = expected.good.bigtiff;
  const f = new R.TiffFile(SOURCES.url(want.file));
  for (const c of want.contigs) {
    for (const [plane, p] of Object.entries(c.planes)) {
      const byOffset = await f.pageAt(p.ifd);
      const byIndex = await f.page(p.page);
      assert.equal(byOffset, byIndex, `${c.name} ${plane}: one page object`);
      const got = await f.readPage(byOffset, 0, c.length);
      assert.ok(bytesOf(got).equals(bin(p.bin)), `${c.name} ${plane}`);
    }
  }
});

test("strips next to each other are read in one request, and strips read before in none", async () => {
  const want = expected.good.strips;
  const ps = want.contigs[0].planes.ps; // (207 strips of 3 rows)
  const g = await R.open(SOURCES.url(want.file));
  const page = await g.page("chr1", "ps");
  assert.ok(page.offsets.length > 100, `${page.offsets.length} strips`);
  const n0 = requests;
  const got = await g.read("chr1", "ps", 0, want.contigs[0].length);
  assert.ok(requests - n0 <= 2, `${requests - n0} requests for ${page.offsets.length} strips`);
  assert.ok(bytesOf(got).equals(bin(ps.bin)));
  const n1 = requests;
  await g.read("chr1", "ps", 1000, 50000);
  assert.equal(requests, n1);
});

test("many reads at once, of a file not opened before, are each right", async () => {
  const want = expected.good.bigendian;
  const g = await R.open(R.fromUrl(base + want.file, { key: "concurrent" }));
  const r = rng(5);
  const jobs = [];
  for (let i = 0; i < 60; i++) {
    const c = want.contigs[i % 2];
    const planes = Object.entries(c.planes);
    const [plane, p] = planes[i % planes.length];
    const a = Math.floor(r() * c.length);
    const b = a + 1 + Math.floor(r() * 20000);
    jobs.push(g.read(c.name, plane, a, b).then((got) => [got, p, a, Math.min(b, c.length)]));
  }
  for (const [got, p, a, b] of await Promise.all(jobs)) {
    const size = TYPED[p.dtype].BYTES_PER_ELEMENT;
    assert.ok(bytesOf(got).equals(bin(p.bin).subarray(a * size, b * size)));
  }
});

test("a small cache: strips are read again, and still right", async () => {
  const want = expected.good.plain;
  R.setCacheBytes(4096);
  try {
    const g = await R.open(R.fromBytes(fs.readFileSync(path.join(OUT, want.file)), "small-cache"));
    for (let k = 0; k < 2; k++) {
      for (const [plane, p] of Object.entries(want.contigs[0].planes)) assert.ok(bytesOf(await g.read("chr1", plane, 0, 60000)).equals(bin(p.bin)), plane);
    }
    assert.ok(R.cacheStats().bytes <= 4096 || R.cacheStats().strips === 1);
  } finally {
    R.setCacheBytes(192 * 1024 * 1024);
  }
});

test("tiles and predictors: the files that have them do", async () => {
  const tiled = await R.open(SOURCES.bytes(expected.good.tiled.file));
  const t = await tiled.page("chr1", "gt");
  assert.ok(t.tiled && t.tileWidth === 32 && t.tileLength === 16 && t.width === 97, "tiles 32 wide over rows of 97");
  const p = await R.open(SOURCES.bytes(expected.good.predictors.file));
  assert.equal((await p.page("chr1", "x-score")).predictor, 2);
  assert.equal((await p.page("chr1", "x-frac")).predictor, 3);
  assert.equal((await tiled.page("chr1", "x-frac")).predictor, 3);
  assert.equal((await p.page("chr1", "gt")).predictor, 1);
});

test("files it must refuse, each with a code a page can put in its own words, and quickly", async () => {
  for (const [name, want] of Object.entries(expected.bad)) {
    for (const kind of ["bytes", "url"]) {
      const source = SOURCES[kind](want.file);
      const t0 = Date.now();
      const whole0 = wholeAnswers;
      const ok = (e) => e.code === want.code && (!want.compression || e.compression === want.compression);
      if (want.at === "open") await assert.rejects(R.open(source), ok, `${name} (${kind})`);
      else {
        const g = await R.open(source);
        await assert.rejects(g.read(want.contig, want.plane, 0, 100), ok, `${name} (${kind})`);
        if (want.compression) assert.equal((await g.read(want.contig, "ref", 0, 100)).length, 100, `${name}: its other pages still read`);
      }
      assert.ok(Date.now() - t0 < 5000, `${name} (${kind}): refused after ${Date.now() - t0} ms`);
      assert.equal(wholeAnswers, whole0, `${name} (${kind}): the whole file was sent`);
    }
  }
  await assert.rejects(R.open(R.fromUrl(base + "no-such-file.gpxil.tif")), { code: "http" });
  // the 200 MB strip: refused once more than its 256 KiB come out, not after all of it
  const g = await R.open(SOURCES.bytes(expected.bad.bomb.file));
  await assert.rejects(g.read("c", "ref", 0, 10), /inflates to more than the 262144 bytes it holds/);
  // 100,000 directories one after another: walked from the bytes read, a few dozen reads in all
  let reads = 0;
  const bytes = R.fromBytes(fs.readFileSync(path.join(OUT, expected.bad.chain.file)));
  const counted = { key: "chain-counted", size: bytes.size, read: (a, b, what) => (reads++, bytes.read(a, b, what)) };
  const t0 = Date.now();
  await assert.rejects((await R.open(counted)).read("c", "gt", 0, 10), { code: "format" });
  assert.ok(reads < 100 && Date.now() - t0 < 5000, `${reads} reads, ${Date.now() - t0} ms`);
  const z = zlib.deflateSync(Buffer.from("ACGT".repeat(1000)));
  assert.equal(new TextDecoder().decode(await R.inflate(z, 4000)), "ACGT".repeat(1000));
  await assert.rejects(R.inflate(z, 3999), { code: "limit" });
  await assert.rejects(R.inflate(z.subarray(0, z.length - 6)), { code: "format" }); // (cut short)
});

test("odd files that read: a tag of 2^30 values, names that differ only in case, a float plane with predictor 2, padding after a zlib stream", async () => {
  const many = await R.open(SOURCES.url(expected.odd["many-values"]));
  assert.equal(
    Array.from(await many.read("c", "ref", 0, 20))
      .map(R.base)
      .join(""),
    "ACGT".repeat(5),
  );
  const g = await R.open(SOURCES.bytes(expected.odd.case));
  const first = async (name) => R.base((await g.read(name, "ref", 0, 1))[0]);
  assert.deepEqual([await first("a"), await first("A"), await first("b"), await first("B")], ["A", "C", "G", "G"]);
  assert.equal(g.contig("chrA").name, "a"); // (ignoring case, the first contig of the name)
  const f = await R.open(SOURCES.url(expected.odd["float-pred2"]));
  assert.ok(bytesOf(await f.read("c", "x-f", 0, 64)).equals(bin("float-pred2.bin")));
  const padded = await R.open(SOURCES.blob(expected.odd.padded));
  assert.equal(
    Array.from(await padded.read("c", "ref", 0, 20))
      .map(R.base)
      .join(""),
    "ACGT".repeat(5),
  );
  const z = Buffer.concat([zlib.deflateSync(Buffer.from("ACGT")), Buffer.alloc(5)]);
  assert.equal(new TextDecoder().decode(await R.inflate(z)), "ACGT"); // (as zlib reads it)
});

test("it loads as a classic script, as an ES module and with require()", async () => {
  const code = fs.readFileSync(READER, "utf8");
  const ctx = vm.createContext({});
  vm.runInContext(code, ctx);
  assert.equal(typeof ctx.GpxilRead.open, "function");
  assert.equal(ctx.GpxilRead.SPEC, "0.3");
  const mjs = path.join(OUT, "gpxil-read.mjs");
  fs.writeFileSync(mjs, code);
  delete globalThis.GpxilRead;
  await import(pathToFileURL(mjs).href);
  assert.equal(typeof globalThis.GpxilRead.open, "function");
  assert.equal(R.SPEC, "0.3");
});

test("what the codes mean", () => {
  assert.equal([0x01, 0x02, 0x04, 0x08, 0x0f, 0x00, 0x11, 0x15].map(R.base).join(""), "ACGTN-ar");
  assert.deepEqual([0, 1, 10, 12, 13, 15, 3].map(R.state), ["REF", "A", "ABSENT", "COMPLEX", "DEL", "UNKNOWN", "reserved 3"]);
  assert.deepEqual(R.flags(0x17), ["PHASED", "INS1", "INS2", "VARIANT_SITE"]);
});
