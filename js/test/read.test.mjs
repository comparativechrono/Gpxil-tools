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

test("windows: across rows and strips, clipped at either end, by alias", async () => {
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

test("files it must refuse, each with a code a page can put in its own words", async () => {
  for (const [name, want] of Object.entries(expected.bad)) {
    const source = SOURCES.bytes(want.file);
    if (want.at === "open") {
      await assert.rejects(R.open(source), (e) => e.code === want.code, name);
      continue;
    }
    const g = await R.open(source);
    await assert.rejects(g.read(want.contig, want.plane, 0, 100), (e) => e.code === want.code && e.compression === want.compression, name);
    assert.equal((await g.read(want.contig, "ref", 0, 100)).length, 100, `${name}: its other pages still read`);
  }
  await assert.rejects(R.open(R.fromUrl(base + "no-such-file.gpxil.tif")), { code: "http" });
});

test("it loads as a classic script, as an ES module and with require()", async () => {
  const code = fs.readFileSync(READER, "utf8");
  const ctx = vm.createContext({});
  vm.runInContext(code, ctx);
  assert.equal(typeof ctx.GpxilRead.open, "function");
  assert.equal(ctx.GpxilRead.SPEC, "0.2");
  const mjs = path.join(OUT, "gpxil-read.mjs");
  fs.writeFileSync(mjs, code);
  delete globalThis.GpxilRead;
  await import(pathToFileURL(mjs).href);
  assert.equal(typeof globalThis.GpxilRead.open, "function");
  assert.equal(R.SPEC, "0.2");
});

test("what the codes mean", () => {
  assert.equal([0x01, 0x02, 0x04, 0x08, 0x0f, 0x00, 0x11, 0x15].map(R.base).join(""), "ACGTN-ar");
  assert.deepEqual([0, 1, 10, 12, 13, 15, 3].map(R.state), ["REF", "A", "ABSENT", "COMPLEX", "DEL", "UNKNOWN", "reserved 3"]);
  assert.deepEqual(R.flags(0x17), ["PHASED", "INS1", "INS2", "VARIANT_SITE"]);
});
