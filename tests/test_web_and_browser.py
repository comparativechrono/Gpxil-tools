"""0.3.0: reading over the web and from file objects, a gpxil file as the reference, VCF regions, and what a
Python without imagecodecs or without sys.stdout.buffer (Pyodide, notebooks) needs."""

from __future__ import annotations

import contextlib
import io
import os
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from simulate import fasta_text, simulate

from gpxiltools import cli
from gpxiltools import codes as C
from gpxiltools import container as K
from gpxiltools.container import GpxilFile, GpxilFormatError, GpxilWriter
from gpxiltools.encode import encode
from gpxiltools.export import iter_vcf_records, write_vcf
from gpxiltools.remote import RangeFile

_RANGE = re.compile(r"bytes=(\d+)-(\d+)$")


class _RangeHandler(SimpleHTTPRequestHandler):
    """Serves files with HTTP range requests, as GitHub Pages does (hide_size: without Content-Range, as a browser
    shows a range from another web site, which hides that header)."""

    ranges = True
    hide_size = False
    log = None

    def send_head(self):
        path = self.translate_path(self.path)
        m = _RANGE.match(self.headers.get("Range", ""))
        if not (self.ranges and m and os.path.isfile(path)):
            return super().send_head()
        size = os.path.getsize(path)
        a, b = int(m.group(1)), min(size - 1, int(m.group(2)))
        self.log.append((a, b))
        fh = open(path, "rb")
        fh.seek(a)
        self.send_response(206)
        self.send_header("Content-Type", "image/tiff")
        if not self.hide_size:
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        self._left = b - a + 1
        return fh

    def copyfile(self, source, outputfile):
        left = getattr(self, "_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        outputfile.write(source.read(left))

    def log_message(self, *a):
        pass


@contextlib.contextmanager
def serve(folder, *, ranges=True, hide_size=False):
    log: list[tuple[int, int]] = []
    handler = type("H", (_RangeHandler,), {"ranges": ranges, "hide_size": hide_size, "log": log})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(handler, directory=str(folder)))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", log
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def sample(tmp_path):
    """A simulated sample on chr1 (with a reference-only chr2), encoded from FASTA."""
    sim = simulate(n=40000, n_sites=400, seed=21)
    fa = tmp_path / "ref.fa"
    fa.write_text(fasta_text({"chr1": sim.reference, "chr2": "ACGTTGCA" * 3000}))
    vcf = tmp_path / "s.vcf"
    vcf.write_text("\n".join(sim.vcf_lines("chr1")) + "\n")
    out = tmp_path / "s.gpxil.tif"
    encode(fa, out, vcf=vcf, width=512, log=lambda m: None)
    return sim, fa, vcf, out


def _same_contents(a: GpxilFile, b: GpxilFile) -> None:
    assert a.contigs == b.contigs
    for name in a.contigs:
        ca, cb = a[name], b[name]
        assert ca.planes == cb.planes
        for p in ca.planes:
            assert np.array_equal(ca.read(p), cb.read(p)), (name, p)
        assert ca.sidecar() == cb.sidecar()


# --------------------------------------------------------------------- the web


def test_a_file_on_a_web_server_reads_as_on_disk(sample):
    _, _, _, out = sample
    with serve(out.parent) as (url, log), GpxilFile(f"{url}/{out.name}") as web, GpxilFile(out) as disk:
        assert web.metadata == disk.metadata
        _same_contents(web, disk)
        assert all(b - a < (1 << 20) for a, b in log), "range requests, not whole-file downloads"


def test_a_region_over_the_web_fetches_only_its_strips(tmp_path):
    rng = np.random.default_rng(5)
    ref = rng.choice(np.frombuffer(b"ACGT", dtype=np.uint8), 10_000_000).tobytes().decode()
    fa = tmp_path / "big.fa"
    fa.write_text(fasta_text({"chrB": ref}))
    out = tmp_path / "big.gpxil.tif"
    encode(fa, out, width=4096, log=lambda m: None)
    size = os.path.getsize(out)
    with serve(tmp_path) as (url, log):
        fh = RangeFile(f"{url}/{out.name}", block=1 << 16)
        with GpxilFile(fh) as g:
            assert g["chrB"].sequence(5_000_000, 5_000_060) == ref[5_000_000:5_000_060]
        assert fh.fetched < size / 8, (fh.fetched, size)
        assert fh.requests == len(log)


def test_a_server_without_range_requests(sample, monkeypatch):
    _, _, _, out = sample
    with serve(out.parent, ranges=False) as (url, _):
        with GpxilFile(f"{url}/{out.name}") as g:  # a small file is downloaded whole
            assert g["chr1"].length == 40000
        monkeypatch.setattr("gpxiltools.remote.WHOLE_FILE_LIMIT", 1000)
        with pytest.raises(OSError, match="range requests"):
            GpxilFile(f"{url}/{out.name}")


def test_a_range_without_the_files_size(sample):
    """In the browser, a range from another web site comes without its Content-Range: the size comes from HEAD."""
    _, _, _, out = sample
    with serve(out.parent, hide_size=True) as (url, log), GpxilFile(f"{url}/{out.name}") as web, GpxilFile(out) as disk:
        assert web._fh.size == os.path.getsize(out)
        _same_contents(web, disk)
        assert all(b - a < (1 << 20) for a, b in log), "range requests, not whole-file downloads"


def test_file_objects(sample):
    _, _, _, out = sample
    with open(out, "rb") as fh, GpxilFile(fh) as g, GpxilFile(out) as disk:
        _same_contents(g, disk)
    with GpxilFile(io.BytesIO(out.read_bytes())) as g:
        assert g.path == "<file object>" and g["chr2"].sequence(0, 8) == "ACGTTGCA"


def test_info_and_validate_take_a_url(sample, capsys):
    _, _, _, out = sample
    with serve(out.parent) as (url, _):
        assert cli.main(["info", f"{url}/{out.name}"]) == 0
        assert cli.main(["validate", f"{url}/{out.name}"]) == 0
    text = capsys.readouterr().out
    assert "chr1" in text and "valid (2 contigs checked, 0 errors" in text


# ---------------------------------------------------------- gpxil as reference


def test_encode_against_a_gpxil_reference(sample, tmp_path):
    sim, fa, vcf, out = sample
    ref_only = tmp_path / "ref.gpxil.tif"
    encode(fa, ref_only, reference_name="SIM1", log=lambda m: None)
    again = tmp_path / "again.gpxil.tif"
    summary = encode(ref_only, again, vcf=vcf, width=512, log=lambda m: None)
    assert summary["contigs"] == 2
    with GpxilFile(out) as a, GpxilFile(again) as b:
        _same_contents(a, b)
        for ea, eb in zip(a.metadata["contigs"], b.metadata["contigs"]):
            assert ea["md5"] == eb["md5"] and ea["refget"] == eb["refget"]
        assert b.metadata["reference"] == {"name": "SIM1", "source": "ref.fa", "via": "ref.gpxil.tif"}
    # only the contigs asked for are read from the reference
    one = tmp_path / "one.gpxil.tif"
    encode(ref_only, one, vcf=vcf, contigs=["chr2"], log=lambda m: None)
    with GpxilFile(one) as g:
        assert g.contigs == ["chr2"]


def test_encode_against_a_gpxil_reference_on_the_web(sample, tmp_path):
    _, fa, vcf, out = sample
    ref_only = tmp_path / "ref.gpxil.tif"
    encode(fa, ref_only, log=lambda m: None)
    again = tmp_path / "web.gpxil.tif"
    with serve(tmp_path) as (url, _):
        encode(f"{url}/ref.gpxil.tif", again, vcf=vcf, width=512, log=lambda m: None)
    with GpxilFile(out) as a, GpxilFile(again) as b:
        _same_contents(a, b)


def test_a_gpxil_file_as_the_reference_of_a_no_ref_file(sample, tmp_path):
    sim, fa, vcf, out = sample
    noref = tmp_path / "noref.gpxil.tif"
    encode(fa, noref, vcf=vcf, include_ref=False, log=lambda m: None)
    with serve(out.parent) as (url, log):
        for reference in (out, f"{url}/{out.name}"):
            with GpxilFile(noref, reference=reference) as g:
                c = g["chr1"]
                assert c.has("ref")
                assert c.sequence(1000, 1100) == sim.reference[1000:1100]
    with GpxilFile(noref) as g:  # and on the command line
        assert not g["chr1"].has("ref")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["fasta", str(noref), "--region", "chr1:1001-1100", "--reference", str(out)]) == 0
    assert buf.getvalue().split("\n", 1)[1].replace("\n", "") == sim.reference[1000:1100]


def test_a_reference_that_does_not_match(sample, tmp_path):
    _, fa, vcf, _ = sample
    noref = tmp_path / "noref.gpxil.tif"
    encode(fa, noref, vcf=vcf, include_ref=False, log=lambda m: None)
    other = tmp_path / "other.fa"
    other.write_text(fasta_text({"chr1": "A" * 40000}))
    wrong = tmp_path / "wrong.gpxil.tif"
    encode(other, wrong, log=lambda m: None)
    with GpxilFile(noref, reference=wrong) as g, pytest.raises(GpxilFormatError, match="does not match"):
        g["chr1"].read("ref", 0, 10)


# ------------------------------------------------------------------ VCF regions


def _overlaps(line: str, s: int, e: int) -> bool:
    f = line.split("\t")
    pos0 = int(f[1]) - 1
    end = pos0 + len(f[3])
    m = re.search(r"END=(\d+)", f[7])
    if m:
        end = int(m.group(1))
    return pos0 < e and end > s


def test_vcf_regions_are_the_whole_contigs_records(tmp_path):
    sim = simulate(n=30000, n_sites=300, seed=9)
    fa = tmp_path / "r.fa"
    fa.write_text(fasta_text({"chr1": sim.reference}))
    vcf = tmp_path / "r.vcf"
    lines = sim.vcf_lines("chr1")
    header, body = lines[:5], lines[5:]
    pos = lambda r: int(r.split("\t")[1])  # noqa: E731
    body = [r for r in body if not (8990 < pos(r) <= 12010 or 19990 < pos(r) <= 20010)]
    # a deletion across the start of a region tested below, and a no-call
    body.append(f"chr1\t9001\t.\t{sim.reference[9000:12001]}\t{sim.reference[9000]}\t50\tPASS\t.\tGT:GQ\t0|1:40")
    body.append(f"chr1\t20001\t.\t{sim.reference[20000]}\t.\t.\tPASS\t.\tGT:GQ\t./.:.")
    vcf.write_text("\n".join(header + sorted(body, key=pos)) + "\n")
    out = tmp_path / "r.gpxil.tif"
    encode(fa, out, vcf=vcf, log=lambda m: None)
    with GpxilFile(out) as g:
        c = g["chr1"]
        whole = list(iter_vcf_records(c, chunk=4096))
        for s, e in ((0, 30000), (10000, 15000), (11999, 12003), (19990, 20010), (2500, 2501), (29000, 30000)):
            part = list(iter_vcf_records(c, chunk=4096, start=s, end=e))
            events = [ln for ln in part if "<*>" not in ln]
            assert events == [ln for ln in whole if "<*>" not in ln and _overlaps(ln, s, e)], (s, e)
            for ln in part:  # no-call blocks are cut at the region's ends
                if "<*>" in ln:
                    pos0 = int(ln.split("\t")[1]) - 1
                    end = int(re.search(r"END=(\d+)", ln).group(1))
                    assert s <= pos0 < end <= e
        # the deletion that starts before 10000 and ends inside it is its own record, whole
        assert any(ln.split("\t")[1] == "9001" for ln in iter_vcf_records(c, start=10000, end=10100))


def test_vcf_region_on_the_command_line(sample, tmp_path):
    _, _, _, out = sample
    dest = tmp_path / "region.vcf"
    assert cli.main(["vcf", str(out), "--region", "chr1:1-20000", "-o", str(dest)]) == 0
    text = dest.read_text()
    assert "##contig=<ID=chr1,length=40000>" in text and "##contig=<ID=chr2" not in text
    assert all(int(ln.split("\t")[1]) <= 20000 for ln in text.splitlines() if not ln.startswith("#"))
    with pytest.raises(SystemExit):
        cli.main(["vcf", str(out), "--region", "chr1", "--contigs", "chr1"])
    buf = io.StringIO()
    with GpxilFile(out) as g:
        write_vcf(g, buf, regions=[("chr1", 0, None)])
    assert buf.getvalue().count("\n") > 100


# ------------------------------------------------ Pythons without some things


def test_fasta_to_a_console_without_a_byte_buffer(sample):
    sim, _, _, out = sample
    buf = io.StringIO()  # notebooks and the browser's Python give text streams
    with contextlib.redirect_stdout(buf):
        assert cli.main(["fasta", str(out), "--region", "chr1:101-160"]) == 0
    assert buf.getvalue().splitlines()[1] == sim.reference[100:160]


def test_missing_codecs_are_said_plainly(tmp_path, monkeypatch):
    assert K.codec_available("deflate") and K.codec_available("none")
    monkeypatch.setattr(K, "codec_available", lambda c: c != "lzw")
    with pytest.raises(GpxilFormatError, match="imagecodecs"):
        GpxilWriter(tmp_path / "x.gpxil.tif", compression="lzw")

    class Page:
        compression = type("Comp", (), {"name": "ZSTD"})()

        def decode(self, *a):
            raise ImportError("No module named 'compression'")

        def asarray(self):
            raise KeyError("<COMPRESSION.ZSTD: 50000> requires the 'imagecodecs' package")

    with pytest.raises(GpxilFormatError, match="compressed with ZSTD.*imagecodecs"):
        K._decode(Page(), b"", 0)
    with pytest.raises(GpxilFormatError, match="imagecodecs"):
        K._asarray(Page())


def test_imagecodecs_is_not_needed_in_the_browser():
    # (the dependency's line of pyproject.toml, read as text: tomllib is Python 3.11 and later)
    with open(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"), encoding="utf-8") as fh:
        (codecs,) = re.findall(r'^\s*"(imagecodecs[^"]*)"', fh.read(), re.M)
    assert "sys_platform != 'emscripten'" in codecs


def test_gt_states_survive_the_web(sample):
    """The genotype plane read over the web decodes to the same haplotype states."""
    _, _, _, out = sample
    with serve(out.parent) as (url, _), GpxilFile(f"{url}/{out.name}") as g:
        gt = g["chr1"].read("gt")
        assert set(np.unique(C.hap1(gt))) <= set(range(16))
