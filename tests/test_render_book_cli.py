import numpy as np
import pytest
import tifffile

from gpxiltools import codes as C
from gpxiltools.book import BookLayout, estimate_pages, make_book
from gpxiltools.cli import main
from gpxiltools.container import GpxilFile
from gpxiltools.fasta import encode_sequence
from gpxiltools.layout import from_hilbert_image, hilbert_d2xy, hilbert_image, hilbert_xy2d
from gpxiltools.palettes import PALETTES, ref_colours
from gpxiltools.render import render_arrays


def test_genotype_mode_colours():
    ref = encode_sequence(b"AAAc")
    gt = np.array([0x00, 0x04, 0x44, 0x00], dtype=np.uint8)  # ref, het A/G, hom G, masked ref
    r = render_arrays(ref, gt, mode="genotype", palette="classic", bases_per_row=4)
    rgb_lut, _ = ref_colours("classic")
    assert r.index.tolist() == [[C.BASE_A, C.BASE_A | C.BASE_G, C.BASE_G, C.BASE_C | C.SOFTMASK]]
    assert tuple(r.to_rgb()[0, 1]) == tuple(rgb_lut[5])  # mean of red and green
    assert r.to_cmyk().shape == (1, 4, 4)


def test_process_palette_is_one_ink_per_base():
    _, cmyk = ref_colours("process")
    assert cmyk[C.BASE_A].tolist() == [255, 0, 0, 0]
    assert cmyk[C.BASE_T].tolist() == [0, 0, 0, 255]
    # a heterozygous A/G site overprints cyan and yellow
    assert cmyk[C.BASE_A | C.BASE_G].tolist() == [255, 0, 255, 0]


def test_haplotypes_mode_interleaves_rows():
    ref = encode_sequence(b"ACGTACGT")
    gt = np.zeros(8, dtype=np.uint8)
    gt[1] = (C.HAP_DEL << 4) | C.BASE_T
    r = render_arrays(ref, gt, mode="haplotypes", bases_per_row=4)
    assert r.index.shape == (4, 4)
    assert r.index[0, 1] == C.NO_BASE and r.index[1, 1] == C.BASE_T


def test_variants_mode_classes():
    ref = encode_sequence(b"ACGTACGT")
    gt = np.array([0, 0x01, 0x22, 0xD0, 0, 0xF0, 0, 0], dtype=np.uint8)
    r = render_arrays(ref, gt, np.zeros(8, np.uint8), mode="variants", bases_per_row=8)
    labels = dict((idx, lab) for lab, idx in r.legend)
    assert labels[r.index[0, 1]] == "heterozygous"
    assert labels[r.index[0, 2]] == "homozygous alternate"
    assert labels[r.index[0, 3]] == "deletion"
    assert labels[r.index[0, 5]] == "no call"


def test_all_palettes_complete():
    for name in PALETTES:
        rgb, cmyk = ref_colours(name)
        assert rgb.shape == (256, 3) and cmyk.shape == (256, 4)


def _lab(rgb):
    """sRGB (0-255) to CIELAB (D65)"""
    c = np.asarray(rgb, float) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def test_n_is_never_a_mixture():
    # N (not known) must not look like any base or mixture, faded or not: in the classic palette a mid-grey N was
    # exactly the mean of C (blue) and T (yellow), so heterozygous C/T sites read as unknown
    for name, pal in PALETTES.items():
        rgb, cmyk = ref_colours(name)
        n = rgb[C.BASE_N]
        assert (rgb[C.BASE_N | C.SOFTMASK] == n).all() and (cmyk[C.BASE_N | C.SOFTMASK] == cmyk[C.BASE_N]).all()  # not faded
        others = [k for k in range(32) if k & 15 != C.BASE_N]
        assert not any((rgb[k] == n).all() for k in others), name
        if pal.mix == "mean":  # (clearly apart: CIE76 colour difference)
            dist = min(float(np.linalg.norm(_lab(rgb[k]) - _lab(n))) for k in others)
            assert dist > 20, f"{name}: N is only {dist:.1f} from another code"
    classic, _ = ref_colours("classic")
    assert classic[C.BASE_C | C.BASE_T].tolist() == [128, 128, 128] and classic[C.BASE_N].tolist() == [32, 32, 32]


def test_hilbert_properties():
    for k in (1, 3, 4):
        n = 1 << k
        d = np.arange(n * n)
        x, y = hilbert_d2xy(k, d)
        assert (np.abs(np.diff(x)) + np.abs(np.diff(y)) == 1).all()  # neighbours stay neighbours
        assert np.array_equal(hilbert_xy2d(k, x, y), d)
    v = np.arange(1000, dtype=np.int32)
    img, k = hilbert_image(v)
    assert k == 5 and np.array_equal(from_hilbert_image(img, 1000), v)


@pytest.fixture
def encoded(tmp_path, sim_files):
    fa, vcf = sim_files
    out = tmp_path / "sim.gpxil.tif"
    assert main(["encode", str(fa), "-v", str(vcf), "-o", str(out), "--width", "500"]) == 0
    return out


def test_cli_end_to_end(tmp_path, encoded, capsys):
    assert main(["info", str(encoded)]) == 0
    assert "chr1" not in capsys.readouterr().out.split("contig")[0]  # contig names come from the FASTA
    assert main(["validate", str(encoded)]) == 0
    png = tmp_path / "r.png"
    assert main(["render", str(encoded), "1:1-5000", "-o", str(png), "--mode", "haplotypes", "--width", "100", "--scale", "2"]) == 0
    from PIL import Image

    assert Image.open(png).size == (200, 200)  # 50 rows of bases x 2 haplotypes x scale 2
    tif = tmp_path / "r.tif"
    assert main(["render", str(encoded), "chr1:1-10000", "-o", str(tif), "--cmyk", "--dpi", "300"]) == 0
    with tifffile.TiffFile(tif) as tf:
        assert tf.pages[0].photometric == 5 and tf.pages[0].shape[-1] == 4  # separated (CMYK)
    hil = tmp_path / "h.png"
    assert main(["render", str(encoded), "1", "-o", str(hil), "--layout", "hilbert", "--mode", "reference"]) == 0
    fa = tmp_path / "h1.fa"
    assert main(["fasta", str(encoded), "--which", "1", "--region", "1:100-200", "-o", str(fa)]) == 0
    assert fa.read_text().startswith(">1:100-200_hap1\n")
    vcf = tmp_path / "out.vcf"
    assert main(["vcf", str(encoded), "-o", str(vcf)]) == 0
    assert "#CHROM" in vcf.read_text()
    pdf = tmp_path / "b.pdf"
    assert main(["book", str(encoded), "-o", str(pdf), "--page", "A5", "--dpi", "72", "--rgb"]) == 0
    assert pdf.read_bytes().startswith(b"%PDF-1.4")


def test_encode_notes_a_vcf_on_another_reference(tmp_path, capsys):
    fa = tmp_path / "r.fa"
    fa.write_text(">c1\nACGTACGTAC\n")
    vcf = tmp_path / "v.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.3\n##contig=<ID=c1,length=10>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
        "c1\t2\t.\tC\tT\t.\tPASS\t.\tGT\t0/1\n"  # REF matches
        "c1\t3\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\n"  # the reference has G
        "c1\t5\t.\tC\tG\t.\tPASS\t.\tGT\t1/1\n"  # the reference has A
    )
    assert main(["encode", str(fa), "-v", str(vcf), "-o", str(tmp_path / "v.gpxil.tif")]) == 0
    err = capsys.readouterr().err
    assert "note: 2 records' REF alleles don't match the reference" in err
    vcf.write_text(vcf.read_text().replace("c1\t5\t.\tC\tG", "c1\t5\t.\tA\tG"))
    assert main(["encode", str(fa), "-v", str(vcf), "-o", str(tmp_path / "w.gpxil.tif")]) == 0
    assert "note: 1 record's REF allele doesn't match" in capsys.readouterr().err


def test_book_structure(tmp_path, encoded):
    pikepdf = pytest.importorskip("pikepdf")
    out = tmp_path / "book.pdf"
    layout = BookLayout(page="A5", dpi=100, margin_mm=10)
    with GpxilFile(encoded) as g:
        info = make_book(g, out, layout=layout, colorspace="cmyk")
    pdf = pikepdf.open(out)
    assert len(pdf.pages) == info["pages"]
    page = pdf.pages[3]  # cover, legend, chapter, first data page
    im = page.Resources.XObject["/Im"]
    assert int(im.Width) == info["bases_per_row"] and str(im.ColorSpace[1]) == "/DeviceCMYK"
    assert estimate_pages(3_100_000_000, 25) > 400
