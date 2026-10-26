"""Generate examples/quickstart.ipynb (run from the repository root).

    python examples/make_quickstart_notebook.py [--execute]
"""

import sys

import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(
    """
# gpxil quick start

Turn a chromosome and one person's genotypes into a gpxil genome pixel image,
look at it, analyse it as an image, and print it.

Data (all public): GRCh38 chromosome 21 from Ensembl, and the Genome in a Bottle
HG002 small-variant benchmark (v5.0q). Runs in Google Colab or any Python 3.10+.
"""
)
code(
    """
import importlib.util

if importlib.util.find_spec("gpxiltools") is None:
    %pip install -q git+https://github.com/comparativechrono/Gpxil-tools scikit-image pysam
"""
)
md("## 1. Download the data")
code(
    """
import os
import urllib.request

os.makedirs("data", exist_ok=True)
ENSEMBL = "https://ftp.ensembl.org/pub/release-116/fasta/homo_sapiens/dna"
GIAB = "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/v5.0q"
files = {
    "chr21.fa.gz": f"{ENSEMBL}/Homo_sapiens.GRCh38.dna_sm.chromosome.21.fa.gz",
    "HG002.vcf.gz": f"{GIAB}/HG002_GRCh38_v5.0q_smvar.vcf.gz",
    "HG002.vcf.gz.tbi": f"{GIAB}/HG002_GRCh38_v5.0q_smvar.vcf.gz.tbi",
    "HG002.benchmark.bed": f"{GIAB}/HG002_GRCh38_v5.0q_smvar.benchmark.bed",
}
for name, url in files.items():
    path = os.path.join("data", name)
    if not os.path.exists(path):
        print("downloading", name)
        urllib.request.urlretrieve(url, path)
print(sorted(os.listdir("data")))
"""
)
md("Keep only chromosome 21 of the VCF (the index makes this quick).")
code(
    """
import pysam

pysam.set_verbosity(0)  # the GIAB index is older than its VCF; harmless
vcf = pysam.TabixFile("data/HG002.vcf.gz")
with open("data/HG002_chr21.vcf", "w") as out:
    out.writelines(h + "\\n" for h in vcf.header)
    out.writelines(r + "\\n" for r in vcf.fetch("chr21"))
"""
)
md(
    """
## 2. Encode

The FASTA calls the chromosome `21`, the VCF `chr21`; gpxiltools matches them.
`--callable` marks everything outside the benchmark regions as unknown rather
than reference.
"""
)
code(
    """
!gpxil encode data/chr21.fa.gz -v data/HG002_chr21.vcf --callable data/HG002.benchmark.bed --ucsc-names --reference-name GRCh38 -o HG002_chr21.gpxil.tif
!gpxil info HG002_chr21.gpxil.tif
!gpxil validate HG002_chr21.gpxil.tif
"""
)
md(
    """
The result is an ordinary TIFF: open it in any image viewer and the first page
is chromosome 21, one pixel per base. In Python every plane is an array.
"""
)
code(
    """
import numpy as np
from gpxiltools import GpxilFile, codes

g = GpxilFile("HG002_chr21.gpxil.tif")
chr21 = g["chr21"]
print(chr21, chr21.width, "bases per row,", chr21.rows, "rows")
start = 42_863_616                               # 0-based offset = VCF POS - 1
ref = chr21.read("ref", start, start + 1024)      # one uint8 per base
gt = chr21.read("gt", start, start + 1024)        # haplotype 1 in the high nibble, 2 in the low
print(chr21.sequence(start, start + 64))
for i in np.flatnonzero(gt)[:12]:
    print(f"chr21:{start + i + 1:,}  ref {codes.IUPAC[ref[i] & 15]}  gt 0x{gt[i]:02x}  {codes.describe_gt_byte(int(gt[i]))}")
"""
)
md("## 3. Look at it")
code(
    """
import matplotlib.pyplot as plt
from gpxiltools.render import render

def show(r, title, scale=1, size=(6, 6)):
    plt.figure(figsize=size)
    plt.imshow(r.scaled(scale).to_rgb(), interpolation="nearest")
    plt.title(title, loc="left", fontsize=10)
    plt.axis("off")
    plt.show()

show(render(chr21, start=30_000_000, end=30_250_000, bases_per_row=500), "250 kb of HG002 chr21, one pixel per base")
"""
)
md(
    """
Zoom in on 1 kb. In **genotype** mode a heterozygous site is drawn in
the mix of its two alleles' colours; **haplotypes** mode draws the two inherited
copies as separate rows; **variants** mode highlights where this genome differs
from the reference.
"""
)
code(
    """
start, bpr, rows = 42_863_616, 64, 16
for mode in ("genotype", "haplotypes", "variants"):
    r = render(chr21, mode=mode, start=start, end=start + bpr * rows, bases_per_row=bpr, mark_insertions=True)
    show(r, mode, scale=6, size=(8, 4 if mode != "haplotypes" else 7))
"""
)
md("The same region in the four built-in palettes; each is defined in sRGB for screens and CMYK for print.")
code(
    """
from gpxiltools.palettes import PALETTES
for name in PALETTES:
    show(render(chr21, palette=name, start=30_100_000, end=30_100_000 + 160 * 40, bases_per_row=160), name, scale=3, size=(8, 2.5))
"""
)
md("## 4. Analyse it as an image")
code(
    """
from skimage.measure import block_reduce

whole_ref = chr21.image("ref")      # (rows, 8192)
whole_gt = chr21.image("gt")
base = whole_ref & 0x0F
acgt = np.isin(base, codes.HAP_SUB_CODES)
gc = np.isin(base, (codes.BASE_C, codes.BASE_G))
variant = (whole_gt != 0) & (whole_gt != codes.GT_UNKNOWN)
blk = (16, 128)
n = block_reduce(acgt, blk, np.sum)
gc_map = np.where(n > 0, block_reduce(gc, blk, np.sum) / np.maximum(n, 1), np.nan)
var_map = np.where(n > 0, block_reduce(variant, blk, np.sum) / np.maximum(n, 1) * 1000, np.nan)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(gc_map, cmap="Blues", interpolation="nearest", aspect="auto")
ax[0].set_title("GC fraction")
ax[1].imshow(var_map, cmap="Oranges", interpolation="nearest", aspect="auto", vmax=np.nanpercentile(var_map, 98))
ax[1].set_title("variant bases per kb")
for a in ax:
    a.axis("off")
plt.show()
"""
)
md("A Hilbert-curve view keeps neighbouring bases together in 2-D: GC-rich and N-rich regions become blocks.")
code(
    """
!gpxil render HG002_chr21.gpxil.tif chr21 -o chr21-hilbert.png --layout hilbert --mode reference
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
plt.figure(figsize=(6, 6))
plt.imshow(Image.open("chr21-hilbert.png").convert("RGB").resize((1024, 1024), Image.BOX))
plt.axis("off")
plt.show()
"""
)
md(
    """
## 5. Print it

A book with one printed dot per base: A4 at 300 dpi holds 6.3 million bases a
page, so chromosome 21 is 8 pages (the whole genome is about 500). The colours
are device CMYK; `--rgb` makes a screen version.
"""
)
code(
    """
!gpxil book HG002_chr21.gpxil.tif -o HG002_chr21_book.pdf --contigs chr21
!gpxil render HG002_chr21.gpxil.tif chr21:30,000,001-30,250,000 -o region-cmyk.tif --width 500 --cmyk --dpi 300
"""
)
md("## 6. Back to sequences and VCF")
code(
    """
!gpxil fasta HG002_chr21.gpxil.tif --which 1 --region chr21:42,863,617-42,864,640 -o hap1.fa
!gpxil fasta HG002_chr21.gpxil.tif --which 2 --region chr21:42,863,617-42,864,640 -o hap2.fa
!gpxil vcf HG002_chr21.gpxil.tif -o HG002_chr21.from-gpxil.vcf
print(open("hap1.fa").read()[:300])
"""
)
md(
    """
A gpxil file of a person, and every faithful picture or print of it, is that
person's genome. HG002 is a consented, public reference genome; treat anyone
else's with the care you would give their VCF.
"""
)

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}})

if __name__ == "__main__":
    path = "examples/quickstart.ipynb"
    if "--execute" in sys.argv:
        from nbclient import NotebookClient

        client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "examples/run"}})
        client.execute()
    nbf.write(nb, path)
    print("wrote", path)
