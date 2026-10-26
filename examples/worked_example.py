"""Build the worked example of the gpxil specification (SPEC.md appendix B).

Writes worked-example.fa, worked-example.vcf, worked-example.gpxil.tif and
worked-example.expected.json into the given directory (default: here).
"""

import json
import os
import sys

from gpxiltools import codes as C
from gpxiltools.container import GpxilFile
from gpxiltools.encode import encode
from gpxiltools.export import haplotype_sequence

FASTA = ">chrT worked example from the gpxil specification, appendix B\nACGTACGTac\n"
VCF = (
    "##fileformat=VCFv4.3\n"
    "##contig=<ID=chrT,length=10>\n"
    '##FILTER=<ID=LowQ,Description="Low quality">\n'
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tEXAMPLE\n"
    "chrT\t2\t.\tC\tT\t.\tPASS\t.\tGT\t0|1\n"
    "chrT\t4\t.\tTAC\tT\t.\tPASS\t.\tGT\t1|0\n"
    "chrT\t8\t.\tT\tTGG\t.\tPASS\t.\tGT\t1|1\n"
    "chrT\t9\t.\tA\tG\t.\tLowQ\t.\tGT\t./1\n"
)


def main(out_dir="."):
    fa = os.path.join(out_dir, "worked-example.fa")
    vcf = os.path.join(out_dir, "worked-example.vcf")
    tif = os.path.join(out_dir, "worked-example.gpxil.tif")
    open(fa, "w").write(FASTA)
    open(vcf, "w").write(VCF)
    encode(fa, tif, vcf=vcf, width=4, reference_name="worked example", log=lambda m: None)
    with GpxilFile(tif) as g:
        c = g["chrT"]
        hexes = lambda a: [f"0x{int(v):02x}" for v in a]  # noqa: E731
        expected = {
            "description": "Expected decoded content of worked-example.gpxil.tif (gpxil SPEC.md appendix B). Offsets are 0-based.",
            "contig": c.name,
            "length": c.length,
            "width": c.width,
            "planes": {p: hexes(c.read(p)) for p in c.planes},
            "images_including_padding": {p: [hexes(row) for row in c.image(p)] for p in c.planes},
            "sidecar": [{"offset": e.offset, "hap": e.hap, "kind": e.kind, "data": e.data} for e in c.sidecar()],
            "haplotype_1": haplotype_sequence(c, 1).decode(),
            "haplotype_2": haplotype_sequence(c, 2).decode(),
            "haplotype_1_uppercase": haplotype_sequence(c, 1, keep_case=False).decode(),
            "haplotype_2_uppercase": haplotype_sequence(c, 2, keep_case=False).decode(),
            "gt_meaning": [C.describe_gt_byte(int(v)) for v in c.read("gt")],
        }
    with open(os.path.join(out_dir, "worked-example.expected.json"), "w") as fh:
        json.dump(expected, fh, indent=1)
    print(json.dumps(expected["planes"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
