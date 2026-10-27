# Benchmarks

Scripts behind the measurements in the gpxil
[rationale](https://github.com/comparativechrono/Gpxil/blob/main/docs/rationale.md).

| File | |
|---|---|
| `bench_representations.py` | size, speed, random access and losslessness of image containers, pixel encodings and layouts against FASTA and .2bit |
| `make_figures.py` | the figures of the specification and READMEs |
| `results/chr21-HG002-v5.0q.json` | output of `bench_representations.py` for GRCh38 chr21 with HG002 |
| `results/wholegenome-HG002-GRCh38.json` | size breakdown of a whole-genome HG002 file |

## Data

All public. Checksums are those published by the providers.

```sh
# GRCh38, Ensembl release 116 (soft-masked)
E=https://ftp.ensembl.org/pub/release-116/fasta/homo_sapiens/dna
curl -O $E/Homo_sapiens.GRCh38.dna_sm.chromosome.21.fa.gz        # sum: 38886 12293
curl -O $E/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz     # sum: 19039 920395

# HG002 (Genome in a Bottle) small-variant benchmark v5.0q, GRCh38
G=https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/v5.0q
curl -O $G/HG002_GRCh38_v5.0q_smvar.vcf.gz                       # md5 c71acc71069bf7cd7f51eb8fb0c1a1ab
curl -O $G/HG002_GRCh38_v5.0q_smvar.vcf.gz.tbi
curl -O $G/HG002_GRCh38_v5.0q_smvar.benchmark.bed                 # md5 3366858af85875cbb3c579412ffa4c56
```

## Running

```sh
pip install -e "..[bench]" scikit-image pysam
python -c "import pysam; open('HG002_chr21.vcf','w').write(''.join(l+'\n' for l in pysam.TabixFile('HG002_GRCh38_v5.0q_smvar.vcf.gz').header) + ''.join(r+'\n' for r in pysam.TabixFile('HG002_GRCh38_v5.0q_smvar.vcf.gz').fetch('chr21')))"
python bench_representations.py --fasta Homo_sapiens.GRCh38.dna_sm.chromosome.21.fa.gz \
    --vcf HG002_chr21.vcf --bed HG002_GRCh38_v5.0q_smvar.benchmark.bed --out chr21.json

gpxil encode Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz -v HG002_GRCh38_v5.0q_smvar.vcf.gz \
    --callable HG002_GRCh38_v5.0q_smvar.benchmark.bed --sex XY --ucsc-names --reference-name GRCh38 \
    -o HG002_GRCh38.gpxil.tif
gpxil validate HG002_GRCh38.gpxil.tif
```

The chr21 benchmark takes about 15 minutes on two cores, most of it in the
slow encoders (WebP, xz). The whole-genome encode took 12.6 minutes and
peaked at 2.0 GB of memory (chromosome 1 is processed in one piece); the
validation took 1.5 minutes and 0.23 GB.
