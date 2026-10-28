# Changes

## 0.5.0 (28 October 2026)

Genomes compared. Files are unchanged.

- **`gpxil compare A B`** compares two genomes base by base, as gpxil's planes
  are aligned to the reference: no VCF records are matched. At each base
  called in both, genotypes are compared as unordered pairs of alleles, so
  phasing isn't compared and one copy equals two of the same allele. Where
  they differ, bases no more than `--gap` (50) apart, widened over any
  repeat their insertions and deletions could move along, form a stretch,
  compared as sequences, with its heterozygous pieces either way round: an
  indel written at another place in a repeat, or a substitution written
  with a deletion, is *the same, written differently*. Each file's events
  (substitutions, indels, COMPLEX alleles, counted as `gpxil stats` counts
  them) are found in the other, missing, there with another genotype, or not
  compared where the other didn't call; with B as the truth that gives
  precision, recall and F1. It lists the first differences.
- **`gpxil compare CHILD --father F --mother M`** checks a child against both
  parents: Mendelian errors (a stretch is first checked as sequences, so
  variants written differently aren't errors), candidate new mutations,
  where each of the child's variants came from (the father, the mother,
  both, or can't be told), and, where a parent's calls are phased, which of
  that parent's haplotypes the child received and the crossovers between
  them.
- Calls outside the regions called with confidence (a file written with a
  BED file keeps them, each an island among bases not called) are left out
  of both, unless `--all-calls`.
- `-o` draws the first region, one pixel a base, its classes over the
  reference faded. `--json` gives every count to programs.
- In Python: `gpxiltools.compare.compare_pair`, `compare_trio`, `picture`
  and `format_compare`. Chromosome 17 takes about 4 s for two genomes and
  6 s for a trio, reading a megabase at a time.
- Checked against VCF records, independently of gpxiltools, on GIAB's HG002
  (v4.2.1 against v5.0q) and the HG002-HG003-HG004 trio, and on 806 indels
  moved along repeats ([Gpxil-scratchpad, experiment 03](https://github.com/comparativechrono/gpxil-scratchpad/tree/main/experiments/03-compare)).

## 0.4.0 (28 October 2026)

Two commands for working with big files. Files are unchanged.

- **`gpxil subset`** copies some contigs, and if you like only some planes,
  of a gpxil file into a new one: a working copy, such as chromosome 17 out
  of a whole genome, without encoding anything again. The strips are copied
  as they are stored, without being decoded, so it is fast, needs about one
  strip of memory, and works on files on the web (adjacent strips are read
  in one request). `--no-ref` leaves out the reference plane (the contigs
  keep their digests); `--planes` keeps some planes; `--reference` puts the
  reference plane back into a file written without it; `--compression` and
  `--level` encode the strips again. The copy keeps each contig's metadata,
  and records what it was copied from (`subset` in the metadata). It
  refuses to leave out the `ps` plane of a contig with several phase sets,
  which would then read as one, and a region of a contig, which gpxil 0.1
  cannot store. In Python: `gpxiltools.subset.subset`, and
  `GpxilWriter.copy_contig`.
- **`gpxil stats`** counts what a file holds, for every contig or for
  regions: the reference's bases (GC, N, soft-masked), bases called, not
  called and partly called, heterozygous and homozygous substitutions with
  transitions and transversions, insertions and deletions (events and
  bases), COMPLEX alleles, the share of heterozygous substitutions phased
  and the phase sets (N50 and the largest), and genotype quality and depth
  (quartiles). It reads a piece at a time, so a whole genome needs little
  memory; regions add up to the whole. `--json` gives the counts to
  programs. In Python: `gpxiltools.stats.stats` and `region_tally`.

## 0.3.1 (28 October 2026)

- **N is near-black** in the `classic` and `okabe-ito` palettes, and is no
  longer faded when soft-masked. In `classic`, a mid-grey N was exactly the
  mean of C (blue) and T (yellow), so in the genotype view a heterozygous C/T
  site, the commonest kind, looked like a base that wasn't known; in
  `okabe-ito`, a grey N was close to a C/T site in a repeat. Every palette's
  N now differs from every base and mixture, faded or not, and a test says
  so. CMYK: N is 90 % black in both. A book's page on how to read it names
  N's colour. Files are unchanged: palettes are rendering choices, so only
  the colour maps stored for image viewers in newly written files differ.
- `gpxil palettes` lists N's colours with the bases'.

## 0.3.0 (27 October 2026)

For big files, files on the web and Python in the browser, as GpxilStudio
needs them. Files written by 0.2.0 and 0.3.0 are the same.

- **Files on web servers.** `GpxilFile("https://…")`, and every command
  (`gpxil info https://…`), reads a file on a web server with HTTP range
  requests, fetching only the strips that a region needs
  (`gpxiltools.remote.RangeFile`). A server that does not answer range
  requests is used only for small files.
- **File objects.** `GpxilFile` also takes an open binary file, such as
  `io.BytesIO`.
- **A gpxil file as the reference.** `gpxil encode` takes a gpxil file (path
  or URL) with the reference plane in place of the FASTA, reading only the
  contigs it encodes; the file records where the reference came from
  (`reference.via`). `--reference` and `GpxilFile(..., reference=...)` take
  one for files written with `--no-ref`, and read just the regions needed,
  after checking the digests.
- **`gpxil vcf --region`** (repeatable): the records whose windows overlap the
  region, exactly as in the whole contig's VCF (a deletion that starts before
  the region is written whole); no-call blocks are cut at the region's ends.
- **imagecodecs** is no longer installed in Pyodide, which has no build of it.
  Only Zstandard, LZW and PackBits need it; a missing codec is now reported
  plainly when writing or reading, and the tests of those codecs are skipped
  without it.
- `gpxil fasta` writes to consoles whose `sys.stdout` has no byte buffer
  (notebooks, the browser).
- A server whose range answers don't say the file's size (as a browser shows
  a range from another web site, hiding `Content-Range`) is asked for it with
  a HEAD request.
- PNG pictures are written without `Image.fromarray(…, mode=…)`, which
  Pillow 11.3 deprecates.
- `gpxil encode` notes how many records' REF alleles don't match the
  reference, which is what a VCF called against another reference (GRCh37 on
  GRCh38) looks like. The count was in the metadata only.

## 0.2.0 (27 October 2026)

Fixes from a review of 0.1.0 against the specification, each with a
regression test in `tests/test_review_regressions.py`. Planes encoded from
ordinary VCFs (such as the HG002 benchmark) are unchanged; the fixes concern
edge cases.

VCF conversion

- A missing genotype (`.`) no longer erases another record's variant; it is
  UNKNOWN only where nothing else was called.
- Unphased genotypes with a `*` allele put the spanning deletion on the
  haplotype that carries it.
- VCF 4.4 genotypes with a leading phase indicator (`|1`) are read.
- `SVLEN` sets the span of `<DEL>` alleles only, using each ALT allele's own
  value; the `SVLEN` of an `<INS>` no longer stretches the record over the
  following bases.
- Single breakends (`G.`) are COMPLEX; breakend mate names keep their case.
- Inserted bases other than A, C, G, T, N become N and are counted.
- IUPAC REF alleles are checked by set intersection (R matches A or G).
- `GQ`, `DP` and `PS` planes are created when the field appears anywhere in
  the VCF, not just in its first records, and overlapping calls resolve them
  field by field.
- Phase-set identifiers that are not integers no longer collide with integer
  ones.
- A COMPLEX allele replaced by a later call loses its sidecar entry; two
  COMPLEX alleles on one base count as a conflict.
- Contigs without calls or callable regions are stored reference only,
  instead of claiming the reference genotype.
- Haploid contigs and `ploidy` are recorded correctly in the metadata;
  diploid calls in haploid regions are counted.
- Callable and haploid BED files may use spaces.
- A sequence allele in a record that also has a symbolic allele covers the
  record's whole span; a `<DEL>` keeps its padding base; an ALT base equal
  to the reference base is REF before precedence is applied; an insertion
  counts as a change of its anchor when overlapping calls resolve
  `GQ`/`DP`/`PS`.
- Missing genotypes leave an ABSENT haplotype 2 unasserted; `|.` is a
  haploid no-call; negative allele numbers are counted, not wrapped.
- The REF check rejects gaps and non-IUPAC characters; ALT `U` reads as T.
- Phase sets are numbered at first appearance, identifiers that end up on
  no base are dropped, and integer identifiers compare as integers.
- A callable BED that has no interval on a contig counts as none; with
  `--no-ref` such contigs are listed as `contigs_without_data`.
- A file that ends up without genotype planes has `sample: null`.

Files and tools

- FASTA names end at a tab as well as a space; empty records are skipped.
- Contig names must be printable ASCII; failed encodes leave no partial file.
- Writers refuse genotype planes without sample metadata and `gq`/`dp`/`ps`
  without a `gt` plane.
- BigTIFF is chosen by size: classic TIFF unless the file would pass 4 GiB,
  in which case the encode restarts as BigTIFF (0.1.0 used BigTIFF for every
  uncompressed file and never for a compressed one).
- `gpxil validate` checks much more (page order, photometric tags, page
  descriptions, sidecar headers, order and duplicates, COMPLEX entries, `ps`
  against PHASED, plane descriptions, `sample` against `gt`, refget without
  md5) and works chunk by chunk.
- `gpxil fasta` streams whole chromosomes; `fasta`, `vcf`, `render` and
  `book` take `--reference` for files written with `--no-ref`.
- VCF export: merged windows are phased only if every site is; `##contig`
  lines use the file's contig names; reserved states read as unknown.
- Errors on the command line are one-line messages instead of tracebacks.

## 0.1.0 (26 October 2026)

First release.
