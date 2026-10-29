# Changes

## 0.8.1 (29 October 2026)

Two fixes found putting 0.8.0's tools in GpxilStudio.

- **Repeats whose own names hold a /** (GRCh38's ALR/Alpha and BSR/Beta, of
  the satellites): a repeat's label as a class or family, in `gpxil
  annotate`, is now its table's class and family, not read off its name, so
  the 3.9 Mb of alpha satellite at chromosome 17's centromere are
  `Satellite/centr`, not `Satellite/centr/ALR`. A RepeatMasker table's
  features keep their class, family and name in `Feature.fields`.
- **Pictures without Pillow's warning**: `gpxil stack`'s pictures give
  `Image.fromarray` no mode, which Pillow 11.3 deprecates (and Pillow 13
  removes), so it no longer warns where warnings are shown.

## 0.8.0 (29 October 2026)

Tools for discovery, made for (and checked against) the analyses of
Albrecht-Buehler's GA-sequences in
[Gpxil-scratchpad](https://github.com/comparativechrono/Gpxil-scratchpad)
(experiments 04 and 05).

- **`gpxil runs FILE --of AG --min 51`**: the runs of a composition, the
  longest stretches whose bases all belong to a set given in IUPAC letters
  (AG or R: Albrecht-Buehler's pure GA-sequences; CT, A, AT ...), as BED,
  with a summary (by strand, lengths, runs a megabase). `--both-strands`
  finds the set's runs on the other strand too (strand `-`); a set that is its
  own complement has one set of runs (strand `.`). On GRCh38's chr1: 1,766
  GA-sequences and 1,848 TC-sequences, as experiment 04 counted them.
- **`gpxil stack FILE FEATURES`**: features stacked, one a row, lined up at
  their starts, ends or centres, each read 5' to 3' on its own strand (a
  sample's substitutions complemented and its insertions moved with it):
  Albrecht-Buehler's GA-complex image, for any features. Sorted as given, by
  position, length, score, name, or the run of a set of bases at the anchor
  (his poly(A)); drawn as `gpxil render` draws bases (any mode), raw or
  labelled, a pixel row a feature or the mean of many (`--bin`); each
  column's bases (and the sample's variants) as a table (`--profile`), and
  the features in the order drawn as BED (`--rows`). The chr1 stack of
  GA-sequences is experiment 04's, pixel for pixel.
- **`gpxil spectrum FILE FEATURES`**: a sample's substitutions in regions
  against the rest of the genome (or a background of its own), with each
  region base's expected count from the background's rate of each
  substitution in its context (3 bases by default: the 96 kinds of
  mutational signatures; 1 to 9), read with a pyrimidine or a purine in the
  middle; observed over expected with an exact binomial interval and test
  (the background's counts are a sample too); by kind, and on each feature's
  own strand with `--stranded` (12 kinds); transitions and transversions;
  heterozygous and homozygous alleles; insertions and deletions per called
  base; only isolated substitutions with `--isolated N`. Counts the bases the
  file calls, leaving out calls outside its confident regions as `gpxil
  compare` does, narrowed by `--callable`. On HG002 (v5.0q), its isolated
  substitutions in GA-sequences are experiment 04's, kind by kind.
- **`gpxil annotate FILE FEATURES --plane x-NAME -o OUT`**: a copy of a file
  (its strips copied as stored) with an extension plane made from features,
  a value a base: a RepeatMasker table's repeats as categories (class,
  family or name, the labels in the plane's metadata), a BED file's names,
  presence, scores, or a bedGraph's values (replication timing, Hi-C
  eigenvectors) as float32 (NaN where there are none). Where features
  overlap, the shorter is drawn. A file's annotations are recorded in its
  metadata (`annotations`). chr1 with RepeatMasker's families: 37 s, 1.8 MB
  more.
- **`gpxil skew FILE`**: strand symmetry in windows at multiples of the
  window along each sequence: GC content; GC, AT, purine and keto skews; the
  cumulative GC skew; and for each k how far k-mers are from their reverse
  complements' counts (Chargaff's second parity rule). `--unmasked` leaves
  out repeats; `--features` counts only bases in features, `--stranded` on
  their own strands (genes: transcription's skews). A table, totals as JSON,
  and a map with a track a measure for each sequence.
- **`gpxil period FILE REGION`**: a stretch wrapped at its repeat period (the
  shortest that fits, or `--period`), a unit a row, the reference beside the
  sample's haplotypes, aligned (substitutions outlined, deletions dashed,
  insertions marked and listed) or as sequences; as text, or a picture.
- Features (`gpxiltools.features`): BED and RepeatMasker tables (UCSC's
  rmsk, with or without its bin column, and RepeatMasker's `.out`), read
  plain or gzipped and chosen by name and length as they are read, so a
  genome's 5.6 million repeats need little memory; `Intervals`, merged and
  cut into masks quickly.
- `GpxilWriter.copy_contig(..., add={name: (dtype, values)})` adds computed
  extension planes while copying a contig, in the specification's page
  order; `render.variant_classes()` gives each base's class as the
  variants mode draws it.
- `gpxil validate` takes NaN as the same as NaN where it compares the planes
  of windows that overlap (a float32 plane's "no value").
- The new commands read regions as `gpxil stats` reads them, each base once:
  a sequence named that only windows hold is all of them, joined where they
  overlap or touch, and regions that overlap are scanned as their union.

## 0.7.0 (28 October 2026)

Version 0.3 of the specification: windows.

- **Windows** (specification section 3.1): a contig can hold a stretch of a
  longer sequence, its parent, and say where in it the stretch lies, so that
  a gene and its surroundings travel in a file of a few kilobytes and keep
  the chromosome's positions. TP53 and 100 kb either side, out of HG002's
  chromosome 17, is 76 kB (12 kB without the reference plane), against
  24.6 MB for the chromosome. A window holds what the whole chromosome's
  planes hold over its stretch; it is named after the region it holds
  (`chr17:7561779-7787546`), and its metadata entry's `window` key names the
  parent, where the window starts in it, and the parent's length, digests
  and other names.
- **`gpxil subset FILE chr17:7,561,779-7,787,546`** cuts a window out of a
  whole contig, or out of another window, reading and encoding again a strip
  at a time, so it works on a whole genome on the web in a second or two. A
  contig or a window named whole is copied as it is stored, as before; the
  name of a sequence that a file holds windows of copies them all. A file
  with windows is gpxil 0.3, and its `subset` record gives the version of the
  file it copied, whose rules its planes follow.
- **`gpxil encode --region chr17:7,561,779-7,787,546`** (repeatable) makes
  windows straight from a VCF, converting only the calls that join up with
  the window through overlapping spans, over the stretch they cover: which
  gives what converting the whole chromosome would. Checked against cutting
  the same windows out of whole chromosomes, on 116 windows of HG002's
  chromosome 17 whose edges cut deletions and insertions, and on dense
  simulated calls of every kind (overlapping indels, symbolic deletions and
  inversions, breakends, reference blocks, `*` alleles, haploid and missing
  genotypes, phase sets): the planes and sidecars are the same, base for
  base, phase sets too. Only the contigs asked for (with `--contigs` too)
  are taken out of the VCF. A gpxil file of windows can't be the reference
  (it holds stretches of its sequences): encode says so.
- **Phase-set identifiers that aren't numbers** (VCF `PS` should be one) are
  numbered, as the specification now requires (section 5.5), from all the
  phased calls of the contig, whichever are converted, so that a window's
  values are its whole parent's, and every window's of one parent alike. A
  number is written in the digits 0 to 9, and compared by value (`077` is
  77): `+77` or `²` is an identifier like any other.
- **Positions on a window are its parent's, in every command**:
  `GpxilFile.locate(name, start, end)` finds a region of a sequence in the
  contig that holds it (a window of just that stretch, or the first that
  holds it all), so `stats`, `render`, `fasta`, `vcf` and `compare` take
  `chr17:7,661,779-7,687,546` on a window file. Names are found as
  gpxil-read.js finds them (`GpxilFile.find`): as written, else a whole
  contig ignoring case, with or without `chr`; a window by its name as
  written, so `17` isn't a window `chr17:1-500`. `GpxilFile.region_of()`
  reads a region given as text as the CLI and gpxil-read.js do, and
  `locate()` takes one. `gpxil info`, `render`, `book` and `stats` label
  windows by their parent's positions. `Contig.window`, `.parent`, `.start`,
  `.end`, `.region()`, and `GpxilFile.windows` and `.windows_of()` say where
  contigs lie. A name ends at the last colon of a region, so a window's name
  can carry a region: `chr17:7561779-7787546:7,661,779-7,687,546`.
- **Windows that overlap or touch are read as one stretch** of their
  sequence (`Stretch`, `GpxilFile.stretches()`): `gpxil vcf` writes the
  records of one window of all the stretch, each once, so a deletion that
  one window cuts and the next holds is written whole (on 116 windows of
  HG002's chromosome 17, in 41 stretches, the 860 records are those of one
  window per stretch); `gpxil compare` compares each base once, and `gpxil
  stats` counts each base once in all. A region is read through the
  stretch that holds it, in every command (`GpxilFile.view()`), so a
  region in one window ends as the region of one window of all the stretch
  would, and a region that windows hold only together is found in them
  (`locate(..., join=True)`). Windows may have different planes: each base
  keeps the planes of the windows that hold it, and where none with the
  sample's planes does, nothing is known of it; so a region found first in
  a window of the reference only still has the calls of a window beside
  it. The reference is read from the windows that have it, where they hold
  the bases (`Stretch.covers()`). A window's copy of a COMPLEX entry at its
  first base (specification section 3.1) is told from the parent's own
  entry, and left out. A sequence named alone is all its windows in
  `stats`, `vcf` and `book`. A deletion at a window's first base is
  written on the base before the window, as N, so that the window's VCF
  encodes back to it (specification section 11.4).
- Regions: `CHROM:POS` is one base; numbers are the digits 0 to 9, with
  commas or spaces, and don't begin with 0, so a name with colons in it
  (`HLA-A*01:01:01:01`) stays a name; a region from position 0 is an error.
  A file's contig named like a region (`HLA-A*01:14`, or even `HLA:0-5`),
  or given with spaces around it, is that contig, and lists of names
  (`--contigs`, `subset`) split at commas but those that group a region's
  digits. Positions go up to 2^53 − 1. gpxil-read.js reads regions the
  same way.
- A window without its reference plane reads its bases from its parent in
  the reference given (a FASTA, a gpxil file of the chromosome, or of a
  window that holds it), checked against its own digests or its parent's.
- `gpxil validate` checks windows: their `window` key, the file's version,
  that a window holds less than all of its parent, that windows of one
  parent agree where they overlap, and, where the file holds the parent
  too, that a window holds what the parent does. The schema of version 0.3
  is bundled. A whole number in a `window` key may be written `2.0`, as
  JSON Schema allows.
- A copy of a copy keeps, in its `subset` record, the version whose rules
  made its planes. A window without the sample's planes doesn't name the
  VCF contig. Windows without their reference plane that give only their
  parent's digests check a FASTA's chromosome once for all of them.
- Files are written as gpxil 0.3. Readers of 0.2 read them, and a window as
  an ordinary contig. Files of 0.1 and 0.2 read as before.
- `js/gpxil-read.js` implements 0.3: a contig's `window`, `parent`, `start`
  and `end`; `locate()` finds regions as gpxiltools does (the tests check it
  against gpxiltools' answers); `parseRegion()` and `regionText()`; windows
  whose metadata says wrong things of where they are are refused.

## 0.6.0 (28 October 2026)

Version 0.2 of the specification, and a reader in JavaScript.

- **gpxil 0.2: unphased calls that overlap go one to each haplotype**
  (specification section 6.5). Calls that overlap are alternatives at one
  place. An ALT allele *occupies* the bases it changes, and the base before
  a plain insertion or deletion; in an unphased genotype of two different
  alleles, an allele that doesn't fit the haplotype the written order gives
  it (an earlier call's ALT allele occupies one of its bases there) goes to
  the other haplotype, if both alleles fit the other way round. So two
  overlapping heterozygous deletions no longer run together on haplotype 2,
  nor does an insertion follow a base another call deletes; split
  multi-allelic records, unphased, go one to each haplotype in either order
  (unless their alleles change different bases of one REF), and a record
  written twice stays heterozygous, whichever way its genotype is written
  (the same allele goes to the haplotype that has it). Counted as
  `overlapping_calls_reassigned` (each contig's `encoding`, the summary, and
  a note from `gpxil encode`). On chromosome 17 of GIAB's v4.2.1 benchmarks
  of HG002, HG003 and HG004 it moves 26 of 320,884 records (on their 22
  autosomes, 687 of 12,079,785); `gpxil compare` then agrees with a check
  from the VCF records on every stretch where HG002's two benchmarks
  differ, and the trio's artefactual Mendelian error is gone.
- Smaller fixes to the VCF conversion. Allele numbers are digits: `²`, a
  digit to Python, stopped the encoder with an error, and ` 1` or `+1` now
  count as `bad_allele_index`, as `-1` did. A one-base ALT allele other than
  A, C, G or T at a reference N is UNKNOWN, as a longer one was, not REF. A
  record with an empty REF is skipped (`skipped_out_of_range`): its
  insertion had no anchor, and could land before the contig. Alleles of
  `Call`s made in Python are read case-insensitively, as the VCF's are.
- Files are written as gpxil 0.2. `gpxil subset` keeps the version of the
  file it copies, any 0.x, whose planes were made by that version's rules.
  The validator knows 0.1 and 0.2, with a schema for each, and checks a
  later 0.x as 0.2; `GpxilWriter(version=...)` writes another.
- **`js/gpxil-read.js`**: a reader of gpxil files for web pages and Node,
  from Blobs, bytes or the web (range requests, adjacent strips in one
  request, decoded strips cached), with the sidecar; any version 0.x, in
  strips or tiles, Deflate or none, with or without a predictor, classic
  TIFF or BigTIFF, either byte order, planes of 8, 16 or 32 bits. Files are
  untrusted: it refuses directories that loop, strips stored in more bytes
  than they could need, that inflate past their size or hold less than
  their rows, and tags, strips or sidecars larger than it takes, each with
  an error code a page can put in its own words; directories one after
  another are walked from the bytes already read. Tested in CI against the
  files gpxiltools writes, and files made to hurt it, after two independent
  reviews. The Gpxil browser and GpxilStudio each keep a copy
  (`js/copy_to.py`); it is not on npm.

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
- `-o` draws a region (`--draw`, or the first compared), one pixel a base,
  its classes over the reference faded. `--json` gives every count to
  programs.
- In Python: `gpxiltools.compare.compare_pair`, `compare_trio`, `picture`
  and `format_compare`. Chromosome 17 takes 2 to 3 s, for two genomes or a
  trio (about 10 s in the browser), reading a megabase at a time.
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
