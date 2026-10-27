# Changes

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
