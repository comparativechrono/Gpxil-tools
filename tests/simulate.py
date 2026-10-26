"""Synthetic genomes with known haplotypes, for end-to-end tests.

Events are placed far enough apart that no two VCF records overlap, so the
true haplotype sequences can be built independently of gpxiltools by plain
string splicing. That makes them an oracle for the encoder and exporter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

BASES = "ACGT"


def random_reference(n: int, rng: random.Random) -> str:
    seq = [rng.choice(BASES) for _ in range(n)]
    # a soft-masked stretch, a homopolymer run, a dinucleotide repeat and an N gap
    for i in range(n // 5, n // 5 + n // 10):
        seq[i] = seq[i].lower()
    for i in range(n // 2, n // 2 + 15):
        seq[i] = "A"
    for k, i in enumerate(range(n // 2 + 40, n // 2 + 80)):
        seq[i] = "AC"[k % 2]
    for i in range(n - n // 20, n - n // 20 + 30):
        seq[i] = "N"
    return "".join(seq)


@dataclass
class Site:
    pos0: int  # 0-based start of REF
    ref: str
    alts: list[str]
    gt: str
    hap_alleles: list[str | None]  # allele string per haplotype (None = no call)


@dataclass
class Simulation:
    reference: str
    sites: list[Site] = field(default_factory=list)

    def vcf_lines(self, chrom: str, sample: str = "SIM") -> list[str]:
        header = [
            "##fileformat=VCFv4.3",
            f"##contig=<ID={chrom},length={len(self.reference)}>",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">',
            f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}",
        ]
        body = []
        for i, s in enumerate(self.sites):
            filt = "LowQual" if i % 17 == 5 else "PASS"
            body.append(f"{chrom}\t{s.pos0 + 1}\t.\t{s.ref}\t{','.join(s.alts)}\t50\t{filt}\t.\tGT:GQ\t{s.gt}:{(i * 7) % 99}")
        return header + body

    def haplotype(self, h: int) -> str:
        """True sequence of haplotype h (1 or 2), uppercase, no-calls as N per REF base."""
        out = []
        prev = 0
        ref = self.reference.upper()
        for s in self.sites:
            out.append(ref[prev : s.pos0])
            allele = s.hap_alleles[h - 1]
            out.append("N" * len(s.ref) if allele is None else allele)
            prev = s.pos0 + len(s.ref)
        out.append(ref[prev:])
        return "".join(out)


def _other(base: str, rng: random.Random, avoid: set[str] = frozenset()) -> str:
    return rng.choice([b for b in BASES if b != base.upper() and b not in avoid])


def simulate(n: int = 20000, n_sites: int = 300, seed: int = 1, phased: bool = True, nocalls: bool = True) -> Simulation:
    rng = random.Random(seed)
    ref = random_reference(n, rng)
    up = ref.upper()
    sim = Simulation(ref)
    spacing = max(30, n // (n_sites + 1))
    pos = 25
    sep = "|" if phased else "/"
    while pos < n - 40 and len(sim.sites) < n_sites:
        if "N" in up[pos - 12 : pos + 14]:
            pos += spacing
            continue
        kind = rng.choice(["snv", "snv", "snv", "mnp", "del", "ins", "complex", "multi"])
        zyg = rng.choice(["10", "01", "11", "11", "10", "01"])
        if kind == "snv":
            r = up[pos]
            alt = _other(r, rng)
            s_ref, alts = r, [alt]
        elif kind == "mnp":
            k = rng.randint(2, 3)
            r = up[pos : pos + k]
            alt = "".join(_other(b, rng) for b in r)
            s_ref, alts = r, [alt]
        elif kind == "del":
            k = rng.randint(1, 10)
            r = up[pos - 1 : pos + k]
            s_ref, alts = r, [r[0]]
            pos -= 1
        elif kind == "ins":
            k = rng.randint(1, 10)
            r = up[pos]
            ins = "".join(rng.choice(BASES) for _ in range(k))
            s_ref, alts = r, [r + ins]
        elif kind == "complex":
            r = up[pos : pos + 2]
            alt = _other(r[0], rng) + "".join(rng.choice(BASES) for _ in range(2))
            s_ref, alts = r, [alt]
        else:  # multi: two different SNV alleles at one site
            r = up[pos]
            a1 = _other(r, rng)
            a2 = _other(r, rng, {a1})
            s_ref, alts = r, [a1, a2]
            zyg = "12" if rng.random() < 0.5 else "21"
        idx = list(zyg)
        if nocalls and kind == "snv" and rng.random() < 0.1:
            idx[rng.randint(0, 1)] = "."
        hap_alleles = []
        for a in idx:
            if a == ".":
                hap_alleles.append(None)
            elif a == "0":
                hap_alleles.append(s_ref)
            else:
                hap_alleles.append(alts[int(a) - 1])
        sim.sites.append(Site(pos, s_ref, alts, sep.join(idx), hap_alleles))
        pos += len(s_ref) + spacing + rng.randint(0, spacing // 2)
    return sim


def fasta_text(records: dict[str, str], width: int = 60) -> str:
    out = []
    for name, seq in records.items():
        out.append(f">{name} simulated")
        out.extend(seq[i : i + width] for i in range(0, len(seq), width))
    return "\n".join(out) + "\n"
