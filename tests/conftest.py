import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from simulate import fasta_text, simulate  # noqa: E402


@pytest.fixture(scope="session")
def sim():
    return simulate(n=20000, n_sites=300, seed=7)


@pytest.fixture(scope="session")
def sim_unphased():
    return simulate(n=12000, n_sites=150, seed=11, phased=False)


@pytest.fixture
def sim_files(tmp_path, sim):
    """FASTA with two contigs ('1' simulated diploid, '2' plain) and a VCF for contig 1 named 'chr1'."""
    fa = tmp_path / "ref.fa"
    other = "ACGT" * 2500
    fa.write_text(fasta_text({"1": sim.reference, "2": other}))
    vcf = tmp_path / "sample.vcf"
    vcf.write_text("\n".join(sim.vcf_lines("chr1")) + "\n")
    return fa, vcf


@pytest.fixture
def datadir():
    """The tests/data directory (the spec's worked example)."""
    from pathlib import Path

    return Path(__file__).parent / "data"
