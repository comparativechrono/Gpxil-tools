"""gpxiltools: read, write and render gpxil genome pixel images.

A gpxil file stores a genome as images: one pixel per base, one page per
chromosome and plane (reference sequence, genotype, flags, ...), in an
ordinary multi-page TIFF. See https://github.com/comparativechrono/Gpxil for
the format specification.
"""

__version__ = "0.4.0"

from .container import GpxilFile, GpxilWriter, open_gpxil  # noqa: E402

__all__ = ["GpxilFile", "GpxilWriter", "open_gpxil", "__version__"]
