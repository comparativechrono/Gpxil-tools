"""Mapping genome positions to pixels.

Storage always uses the raster layout (spec section 4): position ``p`` of a
contig with layout width ``W`` is pixel ``(row, col) = divmod(p, W)``, rows
top to bottom, like lines of text. Any other layout is a view computed from
it. The Hilbert curve view is provided because it keeps neighbouring bases
neighbouring in two dimensions, which suits image-analysis filters; a
2^k x 2^k block of a Hilbert image is always a contiguous run of 4^k bases.
"""

from __future__ import annotations

import math

import numpy as np


def position_to_pixel(pos, width: int):
    """0-based position(s) -> (row, col) in a raster of the given width."""
    pos = np.asarray(pos)
    return pos // width, pos % width


def pixel_to_position(row, col, width: int):
    return np.asarray(row) * width + np.asarray(col)


def raster(values: np.ndarray, width: int, fill=0) -> np.ndarray:
    """1-D per-base array -> 2-D raster image, padded with ``fill``."""
    n = values.shape[0]
    rows = max(1, math.ceil(n / width))
    out = np.full(rows * width, fill, dtype=values.dtype)
    out[:n] = values
    return out.reshape(rows, width)


def hilbert_order(length: int) -> int:
    """Smallest k with 4**k >= length."""
    return max(0, math.ceil(math.log(max(length, 1), 4) - 1e-12))


def hilbert_d2xy(order: int, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert-curve index -> (x, y) for a 2**order square (vectorised)."""
    t = np.asarray(d, dtype=np.int64).copy()
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    s = 1
    n = 1 << order
    while s < n:
        rx = (t >> 1) & 1
        ry = (t ^ rx) & 1
        swap = ry == 0
        flip = swap & (rx == 1)
        x = np.where(flip, s - 1 - x, x)
        y = np.where(flip, s - 1 - y, y)
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    return x, y


def hilbert_xy2d(order: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(x, y) -> Hilbert-curve index for a 2**order square (vectorised)."""
    x = np.asarray(x, dtype=np.int64).copy()
    y = np.asarray(y, dtype=np.int64).copy()
    n = 1 << order
    d = np.zeros_like(x)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        swap = ry == 0
        flip = swap & (rx == 1)
        x = np.where(flip, n - 1 - x, x)
        y = np.where(flip, n - 1 - y, y)
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        s >>= 1
    return d


def hilbert_image(values: np.ndarray, order: int | None = None, fill=0, chunk: int = 1 << 24) -> tuple[np.ndarray, int]:
    """Lay a 1-D per-base array out along a Hilbert curve.

    Returns ``(image, order)`` where image is ``2**order`` square. Positions
    beyond the end of ``values`` are filled with ``fill``.
    """
    n = values.shape[0]
    k = hilbert_order(n) if order is None else order
    side = 1 << k
    if side * side < n:
        raise ValueError(f"order {k} too small for {n} values")
    img = np.full((side, side), fill, dtype=values.dtype)
    for start in range(0, n, chunk):
        d = np.arange(start, min(n, start + chunk), dtype=np.int64)
        x, y = hilbert_d2xy(k, d)
        img[y, x] = values[start : start + d.size]
    return img, k


def from_hilbert_image(img: np.ndarray, length: int, chunk: int = 1 << 24) -> np.ndarray:
    """Inverse of :func:`hilbert_image`."""
    side = img.shape[0]
    k = int(round(math.log2(side)))
    out = np.empty(length, dtype=img.dtype)
    for start in range(0, length, chunk):
        d = np.arange(start, min(length, start + chunk), dtype=np.int64)
        x, y = hilbert_d2xy(k, d)
        out[start : start + d.size] = img[y, x]
    return out
