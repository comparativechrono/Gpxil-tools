"""Annotation planes (``gpxil annotate``): a copy of a gpxil file with an extension plane made from features.

An annotation drawn as a plane (spec 5.6) has a value at every base, so it
lines up with the genome base for base, travels with it, and is read, cut
out (``gpxil subset``) and drawn as any plane is. ``annotate`` copies a file
(its strips as they are stored) and adds one plane, ``x-NAME``, from a BED
file or a RepeatMasker table:

- ``category``: each feature's label a number, 1, 2, 3 ... in the order of
  the labels' names (0: no feature), the labels kept in the plane's
  metadata (``labels``); a repeat's label is its class, its family (the
  default: ``SINE/Alu``) or its name (``SINE/Alu/AluY``), as ``level`` says;
  a BED feature's, its name. uint8 for up to 255 labels, else uint16.
- ``presence``: 1 where a feature is, 0 elsewhere (uint8).
- ``score``: the feature's score (BED's fifth column; a repeat's divergence
  from its consensus, %), float32, NaN where there is no feature.
- ``value``: a bedGraph's values (the fourth column: replication timing,
  a Hi-C compartment's eigenvector, a signal), float32, NaN where none.

Where features overlap, the shorter one is drawn (nested features show),
and of two as long, the later in the file. ``auto`` (the default) takes
``category`` for a RepeatMasker table, ``value`` for a BED file whose fourth
column is all numbers (a bedGraph), ``category`` for one with names, and
``presence`` for one without.

Positions are the sequence's, so a window's plane holds the features over
its own stretch of the sequence (spec 3.1).
"""

from __future__ import annotations

import heapq
import os
import sys
import time
from collections.abc import Callable

import numpy as np

from . import __version__
from .container import GpxilFile, GpxilWriter, path_name
from .features import Feature, layout_of, read_features, sequence_name

KINDS = ("auto", "category", "presence", "score", "value")
LEVELS = ("class", "family", "name")


def label_of(name: str, level: str, rmsk: bool) -> str:
    """A feature's label: a BED feature's name; a repeat's class, family or name (its name is class/family/name, or
    class/name for a repeat whose family is its class)."""
    if not rmsk or level == "name":
        return name
    parts = name.split("/")
    if level == "class":
        return parts[0]
    return "/".join(parts[:-1]) or parts[0]


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def segments(features: list[tuple[int, int, int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features ``(start, end, order, value)`` of one sequence as segments that don't overlap, ``(starts, ends,
    values)`` in order: where features overlap, the value of the shortest, and of two as long, the one later in
    ``order``; segments next to each other with the same value are joined."""
    feats = [f for f in features if f[1] > f[0]]
    if not feats:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0, dtype=np.float64)
    bounds = sorted({x for f in feats for x in (f[0], f[1])})
    by_start = sorted(range(len(feats)), key=lambda i: feats[i][0])
    heap: list = []
    out_s, out_e, out_v = [], [], []
    k = 0
    for a, b in zip(bounds, bounds[1:]):
        while k < len(by_start) and feats[by_start[k]][0] <= a:
            i = by_start[k]
            s, e, order, v = feats[i]
            heapq.heappush(heap, (e - s, -order, e, v))
            k += 1
        while heap and heap[0][2] <= a:
            heapq.heappop(heap)  # (ended: dropped once it reaches the top)
        if not heap:
            continue
        v = heap[0][3]
        if out_e and out_e[-1] == a and out_v[-1] == v:
            out_e[-1] = b
        else:
            out_s.append(a)
            out_e.append(b)
            out_v.append(v)
    return np.array(out_s, dtype=np.int64), np.array(out_e, dtype=np.int64), np.array(out_v, dtype=np.float64)


class Painter:
    """A plane's values for any stretch of a sequence, from its segments (see :func:`segments`)."""

    def __init__(self, starts: np.ndarray, ends: np.ndarray, values: np.ndarray, dtype, none):
        self.starts, self.ends, self.values = starts, ends, values
        self.dtype, self.none = np.dtype(dtype), none

    def __call__(self, a: int, b: int) -> np.ndarray:
        out = np.full(b - a, self.none, dtype=self.dtype)
        i, j = int(np.searchsorted(self.ends, a, side="right")), int(np.searchsorted(self.starts, b, side="left"))
        for s, e, v in zip(self.starts[i:j].tolist(), self.ends[i:j].tolist(), self.values[i:j].tolist()):
            out[max(0, s - a) : min(b, e) - a] = v
        return out


def plane_from(gfile, features: list[Feature], *, kind: str = "auto", level: str = "family", rmsk: bool = False) -> dict:
    """What a plane made of the features holds: ``kind`` (as decided, for ``auto``), ``dtype``, ``none`` (the value
    where there is no feature), ``labels`` (category: number -> label), and ``painters``: a :class:`Painter` for each
    sequence of the file with features (by the file's name for it); and ``unplaced``, how many features are on no
    sequence of the file."""
    if kind not in KINDS:
        raise ValueError(f"--as is one of {', '.join(KINDS)}")
    if level not in LEVELS:
        raise ValueError(f"--level is one of {', '.join(LEVELS)}")
    if kind == "auto":
        if rmsk:
            kind = "category"
        elif features and all(_is_number(f.name) for f in features):
            kind = "value"
        elif any(f.name for f in features):
            kind = "category"
        else:
            kind = "presence"
    labels: dict[str, int] = {}
    if kind == "category":
        names = sorted({label_of(f.name, level, rmsk) for f in features})
        if "" in names:
            names.remove("")
        labels = {x: i for i, x in enumerate(names, 1)}
        dtype = np.uint8 if len(labels) <= 255 else np.uint16 if len(labels) <= 65535 else np.uint32
        none = 0
    elif kind == "presence":
        dtype, none = np.uint8, 0
    else:
        dtype, none = np.float32, np.nan
    by_seq: dict[str, list] = {}
    names_of: dict[str, str | None] = {}
    unplaced = 0
    for i, f in enumerate(features):
        if f.contig not in names_of:
            names_of[f.contig] = sequence_name(gfile, f.contig)
        seq = names_of[f.contig]
        if seq is None:
            unplaced += 1
            continue
        if kind == "category":
            lab = label_of(f.name, level, rmsk)
            if not lab:
                continue
            v = labels[lab]
        elif kind == "presence":
            v = 1
        elif kind == "score":
            if f.score is None:
                continue
            v = f.score
        else:
            if not _is_number(f.name):
                raise ValueError(f"{f.contig}:{f.start + 1}-{f.end}: {f.name!r} is not a number (--as value reads the fourth column)")
            v = float(f.name)
        by_seq.setdefault(seq, []).append((f.start, f.end, i, v))
    painters = {seq: Painter(*segments(fs), dtype, none) for seq, fs in by_seq.items()}
    return {
        "kind": kind,
        "dtype": np.dtype(dtype),
        "none": none,
        "labels": {v: k for k, v in labels.items()},
        "painters": painters,
        "unplaced": unplaced,
    }


def annotate(
    path,
    out,
    features_path,
    *,
    plane: str,
    kind: str = "auto",
    level: str = "family",
    layout: str = "auto",
    match: str | None = None,
    min_length: int = 0,
    description: str | None = None,
    units: str | None = None,
    replace: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Copy the gpxil file ``path`` to ``out`` with an extension plane ``plane`` (``x-…``) made from the features of
    ``features_path`` (see the module's notes). Returns a summary."""
    from .container import _EXT_NAME

    log = log or (lambda msg: print(msg, file=sys.stderr))
    t0 = time.time()
    if not _EXT_NAME.match(plane):
        raise ValueError(f"{plane!r}: name the plane x- and letters, digits, - or _ (x-repeats)")
    if os.path.exists(out) and not str(path).startswith(("http://", "https://")) and os.path.samefile(path, out):
        raise ValueError(f"{out} is the input file: write the copy to another file")
    layout = layout_of(features_path) if layout == "auto" else layout
    features = read_features(features_path, layout, match=match, min_length=min_length)
    with GpxilFile(path) as g:
        if any(plane in g[n].pages for n in g.contigs) and not replace:
            raise ValueError(f"{path_name(g.path)} has a {plane} plane already: give --replace to make it again")
        made = plane_from(g, features, kind=kind, level=level, rmsk=layout == "rmsk")
        meta = g.metadata
        from .subset import _OWN_KEYS

        writer = GpxilWriter(
            out,
            width=int((meta.get("layout") or {}).get("width") or g[g.contigs[0]].width),
            compression=_compression(g),
            palette=meta.get("palette") or "classic",
            reference=meta.get("reference") or {},
            sample=meta.get("sample"),
            metadata={k: v for k, v in meta.items() if k not in _OWN_KEYS},
            require_codec=False,
            version=str(meta["gpxil"]),  # (an extension plane is in every version: the planes copied keep theirs)
        )
        writer.plane_info = {p: {k: v for k, v in (d or {}).items() if k != "dtype"} for p, d in (meta.get("planes") or {}).items()}
        what = {
            "category": "the features' labels, by number (labels)",
            "presence": "1 where a feature is, 0 elsewhere",
            "score": "the features' scores" + (" (a repeat's divergence from its consensus, %)" if layout == "rmsk" else ""),
            "value": "the features' values (a bedGraph's fourth column)",
        }[made["kind"]]
        info = {
            "description": description or f"{what}, from {os.path.basename(str(features_path))}",
            "kind": made["kind"],
            "from": os.path.basename(str(features_path)),
        }
        if units:
            info["units"] = units
        if made["kind"] == "category":
            info["labels"] = {str(k): v for k, v in sorted(made["labels"].items())}
            info["missing_means"] = "0: no feature"
        if made["kind"] in ("score", "value"):
            info["missing_means"] = "NaN: no feature"
        if layout == "rmsk" and made["kind"] == "category":
            info["level"] = level
        writer.plane_info[plane] = info
        before = [a for a in meta.get("annotations") or [] if isinstance(a, dict) and a.get("plane") != plane]
        writer.extra_metadata["annotations"] = before + [
            {
                "plane": plane,
                "from": info["from"],
                "kind": made["kind"],
                "features": len(features),
                "generator": f"gpxiltools {__version__}",
            }
        ]
        dt = made["dtype"]
        try:
            for name in g.contigs:
                c = g[name]
                painter = made["painters"].get(c.parent)
                if painter is None:
                    none = made["none"]

                    def values(a, b, none=none):
                        return np.full(b - a, none, dtype=dt)

                else:

                    def values(a, b, painter=painter, off=c.start):
                        return painter(off + a, off + b)

                keep = [p for p in c.planes if p != plane]
                got = writer.copy_contig(c, keep, add={plane: (dt, values)})
                log(f"  {got.name}: {','.join(got.pages)}  [{time.time() - t0:.0f}s]")
            writer.close()
        except BaseException:
            writer.abort()
            raise
    painted = sum(int((p.ends - p.starts).sum()) for p in made["painters"].values())
    return {
        "plane": plane,
        "kind": made["kind"],
        "dtype": dt.name,
        "labels": len(made["labels"]),
        "features": len(features),
        "unplaced": made["unplaced"],
        "bases_with_a_feature": painted,
        "contigs": len(writer.contigs),
        "seconds": round(time.time() - t0, 1),
        "bytes": os.path.getsize(out),
    }


def _compression(g: GpxilFile) -> str:
    from .container import DEFAULT_COMPRESSION, compression_of

    first = g[g.contigs[0]]
    return compression_of(first._page(first.planes[0])) or DEFAULT_COMPRESSION
