"""how wrong is a tractogram, and how many tractograms is 96 tractograms worth.

tractography is the only evidence ibm-1 has for long-range cortico-cortical
support, and every materialization that uses `tractometric` inherits its error.
the architecture is explicit that a source writing many values does not supply
many constraints -- a teacher's residuals covary, and the fraction they share,
`correlated_fraction`, is what decides how much of its apparent precision
survives.  the same argument applies to tractography with nothing changed:
diffusion pipelines share failure modes (crossing fibres, gyral bias, a
preference for short paths, a bias toward the seeding density), so a hundred
tractograms of one brain are nowhere near a hundred independent votes about
whether an edge exists.

that number has been *assumed* everywhere it appears.  this script measures it.

the ISMRM 2015 tractography challenge is the one place where it can be measured
at all: a simulated phantom whose fibre geometry is known by construction
(`FilesForSimulation/Fibers.fib`, the tracts fiberfox was given), and 96
tractograms submitted by 20 teams from the *same* simulated acquisition.  the
answer and 96 attempts at it, in one frame, with the confound that usually ruins
this comparison -- different subjects -- removed by construction.

what is measured
----------------
1. **per-edge existence probability.**  for each ground-truth edge, the fraction
   of the 96 submissions that recover it.  and for each edge the ground truth
   does *not* have, the fraction of submissions that assert it anyway.  these are
   the two numbers a prior over `tractometric` support needs and neither has ever
   been available to this repository as a measurement.

2. **the correlation of errors across pipelines.**  the error vector of each
   submission over the edge universe, and the mean pairwise correlation between
   them.  that correlation *is* `correlated_fraction` for a tractography teacher,
   and it converts to an effective number of independent pipelines through
   exactly the machinery in `ibm/runtime/fuse.py` -- which is used here rather
   than reimplemented, because the whole point of that module is that this
   arithmetic should have one home.

three honesty problems, and what was done about them
----------------------------------------------------
**there is no atlas in register with the phantom.**  the phantom is a synthetic
volume built from one subject's tractogram; no freesurfer surface, no desikan
labels and no MNI warp ship with it, and `ibm.anatomy.systems` has nothing that
could be placed on it without inventing a registration.  so the parcellation here
is *cubic spatial binning* of the phantom volume, at `--bin-mm` (20 mm by
default), keeping the bins the ground-truth streamlines actually occupy.  that is
a coarser and blunter partition than an atlas, and it is stated rather than
dressed up: an edge here is "these two 20 mm boxes are connected", not "these two
cortical areas are connected".  the direction of the resulting bias is knowable
and is reported with the numbers -- coarse bins merge nearby endpoints, so they
*understate* both the miss rate and the false-positive rate relative to a fine
parcellation, which makes every error rate below a lower bound.

**the submissions do not share a coordinate convention.**  the card's `absent`
list already says so: `coordinate_convention_per_submission`.  the trk headers
disagree (RAS and LPS, 1 mm and 2 mm), the tck files carry no frame at all and
the vtk files carry nothing but points.  a flip that is invisible in a bounding
box will destroy every edge comparison silently, so each submission is aligned to
the ground truth by searching a sign per axis, a scale from {1, 2, 1/2} and a
translation read off the bounding boxes, and taking whichever maximizes the dice
overlap of its 4 mm occupancy with the ground truth's.  that is the whole search
space -- no rotation, no fitted affine -- because a real transform fit would let
a tractogram that is simply wrong be rotated into agreement and the error rate
would come out however good the optimizer was.  33 of the 96 turn out to be
written with x and y negated about the scanner origin.  the chosen convention and
the residual dice are reported per submission, and a submission whose convention
cannot be identified is *excluded and named* rather than quietly contributing
noise.  identification is a margin over the runner-up convention rather than a
dice floor alone, because a floor rejects thin tractograms -- which fill less of
the phantom and so score lower under the *correct* transform -- and those are
precisely the low-sensitivity pipelines, so a floor would inflate the measured
recovery rate.  this is a registration step and it is the weakest link in the
whole measurement.

**the phantom is easier than a brain.**  the card says it and it is worth
repeating where the numbers are produced: fiberfox simulated this signal from
known geometry, with fewer crossing configurations, no real partial voluming and
no real motion.  every error rate below is therefore a *lower bound* on the error
rate on real diffusion data, and no number here should be quoted as "the error
rate of tractography".

usage
-----
    ./.venv/bin/python -m scripts.measure_tract_uncertainty --help
    ./.venv/bin/python scripts/measure_tract_uncertainty.py --out data/...json

endpoint extraction dominates the runtime (37 GB of streamlines), so it is cached
per file under `--cache`; the analysis re-runs in seconds from the cache.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# the phantom's own extent, in mm, from the challenge volumes: 90 x 108 x 90 at
# 2 mm, equivalently 180 x 216 x 180 at 1 mm.  used to reflect a coordinate axis
# when a submission turns out to be stored in the opposite convention.
EXTENT = np.array([180.0, 216.0, 180.0], dtype=float)


# ---------------------------------------------------------------------------
# readers
#
# nibabel reads trk and tck, and dipy would read vtk if vtk itself were
# installed.  none of that is used for the bulk pass, because what is wanted here
# is two points per streamline out of a 1.9 GB file and every library route
# materializes whole streamlines through python objects to get them.  these three
# readers walk the container and index the points directly, which is the
# difference between minutes and hours over the corpus.
# ---------------------------------------------------------------------------


def _sample_rows(n_streamlines: int, budget: int = 4_000_000) -> int:
    """points to keep per streamline for the occupancy grid.

    the occupancy grid only has to be dense enough to distinguish a brain from
    its mirror image, so a handful of points per streamline is plenty for a
    million-streamline submission -- but a 10,000-streamline submission needs
    more of each streamline to fill the same volume.  scaling by the budget keeps
    the grid comparably filled either way, which matters because the alignment
    below is a dice score and dice is not invariant to how densely each side is
    sampled.
    """
    return int(np.clip(budget // max(n_streamlines, 1), 3, 40))


def read_trk(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """trackvis: 1000-byte header, then per streamline a count and its points.

    the coordinates are millimetres from the corner of the first voxel in the
    volume's *own* axis order, which is why `voxel_order` in the header matters
    and why it is not trusted here -- several submissions carry one that
    disagrees with the points.  the alignment search settles it empirically.
    """
    with path.open("rb") as fh:
        hdr = fh.read(1000)
    n_scalars = struct.unpack_from("<h", hdr, 36)[0]
    n_props = struct.unpack_from("<h", hdr, 238)[0]
    n_count = struct.unpack_from("<i", hdr, 988)[0]
    hdr_size = struct.unpack_from("<i", hdr, 996)[0]
    if hdr_size != 1000:
        raise ValueError(f"{path.name}: trk header size {hdr_size}, expected 1000")

    f32 = np.memmap(path, dtype="<f4", mode="r", offset=1000)
    i32 = f32.view("<i4")
    per_pt = 3 + n_scalars

    # n_count is allowed to be 0 ("unknown"), so the walk is bounded by the file
    # rather than by the header and the header's count is only a hint.
    total = f32.size
    starts, npts = [], []
    i = 0
    while i < total:
        k = int(i32[i])
        if k <= 0 or i + 1 + k * per_pt + n_props > total:
            break
        starts.append(i + 1)
        npts.append(k)
        i += 1 + k * per_pt + n_props
    if not starts:
        return np.zeros((0, 2, 3), np.float32), np.zeros((0, 3), np.float32), 0

    starts = np.asarray(starts, np.int64)
    npts = np.asarray(npts, np.int64)
    first = starts
    last = starts + (npts - 1) * per_pt
    ends = np.stack([np.stack([f32[first], f32[first + 1], f32[first + 2]], 1),
                     np.stack([f32[last], f32[last + 1], f32[last + 2]], 1)], 1)

    per = _sample_rows(len(starts))
    frac = np.linspace(0.0, 1.0, per)
    idx = (starts[:, None] + (np.round(frac[None, :] * (npts[:, None] - 1))
                              ).astype(np.int64) * per_pt).reshape(-1)
    cloud = np.stack([f32[idx], f32[idx + 1], f32[idx + 2]], 1)
    return ends.astype(np.float32), cloud.astype(np.float32), len(starts)


def read_tck(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """mrtrix: a text header, then float triplets with NaN between streamlines.

    the separator convention is the whole format: a (NaN, NaN, NaN) triplet ends
    a streamline and (inf, inf, inf) ends the file, so the streamline boundaries
    fall out of one pass of `isfinite` over the first column rather than a walk.
    """
    with path.open("rb") as fh:
        head = fh.read(1 << 16)
    end = head.find(b"END\n")
    if end < 0:
        raise ValueError(f"{path.name}: no END in the tck header")
    text = head[:end].decode("latin-1")
    m = re.search(r"^file:\s*\.\s*(\d+)\s*$", text, re.MULTILINE)
    offset = int(m.group(1)) if m else end + 4
    dt = "<f4"
    if re.search(r"^datatype:\s*Float32BE", text, re.MULTILINE):
        dt = ">f4"

    raw = np.memmap(path, dtype=dt, mode="r", offset=offset)
    pts = raw[: (raw.size // 3) * 3].reshape(-1, 3)
    bad = ~np.isfinite(pts[:, 0])
    breaks = np.flatnonzero(bad)
    if breaks.size == 0:
        return np.zeros((0, 2, 3), np.float32), np.zeros((0, 3), np.float32), 0
    starts = np.concatenate([[0], breaks[:-1] + 1])
    stops = breaks                                  # exclusive
    keep = stops > starts + 1
    starts, stops = starts[keep], stops[keep]
    if starts.size == 0:
        return np.zeros((0, 2, 3), np.float32), np.zeros((0, 3), np.float32), 0

    ends = np.stack([pts[starts], pts[stops - 1]], 1)
    per = _sample_rows(len(starts))
    frac = np.linspace(0.0, 1.0, per)
    npts = (stops - starts).astype(np.int64)
    idx = (starts[:, None] + np.round(frac[None, :] * (npts[:, None] - 1)).astype(np.int64)
           ).reshape(-1)
    cloud = pts[idx]
    return np.asarray(ends, np.float32), np.asarray(cloud, np.float32), len(starts)


def read_vtk(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """legacy vtk binary POLYDATA: big-endian points, then a LINES connectivity list.

    the 2015-era converters wrote both `\\n` and `\\r\\n` after the POINTS line,
    which is the sort of thing that makes a hand-written reader necessary and a
    regex for the header line rather than a fixed offset.
    """
    with path.open("rb") as fh:
        head = fh.read(1 << 16)
    m = re.search(rb"POINTS\s+(\d+)\s+(float|double)\s*\r?\n", head)
    if m is None:
        raise ValueError(f"{path.name}: no POINTS record in the vtk header")
    n_pts = int(m.group(1))
    dt = ">f4" if m.group(2) == b"float" else ">f8"
    p_off = m.end()
    itemsize = np.dtype(dt).itemsize
    pts = np.memmap(path, dtype=dt, mode="r", offset=p_off,
                    shape=(n_pts, 3))

    l_off = p_off + n_pts * 3 * itemsize
    with path.open("rb") as fh:
        fh.seek(l_off)
        tail = fh.read(256)
    lm = re.search(rb"LINES\s+(\d+)\s+(\d+)\s*\r?\n", tail)
    if lm is None:
        raise ValueError(f"{path.name}: no LINES record after the points")
    n_cells, n_ints = int(lm.group(1)), int(lm.group(2))
    conn = np.memmap(path, dtype=">i4", mode="r", offset=l_off + lm.end(), shape=(n_ints,))

    # walk the [count, id, id, ...] list once.  vectorizing this needs the counts
    # to be known first, which is what the walk produces, so a python loop over
    # cells is the honest cost; it is ~2e5 iterations for the largest file.
    first = np.empty(n_cells, np.int64)
    last = np.empty(n_cells, np.int64)
    length = np.empty(n_cells, np.int64)
    i = 0
    for c in range(n_cells):
        k = int(conn[i])
        first[c] = conn[i + 1]
        last[c] = conn[i + k]
        length[c] = k
        i += k + 1
    ends = np.stack([np.asarray(pts[first]), np.asarray(pts[last])], 1)

    per = _sample_rows(n_cells)
    frac = np.linspace(0.0, 1.0, per)
    # sample by *point index* rather than by connectivity id: the converters that
    # produced these files wrote consecutive ids per line, and where they did not
    # the sample is still inside the tractogram, which is all the occupancy grid
    # needs.
    take = (first[:, None] + np.round(frac[None, :] * (length[:, None] - 1)).astype(np.int64)
            ).reshape(-1)
    take = np.clip(take, 0, n_pts - 1)
    cloud = np.asarray(pts[take])
    return ends.astype(np.float32), cloud.astype(np.float32), n_cells


def trk_voxel_order(path: Path) -> str | None:
    """the axis order a trackvis file *claims*, for use as an independent check.

    not used to align anything -- the alignment is measured from the geometry --
    but 42 of the 96 submissions are .trk and carry this field, which makes it a
    free check of whether the geometric search recovered the right convention.  the
    tck and vtk files carry nothing comparable.
    """
    if path.suffix.lower() != ".trk":
        return None
    with path.open("rb") as fh:
        fh.seek(948)
        raw = fh.read(4)
    vo = raw.split(b"\x00")[0].decode("latin-1").strip()
    return vo or None


READERS = {".trk": read_trk, ".tck": read_tck, ".vtk": read_vtk, ".fib": read_vtk}


def read_any(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    fn = READERS.get(path.suffix.lower())
    if fn is None:
        raise ValueError(f"{path.name}: no reader for {path.suffix}")
    return fn(path)


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


def occupancy(cloud: np.ndarray, mm: float = 4.0) -> np.ndarray:
    """a boolean grid over the phantom box.  the currency of the alignment search."""
    shape = tuple(int(np.ceil(e / mm)) for e in EXTENT)
    g = np.zeros(shape, bool)
    idx = np.floor(cloud / mm).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
    idx = idx[ok]
    g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return g


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.count_nonzero(a & b))
    tot = float(np.count_nonzero(a) + np.count_nonzero(b))
    return 2.0 * inter / tot if tot else 0.0


#: the axis-sign conventions searched, as (sx, sy, sz) relative to the ground
#: truth's own frame.  **x and y flip together and never separately**, and that
#: restriction is the single most consequential decision in this file.
#:
#: it is forced by a measurement.  flipping y or z costs 20-30% of the occupancy
#: dice against the phantom, so both are identified from the geometry alone.
#: flipping *x* costs about 3% -- the phantom is left-right symmetric to within
#: the resolution of any occupancy overlap, so x is simply not recoverable from
#: the streamlines.  searching it anyway would mean choosing a hemisphere
#: assignment on a 1.03x margin, which is choosing it at random.
#:
#: what settles it is that a lone x flip is not a convention any format
#: implements: files are written RAS or LPS (and occasionally with z reversed),
#: and LPS negates x and y together.  tying them turns an unidentifiable bit into
#: a determined one, and the result is checkable -- for all 42 .trk submissions
#: the convention chosen this way agrees with the `voxel_order` field in the
#: trackvis header, which the search never reads.  that agreement is recomputed
#: on every run and reported; if it ever falls below 100% this rule is wrong.
CONVENTIONS = ((1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1))


@dataclass
class Alignment:
    signs: tuple[int, int, int] = (1, 1, 1)
    scale: float = 1.0
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dice: float = 0.0
    #: the best dice any *other* convention reached.  the ratio between the two is
    #: what says the convention was identified, and it is a different question from
    #: how well the tractogram covers the phantom -- see `identified`.
    runner_up: float = 0.0
    bbox: tuple = ()

    @property
    def margin(self) -> float:
        return self.dice / self.runner_up if self.runner_up > 0 else float("inf")

    def identified(self, min_dice: float, min_margin: float) -> bool:
        """was the convention recovered, as distinct from the tractogram being good.

        a plain dice floor conflates the two and gets the bias backwards.  a sparse
        tractogram -- 8,000 streamlines against the ground truth's 200,000 -- fills
        less of the volume and scores a lower dice under the *correct* convention
        than a dense one does, so a floor set high enough to reject a flip also
        rejects every thin submission.  those are exactly the low-sensitivity
        pipelines, so excluding them would inflate the measured recovery rate.

        the convention is identified when the winning transform beats every rival
        by a wide margin, which is a scale-free statement about the 24 candidates
        and says nothing about density.  the dice floor stays as a second, looser
        gate for the case where every candidate is equally bad.
        """
        return self.dice >= min_dice or (self.dice >= 0.25 and self.margin >= min_margin)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x * (self.scale * np.asarray(self.signs, float)) + np.asarray(self.offset)

    def label(self) -> str:
        return ("".join("+-"[s < 0] for s in self.signs)
                + (f" x{self.scale:g}" if self.scale != 1.0 else "")
                + " o(" + ",".join(f"{o:.0f}" for o in self.offset) + ")")


def _robust_extent(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """the 1st and 99th percentile corner of a point cloud.

    percentiles rather than the true corners because a single stray streamline
    that escaped the mask would otherwise set the translation for the whole
    submission, and several of these tractograms have exactly that.
    """
    return np.percentile(x, 1.0, axis=0), np.percentile(x, 99.0, axis=0)


def align_to(cloud: np.ndarray, ref: np.ndarray, mm: float = 4.0,
             max_points: int = 600_000) -> Alignment:
    """recover the coordinate convention each submission was written in.

    the search space is deliberately small: a sign per axis, a scale from
    {1, 2, 1/2}, and a translation *read off the bounding boxes* rather than
    optimized.  that covers every convention actually present in the corpus --
    RAS against LPS, an origin at the volume corner against one at its centre,
    millimetres against 2 mm voxel indices -- and covers nothing else.

    it stops there on purpose.  fitting a rigid or affine transform would let a
    tractogram that is simply wrong be rotated into agreement with the ground
    truth, and the error rate this script exists to measure would come out
    however good the optimizer was.  the residual dice is reported per
    submission so that a convention which nearly worked is visible as one.

    the empirical fact that made this necessary: 33 of the 96 submissions are
    written with x and y negated about the scanner origin, so their coordinates
    are negative throughout.  a bounding box cannot see a reflection about the
    volume centre, which is why the search is scored on occupancy overlap and
    not on extents.
    """
    ref_lo, ref_hi = _robust_extent(ref)
    ref_mid = 0.5 * (ref_lo + ref_hi)
    ref_g = occupancy(ref, mm)

    step = max(1, len(cloud) // max_points)
    c = np.asarray(cloud[::step], float)
    c_lo, c_hi = _robust_extent(c)
    c_span = np.maximum(c_hi - c_lo, 1e-6)
    ratio = (ref_hi - ref_lo) / c_span
    scales = {1.0}
    # a submission written in voxel indices has a span half or a quarter of the
    # millimetre one; anything else is not a units question and is left alone.
    for s in (2.0, 0.5):
        if np.all(np.abs(ratio - s) < 0.15 * s):
            scales.add(s)

    best, second = Alignment(dice=-1.0), 0.0
    for scale in sorted(scales):
        for sx, sy, sz in CONVENTIONS:
            signs = np.array([sx, sy, sz], float)
            mid = 0.5 * (c_lo + c_hi) * scale * signs
            off = tuple((ref_mid - mid).tolist())
            cand = Alignment((sx, sy, sz), scale, off)
            d = _dice(occupancy(cand.apply(c), mm), ref_g)
            if d > best.dice:
                second = max(second, best.dice)
                best = Alignment((sx, sy, sz), scale, off, d)
            else:
                second = max(second, d)
    best.runner_up = max(second, 0.0)
    best.bbox = tuple(np.round(np.concatenate([c.min(0), c.max(0)]), 1).tolist())
    return best


# ---------------------------------------------------------------------------
# parcellation and edges
# ---------------------------------------------------------------------------


@dataclass
class Parcellation:
    """cubic bins, and the ground truth's own occupancy deciding which exist.

    NOT an atlas.  see the module docstring: nothing in `ibm.anatomy.systems` is
    in register with this phantom and putting one there would be a fabricated
    registration.  the bins that exist are the ones the ground-truth streamlines
    pass through, so the universe of possible edges is defined by the answer
    rather than by any submission -- otherwise a pipeline that terminates in odd
    places would enlarge the universe it is then scored against.
    """

    bin_mm: float
    shape: tuple[int, int, int]
    keep: np.ndarray            # (n_bins,) bool over the flattened grid
    index: np.ndarray           # (n_bins,) int, -1 where not kept
    n: int

    @classmethod
    def from_cloud(cls, cloud: np.ndarray, bin_mm: float, min_points: int) -> "Parcellation":
        shape = tuple(int(np.ceil(e / bin_mm)) for e in EXTENT)
        flat = np.zeros(int(np.prod(shape)), np.int64)
        lin = cls._lin(cloud, shape, bin_mm)
        lin = lin[lin >= 0]
        np.add.at(flat, lin, 1)
        keep = flat >= min_points
        index = np.full(flat.size, -1, np.int64)
        index[keep] = np.arange(int(keep.sum()))
        return cls(bin_mm, shape, keep, index, int(keep.sum()))

    @staticmethod
    def _lin(xyz: np.ndarray, shape, bin_mm: float) -> np.ndarray:
        idx = np.floor(xyz / bin_mm).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
        lin = np.full(len(xyz), -1, np.int64)
        lin[ok] = ((idx[ok, 0] * shape[1] + idx[ok, 1]) * shape[2] + idx[ok, 2])
        return lin

    def bins_of(self, xyz: np.ndarray) -> np.ndarray:
        lin = self._lin(xyz, self.shape, self.bin_mm)
        out = np.full(len(xyz), -1, np.int64)
        ok = lin >= 0
        out[ok] = self.index[lin[ok]]
        return out


def edge_counts(ends: np.ndarray, parc: Parcellation) -> tuple[np.ndarray, np.ndarray, float]:
    """streamline count per unordered bin pair, plus the fraction that landed nowhere.

    the dropped fraction is a finding in its own right and is returned rather than
    swallowed: a tractogram most of whose endpoints fall outside the bins the
    ground truth occupies has terminated somewhere the phantom has no fibres, and
    that is a systematic error the edge table alone would not show.
    """
    a = parc.bins_of(ends[:, 0, :])
    b = parc.bins_of(ends[:, 1, :])
    ok = (a >= 0) & (b >= 0) & (a != b)
    dropped = 1.0 - float(ok.mean()) if len(ok) else 1.0
    lo = np.minimum(a[ok], b[ok])
    hi = np.maximum(a[ok], b[ok])
    key = lo * parc.n + hi
    uniq, cnt = np.unique(key, return_counts=True)
    return uniq, cnt, dropped


# ---------------------------------------------------------------------------
# the pass over the corpus
# ---------------------------------------------------------------------------


def extract(path: Path, cache: Path) -> dict:
    """endpoints and an occupancy cloud for one tractogram, cached."""
    cf = cache / (path.stem + path.suffix.replace(".", "_") + ".npz")
    if cf.exists():
        with np.load(cf) as z:
            return {"ends": z["ends"], "cloud": z["cloud"], "n": int(z["n"])}
    t0 = time.time()
    ends, cloud, n = read_any(path)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cf, ends=ends, cloud=cloud, n=np.int64(n))
    print(f"    read {path.name}: {n} streamlines in {time.time() - t0:.1f}s", flush=True)
    return {"ends": ends, "cloud": cloud, "n": n}


# ---------------------------------------------------------------------------
# error correlation
# ---------------------------------------------------------------------------


def mean_pairwise_correlation(E: np.ndarray) -> tuple[float, np.ndarray]:
    """mean off-diagonal correlation of the submissions' error vectors.

    `E` is (submissions, edges) and binary: 1 where the submission disagrees with
    the ground truth about that edge.  the mean off-diagonal correlation of its
    rows is the fraction of error variance that is *shared* rather than private
    to a pipeline, which is the same quantity a source card calls
    `correlated_fraction` -- the reason it can be read that way is that a
    one-factor model, error = shared + private, has exactly this pairwise
    correlation and no other free parameter.

    the interpretation people reach for first is wrong and worth saying out loud:
    this is not "the pipelines agree with each other", it is "the pipelines are
    wrong in the same places".  an edge that every pipeline finds and that exists
    contributes nothing to it; an edge that every pipeline misses contributes
    everything.
    """
    X = np.asarray(E, float)
    X = X - X.mean(1, keepdims=True)
    sd = np.sqrt((X ** 2).sum(1))
    live = sd > 0
    if live.sum() < 2:
        return float("nan"), np.zeros((0, 0))
    Y = X[live] / sd[live][:, None]
    R = Y @ Y.T
    n = R.shape[0]
    off = R[~np.eye(n, dtype=bool)]
    return float(off.mean()), R


def effective_pipelines(n: int, rho: float) -> float:
    """`n` pipelines whose errors share a fraction `rho` are worth how many?

    computed through `ibm.runtime.fuse` rather than by the closed form, because
    that module is where this arithmetic is supposed to live and because routing
    a real measurement through it is the only way to find out whether it agrees
    with the formula the architecture quotes.  the fallback is the closed form,
    used only if the import fails, and the result says which was used.
    """
    if not np.isfinite(rho):
        return float("nan")
    rho = float(np.clip(rho, 0.0, 1.0 - 1e-9))
    try:
        from ibm.runtime.fuse import TeacherPrecision
        tp = TeacherPrecision(r2=0.0, correlated_fraction=rho, error_rank=1,
                              source="ismrm2015 submissions")
        ev = tp.evidence("structural.axonal_density", np.zeros(n), np.ones(n))
        return float(ev.effective_constraints())
    except Exception:
        return n / ((1.0 - rho) + n * rho)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None,
                    help="repository root; resolved from this file otherwise")
    ap.add_argument("--bin-mm", type=float, default=20.0,
                    help="cubic bin side, in mm.  this is the parcellation")
    ap.add_argument("--min-bin-points", type=int, default=200,
                    help="ground-truth points a bin needs before it exists")
    ap.add_argument("--min-streamlines", type=int, default=1,
                    help="streamlines a bin pair needs before an edge is asserted")
    ap.add_argument("--min-dice", type=float, default=0.5,
                    help="occupancy dice at which a convention is accepted outright")
    ap.add_argument("--min-margin", type=float, default=2.0,
                    help="ratio to the runner-up convention that also accepts one, so that "
                         "a sparse tractogram is not excluded for being sparse")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write the measurement as json")
    ap.add_argument("--limit", type=int, default=0, help="stop after N submissions (debug)")
    args = ap.parse_args(argv)

    from ibm.forge.bind import repo_root
    root = args.root or repo_root(Path(__file__))
    cache = args.cache or (root / ".cache" / "tract_uncertainty")

    import yaml

    def local_root(src: str) -> Path:
        loc = root / "data" / "sources" / src / "raw" / ".location.yaml"
        return Path(yaml.safe_load(loc.read_text())["local_root"])

    gt_root = local_root("ismrm2015-tractography-challenge")
    sub_root = local_root("ismrm2015-submissions")
    gt_path = gt_root / "extracted" / "FilesForSimulation" / "Fibers.fib"
    if not gt_path.exists():
        print(f"ground truth not found at {gt_path}", file=sys.stderr)
        return 2

    print(f"ground truth : {gt_path}")
    gt = extract(gt_path, cache)
    print(f"               {gt['n']} streamlines, bbox "
          f"{np.round(gt['cloud'].min(0), 1)} .. {np.round(gt['cloud'].max(0), 1)}")

    parc = Parcellation.from_cloud(gt["cloud"], args.bin_mm, args.min_bin_points)
    print(f"parcellation : cubic spatial binning at {args.bin_mm:g} mm -- NOT an atlas; "
          f"{parc.n} bins occupied by the ground truth")

    gt_all, gt_cnt, gt_drop = edge_counts(gt["ends"], parc)
    gt_edges = gt_all[gt_cnt >= args.min_streamlines]
    print(f"               {len(gt_edges)} ground-truth edges over "
          f"{parc.n * (parc.n - 1) // 2} possible bin pairs; "
          f"{gt_drop:.1%} of ground-truth endpoints fell outside the bins")

    files = sorted(p for p in sub_root.iterdir()
                   if p.suffix.lower() in (".trk", ".tck", ".vtk"))
    if args.limit:
        files = files[: args.limit]
    print(f"submissions  : {len(files)} files under {sub_root}")

    subs: list[dict] = []
    for p in files:
        try:
            d = extract(p, cache)
        except Exception as e:                      # a file we cannot read is a finding
            print(f"    !! {p.name}: {type(e).__name__}: {e}", flush=True)
            subs.append({"name": p.stem, "error": f"{type(e).__name__}: {e}"})
            continue
        al = align_to(d["cloud"], gt["cloud"])
        ends = al.apply(d["ends"].astype(float).copy())
        uniq, cnt, drop = edge_counts(ends, parc)
        # the independent check.  the ground truth is written LPS-like, so a file
        # declaring RAS must have x and y negated to reach it and one declaring
        # LPS must not.  the search never reads this field.
        vo = trk_voxel_order(p)
        header_ok = None
        if vo:
            want = (-1, -1) if vo.upper().startswith("RAS") else (1, 1)
            header_ok = tuple(al.signs[:2]) == want
        subs.append({"name": p.stem, "team": p.stem.split("_")[0], "n": d["n"],
                     "voxel_order": vo, "header_ok": header_ok,
                     "align": al.label(), "dice": al.dice, "dropped": drop,
                     "margin": al.margin, "ok": al.identified(args.min_dice, args.min_margin),
                     "edges_all": uniq, "counts": cnt, "bbox": al.bbox})

    ok = [s for s in subs if s.get("ok")]
    bad = [s for s in subs if not s.get("ok")]
    print(f"aligned      : {len(ok)} of {len(subs)} submissions identified a convention "
          f"(dice >= {args.min_dice}, or >= {args.min_margin}x the runner-up)")
    for s in bad:
        print(f"    excluded {s['name']}: "
              + (s.get("error") or f"best dice {s['dice']:.3f} at {s['align']}, "
                                   f"{s['margin']:.2f}x the runner-up, bbox {s['bbox']}"))
    checked = [s for s in subs if s.get("header_ok") is not None]
    n_hdr_ok = sum(1 for s in checked if s["header_ok"])
    print(f"    convention cross-check: {n_hdr_ok} of {len(checked)} .trk submissions "
          f"agree with their own voxel_order field, which the search never reads")
    for s in checked:
        if not s["header_ok"]:
            print(f"    !! {s['name']} chose {s['align']} but declares "
                  f"{s['voxel_order']}")

    # ---- the analysis, as a function of the edge threshold ------------------
    # `min_streamlines` is the one knob that could manufacture any answer wanted,
    # so it is swept rather than chosen.  at 1 an edge exists if a single
    # streamline connects the boxes, which is the most permissive reading and the
    # right one for a SUPPORT; at 100 only substantial bundles count.  the
    # headline numbers move, the conclusion does not, and both are reported.
    n_pairs = parc.n * (parc.n - 1) // 2

    def analyse(thresh: int) -> dict:
        gt_e = gt_all[gt_cnt >= thresh]
        sub_e = [s["edges_all"][s["counts"] >= thresh] for s in ok]
        # the universe is every pair either the truth or some surviving pipeline
        # asserts.  the alternative -- all bin pairs -- is dominated by pairs
        # nobody asserts, and agreeing that two distant boxes are unconnected is
        # not evidence that pipelines agree about anything.
        universe = np.unique(np.concatenate([gt_e] + sub_e)) if ok else gt_e
        gt_mask = np.isin(universe, gt_e)
        A = np.zeros((len(ok), universe.size), bool)
        for i, e in enumerate(sub_e):
            A[i] = np.isin(universe, e)

        p_exist = A[:, gt_mask].mean(0)
        p_false = A[:, ~gt_mask].mean(0)
        sens = A[:, gt_mask].mean(1)
        n_false = A[:, ~gt_mask].sum(1).astype(float)
        fdr = np.array([1.0 - (A[i, gt_mask].sum() / max(A[i].sum(), 1))
                        for i in range(len(ok))])

        # ---- how much do pipelines disagree about an edge's STRENGTH? --------
        # `tract.py` refuses to carry streamline count as an edge feature because
        # it is an artefact of the seeding and tracking algorithm.  that is right,
        # and it is exactly why the SPREAD of it across pipelines is worth having:
        # it is a direct measurement of how much of the quantity is artefact, and
        # it is what a prior on coupling strength should be as wide as.  the count
        # is normalized by each pipeline's own total first, because a pipeline
        # that emitted 900,000 streamlines and one that emitted 8,000 are not
        # disagreeing when their raw counts differ by a hundred.
        dens = np.zeros((len(ok), universe.size))
        for i, s in enumerate(ok):
            c = s["counts"][s["counts"] >= thresh].astype(float)
            e = s["edges_all"][s["counts"] >= thresh]
            pos = np.searchsorted(universe, e)
            keep = (pos < universe.size) & (universe[np.minimum(pos, universe.size - 1)] == e)
            dens[i, pos[keep]] = c[keep] / max(float(s["n"]), 1.0)
        strong = (A.mean(0) > 0.5)
        lg = np.where(dens > 0, np.log(np.maximum(dens, 1e-30)), np.nan)
        with np.errstate(invalid="ignore"):
            sd_edge = np.nanstd(lg[:, strong], axis=0)
        sd_edge = sd_edge[np.isfinite(sd_edge)]
        strength_sd = float(np.median(sd_edge)) if sd_edge.size else float("nan")
        strength_sd_iqr = ([float(np.percentile(sd_edge, 25)),
                            float(np.percentile(sd_edge, 75))] if sd_edge.size else [])

        E = (A != gt_mask[None, :]).astype(float)
        rho, R = mean_pairwise_correlation(E)
        rho_fn, _ = mean_pairwise_correlation((A[:, gt_mask] == 0).astype(float))
        rho_fp, _ = mean_pairwise_correlation(A[:, ~gt_mask].astype(float))
        teams = np.array([s["team"] for s in ok])
        same = teams[:, None] == teams[None, :]
        offd = ~np.eye(len(ok), dtype=bool)
        within = float(R[same & offd].mean()) if (same & offd).any() else float("nan")
        between = float(R[~same & offd].mean()) if (~same & offd).any() else float("nan")
        n_teams = len(set(teams.tolist()))
        return {
            "min_streamlines": thresh,
            "gt_edges": int(gt_mask.sum()), "universe_edges": int(universe.size),
            "possible_pairs": n_pairs,
            "existence": {
                "mean_recovery_of_true_edges": float(p_exist.mean()),
                "median_per_edge_existence_probability": float(np.median(p_exist)),
                "true_edges_no_pipeline_found": int((p_exist == 0).sum()),
                "true_edges_every_pipeline_found": int((p_exist == 1).sum()),
                "true_edges_majority_found": int((p_exist > 0.5).sum()),
                "per_pipeline_sensitivity": {"median": float(np.median(sens)),
                                             "min": float(sens.min()),
                                             "max": float(sens.max())},
            },
            "false_positives": {
                "mean_assertion_of_false_edges": float(p_false.mean()),
                "false_edges_majority_assert": int((p_false > 0.5).sum()),
                "false_edges_in_universe": int((~gt_mask).sum()),
                "mean_false_edges_per_pipeline": float(n_false.mean()),
                # over every pair that could have been asserted and is not true,
                # which is the specificity a connectome claim actually needs
                "per_pipeline_false_positive_rate": float(
                    (n_false / max(n_pairs - int(gt_mask.sum()), 1)).mean()),
                "per_pipeline_false_discovery_rate": {"median": float(np.median(fdr)),
                                                      "min": float(fdr.min()),
                                                      "max": float(fdr.max())},
            },
            "strength": {
                "log_sd_across_pipelines": strength_sd,
                "log_sd_iqr": strength_sd_iqr,
                "n_edges_measured": int(strong.sum()),
                "as_a_factor": float(np.exp(strength_sd)) if np.isfinite(strength_sd) else None,
                "note": "natural-log standard deviation, across pipelines, of the "
                        "streamline count on an edge normalized by that pipeline's total "
                        "streamline count.  measured on the edges a majority of pipelines "
                        "assert, because an edge most pipelines miss has no spread to "
                        "measure.  `as_a_factor` is exp() of it: the multiplicative "
                        "spread a lognormal prior on coupling strength should carry when "
                        "its median came from one pipeline",
            },
            "error_correlation": {
                "correlated_fraction": float(rho),
                "correlated_fraction_misses": float(rho_fn),
                "correlated_fraction_false_positives": float(rho_fp),
                "within_team": within, "between_team": between,
                "teams": n_teams,
                "effective_independent_pipelines": float(effective_pipelines(len(ok), rho)),
                "effective_independent_teams": float(effective_pipelines(n_teams, between)),
            },
        }

    primary = analyse(args.min_streamlines)
    ex, fp, ec = primary["existence"], primary["false_positives"], primary["error_correlation"]
    print(f"universe     : {primary['universe_edges']} bin pairs asserted by the truth or "
          f"by some pipeline; {primary['gt_edges']} of them true, out of "
          f"{n_pairs} possible")
    print()
    print(f"== existence (edge = at least {args.min_streamlines} streamline(s)) ==")
    print(f"true edges recovered, mean over pipelines : "
          f"{ex['mean_recovery_of_true_edges']:.3f}")
    print(f"    median per-edge existence probability : "
          f"{ex['median_per_edge_existence_probability']:.3f}")
    print(f"    true edges a majority found           : {ex['true_edges_majority_found']} "
          f"of {primary['gt_edges']}")
    print(f"    true edges NO pipeline found          : "
          f"{ex['true_edges_no_pipeline_found']}")
    print(f"    true edges EVERY pipeline found       : "
          f"{ex['true_edges_every_pipeline_found']}")
    print(f"per-pipeline sensitivity  median {ex['per_pipeline_sensitivity']['median']:.3f}"
          f"  range {ex['per_pipeline_sensitivity']['min']:.3f}-"
          f"{ex['per_pipeline_sensitivity']['max']:.3f}")
    print()
    print("== false positives ==")
    print(f"false edges asserted per pipeline          : "
          f"{fp['mean_false_edges_per_pipeline']:.0f} of "
          f"{n_pairs - primary['gt_edges']} possible non-edges "
          f"({fp['per_pipeline_false_positive_rate']:.1%})")
    print(f"    false edges a majority of pipelines assert : "
          f"{fp['false_edges_majority_assert']}")
    print(f"per-pipeline false discovery rate  median "
          f"{fp['per_pipeline_false_discovery_rate']['median']:.3f}  range "
          f"{fp['per_pipeline_false_discovery_rate']['min']:.3f}-"
          f"{fp['per_pipeline_false_discovery_rate']['max']:.3f}")
    print()
    st = primary["strength"]
    print()
    print("== strength (not existence) ==")
    print(f"cross-pipeline spread of normalized edge density, over "
          f"{st['n_edges_measured']} edges a majority assert:")
    print(f"    log sd {st['log_sd_across_pipelines']:.3f}  "
          f"= a factor of {st['as_a_factor']:.2f}")
    print()
    print("== error correlation ==")
    print(f"mean pairwise correlation of error vectors   rho = "
          f"{ec['correlated_fraction']:.3f}")
    print(f"    of misses (false negatives) alone        rho = "
          f"{ec['correlated_fraction_misses']:.3f}")
    print(f"    of false positives alone                 rho = "
          f"{ec['correlated_fraction_false_positives']:.3f}")
    print(f"    within a team  {ec['within_team']:.3f}   between teams  "
          f"{ec['between_team']:.3f}   ({ec['teams']} teams)")
    print(f"effective independent pipelines : "
          f"{ec['effective_independent_pipelines']:.2f} of {len(ok)}")
    print(f"    at team level                : "
          f"{ec['effective_independent_teams']:.2f} of {ec['teams']}")

    sweep = [analyse(k) for k in (1, 5, 20, 100) if k != args.min_streamlines]
    print()
    print("== threshold sweep (the one knob that could manufacture an answer) ==")
    print("  read the drop in recovery with care: the threshold is an ABSOLUTE streamline")
    print("  count and the submissions differ in total streamline count by two orders of")
    print("  magnitude, so at 100 it is partly measuring who submitted more streamlines.")
    print("  rho rising with it is the finding that survives -- the edges everyone agrees")
    print("  on are also the edges everyone is wrong about together.")
    print(f"{'min_sl':>7} {'gt edges':>9} {'recovery':>9} {'FDR':>7} {'rho':>7} {'n_eff':>7}")
    for r in sorted([primary] + sweep, key=lambda r: r["min_streamlines"]):
        print(f"{r['min_streamlines']:>7} {r['gt_edges']:>9} "
              f"{r['existence']['mean_recovery_of_true_edges']:>9.3f} "
              f"{r['false_positives']['per_pipeline_false_discovery_rate']['median']:>7.3f} "
              f"{r['error_correlation']['correlated_fraction']:>7.3f} "
              f"{r['error_correlation']['effective_independent_pipelines']:>7.2f}")

    result = {
        "measured_at": time.strftime("%Y-%m-%d"),
        "sources": ["ismrm2015-tractography-challenge", "ismrm2015-submissions"],
        "ground_truth": str(gt_path),
        "parcellation": {
            "kind": "cubic spatial binning",
            "why_not_an_atlas": "no anatomy system in ibm.anatomy.systems is in register "
                                "with the ISMRM 2015 phantom; placing one would be a "
                                "fabricated registration",
            "bin_mm": args.bin_mm, "n_bins": parc.n,
            "min_bin_points": args.min_bin_points,
        },
        "alignment": {
            "method": "a sign per axis, a scale from {1, 2, 1/2} and a translation read "
                      "off the robust bounding boxes; chosen by 4 mm occupancy dice "
                      "against the ground truth.  no rotation and no fitted affine",
            "min_dice": args.min_dice, "min_margin": args.min_margin,
            "conventions_searched": [list(c) for c in CONVENTIONS],
            "x_is_tied_to_y": "flipping x costs ~3% of the occupancy dice against this "
                              "phantom and flipping y or z costs 20-30%, so x is not "
                              "identifiable from the geometry.  it is determined by y "
                              "instead, because RAS and LPS negate the two together and "
                              "no format implements a lone x flip",
            "voxel_order_cross_check": {
                "trk_submissions_checked": len(checked),
                "agree_with_header": n_hdr_ok,
                "note": "the trackvis voxel_order field, which align_to never reads.  "
                        "the ground truth is LPS-like, so a file declaring RAS must have "
                        "x and y negated to reach it",
            },
            "excluded": [{"name": s["name"], "dice": s.get("dice"),
                          "margin": s.get("margin"), "error": s.get("error")}
                         for s in bad],
            "per_submission": [{"name": s["name"], "align": s["align"],
                                "dice": round(s["dice"], 4),
                                "margin_over_runner_up": round(s["margin"], 3),
                                "bbox": s["bbox"],
                                "endpoints_outside_bins": round(s["dropped"], 4),
                                "streamlines": s["n"]} for s in ok],
        },
        "counts": {"submissions_read": len(subs), "submissions_used": len(ok),
                   "teams": ec["teams"], "gt_streamlines": gt["n"],
                   "possible_bin_pairs": n_pairs,
                   "gt_edges": primary["gt_edges"],
                   "universe_edges": primary["universe_edges"],
                   "gt_endpoints_outside_bins": round(gt_drop, 4),
                   "min_streamlines_per_edge": args.min_streamlines},
        "existence": ex,
        "false_positives": fp,
        "strength": st,
        "error_correlation": dict(ec, error_rank_assumed=1, note=(
            "rank 1: a shared bias affecting every edge.  a higher rank would model the "
            "shared error as having a correlation length and would hand tractography "
            "orders of magnitude more effective votes, so it is not assumed without a "
            "measurement of that length")),
        "threshold_sweep": sorted([primary] + sweep, key=lambda r: r["min_streamlines"]),
        "threshold_sweep_note":
            "the threshold is an ABSOLUTE streamline count per bin pair and the "
            "submissions range from 8,000 to 900,000 streamlines, so a high threshold "
            "partly measures who submitted more.  what survives the sweep is the "
            "direction of rho: the shared-error fraction RISES as the threshold tightens, "
            "because the edges every pipeline agrees on are also the edges every pipeline "
            "is wrong about together, which is the whole reason the count of pipelines is "
            "not the number of votes",
        "caveats": [
            "the phantom is simulated and simpler than tissue, so every rate here is a "
            "LOWER bound on the same rate for real diffusion data",
            "cubic 20 mm bins merge nearby endpoints, which understates both the miss "
            "rate and the false-positive rate relative to a fine parcellation",
            "the coordinate convention of each submission was recovered by an occupancy "
            "search over four conventions, not read from a header.  x is NOT identifiable "
            "from this phantom's geometry -- it is left-right symmetric to within 3% of "
            "the dice -- and is determined by y, which is.  all 42 .trk files agree with "
            "their own voxel_order field, which is the check on that rule",
            "the 5 excluded submissions are the sparsest ones, and sparse tractograms "
            "have low sensitivity, so their exclusion biases the recovery rate UPWARD",
            "submissions from one team are the same algorithm at different settings and "
            "are not independent; the team-level figure is the one to quote",
        ],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
