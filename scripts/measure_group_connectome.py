"""what a group connectome costs, measured on real tissue instead of a phantom.

`scripts/measure_tract_uncertainty.py` measured how wrong one tractogram of one
brain is, against the only dataset where the answer is known by construction.
that measurement is tier 1's calibration and it says nothing about tier 2.  a
group connectome is a different object with a different error: somebody else's
subjects, averaged, applied to a subject who has no diffusion imaging at all --
which is the ordinary case in this corpus, because the mne `sample` subject,
eegmmidb, sleep-edfx and erp-core have no dwi and never will.  the phantom cannot
say what that costs, for the plain reason that it is ONE brain.

TractoInferno can, and this script is what asks it.  284 subjects, six sites, one
processing pipeline (tractoflow 2.1.1 then rbx-flow 1.1.0), expert-curated bundle
tractograms per subject.  the questions, in the order the tier-2 prior needs them:

1. **between subjects.**  how often do two people share an edge, and how far apart
   are their strengths when they do.  this is the variance a group average carries
   into a subject it was not measured on.

2. **between sites.**  is a different scanner as different as a different person?
   this is the quantity that decides whether a published group connectome
   transfers at all, and it is the one the phantom structurally cannot supply.

3. **the effective sample count.**  284 subjects are not 284 independent votes
   about a group connectome any more than 96 tractograms were 96 votes about one
   brain.  the same `ibm.runtime.fuse.TeacherPrecision.effective_constraints`
   arithmetic is used here, unchanged, for the same reason it was used there.

4. **what the group explains of an individual.**  hold a subject out, predict
   their connectome from the others, and report how much of their own variation
   survives the prediction.  that number is what a materialization should record
   in provenance when it falls back to tier 2, because it is the difference
   between "this subject's connectivity" and "a connectivity".

the honesty problems, and what was done about them
--------------------------------------------------
**the site labels are not in the release, and this script does not invent them.**
ds003900 v1.1.1 ships 10,884 files and not one of them is a `participants.tsv`;
there is no site column anywhere, the gradient tables have been harmonised to a
single 64-direction b=1000 scheme for every subject, and every volume has been
resampled to 1 mm isotropic -- so the acquisition signature a scanner would
normally leave has been processed away on purpose.  what survives is the FIELD OF
VIEW, which the pipeline crops to the brain but cannot extend: a subject scanned
with 120 slices has a volume 120 voxels deep however large their head is.  over
the 284 subjects that number is sharply multimodal with an empty gap between 120
and 130, and the groups it induces are EXACTLY CONTIGUOUS IN SUBJECT ID:

    block A   sub-1000 .. sub-1096    97 subjects   z <= 120, mixed-parity dims
    block B   sub-1097 .. sub-1158    62 subjects   z <= 120, all dims even
    block C   sub-1159 .. sub-1197    39 subjects   z 130-140, all dims even
    block D   sub-1198 .. sub-1283    86 subjects   z 131-150, mixed-parity dims

the paper's table 1 gives the per-site counts before quality control: BIL&GIN 39,
MRi-Share 20, Bilingualism-and-the-Brain 64, UCLA CNP 130, Stockholm Sleepy Brain
86, mTBI-and-Aging 15.  block C is 39 and block D is 86, both exact; block B is 62
against 64.  four blocks partition all 284 subjects with no remainder.  that is
strong enough to say the id ordering follows the source cohorts and that these are
SITE BOUNDARIES, and far too weak to say which site is which or to split block A,
which by subtraction still holds three cohorts.  so the numbers below are reported
as **between-acquisition-block**, they are a LOWER bound on the true between-site
spread because block A mixes sites, and no block is given a name.

**there is no atlas in register with these subjects.**  same problem the phantom
had and the same answer: cubic spatial binning, 20 mm, and an edge is "these two
boxes are connected".  `ibm.anatomy.systems` has nothing that could be placed on a
tractoinferno subject without inventing a registration, and the release ships no
freesurfer surface and no MNI warp.  what is different here is that the bins have
to mean the same thing in 284 different heads, so each subject is mapped by a
7-parameter normalization -- the centroid and the per-axis standard deviation of
its own white+grey mask, matched to the cohort median -- and nothing else.  no
rotation and no fitted warp, because a fitted warp would let two subjects be
deformed into agreement and the between-subject spread would come out however good
the registration was.  the residual is measured rather than assumed: the scatter
of each named bundle's endpoint centroid across subjects is 3.6 mm for the
pyramidal tract and 11.0 mm for the frontal callosum, median 7.3, which is inside a
20 mm bin and is reported with the results as the reason the bins are not smaller.
it is not misregistration alone -- a bundle's endpoint territory genuinely differs
between people and this cannot separate the two -- so it is an upper bound on the
registration error and a bound on nothing else.

**these are bundle tractograms, not whole-brain tractograms.**  the release ships
32 expert-curated bundles per subject and no whole-brain reference, so the edge
universe is the union of those bundles' terminations and nothing else -- there is
no claim here about connectivity the rbx bundle set does not cover.  worse and
more interesting: no bundle is present in all 284 subjects.  the best (MdLF) is
there for 94% of them, IFOF for 57% and the fornix for 14%.  a missing
bundle removes its edges from that subject entirely, so the raw edge
reproducibility below confounds "these two people differ" with "the segmentation
failed", and both are reported: `P(edge)` unconditional and `P(edge | the subject
has the bundle that carries it)`.

**there is no ground truth on real tissue.**  the phantom's `rho` -- the fraction
of a pipeline's error shared with other pipelines -- cannot be reproduced here at
all, because computing it needs an answer to be wrong about.  what can be computed
is the correlation of subjects' disagreement with the group's own consensus, and
that is a DIFFERENT and systematically smaller number, since any error every
subject makes is absorbed into the consensus and disappears.  it is reported as a
lower bound and the phantom's figure is the one that still governs the shared part,
with the additional aggravation that all 284 subjects went through ONE pipeline,
where the phantom's 0.317 was measured across twenty different ones.

usage
-----
    ./.venv/bin/python scripts/measure_group_connectome.py --help
    ./.venv/bin/python scripts/measure_group_connectome.py \
        --out data/sources/tractoinferno/evidence/group_connectome@1/measured.json

the site arm needs subjects from blocks B, C and D, which the local root does not
hold -- it holds the first 30 subjects by id, all of them block A.  they are
streamed from the openneuro s3 urls carried in the dataset's own
`.openneuro-manifest-1.1.1.json`, which IS in the local root, into `--cache`/cohort
and read from there on later runs.  8 bundles per subject rather than all 32,
because 5 GiB is enough to answer the question and 40 is not a better answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# the phantom script's trk walker, imported rather than copied.  it returns two
# points per streamline without materializing the streamline, which is the
# difference between minutes and hours over 8.8 GiB of bundles -- and a second
# copy of a binary-format walker is a second place for an offset to be wrong.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.measure_tract_uncertainty import read_trk  # noqa: E402

#: the acquisition-geometry blocks, as recovered above.  inclusive id ranges.
#: NOT site labels -- see the module docstring for exactly how far this goes.
BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("A", 1000, 1096),
    ("B", 1097, 1158),
    ("C", 1159, 1197),
    ("D", 1198, 1283),
)

#: the bundles the site arm fetches.  the eight highest-availability bundles that
#: also spread over the lobes: frontal, temporal, parietal, occipital and two
#: callosal.  a site comparison restricted to one lobe would measure the lobe.
SITE_BUNDLES = ("AF_L", "AF_R", "ILF_L", "ILF_R", "SLF_L", "SLF_R",
                "CC_Oc", "CC_Fr_1")


def block_of(sub: str) -> str:
    n = int(sub.split("-")[1])
    for name, lo, hi in BLOCKS:
        if lo <= n <= hi:
            return name
    return "?"


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------


@dataclass
class Subject:
    """one subject's paths, wherever its bytes happen to live.

    the local root and the cohort cache are the same tree with different subsets
    of it present, so a subject is a directory and a list of bundles and the code
    below never asks which of the two it came from.
    """

    sub: str
    split: str
    root: Path
    bundles: tuple[str, ...]
    source: str = "local"

    @property
    def block(self) -> str:
        return block_of(self.sub)

    def trk(self, bundle: str) -> Path:
        return self.root / "tractography" / f"{self.sub}__{bundle}.trk"

    def mask(self, which: str) -> Path:
        return self.root / "mask" / f"{self.sub}__mask_{which}.nii.gz"


def discover(deriv: Path, source: str = "local") -> dict[str, Subject]:
    out: dict[str, Subject] = {}
    for split in ("trainset", "validset", "testset"):
        d = deriv / split
        if not d.is_dir():
            continue
        for sd in sorted(d.iterdir()):
            if not sd.name.startswith("sub-"):
                continue
            trk = sorted((sd / "tractography").glob("*.trk"))
            if not trk:
                continue
            out[sd.name] = Subject(sd.name, split, sd,
                                   tuple(p.name.split("__")[1][:-4] for p in trk), source)
    return out


def manifest_urls(local_root: Path) -> dict[str, str]:
    """path -> s3 url, from the openneuro manifest the dataset ships with itself.

    read from the local root rather than from the network, so the url list is a
    property of the acquired bytes and not of whatever openneuro serves today.
    """
    man = local_root / ".openneuro-manifest-1.1.1.json"
    out: dict[str, str] = {}

    def walk(x, pre=""):
        if isinstance(x, list):
            if len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], (dict, list)):
                walk(x[1], pre + "/" + x[0])
            else:
                for e in x:
                    walk(e, pre)
        elif isinstance(x, dict):
            if "filename" in x:
                urls = x.get("urls") or [None]
                out[pre] = urls[0]
            else:
                for k, v in x.items():
                    walk(v, pre + "/" + k)

    walk(json.loads(man.read_text()))
    return out


def fetch_cohort(local_root: Path, cache: Path, picks: dict[str, list[str]],
                 bundles=SITE_BUNDLES) -> Path:
    """mirror the site-arm subjects into `cache`, skipping what is already there.

    a partial mirror of a public dataset rather than an addition to the acquired
    root, because the root's `.location.yaml` records a complete-subject count and
    a subset that looks like a subject but carries 8 of its 32 bundles would make
    that record a lie.
    """
    urls = manifest_urls(local_root)
    deriv = cache / "derivatives"
    todo: list[tuple[str, Path]] = []
    for _, subs in picks.items():
        for s in subs:
            for path, url in urls.items():
                if f"/{s}/" not in path or url is None:
                    continue
                name = path.rsplit("/", 1)[-1]
                want = (any(name == f"{s}__{b}.trk" for b in bundles)
                        or name in (f"{s}__mask_wm.nii.gz", f"{s}__mask_gm.nii.gz"))
                if not want:
                    continue
                parts = path.strip("/").split("/")
                dest = deriv.joinpath(*parts[1:-1])
                dest.mkdir(parents=True, exist_ok=True)
                if not (dest / name).exists():
                    todo.append((url, dest / name))
    for i, (url, dest) in enumerate(todo):
        t0 = time.time()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as r, tmp.open("wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
        print(f"    [{i + 1}/{len(todo)}] {dest.name} "
              f"{dest.stat().st_size / 2**20:.0f} MiB in {time.time() - t0:.1f}s", flush=True)
    return deriv


def pick_cohort(geometry: list[dict], per_block: int,
                bundles=SITE_BUNDLES) -> dict[str, list[str]]:
    """`per_block` subjects from each block, spread evenly over the id range.

    evenly spaced rather than the first N, because the ids are ordered and the
    first N of a block would sample one end of whatever the ordering within a
    cohort is.  a subject missing more than two of the eight bundles is skipped:
    the site arm compares strengths on a fixed bundle set, and a subject with two
    of them contributes noise about the bundle set rather than about the site.
    """
    have: dict[str, set] = {}
    for row in geometry:
        have.setdefault(row["sub"], set()).add(row["bundle"])
    out: dict[str, list[str]] = {}
    for name, lo, hi in BLOCKS:
        subs = [s for s in sorted(have) if lo <= int(s.split("-")[1]) <= hi]
        ok = [s for s in subs if sum(b in have[s] for b in bundles) >= len(bundles) - 2]
        if not ok:
            continue
        idx = sorted(set(np.linspace(0, len(ok) - 1, per_block).round().astype(int)))
        out[name] = [ok[i] for i in idx]
    return out


# ---------------------------------------------------------------------------
# the cohort geometry table
# ---------------------------------------------------------------------------


def cohort_geometry(local_root: Path, cache: Path) -> list[dict]:
    """one row per (subject, bundle) over all 284: streamline count and volume dims.

    a trk header is 1000 bytes and carries both, so the whole cohort is a few
    megabytes of http range requests rather than 120 GiB of streamlines.  this is
    what the block recovery in the module docstring is computed from, and it is
    also the only measurement here that covers every subject rather than a sample.
    """
    import struct

    cf = cache / "cohort_geometry.json"
    if cf.exists():
        return json.loads(cf.read_text())

    urls = manifest_urls(local_root)
    trks = {p: u for p, u in urls.items() if p.endswith(".trk") and u}
    rows: list[dict] = []
    hdrs = cache / "headers"
    hdrs.mkdir(parents=True, exist_ok=True)
    for i, (path, url) in enumerate(sorted(trks.items())):
        parts = path.strip("/").split("/")
        sub, name = parts[2], parts[-1]
        hp = hdrs / f"{sub}__{name}"
        if not hp.exists():
            req = urllib.request.Request(url, headers={"Range": "bytes=0-999"})
            with urllib.request.urlopen(req) as r:
                hp.write_bytes(r.read(1000))
        h = hp.read_bytes()
        if len(h) < 1000 or h[:5] != b"TRACK":
            continue
        rows.append({"sub": sub, "split": parts[1],
                     "bundle": name.split("__")[1][:-4],
                     "n": struct.unpack_from("<i", h, 988)[0],
                     "dims": list(struct.unpack_from("<3h", h, 6))})
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(trks)} headers", flush=True)
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(rows))
    return rows


# ---------------------------------------------------------------------------
# normalization and parcellation
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """the 7-parameter map from one subject's voxel grid into the shared bins.

    translation and one scale per axis, taken from the first two moments of the
    subject's own white+grey mask.  that is the whole transform.  a rotation would
    need an axis assignment and the brain's second and third principal axes are
    close enough in length that the assignment flips between subjects; a fitted
    warp would let two subjects be deformed into agreement, which is the one thing
    a between-subject measurement must not allow.
    """

    centre: np.ndarray
    scale: np.ndarray
    ref_centre: np.ndarray
    ref_scale: np.ndarray

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return (xyz - self.centre) / self.scale * self.ref_scale + self.ref_centre


def mask_moments(subj: Subject) -> tuple[np.ndarray, np.ndarray, int]:
    import nibabel as nib

    wm = nib.load(subj.mask("wm")).get_fdata(dtype=np.float32) > 0.5
    gm = nib.load(subj.mask("gm")).get_fdata(dtype=np.float32) > 0.5
    b = wm | gm
    idx = np.argwhere(b).astype(np.float64)
    return idx.mean(0), idx.std(0), int(b.shape[2])


@dataclass
class Bins:
    """cubic bins in the normalized frame, kept where the cohort actually ends.

    a bin exists if at least `keep_frac` of the subjects put at least
    `min_points` endpoints in it.  defining it from the cohort rather than from
    any one subject is the same decision the phantom script took when it defined
    its bins from the ground truth: otherwise a subject that terminates in odd
    places enlarges the universe it is then scored against.
    """

    bin_mm: float
    origin: np.ndarray
    shape: tuple[int, int, int]
    index: np.ndarray
    n: int

    def linear(self, xyz: np.ndarray) -> np.ndarray:
        idx = np.floor((xyz - self.origin) / self.bin_mm).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=1)
        out = np.full(len(xyz), -1, np.int64)
        out[ok] = (idx[ok, 0] * self.shape[1] + idx[ok, 1]) * self.shape[2] + idx[ok, 2]
        return out

    def parcels_of(self, xyz: np.ndarray) -> np.ndarray:
        lin = self.linear(xyz)
        out = np.full(len(xyz), -1, np.int64)
        ok = lin >= 0
        out[ok] = self.index[lin[ok]]
        return out


def build_bins(clouds: dict[str, np.ndarray], bin_mm: float, min_points: int,
               keep_frac: float) -> Bins:
    lo = np.min([c.min(0) for c in clouds.values()], axis=0)
    hi = np.max([c.max(0) for c in clouds.values()], axis=0)
    origin = np.floor(lo / bin_mm) * bin_mm
    shape = tuple(int(np.ceil((hi[i] - origin[i]) / bin_mm)) + 1 for i in range(3))
    n_flat = int(np.prod(shape))
    grid = Bins(bin_mm, origin, shape, np.arange(n_flat), n_flat)
    occ = np.zeros(n_flat, np.int64)
    for c in clouds.values():
        lin = grid.linear(c)
        occ += np.bincount(lin[lin >= 0], minlength=n_flat) >= min_points
    keep = occ >= keep_frac * len(clouds)
    index = np.full(n_flat, -1, np.int64)
    index[keep] = np.arange(int(keep.sum()))
    return Bins(bin_mm, origin, shape, index, int(keep.sum()))


# ---------------------------------------------------------------------------
# per-subject edge tables
# ---------------------------------------------------------------------------


@dataclass
class Ends:
    """a subject's streamline endpoints in the shared frame, plus their bundles."""

    xyz: np.ndarray                  # (m, 2, 3)
    bundle: np.ndarray               # (m,) str
    n: int
    z_extent: int
    per_bundle: dict = field(default_factory=dict)


def load_ends(subj: Subject, frame: Frame, bundles=None) -> Ends:
    keep = tuple(bundles) if bundles else subj.bundles
    E, lab, per = [], [], {}
    for b in keep:
        p = subj.trk(b)
        if not p.exists():
            continue
        ends, _cloud, n = read_trk(p)
        if not len(ends):
            continue
        E.append(ends)
        lab += [b] * len(ends)
        per[b] = n
    if not E:
        return Ends(np.zeros((0, 2, 3), np.float32), np.zeros(0, object), 0, 0, {})
    A = np.concatenate(E).astype(np.float64)
    y = frame(A.reshape(-1, 3)).reshape(-1, 2, 3)
    return Ends(y.astype(np.float32), np.asarray(lab), len(y), 0, per)


def edge_table(ends: Ends, bins: Bins) -> tuple[np.ndarray, np.ndarray, float]:
    """streamline count per unordered parcel pair, and the fraction placed."""
    a = bins.parcels_of(ends.xyz[:, 0, :])
    b = bins.parcels_of(ends.xyz[:, 1, :])
    ok = (a >= 0) & (b >= 0) & (a != b)
    placed = float(ok.mean()) if len(ok) else 0.0
    key = np.minimum(a[ok], b[ok]) * bins.n + np.maximum(a[ok], b[ok])
    u, c = np.unique(key, return_counts=True)
    return u, c, placed


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def mean_pairwise_correlation(E: np.ndarray) -> float:
    """mean off-diagonal correlation of row-centred error vectors.

    identical in form to the phantom script's function of the same name, and
    deliberately so: the quantity a source card calls `correlated_fraction` is
    defined by a one-factor model whose only free parameter is this correlation,
    and computing it two different ways in two scripts would make the two
    measurements incomparable for no gain.
    """
    X = np.asarray(E, float)
    X = X - X.mean(1, keepdims=True)
    sd = np.sqrt((X ** 2).sum(1))
    live = sd > 0
    if live.sum() < 2:
        return float("nan")
    Y = X[live] / sd[live][:, None]
    R = Y @ Y.T
    m = R.shape[0]
    return float(R[~np.eye(m, dtype=bool)].mean())


def effective_n(n: int, rho: float) -> float:
    """`n` observations sharing a fraction `rho` of their error are worth how many.

    routed through `ibm.runtime.fuse` exactly as the phantom script routes it,
    because the point of that module is that this arithmetic has one home and
    because the ceiling it produces -- 1/rho, whatever n is -- is the finding.
    """
    if not np.isfinite(rho):
        return float("nan")
    rho = float(np.clip(rho, 0.0, 1.0 - 1e-9))
    try:
        from ibm.runtime.fuse import TeacherPrecision
        tp = TeacherPrecision(r2=0.0, correlated_fraction=rho, error_rank=1,
                              source="tractoinferno subjects")
        ev = tp.evidence("structural.axonal_density", np.zeros(n), np.ones(n))
        return float(ev.effective_constraints())
    except Exception:
        return n / ((1.0 - rho) + n * rho)


def strength_matrix(tables: dict[str, tuple], subs: list[str], n_pairs: int,
                    n_bins: int) -> np.ndarray:
    """(subjects, pairs) of streamline count normalized by the subject's own total.

    normalized because a subject's total streamline yield is a property of the
    seeding and of how many bundles rbx managed to segment, not of their
    connectivity -- the same normalization the phantom applied across pipelines,
    for the same reason.
    """
    lo = np.arange(n_bins * n_bins) // n_bins
    hi = np.arange(n_bins * n_bins) % n_bins
    valid = lo < hi
    X = np.zeros((len(subs), n_bins * n_bins))
    for i, s in enumerate(subs):
        u, c, _placed = tables[s]
        X[i, u] = c / max(c.sum(), 1)
    return X[:, valid]


def variance_components(L: np.ndarray, groups: np.ndarray) -> dict:
    """between-group and within-group variance of a log-strength column.

    a one-way random-effects split, computed from the group means rather than by
    an anova table, with the between-group term debiased by the within-group term
    it contains -- without that correction a partition into groups of 12 reports a
    between-group variance of roughly var/12 even when the grouping is noise, and
    the number would look like a site effect.  clipped at zero, which is where an
    estimate of a variance is allowed to be but a variance is not.
    """
    gs = np.unique(groups)
    means = np.array([np.nanmean(L[groups == g], axis=0) for g in gs])
    ns = np.array([int((groups == g).sum()) for g in gs])
    within = np.array([np.nanvar(L[groups == g], axis=0, ddof=1) for g in gs])
    w = np.nansum(within * (ns - 1)[:, None], 0) / max(int(ns.sum() - len(gs)), 1)
    raw = np.nanvar(means, axis=0, ddof=1)
    between = np.maximum(raw - w * np.mean(1.0 / ns), 0.0)
    return {"between": between, "within": w, "group_means": means, "groups": gs}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--bin-mm", type=float, default=20.0,
                    help="cubic bin side in the normalized frame.  this is the parcellation")
    ap.add_argument("--min-bin-points", type=int, default=200,
                    help="endpoints a subject must put in a bin for it to count as occupied")
    ap.add_argument("--keep-frac", type=float, default=0.75,
                    help="fraction of subjects that must occupy a bin for it to exist")
    ap.add_argument("--min-z", type=int, default=114,
                    help="volumes shallower than this are field-of-view truncated and are "
                         "held out of the main arm; they are reported separately")
    ap.add_argument("--per-block", type=int, default=12,
                    help="subjects per acquisition block in the site arm")
    ap.add_argument("--no-fetch", action="store_true",
                    help="skip the site arm rather than streaming its subjects")
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from ibm.forge.bind import repo_root
    import yaml

    root = args.root or repo_root(Path(__file__))
    cache = args.cache or (root / ".cache" / "group_connectome")
    cache.mkdir(parents=True, exist_ok=True)
    loc = yaml.safe_load(
        (root / "data" / "sources" / "tractoinferno" / "raw" / ".location.yaml").read_text())
    local_root = Path(loc["local_root"])
    deriv = local_root / "derivatives"
    if not deriv.is_dir():
        print(f"tractoinferno derivatives not found at {deriv}", file=sys.stderr)
        return 2

    out: dict = {"measured_at": time.strftime("%Y-%m-%d"),
                 "sources": ["tractoinferno (openneuro ds003900 v1.1.1)"],
                 "parameters": {"bin_mm": args.bin_mm, "min_bin_points": args.min_bin_points,
                                "keep_frac": args.keep_frac, "min_z": args.min_z,
                                "per_block": args.per_block}}

    # ---------------- cohort geometry, all 284 --------------------------------
    print("cohort geometry (trk headers over all 284 subjects)")
    geom = cohort_geometry(local_root, cache)
    dims = {r["sub"]: r["dims"] for r in geom}
    bundles_of: dict[str, set] = {}
    counts_of: dict[str, dict] = {}
    for r in geom:
        bundles_of.setdefault(r["sub"], set()).add(r["bundle"])
        counts_of.setdefault(r["sub"], {})[r["bundle"]] = r["n"]
    all_b = sorted({b for v in bundles_of.values() for b in v})
    avail = {b: sum(b in v for v in bundles_of.values()) / len(bundles_of) for b in all_b}
    blocks_n = {name: sum(1 for s in dims if block_of(s) == name) for name, _, _ in BLOCKS}
    zvals, zcounts = np.unique([d[2] for d in dims.values()], return_counts=True)
    out["cohort"] = {
        "n_subjects": len(dims), "n_tractograms": len(geom),
        "bundles": len(all_b),
        "bundle_availability": {k: round(v, 4) for k, v in sorted(avail.items())},
        "mean_bundles_per_subject": round(float(np.mean([len(v) for v in bundles_of.values()])), 2),
        "acquisition_blocks": blocks_n,
        # the histogram the block recovery rests on.  the empty span between 120
        # and 130 is the finding: a brain's depth is continuous and a field of
        # view is not.
        "z_extent_histogram": {str(int(k)): int(v) for k, v in zip(zvals, zcounts)},
    }
    print(f"    {len(dims)} subjects, {len(geom)} tractograms, {len(all_b)} bundle names")
    print(f"    bundle availability {min(avail.values()):.2f}-{max(avail.values()):.2f}; "
          f"no bundle is present in every subject")
    print(f"    acquisition blocks {blocks_n}")

    # a cohort-wide strength spread that needs no registration at all: the
    # streamline count of a named bundle, which is defined per subject without
    # placing anything in a common frame.  it is a weaker observable than an edge
    # -- a bundle is one edge of a 32-edge connectome -- but it is measured on all
    # 284 rather than on a sample, so it says whether the sample is atypical.
    per_b = {}
    for b in all_b:
        tot = {s: sum(counts_of[s].values()) for s in counts_of if b in counts_of[s]}
        v = np.array([counts_of[s][b] / tot[s] for s in tot if tot[s] > 0])
        v = v[v > 0]
        if len(v) > 3:
            per_b[b] = float(np.std(np.log(v), ddof=1))
    out["cohort"]["bundle_strength_log_sd"] = {k: round(v, 4) for k, v in sorted(per_b.items())}
    out["cohort"]["bundle_strength_log_sd_median"] = round(float(np.median(list(per_b.values()))), 4)
    print(f"    per-bundle share of a subject's streamlines: log sd across subjects "
          f"median {np.median(list(per_b.values())):.3f} "
          f"(x{np.exp(np.median(list(per_b.values()))):.2f}), all 284 subjects")

    # ---------------- the local arm: full bundle sets --------------------------
    local = discover(deriv, "local")
    print(f"\nlocal root: {len(local)} subjects with tractograms under {deriv}")
    moments = {s: mask_moments(v) for s, v in local.items()}
    full = [s for s in local if moments[s][2] >= args.min_z]
    trunc = [s for s in local if moments[s][2] < args.min_z]
    ref_c = np.median([moments[s][0] for s in full], 0)
    ref_s = np.median([moments[s][1] for s in full], 0)
    frames = {s: Frame(moments[s][0], moments[s][1], ref_c, ref_s) for s in local}
    print(f"    {len(full)} with full head coverage, {len(trunc)} field-of-view truncated "
          f"({', '.join(sorted(trunc)) or 'none'})")

    t0 = time.time()
    ends = {s: load_ends(local[s], frames[s]) for s in sorted(local)}
    print(f"    endpoints for {len(ends)} subjects in {time.time() - t0:.0f}s "
          f"({sum(e.n for e in ends.values()):,} streamlines)")

    # registration residual, measured rather than asserted
    cent: dict[str, list] = {}
    for s, e in ends.items():
        if s not in full:
            continue
        for b in np.unique(e.bundle):
            cent.setdefault(b, []).append(e.xyz[e.bundle == b].reshape(-1, 3).mean(0))
    resid = {b: float(np.linalg.norm(np.std(np.array(v), axis=0)))
             for b, v in cent.items() if len(v) >= 5}
    out["registration"] = {
        "method": "7-parameter: centroid and per-axis sd of the subject's own white+grey "
                  "mask, matched to the cohort median.  no rotation, no fitted warp",
        "bundle_endpoint_centroid_scatter_mm": {k: round(v, 2) for k, v in sorted(resid.items())},
        "median_mm": round(float(np.median(list(resid.values()))), 2),
        "range_mm": [round(min(resid.values()), 2), round(max(resid.values()), 2)],
    }
    print(f"    bundle endpoint centroid scatter across subjects: "
          f"{min(resid.values()):.1f}-{max(resid.values()):.1f} mm, "
          f"median {np.median(list(resid.values())):.1f} -- inside a {args.bin_mm:g} mm bin")

    clouds = {s: ends[s].xyz.reshape(-1, 3) for s in full}
    bins = build_bins(clouds, args.bin_mm, args.min_bin_points, args.keep_frac)
    tables = {s: edge_table(ends[s], bins) for s in sorted(local)}
    n_pairs = bins.n * (bins.n - 1) // 2
    placed = np.array([tables[s][2] for s in full])
    print(f"\nparcellation: {bins.n} bins of {args.bin_mm:g} mm, {n_pairs} possible pairs; "
          f"{placed.mean():.1%} of endpoints placed (range {placed.min():.1%}-{placed.max():.1%})")

    X = strength_matrix(tables, full, n_pairs, bins.n)
    B = X > 0
    n = len(full)
    p = B.mean(0)
    seen = p > 0
    out["parcellation"] = {
        "kind": "cubic spatial binning in a 7-parameter normalized frame, NOT an atlas",
        "bin_mm": args.bin_mm, "n_bins": bins.n, "possible_pairs": n_pairs,
        "endpoints_placed_mean": round(float(placed.mean()), 4),
        "n_subjects": n, "subjects": sorted(full),
        "blocks_represented": sorted({block_of(s) for s in full}),
    }

    # ---------------- between subjects ----------------------------------------
    thresholds = {}
    for thr in (0.0, 1e-5, 1e-4, 1e-3):
        Bt = X > thr
        pt = Bt.mean(0)
        st = pt > 0
        thresholds[f"{thr:g}"] = {
            "edges_asserted_by_someone": int(st.sum()),
            "fraction_of_possible_pairs": round(float(st.mean()), 4),
            "found_by_all": int((pt == 1).sum()),
            "found_by_exactly_one": int((pt * n == 1).sum()),
            "mean_existence_probability": round(float(pt[st].mean()), 4),
            "median_existence_probability": round(float(np.median(pt[st])), 4),
        }
    print("\nbetween subjects -- edge existence")
    for k, v in thresholds.items():
        print(f"    strength > {k:>6}: {v['edges_asserted_by_someone']:5d} edges, "
              f"{v['found_by_all']:3d} found by all {n}, "
              f"{v['found_by_exactly_one']:4d} by exactly one, "
              f"mean P(edge) {v['mean_existence_probability']:.3f}")

    # conditional on the bundle that carries the edge having been segmented
    owner: dict[int, str] = {}
    tally: dict[int, dict] = {}
    per_bundle_tab: dict[tuple[str, str], set] = {}
    for s in full:
        e = ends[s]
        a = bins.parcels_of(e.xyz[:, 0, :])
        b = bins.parcels_of(e.xyz[:, 1, :])
        ok = (a >= 0) & (b >= 0) & (a != b)
        key = np.minimum(a, b) * bins.n + np.maximum(a, b)
        for bun in np.unique(e.bundle):
            m = ok & (e.bundle == bun)
            u, c = np.unique(key[m], return_counts=True)
            per_bundle_tab[(s, bun)] = set(u.tolist())
            for kk, vv in zip(u.tolist(), c.tolist()):
                tally.setdefault(kk, {}).setdefault(bun, 0)
                tally[kk][bun] += vv
    for kk, d in tally.items():
        owner[kk] = max(d, key=d.get)
    have = {s: set(np.unique(ends[s].bundle).tolist()) for s in full}
    p_cond, p_unc, availability = [], [], []
    for kk, ob in owner.items():
        elig = [s for s in full if ob in have[s]]
        if not elig:
            continue
        got = sum(1 for s in elig if kk in per_bundle_tab.get((s, ob), ()))
        p_cond.append(got / len(elig))
        p_unc.append(sum(1 for s in full if any(kk in per_bundle_tab.get((s, bb), ())
                                                for bb in have[s])) / n)
        availability.append(len(elig) / n)
    p_cond = np.array(p_cond)
    p_unc = np.array(p_unc)
    print(f"    P(edge | any subject)                        {p_unc.mean():.3f}")
    print(f"    P(edge | the subject has the owning bundle)  {p_cond.mean():.3f}  "
          f"(mean bundle availability {np.mean(availability):.3f})")

    # strength spread
    strength = {}
    for frac, label in ((0.5, "half"), (0.75, "three_quarters"), (1.0, "all")):
        sel = B.sum(0) >= max(int(round(frac * n)), 2)
        if sel.sum() < 5:
            continue
        L = np.log(np.where(X[:, sel] > 0, X[:, sel], np.nan))
        sd = np.nanstd(L, axis=0, ddof=1)
        strength[label] = {"n_edges": int(sel.sum()),
                           "log_sd_mean": round(float(np.nanmean(sd)), 4),
                           "log_sd_median": round(float(np.nanmedian(sd)), 4),
                           "factor": round(float(np.exp(np.nanmean(sd))), 3)}
    print("\nbetween subjects -- edge strength")
    for k, v in strength.items():
        print(f"    edges in >= {k:15s} {v['n_edges']:5d}: log sd {v['log_sd_mean']:.3f} "
              f"-> a factor of x{v['factor']:.2f}")

    # error correlation, with the group consensus standing in for a truth
    selu = B.sum(0) >= 1
    Braw = B[:, selu].astype(float)
    cons = (p[selu] >= 0.5).astype(float)
    rho_dis = mean_pairwise_correlation(np.abs(Braw - cons[None, :]))
    rho_raw = mean_pairwise_correlation(Braw)
    selall = B.sum(0) >= n
    rho_log = (mean_pairwise_correlation(np.log(X[:, selall]))
               if selall.sum() >= 5 else float("nan"))
    neff = effective_n(n, rho_dis)
    out["between_subject"] = {
        "existence": thresholds,
        "mean_existence_probability": round(float(p_unc.mean()), 4),
        "mean_existence_probability_given_bundle": round(float(p_cond.mean()), 4),
        "mean_bundle_availability": round(float(np.mean(availability)), 4),
        "strength": strength,
        "strength_log_sd": strength.get("half", {}).get("log_sd_mean"),
        "error_correlation": {
            "rho_disagreement_with_group_consensus": round(rho_dis, 4),
            "rho_binary_connectomes_uncentred": round(rho_raw, 4),
            "rho_log_strength": None if not np.isfinite(rho_log) else round(rho_log, 4),
            "effective_independent_subjects": round(neff, 3),
            "ceiling_however_many_subjects": round(1.0 / max(rho_dis, 1e-9), 2),
            "read_it_as": "a LOWER bound on the shared fraction.  the reference is the "
                          "subjects' own consensus, so any error every subject makes is "
                          "invisible to it -- and all 284 went through ONE pipeline, where "
                          "the phantom's 0.317 was measured across twenty",
        },
    }
    print("\nbetween subjects -- error correlation")
    print(f"    rho (disagreement with the group's own consensus) {rho_dis:.3f}")
    print(f"    {n} subjects are worth {neff:.2f} independent ones; the ceiling is "
          f"{1.0 / max(rho_dis, 1e-9):.1f} however many arrive")

    # ---------------- what the group explains of an individual -----------------
    print("\nleave-one-subject-out: what a group connectome explains")
    from scipy import stats as _st
    aucs, briers, accs, r2s, rs, sds = [], [], [], [], [], []
    for i in range(n):
        o = np.delete(np.arange(n), i)
        pg = B[o].mean(0)
        y = B[i]
        sel = (pg > 0) | y
        rank = _st.rankdata(pg[sel])
        yy = y[sel]
        n1, n0 = int(yy.sum()), int((~yy).sum())
        if n1 and n0:
            aucs.append((rank[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
            briers.append(float(np.mean((pg[sel] - yy) ** 2)))
            accs.append(float(np.mean((pg[sel] >= 0.5) == yy)))
        sel2 = (B[o].sum(0) >= len(o) * 0.5) & B[i]
        if sel2.sum() < 50:
            continue
        g = np.log(np.where(X[o][:, sel2] > 0, X[o][:, sel2], np.nan))
        gm = np.nanmean(g, axis=0)
        yv = np.log(X[i, sel2])
        # both centred: a subject's overall streamline yield is seeding and
        # segmentation success, not connectivity, and leaving it in would let the
        # group score credit for predicting how many streamlines rbx emitted
        yc, gc = yv - yv.mean(), gm - gm.mean()
        r2s.append(1 - float(np.sum((yc - gc) ** 2) / np.sum(yc ** 2)))
        rs.append(float(np.corrcoef(yc, gc)[0, 1]))
        sds.append(float(np.std(yc - gc)))
    selv = B.sum(0) >= n * args.keep_frac
    Lv = np.log(np.where(X[:, selv] > 0, X[:, selv], np.nan))
    Lv = Lv - np.nanmean(Lv, axis=1, keepdims=True)
    prof = np.nanmean(Lv, axis=0)
    tot_v = float(np.nanvar(Lv))
    grp_v = float(np.nanvar(prof))
    out["group_cost"] = {
        "existence_auc": round(float(np.mean(aucs)), 4),
        "existence_auc_sd": round(float(np.std(aucs)), 4),
        "existence_accuracy_at_half": round(float(np.mean(accs)), 4),
        "existence_brier": round(float(np.mean(briers)), 4),
        "log_strength_r2": round(float(np.mean(r2s)), 4),
        "log_strength_r2_sd": round(float(np.std(r2s)), 4),
        "log_strength_r": round(float(np.mean(rs)), 4),
        "residual_log_sd": round(float(np.mean(sds)), 4),
        "residual_factor": round(float(np.exp(np.mean(sds))), 3),
        "variance_share_group_profile": round(grp_v / tot_v, 4),
        "variance_share_individual": round(1 - grp_v / tot_v, 4),
    }
    print(f"    held-out edge existence: AUC {np.mean(aucs):.3f}, "
          f"accuracy {np.mean(accs):.3f} at p>=0.5")
    print(f"    held-out edge strength : R2 {np.mean(r2s):.3f} (r {np.mean(rs):.3f}), "
          f"residual x{np.exp(np.mean(sds)):.2f}")
    print(f"    of a subject's log-strength variance, {grp_v / tot_v:.1%} is the group "
          f"profile and {1 - grp_v / tot_v:.1%} is theirs alone")

    # ---------------- the site arm --------------------------------------------
    if args.no_fetch:
        out["between_block"] = {"measured": False,
                                "why": "--no-fetch: the site arm was not run"}
    else:
        print("\nsite arm: acquisition blocks B, C and D are not in the local root")
        picks = pick_cohort(geom, args.per_block)
        picks_local = {"A": [s for s in sorted(full) if block_of(s) == "A"]}
        idx = np.linspace(0, len(picks_local["A"]) - 1, args.per_block).round().astype(int)
        picks_local["A"] = [picks_local["A"][i] for i in sorted(set(idx))]
        need = {k: v for k, v in picks.items() if k != "A"}
        cdir = fetch_cohort(local_root, cache / "cohort", need)
        remote = discover(cdir, "cohort")
        chosen: dict[str, Subject] = {}
        groups: list[str] = []
        for blk, subs in list(picks_local.items()) + list(need.items()):
            for s in subs:
                src = local.get(s) or remote.get(s)
                if src is None:
                    continue
                chosen[s] = src
                groups.append(blk)
        print(f"    {len(chosen)} subjects: " +
              ", ".join(f"{b}={sum(1 for g in groups if g == b)}"
                        for b in sorted(set(groups))))

        mom2 = {s: (moments[s] if s in moments else mask_moments(v))
                for s, v in chosen.items()}
        rc = np.median([mom2[s][0] for s in chosen], 0)
        rs2 = np.median([mom2[s][1] for s in chosen], 0)
        e2 = {s: load_ends(v, Frame(mom2[s][0], mom2[s][1], rc, rs2), SITE_BUNDLES)
              for s, v in chosen.items()}
        keep_s = [s for s in sorted(chosen) if e2[s].n > 0]
        g2 = np.array([groups[list(chosen).index(s)] for s in keep_s])
        bins2 = build_bins({s: e2[s].xyz.reshape(-1, 3) for s in keep_s},
                           args.bin_mm, args.min_bin_points, args.keep_frac)
        tab2 = {s: edge_table(e2[s], bins2) for s in keep_s}
        X2 = strength_matrix(tab2, keep_s, bins2.n * (bins2.n - 1) // 2, bins2.n)
        B2 = X2 > 0
        sel = B2.sum(0) >= max(int(round(0.75 * len(keep_s))), 4)
        L2 = np.log(np.where(X2[:, sel] > 0, X2[:, sel], np.nan))
        L2 = L2 - np.nanmean(L2, axis=1, keepdims=True)
        vc = variance_components(L2, g2)
        btw = float(np.nanmean(vc["between"]))
        wit = float(np.nanmean(vc["within"]))
        # the null: the same statistic with the block labels shuffled.  without it
        # a between-group variance is uninterpretable, because any partition of a
        # noisy sample has one.
        rng = np.random.default_rng(0)
        null = []
        for _ in range(500):
            null.append(float(np.nanmean(variance_components(L2, rng.permutation(g2))["between"])))
        null = np.array(null)
        pval = float((null >= btw).mean())

        # existence, per block
        pb = {b: (B2[g2 == b].mean(0)) for b in sorted(set(g2))}
        pb_sd = float(np.mean(np.std(np.array([pb[b][sel] for b in sorted(pb)]), axis=0)))

        # the transfer test: predict a subject from their own block, and from the
        # others.  this is the question "is a different scanner as different as a
        # different person" asked in the only form that has an operational answer.
        within_r2, across_r2 = [], []
        for i, s in enumerate(keep_s):
            own = np.array([j for j in range(len(keep_s)) if g2[j] == g2[i] and j != i])
            oth = np.array([j for j in range(len(keep_s)) if g2[j] != g2[i]])
            if len(own) < 3 or len(oth) < 3:
                continue
            y = L2[i]
            for src, acc in ((own, within_r2), (oth, across_r2)):
                gm = np.nanmean(L2[src], axis=0)
                m = np.isfinite(y) & np.isfinite(gm)
                if m.sum() < 50:
                    continue
                yc = y[m] - y[m].mean()
                gc = gm[m] - gm[m].mean()
                acc.append(1 - float(np.sum((yc - gc) ** 2) / np.sum(yc ** 2)))
        out["between_block"] = {
            "measured": True,
            "what_a_block_is": "an acquisition-geometry block recovered from the "
                               "field-of-view signature and contiguous in subject id; "
                               "block C is exactly 39 subjects and block D exactly 86, "
                               "matching two of the paper's per-site counts.  NOT a site "
                               "label -- block A still holds three cohorts, so every "
                               "number here is a LOWER bound on the between-site spread",
            "blocks": {b: int((g2 == b).sum()) for b in sorted(set(g2))},
            "subjects": {b: [s for s, gg in zip(keep_s, g2) if gg == b] for b in sorted(set(g2))},
            "bundles": list(SITE_BUNDLES),
            "n_bins": bins2.n, "n_edges_compared": int(sel.sum()),
            "between_block_log_var": round(btw, 4),
            "within_block_log_var": round(wit, 4),
            "between_block_log_sd": round(float(np.sqrt(btw)), 4),
            "within_block_log_sd": round(float(np.sqrt(wit)), 4),
            "icc_block": round(btw / max(btw + wit, 1e-12), 4),
            "permutation_null_mean": round(float(null.mean()), 4),
            "permutation_p": pval,
            "existence_sd_across_blocks": round(pb_sd, 4),
            "loso_r2_within_block": round(float(np.mean(within_r2)), 4) if within_r2 else None,
            "loso_r2_across_blocks": round(float(np.mean(across_r2)), 4) if across_r2 else None,
        }
        print(f"    between-block log var {btw:.4f} (sd {np.sqrt(btw):.3f}), "
              f"within-block {wit:.4f} (sd {np.sqrt(wit):.3f}), ICC {btw / (btw + wit):.3f}")
        print(f"    permutation null {null.mean():.4f} +- {null.std():.4f}, p = {pval:.3f}")
        if within_r2 and across_r2:
            print(f"    predicting a held-out subject: R2 {np.mean(within_r2):.3f} from their "
                  f"own block, {np.mean(across_r2):.3f} from the others")

    # ---------------- the truncated subjects ----------------------------------
    if trunc:
        Xt = strength_matrix(tables, sorted(trunc), n_pairs, bins.n)
        Bt = Xt > 0
        out["field_of_view_truncated"] = {
            "subjects": sorted(trunc),
            "why": "z extent below --min-z: the acquisition did not cover the whole head, "
                   "so the moment normalization mis-scales them and their inferior "
                   "terminations are absent rather than sparse",
            "mean_edges": round(float(Bt.sum(1).mean()), 1),
            "mean_edges_full_coverage": round(float(B.sum(1).mean()), 1),
        }
        print(f"\nfield-of-view truncated ({len(trunc)}): {Bt.sum(1).mean():.0f} edges "
              f"against {B.sum(1).mean():.0f} for full coverage -- held out of every "
              f"number above")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, sort_keys=False))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
