"""the cortical sheet the pretraining loops actually run on: surface, not sphere.

`docs/DISCONNECTS.md` row 2.  `scripts/pretrain_video_loop.py:cortical_sites()`
placed its sites on a **spherical shell** area-matched to the measured white
surface, and `cortical_regions()` cut six lobe labels out of it by coordinate
thresholds -- its own docstring called that "a geometric convention on the
spherical proxy, NOT an atlas".  The cost was not cosmetic: **a sphere has no
lateral sulcus**, so the insula is not a separable target on it, and the
interoceptive port entered a 4.5% evenly-spaced subsample of a `frontal` label
that was itself a cut at `y/r > 0.55`.

This module holds the replacement.  Sites are **fsaverage white-surface
vertices** and regions are **`?h.aparc.annot`**, the Desikan-Killiany
parcellation -- 34 gyral labels per hemisphere, 68 in total, insula and all four
cingulate divisions among them.

Three things it is careful about.

**A parcellation is a COVER, not a component.**  `data/sources/desikan2006/card.yaml`
is emphatic and ARCHITECTURE.md §2 is the rule: a label set says which points a
name collects, it does not say what value anything takes at them.  So nothing
here returns a *value* per region.  It returns membership, and membership is used
to say where a port enters and which sites a fascicle could join -- supports for
interaction, never likelihoods.

**Gyral is not areal.**  The same card says so: sulcal and gyral boundaries do
not coincide with areal boundaries, and DK's popularity is a fact about
FreeSurfer's defaults.  It is held here because it is the frame nearly every
published structural connectome is expressed in, which is exactly what makes it a
required crosswalk to the tract topology -- not because it is the right cover.

**Sampling is area-weighted, and on the CPU.**  Vertices of an fsaverage surface
are near-uniform on the *sphere*, not on the *white surface*: sampling them
uniformly would put more sites where the spherical map compressed the sheet.
Weighting each vertex by a third of its incident triangle area makes site density
uniform per mm² of cortex, which is what makes `exp(-d/l)` in millimetres mean
what it says.  The draw runs entirely on a CPU generator and is moved to the
device afterwards, so -- unlike the long-range draw this replaces -- **the same
seed gives the same sheet on cpu and on cuda**.
"""
from __future__ import annotations

import functools
import hashlib
import os

import numpy as np

from ibm.anatomy.systems import DK_GYRI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSAVERAGE = os.path.join(ROOT, "data/sources/fsaverage/raw/fsaverage")
APARC = os.path.join(ROOT, "data/sources/desikan2006/raw/fsaverage/label")
DKT_SUBJECT = os.path.join(ROOT, "data/sources/dkt-atlas/raw/somato-01")

HEMIS = ("lh", "rh")

#: the 68 hemisphere-qualified DK labels, in the order `ibm.anatomy.systems`
#: declares the gyri and with the hemisphere prefixed.  the prefix matters:
#: `ibm/topologies/tract.py` is about fascicles, and the corpus callosum joins
#: homotopic points that a hemisphere-blind label would merge into one node,
#: erasing the single clearest case the topology exists to represent.
REGIONS: tuple[str, ...] = tuple(f"{h}.{g}" for h in HEMIS for g in DK_GYRI)
REGION_INDEX = {r: i for i, r in enumerate(REGIONS)}

#: coarse lobes, as a UNION OF ATLAS LABELS rather than a coordinate cut.  the
#: six names the spherical proxy used are kept so that every existing caller of
#: `region_index(pos, "occipital")` keeps working and keeps meaning roughly what
#: it meant -- and two more are added, which is the entire point of the change:
#: `insula` and `cingulate` did not exist as addressable targets before.
#:
#: paracentral goes to frontal by the usual convention.  it straddles the
#: midline continuation of the central sulcus and could as defensibly be split;
#: it is 1.6% of cortex and it is named here rather than being an accident of a
#: threshold, which is the difference that matters.
LOBE_MEMBERS: dict[str, tuple[str, ...]] = {
    "occipital": ("lateraloccipital", "lingual", "cuneus", "pericalcarine"),
    "temporal": ("superiortemporal", "middletemporal", "inferiortemporal",
                 "bankssts", "fusiform", "transversetemporal", "entorhinal",
                 "temporalpole", "parahippocampal"),
    "parietal": ("superiorparietal", "inferiorparietal", "supramarginal",
                 "precuneus"),
    "frontal": ("superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal",
                "parsopercularis", "parstriangularis", "parsorbitalis",
                "lateralorbitofrontal", "medialorbitofrontal", "frontalpole",
                "paracentral"),
    "precentral": ("precentral",),
    "postcentral": ("postcentral",),
    "insula": ("insula",),
    "cingulate": ("rostralanteriorcingulate", "caudalanteriorcingulate",
                  "posteriorcingulate", "isthmuscingulate"),
}
LOBES: tuple[str, ...] = tuple(LOBE_MEMBERS)

#: every DK gyrus must land in exactly one lobe, or a "lobe" is a subset with an
#: unstated remainder and a port that claims to enter "frontal" enters something
#: smaller.  checked at import because the failure is silent otherwise.
_seen: list[str] = [g for v in LOBE_MEMBERS.values() for g in v]
assert sorted(_seen) == sorted(DK_GYRI), (
    f"lobe cover is not a partition of DK_GYRI: "
    f"missing {sorted(set(DK_GYRI) - set(_seen))}, "
    f"extra {sorted(set(_seen) - set(DK_GYRI))}, "
    f"duplicated {sorted({g for g in _seen if _seen.count(g) > 1})}")
del _seen

#: the interoceptive target, now that there is one.  posterior and anterior
#: insula plus anterior cingulate: `ibm/interoception.py` names exactly these
#: three DK labels as the real target and then says it cannot have them.
INTEROCEPTIVE_REGIONS = ("insula", "rostralanteriorcingulate",
                         "caudalanteriorcingulate")


class SurfaceMissing(RuntimeError):
    """the fsaverage payload is not staged."""


def _need(path: str) -> str:
    if not os.path.exists(path):
        raise SurfaceMissing(
            f"{path} is missing.  the cortical surface and its parcellation are "
            "payloads, not code: run\n\n"
            "    PYTHONPATH=. .venv/bin/python scripts/fetch_cortical_atlases.py\n\n"
            "to stage them (they are gitignored under data/sources/*/raw).")
    return path


def _vertex_areas(xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """a third of each incident triangle's area, per vertex.

    the standard barycentric lumping.  it is not the Voronoi area and does not
    need to be: what it is used for is a sampling weight, where the requirement
    is that the weights sum to the surface area with no vertex systematically
    over- or under-counted, and thirds satisfy that exactly.
    """
    p = xyz[faces]                                        # (F, 3, 3)
    a = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    out = np.zeros(len(xyz))
    for c in range(3):
        np.add.at(out, faces[:, c], a / 3.0)
    return out


@functools.lru_cache(maxsize=4)
def load_sheet(surface: str = "white", annot: str = "aparc") -> dict:
    """the cortical sheet: vertices, per-vertex area, and an atlas label each.

    both hemispheres, concatenated, **cortex only** -- `?h.cortex.label` removes
    the medial wall, and a site there is a site on a piece of surface with no
    cortex under it.  the `unknown` and `corpuscallosum` annot entries are
    dropped for the same reason, and a vertex inside `cortex.label` that still
    carries one is dropped too rather than being quietly assigned somewhere.

    fsaverage lh and rh surfaces already live in one frame (lh spans x in
    [-66, 2] mm, rh in [-1, 67] mm), so no hemisphere offset is applied and the
    euclidean distance between a left and a right site is the real one.
    """
    import nibabel.freesurfer as fsio

    xyz_l, area_l, lab_l, hemi_l, vert_l = [], [], [], [], []
    for hi, h in enumerate(HEMIS):
        v, f = fsio.read_geometry(_need(f"{FSAVERAGE}/surf/{h}.{surface}"))
        a = _vertex_areas(v, f)
        raw, _, names = fsio.read_annot(_need(f"{APARC}/{h}.{annot}.annot"))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        cortex = fsio.read_label(_need(f"{FSAVERAGE}/label/{h}.cortex.label"))

        # map annot entry -> index into REGIONS; -1 for anything that is not one
        # of the 34 gyri (unknown, corpuscallosum, and the -1 no-label code).
        table = np.full(len(names), -1, dtype=np.int64)
        for k, nm in enumerate(names):
            key = f"{h}.{nm}"
            if key in REGION_INDEX:
                table[k] = REGION_INDEX[key]
        lab = np.where(raw >= 0, table[np.clip(raw, 0, None)], -1)

        keep = np.zeros(len(v), dtype=bool)
        keep[cortex] = True
        keep &= lab >= 0
        xyz_l.append(v[keep].astype(np.float64))
        area_l.append(a[keep])
        lab_l.append(lab[keep])
        hemi_l.append(np.full(int(keep.sum()), hi, dtype=np.int64))
        vert_l.append(np.nonzero(keep)[0].astype(np.int64))

    xyz = np.concatenate(xyz_l)
    out = {
        "surface": surface, "annot": annot,
        "xyz": xyz,
        "area": np.concatenate(area_l),
        "region": np.concatenate(lab_l),
        "hemi": np.concatenate(hemi_l),
        "vertex": np.concatenate(vert_l),
        "n_vertices": len(xyz),
        "total_area_mm2": float(np.concatenate(area_l).sum()),
    }
    return out


@functools.lru_cache(maxsize=4)
def _kdtree(surface: str = "white", annot: str = "aparc"):
    from scipy.spatial import cKDTree
    return cKDTree(load_sheet(surface, annot)["xyz"])


def sample_sites(n: int, seed: int = 0, surface: str = "white",
                 annot: str = "aparc"):
    """`n` distinct cortical vertices, drawn with probability proportional to area.

    returns `(xyz (n, 3) float32, region (n,) int64)`.

    the draw is **exponential-race weighted sampling without replacement**
    (Efraimidis-Spirakis): key_i = -log(u_i) / w_i, take the n smallest.  it is
    exact, it is one vectorised pass, and it is a pure function of `seed` -- no
    device generator, no global RNG.  the previous long-range draw was a
    `torch.randint` on a device generator and CLAUDE.md records what that cost:
    seed 0 reproduces on the same device only, and cpu and cuda drew unrelated
    graphs.  positions no longer have that property.

    sites are sorted by their draw order and then by vertex index, so the site
    ORDER is reproducible too.  that matters more than it looks: several heads in
    the pretraining loop read `linspace(0, n-1, read_sites)`, which is a
    statement about site order, and an order that changed between runs would make
    those readouts incomparable.
    """
    sheet = load_sheet(surface, annot)
    v = sheet["n_vertices"]
    if n > v:
        raise ValueError(
            f"asked for {n} sites but the cortex has {v} vertices at "
            f"fsaverage resolution.  sampling with replacement would put two "
            f"sites at one point, which is not a finer sheet.")
    rng = np.random.default_rng(seed)
    w = sheet["area"]
    key = -np.log(rng.random(v)) / np.maximum(w, 1e-12)
    take = np.sort(np.argpartition(key, n - 1)[:n])
    return (sheet["xyz"][take].astype(np.float32),
            sheet["region"][take].copy())


def _fingerprint(pos) -> str:
    a = np.ascontiguousarray(np.asarray(pos, dtype=np.float32))
    return hashlib.sha1(a.tobytes()).hexdigest() + f":{a.shape}"


_REGION_CACHE: dict[str, np.ndarray] = {}


def regions_at(pos, surface: str = "white", annot: str = "aparc") -> np.ndarray:
    """the atlas label of the nearest cortical vertex to each position.

    for sites produced by `sample_sites` the nearest vertex IS the site, so this
    is exact rather than an interpolation; it is written as a lookup so that a
    position that came from somewhere else -- a checkpoint, a materialization,
    a perturbed sheet -- still gets an answer instead of an index error.
    """
    key = _fingerprint(pos) + f"|{surface}|{annot}"
    got = _REGION_CACHE.get(key)
    if got is None:
        _, i = _kdtree(surface, annot).query(
            np.asarray(pos, dtype=np.float64), k=1, workers=-1)
        got = load_sheet(surface, annot)["region"][i]
        if len(_REGION_CACHE) > 8:
            _REGION_CACHE.clear()
        _REGION_CACHE[key] = got
    return got


def resolve(name: str) -> tuple[int, ...]:
    """the REGIONS indices a name selects.

    accepts a lobe (`"occipital"`, `"insula"`, `"cingulate"`), a bare DK gyrus
    (`"lateraloccipital"` -- both hemispheres), or a hemisphere-qualified label
    (`"lh.insula"`).  a name that resolves to nothing raises, with the
    alternatives, because a port that silently selects zero sites is a drive
    that goes nowhere and a readout that reads nothing.
    """
    if name in LOBE_MEMBERS:
        return tuple(REGION_INDEX[f"{h}.{g}"]
                     for g in LOBE_MEMBERS[name] for h in HEMIS)
    if name in REGION_INDEX:
        return (REGION_INDEX[name],)
    if name in DK_GYRI:
        return tuple(REGION_INDEX[f"{h}.{name}"] for h in HEMIS)
    raise KeyError(
        f"{name!r} is not a lobe {LOBES}, a DK gyrus, or a hemisphere-qualified "
        f"label like 'lh.insula'")


def area_fractions(surface: str = "white", annot: str = "aparc") -> dict:
    """what fraction of the white surface each lobe and each gyrus occupies.

    reported rather than assumed.  the spherical proxy's thresholds were set to
    "the published lobe fractions"; this is the measurement those were guessing
    at, on the actual surface, and it is what the interoceptive port's
    `PORT_FRACTION` should be read against.
    """
    sheet = load_sheet(surface, annot)
    a, lab = sheet["area"], sheet["region"]
    tot = a.sum()
    per_region = {r: float(a[lab == i].sum() / tot) for i, r in enumerate(REGIONS)}
    per_lobe = {
        L: float(sum(a[np.isin(lab, resolve(L))]) / tot) for L in LOBES}
    per_gyrus = {
        g: per_region[f"lh.{g}"] + per_region[f"rh.{g}"] for g in DK_GYRI}
    return {"total_area_mm2": float(tot), "lobe": per_lobe,
            "gyrus": per_gyrus, "region": per_region}
