"""subject geometry: the files `ibm.materialize.sites` refuses to invent.

`sites.py` is explicit that nothing there synthesizes anatomy -- the octree, the
geodesic poisson disk and the tree walk are all implemented, and what is missing
when a build stops is a file with a subject's pial surface in it.  this module is
that file, read.  it turns one real reconstruction on disk into the
`GeometrySet` the samplers expect, and it does nothing else: no resampling, no
smoothing, no atlas, no dynamics.

three things it takes seriously, because each of them is a way a head model goes
quietly wrong.

*frames.*  a FreeSurfer reconstruction speaks surface RAS -- the T1's own
tkrRAS, with the volume's `c_ras` offset removed -- and every surface, every BEM
boundary and every voxel label in `subjects/<id>/` is in it.  electrode positions
are not: they are digitised in the instrument's head frame, defined by the
subject's own fiducials, and the only thing that relates the two is a
coregistration someone ran.  so surfaces and volumes come back in one frame,
electrodes come back in theirs, and the transform between them is returned as a
separate object carrying its residual rather than being applied silently.

*occupancy is a fraction, not a mask.*  `VolumeGeometry.occupancy` is soft on
purpose so that an octree leaf straddling the pial surface knows it is half csf.
what this module can honestly supply is limited by what the files contain: an
`aseg` is a hard label map, so the parenchyma occupancy it yields is binary at
the voxel scale and only becomes fractional through the trilinear-free
supersampling done here.  that limit is written into the geometry's `source`
string, which is what ends up in provenance.

*nothing here is a template.*  every function takes a path.  the one convenience
entry point, `sample_subject`, resolves its paths from the `mne-sample` card's
own `.location.yaml` and raises naming that card when the bytes have not been
acquired -- because a hardcoded absolute path is the failure mode that works on
one machine and silently reads nothing on the next.

reading is deliberately done without nibabel.  the two FreeSurfer binary formats
this needs -- the triangle surface and the MGH volume -- are small, fixed and
fully specified, and depending on a package that is not in the environment would
turn "the anatomy is right there on disk" into "not implemented", which is the
lie `MissingData` exists to avoid.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ibm.materialize.sites import (
    DiscreteGeometry, Geometry, GeometrySet, MissingData, SurfaceGeometry, VolumeGeometry,
)

#: the frame a FreeSurfer reconstruction is actually in.  `ibm.frames` calls the
#: subject's native anatomical volume `subject_t1` and the reconstructed sheet
#: `subject_surf`, and declares no warp between them -- correctly, because for a
#: FreeSurfer subject there is no warp: `lh.white`, `bem/outer_skin.surf` and
#: `aseg.mgz`'s tkrRAS are literally the same millimetre coordinates.  declaring
#: the surfaces in `subject_surf` and then asking the materializer for a chain
#: into `subject_t1` would invent a registration step that was never run and
#: attribute a residual to it, so everything read out of `subjects/<id>/` is
#: labelled with the frame it is in: `subject_t1`.
ANATOMICAL_FRAME = "subject_t1"

#: the frame digitised electrode positions arrive in.  `ibm.frames` declares
#: `eeg_cap -> subject_t1` as a digitised warp with a 5 mm residual, which is the
#: honest number for a template montage; a real coregistration does better, and
#: `Coregistration` carries the measured value beside the declared one.
MONTAGE_FRAME = "eeg_cap"

_SURFACE_MAGIC = 0xFFFFFE
_CURV_MAGIC = 0xFFFFFF
_MGH_DTYPE = {0: ">u1", 1: ">i4", 3: ">f4", 4: ">i2"}

#: `aseg` labels that are brain parenchyma: cortical and subcortical grey, white
#: matter, cerebellum, brainstem, and the white-matter hypointensities that are
#: still tissue.  ventricles, the subarachnoid space, choroid plexus and the
#: vessel label are deliberately absent -- `tissue` is declared as parenchyma and
#: `csf_space` is a different support with a different notion of adjacency, so
#: folding csf into the tissue mask would put population state in a ventricle.
ASEG_PARENCHYMA: frozenset[int] = frozenset({
    2, 3, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 26, 27, 28,
    41, 42, 46, 47, 48, 49, 50, 51, 52, 53, 54, 58, 59, 60,
    77, 78, 79, 96, 97, 251, 252, 253, 254, 255,
})

#: `aseg` labels that are csf: the ventricular system, the cisterns FreeSurfer
#: labels, and choroid plexus.  kept beside the parenchyma set rather than
#: inferred from it, because "not tissue" also contains skull, scalp and air.
ASEG_CSF: frozenset[int] = frozenset({4, 5, 14, 15, 24, 31, 43, 44, 63, 72})


# ---------------------------------------------------------------------------
# the two FreeSurfer binary formats
# ---------------------------------------------------------------------------


def read_freesurfer_surface(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """a FreeSurfer triangle surface: vertices in surface RAS (mm) and faces.

    the format is three magic bytes, a free-text comment terminated by a blank
    line, two big-endian counts and then the coordinates and the triangles.  it
    is read here rather than through a library because the alternative in this
    environment is to raise "not implemented" over a file that is sitting on
    disk in a format that has not changed in twenty years.
    """
    p = Path(path)
    if not p.is_file():
        raise MissingData("read_freesurfer_surface", str(p),
                          "a FreeSurfer triangle surface (lh.white, rh.pial, "
                          "bem/outer_skin.surf, ...)",
                          "recon-all on the subject's T1, or mne watershed_bem for the "
                          "conductor boundaries")
    b = p.read_bytes()
    magic = int.from_bytes(b[:3], "big")
    if magic != _SURFACE_MAGIC:
        raise ValueError(f"{p}: magic {magic:#x} is not a FreeSurfer triangle surface "
                         f"({_SURFACE_MAGIC:#x}); a quadrangle or curvature file was passed")
    off = b.index(b"\n", 3) + 1
    if b[off:off + 1] == b"\n":
        off += 1
    n_v, n_f = struct.unpack(">ii", b[off:off + 8])
    off += 8
    v = np.frombuffer(b, ">f4", n_v * 3, off).reshape(n_v, 3).astype(np.float64)
    off += n_v * 12
    f = np.frombuffer(b, ">i4", n_f * 3, off).reshape(n_f, 3).astype(np.int64)
    return v, f


def read_freesurfer_curv(path: str | Path) -> np.ndarray:
    """a per-vertex scalar (`lh.thickness`, `lh.curv`) in the "new" curv format."""
    p = Path(path)
    if not p.is_file():
        raise MissingData("read_freesurfer_curv", str(p), "a per-vertex FreeSurfer scalar",
                          "recon-all writes lh.thickness / rh.thickness beside the surfaces")
    b = p.read_bytes()
    magic = int.from_bytes(b[:3], "big")
    if magic != _CURV_MAGIC:
        raise ValueError(f"{p}: magic {magic:#x} is not a new-format curv file")
    n_v, _n_f, per = struct.unpack(">iii", b[3:15])
    return np.frombuffer(b, ">f4", n_v * per, 15).reshape(n_v, per).astype(np.float64).squeeze()


@dataclass(frozen=True)
class MGHVolume:
    """an MGH/MGZ volume with both of its affines, and why both are kept.

    `vox2ras` puts a voxel in the scanner's RAS; `vox2ras_tkr` puts it in surface
    RAS, which is the frame the reconstructed surfaces and the BEM boundaries
    live in.  the two differ by `c_ras`, a pure translation of a few centimetres,
    and using the wrong one is the classic way a source grid ends up shifted off
    the cortex while every number involved stays entirely plausible.  so both are
    carried and the caller has to say which frame it wants.
    """

    data: np.ndarray
    vox2ras: np.ndarray
    vox2ras_tkr: np.ndarray
    c_ras: np.ndarray
    zooms: np.ndarray
    path: str = ""

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)


def read_mgh(path: str | Path) -> MGHVolume:
    """an MGH/MGZ volume: the 284-byte header, the data, and the two affines."""
    p = Path(path)
    if not p.is_file():
        raise MissingData("read_mgh", str(p), "a FreeSurfer MGH/MGZ volume (T1.mgz, aseg.mgz)",
                          "recon-all writes them into subjects/<id>/mri/")
    opener = gzip.open if p.suffix == ".mgz" else open
    with opener(p, "rb") as fh:                                     # type: ignore[operator]
        b = fh.read()
    _ver, w, h, d, n_frames, dtype, _dof = struct.unpack(">iiiiiii", b[:28])
    good_ras = struct.unpack(">h", b[28:30])[0]
    if dtype not in _MGH_DTYPE:
        raise ValueError(f"{p}: MGH type {dtype} is not one of {sorted(_MGH_DTYPE)}")
    zooms = np.array(struct.unpack(">fff", b[30:42]), float)
    dircos = np.array(struct.unpack(">fffffffff", b[42:78]), float).reshape(3, 3).T
    c_ras = np.array(struct.unpack(">fff", b[78:90]), float)
    if not good_ras:
        # the header says its direction cosines are not trustworthy.  refusing is
        # the only honest move: a volume whose orientation is unknown cannot be
        # placed in a frame, and guessing LIA here would put the mask in a
        # plausible-looking wrong place.
        raise ValueError(f"{p}: goodRASFlag is unset, so this volume declares no orientation "
                         "and cannot be brought into an anatomical frame")
    data = (np.frombuffer(b, _MGH_DTYPE[dtype], w * h * d * n_frames, 284)
            .reshape(n_frames, d, h, w).transpose(3, 2, 1, 0))
    if n_frames == 1:
        data = data[..., 0]
    m = dircos * zooms
    vox2ras = np.eye(4)
    vox2ras[:3, :3] = m
    vox2ras[:3, 3] = c_ras - m @ (np.array([w, h, d], float) / 2.0)
    tkr = np.array([[-zooms[0], 0.0, 0.0, zooms[0] * w / 2.0],
                    [0.0, 0.0, zooms[2], -zooms[2] * d / 2.0],
                    [0.0, -zooms[1], 0.0, zooms[1] * h / 2.0],
                    [0.0, 0.0, 0.0, 1.0]])
    return MGHVolume(np.ascontiguousarray(data), vox2ras, tkr, c_ras, zooms, str(p))


# ---------------------------------------------------------------------------
# turning a closed surface into an occupancy field
# ---------------------------------------------------------------------------


def occupancy_from_closed_surface(vertices: np.ndarray, faces: np.ndarray, *,
                                  voxel_mm: float = 1.0, margin_mm: float = 3.0,
                                  supersample: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """rasterize a watertight surface and fill it: (occupancy, affine).

    the conductor boundary this is applied to is a *surface*, and the octree needs
    a *volume*.  the conversion is done by rasterizing every triangle densely
    enough that the shell it paints has no holes at the raster's own scale, then
    flood-filling the interior, rather than by an inside/outside test per query
    point: `octree_sites` evaluates occupancy at nine points per cell per level,
    which is millions of queries, and a ray cast per query would dominate the
    build by orders of magnitude while a 1 mm raster is exact to well under the
    coarsest spacing any request here asks for.

    the returned occupancy is binary, and that is a statement about the input: a
    BEM boundary *is* a hard surface, so there is no partial volume to represent.
    a tissue mask read from an `aseg` is the same and for the same reason.
    """
    from scipy import ndimage

    v = np.asarray(vertices, float).reshape(-1, 3)
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    lo = v.min(axis=0) - margin_mm
    hi = v.max(axis=0) + margin_mm
    shape = np.maximum(np.ceil((hi - lo) / voxel_mm).astype(int) + 1, 2)

    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    longest = max(float(np.linalg.norm(b - a, axis=1).max()),
                  float(np.linalg.norm(c - a, axis=1).max()),
                  float(np.linalg.norm(c - b, axis=1).max()))
    n = max(int(np.ceil(longest * supersample / voxel_mm)), 1)
    # barycentric lattice on the reference triangle, shared by every face
    ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = (ii + jj) <= n
    u = (ii[keep] / n).astype(float)
    w = (jj[keep] / n).astype(float)
    t = 1.0 - u - w

    grid = np.zeros(tuple(shape), bool)
    chunk = max(1, int(4_000_000 / max(len(u), 1)))
    for s in range(0, len(f), chunk):
        pts = (t[None, :, None] * a[s:s + chunk, None, :]
               + u[None, :, None] * b[s:s + chunk, None, :]
               + w[None, :, None] * c[s:s + chunk, None, :]).reshape(-1, 3)
        idx = np.rint((pts - lo) / voxel_mm).astype(np.int64)
        np.clip(idx, 0, np.array(shape) - 1, out=idx)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    filled = ndimage.binary_fill_holes(grid)
    if filled is None:                                             # pragma: no cover
        filled = grid
    affine = np.eye(4)
    affine[:3, :3] = np.eye(3) * voxel_mm
    affine[:3, 3] = lo
    return filled.astype(np.float32), affine


def _bounds(occ: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """the mm bounding box of the occupied voxels, not of the array.

    an octree's root cube is the largest side of these bounds, and taking them
    from the array would make the root the size of the scan's field of view --
    one extra level of refinement over empty air, everywhere, for every build.
    """
    idx = np.array(np.nonzero(occ > 0.0))
    if not idx.size:
        raise MissingData("geometry", "an occupied voxel",
                          "the mask is empty everywhere, so grid(R, r) has nothing to sample",
                          "the segmentation this mask was built from")
    corners = np.stack([idx.min(axis=1) - 0.5, idx.max(axis=1) + 0.5])
    h = np.concatenate([corners, np.ones((2, 1))], axis=1) @ np.asarray(affine, float).T
    return np.stack([h[:, :3].min(axis=0), h[:, :3].max(axis=0)])


# ---------------------------------------------------------------------------
# a reconstruction
# ---------------------------------------------------------------------------


def from_freesurfer(subject_dir: str | Path, *, surface: str = "white",
                    depth_levels: int = 6, voxel_mm: float = 1.0,
                    frame: str = ANATOMICAL_FRAME) -> dict[str, Geometry]:
    """cortical_surface, cortical_depth and tissue from one `subjects/<id>/`.

    the sheet is taken from `?h.white` rather than `?h.pial` by default, and the
    choice is not cosmetic.  the electromagnetic field is sourced by the
    transmembrane current of pyramidal cells whose somata sit near the grey/white
    boundary and whose dendritic trees point along the local normal; the white
    surface's normal is the current's direction, and the pial surface's normal is
    the same direction contaminated by where the cortex happened to bulge.  the
    two hemispheres are concatenated into one mesh with disjoint components,
    which is exactly right: there is no cortico-cortical *surface* adjacency
    across the midline, and a geodesic sampler that could walk from one
    hemisphere to the other would be describing a sheet nobody has.

    `tissue` comes from `aseg` and not from the ribbon, because the support is
    declared as the whole parenchyma -- subcortical structures, brainstem and
    cerebellum included -- and a ribbon mask would silently drop the thalamus
    that every neuromodulatory and afferent process in the trace writes to.
    """
    root = Path(subject_dir)
    surf = root / "surf"
    out: dict[str, Geometry] = {}

    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    n = 0
    for hemi in ("lh", "rh"):
        v, f = read_freesurfer_surface(surf / f"{hemi}.{surface}")
        verts.append(v)
        faces.append(f + n)
        n += len(v)
    out["cortical_surface"] = SurfaceGeometry(
        "cortical_surface", frame, np.concatenate(verts), np.concatenate(faces),
        source=f"{root.name}: lh.{surface} + rh.{surface} ({n:,} vertices), surface RAS")

    thickness: list[np.ndarray] = []
    for hemi in ("lh", "rh"):
        p = surf / f"{hemi}.thickness"
        if p.is_file():
            thickness.append(read_freesurfer_curv(p))
    if thickness:
        th = np.concatenate(thickness)
        th = th[np.isfinite(th) & (th > 0)]
        note = (f"mean cortical thickness {th.mean():.2f} mm (5th-95th pct "
                f"{np.percentile(th, 5):.2f}-{np.percentile(th, 95):.2f} mm)")
    else:
        note = "no ?h.thickness on disk; the proportion cannot be converted to mm"
    out["cortical_depth"] = _depth_geometry(depth_levels, frame, f"{root.name}: {note}")

    aseg = read_mgh(root / "mri" / "aseg.mgz")
    occ = _label_occupancy(aseg.data, ASEG_PARENCHYMA)
    out["tissue"] = VolumeGeometry(
        "tissue", frame, _bounds(occ, aseg.vox2ras_tkr), occ, aseg.vox2ras_tkr,
        source=(f"{root.name}: aseg.mgz, {len(ASEG_PARENCHYMA)} parenchyma labels, "
                f"{int((occ > 0).sum()):,} voxels at "
                f"{aseg.zooms[0]:g} mm; a hard label map, so occupancy is binary and a leaf "
                f"straddling the pial surface is rounded rather than fractional"))

    # the same `aseg` delimits two more declared supports, and reporting them as
    # absent while the file sits open was putting a real limitation and an
    # unopened label in the same column.  `ASEG_CSF` has been beside
    # `ASEG_PARENCHYMA` since this module was written and nothing read it.
    csf = _label_occupancy(aseg.data, ASEG_CSF)
    out["csf_space"] = VolumeGeometry(
        "csf_space", frame, _bounds(csf, aseg.vox2ras_tkr), csf, aseg.vox2ras_tkr,
        source=(f"{root.name}: aseg.mgz, {len(ASEG_CSF)} csf labels, "
                f"{int((csf > 0).sum()):,} voxels.  this is the VENTRICULAR system and what "
                f"FreeSurfer labels of the cisterns; the subarachnoid space over the convexity "
                f"is not labelled by recon-all and is therefore not in here, which matters for "
                f"a glymphatic model and not for a ventricular one"))
    # the interstitial space shares the parenchyma's extent and is not a copy of
    # it: the extracellular space is interdigitated with the cells throughout, so
    # at any resolution a materialization can afford the two occupy the same
    # millimetres.  what distinguishes the support is its volume fraction and
    # tortuosity, which are field state rather than extent.
    out["interstitial"] = VolumeGeometry(
        "interstitial", frame, _bounds(occ, aseg.vox2ras_tkr), occ, aseg.vox2ras_tkr,
        source=(f"{root.name}: aseg.mgz parenchyma -- the interstitial space is interdigitated "
                f"with the cells and has no boundary of its own; its 20% volume fraction and "
                f"1.6 tortuosity are field properties, not extent"))
    return out


def read_talairach_xfm(path: str | Path) -> np.ndarray:
    """the linear scanner-RAS -> MNI305 transform `recon-all` writes.

    read rather than recomputed, and read from the subject's own directory,
    because it is the only transform into a template frame that this subject
    actually has.  a materialization that needs a population atlas -- an arterial
    territory, a group connectome's parcel centroids -- needs exactly this and
    nothing else stands in for it.

    it is an AFFINE, twelve parameters, and that is the honest limit of what it
    can do: it matches head size and gross orientation and it does not match a
    sulcus.  `ibm.frames` carries the residual, and every position warped through
    it inherits a systematic, spatially correlated displacement rather than a
    random one.
    """
    p = Path(path)
    if not p.is_file():
        raise MissingData("read_talairach_xfm", str(p),
                          "the linear talairach transform recon-all writes",
                          "subjects/<id>/mri/transforms/talairach.xfm")
    text = p.read_text()
    i = text.find("Linear_Transform")
    if i < 0:
        raise ValueError(f"{p}: no Linear_Transform block")
    rows = []
    for line in text[i:].splitlines()[1:]:
        bits = line.replace(";", "").split()
        if len(bits) != 4:
            break
        rows.append([float(x) for x in bits])
        if len(rows) == 3:
            break
    if len(rows) != 3:
        raise ValueError(f"{p}: Linear_Transform has {len(rows)} rows, not 3")
    return np.vstack([np.array(rows, float), [0.0, 0.0, 0.0, 1.0]])


def template_transform(subject_dir: str | Path, *, dst: str = "mni152") -> np.ndarray:
    """surface RAS -> a template frame, as one 4x4, composed rather than assumed.

    three transforms in a row, and each one is a place this has historically gone
    wrong silently:

        tkrRAS -> scanner RAS      inv(Torig) then Norig, from the T1's own header.
                                   they differ by `c_ras`, a few centimetres, and
                                   using the wrong one shifts a source grid off the
                                   cortex while every number stays plausible.
        scanner RAS -> MNI305      `talairach.xfm`, this subject's own affine.
        MNI305 -> MNI152           a published constant.  FreeSurfer targets MNI305
                                   and every volumetric atlas in this corpus is
                                   distributed in MNI152; the two differ by about a
                                   millimetre, and NOT applying it costs the same
                                   millimetre silently.
    """
    from ibm.materialize.substrate import MNI305_TO_MNI152

    root = Path(subject_dir)
    t1 = read_mgh(root / "mri" / "T1.mgz")
    xfm = read_talairach_xfm(root / "mri" / "transforms" / "talairach.xfm")
    m = xfm @ np.asarray(t1.vox2ras, float) @ np.linalg.inv(np.asarray(t1.vox2ras_tkr, float))
    if dst == "mni305":
        return m
    if dst != "mni152":
        raise ValueError(f"template_transform knows mni305 and mni152, not {dst!r}")
    return MNI305_TO_MNI152 @ m


def _depth_geometry(levels: int, frame: str, note: str) -> DiscreteGeometry:
    """cortical depth as a small set of normalized levels.

    the honest caveat first: the three columns of `xyz` here are *not* positions.
    `cortical_depth` is declared as normalized depth through the ribbon, pial to
    white, and §1 is explicit that it is a proportion rather than a distance
    because cortical thickness varies two-fold across the sheet.  there is no
    sampler in `sites.py` for a one-dimensional manifold, and forcing one would
    mean inventing a metric; carrying the levels as a discrete set is the
    truthful encoding, since what a laminar process actually needs is an ordered
    set of proportions and the per-vertex thickness that converts them.  the
    first column is the proportion, the other two are zero, and any process that
    reads them as millimetres in the subject's head is wrong.
    """
    d = (np.arange(levels) + 0.5) / levels
    xyz = np.stack([d, np.zeros(levels), np.zeros(levels)], axis=1)
    ids = tuple(f"depth_{x:.3f}" for x in d)
    return DiscreteGeometry("cortical_depth", frame, xyz, ids,
                            source=f"{levels} normalized pial-to-white levels; {note}.  the "
                                   "columns are (proportion, 0, 0) and are not millimetres")


def _label_occupancy(labels: np.ndarray, keep: Iterable[int]) -> np.ndarray:
    want = np.zeros(int(np.asarray(labels).max()) + 1, np.float32)
    for k in keep:
        if k < len(want):
            want[k] = 1.0
    return want[np.asarray(labels, np.int64)]


def from_bem(bem_dir: str | Path, *, voxel_mm: float = 1.0,
             frame: str = ANATOMICAL_FRAME) -> dict[str, Geometry]:
    """head_volume and scalp from a three-layer BEM.

    the outer skin boundary is the conductor's outer surface, so it is the
    honest extent of `head_volume` -- "brain, csf, skull, scalp, air cavities" is
    exactly what a filled `outer_skin` contains, and nothing else on disk
    delimits it.  it doubles as the `scalp` support, which is a surface and stays
    one: the electrodes sit *on* it, and flattening it into the volume would lose
    the normal that the electrode-skin coupling is defined against.

    the inner skull and outer skull boundaries are read and reported but are not
    supports of their own.  `ibm.fields.supports` declares no skull support, and
    inventing one here to hold the two surfaces would be adding to the ontology
    from a loader, which is precisely the thing the registry exists to refuse.
    they belong to the *material* field as conductivity over `head_volume`, and
    the layer conductivities are carried in the geometry's source string so that
    provenance can say which head model the lead field was solved on.
    """
    layers = read_bem_layers(bem_dir)
    skin = layers.get("outer_skin")
    if skin is None:
        raise MissingData("from_bem", "the outer skin boundary",
                          "the outermost surface of the volume conductor, which is what bounds "
                          "the head_volume support",
                          f"{bem_dir}: a *-bem.fif written by mne.make_bem_model, or "
                          "outer_skin.surf from mne watershed_bem")
    v, f, sigma = skin
    occ, affine = occupancy_from_closed_surface(v, f, voxel_mm=voxel_mm)
    desc = ", ".join(f"{k} sigma={s:g} S/m" for k, (_, _, s) in sorted(layers.items())
                     if s is not None)
    src = (f"{Path(bem_dir).name}: outer_skin filled at {voxel_mm:g} mm "
           f"({int((occ > 0).sum()):,} voxels)" + (f"; layers {desc}" if desc else ""))
    return {
        "head_volume": VolumeGeometry("head_volume", frame, _bounds(occ, affine), occ, affine,
                                      source=src),
        "scalp": SurfaceGeometry("scalp", frame, v, f,
                                 source=f"{Path(bem_dir).name}: outer_skin ({len(v):,} vertices)"),
    }


def read_bem_layers(bem_dir: str | Path
                    ) -> dict[str, tuple[np.ndarray, np.ndarray, float | None]]:
    """the conductor boundaries, preferring the FIF that carries conductivities.

    a `*-bem.fif` holds the same three surfaces as the `.surf` files *plus* the
    conductivity of each layer, and the conductivity is the half of a head model
    that actually sets the lead field: the skull-to-brain ratio is the parameter
    an eeg forward solution is most sensitive to and least able to constrain from
    its own data.  so the FIF is read when it exists and the bare surfaces are
    the fallback, with the difference recorded rather than smoothed over.
    """
    root = Path(bem_dir)
    fifs = sorted(root.glob("*-bem.fif"))
    names = {4: "outer_skin", 3: "outer_skull", 1: "inner_skull"}
    out: dict[str, tuple[np.ndarray, np.ndarray, float | None]] = {}
    if fifs:
        import mne

        # the richest model wins: a 3-layer file beats a 1-layer one, and both
        # exist side by side in a typical subjects/<id>/bem.
        best: list[Any] = []
        for p in fifs:
            try:
                surfs = mne.read_bem_surfaces(str(p), verbose="ERROR")
            except Exception:                                      # pragma: no cover
                continue
            if len(surfs) > len(best):
                best = list(surfs)
        for s in best:
            key = names.get(int(s["id"]), f"bem_{int(s['id'])}")
            out[key] = (np.asarray(s["rr"], float) * 1000.0,       # fif stores metres
                        np.asarray(s["tris"], np.int64),
                        None if s.get("sigma") is None else float(s["sigma"]))
    if out:
        return out
    for key in names.values():
        p = root / f"{key}.surf"
        if p.is_file():
            v, f = read_freesurfer_surface(p)
            out[key] = (v, f, None)
    if not out:
        raise MissingData("read_bem_layers", str(root),
                          "a boundary-element head model: either a *-bem.fif or the "
                          "inner_skull/outer_skull/outer_skin .surf triplet",
                          "mne watershed_bem followed by mne.make_bem_model, or the flash "
                          "multi-echo route where a flash5 is available")
    return out


# ---------------------------------------------------------------------------
# the montage, and the one transform that relates it to the head
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coregistration:
    """the digitised-to-anatomical transform, with both of its residuals.

    two numbers, and they answer different questions.  `declared_mm` is what
    `ibm.frames` says a warp of this kind costs -- 5 mm for `eeg_cap ->
    subject_t1`, which is the honest figure for a montage that was never
    digitised -- and it is the number a materialization must carry when it has no
    better one.  `measured_mm` is what *this* coregistration achieves, computed
    against geometry that was not used to fit it: the distance from every
    transformed electrode to the scalp surface it is supposed to be sitting on.
    reporting only the first understates a good coregistration; reporting only
    the second hides that the transform itself could be wrong in a way its own
    residual cannot see.
    """

    src_frame: str
    dst_frame: str
    matrix: np.ndarray                  # (4, 4), millimetres
    source: str = ""
    declared_mm: float | None = None
    measured_mm: float | None = None
    measured_note: str = ""

    def apply(self, xyz: Any) -> np.ndarray:
        p = np.asarray(xyz, float).reshape(-1, 3)
        h = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ np.asarray(self.matrix, float).T
        return h[:, :3]

    def inverse(self) -> "Coregistration":
        return Coregistration(self.dst_frame, self.src_frame, np.linalg.inv(self.matrix),
                              self.source + " (inverted)", self.declared_mm, self.measured_mm,
                              self.measured_note)

    def warp(self, xyz: Any, src: str, dst: str) -> np.ndarray:
        """the callable `build(warp=...)` and `RegionResolver` expect.

        it refuses any pair it was not built for instead of returning the
        identity, because an unrecognised frame silently passed through is the
        misregistration `ibm.frames` exists to prevent -- and it looks exactly
        like success.
        """
        p = np.asarray(xyz, float).reshape(-1, 3)
        if src == dst:
            return p
        if (src, dst) == (self.src_frame, self.dst_frame):
            return self.apply(p)
        if (src, dst) == (self.dst_frame, self.src_frame):
            return self.inverse().apply(p)
        raise MissingData(
            "coregistration", f"warp({src!r} -> {dst!r})",
            f"this coregistration relates {self.src_frame!r} and {self.dst_frame!r} and nothing "
            "else; another pair needs its own measured transform",
            "the registration output for that pair -- a -trans.fif, a surface registration, or "
            "digitised fiducials")

    def describe(self) -> str:
        t = np.asarray(self.matrix, float)
        shift = float(np.linalg.norm(t[:3, 3]))
        ang = float(np.degrees(np.arccos(np.clip((np.trace(t[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
        lines = [f"coregistration {self.src_frame} -> {self.dst_frame}: "
                 f"{ang:.1f} deg rotation, {shift:.1f} mm translation",
                 f"  source            {self.source}"]
        if self.declared_mm is not None:
            lines.append(f"  declared residual {self.declared_mm:.1f} mm "
                         "(ibm.frames, for a warp of this kind)")
        if self.measured_mm is not None:
            lines.append(f"  measured residual {self.measured_mm:.2f} mm  {self.measured_note}")
        return "\n".join(lines)


def from_montage(raw_or_fif: Any, *, support: str = "sensor_array",
                 frame: str = MONTAGE_FRAME, kind: str = "eeg"
                 ) -> tuple[DiscreteGeometry, tuple[str, ...]]:
    """the real electrode positions, left in the frame they were digitised in.

    `discrete_sites` says an electrode is where it is and r(q) has no opinion
    about it; this is the other half of that statement -- the positions come off
    the recording, in metres in the instrument's head frame, and are converted to
    millimetres and nothing else.  they are *not* carried into the anatomical
    frame here, because that step needs a coregistration that is a separate
    measurement with its own residual, and doing it silently inside a loader is
    how a montage ends up 8 mm above the scalp with no record of why.
    """
    import mne

    info = _info_of(raw_or_fif)
    picks = mne.pick_types(info, meg=(kind == "meg"), eeg=(kind == "eeg"), exclude=())
    if not len(picks):
        raise MissingData("from_montage", f"{kind} channels",
                          f"channel positions for a {kind} montage; this recording has none",
                          "a raw FIF with digitised electrode positions, or a montage applied "
                          "with raw.set_montage()")
    xyz = np.array([info["chs"][int(i)]["loc"][:3] for i in picks], float) * 1000.0
    ids = tuple(str(info["ch_names"][int(i)]) for i in picks)
    bad = ~np.isfinite(xyz).all(axis=1) | (np.linalg.norm(xyz, axis=1) < 1e-6)
    if bad.any():
        raise MissingData("from_montage", "electrode positions",
                          f"{int(bad.sum())} of {len(ids)} channels carry a zero or non-finite "
                          "position, which means the montage was never applied",
                          "raw.set_montage(...) with a digitisation or a standard cap")
    n_dig = len(info["dig"] or ())
    return DiscreteGeometry(
        support, frame, xyz, ids,
        source=(f"{len(ids)} {kind} contacts from {getattr(raw_or_fif, 'filenames', [''])[0] or raw_or_fif}"
                if not isinstance(raw_or_fif, (str, Path)) else
                f"{len(ids)} {kind} contacts from {Path(str(raw_or_fif)).name}")
        + f"; {n_dig} digitised points; positions in the {frame} frame, in mm"), ids


def _info_of(raw_or_fif: Any) -> Any:
    import mne

    if isinstance(raw_or_fif, (str, Path)):
        return mne.io.read_raw_fif(str(raw_or_fif), preload=False, verbose="ERROR").info
    return getattr(raw_or_fif, "info", raw_or_fif)


def coregistration_from_trans(trans_fif: str | Path, *,
                              src_frame: str = MONTAGE_FRAME,
                              dst_frame: str = ANATOMICAL_FRAME,
                              scalp: SurfaceGeometry | None = None,
                              electrodes: np.ndarray | None = None) -> Coregistration:
    """the measured head -> MRI transform, checked against the scalp it implies.

    the declared residual comes from `ibm.frames`, which is the right default and
    a poor description of a coregistration that was actually run.  so where a
    scalp surface and the electrodes are both available this also measures one:
    the distance from each transformed contact to the nearest point of the scalp.
    an electrode is a disc sitting on skin, so that distance should be a
    millimetre or two of gel and hair; a systematically larger one is the
    signature of a transform that is subtly wrong, and it is visible here without
    any ground truth beyond the anatomy itself.
    """
    import mne

    from ibm import frames as _frames

    t = mne.read_trans(str(trans_fif), verbose="ERROR")
    m = np.asarray(t["trans"], float).copy()
    m[:3, 3] *= 1000.0                                   # fif stores the shift in metres
    declared = _frames.residual_mm(src_frame, dst_frame)

    measured: float | None = None
    note = ""
    if scalp is not None and electrodes is not None and len(electrodes):
        p = np.concatenate([np.asarray(electrodes, float).reshape(-1, 3),
                            np.ones((len(electrodes), 1))], axis=1) @ m.T
        v = np.asarray(scalp.vertices, float)
        d = np.empty(len(p))
        for i in range(0, len(p), 64):
            blk = p[i:i + 64, :3]
            d[i:i + 64] = np.sqrt(((blk[:, None, :] - v[None, :, :]) ** 2).sum(-1)).min(axis=1)
        measured = float(np.median(d))
        note = (f"median contact-to-scalp distance over {len(p)} electrodes "
                f"(min {d.min():.2f}, max {d.max():.2f} mm); an electrode sits on skin, so this "
                "is gel and hair plus whatever the transform got wrong")
    return Coregistration(src_frame, dst_frame, m,
                          source=f"{Path(trans_fif).name}: {t['from']} -> {t['to']}, "
                                 "a measured fiducial/head-shape coregistration",
                          declared_mm=declared, measured_mm=measured, measured_note=note)


# ---------------------------------------------------------------------------
# the lead field, which is the one thing positions cannot give
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadField:
    """what each sensor sees per unit source current, for one set of positions.

    `ibm.topologies.em` is blunt about why this cannot be a distance function:
    the skull is two orders of magnitude less conductive than brain, the csf
    beneath it shunts current tangentially, and a source's orientation decides
    whether it is visible at all -- so a 1/r^2 kernel gets the *ordering* of
    sensor sensitivities wrong and not merely its scale.  the only honest way to
    obtain the topology is to solve the forward problem on this subject's head
    model, which is what this holds.

    `solved` is the part that must not be lost.  a boundary-element solver
    discards source positions outside its innermost boundary, and an adaptive
    octree over a parenchyma mask puts a fair number of cell centres exactly
    there -- a 10 mm cell can be mostly grey matter with its centre in the
    subarachnoid space.  those rows come back as zero, which reads as "this
    sensor cannot see this source" when what it means is "the head model does not
    cover this source", and the two are very different claims.
    """

    data: np.ndarray                    # (n_sensor, n_source, 3), V per A.m
    sensor_ids: tuple[str, ...]
    solved: np.ndarray                  # (n_source,) bool
    source: str = ""
    note: str = ""

    @property
    def n_sensor(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_source(self) -> int:
        return int(self.data.shape[1])

    def describe(self) -> str:
        mag = np.linalg.norm(self.data, axis=2)
        live = mag[:, self.solved]
        n_out = int((~self.solved).sum())
        lines = [f"lead field {self.n_sensor} sensors x {self.n_source} sources x 3 orientations",
                 f"  source            {self.source}",
                 f"  magnitude         median {np.median(live):.3g}, "
                 f"peak {live.max():.3g} (V per A.m)",
                 f"  dynamic range     {live.max() / max(np.median(live), 1e-30):.0f}x between "
                 "the peak and the median entry, which is the skull doing the smoothing",
                 f"  solved            {int(self.solved.sum()):,} of {self.n_source:,} sources; "
                 f"{n_out:,} fall outside the head model's innermost boundary and carry a zero "
                 "row"]
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)


def bem_lead_field(source_xyz_mm: Any, *, raw_fif: str | Path, trans_fif: str | Path,
                   bem_sol: str | Path, sensor_ids: Sequence[str] | None = None,
                   kind: str = "eeg", frame: str = ANATOMICAL_FRAME) -> LeadField:
    """solve the forward problem over *these* positions and this subject's BEM.

    deliberately not the precomputed `*-fwd.fif` sitting beside the recording.
    that solution is for an oct-6 cortical source space, and a lead field computed
    for a different source space cannot be reindexed onto an octree -- the
    builder in `ibm.topologies.em` refuses exactly that, and it is right to: the
    columns are positions, and interpolating between them across a sulcus mixes
    two sources whose orientations are opposite.  so the head model is reused and
    the solution is not, which is the part that is actually subject-specific.

    the sources are given in the anatomical frame in millimetres, because that is
    where the site table is; the coregistration to the sensors' frame is the
    `-trans.fif` and is applied by the solver rather than here, so it is the same
    transform the site table records rather than a second copy of it.
    """
    import mne

    xyz = np.asarray(source_xyz_mm, float).reshape(-1, 3)
    if frame != ANATOMICAL_FRAME:
        raise ValueError(f"bem_lead_field wants sources in {ANATOMICAL_FRAME!r} (surface RAS); "
                         f"it was handed {frame!r}, and moving them is the caller's decision")
    src = mne.setup_volume_source_space(
        pos=dict(rr=xyz / 1000.0, nn=np.tile([0.0, 0.0, 1.0], (len(xyz), 1))), verbose="ERROR")
    info = mne.io.read_raw_fif(str(raw_fif), preload=False, verbose="ERROR").info
    picks = mne.pick_types(info, meg=(kind == "meg"), eeg=(kind == "eeg"), exclude=())
    info = mne.pick_info(info, picks)
    fwd = mne.make_forward_solution(info, trans=str(trans_fif), src=src, bem=str(bem_sol),
                                    meg=(kind == "meg"), eeg=(kind == "eeg"), verbose="ERROR")
    rows = list(fwd["sol"]["row_names"])
    sol = np.asarray(fwd["sol"]["data"], float).reshape(len(rows), -1, 3)
    solved = np.asarray(fwd["src"][0]["inuse"], bool)

    want = list(sensor_ids) if sensor_ids is not None else rows
    missing = [c for c in want if c not in rows]
    if missing:
        raise MissingData("bem_lead_field", ", ".join(missing[:6]),
                          "a forward solution row for every materialized sensor; the montage "
                          "and the recording disagree about which channels exist",
                          "the same raw file the montage positions came from")
    order = np.array([rows.index(c) for c in want], np.int64)

    data = np.zeros((len(want), len(xyz), 3))
    data[:, solved, :] = sol[order]
    return LeadField(
        data, tuple(want), solved,
        source=(f"mne BEM forward over {Path(bem_sol).name} with {Path(trans_fif).name}, "
                f"free orientation, {len(want)} {kind} channels"),
        note=(f"{int((~solved).sum()):,} of {len(xyz):,} source positions "
              f"({1.0 - float(solved.mean()):.1%}) lie outside the inner skull.  a large "
              "fraction means the sources are octree cell centres: a conservative octree keeps "
              "any cell that touches parenchyma, and at 10 mm many such centres sit in the "
              "subarachnoid space.  a fraction near zero means they are surface positions, "
              "which lie on the white boundary by construction and can only fall out where the "
              "bem's inner-skull surface and the freesurfer reconstruction disagree"))


# ---------------------------------------------------------------------------
# the mne sample subject, resolved from its own card
# ---------------------------------------------------------------------------

#: the card whose `.location.yaml` says where these bytes are.  named as a
#: constant so the error message and the loader cannot drift apart.
SAMPLE_CARD = "mne-sample"


@dataclass(frozen=True)
class SamplePaths:
    """the files `sample_subject` reads, resolved and checked before anything opens."""

    root: Path
    subject_dir: Path
    bem_dir: Path
    raw_fif: Path
    trans_fif: Path
    forward_fif: Path

    def describe(self) -> str:
        return "\n".join(f"  {k:12s} {v}" for k, v in
                         (("dataset", self.root), ("subject", self.subject_dir),
                          ("bem", self.bem_dir), ("raw", self.raw_fif),
                          ("trans", self.trans_fif), ("forward", self.forward_fif)))


def sample_paths(subject: str = "sample") -> SamplePaths:
    """where the mne sample subject's bytes are, per `data/sources/mne-sample`.

    the path is *never* written down here.  `.location.yaml` is committed and the
    bytes are not, so the card is the single place that knows, and a loader that
    hardcoded the answer would work on the machine it was written on and read
    nothing on any other -- while reporting a result computed over an empty file
    list, which is worse than failing.
    """
    from ibm.forge.spectra import local_root

    root = Path(local_root(SAMPLE_CARD))
    hits = sorted(root.glob("*/MNE-sample-data")) or sorted(root.glob("MNE-sample-data"))
    if not hits:
        hits = sorted(p.parent for p in root.glob("*/*/MNE-sample-data/subjects"))
    if not hits:
        raise MissingData(
            "sample_paths", f"MNE-sample-data under {root}",
            "the unpacked MNE sample dataset, containing subjects/<id> and MEG/<id>",
            f"data/sources/{SAMPLE_CARD}/raw/.location.yaml points at {root}, which does not "
            "contain an MNE-sample-data directory; re-acquire the card's bytes")
    base = hits[0]
    p = SamplePaths(
        root=base,
        subject_dir=base / "subjects" / subject,
        bem_dir=base / "subjects" / subject / "bem",
        raw_fif=base / "MEG" / subject / f"{subject}_audvis_raw.fif",
        trans_fif=base / "MEG" / subject / f"{subject}_audvis_raw-trans.fif",
        forward_fif=base / "MEG" / subject / f"{subject}_audvis-meg-eeg-oct-6-fwd.fif")
    missing = [str(x) for x in (p.subject_dir, p.bem_dir, p.raw_fif, p.trans_fif) if not x.exists()]
    if missing:
        raise MissingData(
            "sample_paths", ", ".join(missing),
            "a complete FreeSurfer reconstruction with a BEM plus the raw recording and its "
            "coregistration",
            f"the {SAMPLE_CARD} card; its local_root is {root} and these paths are missing "
            "under it")
    return p


def sample_subject(*, subject: str = "sample", surface: str = "white",
                   voxel_mm: float = 1.0, paths: SamplePaths | None = None) -> GeometrySet:
    """a ready `GeometrySet` for the mne sample subject.

    what it contains is what this dataset actually has: a cortical sheet, a
    parenchyma mask, a filled three-layer conductor, its scalp, a set of
    normalized cortical depths, and a 59-electrode montage in its own frame.
    what it does *not* contain is everything else `ibm.fields.supports` declares
    -- there is no vascular segmentation, no retina, no cochlea, no body and no
    implanted array in an mne sample subject, and this returns a set that is
    silent about them rather than a set of plausible stand-ins.  a build that
    genuinely needs one of those will say which support and which file, which is
    the whole design of `GeometrySet.require`.
    """
    p = paths or sample_paths(subject)
    g: dict[str, Geometry] = {}
    g.update(from_freesurfer(p.subject_dir, surface=surface, voxel_mm=voxel_mm))
    g.update(from_bem(p.bem_dir, voxel_mm=voxel_mm))
    montage, _ids = from_montage(p.raw_fif)
    g[montage.support] = montage
    return GeometrySet(g, subject=subject)


def sample_lead_field(source_xyz_mm: Any, *, subject: str = "sample",
                      sensor_ids: Sequence[str] | None = None,
                      paths: SamplePaths | None = None) -> LeadField:
    """the sample subject's own three-layer BEM solution, for these source positions.

    the solution file is the 5120-per-layer one that ships with the dataset, so
    the skull and scalp conductivities are the dataset's own and not a guess; the
    only thing computed here is the solve over the octree's positions.
    """
    p = paths or sample_paths(subject)
    sols = sorted(p.bem_dir.glob("*-bem-sol.fif"))
    if not sols:
        raise MissingData("sample_lead_field", f"{p.bem_dir}/*-bem-sol.fif",
                          "a precomputed boundary-element solution for this head model",
                          "mne.make_bem_solution over the three-layer model in this bem/")
    # the richest model wins, exactly as in read_bem_layers: a 5120-5120-5120
    # solution carries skull and scalp, a 5120 one is the brain alone.
    best = max(sols, key=lambda q: q.name.count("5120"))
    return bem_lead_field(source_xyz_mm, raw_fif=p.raw_fif, trans_fif=p.trans_fif,
                          bem_sol=best, sensor_ids=sensor_ids)


def sample_coregistration(*, subject: str = "sample", geometry: GeometrySet | None = None,
                          paths: SamplePaths | None = None) -> Coregistration:
    """the sample subject's real head -> MRI transform, with its measured residual."""
    p = paths or sample_paths(subject)
    g = geometry or sample_subject(subject=subject, paths=p)
    scalp = g.get("scalp")
    montage = g.get("sensor_array")
    return coregistration_from_trans(
        p.trans_fif,
        scalp=scalp if isinstance(scalp, SurfaceGeometry) else None,
        electrodes=None if montage is None else np.asarray(montage.xyz, float))


__all__ = [
    "ANATOMICAL_FRAME", "MONTAGE_FRAME", "ASEG_PARENCHYMA", "ASEG_CSF", "SAMPLE_CARD",
    "MGHVolume", "Coregistration", "SamplePaths",
    "read_freesurfer_surface", "read_freesurfer_curv", "read_mgh",
    "occupancy_from_closed_surface", "from_freesurfer", "from_bem", "read_bem_layers",
    "from_montage", "coregistration_from_trans", "LeadField", "bem_lead_field",
    "sample_lead_field",
    "sample_paths", "sample_subject", "sample_coregistration",
]
