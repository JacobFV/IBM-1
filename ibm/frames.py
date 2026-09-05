"""coordinate frames.

every position, atlas, tractogram, head model and dataset declares the frame it
speaks; the materializer inserts warps between them and nothing is ever compared
across frames implicitly.  a support (§1) carries whatever frame its positions
live in.

this is cheap now and brutal to retrofit.  silent misregistration is the dominant
failure mode of multimodal neuroimaging, and it does not announce itself -- the
numbers stay plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from ibm.registry import REGISTRY


@dataclass(frozen=True)
class Frame:
    name: str
    doc: str
    kind: str            # "volume" | "surface" | "device" | "sensor" | "screen" | "body"
    units: str = "mm"
    handedness: str = "RAS"


@dataclass(frozen=True)
class Warp:
    """a declared map between two frames.

    `invertible` and `subject_specific` are not decoration.  a template warp
    applied to a subject displaces every position in a head-size-dependent,
    spatially correlated way, so the resulting error is systematic rather than
    random and must be carried as uncertainty in everything derived from it.
    """
    src: str
    dst: str
    method: str          # "affine" | "nonlinear" | "rigid" | "digitised" | "fitted" | "identity"
    subject_specific: bool
    invertible: bool = True
    residual_mm: float | None = None
    note: str = ""


FRAMES: dict[str, Frame] = {}
WARPS: dict[tuple[str, str], Warp] = {}


def frame(f: Frame) -> Frame:
    if f.name in FRAMES:
        raise ValueError(f"frame {f.name!r} already registered")
    FRAMES[f.name] = f
    return f


def warp(w: Warp) -> Warp:
    WARPS[(w.src, w.dst)] = w
    return w


# -- template and subject volumes ------------------------------------------
frame(Frame("mni152", "MNI152 nonlinear template volume", "volume"))
frame(Frame("mni305", "MNI305 template volume", "volume"))
frame(Frame("subject_t1", "native anatomical volume of one participant", "volume"))
frame(Frame("subject_dwi", "native diffusion volume; distorted relative to T1", "volume"))
frame(Frame("subject_bold", "native functional volume; distorted relative to T1", "volume"))

# -- surfaces ---------------------------------------------------------------
frame(Frame("fsaverage", "FreeSurfer average surface", "surface"))
frame(Frame("fslr32k", "HCP fs_LR 32k surface", "surface"))
frame(Frame("subject_surf", "native reconstructed cortical surface", "surface"))

# -- sensors and devices ----------------------------------------------------
frame(Frame("eeg_cap", "electrode positions on the scalp", "sensor"))
frame(Frame("meg_dewar", "gradiometer and magnetometer positions in the dewar", "sensor"))
frame(Frame("meg_head", "MEG sensors after head-position transformation", "sensor"))
frame(Frame("nirs_cap", "optode positions on the scalp", "sensor"))
frame(Frame("electrode_grid", "implanted grid, strip or depth contacts", "device"))
frame(Frame("array", "intracortical microelectrode array", "device"))
frame(Frame("coil", "TMS coil position and orientation", "device"))
frame(Frame("transducer", "focused-ultrasound transducer geometry", "device"))
frame(Frame("scanner", "MR scanner bore and gradient frame", "device"))

# -- body, world and stimulus ----------------------------------------------
frame(Frame("body", "participant body segments and joints", "body"))
frame(Frame("eye", "eye-centred, the frame retinal input is actually defined in", "body"))
frame(Frame("display", "stimulus screen in pixels, with a viewing distance", "screen", units="px"))
frame(Frame("world", "room or scene coordinates", "volume"))
frame(Frame("audio", "free-field acoustic coordinates relative to the head", "world"))

# -- warps ------------------------------------------------------------------
warp(Warp("subject_t1", "mni152", "nonlinear", True, residual_mm=2.0,
          note="registration residual is regionally structured; largest at the cortical rim"))
warp(Warp("subject_surf", "fsaverage", "nonlinear", True, residual_mm=3.0,
          note="surface-based registration; folding-driven, so the residual concentrates in tertiary sulci"))
warp(Warp("subject_dwi", "subject_t1", "nonlinear", True, residual_mm=1.5,
          note="susceptibility distortion; without a fieldmap this is the dominant error in tractography endpoints"))
warp(Warp("subject_bold", "subject_t1", "nonlinear", True, residual_mm=1.5))
warp(Warp("meg_dewar", "meg_head", "rigid", True, note="from continuous head-position indicators"))
warp(Warp("meg_head", "subject_t1", "digitised", True, residual_mm=3.0,
          note="from fiducial and head-shape digitisation; the single largest error in MEG source estimates"))
warp(Warp("eeg_cap", "subject_t1", "digitised", True, residual_mm=5.0,
          note="digitised positions if measured; a template montage otherwise, which is systematic error, not noise"))
warp(Warp("electrode_grid", "subject_t1", "fitted", True, residual_mm=2.0,
          note="post-implant CT to pre-implant T1, plus brain shift, which a rigid fit does not model"))
warp(Warp("display", "eye", "fitted", True,
          note="requires viewing distance and gaze; without it a screen coordinate is not retinal input"))


def path(src: str, dst: str) -> list[Warp] | None:
    """shortest declared warp chain, or None if the frames are not connected.

    a missing chain is a hard error at materialization, not something to
    approximate around.
    """
    if src == dst:
        return []
    seen = {src}
    queue: list[tuple[str, list[Warp]]] = [(src, [])]
    while queue:
        node, acc = queue.pop(0)
        for (a, b), w in WARPS.items():
            for nxt, ww in ((b, w),) if a == node else ((a, w),) if (b == node and w.invertible) else ():
                if nxt in seen:
                    continue
                if nxt == dst:
                    return acc + [ww]
                seen.add(nxt)
                queue.append((nxt, acc + [ww]))
    return None


def residual_mm(src: str, dst: str) -> float | None:
    """accumulated registration residual along a warp chain, in quadrature."""
    p = path(src, dst)
    if p is None:
        return None
    rs = [w.residual_mm for w in p if w.residual_mm is not None]
    return (sum(r * r for r in rs)) ** 0.5 if rs else 0.0
