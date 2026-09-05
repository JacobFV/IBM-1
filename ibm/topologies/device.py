"""device coupling: where an instrument touches the organism.

the topology that carries the boundary between the model and the experiment.  a
recording contact, a stimulating electrode, a coil, a transducer and a display all
have a physical extent and a physical relation to tissue, and every one of them is
a *different* relation -- which is why this is one topology with a reach and a
contact area rather than a family of per-instrument graphs.  what they share is
the structure: a discrete set of device elements, a medium, and a sensitivity or
delivery volume around each element that is a property of the element rather than
of the tissue.

the metric is euclidean distance in the head, and here that is right for a
specific reason: the coupling is through the immediately surrounding medium.  a
depth electrode contact records the extracellular potential in the tissue around
it, and a millimetre of white matter between it and a gyrus does not care that the
gyrus is a different area.  this is exactly where the cortical-surface metric would
be wrong -- a subdural contact over a sulcus records from both banks, which are
centimetres apart along the sheet -- and it is why `device_coupling` cannot be
folded into `cortical_surface` even for a grid that sits on cortex.

it is not the electromagnetic topology either, and the difference is what the edge
means.  an em edge says a current contributes to a field through a solved forward
model; a device edge says an element is close enough to a piece of tissue to
matter to it at all.  the device topology is the cheap geometric statement that
prunes the expensive one, and it is also the whole story for instruments where no
forward model is involved -- an intracortical contact, a thermistor, a display
whose pixels reach a retinal position.

three features and one honest gap.  `contact_area_mm2` because a contact's
sensitivity volume scales with its surface and a subdural disc and an intracortical
tip differ in it by four orders of magnitude.  `orientation` because directional
elements exist -- a coil's field direction decides which sulcal wall is stimulated,
and a gradiometer is blind along one axis.  `resistance` for contact impedance,
which sets the noise floor and the delivered current density and is the quantity
that drifts over an implant's lifetime.  the gap: impedance is left as nan when the
materialization does not supply it, rather than filled with a plausible kilohm,
because a made-up impedance propagates into a delivered current and looks like a
measurement.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance


@B.builder(
    "device_proximity",
    produces=("distance_mm", "contact_area_mm2", "orientation", "resistance"),
    supports=("implanted_array", "sensor_array", "stimulator", "display", "tissue"),
    directed=True,
    metric="euclidean distance through the immediately surrounding medium",
    doc="element-to-medium edges within each element's reach")
def device_proximity(sites, *, device_support: str = "implanted_array",
                     medium_support: str = "tissue", reach_mm: float | None = None,
                     k: int | None = None, direction: str = "record"):
    """which medium sites each device element couples to.

    `direction` decides the edge orientation and nothing else -- "record" points
    medium -> device, "stimulate" points device -> medium.  it is a parameter
    rather than two builders because the geometry is identical and reciprocity
    makes it genuinely the same relation; only the process differs.

    `reach_mm` defaults from the element's own `reach_mm` column if it has one and
    otherwise to 3 mm, which is roughly the radius within which a clinical
    subdural or depth contact's recorded potential is dominated by local sources.
    it is a property of the element -- an intracortical tip's reach is tens of
    microns and a scalp electrode's is the whole head -- so a materialization that
    knows its hardware should say so rather than accept the default.
    """
    np = B._numpy("device_proximity")
    dev = sites.require(device_support, "device_proximity",
                        "positions of the instrument's elements")
    med = sites.require(medium_support, "device_proximity",
                        "positions of the medium the instrument couples to")
    if direction not in ("record", "stimulate"):
        raise ValueError(f"direction must be 'record' or 'stimulate', not {direction!r}")

    dxyz = np.asarray(dev.xyz, dtype=float)
    mxyz = np.asarray(med.xyz, dtype=float)
    per_elem = dev.opt("reach_mm")
    reach = (np.asarray(per_elem, dtype=float) if per_elem is not None
             else np.full(dev.n, float(reach_mm) if reach_mm is not None else 3.0))
    if reach_mm is not None:
        reach = np.minimum(reach, float(reach_mm))

    kk = int(k) if k is not None else min(med.n, 256)
    idx, dist = B.nearest(dxyz, mxyz, kk, "device_proximity")
    ok = dist <= reach[:, None]
    if not ok.any():
        return B.empty("device_coupling", sites.n_total,
                       ("distance_mm", "contact_area_mm2", "orientation", "resistance"),
                       directed=True,
                       note=(f"no {medium_support} site is within reach of any "
                             f"{device_support} element; check the frame -- device "
                             "positions and medium positions must already be warped "
                             "into a common one"))
    rows = np.repeat(np.arange(dev.n), idx.shape[1])[ok.ravel()]
    cols = idx.ravel()[ok.ravel()]
    d = dist.ravel()[ok.ravel()]

    area = dev.opt("contact_area_mm2")
    area = (np.asarray(area, dtype=float)[rows] if area is not None
            else dev.spacing(np)[rows] ** 2)
    imp = dev.opt("impedance_ohm")
    res = (np.asarray(imp, dtype=float)[rows] if imp is not None
           else np.full(len(rows), np.nan))

    orient = dev.opt("orientation")
    if orient is not None:
        u = B.unit(np.asarray(orient, dtype=float), np)[rows]
    else:
        u = B.unit(mxyz[cols] - dxyz[rows], np)

    if direction == "record":
        src, dst = cols + med.offset, rows + dev.offset
    else:
        src, dst = rows + dev.offset, cols + med.offset

    return B.EdgeSet(
        "device_coupling", src, dst, sites.n_total,
        {"distance_mm": d, "contact_area_mm2": area, "orientation": u, "resistance": res},
        directed=True,
        note=(f"{len(d)} {direction} edges, {device_support} <-> {medium_support}; "
              f"reach from {'the element column' if per_elem is not None else 'the argument'}"
              + ("" if imp is not None else
                 "; impedance is nan -- no impedance_ohm column was supplied, and a "
                 "plausible default would turn into a delivered current density")))


DEVICE_COUPLING = REGISTRY.topology(Topology(
    "device_coupling",
    "the boundary between the model and the experiment: which tissue, scalp or receptor "
    "positions an instrument element is physically close enough to act on or sense.  its "
    "metric is euclidean distance through the immediately surrounding medium, which is the "
    "right one because coupling is local and indifferent to anatomy -- a subdural contact "
    "over a sulcus records from both banks, and those banks are centimetres apart along the "
    "cortical sheet, so the surface metric would be exactly wrong here.  it is distinct from "
    "the electromagnetic topology in what an edge asserts: an em edge is a solved forward "
    "coupling, a device edge is the far cheaper geometric statement that an element is near "
    "enough to matter, which prunes the forward model and is the whole story for instruments "
    "that have none.  reach, contact area and orientation are properties of the element "
    "rather than of the tissue, and contact impedance is left missing rather than "
    "invented when the materialization does not supply it",
    on=("sensor_array", "implanted_array", "stimulator", "scanner_element", "display",
        "scalp", "tissue", "head_volume", "retina"),
    edge_features=("distance_mm", "contact_area_mm2", "orientation", "resistance"),
    directed=True,
    builder="device_proximity",
    provenance=Provenance.PHYSICS))
