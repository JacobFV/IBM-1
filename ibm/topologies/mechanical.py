"""mechanical adjacency: what is in contact with what.

the topology that most sharply contradicts the cortical-surface one, over the same
positions.  two points on facing banks of a sulcus are centimetres apart along the
sheet and no axon joins them directly -- and mechanically they are in contact.
they are separated by a film of csf, they shear against each other during head
rotation, and a pressure wave crosses from one to the other in microseconds.
`cortical_surface` is right for lateral neural propagation and would be
catastrophically wrong for a mechanical process; `mechanical` is right for
traction and would be equally wrong for the neural one.  the same is true against
`tractometric`: a fascicle transmits action potentials over 150 mm of path and
transmits essentially no stress along it that the surrounding tissue does not
transmit faster in a straight line.

so this is a radius graph in the head volume, with euclidean distance, and the
euclidean metric is correct for once because stress is transmitted by contact
between adjacent material points regardless of what tissue type they are.  it is
defined over `head_volume` rather than `tissue` for exactly that reason: skull,
csf, scalp and brain are one mechanical body, the skull is the boundary condition
for everything inside it, and a mechanical topology restricted to parenchyma would
have no way to express that the brain is floating in fluid inside a rigid shell,
which is the single most important mechanical fact about it.

`contact_area_mm2` and `orientation` are both needed because stress is a tensor,
not a scalar.  the traction across an interface is the stress tensor contracted
with the interface normal, so a process needs the direction of the edge and the
area it acts over; a distance alone would let a mechanical process behave
isotropically, which is wrong wherever the tissue is not -- and white matter,
under shear, is very much not.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance


@B.builder(
    "mechanical_stencil",
    produces=("distance_mm", "contact_area_mm2", "orientation"),
    supports=("head_volume",),
    directed=False,
    metric="euclidean distance between material points in contact",
    doc="a contact stencil over the head as one mechanical body")
def mechanical_stencil(sites, *, support: str = "head_volume",
                       radius_mm: float | None = None, tissue_column: str = "tissue_class",
                       across_tissue_boundaries: bool = True):
    """contact adjacency in the head volume.

    `across_tissue_boundaries` is on by default and is worth being able to turn
    off.  brain and skull are in contact through csf, and stress crosses that
    interface -- but it is a *sliding* interface, not a bonded one, and a model
    that treats it as bonded makes the brain rigid.  a process that wants to
    handle the interface explicitly asks for a stencil that stops at tissue
    boundaries and supplies its own coupling across them; the default keeps the
    edges and lets the process decide, because deleting them by default would
    silently disconnect the brain from its container.
    """
    np = B._numpy("mechanical_stencil")
    t = sites.require(support, "mechanical_stencil",
                      "positions through the head as a mechanical body -- brain, csf, "
                      "skull and scalp, not parenchyma alone")
    xyz = np.asarray(t.xyz, dtype=float)
    sp = t.spacing(np)
    r = float(radius_mm) if radius_mm is not None else 2.0 * float(np.max(sp))

    i, j, d = B.pairs_within(xyz, r, "mechanical_stencil")
    if len(d) == 0:
        return B.empty("mechanical", sites.n_total,
                       ("distance_mm", "contact_area_mm2", "orientation"))

    note_extra = ""
    if not across_tissue_boundaries:
        cls = t.opt(tissue_column)
        if cls is None:
            raise B.MissingInput(
                "mechanical_stencil", f"{support}.columns[{tissue_column!r}]",
                "an (n,) tissue-class label per site (grey, white, csf, skull, "
                "scalp, air), needed to stop the stencil at a tissue boundary",
                "a segmentation of the T1 -- FreeSurfer aseg, simnibs charm, or any "
                "five-tissue segmentation (data/sources: freesurfer, simnibs, ants)")
        c = np.asarray(cls)
        keep = c[i] == c[j]
        i, j, d = i[keep], j[keep], d[keep]
        note_extra = "; stencil stops at tissue boundaries"
        if len(d) == 0:
            return B.empty("mechanical", sites.n_total,
                           ("distance_mm", "contact_area_mm2", "orientation"))

    area = np.minimum(sp[i], sp[j]) ** 2
    return B.EdgeSet(
        "mechanical", i + t.offset, j + t.offset, sites.n_total,
        {"distance_mm": d, "contact_area_mm2": area,
         "orientation": B.unit(xyz[j] - xyz[i], np)},
        directed=False,
        note=f"contact stencil, radius {r:.3g} mm over {support}{note_extra}")


MECHANICAL = REGISTRY.topology(Topology(
    "mechanical",
    "contact adjacency through the head as one mechanical body.  euclidean distance is the "
    "right metric here and it directly contradicts the cortical-surface one over the same "
    "positions: two points on facing sulcal banks are centimetres apart along the sheet and "
    "in mechanical contact across a film of csf, shearing against each other and passing a "
    "pressure wave in microseconds.  it is defined over the head volume rather than "
    "parenchyma because skull, csf, scalp and brain form one body -- the skull is the "
    "boundary condition for everything inside it, and a brain-only mechanical topology "
    "cannot say that the brain floats in fluid inside a rigid shell.  contact area and edge "
    "orientation are carried because traction is a stress tensor contracted with an "
    "interface normal; a scalar distance would force a mechanical process to be isotropic, "
    "which white matter under shear is not",
    on=("head_volume",),
    edge_features=("distance_mm", "contact_area_mm2", "orientation"),
    directed=False,
    builder="mechanical_stencil",
    provenance=Provenance.PHYSICS))
