"""interstitial adjacency: diffusion through a tortuous, crowded volume.

it looks like `local` -- a radius graph in the parenchyma -- and it is a different
topology because the geometry it describes is a *material property* rather than a
mesh.  the extracellular space is about 20% of tissue volume and is a connected
film 20-60 nm wide winding between cells, so a molecule crossing 100 microns of
tissue travels a path noticeably longer than 100 microns and does so through a
fifth of the cross-section.  the two corrections are both real and they are
different: volume fraction alpha scales how much of a concentration a given amount
of substance produces, and tortuosity lambda scales how fast it spreads, with
D_eff = D_free / lambda^2.  the canonical values -- alpha about 0.2, lambda about
1.6 in healthy cortex -- are among the better-measured numbers in the field, and
they both change under the conditions ibm-1 most wants to model: alpha falls
sharply and lambda rises during ischaemia, spreading depolarization and sleep-wake
transitions.

so the edge feature that matters here is a resistance, not a distance.  a plain
distance would let a process apply a free-solution diffusivity and be wrong by a
factor of two and a half in normal tissue and by much more in swollen tissue.
carrying lambda^2 d / (alpha A) per edge means the geometry and the material state
enter in the place where they belong, and a process that later learns alpha from
a diffusion-weighted measurement changes one column rather than its own equations.

`contact_area_mm2` is the cross-section available for flux between two adjacent
sampling cells -- for a lattice, the face they share.  it is separate from
distance because a coarse materialization has short thick edges and a fine one has
long thin ones relative to their spacing, and only the ratio d/A is the transport
geometry.  the same reasoning is why the metabolic-exchange topology carries an
area too: flux is per unit area everywhere in this part of the ontology.
"""

from __future__ import annotations

from ibm.registry import REGISTRY, Topology
from ibm.topologies import builders as B
from ibm.vocabulary import Provenance


@B.builder(
    "interstitial_radius",
    produces=("distance_mm", "contact_area_mm2", "resistance"),
    supports=("interstitial",),
    directed=False,
    metric="euclidean distance corrected by tortuosity and volume fraction",
    doc="a radius graph over the interstitium with per-edge diffusive resistance")
def interstitial_radius(sites, *, support: str = "interstitial",
                        radius_mm: float | None = None,
                        tortuosity: float = 1.6, volume_fraction: float = 0.2):
    """local diffusive adjacency in the extracellular space.

    tortuosity and volume fraction are taken from the `tortuosity` and
    `volume_fraction` site columns when the materialization carries them -- both
    are declared fields, so a process that changes them changes this edge set's
    inputs rather than a constant -- and from the arguments otherwise.  the
    defaults are the healthy-cortex values from real-time iontophoresis
    measurements and they are not universal: white matter is anisotropic enough
    that a scalar lambda is a summary, and both numbers move by tens of percent
    across the sleep-wake cycle.

    the resistance is lambda^2 d / (alpha A), the geometric factor by which a free
    diffusivity must be divided along this edge.  it is left in that form -- with
    the diffusivity of the actual species factored out -- because potassium,
    glutamate and adenosine share this geometry and differ only in D.
    """
    np = B._numpy("interstitial_radius")
    t = sites.require(support, "interstitial_radius", "interstitial sampling positions")
    xyz = np.asarray(t.xyz, dtype=float)
    sp = t.spacing(np)
    if t.n == 0:
        # R selected nothing on this support.  that is a legitimate
        # materialization -- a request whose region simply does not reach here --
        # and the default radius is two site spacings, which an empty table has
        # none of.  an empty edge set with its columns present is the answer;
        # `np.max` of nothing is a crash three modules from the cause.
        return B.empty("interstitial", sites.n_total, ("distance_mm", "contact_area_mm2", "resistance"),
                       note="no sites on {!r}: R selects none of it".format(support))
    r = float(radius_mm) if radius_mm is not None else 2.0 * float(np.max(sp))

    i, j, d = B.pairs_within(xyz, r, "interstitial_radius")
    if len(d) == 0:
        return B.empty("interstitial", sites.n_total,
                       ("distance_mm", "contact_area_mm2", "resistance"))

    lam = t.opt("tortuosity")
    lam = np.full(t.n, float(tortuosity)) if lam is None else np.asarray(lam, dtype=float)
    alpha = t.opt("volume_fraction")
    alpha = (np.full(t.n, float(volume_fraction)) if alpha is None
             else np.asarray(alpha, dtype=float))

    # the shared cross-section between two sampling cells: the smaller cell's
    # face.  on a uniform lattice this is exactly the voxel face; on a
    # non-uniform one it is the conservative choice, since flux is limited by the
    # narrower of the two.
    area = np.minimum(sp[i], sp[j]) ** 2
    lam_e = 0.5 * (lam[i] + lam[j])
    alpha_e = np.maximum(0.5 * (alpha[i] + alpha[j]), 1e-6)
    resistance = (lam_e ** 2) * d / (alpha_e * np.maximum(area, 1e-12))

    return B.EdgeSet(
        "interstitial", i + t.offset, j + t.offset, sites.n_total,
        {"distance_mm": d, "contact_area_mm2": area, "resistance": resistance},
        directed=False,
        note=(f"radius {r:.3g} mm; lambda from "
              f"{'column' if t.opt('tortuosity') is not None else f'default {tortuosity}'}, "
              f"alpha from "
              f"{'column' if t.opt('volume_fraction') is not None else f'default {volume_fraction}'}; "
              "resistance is the factor dividing a free-solution diffusivity"))


INTERSTITIAL = REGISTRY.topology(Topology(
    "interstitial",
    "diffusive adjacency through the extracellular space.  spatially it looks like the local "
    "topology and physically it is not, because the interstitium is a 20-60 nm film "
    "occupying a fifth of tissue volume and winding between cells: a molecule crossing 100 "
    "microns of tissue travels further than 100 microns through a fifth of the "
    "cross-section.  the two corrections are independent -- volume fraction sets how much "
    "concentration a quantity produces, tortuosity sets how fast it spreads, D_eff = "
    "D_free / lambda^2 -- and both change sharply in ischaemia, spreading depolarization "
    "and sleep.  so the edge carries a resistance, lambda^2 d / (alpha A), with the "
    "species' free diffusivity factored out, rather than a bare distance that would invite a "
    "process to use a free-solution constant and be wrong by a factor of two and a half",
    on=("interstitial",),
    edge_features=("distance_mm", "contact_area_mm2", "resistance"),
    directed=False,
    builder="interstitial_radius",
    provenance=Provenance.PHYSICS))
