"""interaction topologies (ARCHITECTURE.md §3).

    T(i, j) != 0  <=>  i and j may interact through T

importing this package registers every topology and every builder.  a topology
says which materialized state variables may interact through a class of process;
it is a *support* for interaction, not a weighting of it, so the geometry it
carries -- distance, geodesic path, tract length and its delay, contact area,
orientation, resistance -- is descriptive, and how strongly a pair actually
couples belongs to the process's parameters.

there is no universal interaction graph, and the modules here are the argument for
that rather than an illustration of it.  the same two cortical positions on facing
banks of a sulcus are:

    local          one millimetre apart, and potassium crosses between them
    cortical_surface  centimetres apart, and no horizontal axon joins them
    mechanical     in contact, shearing across a film of csf
    tractometric   possibly joined by a fascicle, with a delay neither of the
                   above has
    electromagnetic  contributing to the same sensor with a weight that depends
                   on skull thickness and dipole orientation, instantaneously

none of these five is a reweighting of another; three of them disagree about which
pairs exist at all.  a model with one graph has to pick one of these to be true.

the eleventh line of the argument is `neuromodulatory_projection`, which is dense
where the others are sparse and writes parameters rather than state, and the
twelfth is `afferent_pathway`, whose two ends are in coordinate frames that were
never registered to each other so that no distance between them means anything.

`builders` carries the protocol, the site tables and the shared geometry; the
modules here register declarations and the functions that turn them into edges.
"""

from __future__ import annotations

from ibm.topologies import association  # noqa: F401
from ibm.topologies import builders
from ibm.topologies.builders import (
    BUILDERS, BuilderSpec, EdgeSet, MissingInput, SiteTable, Sites, as_sites,
    build, empty, get,
)

# order is irrelevant to correctness -- registration is idempotent per name and
# collisions raise -- but efferent imports afferent's relay algebra, so the two
# are kept adjacent to make that dependency visible rather than incidental.
from ibm.topologies import (          # noqa: F401  (imported for registration)
    local, surface, laminar, microcircuit, tract, vascular, interstitial, csf,
    em, mechanical, metabolic, afferent, efferent, device, neuromodulatory,
)

from ibm.topologies.afferent import AFFERENT_PATHWAY
from ibm.topologies.csf import CSF
from ibm.topologies.device import DEVICE_COUPLING
from ibm.topologies.efferent import EFFERENT_PATHWAY
from ibm.topologies.em import ELECTROMAGNETIC
from ibm.topologies.interstitial import INTERSTITIAL
from ibm.topologies.laminar import LAMINAR
from ibm.topologies.local import LOCAL
from ibm.topologies.mechanical import MECHANICAL
from ibm.topologies.metabolic import METABOLIC_EXCHANGE
from ibm.topologies.microcircuit import MICROCIRCUIT
from ibm.topologies.neuromodulatory import NEUROMODULATORY_PROJECTION
from ibm.topologies.surface import CORTICAL_SURFACE
from ibm.topologies.tract import TRACTOMETRIC
from ibm.topologies.vascular import VASCULAR

__all__ = [
    "builders", "BUILDERS", "BuilderSpec", "EdgeSet", "MissingInput", "SiteTable",
    "Sites", "as_sites", "build", "empty", "get",
    "LOCAL", "CORTICAL_SURFACE", "LAMINAR", "MICROCIRCUIT", "TRACTOMETRIC",
    "VASCULAR", "CSF", "INTERSTITIAL", "ELECTROMAGNETIC", "MECHANICAL",
    "METABOLIC_EXCHANGE", "AFFERENT_PATHWAY", "EFFERENT_PATHWAY", "DEVICE_COUPLING",
    "NEUROMODULATORY_PROJECTION",
]
