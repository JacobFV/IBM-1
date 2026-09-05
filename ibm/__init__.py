"""implicit brain model (ibm-1).

    ibm = (fields, anatomy, topologies, processes)

everything else -- resolution, bandwidth, datasets, observations, interventions,
explicit models -- is expressed through those four.  see docs/ARCHITECTURE.md.

nothing in this package computes until it is materialized.  the ontology declares
what exists; `ibm.materialize` turns a request into a runnable model; `ibm.runtime`
runs it; `ibm.forge` updates p(theta | D).
"""

__version__ = "0.1.0"

from ibm.registry import REGISTRY

__all__ = ["REGISTRY", "load_all"]


def load_all(seal: bool = True, strict: bool = False):
    """import every ontology module so the registry is populated, then seal it."""
    import importlib

    for m in (
        "ibm.frames",
        "ibm.fields", "ibm.anatomy", "ibm.topologies", "ibm.processes",
    ):
        importlib.import_module(m)
    if seal:
        REGISTRY.seal(strict=strict)
    return REGISTRY
