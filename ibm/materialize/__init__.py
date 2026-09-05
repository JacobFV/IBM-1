"""M = materialize(R, r, B, F, A, T, P).

the package that turns a symbolic request into something `ibm.runtime` can step.
it is arranged as a pipeline of increasingly expensive stages, and the boundaries
between them are the point rather than an organizing convenience:

    request.py     R, r, B and the budget.  symbolic, microseconds, hashable
    trace.py       every process reachable from a target (§7).  still symbolic
    cache.py       the content address of everything below, computed from the above
    sites.py       grid(R, r): octrees, poisson disks, tree traversals.  minutes
    build.py       edges, f, frames, layout.  the assembly
    model.py       what came out, and what it costs on both axes
    provenance.py  what it is resting on
    library/       the named explicit models, as requests plus their accounting

the first three stages compute no geometry at all, which is what makes the fourth
cacheable: the key for an octree over a segmented head is known long before the
octree exists.

two things this package deliberately does not contain.  it holds no dynamics --
stepping state is `ibm.runtime`'s job, and a materialized model that could
override a process would have become an independently defined brain model, which
§7 says these are not.  and it declares nothing: no component, no topology, no
process.  every model in `library/` is a slice of the one registry, which is the
whole reason two of them cannot disagree about the dynamics of the same state
variable.

`library` is not imported here.  it binds source cards and named models, and
importing it as a side effect of touching a request would make the cheap symbolic
stage drag in the entire model catalogue.
"""

from __future__ import annotations

from ibm.materialize.build import (
    CoarseningReport,
    MaterializationIncomplete,
    ResolutionVerdict,
    build,
    coarsening_report,
    earns_its_cost,
    resolution_earns_its_cost,
    select_implementation,
)
from ibm.materialize.cache import Cache, spec_hash
from ibm.materialize.model import Cost, MaterializedModel, RegionWeights, account
from ibm.materialize.provenance import (
    Basis,
    ConversionRecord,
    FrameRecord,
    GeometryRecord,
    ParameterFate,
    Provenance,
    Selection,
    ValidityBreach,
)
from ibm.materialize.request import (
    Budget,
    BudgetExceeded,
    DeviceSpec,
    MaterializationRequest,
    SubjectSpec,
    Window,
    graded,
    graded_around,
    uniform,
)
from ibm.materialize.sites import (
    DiscreteGeometry,
    GeometrySet,
    MissingData,
    RegionResolver,
    SiteLayout,
    SurfaceGeometry,
    TreeGeometry,
    VolumeGeometry,
    build_sites,
)
from ibm.materialize.trace import CoreClique, Grounding, Trace, core_clique, trace

__all__ = [
    # the request
    "MaterializationRequest", "Budget", "BudgetExceeded", "Window", "SubjectSpec",
    "DeviceSpec", "uniform", "graded", "graded_around",
    # tracing
    "trace", "Trace", "Grounding", "core_clique", "CoreClique",
    # geometry and sites
    "GeometrySet", "VolumeGeometry", "SurfaceGeometry", "TreeGeometry", "DiscreteGeometry",
    "RegionResolver", "SiteLayout", "build_sites", "MissingData",
    # building
    "build", "MaterializationIncomplete", "select_implementation",
    "resolution_earns_its_cost", "earns_its_cost", "ResolutionVerdict",
    "coarsening_report", "CoarseningReport",
    # the result
    "MaterializedModel", "Cost", "RegionWeights", "account",
    # provenance
    "Provenance", "Basis", "Selection", "ParameterFate", "ConversionRecord",
    "ValidityBreach", "FrameRecord", "GeometryRecord",
    # caching
    "Cache", "spec_hash",
]
