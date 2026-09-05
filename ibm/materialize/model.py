"""the materialized model: what a request turned into.

    M = materialize(R, r, B, F, A, T, P)

M is not a class of brain model.  it is one lazy view of the single implicit
model -- a site table, an allocation plan, a set of edge sets, a chosen f per
process, and an honest record of what all of that is resting on.  there is
deliberately no dynamics here: stepping the state is `ibm.runtime`'s job, and a
`MaterializedModel` that could override a process would have quietly become an
independently defined brain model, which is exactly the thing §7 says these are
not.

the cost accounting is the part worth reading twice.  ARCHITECTURE.md §1 makes
bandwidth co-equal with spatial resolution and says plainly that *the two budgets
multiply*, and a report that gives one number for "size" hides which of the two
axes is actually expensive.  so `Cost` carries sites and spectral coefficients
separately, reports their product, and names which axis dominates -- because the
remedy differs entirely.  a model that is spectral-coefficient-bound is fixed by
narrowing B(q) or shortening the window and needs no anatomy at all; a model that
is site-bound is fixed by coarsening r(q) somewhere, and `earns_its_cost` in
`build.py` says where that is free.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field, replace
from typing import Any, Mapping

import numpy as np

from ibm.fields.uncertainty.spectral import TemporalBasis
from ibm.materialize.provenance import Basis, Provenance
from ibm.materialize.request import Budget, MaterializationRequest
from ibm.materialize.sites import RegionResolver, SiteLayout
from ibm.materialize.trace import Trace
from ibm.registry import Implementation, Intervention, Observation
from ibm.runtime.state import Block, Layout
from ibm.topologies.builders import EdgeSet, SiteTable, Sites
from ibm.vocabulary import Everywhere, Prior, Region

_BYTES_PER_NUMBER = 8            # float64 / complex128 counted as two of them


# ---------------------------------------------------------------------------
# region weights
# ---------------------------------------------------------------------------


class RegionWeights:
    """(component, region) -> soft membership over that component's sites.

    a callable object rather than a dict because the set of regions a process may
    ask about is not known until the process runs: a learned f is allowed to build
    a region expression from state, and precomputing every possible answer is not
    an option.  results are memoized on the expression itself, which is safe
    because regions are frozen dataclasses and therefore hashable by value.

    this is the object `ibm.runtime.state.Layout.weights` delegates to, and the
    reason that delegation exists: a region is symbolic until materialization, so
    the only thing that can turn `Anat("cortical_layers", "iv")` into numbers is
    the model that built the site table.
    """

    __slots__ = ("sites", "resolver", "support_of", "_cache")

    def __init__(self, sites: Sites, resolver: RegionResolver,
                 support_of: Mapping[str, str]) -> None:
        self.sites = sites
        self.resolver = resolver
        self.support_of = dict(support_of)
        self._cache: dict[tuple[str, Any], np.ndarray] = {}

    def __call__(self, component: str, region: Region) -> np.ndarray:
        support = self.support_of.get(component)
        if support is None:
            raise KeyError(f"component {component!r} is not materialized in this model")
        table = self.sites[support]
        if isinstance(region, Everywhere):
            return np.ones(table.n)
        key = (support, region)
        try:
            hit = self._cache.get(key)
        except TypeError:                                          # unhashable region
            return self.resolver.weights(region, table.xyz, support)
        if hit is None:
            hit = self.resolver.weights(region, table.xyz, support)
            self._cache[key] = hit
        return hit

    def coverage(self, component: str, region: Region) -> float:
        """the fraction of a component's sites the region actually selects.

        worth having as a named thing: a selector that resolves to a coverage of
        zero is not an empty coupling, it is a materialization whose region and
        whose r(q) disagree -- the process was declared over a structure the
        request never sampled.
        """
        w = self(component, region)
        return float(w.mean()) if len(w) else 0.0


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cost:
    """the two laziness axes, their product, and the bytes that follow.

    `state_variables` is §1's own unit: one component of one field at one
    position.  `spectral_coefficients` is the temporal twin -- retained laplacian
    components summed over those variables.  neither bounds the other, which is
    why both are here and why the ratio between them is the single most
    informative number about a materialization's shape.  a whole-brain
    haemodynamic model and a single-electrode spike model can have the same byte
    count and nothing else in common.
    """

    state_variables: int = 0
    spectral_coefficients: int = 0
    numbers: int = 0
    edges: int = 0
    state_bytes: int = 0
    edge_bytes: int = 0
    per_component: Mapping[str, int] = _field(default_factory=dict)
    per_support: Mapping[str, int] = _field(default_factory=dict)

    @property
    def n_bytes(self) -> int:
        return self.state_bytes + self.edge_bytes

    @property
    def mean_bandwidth(self) -> float:
        """retained spectral components per state variable.  the shape, in one number."""
        return self.spectral_coefficients / self.state_variables if self.state_variables else 0.0

    @property
    def dominant_axis(self) -> str:
        """which budget to fix first.  the two remedies are unrelated.

        a bandwidth-dominated model is narrowed with B(q) or a shorter window and
        needs no anatomy touched at all; a site-dominated one is coarsened with
        r(q), and the §1 criterion says where that coarsening is free.
        """
        return "bandwidth" if self.mean_bandwidth > 8.0 else "resolution"

    def check(self, budget: Budget) -> list[str]:
        return budget.check(state_variables=self.state_variables,
                            spectral_coefficients=self.spectral_coefficients,
                            n_bytes=self.n_bytes, edges=self.edges)

    def describe(self) -> str:
        top = sorted(self.per_component.items(), key=lambda kv: -kv[1])[:6]
        lines = [f"  state variables       {self.state_variables:>15,}   (sites x components)",
                 f"  spectral coefficients {self.spectral_coefficients:>15,}   "
                 f"({self.mean_bandwidth:.1f} per variable)",
                 f"  numbers stored        {self.numbers:>15,}",
                 f"  edges                 {self.edges:>15,}",
                 f"  bytes                 {self.n_bytes / 2 ** 30:>15.2f} GiB   "
                 f"(state {self.state_bytes / 2 ** 30:.2f} + edges "
                 f"{self.edge_bytes / 2 ** 30:.2f})",
                 f"  dominant axis         {self.dominant_axis:>15s}"]
        if self.per_support:
            lines.append("  sites per support:    "
                         + ", ".join(f"{k} {v:,}" for k, v in sorted(self.per_support.items())))
        if top:
            lines.append("  costliest blocks:     "
                         + ", ".join(f"{k} {v:,}" for k, v in top))
        return "\n".join(lines)


def account(layout: Layout, sites: Sites,
            edges: Mapping[str, EdgeSet] | None = None) -> Cost:
    """count what a layout and its edge sets actually cost.

    counted from the allocation plan rather than from allocated arrays, because
    the whole point of checking a budget is to check it before anything is
    allocated -- an octree that discovers it is out of memory by running out of
    memory is the failure `Budget` exists to prevent.
    """
    n_vars = sum(b.n_sites for b in layout)
    n_coeff = sum(b.n_sites * max(b.k, 1) for b in layout if b.uncertainty != "scalar")
    numbers = sum(b.cost for b in layout)
    per_component = {b.component: b.cost for b in layout}
    per_support = {s: t.n for s, t in sites.tables.items()}

    n_edges, edge_bytes = 0, 0
    for e in (edges or {}).values():
        n_edges += e.n_edges
        edge_bytes += e.n_edges * 2 * 8
        for v in e.features.values():
            a = np.asarray(v)
            edge_bytes += int(a.size * a.dtype.itemsize) if a.dtype.kind in "fiub" else 0
    return Cost(state_variables=n_vars, spectral_coefficients=n_coeff, numbers=numbers,
                edges=n_edges, state_bytes=numbers * _BYTES_PER_NUMBER, edge_bytes=edge_bytes,
                per_component=per_component, per_support=per_support)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterializedModel:
    """one lazy view of the implicit model, ready for `ibm.runtime` to step.

    frozen, and the freezing matters: a materialization is a claim about what was
    built, and provenance is only trustworthy if the thing it describes cannot
    have been edited since.  a model that needs different sites or a different f
    is a different request, which is cheap -- that is what laziness bought.
    """

    request: MaterializationRequest
    trace: Trace
    sites: Sites
    site_layout: SiteLayout
    layout: Layout
    basis: TemporalBasis
    edges: Mapping[str, EdgeSet] = _field(default_factory=dict)
    implementations: Mapping[str, Implementation] = _field(default_factory=dict)
    priors: Mapping[str, Prior] = _field(default_factory=dict)      # "process.param" -> prior
    #: components an intervention pins.  clamped state is still ordinary state
    #: (§6); the clamp is a constraint applied by the runtime, not a different
    #: kind of variable, which is why this is a mapping and not a separate block.
    clamps: Mapping[str, Intervention] = _field(default_factory=dict)
    observations: Mapping[str, Observation] = _field(default_factory=dict)
    region_weights: RegionWeights | None = None
    provenance: Provenance = _field(default_factory=lambda: Provenance("unnamed"))
    cost: Cost = _field(default_factory=Cost)
    #: r(q) actually achieved per support, as (min, median, max) in mm.  the
    #: request asked for a spacing rule; this is what the sampler could deliver
    #: given each support's `min_spacing_mm` floor and the budget.
    achieved_spacing_mm: Mapping[str, tuple[float, float, float]] = _field(default_factory=dict)
    notes: tuple[str, ...] = ()

    # -- access ----------------------------------------------------------

    @property
    def name(self) -> str:
        return self.request.name

    @property
    def components(self) -> tuple[str, ...]:
        return self.layout.components

    @property
    def processes(self) -> tuple[str, ...]:
        return self.trace.processes

    def table(self, support: str) -> SiteTable:
        t = self.sites.get(support)
        if t is None:
            raise KeyError(f"{self.name}: support {support!r} was not materialized; this model "
                           f"holds {', '.join(sorted(self.sites.tables)) or 'none'}")
        return t

    def block(self, component: str) -> Block:
        return self.layout[component]

    def edge_set(self, topology: str) -> EdgeSet:
        e = self.edges.get(topology)
        if e is None:
            raise KeyError(
                f"{self.name}: topology {topology!r} has no edge set.  either the trace never "
                f"reached a process using it (it holds {', '.join(sorted(self.edges)) or 'none'}) "
                "or its builder could not run -- check provenance.missing")
        return e

    def implementation(self, process: str) -> Implementation:
        i = self.implementations.get(process)
        if i is None:
            raise KeyError(
                f"{self.name}: no f was selected for {process!r}.  §5 permits a process to exist "
                "in the ontology without a high-confidence implementation, so this is a gap in "
                "the ontology rather than in the build; provenance.unimplemented lists them")
        return i

    def weights(self, component: str, region: Region) -> np.ndarray:
        if self.region_weights is None:
            return np.ones(self.layout[component].n_sites)
        return self.region_weights(component, region)

    def positions(self, component: str) -> np.ndarray:
        """the positions of one component's state variables, in the model's frame."""
        return np.asarray(self.table(self.site_layout.support_of(component)).xyz)

    def volumes(self, component: str) -> np.ndarray:
        """per-site measure: mm^3 on a volume or tree support, mm^2 on a surface.

        the quantity a density becomes an amount through, and the one that adaptive
        sampling makes vary by orders of magnitude within a single block.  a
        process that sums a per-unit-volume rate over sites without this is wrong
        by exactly the local refinement factor, everywhere it refined.
        """
        t = self.table(self.site_layout.support_of(component))
        v = t.columns.get("volume_mm3")
        a = t.columns.get("area_mm2")
        if v is not None and np.isfinite(np.asarray(v, float)).any():
            return np.asarray(v, float)
        if a is not None:
            return np.asarray(a, float)
        return np.full(t.n, np.nan)

    # -- the §7 question -------------------------------------------------

    def rests_on(self, component: str) -> Basis:
        return self.provenance.rests_on(component, trace=self.trace)

    def audit(self) -> str:
        """the provenance report, with a `rests_on` for every target."""
        return self.provenance.audit(self.request.target_components)

    # -- derivation ------------------------------------------------------

    def with_provenance(self, p: Provenance) -> "MaterializedModel":
        return replace(self, provenance=p)

    # -- presentation ----------------------------------------------------

    def describe(self) -> str:
        """what was materialized, in the order someone reading it needs it.

        shape first (what is this model of, and how big), then the two budgets and
        which one dominates, then the honesty -- because a size report that omits
        which half of the model is running on `weak()` priors is the report that
        gets pasted into a paper.
        """
        w = self.request.window
        lines = [f"materialized {self.name!r} for subject {self.request.subject.id!r} "
                 f"in frame {self.request.frame!r}",
                 f"  targets     {', '.join(self.request.target_components) or '(none)'}",
                 f"  traced      {len(self.trace.components)} components, "
                 f"{len(self.trace.processes)} processes, "
                 f"{len(self.trace.topologies)} topologies, "
                 f"{len(self.trace.supports)} supports",
                 f"  window      n={w.n} dt={w.dt:g}s ({w.duration_s:g}s, "
                 f"nyquist {w.nyquist_hz:g} Hz), basis k={self.basis.k}",
                 "  sites"]
        for s in sorted(self.sites.tables):
            t = self.sites[s]
            lo, mid, hi = self.achieved_spacing_mm.get(s, (float("nan"),) * 3)
            lines.append(f"    {s:22s} {t.n:>10,} sites  r(q) {lo:g}/{mid:g}/{hi:g} mm  "
                         f"frame {t.frame}")
        if self.edges:
            lines.append("  edges")
            for k in sorted(self.edges):
                lines.append(f"    {self.edges[k].describe()}")
        lines.append("  cost")
        lines.append(self.cost.describe())

        sel = self.provenance.selections
        lti = sum(1 for s in sel if s.form.value == "lti")
        lines.append(f"  f           {lti}/{len(sel)} processes are LTI (exact at any timestep, "
                     "diagonal in the temporal basis)")
        n_par = len(self.provenance.parameters)
        moved = sum(1 for p in self.provenance.parameters if p.status == "moved")
        info = sum(1 for p in self.provenance.parameters if p.status == "informative prior")
        lines.append(f"  theta       {n_par} entries: {moved} moved by evidence, {info} at an "
                     f"informative prior, {n_par - moved - info} at a weak or speculative one")
        lossy = [c for c in self.provenance.conversions if c.lossy]
        if lossy:
            lines.append(f"  ! {len(lossy)} lossy uncertainty conversion(s); "
                         "model.audit() names what each destroys")
        if self.provenance.breaches:
            lines.append(f"  ! {len(self.provenance.breaches)} process(es) running outside the "
                         "regime their f was written for")
        if self.provenance.unimplemented:
            lines.append(f"  ! {len(self.provenance.unimplemented)} traced process(es) with no "
                         "implementation at all")
        if self.provenance.uses_template_geometry:
            lines.append("  ! built on template geometry: systematic positional error, not noise")
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<MaterializedModel {self.name!r} {self.cost.state_variables:,} vars x "
                f"{self.cost.mean_bandwidth:.1f} components, "
                f"{self.cost.n_bytes / 2 ** 30:.2f} GiB>")


__all__ = ["MaterializedModel", "Cost", "RegionWeights", "account"]
