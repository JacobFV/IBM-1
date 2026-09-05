"""turning a traced request into a runnable model.

this is where laziness stops being free.  everything before it -- the request,
the trace -- is symbolic and costs microseconds; from here on an octree gets
refined, an atlas gets resampled, a geodesic graph gets built and a lead field
gets solved, and the reason `cache.py` exists is that those take minutes and the
spec that determines them is known long before any of them run.

the order of the steps is not arbitrary and is worth stating, because each one
depends on the last being honest:

1. **trace** -- every process reachable from a target (§7)
2. **sites** -- grid(R, r) per support, with R restricting and r(q) refining
3. **frames** -- warp chains inserted, residuals carried as systematic error
4. **regions** -- symbolic regions become per-site weights against anatomy
5. **edges** -- topologies applied to the materialized state graph
6. **f** -- one implementation per process, by policy, with the rejects recorded
7. **layout** -- one block per component, in that component's form of uncertainty
8. **provenance** -- what all of the above is resting on

frames come third and not later because steps 4 and 5 are both statements about
distance: a region expression and a cross-support topology evaluated on positions
that are not yet in one frame produce numbers that look like geometry and are
not.

and one thing that is not a step but a question, asked in `earns_its_cost`:
ARCHITECTURE.md §1 says resolution earns its cost only where coarse-graining
fails to commute with the dynamics, and that r(q) should be derived from *that*
rather than from proximity to whatever we happen to be measuring.  that criterion
is decidable from what this module already knows -- the form of every process
acting on a region, the heterogeneity inside a coarse cell, and the length scale
of the topologies there -- so it is implemented as a function that returns a
verdict per region, and a fine materialization it calls waste really is waste:
under those three conditions the coarse and fine models are exactly equivalent,
not approximately.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ibm import frames as _frames
from ibm.materialize.model import Cost, MaterializedModel, RegionWeights, account
from ibm.materialize.provenance import (
    ConversionRecord, FrameRecord, GeometryRecord, Provenance, Selection, ValidityBreach,
    fates,
)
from ibm.materialize.request import Budget, BudgetExceeded, MaterializationRequest
from ibm.materialize.sites import (
    GeometrySet, MissingData, RegionResolver, SiteLayout, build_sites,
)
from ibm.materialize.trace import Trace, trace as trace_request
from ibm.registry import REGISTRY, Form, Implementation, Process
from ibm.runtime.state import Block, Layout
from ibm.topologies import builders as _builders
from ibm.topologies.builders import EdgeSet, MissingInput, SiteTable, Sites
from ibm.vocabulary import Band, Everywhere, FULL, Prior, Provenance as Prov, Tying


class MaterializationIncomplete(RuntimeError):
    """the build could not finish, with every reason at once.

    collected rather than raised at the first failure, for the same reason
    `Budget.check` returns every violation: a request that is missing a surface, a
    tractogram and a head model should be re-scoped once, not three times, and
    finding out about the third only after fixing the first two is a bad afternoon.
    """

    def __init__(self, request: str, problems: Sequence[str]) -> None:
        body = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(problems))
        super().__init__(f"cannot materialize {request!r}; {len(problems)} problem(s):\n{body}")
        self.problems = tuple(problems)


# ---------------------------------------------------------------------------
# selecting f
# ---------------------------------------------------------------------------

#: how much a declared provenance is worth when choosing between candidate f.
#: physics outranks a fit because a conservation law does not overfit; a fit
#: outranks the literature because it is this corpus rather than someone else's;
#: speculative sits at zero because "the functional form itself is a guess".
_PROV_SCORE: dict[Prov, float] = {
    Prov.PHYSICS: 6.0, Prov.FIT: 5.0, Prov.LITERATURE: 4.0, Prov.ATLAS: 4.0,
    Prov.DISTILLED: 3.0, Prov.WEAK: 1.0, Prov.SPECULATIVE: 0.0,
}

#: policy -> what it rewards.  a policy is a statement about what this
#: materialization is *for*, not about which f is true: `prefer_lti` is right for
#: a whole-brain forward model that must stay diagonal in the temporal basis, and
#: wrong for the one materialization whose entire point is the nonlinearity.
POLICIES = ("prefer_lti", "prefer_analytic", "prefer_learned", "prefer_evidence", "cheapest")


def score_implementation(impl: Implementation, *, policy: str = "prefer_lti",
                         spacing_mm: float = float("nan"), band: Band = FULL,
                         n_sites: int = 1) -> tuple[float, str]:
    """rank one candidate f under a policy, and say why in one line.

    validity dominates every other term on purpose.  an implementation whose form
    is meaningless at the spacing this request materializes at is not a slightly
    worse choice than one that is meaningful there; it is the wrong object, and
    only gets selected when nothing else exists -- in which case the breach is
    recorded rather than the selection quietly reversed.
    """
    s = _PROV_SCORE.get(impl.provenance, 1.0)
    why: list[str] = [impl.provenance.value]

    proc = REGISTRY.processes.get(impl.process)
    if proc is not None and np.isfinite(spacing_mm):
        v = proc.validity.violations(float(spacing_mm), band)
        if v:
            s -= 10.0 * len(v)
            why.append(f"{len(v)} validity violation(s) at {spacing_mm:g} mm")

    if impl.form is Form.LTI:
        if impl.transfer is None and impl.fn is None:
            s -= 5.0
            why.append("declared LTI but supplies no transfer function")
        else:
            why.append("lti: exact at any timestep, diagonal in the temporal basis")
    if policy == "prefer_lti" and impl.form is Form.LTI:
        s += 6.0
    elif policy == "prefer_analytic" and impl.form in (Form.LTI, Form.RATE):
        s += 4.0
    elif policy == "prefer_learned" and impl.form is Form.LEARNED:
        s += 6.0
    elif policy == "prefer_evidence" and impl.provenance in (Prov.FIT, Prov.DISTILLED):
        s += 6.0
    elif policy == "cheapest":
        if impl.form is Form.LTI:
            s += 6.0
        if impl.state_dependent_weights:
            s -= 4.0
            why.append("state-dependent weights: the edge weights are recomputed every step")
        if impl.tying is Tying.PER_SITE:
            s -= 2.0
            why.append(f"per-site theta over {n_sites:,} sites")
    if impl.fn is None and impl.transfer is None and impl.form is not Form.TABLE:
        s -= 3.0
        why.append("declared but not callable")
    return s, "; ".join(why)


def select_implementation(process: str, *, policy: str = "prefer_lti",
                          spacing_mm: float = float("nan"), band: Band = FULL,
                          n_sites: int = 1,
                          override: Mapping[str, str] | None = None
                          ) -> tuple[Implementation | None, Selection | None]:
    """choose one f for one process, and record what was passed over.

    an explicit override wins unconditionally and is still scored, because the
    interesting case is an override that the policy would not have chosen -- that
    disagreement is exactly what a reader of the provenance wants to see.
    """
    candidates = REGISTRY.impls_of(process)
    if not candidates:
        return None, None
    scored = sorted(
        ((score_implementation(i, policy=policy, spacing_mm=spacing_mm, band=band,
                               n_sites=n_sites), i) for i in candidates),
        key=lambda t: (-t[0][0], t[1].name))
    want = (override or {}).get(process)
    chosen = next((i for (_, _), i in scored if i.name == want), None) if want else None
    if want and chosen is None:
        raise KeyError(f"process {process!r} has no implementation named {want!r}; it has "
                       f"{', '.join(i.name for i in candidates)}")
    if chosen is None:
        (best_score, best_why), chosen = scored[0]
    else:
        best_score, best_why = next((sc, w) for (sc, w), i in scored if i is chosen)
        best_why = f"{best_why} [selected by override, not by policy {policy!r}]"
    rejected = tuple((i.name, w) for (sc, w), i in scored if i is not chosen)
    return chosen, Selection(process, chosen.name, chosen.form, chosen.provenance,
                             float(best_score), best_why, rejected, policy)


# ---------------------------------------------------------------------------
# when resolution earns its cost -- ARCHITECTURE.md §1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionVerdict:
    """whether refining r(q) over a region can change the answer at all.

    the §1 criterion, made decidable.  if the state over a region is smooth and
    every process acting there is linear, coarse-graining *commutes* with the
    dynamics -- the coarse and fine materializations are exactly equal, not close
    -- and the fine one is waste.  three things break the commutation and each has
    an observable proxy at build time:

    - a nonlinearity whose average is not the average's image.  the proxy is
      exact: `Form`, on the implementation actually selected.  an LTI process is
      diagonal in the temporal basis and commutes with any linear coarse-graining
      operator by construction.
    - heterogeneity inside a coarse cell.  the proxy is the spread of the
      anatomical memberships and material parameters over the sites a coarse cell
      would absorb; a cell straddling two cortical areas or a grey/white boundary
      does not average.
    - a topology whose edges do not survive coarsening.  the proxy is the edge
      length scale against the coarse spacing: a lateral topology with a 1 mm
      length constant materialized at 4 mm has no edges left to speak of, and a
      vascular tree loses its bifurcations outright.

    `earns` is deliberately conservative -- anything unknown counts as breaking --
    because the failure modes are asymmetric.  materializing too finely wastes
    memory, which is visible and annoying; coarse-graining a nonlinearity produces
    a plausible wrong answer, which is neither.
    """

    region: str
    support: str
    fine_mm: float
    coarse_mm: float
    earns: bool
    reasons: tuple[str, ...] = ()
    commuting: tuple[str, ...] = ()
    breaking: tuple[str, ...] = ()
    n_sites_fine: int = 0
    n_sites_coarse: int = 0
    wasted_state_variables: int = 0
    wasted_coefficients: int = 0

    @property
    def factor(self) -> float:
        return (self.n_sites_fine / self.n_sites_coarse) if self.n_sites_coarse else float("nan")

    def describe(self) -> str:
        head = ("EARNS" if self.earns else "WASTE")
        lines = [f"[{head}] {self.region or self.support}: {self.fine_mm:g} mm vs "
                 f"{self.coarse_mm:g} mm, {self.n_sites_fine:,} vs {self.n_sites_coarse:,} sites"]
        for r in self.reasons:
            lines.append(f"    {r}")
        if not self.earns:
            lines.append(f"    every process here is LTI and the state is smooth, so "
                         f"coarse-graining commutes and the two materializations are exactly "
                         f"equivalent: {self.wasted_state_variables:,} state variables and "
                         f"{self.wasted_coefficients:,} spectral coefficients bought nothing")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def resolution_earns_its_cost(
        processes: Iterable[str],
        implementations: Mapping[str, Implementation], *,
        fine_mm: float, coarse_mm: float,
        region: str = "", support: str = "",
        n_sites_fine: int = 0, n_sites_coarse: int = 0,
        k_components: int = 1,
        heterogeneity: float = 0.0,
        edge_length_mm: Mapping[str, float] | None = None,
        smooth: bool = True) -> ResolutionVerdict:
    """§1's criterion, as an actual function.

    `heterogeneity` is the caller's measurement of how much the sub-cell structure
    varies -- the standard deviation of anatomical membership or of a material
    parameter across the sites one coarse cell would absorb, normalized to [0, 1].
    zero means the coarse cell is homogeneous and its average is a faithful
    representative; anything above the threshold means it is not, and no amount of
    linearity rescues that.

    `smooth` is the escape hatch for a caller who knows the state is not smooth for
    a reason this function cannot see -- a travelling wave front, a seizure edge, a
    stimulation focus with a steep field gradient.  passing False forces `earns`.
    """
    procs = list(processes)
    breaking: list[str] = []
    commuting: list[str] = []
    reasons: list[str] = []

    for pid in procs:
        impl = implementations.get(pid)
        if impl is None:
            breaking.append(pid)
            reasons.append(f"{pid}: no f selected, so nothing here can be shown to commute")
            continue
        if impl.state_dependent_weights:
            breaking.append(pid)
            reasons.append(f"{pid} ({impl.name}): sets its own interaction weights from state, "
                           "so the coupling is nonlinear even where its form is not")
            continue
        if impl.form is Form.LTI:
            commuting.append(pid)
            continue
        if impl.form is Form.TABLE:
            if heterogeneity > 0.0:
                breaking.append(pid)
                reasons.append(f"{pid} ({impl.name}): a table lookup over a coarse cell that is "
                               f"not homogeneous (heterogeneity {heterogeneity:.2f})")
            else:
                commuting.append(pid)
            continue
        breaking.append(pid)
        reasons.append(f"{pid} ({impl.name}): form {impl.form.value} is not linear, so the "
                       "average of the dynamics is not the dynamics of the average")

    if not smooth:
        reasons.append("the caller declares the state over this region is not smooth")
    if heterogeneity > 0.15:
        reasons.append(f"sub-cell heterogeneity {heterogeneity:.2f}: a {coarse_mm:g} mm cell here "
                       "absorbs sites that do not average -- an area boundary, a tissue "
                       "boundary or a laminar transition")
    for topo, ell in sorted((edge_length_mm or {}).items()):
        if ell < coarse_mm:
            reasons.append(f"topology {topo!r} has a {ell:g} mm length scale, shorter than the "
                           f"{coarse_mm:g} mm coarse spacing: its edges do not survive coarsening")

    earns = bool(breaking) or not smooth or heterogeneity > 0.15 or any(
        ell < coarse_mm for ell in (edge_length_mm or {}).values())

    if not earns and not reasons:
        reasons.append(f"all {len(commuting)} process(es) here are LTI, the coarse cells are "
                       "homogeneous, and every topology outlives the coarse spacing")

    wasted = max(n_sites_fine - n_sites_coarse, 0)
    return ResolutionVerdict(
        region=region, support=support, fine_mm=float(fine_mm), coarse_mm=float(coarse_mm),
        earns=earns, reasons=tuple(reasons), commuting=tuple(sorted(commuting)),
        breaking=tuple(sorted(breaking)), n_sites_fine=int(n_sites_fine),
        n_sites_coarse=int(n_sites_coarse),
        wasted_state_variables=0 if earns else wasted,
        wasted_coefficients=0 if earns else wasted * max(int(k_components), 1))


def earns_its_cost(model: MaterializedModel, *, factor: float = 2.0,
                   heterogeneity: Mapping[str, float] | None = None,
                   smooth: Mapping[str, bool] | None = None) -> tuple[ResolutionVerdict, ...]:
    """apply the §1 criterion to every r(q) clause of a built model.

    one verdict per rule plus one for the default clause, because the default
    clause is a claim too: `ibm.materialize.library.res` says outright that it
    asserts coarse-graining commutes everywhere the named rules did not reach, and
    a materialization whose default region is full of nonlinear processes has made
    that claim falsely.
    """
    het = dict(heterogeneity or {})
    sm = dict(smooth or {})
    out: list[ResolutionVerdict] = []
    r = model.request.resolution
    clauses: list[tuple[str, float]] = [
        (f"rule[{i}] {type(rule.region).__name__}", float(rule.spacing_mm))
        for i, rule in enumerate(r.rules)]
    clauses.append(("default", float(r.default_mm)))

    lengths = _edge_length_scales(model)
    for support, table in sorted(model.sites.tables.items()):
        if table.columns.get("kind") == "discrete":
            continue
        procs = _processes_on(model, support)
        spacing = np.asarray(table.spacing(np), float)
        spacing = spacing[np.isfinite(spacing)]
        if not len(spacing):
            continue
        k = max((model.layout[c].k for c in model.layout.components
                 if model.site_layout.support_of(c) == support), default=1)
        for name, mm in clauses:
            sel = spacing <= mm * 1.001
            n_fine = int(sel.sum())
            if not n_fine:
                continue
            n_coarse = max(int(round(n_fine / (factor ** _dim(support)))), 1)
            out.append(resolution_earns_its_cost(
                procs, model.implementations, fine_mm=mm, coarse_mm=mm * factor,
                region=f"{support}:{name}", support=support,
                n_sites_fine=n_fine, n_sites_coarse=n_coarse, k_components=k,
                heterogeneity=het.get(support, _heterogeneity(table)),
                edge_length_mm={t: l for t, l in lengths.items()
                                if support in REGISTRY.topologies[t].on},
                smooth=sm.get(support, True)))
    return tuple(out)


def _dim(support: str) -> int:
    s = REGISTRY.supports.get(support)
    return int(s.dimension) if s is not None else 3


def _processes_on(model: MaterializedModel, support: str) -> list[str]:
    """the traced processes that write a component living on this support."""
    out = []
    for pid in model.trace.processes:
        p = REGISTRY.processes.get(pid)
        if p is None:
            continue
        for s in p.outputs:
            if any(model.site_layout.support_of(v) == support
                   for v in s.vars if v in model.site_layout):
                out.append(pid)
                break
    return out


def _heterogeneity(table: SiteTable) -> float:
    """a cheap, honest proxy: how uneven the anatomical memberships already are.

    if no partitioning system was resampled onto this table there is nothing to
    measure, and the function returns zero -- which makes the verdict *less*
    conservative, so the docstring says so out loud rather than letting an absent
    atlas quietly argue for coarsening.
    """
    parts = getattr(table, "partitions", None) or {}
    if not parts:
        occ = table.columns.get("occupancy")
        if occ is None:
            return 0.0
        occ = np.asarray(occ, float)
        return float(np.mean((occ > 0.02) & (occ < 0.98)))
    spread = []
    for a in parts.values():
        a = np.asarray(a, float)
        if a.ndim == 2 and a.shape[1] > 1:
            spread.append(float(np.mean(1.0 - a.max(axis=1))))
    return float(np.mean(spread)) if spread else 0.0


def _edge_length_scales(model: MaterializedModel) -> dict[str, float]:
    """median edge length per built topology: the scale that must survive coarsening."""
    out: dict[str, float] = {}
    for name, e in model.edges.items():
        if name not in REGISTRY.topologies or not e.n_edges:
            continue
        d = e.features.get("distance_mm", e.features.get("length_mm"))
        if d is None:
            continue
        d = np.asarray(d, float)
        d = d[np.isfinite(d) & (d > 0)]
        if len(d):
            out[name] = float(np.median(d))
    return out


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------


def build(request: MaterializationRequest, *,
          geometry: GeometrySet | Mapping[str, Any] | None = None,
          anatomy: Callable[[str, str, Any, str], Any] | None = None,
          warp: Callable[[Any, str, str], Any] | None = None,
          geodesic: Callable[[str, Any, Any], Any] | None = None,
          topology_inputs: Mapping[str, Mapping[str, Any]] | None = None,
          implementation_override: Mapping[str, str] | None = None,
          moved_parameters: Mapping[str, Iterable[str]] | None = None,
          cache: Any = None,
          stop_at_observations: bool = False,
          strict: bool = True) -> MaterializedModel:
    """materialize one request into a runnable model.

    every argument after `request` is external data or an external decision.  none
    of it can be invented, and none of it is silently defaulted: an absent atlas
    makes `Anat(...)` regions raise with the system and label named, an absent
    tractogram makes the tract topology raise with the file named, and an absent
    subject surface makes the surface sampler raise rather than sampling a
    template and reporting subject coordinates.

    `strict=False` builds anyway and records every gap in provenance.  it exists
    for cost estimation and for the coarsening argument, where the question is how
    big a thing would be rather than what it would predict, and a model built that
    way says so in its own `describe()`.
    """
    problems: list[str] = []
    notes: list[str] = []

    # 1. trace -----------------------------------------------------------
    tr = trace_request(request, stop_at_observations=stop_at_observations)
    if tr.notes:
        notes.extend(tr.notes)

    # 2. sites -----------------------------------------------------------
    geom = geometry if isinstance(geometry, GeometrySet) else GeometrySet(
        dict(geometry or {}), request.subject.id)
    resolver = RegionResolver(frame=request.frame, anchors=request.anchors(),
                              anatomy=anatomy, warp=warp, geodesic=geodesic)
    try:
        raw = build_sites(request, tr.supports, geom, resolver=resolver, cache=cache)
    except (MissingData, MissingInput) as exc:
        if strict:
            raise
        problems.append(str(exc))
        raw = Sites({})
    sites = _restrict_to_scope(raw, request, resolver, notes)

    support_of = {c: _support_of(c) for c in tr.components}
    unplaced = sorted({c for c, s in support_of.items() if s not in sites.tables})
    if unplaced:
        problems.append(
            f"{len(unplaced)} traced component(s) live on a support with no site table: "
            + ", ".join(f"{c} on {support_of[c]!r}" for c in unplaced[:6]))

    # 3. frames ----------------------------------------------------------
    # before regions and before edges, deliberately.  a region expression and a
    # cross-support topology are both statements about distance, and evaluating
    # either against positions that have not yet been brought into one frame is
    # the silent misregistration ibm.frames exists to prevent.
    frame_records, sites, frame_problems = _resolve_frames(sites, request, warp)
    if strict:
        problems.extend(frame_problems)
    else:
        notes.extend(frame_problems)

    # 4. regions ---------------------------------------------------------
    weights = RegionWeights(sites, resolver, support_of)

    # 5. edges -----------------------------------------------------------
    edges, edge_problems = _build_edges(tr, sites, topology_inputs or {}, request.budget)
    problems.extend(edge_problems)

    # 6. f ---------------------------------------------------------------
    selections: list[Selection] = []
    impls: dict[str, Implementation] = {}
    priors: dict[str, Prior] = {}
    param_fates = []
    unimplemented: list[str] = []
    breaches: list[ValidityBreach] = []
    spacing = {s: _spacing_stats(t) for s, t in sites.tables.items()}

    for pid in tr.processes:
        proc = REGISTRY.processes[pid]
        support, n_sites, mm = _process_scale(proc, support_of, sites, spacing)
        band = _process_band(proc, tr)
        impl, sel = select_implementation(pid, policy=request.policy, spacing_mm=mm,
                                          band=band, n_sites=n_sites,
                                          override=implementation_override)
        if impl is None or sel is None:
            unimplemented.append(pid)
            continue
        impls[pid] = impl
        selections.append(sel)
        n_part = _n_partitions(sites.get(support))
        for f in fates(impl, n_sites=n_sites, n_partitions=n_part):
            param_fates.append(f)
            priors[f"{pid}.{f.name}"] = f.prior
        v = proc.validity.violations(mm, band) if np.isfinite(mm) else []
        if v:
            breaches.append(ValidityBreach(pid, impl.name, support, mm, band, proc.validity,
                                           tuple(v), n_sites))

    # 7. layout ----------------------------------------------------------
    basis = request.window.basis()
    blocks: list[Block] = []
    pairs: list[tuple[str, str, int]] = []
    for cid in tr.components:
        support = support_of[cid]
        table = sites.get(support)
        if table is None:
            continue
        comp = REGISTRY.components[cid]
        band = tr.bands.get(cid, comp.band) & comp.band
        b = basis.truncated(band) if comp.uncertainty != "scalar" else None
        blocks.append(Block(cid, comp.uncertainty, table.n, band, b, None, support,
                            spacing[support][1]))
        pairs.append((cid, support, table.n))
    layout = Layout(tuple(blocks), basis, weights)
    site_layout = SiteLayout.of(pairs)

    cost = account(layout, sites, edges)
    violations = cost.check(request.budget)
    if violations:
        if strict:
            raise BudgetExceeded("; ".join(violations))
        notes.extend(violations)

    # 8. provenance ------------------------------------------------------
    conversions = _conversions(tr)
    geometry_records = tuple(
        GeometryRecord(s, str(t.columns.get("geometry_source", "") or ""), t.n,
                       "template" in str(t.columns.get("geometry_source", "")).lower(),
                       t.frame)
        for s, t in sorted(sites.tables.items()))
    prov = Provenance(
        request=request.name, subject=request.subject.id, policy=request.policy,
        selections=tuple(selections), parameters=tuple(param_fates),
        conversions=conversions, breaches=tuple(breaches), frames=tuple(frame_records),
        geometry=geometry_records, unimplemented=tuple(sorted(unimplemented)),
        missing=tuple(problems) if not strict else (), notes=tuple(notes), trace=tr)
    if moved_parameters:
        prov = prov.with_evidence(moved_parameters)

    if problems and strict:
        raise MaterializationIncomplete(request.name, problems)

    clamps = {v: REGISTRY.interventions[i] for i in request.interventions
              if i in REGISTRY.interventions
              for v in REGISTRY.interventions[i].constrains.vars}
    observations = {o: REGISTRY.observations[o] for o in request.observations
                    if o in REGISTRY.observations}

    return MaterializedModel(
        request=request, trace=tr, sites=sites, site_layout=site_layout, layout=layout,
        basis=basis, edges=edges, implementations=impls, priors=priors, clamps=clamps,
        observations=observations, region_weights=weights, provenance=prov, cost=cost,
        achieved_spacing_mm=spacing, notes=tuple(notes))


# ---------------------------------------------------------------------------
# build steps
# ---------------------------------------------------------------------------


def _support_of(cid: str) -> str:
    c = REGISTRY.components.get(cid)
    if c is None:
        return ""
    if c.support:
        return c.support
    f = REGISTRY.fields.get(c.field)
    return f.support if f is not None else ""


def _spacing_stats(t: SiteTable) -> tuple[float, float, float]:
    s = np.asarray(t.spacing(np), float)
    s = s[np.isfinite(s)]
    if not len(s):
        return (float("nan"),) * 3
    return float(s.min()), float(np.median(s)), float(s.max())


def _n_partitions(t: SiteTable | None) -> int:
    if t is None:
        return 1
    n = 0
    for a in (getattr(t, "partitions", None) or {}).values():
        a = np.asarray(a)
        n += int(a.shape[1]) if a.ndim == 2 else 1
    return max(n, 1)


def _restrict_to_scope(sites: Sites, request: MaterializationRequest,
                       resolver: RegionResolver, notes: list[str]) -> Sites:
    """apply R, after r(q) has done the sampling.

    the two are separate for a reason: r(q) decides *how finely* to sample and R
    decides *where*, and evaluating R first would mean a region expression could
    only ever be tested at the coarse level a preliminary grid happened to use.
    sampling then restricting is exact, and the only cost is sites generated
    outside R and immediately dropped -- which the octree's occupancy test has
    usually already pruned.

    discrete supports are never restricted.  an electrode outside the named region
    is still part of the instrument, and a materialization that dropped half a
    montage because the region was cortical would be describing a different device.
    """
    if not request.regions:
        return sites
    scope = request.scope
    if isinstance(scope, Everywhere):
        return sites
    out: dict[str, SiteTable] = {}
    offset = 0
    for s in sorted(sites.tables):
        t = sites[s]
        if t.columns.get("kind") == "discrete":
            out[s] = replace(t, offset=offset)
            offset += t.n
            continue
        w = resolver.weights(scope, t.xyz, s)
        keep = np.flatnonzero(w > 0.0)
        if len(keep) == t.n:
            out[s] = replace(t, offset=offset)
        elif not len(keep):
            notes.append(f"region R selects none of the {t.n:,} sampled sites on {s!r}: the "
                         "named regions and r(q) disagree about where this support is")
            out[s] = _take(t, keep, offset)
        else:
            notes.append(f"R kept {len(keep):,} of {t.n:,} sites on {s!r}")
            out[s] = _take(t, keep, offset)
        offset += out[s].n
    return Sites(out)


def _take(t: SiteTable, keep: np.ndarray, offset: int) -> SiteTable:
    """subset a site table, carrying only the columns that are per-site.

    a surface table also carries its mesh, and a tree table carries a parent array
    whose *values* are site indices; neither survives naive masking, so the mesh
    is passed through untouched and the parent array is remapped.  silently
    truncating either would produce a table that looks right and indexes into
    nothing.
    """
    n = t.n
    cols: dict[str, Any] = {}
    remap = np.full(n, -1, np.int64)
    remap[keep] = np.arange(len(keep))
    for k, v in t.columns.items():
        a = np.asarray(v) if isinstance(v, np.ndarray) else v
        if isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[0] == n:
            if k == "parent":
                p = a[keep]
                cols[k] = np.where(p >= 0, remap[np.clip(p, 0, n - 1)], -1)
            else:
                cols[k] = a[keep]
        else:
            cols[k] = v
    sp = np.asarray(t.spacing(np), float)[keep]
    return SiteTable(t.support, t.frame, np.asarray(t.xyz)[keep], offset, sp, cols,
                     {k: np.asarray(v)[keep] for k, v in (t.partitions or {}).items()})


def _build_edges(tr: Trace, sites: Sites, inputs: Mapping[str, Mapping[str, Any]],
                 budget: Budget) -> tuple[dict[str, EdgeSet], list[str]]:
    """apply every traced topology to the materialized state graph.

    a topology declares which state variables *may* interact; the edge set is that
    declaration applied to the sites this request happened to instantiate (§3).
    the two are separate objects because the declaration is ontology and must seal
    without touching data, while the edge set is arithmetic over one octree.
    """
    out: dict[str, EdgeSet] = {}
    problems: list[str] = []
    total = 0
    for name in tr.topologies:
        topo = REGISTRY.topologies.get(name)
        if topo is None:
            problems.append(f"topology {name!r} is used by a traced process but is not registered")
            continue
        if not topo.builder:
            problems.append(
                f"topology {name!r} declares no builder, so it cannot become an edge set.  a "
                "process over it has a declared support for interaction and no way to compute "
                "it; register a builder in ibm.topologies")
            continue
        try:
            spec = _builders.get(topo.builder)
        except KeyError as exc:
            problems.append(str(exc))
            continue
        try:
            e = spec(sites, **dict(inputs.get(name, {})))
        except MissingInput as exc:
            problems.append(str(exc))
            continue
        except MissingData as exc:
            problems.append(str(exc))
            continue
        out[name] = e
        total += e.n_edges
        if total > budget.max_edges:
            raise BudgetExceeded(
                f"edge sets reached {total:,} edges at topology {name!r}, past the "
                f"{budget.max_edges:,} ceiling.  a local topology's edge count grows as the cube "
                "of its radius over the spacing, so this is usually one radius that was written "
                "for a coarser r(q) than the request materializes at")
    return out, problems


def _process_scale(proc: Process, support_of: Mapping[str, str], sites: Sites,
                   spacing: Mapping[str, tuple[float, float, float]]
                   ) -> tuple[str, int, float]:
    """the support, site count and median spacing a process is being run at.

    taken from the *outputs*, because validity is a statement about the state the
    form produces.  a coupling that reads a whole-brain field and writes a cortical
    one is being asked to be meaningful at the cortical spacing, not at the coarse
    one it happens to read.
    """
    for s in proc.outputs:
        for v in s.vars:
            sup = support_of.get(v)
            t = sites.get(sup) if sup else None
            if t is not None:
                return sup, t.n, spacing.get(sup, (float("nan"),) * 3)[1]
    return "", 0, float("nan")


def _process_band(proc: Process, tr: Trace) -> Band:
    """the band a process is actually materialized over: its outputs', narrowed by B(q)."""
    out: Band | None = None
    for s in proc.outputs:
        for v in s.vars:
            b = tr.bands.get(v)
            if b is None:
                continue
            cand = s.band & b
            out = cand if out is None else Band(min(out.lo_hz, cand.lo_hz),
                                                max(out.hi_hz, cand.hi_hz))
    return out or FULL


def _conversions(tr: Trace) -> tuple[ConversionRecord, ...]:
    """every place a belief changes form, with what the registry says it destroys.

    read off the declarations rather than observed at run time, which is the only
    way this can be in the record *before* anything runs.  the registry already
    refuses to seal a process coupling two forms with no declared conversion (§8),
    so every pair found here has an entry.
    """
    out: list[ConversionRecord] = []
    for pid in tr.processes:
        p = REGISTRY.processes.get(pid)
        if p is None:
            continue
        ins = {v: REGISTRY.components[v].uncertainty for s in p.inputs for v in s.vars
               if v in REGISTRY.components}
        outs = {v: REGISTRY.components[v].uncertainty for s in p.outputs for v in s.vars
                if v in REGISTRY.components}
        for vi, fi in sorted(ins.items()):
            for vo, fo in sorted(outs.items()):
                if fi == fo:
                    continue
                conv = REGISTRY.conversions.get((fi, fo))
                out.append(ConversionRecord(
                    pid, vi, vo, fi, fo,
                    bool(getattr(conv, "lossy", True)),
                    str(getattr(conv, "destroys", "")) or
                    ("no conversion is registered for this pair, which the registry should have "
                     "refused to seal" if conv is None else "")))
    return tuple(out)


def _resolve_frames(sites: Sites, request: MaterializationRequest,
                    warp: Callable[[Any, str, str], Any] | None
                    ) -> tuple[list[FrameRecord], Sites, list[str]]:
    """insert the warp chain from every support's frame into the model's frame.

    a missing chain is a hard error and not something to approximate around: two
    sets of positions in unrelated frames have no meaningful distance between
    them, and every topology built across the pair would be fiction that looks
    like geometry.  a *declared* chain with no transform behind it is a different
    situation -- the model knows what warp it needs and what that warp's residual
    would be -- so the record is written and the positions are left where they are,
    with the gap named rather than papered over.
    """
    records: list[FrameRecord] = []
    problems: list[str] = []
    out: dict[str, SiteTable] = {}
    for s in sorted(sites.tables):
        t = sites[s]
        dst = request.frame
        if t.frame == dst:
            out[s] = t
            records.append(FrameRecord(s, t.frame, dst, (), 0.0, True, "already in frame"))
            continue
        chain = _frames.path(t.frame, dst)
        if chain is None:
            problems.append(
                f"support {s!r} has positions in frame {t.frame!r} and the model is being built "
                f"in {dst!r}, and ibm.frames declares no warp chain between them.  that is a "
                "modelling gap, not a missing file: declare the warp in ibm.frames, or build "
                "this request in a frame the support can reach")
            out[s] = t
            continue
        res = _frames.residual_mm(t.frame, dst)
        records.append(FrameRecord(s, t.frame, dst, tuple(w.method for w in chain), res,
                                   all(w.subject_specific for w in chain),
                                   "; ".join(w.note for w in chain if w.note)))
        if warp is None:
            problems.append(
                f"support {s!r} needs the {t.frame!r} -> {dst!r} warp "
                f"({' -> '.join(w.method for w in chain)}, residual "
                f"{'unknown' if res is None else f'{res:.1f} mm'}) and no transform was supplied. "
                "pass warp=... to build(); its positions are being left in their own frame, so "
                "any topology built across this support and another is not geometry")
            out[s] = t
            continue
        xyz = np.asarray(warp(np.asarray(t.xyz, float), t.frame, dst), float).reshape(-1, 3)
        out[s] = replace(t, xyz=xyz, frame=dst)
    return records, Sites(out), problems


# ---------------------------------------------------------------------------
# the coarsening argument, run rather than asserted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoarseningReport:
    """the fine and coarse materializations side by side, and the verdict.

    `MaterializationRequest.coarsened` exists precisely so this can be run: §1's
    claim is that under linearity and smoothness the two are *exactly* equivalent,
    which is a checkable statement rather than a stylistic preference.  what this
    reports is the structural half -- whether anything upstream could break the
    commutation, and what the fine model paid for the privilege.  the numeric half
    (step both, compare trajectories) belongs to `ibm.runtime` and needs a state
    vector, not a layout.
    """

    fine: Cost
    coarse: Cost
    factor: float
    verdicts: tuple[ResolutionVerdict, ...]

    @property
    def wasted(self) -> bool:
        return all(not v.earns for v in self.verdicts) and bool(self.verdicts)

    def describe(self) -> str:
        lines = [f"coarsening by {self.factor:g}x:",
                 f"  fine   {self.fine.state_variables:>12,} vars  "
                 f"{self.fine.spectral_coefficients:>15,} coefficients  "
                 f"{self.fine.n_bytes / 2 ** 30:.2f} GiB",
                 f"  coarse {self.coarse.state_variables:>12,} vars  "
                 f"{self.coarse.spectral_coefficients:>15,} coefficients  "
                 f"{self.coarse.n_bytes / 2 ** 30:.2f} GiB"]
        lines += ["  " + v.describe() for v in self.verdicts]
        if self.wasted:
            lines.append("  => every region commutes: the fine materialization is exactly "
                         "equivalent to the coarse one and its extra cost bought nothing")
        return "\n".join(lines)


def coarsening_report(request: MaterializationRequest, *, factor: float = 2.0,
                      **build_kw: Any) -> CoarseningReport:
    """materialize a request and its coarsened twin, and compare.

    built with `strict=False` unless the caller overrides, because the question
    here is how big a thing would be and whether refining it could matter -- and
    that question is answerable with a missing tractogram, whereas a prediction is
    not.
    """
    build_kw.setdefault("strict", False)
    fine = build(request, **build_kw)
    coarse = build(request.coarsened(factor), **build_kw)
    verdicts = earns_its_cost(fine, factor=factor)
    return CoarseningReport(fine.cost, coarse.cost, float(factor), verdicts)


__all__ = ["build", "MaterializationIncomplete", "ResolutionVerdict",
           "resolution_earns_its_cost", "earns_its_cost", "select_implementation",
           "score_implementation", "CoarseningReport", "coarsening_report", "POLICIES"]
