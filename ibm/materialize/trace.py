"""dependency tracing: which processes and components a target actually needs.

ARCHITECTURE.md §7 settles the hard question before it is asked.  *every process
reachable from a target is materialized*, and the process graph is finite.  so
this is a backward breadth-first search over a bipartite component <-> process
graph with a visited set, and it terminates for the same reason any traversal of
a finite graph terminates.  there is no fixed point to iterate to, no closure
operator, and no convergence criterion; the recurrent structure of the brain
means the graph has cycles, and a cycle is a revisit, not a divergence.

what is worth computing is everything else.

*where the trace bottoms out.*  a component with no writers must be exogenous, or
clamped by an intervention, or the registry would have refused to seal (§8: "read
but never written, not exogenous, and not clamped -- the graph has a hole here").
recording which of the three applies is how a reader learns whether a
materialization is grounded in physics, in an experimenter's waveform, or in
nothing but a prior.

*the core clique.*  §7 observes that because the graph is finite, the core clique
is shared across most materializations and R, r and B are the levers by which
they actually differ.  that is a claim about this codebase that can be checked
rather than asserted: trace several requests and intersect.  when the intersection
is small, the ontology has fragmented into unrelated sub-models and the "one
implicit model" story has quietly stopped being true.

*what the trace pulled in that the request did not ask for.*  the four ontology
tuples on a request are additive hints, and the honest report is the difference
between what was named and what was reached -- usually large, because asking for
scalp potential drags in the whole cortical column, the volume conductor, and
every ionic process underneath.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field as _field
from typing import Any, Iterable, Mapping, Sequence

from ibm.materialize.request import MaterializationRequest
from ibm.registry import REGISTRY, Process
from ibm.vocabulary import Band, Region, field_of


#: why the backward walk stopped at a component.  ordered by how much a reader
#: should trust what sits underneath: physics and instruments first, priors last.
GROUND_EXOGENOUS = "exogenous"          # declared as given: a material constant, a stimulus
GROUND_INTERVENTION = "intervention"    # clamped by the experimenter (§6)
GROUND_OBSERVATION = "observation"      # evidence enters here (§6)
GROUND_UNWRITTEN = "unwritten"          # nothing writes it and nothing declared it given
GROUND_INTERIOR = "interior"            # ordinary state with writers; the walk continued


@dataclass(frozen=True)
class Grounding:
    """one leaf of the backward walk and the reason it is one."""

    component: str
    kind: str
    via: tuple[str, ...] = ()           # intervention or observation ids, when relevant
    note: str = ""

    def __str__(self) -> str:
        v = f" via {', '.join(self.via)}" if self.via else ""
        return f"{self.component} <- {self.kind}{v}" + (f" ({self.note})" if self.note else "")


@dataclass(frozen=True)
class Trace:
    """the reachable sub-ontology of one request.

    frozen and cheap: no geometry, no atlas, no arithmetic.  a trace is the second
    symbolic stage after the request, and holding the two apart is what lets the
    cache key for an expensive artefact be computed before the artefact exists.
    """

    request_name: str
    components: tuple[str, ...] = ()
    processes: tuple[str, ...] = ()
    topologies: tuple[str, ...] = ()
    supports: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    anatomies: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    interventions: tuple[str, ...] = ()
    groundings: tuple[Grounding, ...] = ()
    #: component -> the processes that write it, restricted to the traced set
    writers: Mapping[str, tuple[str, ...]] = _field(default_factory=dict)
    #: component -> the processes that read it, restricted to the traced set
    readers: Mapping[str, tuple[str, ...]] = _field(default_factory=dict)
    #: process -> the processes it depends on directly (through its inputs)
    depends_on: Mapping[str, tuple[str, ...]] = _field(default_factory=dict)
    #: component -> widest band any traced process touches it over, intersected
    #: with the request's B(q).  the layout allocates against this.
    bands: Mapping[str, Band] = _field(default_factory=dict)
    #: component -> the regions traced selectors restrict it to
    regions: Mapping[str, tuple[Region, ...]] = _field(default_factory=dict)
    #: what the request named but the trace never reached
    unreached_hints: tuple[str, ...] = ()
    #: processes pulled in because they write another traced process's theta
    parameter_writers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    # -- membership ------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self.components or name in self.processes

    @property
    def n(self) -> tuple[int, int]:
        return len(self.components), len(self.processes)

    def component_set(self) -> frozenset[str]:
        return frozenset(self.components)

    def process_set(self) -> frozenset[str]:
        return frozenset(self.processes)

    def grounding(self, component: str) -> Grounding:
        for g in self.groundings:
            if g.component == component:
                return g
        return Grounding(component, GROUND_INTERIOR)

    def grounded_at(self, kind: str) -> tuple[str, ...]:
        return tuple(g.component for g in self.groundings if g.kind == kind)

    # -- backward walk over the traced graph -----------------------------

    def upstream(self, component: str, *, max_depth: int | None = None) -> tuple[str, ...]:
        """every traced process that can influence a component, at any depth.

        the same walk as the trace itself, restricted to what the trace kept, and
        the thing provenance stands on: "is this prediction resting on constrained
        structure or on a prior?" is a question about exactly this set.
        """
        seen_c: set[str] = {component}
        seen_p: set[str] = set()
        frontier = deque([(component, 0)])
        while frontier:
            c, d = frontier.popleft()
            if max_depth is not None and d >= max_depth:
                continue
            for p in self.writers.get(c, ()):
                if p in seen_p:
                    continue
                seen_p.add(p)
                proc = REGISTRY.processes.get(p)
                if proc is None:
                    continue
                for s in proc.inputs:
                    for v in s.vars:
                        if v in seen_c or v not in self.component_set():
                            continue
                        seen_c.add(v)
                        frontier.append((v, d + 1))
        return tuple(sorted(seen_p))

    def downstream(self, component: str) -> tuple[str, ...]:
        """every traced process a component can influence.  the mirror walk."""
        seen_c, seen_p = {component}, set()
        frontier = deque([component])
        while frontier:
            c = frontier.popleft()
            for p in self.readers.get(c, ()):
                if p in seen_p:
                    continue
                seen_p.add(p)
                proc = REGISTRY.processes.get(p)
                if proc is None:
                    continue
                for s in proc.outputs:
                    for v in s.vars:
                        if v in self.component_set() and v not in seen_c:
                            seen_c.add(v)
                            frontier.append(v)
        return tuple(sorted(seen_p))

    # -- comparison ------------------------------------------------------

    def shared_with(self, other: "Trace") -> frozenset[str]:
        return self.process_set() & other.process_set()

    def describe(self) -> str:
        by_kind: dict[str, list[str]] = {}
        for g in self.groundings:
            by_kind.setdefault(g.kind, []).append(g.component)
        lines = [f"trace of {self.request_name!r}",
                 f"  {len(self.components)} components over {len(self.fields)} fields and "
                 f"{len(self.supports)} supports",
                 f"  {len(self.processes)} processes over {len(self.topologies)} topologies"
                 + (f" ({len(self.parameter_writers)} of them write theta)"
                    if self.parameter_writers else ""),
                 f"  {len(self.anatomies)} anatomical systems"]
        for kind in (GROUND_EXOGENOUS, GROUND_INTERVENTION, GROUND_OBSERVATION,
                     GROUND_UNWRITTEN):
            got = sorted(by_kind.get(kind, ()))
            if got:
                head = ", ".join(got[:6]) + (f" (+{len(got) - 6})" if len(got) > 6 else "")
                lines.append(f"  grounded at {kind:13s} {len(got):3d}: {head}")
        if self.unreached_hints:
            lines.append("  named on the request but never reached: "
                         + ", ".join(self.unreached_hints))
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the walk
# ---------------------------------------------------------------------------


def trace(request: MaterializationRequest, *, stop_at_observations: bool = False,
          registry: Any = None) -> Trace:
    """backward reachability from the request's targets.  §7's rule, literally.

    `stop_at_observations` is the one policy knob, and it is off by default.  an
    observation is evidence about state, not a substitute for the dynamics that
    produced it (§6), so a forward model must keep tracing through an observed
    component -- the electrode voltage is still produced by the volume conductor
    even when it is measured.  an inverse or decoding materialization is the case
    where the flag earns its keep: there the observed component is being
    conditioned on, and everything upstream of it exists only to be inferred.
    either way the boundary is recorded, so the two are never confused.
    """
    reg = registry or REGISTRY
    seeds: list[str] = list(request.target_components)
    for f in request.fields:
        seeds += [c.id for c in reg.of_field(f)]
    for pid in request.processes:
        p = reg.processes.get(pid)
        if p is not None:
            seeds += [v for s in p.outputs for v in s.vars]
    for oid in request.observations:
        o = reg.observations.get(oid)
        if o is not None:
            seeds += list(o.observes.vars)
    for iid in request.interventions:
        iv = reg.interventions.get(iid)
        if iv is not None:
            seeds += list(iv.constrains.vars)

    unknown = [c for c in dict.fromkeys(seeds) if c not in reg.components
               and c not in reg._alias]
    if unknown:
        raise KeyError(
            f"request {request.name!r} targets unregistered components {unknown}.  a typo here "
            "is a declaration error, not a silently empty materialization (ARCHITECTURE.md §8); "
            "check docs/CONTRACT.md for the fixed ids")

    clamped: dict[str, list[str]] = {}
    for iid in request.interventions:
        iv = reg.interventions.get(iid)
        if iv is None:
            raise KeyError(f"request {request.name!r} names unregistered intervention {iid!r}")
        for v in iv.constrains.vars:
            clamped.setdefault(v, []).append(iid)
    observed: dict[str, list[str]] = {}
    for oid in request.observations:
        o = reg.observations.get(oid)
        if o is None:
            raise KeyError(f"request {request.name!r} names unregistered observation {oid!r}")
        for v in o.observes.vars:
            observed.setdefault(v, []).append(oid)

    components: dict[str, None] = {}
    processes: dict[str, None] = {}
    groundings: list[Grounding] = []
    writers: dict[str, list[str]] = {}
    readers: dict[str, list[str]] = {}
    depends: dict[str, set[str]] = {}
    bands: dict[str, Band] = {}
    regions: dict[str, list[Region]] = {}

    frontier: deque[str] = deque()
    for c in dict.fromkeys(seeds):
        cid = reg.resolve(c).id
        if cid not in components:
            components[cid] = None
            frontier.append(cid)

    for s in request.targets:
        for v in s.vars:
            cid = reg.resolve(v).id
            _widen(bands, cid, s.band)
            regions.setdefault(cid, []).append(s.region)

    def drain() -> None:
        """the whole traversal.  a queue, a visited set, and a finite graph."""
        while frontier:
            cid = frontier.popleft()
            comp = reg.components[cid]
            ws = [p for p in reg.processes.values() if any(cid in s.vars for s in p.outputs)]
            readers[cid] = tuple(p.id for p in reg.processes.values()
                                 if any(cid in s.vars for s in p.inputs))

            if cid in clamped:
                groundings.append(Grounding(cid, GROUND_INTERVENTION, tuple(clamped[cid]),
                                            "externally constrained state; its writers are not "
                                            "needed to produce it"))
                writers[cid] = ()
                continue
            if stop_at_observations and cid in observed:
                groundings.append(Grounding(cid, GROUND_OBSERVATION, tuple(observed[cid]),
                                            "conditioned on; the chain that produces it is not "
                                            "materialized under this policy"))
                writers[cid] = ()
                continue
            if cid in observed:
                groundings.append(Grounding(cid, GROUND_OBSERVATION, tuple(observed[cid]),
                                            "evidence enters here, and the process chain that "
                                            "produces it is still materialized"))
            if not ws:
                kind = GROUND_EXOGENOUS if comp.exogenous else GROUND_UNWRITTEN
                note = ("declared exogenous: a material constant, a stimulus or a device drive"
                        if comp.exogenous else
                        "nothing writes it, it is not exogenous, and no intervention clamps it "
                        "-- the graph has a hole here and the registry should have refused to "
                        "seal")
                groundings.append(Grounding(cid, kind, (), note))
                writers[cid] = ()
                continue

            writers[cid] = tuple(p.id for p in ws)
            for p in ws:
                _admit(p, reg, components, processes, frontier, bands, regions, depends, request)

    drain()

    # processes that write another traced process's theta are upstream of it even
    # though they write no component it reads.  neuromodulation and plasticity are
    # the whole reason §4 allows a process to apply pressure to parameters, and a
    # materialization that dropped them would report a gain as constant when the
    # ontology says something is modulating it.  one more pass per admitted
    # process at worst, and the process table is finite.
    param_writers: list[str] = []
    while True:
        fresh = [p for p in reg.processes.values()
                 if p.writes == "parameters" and p.id not in processes
                 and any(t in processes for t in p.targets)]
        if not fresh:
            break
        for p in fresh:
            param_writers.append(p.id)
            _admit(p, reg, components, processes, frontier, bands, regions, depends, request)
        drain()

    # additive hints from the request that the walk never reached
    named = (set(request.processes) | set(request.topologies)
             | set(request.anatomy) | set(request.fields))
    reached = (set(processes) | {reg.processes[p].topology for p in processes
                                 if p in reg.processes}
               | {field_of(c) for c in components})
    topologies = tuple(sorted({reg.processes[p].topology for p in processes
                               if p in reg.processes}))
    supports = tuple(sorted({_support_of(reg, c) for c in components} - {""}))
    fields = tuple(sorted({field_of(c) for c in components}))
    anatomies = tuple(sorted(set(request.anatomy) | _atlas_systems(regions, request)))
    reached |= set(anatomies) | set(topologies)

    notes: list[str] = []
    holes = [g.component for g in groundings if g.kind == GROUND_UNWRITTEN]
    if holes:
        notes.append(f"{len(holes)} component(s) read but never written and not declared "
                     f"exogenous: {', '.join(sorted(holes)[:5])}")
    if not processes:
        notes.append("no process reaches the targets: this materialization would allocate "
                     "state with no dynamics over it")

    return Trace(
        request_name=request.name,
        components=tuple(sorted(components)),
        processes=tuple(sorted(processes)),
        topologies=topologies,
        supports=supports,
        fields=fields,
        anatomies=anatomies,
        observations=tuple(request.observations),
        interventions=tuple(request.interventions),
        groundings=tuple(sorted(groundings, key=lambda g: (g.kind, g.component))),
        writers={k: tuple(v) for k, v in writers.items()},
        readers={k: tuple(v) for k, v in readers.items()},
        depends_on={k: tuple(sorted(v)) for k, v in depends.items()},
        bands={c: request.band_for(c, bands.get(c, reg.components[c].band))
               for c in sorted(components)},
        regions={k: tuple(v) for k, v in regions.items()},
        unreached_hints=tuple(sorted(named - reached)),
        parameter_writers=tuple(sorted(set(param_writers))),
        notes=tuple(notes))


def _admit(p: Process, reg: Any, components: dict, processes: dict, frontier: deque,
           bands: dict, regions: dict, depends: dict,
           request: MaterializationRequest) -> None:
    """admit a process and enqueue every component it touches.

    outputs are enqueued as well as inputs, which is not an oversight.  a process
    writes all of its outputs or none of them, so a materialization that admitted
    a process for one output and left its siblings unallocated would be running an
    f whose return value has nowhere to go.  the consequence is that the traced
    set grows towards the core clique rather than staying a narrow chain -- which
    is exactly what §7 predicts, and is the reason it can say that R, r and B are
    the levers by which materializations actually differ.
    """
    processes.setdefault(p.id, None)
    dep = depends.setdefault(p.id, set())
    for s in p.inputs:
        for v in s.vars:
            if v not in reg.components and v not in reg._alias:
                continue
            cid = reg.resolve(v).id
            _widen(bands, cid, s.band)
            regions.setdefault(cid, []).append(s.region)
            for q in reg.processes.values():
                if any(cid in o.vars for o in q.outputs):
                    dep.add(q.id)
            if cid not in components:
                components[cid] = None
                frontier.append(cid)
    for s in p.outputs:
        for v in s.vars:
            if v in reg.components or v in reg._alias:
                cid = reg.resolve(v).id
                _widen(bands, cid, s.band)
                if cid not in components:
                    components[cid] = None
                    frontier.append(cid)


def _widen(bands: dict[str, Band], cid: str, band: Band) -> None:
    """a component must be materialized over the union of the bands its processes touch.

    union rather than intersection, and it is worth being explicit about why: a
    component read by one process over 0.5-300 Hz and by another over 0-0.5 Hz has
    to carry both, or one of the two reads silently returns an empty spectrum.  the
    request's B(q) then narrows the union, because that is a deliberate budget
    decision rather than an accident of which processes happened to be traced.
    """
    cur = bands.get(cid)
    bands[cid] = band if cur is None else Band(min(cur.lo_hz, band.lo_hz),
                                               max(cur.hi_hz, band.hi_hz))


def _support_of(reg: Any, cid: str) -> str:
    c = reg.components.get(cid)
    if c is None:
        return ""
    if c.support:
        return c.support
    f = reg.fields.get(c.field)
    return f.support if f is not None else ""


def _atlas_systems(regions: Mapping[str, Sequence[Region]],
                   request: MaterializationRequest) -> set[str]:
    """every anatomical partitioning system a traced selector or rule mentions."""
    from ibm.vocabulary import Anat, Difference, Intersect, Union as _U

    out: set[str] = set()

    def walk(r: Region) -> None:
        if isinstance(r, Anat):
            out.add(r.system)
        elif isinstance(r, (_U, Intersect)):
            for p in r.parts:
                walk(p)
        elif isinstance(r, Difference):
            walk(r.left)
            walk(r.right)

    for rs in regions.values():
        for r in rs:
            walk(r)
    for _, r in request.regions:
        walk(r)
    for rule in request.resolution.rules:
        walk(rule.region)
    return out


# ---------------------------------------------------------------------------
# the core clique
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreClique:
    """what several materializations have in common, and what each has alone.

    §7's claim is that R, r and B are the levers by which materializations
    actually differ -- that they are views of one implicit model rather than
    separate brain models.  the check is arithmetic: intersect the traced process
    sets.  a large `shared` fraction is that claim holding; a small one means the
    library has drifted into unrelated sub-models and two of them could disagree
    about the dynamics of the same state variable without anything noticing.
    """

    shared_processes: tuple[str, ...]
    shared_components: tuple[str, ...]
    per_model: Mapping[str, tuple[str, ...]]         # name -> processes it alone needs
    n_models: int = 0

    @property
    def fraction(self) -> float:
        total = len(self.shared_processes) + sum(len(v) for v in self.per_model.values())
        return len(self.shared_processes) / total if total else 0.0

    def describe(self) -> str:
        lines = [f"core clique over {self.n_models} materializations: "
                 f"{len(self.shared_processes)} processes and "
                 f"{len(self.shared_components)} components shared by all "
                 f"({self.fraction:.0%} of the union)"]
        for name, only in sorted(self.per_model.items()):
            lines.append(f"  {name:24s} +{len(only):3d} of its own"
                         + (f": {', '.join(only[:4])}" if only else ""))
        if self.fraction < 0.2 and self.n_models > 1:
            lines.append("  ! the shared fraction is small.  §7 says these should be views of "
                         "one implicit model; at this overlap they are closer to separate models "
                         "that happen to share a registry")
        return "\n".join(lines)


def core_clique(traces: Iterable[Trace]) -> CoreClique:
    """intersect several traces.  the check behind §7's "shared core clique"."""
    ts = list(traces)
    if not ts:
        return CoreClique((), (), {}, 0)
    shared_p = set(ts[0].processes)
    shared_c = set(ts[0].components)
    for t in ts[1:]:
        shared_p &= set(t.processes)
        shared_c &= set(t.components)
    per = {t.request_name: tuple(sorted(set(t.processes) - shared_p)) for t in ts}
    return CoreClique(tuple(sorted(shared_p)), tuple(sorted(shared_c)), per, len(ts))


def trace_all(requests: Iterable[MaterializationRequest], **kw) -> dict[str, Trace]:
    return {r.name: trace(r, **kw) for r in requests}


__all__ = ["Trace", "Grounding", "CoreClique", "trace", "trace_all", "core_clique",
           "GROUND_EXOGENOUS", "GROUND_INTERVENTION", "GROUND_OBSERVATION",
           "GROUND_UNWRITTEN", "GROUND_INTERIOR"]
