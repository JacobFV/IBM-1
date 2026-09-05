"""the registry: one place where every component and every process exists.

nothing exists unless it is registered here.  ids are checked against a grammar
and a controlled vocabulary, near-duplicates are refused rather than warned
about, and `seal()` rejects a graph with dangling inputs, dead state,
unregistered topologies, band violations, or a coupling between two forms of
uncertain state with no declared conversion.

registration is declarative and happens at import.  `ibm.load_all()` imports the
ontology modules and seals.

the whole ontology is printable: `REGISTRY.table("components")`,
`REGISTRY.table("processes")`, `REGISTRY.summary()`.  if that ever stops being
possible, the ontology has begun to sprawl.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field as _field
from enum import Enum
from typing import Any, Callable

from ibm import vocabulary as V
from ibm.vocabulary import Band, FULL, Prior, Provenance, Sel, Tying, Validity


# ---------------------------------------------------------------------------
# declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Support:
    """a domain a field is indexed over.

    fields do not share a support.  the cortical surface, the vascular tree, the
    interstitial volume and a sensor array have different notions of adjacency
    and distance, which is why there is no universal spatial operator in ibm-1
    and why spatial structure is carried by topologies rather than by a spectrum.
    """
    name: str
    doc: str
    kind: str                     # volume | surface | tree | discrete | manifold
    dimension: int
    frame: str
    extent: str = ""              # prose: what part of the organism it covers
    min_spacing_mm: float = 0.0   # finest sampling that is physically meaningful


@dataclass(frozen=True)
class Field:
    name: str
    doc: str
    support: str
    provenance: Provenance = Provenance.LITERATURE


@dataclass(frozen=True)
class Component:
    """one state variable of the ontology: a field component with a type.

    the ontology declares components; materialization instantiates state
    variables, one component at one position.  `uncertainty` names how a belief
    about this component is carried -- scalar where one timescale matters,
    spectral where several interact within the component itself.
    """
    id: str
    field: str
    doc: str
    units: str
    uncertainty: str = "scalar"          # registered uncertainty form
    band: Band = FULL                    # widest band that is meaningful
    prior: str | None = None             # registered prior builder
    bounds: tuple[float, float] = (-math.inf, math.inf)
    timescale_s: float = 1e-3
    provenance: Provenance = Provenance.LITERATURE
    aliases: tuple[str, ...] = ()
    tags: frozenset[str] = frozenset()
    support: str | None = None           # overrides the field's support
    #: additional supports this component may be indexed on.  a support is a
    #: *sampling* of a domain, not a different place: cortical population activity
    #: is the same quantity whether you index it by volume position or by a column
    #: node seeded on the folded sheet, and which one a materialization wants is
    #: decided by the processes it needs, not by the component.  at 2 mm a volume
    #: graph mis-connects 0.1% of cortical pairs across a sulcus; at 10 mm it
    #: mis-connects 58%, so the sheet indexing is not a refinement of the volume
    #: one -- it is the only indexing on which lateral propagation is expressible.
    #: a materialization must place each position on exactly ONE of these; see
    #: `supports_of` and the overlap check in ibm.materialize.build.
    alt_supports: tuple[str, ...] = ()
    #: a component read but never written must be exogenous -- a stimulus, a
    #: device drive, a material constant -- or the graph has a hole in it.
    exogenous: bool = False


@dataclass(frozen=True)
class Anatomy:
    """a partitioning system: a soft map from position to named partitions."""
    name: str
    doc: str
    labels: tuple[str, ...]
    frame: str
    crisp: bool = False
    hierarchical: bool = False
    covers: str = "brain"
    source: str = ""
    provenance: Provenance = Provenance.ATLAS


@dataclass(frozen=True)
class Topology:
    """which materialized state variables may interact through a class of process.

    a support for interaction, not a weighting of it.  geometry it carries
    (distance, tract length, orientation, contact area) is descriptive; how
    strongly a pair actually interacts is the process's business and may be
    static, fitted, or computed from state.
    """
    name: str
    doc: str
    on: tuple[str, ...]                  # supports it is defined over
    edge_features: tuple[str, ...] = ()
    directed: bool = False
    builder: str | None = None
    provenance: Provenance = Provenance.LITERATURE


class Form(str, Enum):
    """the shape of a process's dynamics.

    LTI is the one worth separating: linear time-invariant couplings are diagonal
    in the temporal laplacian basis, so they are exact at any timestep, carry
    delays as phase ramps, and never enter the time domain.  most of the
    inventory's transport, propagation and observation processes are LTI, and
    keeping that boundary explicit in the ontology is the main lever on cost.
    """
    LTI = "lti"
    RATE = "rate"                # nonlinear rate law, evaluated in time
    TABLE = "table"              # atlas or lookup
    LEARNED = "learned"          # fitted module
    CONSTRAINT = "constraint"    # algebraic; the stiff limit of pressure


@dataclass(frozen=True)
class Implementation:
    """a candidate form of f for one process.

    not a separate primitive.  a different f is a different p(theta) over a
    differently shaped theta, and (I, O, T) is untouched -- which is what lets a
    process begin analytic and become learned without any declaration changing.
    """
    name: str
    process: str
    doc: str
    form: Form
    params: dict[str, Prior] = _field(default_factory=dict)
    tying: Tying = Tying.GLOBAL
    fn: Callable[..., Any] | None = None
    transfer: Callable[..., Any] | None = None   # H(omega), for Form.LTI
    state_dependent_weights: bool = False        # attention-like
    differentiable: bool = True
    provenance: Provenance = Provenance.WEAK
    source: str = ""


@dataclass(frozen=True)
class Process:
    """P = (I, O, T, f, theta).  this declares (I, O, T); f lives in implementations."""
    id: str
    doc: str
    inputs: tuple[Sel, ...]
    outputs: tuple[Sel, ...]
    topology: str
    #: "state" applies pressure to the outputs; "parameters" applies pressure to
    #: another process's theta.  neuromodulation and plasticity write parameters;
    #: everything else writes state.  no additional mechanism is needed, since
    #: theta is already part of the schema.
    writes: str = "state"
    targets: tuple[str, ...] = ()        # process ids, when writes == "parameters"
    timescale_s: float = 1e-3
    validity: Validity = _field(default_factory=Validity)
    provenance: Provenance = Provenance.WEAK
    tags: frozenset[str] = frozenset()
    notes: str = ""


@dataclass(frozen=True)
class Observation:
    """external evidence about a state variable.  not a primitive (§6).

    the coupling that produces the observed variable is an ordinary process; this
    only carries the likelihood attached to it, and the band over which that
    likelihood has any precision at all.
    """
    id: str
    doc: str
    observes: Sel
    modality: str
    noise: str = "gaussian"
    band: Band = FULL
    frame: str = ""
    via: tuple[str, ...] = ()            # process chain producing the observed state


@dataclass(frozen=True)
class Intervention:
    """externally constrained state or process input.  not a primitive (§6)."""
    id: str
    doc: str
    constrains: Sel
    modality: str
    waveform: str = "arbitrary"
    frame: str = ""


@dataclass
class Problem:
    severity: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity.upper():7s} {self.where}: {self.message}"


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


class Registry:
    def __init__(self) -> None:
        self.supports: dict[str, Support] = {}
        self.fields: dict[str, Field] = {}
        self.components: dict[str, Component] = {}
        self.anatomies: dict[str, Anatomy] = {}
        self.topologies: dict[str, Topology] = {}
        self.processes: dict[str, Process] = {}
        self.implementations: dict[str, Implementation] = {}
        self.observations: dict[str, Observation] = {}
        self.interventions: dict[str, Intervention] = {}
        self.uncertainty: dict[str, Any] = {}          # name -> UncertaintyForm
        self.conversions: dict[tuple[str, str], Any] = {}
        self._alias: dict[str, str] = {}
        self._sealed = False

    # -- registration ----------------------------------------------------

    def _guard(self, kind: str, key: str, table: dict) -> None:
        if self._sealed:
            raise RuntimeError(f"registry sealed; cannot register {kind} {key!r}")
        if key in table:
            raise ValueError(f"{kind} {key!r} already registered")

    def support(self, d: Support) -> Support:
        self._guard("support", d.name, self.supports); self.supports[d.name] = d; return d

    def field(self, d: Field) -> Field:
        self._guard("field", d.name, self.fields); V.validate_id(d.name)
        self.fields[d.name] = d; return d

    def component(self, d: Component) -> Component:
        self._guard("component", d.id, self.components)
        V.validate_id(d.id)
        if V.field_of(d.id) != d.field:
            raise ValueError(f"component {d.id!r} declares field {d.field!r} but its id says "
                             f"{V.field_of(d.id)!r}; the id must begin with its field")
        nf = V.normal_form(d.id)
        for e in self.components.values():
            if V.normal_form(e.id) == nf and e.id not in d.aliases and d.id not in e.aliases:
                raise ValueError(
                    f"component {d.id!r} collides with {e.id!r}: both normalize to "
                    f"{'+'.join(nf)}.  either they are the same concept -- register one and "
                    "reference it -- or ibm.vocabulary.SYNONYMS needs a distinction it lacks.")
        for a in d.aliases:
            self._alias[a] = d.id
        self.components[d.id] = d; return d

    def anatomy(self, d: Anatomy) -> Anatomy:
        self._guard("anatomy", d.name, self.anatomies); self.anatomies[d.name] = d; return d

    def topology(self, d: Topology) -> Topology:
        self._guard("topology", d.name, self.topologies); self.topologies[d.name] = d; return d

    def process(self, d: Process) -> Process:
        self._guard("process", d.id, self.processes); V.validate_id(d.id)
        self.processes[d.id] = d; return d

    def implementation(self, d: Implementation) -> Implementation:
        key = f"{d.process}:{d.name}"
        self._guard("implementation", key, self.implementations)
        self.implementations[key] = d; return d

    def observation(self, d: Observation) -> Observation:
        self._guard("observation", d.id, self.observations); self.observations[d.id] = d; return d

    def intervention(self, d: Intervention) -> Intervention:
        self._guard("intervention", d.id, self.interventions); self.interventions[d.id] = d; return d

    def uncertainty_form(self, d) -> Any:
        self._guard("uncertainty form", d.name, self.uncertainty); self.uncertainty[d.name] = d; return d

    def conversion(self, d) -> Any:
        self.conversions[(d.src, d.dst)] = d; return d

    # -- resolution ------------------------------------------------------

    def resolve(self, id_: str) -> Component:
        if id_ in self.components: return self.components[id_]
        if id_ in self._alias: return self.components[self._alias[id_]]
        raise KeyError(f"no registered component {id_!r}")

    def supports_of(self, cid: str) -> tuple[str, ...]:
        """every support this component may be indexed on, primary first.

        a topology can only relate sites on its own support, so a component that
        is never admissible on `cortical_surface` makes every surface topology
        unbuildable no matter how good the mesh is.  that failure is invisible to
        `seal()` -- nothing is inconsistent, there is simply nothing to relate --
        and it only appears when a materialization asks for sites.
        """
        c = self.resolve(cid)
        primary = c.support or (self.fields[c.field].support if c.field in self.fields else "")
        out = [primary] if primary else []
        out += [s for s in c.alt_supports if s != primary]
        return tuple(out)

    def components_on(self, support: str) -> list[Component]:
        """components a materialization may index on this support."""
        return [c for c in self.components.values() if support in self.supports_of(c.id)]

    def of_field(self, field: str) -> list[Component]:
        return [c for c in self.components.values() if c.field == field]

    def impls_of(self, process: str) -> list[Implementation]:
        return [i for i in self.implementations.values() if i.process == process]

    def writers(self, cid: str) -> list[Process]:
        return [p for p in self.processes.values() if any(cid in s.vars for s in p.outputs)]

    def readers(self, cid: str) -> list[Process]:
        return [p for p in self.processes.values() if any(cid in s.vars for s in p.inputs)]

    # -- validation ------------------------------------------------------

    def check(self) -> list[Problem]:
        p: list[Problem] = []
        add = lambda s, w, m: p.append(Problem(s, w, m))

        for f in self.fields.values():
            if f.support not in self.supports:
                add("error", f.name, f"support {f.support!r} is not registered")
            if not self.of_field(f.name):
                add("warning", f.name, "field declares no components")

        for c in self.components.values():
            if c.field not in self.fields:
                add("error", c.id, f"field {c.field!r} is not registered")
            if c.uncertainty not in self.uncertainty:
                add("error", c.id, f"uncertainty form {c.uncertainty!r} is not registered")
            if c.support and c.support not in self.supports:
                add("error", c.id, f"support {c.support!r} is not registered")
            for alt in c.alt_supports:
                if alt not in self.supports:
                    add("error", c.id, f"alternative support {alt!r} is not registered")
                if alt == (c.support or self.fields[c.field].support if c.field in self.fields else None):
                    add("warning", c.id, f"lists {alt!r} as an alternative support and as its primary")

        for t in self.topologies.values():
            for s in t.on:
                if s not in self.supports:
                    add("error", t.name, f"defined over unregistered support {s!r}")

        for a in self.anatomies.values():
            if not a.labels:
                add("warning", a.name, "partitioning system declares no labels")

        for proc in self.processes.values():
            if proc.topology not in self.topologies:
                add("error", proc.id, f"topology {proc.topology!r} is not registered")
            if not self.impls_of(proc.id):
                add("warning", proc.id, "no implementation; exists in the ontology but cannot run")
            if not proc.outputs:
                add("error", proc.id, "writes nothing")
            for s in proc.inputs + proc.outputs:
                for cid in s.vars:
                    if cid not in self.components and cid not in self._alias:
                        add("error", proc.id, f"selector names unregistered component {cid!r}")
            if proc.writes == "parameters":
                for t in proc.targets:
                    if t not in self.processes:
                        add("error", proc.id, f"writes parameters of unregistered process {t!r}")
                if not proc.targets:
                    add("error", proc.id, "writes parameters but names no target process")
            for s in proc.outputs:
                for cid in s.vars:
                    c = self.components.get(cid)
                    if not c: continue
                    if s.band.hi_hz > c.band.hi_hz:
                        add("warning", proc.id, f"writes {cid!r} up to {s.band.hi_hz} Hz but the "
                            f"component is meaningful only to {c.band.hi_hz} Hz")
            ins = {self.components[v].uncertainty for s in proc.inputs for v in s.vars
                   if v in self.components}
            outs = {self.components[v].uncertainty for s in proc.outputs for v in s.vars
                    if v in self.components}
            for a in ins:
                for b in outs:
                    if a != b and (a, b) not in self.conversions:
                        add("error", proc.id, f"couples {a!r} inputs to {b!r} outputs with no "
                            f"registered conversion {a} -> {b}")

        for i in self.implementations.values():
            if i.process not in self.processes:
                add("error", i.name, f"implements unregistered process {i.process!r}")
            if i.form is Form.LTI and i.transfer is None and i.fn is None:
                add("warning", f"{i.process}:{i.name}", "declared LTI but supplies no transfer function")

        for o in self.observations.values():
            for cid in o.observes.vars:
                if cid not in self.components:
                    add("error", o.id, f"observes unregistered component {cid!r}")
            for v in o.via:
                if v not in self.processes:
                    add("error", o.id, f"names unregistered process {v!r} in its chain")
        for iv in self.interventions.values():
            for cid in iv.constrains.vars:
                if cid not in self.components:
                    add("error", iv.id, f"constrains unregistered component {cid!r}")

        read, written = set(), set()
        for proc in self.processes.values():
            read.update(v for s in proc.inputs for v in s.vars)
            written.update(v for s in proc.outputs for v in s.vars)
        clamped = {v for iv in self.interventions.values() for v in iv.constrains.vars}
        for cid, c in self.components.items():
            if cid not in read and cid not in written:
                add("warning", cid, "dead: no process reads or writes it")
            elif cid not in written and not c.exogenous and cid not in clamped:
                add("error", cid, "read but never written, not exogenous, and not clamped by any "
                    "intervention -- the graph has a hole here")
            elif cid not in read and cid not in {v for o in self.observations.values()
                                                 for v in o.observes.vars}:
                add("warning", cid, "written but never read or observed")

        for col in V.find_collisions(sorted(self.components)):
            add("warning", "ids", str(col))
        return p

    def seal(self, strict: bool = True) -> "Registry":
        probs = self.check()
        errs = [x for x in probs if x.severity == "error"]
        if errs and strict:
            raise ValueError("ontology does not validate:\n" + "\n".join(str(e) for e in errs))
        self._sealed = True
        return self

    # -- presentation ----------------------------------------------------

    def table(self, kind: str = "components") -> str:
        if kind == "components":
            head = ("id", "unc", "units", "band Hz", "tau s", "prov", "w", "r")
            rows = [(c.id, c.uncertainty, c.units or "-",
                     f"{c.band.lo_hz:g}-{'inf' if math.isinf(c.band.hi_hz) else f'{c.band.hi_hz:g}'}",
                     f"{c.timescale_s:g}", c.provenance.value,
                     str(len(self.writers(c.id))), str(len(self.readers(c.id))))
                    for c in sorted(self.components.values(), key=lambda x: x.id)]
        elif kind == "processes":
            head = ("id", "topology", "writes", "in", "out", "tau s", "impls", "prov")
            rows = [(p.id, p.topology, p.writes,
                     str(sum(len(s.vars) for s in p.inputs)),
                     str(sum(len(s.vars) for s in p.outputs)),
                     f"{p.timescale_s:g}", str(len(self.impls_of(p.id))), p.provenance.value)
                    for p in sorted(self.processes.values(), key=lambda x: x.id)]
        elif kind == "topologies":
            head = ("name", "on", "directed", "features")
            rows = [(t.name, ",".join(t.on), str(t.directed), ",".join(t.edge_features))
                    for t in sorted(self.topologies.values(), key=lambda x: x.name)]
        elif kind == "observations":
            head = ("id", "modality", "observes", "band Hz")
            rows = [(o.id, o.modality, ",".join(o.observes.vars)[:60],
                     f"{o.band.lo_hz:g}-{'inf' if math.isinf(o.band.hi_hz) else f'{o.band.hi_hz:g}'}")
                    for o in sorted(self.observations.values(), key=lambda x: x.id)]
        else:
            raise ValueError(f"unknown table {kind!r}")
        w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(head)]
        line = "  ".join(h.ljust(x) for h, x in zip(head, w))
        rule = "  ".join("-" * x for x in w)
        body = "\n".join("  ".join(c.ljust(x) for c, x in zip(r, w)) for r in rows)
        return f"{line}\n{rule}\n{body}"

    def summary(self) -> str:
        byf: dict[str, int] = defaultdict(int)
        for c in self.components.values():
            byf[c.field] += 1
        out = [f"{len(self.fields)} fields · {len(self.components)} components · "
               f"{len(self.supports)} supports · {len(self.anatomies)} anatomical systems · "
               f"{len(self.topologies)} topologies · {len(self.processes)} processes · "
               f"{len(self.implementations)} implementations · "
               f"{len(self.observations)} observations · {len(self.interventions)} interventions"]
        out += [f"  {f:26s} {n:3d}" for f, n in sorted(byf.items())]
        return "\n".join(out)


#: the process graph and state graph of ibm-1.  there is deliberately only one.
REGISTRY = Registry()
