"""what a materialized number is actually resting on.

ARCHITECTURE.md §7 asks for four things by name -- which f was selected per
process, which parameters were moved off their prior by evidence and which were
not, where a process was run outside the regime its f is meaningful in, and, from
§8, every lossy uncertainty conversion and what it destroyed -- and then states
the reason: *a prediction resting on prior-dominated structure must not be
presented with the confidence of one resting on constrained structure.*

that sentence is the whole design brief for this module.  it is not a logging
requirement.  a materialization of this ontology will routinely produce a
plausible-looking trajectory for a component whose entire upstream chain is
`weak()` priors and a speculative transfer function, and there is nothing in the
numbers themselves that distinguishes it from one grounded in a measured lead
field.  the difference is structural and is knowable at build time, so it is
recorded at build time and answered on demand by `Provenance.rests_on`.

the honesty this module can offer has a hard edge, and it is worth stating.
before fitting, "moved off its prior" is not observable; what *is* observable is
the declared provenance of the prior itself and whether any source card binds to
it.  so `ParameterFate` distinguishes three states -- moved by evidence,
informative prior not yet moved, and weak or speculative prior -- and never
reports the second as the first.  `ibm.forge` supplies the real `moved` set once
a posterior exists, and until it does the record says so.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from ibm.registry import REGISTRY, Form, Implementation
from ibm.vocabulary import Band, Prior, Provenance as Prov, Tying, Validity


#: priors that already encode a measurement or a conservation law, even though no
#: fitting in this codebase has moved them.
INFORMATIVE = frozenset({Prov.PHYSICS, Prov.LITERATURE, Prov.ATLAS})
#: priors that are placeholders for dynamics the science has not pinned.  a
#: posterior over these stays where it started and the structure comes out smooth.
UNINFORMED = frozenset({Prov.WEAK, Prov.SPECULATIVE})
#: priors that a corpus has actually moved.
MOVED = frozenset({Prov.FIT, Prov.DISTILLED})


# ---------------------------------------------------------------------------
# the records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """which f was chosen for one process, and what was passed over.

    `rejected` is kept because the choice is only informative next to its
    alternatives.  a process with one implementation was not "selected" in any
    meaningful sense, and a report that presents those two cases identically is
    hiding the fact that half the model had no choice to make.
    """

    process: str
    implementation: str
    form: Form
    provenance: Prov
    score: float = 0.0
    why: str = ""
    rejected: tuple[tuple[str, str], ...] = ()      # (implementation, reason)
    policy: str = ""

    @property
    def was_a_choice(self) -> bool:
        return bool(self.rejected)


@dataclass(frozen=True)
class ParameterFate:
    """one entry of theta and what, if anything, has constrained it.

    `n_effective` is the count `Tying` implies -- one number globally, one per
    partition, one per site -- and it is the number that decides whether any
    corpus could have moved this parameter.  a per-site parameterization over 10^5
    positions is a legitimate declaration (§4) even when nothing can distinguish
    its entries; recording the count is how the report can say that out loud
    instead of presenting 10^5 smooth numbers as if they were estimates.
    """

    process: str
    implementation: str
    name: str
    prior: Prior
    tying: Tying = Tying.GLOBAL
    n_effective: int = 1
    moved: bool = False
    evidence: tuple[str, ...] = ()                  # source card ids that moved it
    note: str = ""

    @property
    def status(self) -> str:
        if self.moved or self.prior.provenance in MOVED:
            return "moved"
        if self.prior.provenance in INFORMATIVE:
            return "informative prior"
        return "prior only"

    def __str__(self) -> str:
        return (f"{self.process}.{self.name:22s} {self.status:18s} "
                f"{self.prior.provenance.value:12s} {self.tying.value} x{self.n_effective}")


@dataclass(frozen=True)
class ConversionRecord:
    """a place where a belief changed form, and what that cost.

    every lossy conversion declares what it destroys (§8) and this is where the
    declaration is cashed in.  spectral -> scalar at a neurovascular coupling is
    the canonical case: it is the correct conversion, blood genuinely has no
    structure above half a hertz, and it is *also* the exact point at which the
    model stops being able to answer why two activity patterns of equal mean power
    produce different haemodynamics.  both facts belong in the record.
    """

    process: str
    component_in: str
    component_out: str
    src: str
    dst: str
    lossy: bool
    destroys: str = ""

    def __str__(self) -> str:
        tag = "lossy" if self.lossy else "exact"
        return (f"{self.process}: {self.component_in} ({self.src}) -> "
                f"{self.component_out} ({self.dst}) [{tag}]"
                + (f" -- destroys {self.destroys}" if self.lossy and self.destroys else ""))


@dataclass(frozen=True)
class ValidityBreach:
    """a process materialized outside the regime its f was written for.

    not an error, and deliberately not one.  §7 materializes every process
    reachable from a target, so resolution and bandwidth are the only levers and a
    process *will* be asked to run at a spacing its form was never validated at.
    refusing would make whole materializations impossible; complying silently
    would make them untrustworthy.  recording is the only remaining option.
    """

    process: str
    implementation: str
    support: str
    spacing_mm: float
    band: Band
    validity: Validity
    messages: tuple[str, ...] = ()
    n_sites: int = 0

    def __str__(self) -> str:
        return (f"{self.process} ({self.implementation}) on {self.support} at "
                f"{self.spacing_mm:g} mm over {self.band}: " + "; ".join(self.messages))


@dataclass(frozen=True)
class FrameRecord:
    """a warp chain inserted between two frames, and the error it carries.

    `residual_mm` is in quadrature along the chain and is a *systematic*,
    spatially correlated displacement rather than noise (see `ibm.frames.Warp`),
    which is why it is recorded per chain and not folded into a per-site variance.
    the eeg cap -> subject chain at 5 mm is not a 5 mm jitter; it is the whole
    montage sitting 5 mm off in one direction.
    """

    what: str                                       # support or device name
    src: str
    dst: str
    methods: tuple[str, ...] = ()
    residual_mm: float | None = None
    subject_specific: bool = True
    note: str = ""

    def __str__(self) -> str:
        r = "unknown" if self.residual_mm is None else f"{self.residual_mm:.1f} mm"
        chain = " -> ".join(self.methods) or "identity"
        return f"{self.what}: {self.src} -> {self.dst} via {chain}, residual {r}"


@dataclass(frozen=True)
class TierRecord:
    """which rung of a prior ladder one piece of structure actually came from.

    §7's sentence -- a prediction resting on prior-dominated structure must not be
    presented with the confidence of one resting on constrained structure -- is
    usually about *parameters*, and `ParameterFate` answers it for those.  it is
    equally about *structure*, and nothing answered it there.  a tractometric edge
    set built from this subject's diffusion imaging and one expanded from a
    published group matrix have the same type, the same features and the same name
    downstream; the only thing that ever distinguished them was which file the
    build happened to open, and that was not written down.

    this is that written down.  the three prior modules each declare their own
    three-rung `Tier`, and they are deliberately not one enum: "this subject's
    tractogram" and "this subject's angiogram" are different claims and their
    middle rungs cost measurably different amounts.  what a report needs is not a
    shared enum but a shared *shape* -- which ladder, which rung, is it this
    subject's, and what does standing here cost -- so that is what this carries.

    `cost` must never be empty.  a tier without its measured penalty is a label,
    and a label is exactly what lets a group connectome be quoted with a subject
    scan's authority.
    """

    what: str                       # the topology, support or system this describes
    ladder: str                     # "tract_prior.Tier", "vascular_prior.Tier", ...
    tier: str                       # the enum value actually reached
    rank: int                       # 0 = this subject's own data, larger = weaker
    n_rungs: int = 3
    source: str = ""                # where the bytes came from
    cost: str = ""                  # what standing on this rung measurably costs
    note: str = ""

    @property
    def subject_specific(self) -> bool:
        return self.rank == 0

    def __str__(self) -> str:
        who = "this subject" if self.subject_specific else "a population"
        return (f"{self.what}: tier {self.tier} (rung {self.rank + 1} of {self.n_rungs}, "
                f"{who})" + (f" -- {self.cost}" if self.cost else ""))


@dataclass(frozen=True)
class GeometryRecord:
    """where one support's positions came from.

    the template flag is load-bearing.  a template standing in for a subject
    displaces every position in a head-size-dependent, spatially correlated way,
    so a materialization built on one has a systematic error in every derived
    quantity and `rests_on` says so for every component on that support.
    """

    support: str
    source: str
    n_sites: int = 0
    is_template: bool = False
    frame: str = ""


# ---------------------------------------------------------------------------
# the answer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Basis:
    """the answer to "is this resting on constrained structure or on a prior?".

    a verdict, the arithmetic behind it, and -- more useful than either -- the
    named weakest links.  a fraction is a summary; "the whole chain runs through
    `neurovascular_coupling`, whose efficacy is `speculative()`" is something a
    reader can act on.
    """

    component: str
    verdict: str                                    # constrained | mixed | prior
    fraction_moved: float = 0.0
    fraction_informative: float = 0.0
    n_parameters: int = 0
    processes: tuple[str, ...] = ()
    prior_dominated: tuple[str, ...] = ()
    speculative_forms: tuple[str, ...] = ()
    lossy_conversions: tuple[ConversionRecord, ...] = ()
    validity_breaches: tuple[ValidityBreach, ...] = ()
    grounded_at: tuple[str, ...] = ()
    #: the rungs the STRUCTURE upstream of this component is standing on -- the
    #: topologies, supports and partitioning systems that fell through to a
    #: population prior because this subject had no measurement of them.
    tiers: tuple[TierRecord, ...] = ()
    caveats: tuple[str, ...] = ()

    @property
    def rests_on_population(self) -> tuple[TierRecord, ...]:
        """the rungs upstream of here that are somebody else's anatomy.

        the question §7 actually wants asked of a materialization, and the reason
        `tiers` is on this object rather than only on the provenance record: a
        component whose parameters are all constrained by evidence is not
        constrained if the graph those parameters live on came out of a group
        average.  a fit over a population connectome moves the coupling on edges
        this subject may not have.
        """
        return tuple(t for t in self.tiers if not t.subject_specific)

    @property
    def trustworthy(self) -> bool:
        """deliberately strict.  a single speculative form anywhere upstream, a hole
        in the graph, or structure that is somebody else's, disqualifies the whole
        chain -- because each of them does."""
        return (self.verdict == "constrained" and not self.speculative_forms
                and not self.validity_breaches and not self.rests_on_population)

    def describe(self) -> str:
        lines = [f"{self.component}: {self.verdict.upper()}",
                 f"  {self.n_parameters} parameters over {len(self.processes)} upstream "
                 f"processes; {self.fraction_moved:.0%} moved by evidence, "
                 f"{self.fraction_informative:.0%} at an informative prior"]
        if self.prior_dominated:
            lines.append("  prior-dominated upstream: " + ", ".join(self.prior_dominated[:8]))
        if self.speculative_forms:
            lines.append("  the functional form itself is a guess in: "
                         + ", ".join(self.speculative_forms))
        for c in self.lossy_conversions:
            lines.append(f"  lossy upstream -- {c}")
        for v in self.validity_breaches:
            lines.append(f"  out of regime -- {v}")
        if self.grounded_at:
            lines.append("  bottoms out at: " + ", ".join(self.grounded_at[:8]))
        for t in self.tiers:
            lines.append(f"  {'structure' if t.subject_specific else 'POPULATION STRUCTURE'} -- {t}")
        for c in self.caveats:
            lines.append(f"  ! {c}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """everything a materialized model knows about its own trustworthiness.

    built once, by `ibm.materialize.build`, and then read rather than appended to
    -- which is why it is frozen.  a provenance record that could be edited after
    the fact is a record of what someone last claimed rather than of what was
    actually materialized, and the whole value here is that the two cannot drift
    apart.
    """

    request: str
    subject: str = "template"
    policy: str = ""
    selections: tuple[Selection, ...] = ()
    parameters: tuple[ParameterFate, ...] = ()
    conversions: tuple[ConversionRecord, ...] = ()
    breaches: tuple[ValidityBreach, ...] = ()
    frames: tuple[FrameRecord, ...] = ()
    geometry: tuple[GeometryRecord, ...] = ()
    #: every piece of structure that fell through to a prior tier, keyed by the
    #: topology, support or partitioning system it stands in for.  a build with an
    #: empty tuple here used this subject's own anatomy for everything it built.
    tiers: tuple[TierRecord, ...] = ()
    #: processes reached by the trace with no implementation at all.  §5 permits a
    #: process to exist in the ontology without a high-confidence f; a
    #: materialization that reaches one has a gap, not a bug.
    unimplemented: tuple[str, ...] = ()
    #: topologies or geometry the build could not obtain, kept rather than raised
    #: when the caller asked to proceed anyway
    missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    #: the trace this was built from; `rests_on` walks it
    trace: Any = None

    # -- lookup ----------------------------------------------------------

    def selection(self, process: str) -> Selection | None:
        for s in self.selections:
            if s.process == process:
                return s
        return None

    def params_of(self, process: str) -> tuple[ParameterFate, ...]:
        return tuple(p for p in self.parameters if p.process == process)

    def breaches_of(self, process: str) -> tuple[ValidityBreach, ...]:
        return tuple(b for b in self.breaches if b.process == process)

    def tier_of(self, what: str) -> TierRecord | None:
        """the rung one topology, support or system ended up on, or `None`.

        `None` means "this subject's own", not "unknown".  a fall-through is
        recorded when it happens and nothing else writes here, so the absence of a
        record is the positive statement that no substitution was made.
        """
        for t in self.tiers:
            if t.what == what:
                return t
        return None

    @property
    def rests_on_population(self) -> tuple[TierRecord, ...]:
        """every piece of structure in this model that is somebody else's anatomy."""
        return tuple(t for t in self.tiers if not t.subject_specific)

    @property
    def uses_template_geometry(self) -> bool:
        return any(g.is_template for g in self.geometry)

    def with_evidence(self, moved: Mapping[str, Iterable[str]]) -> "Provenance":
        """fold in what a fit actually moved, keyed by ``"process.param"``.

        `ibm.forge` calls this once a posterior exists.  until then every fate
        reports its declared prior provenance and nothing pretends to be an
        estimate, which is the difference between an honest record and a
        flattering one.
        """
        out = []
        for p in self.parameters:
            got = moved.get(f"{p.process}.{p.name}")
            out.append(replace(p, moved=True, evidence=tuple(got)) if got else p)
        return replace(self, parameters=tuple(out))

    # -- the §7 question -------------------------------------------------

    def rests_on(self, component: str, *, trace: Any = None) -> Basis:
        """what this component's prediction is actually standing on.

        walks backwards from the component over the traced graph and aggregates
        every parameter, every lossy conversion and every out-of-regime process it
        can reach.  the walk is backward and not local on purpose: a component
        whose own process is beautifully constrained is not constrained if its
        only input comes through a speculative one, and the local view would
        report the first and miss the second.
        """
        t = trace or self.trace
        if t is None:
            raise ValueError(
                "this provenance record carries no trace, so it cannot walk backwards from "
                f"{component!r}.  pass trace=... or build the model through "
                "ibm.materialize.build, which attaches one")
        procs = tuple(t.upstream(component))
        fates = [p for p in self.parameters if p.process in procs]
        n = len(fates)
        n_moved = sum(1 for f in fates if f.status == "moved")
        n_info = sum(1 for f in fates if f.status == "informative prior")
        f_moved = n_moved / n if n else 0.0
        f_info = (n_moved + n_info) / n if n else 0.0

        prior_dominated = tuple(sorted(
            {f.process for f in fates if f.status == "prior only"}))
        speculative = tuple(sorted(
            {s.process for s in self.selections
             if s.process in procs and s.provenance is Prov.SPECULATIVE}
            | {f.process for f in fates if f.prior.provenance is Prov.SPECULATIVE}))
        lossy = tuple(c for c in self.conversions if c.lossy and c.process in procs)
        breaches = tuple(b for b in self.breaches if b.process in procs)

        if n == 0:
            verdict = "prior"
        elif f_moved >= 0.5:
            verdict = "constrained"
        elif f_moved == 0.0 and (n - n_info) / n > 0.5:
            verdict = "prior"
        else:
            verdict = "mixed"

        caveats: list[str] = []
        if self.uses_template_geometry:
            caveats.append("built on template geometry: every position carries a systematic, "
                           "head-size-dependent displacement, not a random one")
        big = [r for r in self.frames if (r.residual_mm or 0.0) >= 3.0]
        for r in big:
            caveats.append(f"registration residual {r.residual_mm:.1f} mm on {r.what} "
                           f"({r.src} -> {r.dst}) is systematic and spatially correlated")
        unimpl = [p for p in self.unimplemented if p in procs]
        if unimpl:
            caveats.append("upstream processes with no implementation at all: "
                           + ", ".join(sorted(unimpl)))
        # what the upstream chain bottoms out at: the components those processes
        # read that the trace declared exogenous, clamped or observed.  this is
        # the difference between "grounded in a material constant" and "grounded
        # in nothing", and it is the first thing a sceptical reader should see.
        upstream_components = {
            v for p in procs
            for s in (REGISTRY.processes[p].inputs if p in REGISTRY.processes else ())
            for v in s.vars}
        grounded = tuple(sorted(
            f"{g.component} <- {g.kind}" for g in getattr(t, "groundings", ())
            if g.component in upstream_components and g.kind != "interior"))

        # the structural half of the same question.  a topology one of these
        # processes is declared over, a support one of their variables is indexed
        # on, or a partitioning system the whole materialization was carved by --
        # each of those can have fallen through to a population, and none of them
        # shows up in a parameter fate.  systems are attributed to every component
        # rather than traced, because R is one expression evaluated once: a
        # materialization carved by a template parcellation is carved that way
        # everywhere, not only where the atlas is read.
        topos = {REGISTRY.processes[p].topology for p in procs if p in REGISTRY.processes}
        supports = {s for c in upstream_components | {component}
                    for s in REGISTRY.supports_of(c)}
        tiers = tuple(t for t in self.tiers
                      if t.what in topos or t.what in supports
                      or t.what in REGISTRY.anatomies
                      or t.what.split(".", 1)[0] in topos | supports)
        for t in tiers:
            if not t.subject_specific:
                caveats.append(f"{t.what} is not this subject's: {t}")

        return Basis(component, verdict, f_moved, f_info, n, procs, prior_dominated,
                     speculative, lossy, breaches, grounded, tiers, tuple(caveats))

    # -- presentation ----------------------------------------------------

    def audit(self, components: Sequence[str] = ()) -> str:
        """the report §7 asks for, as one page.

        ordered worst-first, because a provenance report that opens with what went
        right is a marketing document.
        """
        lines = [f"provenance for {self.request!r}  subject={self.subject}  "
                 f"policy={self.policy or 'default'}"]

        spec = [s for s in self.selections if s.provenance is Prov.SPECULATIVE]
        weak = [s for s in self.selections if s.provenance is Prov.WEAK]
        lines.append(f"  f selected for {len(self.selections)} processes "
                     f"({sum(1 for s in self.selections if s.was_a_choice)} had alternatives); "
                     f"{len(spec)} speculative, {len(weak)} weak")
        if self.unimplemented:
            lines.append("  no implementation at all: " + ", ".join(sorted(self.unimplemented)))

        n = len(self.parameters)
        moved = sum(1 for p in self.parameters if p.status == "moved")
        info = sum(1 for p in self.parameters if p.status == "informative prior")
        eff = sum(p.n_effective for p in self.parameters)
        lines.append(f"  theta: {n} declared entries ({eff:,} effective numbers under their "
                     f"tying); {moved} moved by evidence, {info} at an informative prior, "
                     f"{n - moved - info} at a weak or speculative prior")

        lossy = [c for c in self.conversions if c.lossy]
        if lossy:
            lines.append(f"  {len(lossy)} lossy uncertainty conversion(s):")
            lines += [f"    {c}" for c in lossy]
        if self.breaches:
            lines.append(f"  {len(self.breaches)} process(es) outside their validity regime:")
            lines += [f"    {b}" for b in self.breaches]
        if self.frames:
            lines.append("  frames:")
            lines += [f"    {r}" for r in self.frames]
        if self.tiers:
            pop = self.rests_on_population
            lines.append(f"  {len(self.tiers)} piece(s) of structure came from a prior tier, "
                         f"{len(pop)} of them a population's rather than this subject's:")
            lines += [f"    {t}" for t in self.tiers]
        if self.geometry:
            lines.append("  geometry:")
            lines += [f"    {g.support}: {g.n_sites:,} sites from {g.source or 'unnamed source'}"
                      + ("  [TEMPLATE]" if g.is_template else "") for g in self.geometry]
        if self.missing:
            lines.append("  missing: " + "; ".join(self.missing))
        lines += [f"  ! {x}" for x in self.notes]

        for c in components:
            lines.append("")
            lines.append(self.rests_on(c).describe())
        return "\n".join(lines)

    def describe(self) -> str:
        return self.audit()


# ---------------------------------------------------------------------------
# construction helpers, used by build.py
# ---------------------------------------------------------------------------


def effective_count(tying: Tying, *, n_sites: int = 1, n_partitions: int = 1,
                    embedding_dim: int = 0) -> int:
    """how many numbers one declared parameter actually is.

    the count §4 says determines what data could move it off its prior.  an
    embedding is counted at its dimension rather than at the site count, because
    that is the size of the thing being fitted -- the per-site values are a
    function of it, not free.
    """
    if tying is Tying.GLOBAL:
        return 1
    if tying is Tying.PER_PARTITION:
        return max(int(n_partitions), 1)
    if tying is Tying.EMBEDDING:
        return max(int(embedding_dim), 1)
    return max(int(n_sites), 1)


def fates(impl: Implementation, *, n_sites: int = 1, n_partitions: int = 1,
          embedding_dim: int = 0) -> tuple[ParameterFate, ...]:
    """every entry of one implementation's theta, with its declared status."""
    return tuple(
        ParameterFate(
            process=impl.process, implementation=impl.name, name=name, prior=prior,
            tying=impl.tying,
            n_effective=effective_count(impl.tying, n_sites=n_sites,
                                        n_partitions=n_partitions,
                                        embedding_dim=embedding_dim),
            moved=prior.provenance in MOVED,
            evidence=(impl.source,) if impl.source and prior.provenance in MOVED else (),
            note=prior.note)
        for name, prior in sorted(impl.params.items()))


__all__ = ["Provenance", "Selection", "ParameterFate", "ConversionRecord", "ValidityBreach",
           "FrameRecord", "GeometryRecord", "Basis", "fates", "effective_count",
           "INFORMATIVE", "UNINFORMED", "MOVED"]
