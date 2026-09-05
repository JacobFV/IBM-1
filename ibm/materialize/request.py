"""what to materialize, before anything is built.

    M = materialize(R, r, B, F, A, T, P)

a request is symbolic and cheap.  it names targets, regions, a resolution rule
set r(q), bands, a subject, the devices in play and a window, and it commits to
nothing: no geometry is read, no atlas is resampled, no eigendecomposition is
run.  that separation is what makes the cache in `cache.py` possible -- the
request hashes to a key long before the expensive artifacts it implies exist.

the budget is not a safety rail bolted on afterwards.  ARCHITECTURE.md §1 makes
bandwidth co-equal with spatial resolution as a laziness axis, and the two
budgets *multiply*: a materialization is sites x retained spectral components,
and either axis alone tells you almost nothing about what a request costs.  a
whole-brain hemodynamic model is spatially broad and temporally narrow; a
single-electrode spike model is spatially tiny and temporally wide; both are
affordable and their product is not.  so `Budget` carries both, and reports both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field, replace
from typing import Any, Iterable, Sequence

import numpy as np

from ibm.fields.uncertainty.spectral import TemporalBasis
from ibm.vocabulary import (
    Anat, Ball, Band, Everywhere, FULL, HEMODYNAMIC, LFP, Near, OnSupport, Region,
    Resolution, ResolutionRule, Sel, Union, sel,
)


class BudgetExceeded(Exception):
    """a materialization would cost more than the request allows.

    raised eagerly, from inside the site builders, rather than after allocating:
    an octree that refines past its budget is a machine that discovers it is out
    of memory by running out of memory.
    """


class TierCeilingExceeded(Exception):
    """a materialization fell through to a prior tier the request had forbidden.

    the counterpart of `BudgetExceeded` for structure rather than cost, and it
    exists for the same reason: the failure it names is one a caller can only act
    on if it is raised rather than recorded.  a fall-through is a legitimate
    materialization -- §1 says structure that no evidence constrains stays at its
    prior, and that is the machinery working -- so the default is to allow it and
    record the rung.  what a model may not do is fall through SILENTLY when its
    entire claim depends on the structure being this subject's: a virtual-lesion
    model over a group connectome resects somebody else's fascicle, and the number
    it produces is not wrong in a way any inspection of it would reveal.
    """


#: the ceilings `MaterializationRequest.max_tier` takes, as ranks rather than as
#: one enum.  the three prior modules each declare their own three-rung `Tier` and
#: they are not interchangeable -- "this subject's tractogram" and "this subject's
#: angiogram" are different claims -- but their RANKS mean the same thing, which
#: is the only thing a request can usefully constrain.
SUBJECT_ONLY = 0        # nothing but this subject's own measurements
POPULATION_OK = 1       # a group connectome, a population atlas, a species' microcircuit
ANY_TIER = 2            # a distance prior, a generative synthesis: a shape, not a measurement


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """the two laziness axes and their product.

    `max_state_variables` counts one component of one field at one position --
    the architecture's definition of a state variable, and the thing that scales
    with r(q).  `max_spectral_coefficients` counts retained temporal laplacian
    components summed over those variables, and scales with B(q).  neither bounds
    the other: 10^6 sites at DC only, and 10^3 sites carrying a full 500-component
    spectrum, are both perfectly affordable and have wildly different shapes.

    `max_bytes` is the only one that is a hardware fact rather than a modelling
    choice, and it is the product plus the edge sets.
    """

    max_state_variables: int = 2_000_000
    max_spectral_coefficients: int = 200_000_000
    max_bytes: int = 8 << 30
    max_sites_per_support: int = 1_000_000
    max_edges: int = 200_000_000
    max_octree_level: int = 14

    def check(self, *, state_variables: int = 0, spectral_coefficients: int = 0,
              n_bytes: int = 0, edges: int = 0) -> list[str]:
        """every violation, not the first: a request that blows three ceilings
        should be re-scoped once rather than three times."""
        out: list[str] = []
        if state_variables > self.max_state_variables:
            out.append(f"state variables {state_variables:,} > budget {self.max_state_variables:,} "
                       f"({state_variables / self.max_state_variables:.1f}x): coarsen r(q)")
        if spectral_coefficients > self.max_spectral_coefficients:
            out.append(f"spectral coefficients {spectral_coefficients:,} > budget "
                       f"{self.max_spectral_coefficients:,} "
                       f"({spectral_coefficients / self.max_spectral_coefficients:.1f}x): narrow B(q) "
                       "or shorten the window")
        if n_bytes > self.max_bytes:
            out.append(f"{n_bytes / 2**30:.2f} GiB > budget {self.max_bytes / 2**30:.2f} GiB")
        if edges > self.max_edges:
            out.append(f"edges {edges:,} > budget {self.max_edges:,}: the topology radius is too "
                       "wide for this r(q)")
        return out

    def enforce(self, **kw) -> None:
        v = self.check(**kw)
        if v:
            raise BudgetExceeded("; ".join(v))


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """the window a belief is a belief *about*.

    a state variable is not a value at an instant but a belief about a trajectory
    over a window (§1), so the window is part of the request rather than a
    runtime detail.  `n` is set by the fastest *nonlinear* process in the traced
    graph, not the fastest process: linear and delayed couplings are exact at any
    n, and only a pointwise nonlinearity aliases if the window undersamples it.
    """

    n: int = 512
    dt: float = 2e-3
    overlap: float = 0.25

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError("a window of fewer than two samples carries no spectrum")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")

    @property
    def duration_s(self) -> float:
        return self.n * self.dt

    @property
    def nyquist_hz(self) -> float:
        return 0.5 / self.dt

    @property
    def resolution_hz(self) -> float:
        return 1.0 / self.duration_s

    def basis(self, band: Band | None = None) -> TemporalBasis:
        b = TemporalBasis(self.n, self.dt)
        return b.truncated(band) if band is not None else b

    def k_for(self, band: Band) -> int:
        """retained spectral components inside a band.  the temporal half of cost."""
        b = TemporalBasis(self.n, self.dt)
        return int(b.band_mask(band).sum())

    def covers(self, band: Band) -> tuple[bool, str]:
        if band.hi_hz > self.nyquist_hz:
            return False, (f"band reaches {band.hi_hz:g} Hz but dt={self.dt:g}s gives a nyquist of "
                           f"{self.nyquist_hz:g} Hz -- everything above it aliases")
        if band.lo_hz and band.lo_hz < self.resolution_hz:
            return False, (f"band starts at {band.lo_hz:g} Hz but a {self.duration_s:g}s window "
                           f"resolves no finer than {self.resolution_hz:g} Hz")
        return True, ""


# ---------------------------------------------------------------------------
# subject and devices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectSpec:
    """whose head this is.

    named rather than optional because every warp in `ibm.frames` that matters is
    subject-specific, and a template stands in for a subject with a systematic,
    head-size-dependent error rather than a random one.  a request that does not
    say which subject it speaks about cannot record honest registration residuals.
    """

    id: str = "template"
    frame: str = "subject_t1"
    surface_frame: str = "subject_surf"
    template: str | None = "mni152"
    note: str = ""

    @property
    def is_template(self) -> bool:
        return self.id == "template"


@dataclass(frozen=True)
class DeviceSpec:
    """an instrument in the materialization, as ordinary state on its own support.

    a device is not an input or an output of the model (§1).  it is state with a
    support, a frame and a coupling process, and it appears in a request only
    because its *positions* are the anchor r(q) refines around and because its
    frame needs a warp chain to the subject's.
    """

    name: str
    support: str
    frame: str
    role: str = "observe"                 # observe | stimulate | both
    n_elements: int = 0
    positions: Any = None                 # (m, 3) in `frame`, if known
    element_ids: tuple[str, ...] = ()
    note: str = ""

    def anchor_positions(self) -> np.ndarray | None:
        return None if self.positions is None else np.asarray(self.positions, float).reshape(-1, 3)


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterializationRequest:
    """R, r, B, F, A, T, P plus the things a real build needs to be honest.

    `targets` is what the model is *for*.  everything else is either a lever on
    cost or a declaration of context.  the four ontology tuples (`fields`,
    `anatomy`, `topologies`, `processes`) are additive hints: leaving them empty
    is the normal case, because dependency tracing already determines what must
    be instantiated and every process reachable from a target is materialized
    (§7).  naming them narrows or widens that set deliberately, and the trace
    records the difference.
    """

    name: str
    targets: tuple[Sel, ...] = ()
    regions: tuple[tuple[str, Region], ...] = ()      # named R, in priority order
    resolution: Resolution = _field(default_factory=Resolution)
    bands: tuple[tuple[str, Band], ...] = ()          # component or field id -> B override
    fields: tuple[str, ...] = ()
    anatomy: tuple[str, ...] = ()
    topologies: tuple[str, ...] = ()
    processes: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    interventions: tuple[str, ...] = ()
    devices: tuple[DeviceSpec, ...] = ()
    subject: SubjectSpec = _field(default_factory=SubjectSpec)
    window: Window = _field(default_factory=Window)
    budget: Budget = _field(default_factory=Budget)
    frame: str = "subject_t1"
    policy: str = "prefer_lti"
    seed: int = 0
    allow_template_geometry: bool = False
    #: how far down a prior ladder this materialization may fall when the subject
    #: has no measurement of a piece of structure.  `None` is no ceiling, which is
    #: the right default: §1 says structure that evidence does not constrain stays
    #: at its prior, so a fall-through is a legitimate materialization and refusing
    #: it by default would put thirty-four of the library's forty models back where
    #: they were.  what the ceiling buys is the ability of a model whose whole
    #: claim is about ONE subject to say so and fail loudly, rather than quietly
    #: resecting a fascicle from a population average.  see `SUBJECT_ONLY`.
    max_tier: int | None = None
    #: per-piece overrides on that ceiling, keyed by topology, support or
    #: partitioning system.  the common case is a model that must be
    #: subject-specific in exactly one respect -- `virtual_lesion` about its
    #: tractometric edges, `dbs_response` about its contact positions -- and would
    #: be needlessly refused by a blanket `max_tier=SUBJECT_ONLY`.
    tier_ceilings: tuple[tuple[str, int], ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "bands", tuple(self.bands))
        object.__setattr__(self, "devices", tuple(self.devices))
        object.__setattr__(self, "tier_ceilings", tuple(self.tier_ceilings))

    # -- accessors -------------------------------------------------------

    @property
    def target_components(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(v for s in self.targets for v in s.vars))

    def region(self, name: str) -> Region:
        for n, r in self.regions:
            if n == name:
                return r
        raise KeyError(f"request {self.name!r} declares no region {name!r}; it has "
                       f"{[n for n, _ in self.regions] or 'none'}")

    @property
    def scope(self) -> Region:
        """the union of named regions; Everywhere if none were named."""
        if not self.regions:
            return Everywhere()
        return Union(tuple(r for _, r in self.regions))

    def band_for(self, component: str, declared: Band = FULL) -> Band:
        """B for a component: the declared band narrowed by the most specific override.

        component id beats field id beats the resolution default, because an
        override written against `neural.exc.activity` is a statement about that
        variable and one written against `neural` is a statement about a habit.
        """
        out = declared
        field = component.split(".", 1)[0]
        for key, b in self.bands:
            if key == field:
                out = out & b
        for key, b in self.bands:
            if key == component:
                out = out & b
        return out & self.resolution.default_band

    def tier_ceiling(self, what: str) -> int | None:
        """the deepest rung `what` may come from, most specific override winning.

        the same shape `band_for` has and for the same reason: a ceiling written
        against `tractometric` is a statement about that structure and one written
        as `max_tier` is a statement about the materialization's habits.  a
        per-piece entry therefore overrides the blanket one in BOTH directions --
        it may loosen as well as tighten -- because the case it exists for is a
        model that is strict about one thing and indifferent about the rest.
        """
        for key, r in self.tier_ceilings:
            if key == what:
                return int(r)
        return self.max_tier

    def admits_tier(self, what: str, rank: int) -> bool:
        ceiling = self.tier_ceiling(what)
        return ceiling is None or int(rank) <= int(ceiling)

    def anchors(self) -> dict[str, np.ndarray]:
        """device positions, keyed by name, for `Near(...)` regions to refine around."""
        out: dict[str, np.ndarray] = {}
        for d in self.devices:
            p = d.anchor_positions()
            if p is not None:
                out[d.name] = p
        return out

    # -- derivation ------------------------------------------------------

    def with_resolution(self, resolution: Resolution) -> "MaterializationRequest":
        return replace(self, resolution=resolution)

    def coarsened(self, factor: float = 2.0) -> "MaterializationRequest":
        """the same request one notch cheaper, for the equivalence check in build.py.

        materializing this alongside the fine request is how the §1 criterion is
        *tested* rather than asserted: where every process is LTI and the state is
        smooth the two agree exactly, and the fine one was waste.
        """
        r = self.resolution
        return replace(self, name=f"{self.name}@{factor:g}x",
                       resolution=Resolution(
                           tuple(ResolutionRule(x.region, x.spacing_mm * factor, x.band)
                                 for x in r.rules),
                           r.default_mm * factor, r.default_band))

    def narrowed(self, band: Band) -> "MaterializationRequest":
        r = self.resolution
        return replace(self, resolution=Resolution(r.rules, r.default_mm, r.default_band & band))

    # -- identity --------------------------------------------------------

    def spec(self) -> dict[str, Any]:
        """canonical, hashable description.  the cache key is a hash of this.

        deliberately excludes nothing that changes the answer and includes nothing
        that does not: `notes` is prose, and `name` is a label, but both are cheap
        and keeping them in makes a cache entry self-identifying when someone is
        staring at a directory of hashes wondering what they are.
        """
        return {
            "name": self.name,
            "targets": [{"vars": list(s.vars), "region": s.region,
                         "band": (s.band.lo_hz, s.band.hi_hz)} for s in self.targets],
            "regions": [(n, r) for n, r in self.regions],
            "resolution": self.resolution,
            "bands": [(k, (b.lo_hz, b.hi_hz)) for k, b in self.bands],
            "fields": list(self.fields), "anatomy": list(self.anatomy),
            "topologies": list(self.topologies), "processes": list(self.processes),
            "observations": list(self.observations), "interventions": list(self.interventions),
            "devices": [d for d in self.devices],
            "subject": self.subject, "window": self.window, "frame": self.frame,
            "policy": self.policy, "seed": self.seed,
            "allow_template_geometry": self.allow_template_geometry,
            "max_tier": self.max_tier,
            "tier_ceilings": [list(x) for x in self.tier_ceilings],
        }

    def describe(self) -> str:
        lines = [f"request {self.name!r}  subject={self.subject.id}  frame={self.frame}",
                 f"  targets     {', '.join(self.target_components) or '(none)'}",
                 f"  regions     {', '.join(n for n, _ in self.regions) or 'everywhere'}",
                 f"  r(q)        {len(self.resolution.rules)} rules, default "
                 f"{self.resolution.default_mm:g} mm",
                 f"  B(q)        default {self.resolution.default_band}"
                 + (f", {len(self.bands)} overrides" if self.bands else ""),
                 f"  window      n={self.window.n} dt={self.window.dt:g}s "
                 f"({self.window.duration_s:g}s, nyquist {self.window.nyquist_hz:g} Hz)",
                 f"  devices     {', '.join(d.name for d in self.devices) or '(none)'}",
                 f"  budget      {self.budget.max_state_variables:,} state vars x "
                 f"{self.budget.max_spectral_coefficients:,} coefficients, "
                 f"{self.budget.max_bytes / 2**30:.1f} GiB"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# resolution helpers
# ---------------------------------------------------------------------------


def graded_around(anchor: str, *,
                  fine_mm: float = 0.05, fine_radius_mm: float = 2.0,
                  mid_mm: float = 0.5, mid_radius_mm: float = 20.0,
                  connected: Region | None = None, connected_mm: float = 2.0,
                  default_mm: float = 10.0,
                  band: Band = FULL, default_band: Band = Band(0.0, 100.0)) -> Resolution:
    """the r(q) of ARCHITECTURE.md §1, verbatim, around a named device.

    kept as a helper and *not* as the recommended default, because §1 is explicit
    that r(q) should be derived from where coarse-graining fails to commute rather
    than from proximity to whatever we happen to be measuring.  proximity is a
    decent proxy only because instruments are usually placed where the interesting
    nonlinearity is; `build.resolution_earns_its_cost` is the actual criterion.
    """
    rules = [ResolutionRule(Near(anchor, fine_radius_mm), fine_mm, band),
             ResolutionRule(Near(anchor, mid_radius_mm), mid_mm, band)]
    if connected is not None:
        rules.append(ResolutionRule(connected, connected_mm, band))
    return Resolution(tuple(rules), default_mm, default_band)


def uniform(spacing_mm: float, band: Band = Band(0.0, 100.0)) -> Resolution:
    return Resolution((), spacing_mm, band)


def graded(*rules: tuple[Region, float] | tuple[Region, float, Band],
           default_mm: float = 4.0, default_band: Band = Band(0.0, 100.0)) -> Resolution:
    out = tuple(ResolutionRule(r[0], r[1], r[2] if len(r) > 2 else FULL) for r in rules)
    return Resolution(out, default_mm, default_band)


# ---------------------------------------------------------------------------
# common requests
# ---------------------------------------------------------------------------
#
# these are the shapes of §7's model library, not the library itself: they build
# a request, and `ibm.materialize.library` names and binds the actual models.
# component ids are the ones fixed in docs/CONTRACT.md; a typo here becomes a
# dangling selector at seal time rather than a silently empty materialization.


def eeg_forward(subject: SubjectSpec | None = None, *, montage: DeviceSpec | None = None,
                spacing_mm: float = 3.0, band: Band = LFP,
                window: Window | None = None) -> MaterializationRequest:
    """scalp potential from population state: spatially broad, temporally wide.

    coarse in space on purpose.  the forward map from cortical current to scalp
    potential is a low-pass spatial filter through skull and scalp, so the sensor
    cannot see the difference between a 1 mm and a 3 mm source grid, and refining
    it buys nothing the measurement can constrain.
    """
    dev = montage or DeviceSpec("eeg", "sensor_array", "eeg_cap", "observe", note="scalp montage")
    return MaterializationRequest(
        name="eeg-forward",
        targets=(sel("device.contact_potential", band=band),),
        resolution=uniform(spacing_mm, band),
        anatomy=("cortical_areas",),
        devices=(dev,),
        subject=subject or SubjectSpec(),
        window=window or Window(n=1024, dt=1e-3),
        bands=(("neural", band), ("electromagnetic", band)),
        notes=eeg_forward.__doc__ or "")


def meg_forward(subject: SubjectSpec | None = None, *, dewar: DeviceSpec | None = None,
                spacing_mm: float = 3.0, band: Band = LFP) -> MaterializationRequest:
    dev = dewar or DeviceSpec("meg", "sensor_array", "meg_head", "observe")
    return MaterializationRequest(
        name="meg-forward",
        targets=(sel("electromagnetic.bfield", band=band),),
        resolution=uniform(spacing_mm, band),
        devices=(dev,),
        subject=subject or SubjectSpec(),
        window=Window(n=2048, dt=1e-3),
        bands=(("neural", band), ("electromagnetic", band)),
        notes="magnetic field outside the head from transmembrane current")


def invasive_local(anchor: str = "array", subject: SubjectSpec | None = None, *,
                   device: DeviceSpec | None = None,
                   band: Band = Band(0.5, 5000.0)) -> MaterializationRequest:
    """the opposite corner of the budget: spatially tiny, temporally wide.

    §1's worked example.  50 um within 2 mm of the contacts, 500 um out to 2 cm,
    2 mm through structures connected to that neighbourhood, and coarse everywhere
    else -- and the whole thing is affordable only because the fine region is a
    few cubic centimetres while the band is five decades wide.
    """
    dev = device or DeviceSpec(anchor, "implanted_array", "array", "both")
    return MaterializationRequest(
        name="invasive-local",
        targets=(sel("device.contact_potential", region=Near(anchor, 5.0), band=band),
                 sel("neural.exc.activity", region=Near(anchor, 2.0), band=band)),
        regions=(("neighbourhood", Near(anchor, 20.0)),),
        resolution=graded_around(anchor, fine_mm=0.05, mid_mm=0.5, default_mm=6.0,
                                 band=band, default_band=LFP),
        devices=(dev,),
        subject=subject or SubjectSpec(),
        window=Window(n=8192, dt=1e-4),
        bands=(("neural", band),),
        notes=invasive_local.__doc__ or "")


def bold_forward(subject: SubjectSpec | None = None, *, spacing_mm: float = 2.0,
                 band: Band = HEMODYNAMIC) -> MaterializationRequest:
    """blood state from population activity: spatially broad, temporally narrow.

    the band is the whole point.  blood has no meaningful structure above roughly
    0.5 Hz, so the hemodynamic components carry a handful of spectral coefficients
    each and a whole-brain 2 mm materialization fits comfortably -- the same site
    count at LFP bandwidth would not.
    """
    return MaterializationRequest(
        name="bold-forward",
        targets=(sel("blood.deoxyhemoglobin", "blood.volume", "blood.flow", band=band),),
        resolution=uniform(spacing_mm, band),
        anatomy=("cortical_areas", "vascular_territories"),
        subject=subject or SubjectSpec(),
        window=Window(n=256, dt=0.72),
        bands=(("blood", band), ("metabolic", band), ("neural", Band(0.0, 100.0))),
        notes=bold_forward.__doc__ or "")


def stimulation_response(modality: str = "tms", subject: SubjectSpec | None = None, *,
                         anchor: str = "coil", spacing_mm: float = 1.0,
                         band: Band = LFP) -> MaterializationRequest:
    """an intervention on device state, propagating through ordinary processes.

    nothing here is special-cased.  the coil current is clamped, the em field is
    an ordinary process output, and the population response is ordinary dynamics;
    the only unusual thing about the request is that r(q) is fine in the focus,
    because that is exactly where the field gradient makes coarse-graining fail.
    """
    frame = {"tms": "coil", "tes": "eeg_cap", "tfus": "transducer", "dbs": "electrode_grid"}
    dev = DeviceSpec(anchor, "stimulator", frame.get(modality, "coil"), "stimulate")
    return MaterializationRequest(
        name=f"{modality}-response",
        targets=(sel("neural.exc.activity", "neural.inh.activity", band=band),),
        regions=(("focus", Near(anchor, 30.0)),),
        resolution=graded(( Near(anchor, 30.0), spacing_mm), default_mm=4.0, default_band=band),
        interventions=(f"{modality}_drive",),
        devices=(dev,),
        subject=subject or SubjectSpec(),
        window=Window(n=4096, dt=1e-4),
        notes=stimulation_response.__doc__ or "")


def resting_state(subject: SubjectSpec | None = None, *, spacing_mm: float = 5.0,
                  band: Band = Band(0.0, 0.25)) -> MaterializationRequest:
    """the cheapest useful materialization: coarse in both axes.

    worth keeping as a named request because it is the natural baseline against
    which `resolution_earns_its_cost` is argued -- if a finer materialization
    does not differ from this one, it was not worth building.
    """
    return MaterializationRequest(
        name="resting-state-fc",
        targets=(sel("neural.exc.activity", band=band),),
        resolution=uniform(spacing_mm, band),
        anatomy=("cortical_areas",),
        subject=subject or SubjectSpec(),
        window=Window(n=512, dt=1.0),
        notes=resting_state.__doc__ or "")


def thermal_safety(subject: SubjectSpec | None = None, *, anchor: str = "transducer",
                   spacing_mm: float = 1.0) -> MaterializationRequest:
    """temperature under a stimulator.  slow, so the band collapses to near-DC.

    the state variable count is what costs here, not the spectrum: heat diffusion
    is smooth in time and steep in space around a focus.
    """
    return MaterializationRequest(
        name="thermal-safety",
        targets=(sel("thermal.temperature", band=Band(0.0, 1.0)),),
        regions=(("focus", Near(anchor, 40.0)),),
        resolution=graded((Near(anchor, 40.0), spacing_mm), default_mm=5.0,
                          default_band=Band(0.0, 1.0)),
        devices=(DeviceSpec(anchor, "stimulator", "transducer", "stimulate"),),
        subject=subject or SubjectSpec(),
        window=Window(n=512, dt=0.5),
        notes=thermal_safety.__doc__ or "")


COMMON: dict[str, Any] = {
    "eeg-forward": eeg_forward,
    "meg-forward": meg_forward,
    "invasive-local": invasive_local,
    "bold-forward": bold_forward,
    "stimulation-response": stimulation_response,
    "resting-state-fc": resting_state,
    "thermal-safety": thermal_safety,
}


__all__ = ["Budget", "BudgetExceeded", "Window", "SubjectSpec", "DeviceSpec",
           "MaterializationRequest", "graded_around", "graded", "uniform", "COMMON",
           "eeg_forward", "meg_forward", "invasive_local", "bold_forward",
           "stimulation_response", "resting_state", "thermal_safety"]
