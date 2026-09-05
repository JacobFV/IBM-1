"""the named explicit models.

    M = materialize(R, r, B, F, A, T, P)

every entry in this package is a *lazy materialization of the one implicit
model*, not an independently defined brain model.  none of them declares a
component, a topology or a process; those are declared once, in the registry,
and a model here only says which slice of that ontology to instantiate and how
finely.  if two models in this library disagree about the dynamics of the same
state variable, one of them is wrong, because they are reading the same
declaration -- which is the whole reason the library is written this way.

ARCHITECTURE.md §7 puts it plainly: the process graph is finite, the core clique
is shared across most materializations, and `R`, `r` and `B` are the levers by
which materializations actually differ.  so the interesting content of every
module here is the `Resolution` -- r(q) and B(q) together -- and the honest
accounting of which parts of the result will be moved by evidence and which will
sit at their prior.

## why r(q) is written the way it is

resolution earns its cost only where coarse-graining fails to commute with the
dynamics (§1): a nonlinearity whose average is not the average's image,
heterogeneity inside a coarse cell, or a topology whose edges do not survive
coarsening.  r(q) is derived from *that*, not from proximity to whatever we
happen to be measuring -- although the two coincide more often than is
comfortable, because instruments are placed where the interesting structure is.

the recurring shapes:

- **an invasive contact** sits inside a volume where the electromagnetic kernel
  falls off as 1/r and the laminar dipole layer is 300 um thick.  averaging a
  laminar current profile over a millimetre destroys the very quantity the
  contact measures, so 50 um near the contact is not luxury, it is the minimum
  at which the forward map means anything.  ten centimetres away the same field
  is a smooth multipole and 10 mm is exact.
- **a scalp sensor** sees a field that has already been low-pass filtered in
  space by the skull.  the skull's smoothing kernel is centimetres wide, so no
  amount of source-side refinement changes the prediction: eeg is spatially
  broad and coarse by physics, not by budget.  its *bandwidth* is the opposite
  -- a millisecond-resolved measurement constrains 0.5-100 Hz honestly.
- **a bold voxel** integrates over a vascular unit and a several-second
  impulse response.  the spatial story is the interesting one (a 2 mm voxel
  straddles two cortical areas and both banks of a sulcus, and the vascular
  tree's edges do not survive coarsening), and the temporal story is finished
  above 0.25 Hz.  fine in space, narrow in band.
- **a macro surrogate** wants neither.  it is coarse in both, deliberately,
  because it exists to be run 10^4 times.

## what this package must record and usually does not

§7 also requires that a prediction resting on prior-dominated structure not be
presented with the confidence of one resting on constrained structure.  a model
that only says what it materializes cannot honour that, so every `NamedModel`
carries `constrained` and `prior_dominated` up front: the first names what the
listed sources can actually move, the second names what will come out smooth
because nothing in the corpus distinguishes it.  writing them before fitting,
rather than reading them off a posterior afterwards, is the point.

## on the request interface

`MaterializationRequest` is declared in `ibm.materialize.request`.  the
conventions this package assumes, since the request is deliberately generic:

- `regions` -- the symbolic R, before r(q) refines it
- `resolution` -- r(q) and B(q) as ordered `ResolutionRule`s; first match wins
- `bands` -- the widest band materialized per *field*, keyed by field name.  the
  per-region band lives in the resolution rules; this is the ceiling
- `devices` -- instrument supports from `ibm.fields.supports` that must exist
  for the model to mean anything
- `window` -- seconds of trajectory the belief is carried over, since a state
  variable is a belief about a trajectory and not a value at an instant
- `budget` -- ceiling on materialized state variables
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as _field

from ibm.materialize.request import MaterializationRequest
from ibm.registry import REGISTRY
# imported for its side effect: the partitioning systems must be registered
# before `anat()` can check a label against them, and a library module that
# named a label no system declares is the same class of error as a dangling
# selector -- it should fail at import, not evaluate to an empty region.
from ibm import anatomy as _anatomy  # noqa: F401
from ibm.vocabulary import (
    ALPHA,
    Anat,
    BETA,
    Ball,
    Band,
    DC,
    DELTA,
    Difference,
    Everywhere,
    FULL,
    GAMMA,
    HEMODYNAMIC,
    HIGH_GAMMA,
    LFP,
    Near,
    OnSupport,
    Resolution,
    ResolutionRule,
    SPIKE,
    STRUCTURAL,
    Sel,
    THETA,
    ULTRASLOW,
    Union,
    sel,
    validate_id,
    within,)

#: source cards are named with hyphens (`erp-core`, `7t-qsm-venograms`,
#: `wan2.2-ti2v-5b`) and so do not satisfy the ibm id grammar.  they get their
#: own, looser check -- the point is only to catch a typo that would silently
#: bind a model to a card that does not exist.
_CARD = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")


@dataclass(frozen=True)
class NamedModel:
    """one named materialization of the implicit model.

    this is a *request plus its accounting*, not a model class.  there is no
    behaviour here and there deliberately cannot be: behaviour belongs to the
    processes the request traces to, and a named model that could override a
    process would have quietly become an independently defined brain model.
    """

    id: str
    doc: str
    request: MaterializationRequest

    #: which observations contribute a likelihood.  ids into the observation
    #: table; the coupling that produces the observed variable is an ordinary
    #: process, and these carry only the evidence attached to it (§6).
    observations: tuple[str, ...] = ()
    #: which state this materialization will accept being clamped.  an
    #: intervention is externally constrained state, not an input (§6), so a
    #: model that accepts none is not thereby passive -- it simply has no
    #: variable an experimenter is allowed to pin.
    interventions: tuple[str, ...] = ()

    #: card ids in `data/sources/`.  fit sources may move theta; eval sources
    #: never do.  keeping the two lists separate here rather than in the fitting
    #: code is what stops a model being scored on what trained it.
    fit_sources: tuple[str, ...] = ()
    eval_sources: tuple[str, ...] = ()
    #: teacher cards.  a teacher supplies distillation targets and never an
    #: observation likelihood, and its precision is calibrated from its own
    #: reported accuracy -- see ARCHITECTURE.md "distillation precision".
    teachers: tuple[str, ...] = ()
    #: negative controls: sources that must produce a null.  if one does not,
    #: the pipeline is wrong and the model's other numbers mean nothing.
    negative_controls: tuple[str, ...] = ()

    #: written before fitting, not read off a posterior afterwards.
    constrained: tuple[str, ...] = ()
    prior_dominated: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        validate_id(self.id)
        for group in (self.fit_sources, self.eval_sources, self.teachers,
                      self.negative_controls):
            for card in group:
                if not _CARD.match(card):
                    raise ValueError(f"{self.id}: {card!r} is not a source card id")
        if not self.constrained and not self.prior_dominated:
            raise ValueError(
                f"{self.id}: a model must state what evidence moves and what stays "
                "at its prior -- §7 forbids presenting the two alike"
            )

    @property
    def cards(self) -> tuple[str, ...]:
        seen = dict.fromkeys(
            (*self.fit_sources, *self.eval_sources, *self.teachers,
             *self.negative_controls)
        )
        return tuple(seen)


#: the library, keyed by id.
MODELS: dict[str, NamedModel] = {}


def register(m: NamedModel) -> NamedModel:
    if m.id in MODELS:
        raise ValueError(f"{m.id} declared twice")
    MODELS[m.id] = m
    return m


# ---------------------------------------------------------------------------
# shorthands
# ---------------------------------------------------------------------------


def rule(region, spacing_mm: float, band: Band = FULL) -> ResolutionRule:
    """one clause of r(q).  rules evaluate in order and the first match wins,
    so write them from the finest neighbourhood outwards."""
    return ResolutionRule(region=region, spacing_mm=spacing_mm, band=band)


def res(*rules: ResolutionRule, default_mm: float = 10.0,
        default_band: Band = Band(0.0, 100.0)) -> Resolution:
    """r(q) with an explicit fallback.

    the default clause is not a shrug.  it is the statement that everywhere the
    named rules did not reach, the state is smooth enough and the processes
    acting on it linear enough that coarse-graining commutes -- which is exactly
    the condition under which a finer materialization would be waste.
    """
    return Resolution(rules=tuple(rules), default_mm=default_mm,
                      default_band=default_band)


def anat(system: str, *labels: str):
    """one or more labels of one partitioning system, checked at import time.

    this exists because a mistyped label is invisible.  §8 refuses a process
    whose selector names an unregistered component -- "a typo here is a
    declaration error, not a silently empty materialization" -- and a region is
    the same object one level down: `anat("cortical_areas", "pericalcarine")` looks
    perfectly reasonable and selects nothing at all, because the declared system
    is a desikan-killiany parcellation whose occipital labels are
    `pericalcarine`, `cuneus`, `lingual` and `lateraloccipital`.  a model written
    that way materializes at its *default* spacing everywhere and reports
    success; the refinement it argued for in its docstring never happened.

    several labels union, because an anatomical region a model reasons about is
    usually more than one parcel of whatever atlas is available -- "inferior
    frontal" is `parsopercularis` and `parstriangularis`, and writing the union
    out is more honest than picking one and pretending the other is elsewhere.
    """
    a = REGISTRY.anatomies.get(system)
    if a is None:
        raise ValueError(
            f"no registered partitioning system {system!r}; known: "
            f"{', '.join(sorted(REGISTRY.anatomies))}")
    if not labels:
        raise ValueError(f"anat({system!r}) names no label")
    bad = [l for l in labels if l not in a.labels]
    if bad:
        raise ValueError(
            f"{system!r} declares no label(s) {bad}.  a region naming a label the system does "
            f"not have selects nothing and refines nothing, silently.  {system!r} declares: "
            f"{', '.join(a.labels)}")
    parts = tuple(Anat(system, l) for l in labels)
    return parts[0] if len(parts) == 1 else Union(parts)


#: how far from the subject's white surface a parenchyma voxel has to be before
#: the volume is allowed to carry state at it.  two constraints, and the binding
#: one is the second.
#:
#: - anatomy: the cortical ribbon runs outward from the white surface, 2-3 mm
#:   thick with a 95th percentile near 4 mm, so 6 mm clears it everywhere.
#: - the disjointness rule: `build._support_overlap` counts a column node as
#:   inside a `tissue` cell when it is within half that cell's width, so an
#:   exclusion smaller than half the volume spacing is an overlap by
#:   construction, however thin the ribbon happens to be.  every model that uses
#:   `subcortex()` samples the volume at 4-10 mm, and 6 mm clears half of all of
#:   them.
#:
#: it stops short of the deep grey it is not supposed to remove: the putamen and
#: the thalamus sit 8-20 mm inside the white surface, so a 6 mm shell takes the
#: ribbon and the immediately subjacent white matter and leaves the structures
#: the volume block exists to carry.
RIBBON_MM = 6.0


def subcortex(exclusion_mm: float = RIBBON_MM):
    """the parenchyma the cortical sheet does not already index.

    the region every model in this library uses to make a two-support placement
    of the neural field a *partition* rather than a second copy of it.  the sheet
    and the parenchyma volume are two samplings of one domain, so naming both in
    R instantiates each neural component twice -- once on column nodes, once on
    voxels -- and the two blocks are a partition only if they cover disjoint
    positions.  carving the ribbon out of the volume is how R says "the volume
    carries what the sheet does not", and `build._overlap_check` then measures
    that rather than taking it on trust.

    `Near("cortex", ...)` is an ordinary landmark region -- `Near` is declared
    over device *or landmark* positions -- and the landmark is external geometry
    exactly as an atlas is: a caller hands the subject's own white-surface
    vertices to `build(anchors={"cortex": ...})`.  a build that does not supply
    them gets a `MissingData` naming the landmark, which is the correct answer:
    without the sheet's own vertices there is no way to say which millimetres it
    already covers, and a volume that guessed would double-count.

    it is a region and not a filtered geometry on purpose.  R is the part of a
    request that decides *where*, and "the parenchyma the sheet does not index"
    is a statement about where rather than a different notion of what tissue is.
    """
    return Difference(OnSupport("tissue"), Near("cortex", exclusion_mm))


def cards_used() -> tuple[str, ...]:
    """every source card the library binds to, for diffing against
    `ls data/sources`.  a dangling card id here is the same class of error as a
    dangling selector in the registry, and should fail as loudly."""
    seen: dict[str, None] = {}
    for m in MODELS.values():
        for c in m.cards:
            seen[c] = None
    return tuple(sorted(seen))


# importing the group modules is what populates MODELS.  the import is at the
# bottom because each module imports `register`, `rule` and `res` from here.
from ibm.materialize.library import (  # noqa: E402,F401
    decoding,
    electrophysiology,
    hemodynamic,
    slow,
    state,
    stimulation,
    surrogate,)

__all__ = [
    "MODELS",
    "NamedModel",
    "RIBBON_MM",
    "anat",
    "cards_used",
    "register",
    "res",
    "rule",
    "subcortex",
]
