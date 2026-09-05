"""the words the ontology is written in.

two jobs.  first, the value types every declaration uses: bands, regions,
selectors, priors, validity.  second, the controlled vocabulary that stops the
ontology sprawling -- an id grammar, a synonym table, and collision detection.

the failure this guards against is not that the model is wrong.  it is that after
two years there are hundreds of components declared across dozens of files,
several pairs of which are the same thing under different names and several more
of which almost are, until no one can say which processes are actually coupled.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field as _field
from enum import Enum

# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------

_ID = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*(?:\.[a-z][a-z0-9]*(?:_[a-z0-9]+)*)*$")

#: tokens meaning the same thing collapse onto one canonical form.  edit this
#: table when a genuinely new concept arrives; do not work around it by inventing
#: a fresh spelling.
SYNONYMS: dict[str, str] = {
    "rate": "activity", "firing": "activity", "spiking": "activity", "spike": "activity",
    "fr": "activity", "act": "activity", "discharge": "activity",
    "potential": "potential", "voltage": "potential", "v": "potential", "vm": "potential",
    "depolarization": "potential", "membrane": "potential",
    "e": "exc", "excitatory": "exc", "pyr": "exc", "pyramidal": "exc",
    "i": "inh", "inhibitory": "inh", "interneuron": "inh",
    "conc": "concentration", "level": "concentration", "amount": "concentration",
    "flow": "flow", "flux": "flow", "perfusion": "flow", "cbf": "flow",
    "vol": "volume", "cbv": "volume", "press": "pressure",
    "temp": "temperature", "cond": "conductivity", "perm": "permittivity",
    "oxy": "oxygenation", "deoxy": "deoxyhemoglobin", "dhb": "deoxyhemoglobin",
    "glu": "glucose", "o2": "oxygen", "syn": "synaptic", "adapt": "adaptation",
    "mod": "modulator", "disp": "displacement", "vel": "velocity",
}

STOPWORDS = frozenset({"state", "value", "var", "variable", "component", "the", "of"})


def validate_id(id_: str) -> None:
    if not _ID.match(id_):
        raise ValueError(
            f"invalid id {id_!r}: expected dotted lowercase snake_case, e.g. 'neural.exc.activity'"
        )


def tokens(id_: str) -> list[str]:
    return [t for part in id_.split(".") for t in part.split("_")]


def normal_form(id_: str) -> tuple[str, ...]:
    """canonical token multiset.  two ids sharing one are the same concept twice."""
    ts = [SYNONYMS.get(t, t) for t in tokens(id_)]
    return tuple(sorted(t for t in ts if t not in STOPWORDS))


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


@dataclass(frozen=True)
class Collision:
    a: str
    b: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.a!r} ~ {self.b!r}: {self.detail}"


def find_collisions(ids: list[str], threshold: float = 0.90) -> list[Collision]:
    out: list[Collision] = []
    byn: dict[tuple[str, ...], list[str]] = {}
    for i in ids:
        byn.setdefault(normal_form(i), []).append(i)
    for nf, grp in byn.items():
        for a, b in zip(grp, grp[1:]):
            out.append(Collision(a, b, "normal_form", f"both normalize to {'+'.join(nf)}"))
    for k, a in enumerate(ids):
        for b in ids[k + 1:]:
            if normal_form(a) != normal_form(b) and similarity(a, b) >= threshold:
                out.append(Collision(a, b, "similar", f"string similarity {similarity(a,b):.2f}"))
    return out


def field_of(id_: str) -> str:
    return id_.split(".", 1)[0]


# ---------------------------------------------------------------------------
# bands -- the temporal laziness axis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Band:
    """half-open frequency interval in Hz.

    bandwidth is co-equal with spatial resolution as a materialization lever
    (ARCHITECTURE.md §1).  a band on a selector says which temporal laplacian
    components a coupling touches; a band on a variable says which are meaningful
    at all.
    """

    lo_hz: float = 0.0
    hi_hz: float = math.inf

    def __post_init__(self) -> None:
        # an EMPTY band is a legal value, not an error.  intersection produces one
        # whenever a component's meaningful band and a materialization's requested
        # band do not overlap, and the meaning is a decision the materializer has
        # to make -- do not instantiate this component here -- rather than a
        # contradiction.  refusing to represent it only moves the failure to the
        # caller, which is where this first bit: a whole-brain request whose floor
        # sat exactly on a hemodynamic component's ceiling crashed the tracer.
        # inverted bands are still an error, because those are a real mistake.
        if self.hi_hz < self.lo_hz:
            raise ValueError(f"inverted band [{self.lo_hz}, {self.hi_hz}): hi below lo")

    @property
    def is_empty(self) -> bool:
        """no frequency lies inside.  a component whose materialized band is empty
        contributes nothing and should not be allocated."""
        return self.hi_hz <= self.lo_hz

    def __bool__(self) -> bool:
        return not self.is_empty

    def __and__(self, o: "Band") -> "Band":
        lo, hi = max(self.lo_hz, o.lo_hz), min(self.hi_hz, o.hi_hz)
        return Band(lo, max(lo, hi))

    def intersects(self, o: "Band") -> bool:
        return max(self.lo_hz, o.lo_hz) < min(self.hi_hz, o.hi_hz)

    def covers(self, o: "Band") -> bool:
        if o.is_empty:
            return True
        if self.is_empty:
            return False
        return self.lo_hz <= o.lo_hz and o.hi_hz <= self.hi_hz

    def __repr__(self) -> str:
        if self.is_empty:
            return "Band(empty)"
        hi = "inf" if math.isinf(self.hi_hz) else f"{self.hi_hz:g}"
        return f"Band({self.lo_hz:g}-{hi}Hz)"


DC          = Band(0.0, 0.01)
ULTRASLOW   = Band(0.0, 0.1)
HEMODYNAMIC = Band(0.0, 0.5)
DELTA       = Band(0.5, 4.0)
THETA       = Band(4.0, 8.0)
ALPHA       = Band(8.0, 13.0)
BETA        = Band(13.0, 30.0)
GAMMA       = Band(30.0, 90.0)
HIGH_GAMMA  = Band(70.0, 200.0)
LFP         = Band(0.5, 300.0)
SPIKE       = Band(300.0, 5000.0)
STRUCTURAL  = Band(0.0, 0.001)
FULL        = Band(0.0, math.inf)


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------


class Region:
    """symbolic set expression over anatomical and geometric primitives.

    regions stay symbolic until materialization, where they become weight vectors
    over the site table.  membership is soft: a probabilistic atlas contributes
    weights in [0,1] and those weights are used directly as process gains rather
    than thresholded.  a partition boundary is a gradient, not a wall.
    """

    def __and__(self, o: "Region") -> "Region": return Intersect((self, o))
    def __or__(self, o: "Region") -> "Region":  return Union((self, o))
    def __sub__(self, o: "Region") -> "Region": return Difference(self, o)
    def __invert__(self) -> "Region":           return Difference(Everywhere(), self)


@dataclass(frozen=True)
class Everywhere(Region):
    """the whole support of whichever field the selector names."""


@dataclass(frozen=True)
class Anat(Region):
    """membership in one label of one anatomical partitioning system."""
    system: str
    label: str
    threshold: float = 0.0     # 0.0 keeps soft weights


@dataclass(frozen=True)
class Ball(Region):
    center: tuple[float, float, float]
    radius_mm: float
    frame: str = "mni152"


@dataclass(frozen=True)
class Near(Region):
    """within a distance of named device or landmark positions.

    where the highest-resolution materializations live: electrode, coil, probe
    and array neighbourhoods.
    """
    anchor: str
    radius_mm: float
    metric: str = "euclidean"   # or "geodesic" on a surface support


@dataclass(frozen=True)
class OnSupport(Region):
    """restrict to positions lying on a named support."""
    support: str


@dataclass(frozen=True)
class Union(Region):      parts: tuple[Region, ...]
@dataclass(frozen=True)
class Intersect(Region):  parts: tuple[Region, ...]
@dataclass(frozen=True)
class Difference(Region):
    left: Region
    right: Region


# ---------------------------------------------------------------------------
# selectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sel:
    """a precise slice of state: which variables, where, in which band.

    ARCHITECTURE.md §4 requires that process definitions avoid generic statements
    like `neural -> neural`.  a Sel is that requirement made mechanical: the
    narrowest thing you can write is still explicit about variables, region and
    band, and the registry checks every name at seal time.
    """

    vars: tuple[str, ...]
    region: Region = _field(default_factory=Everywhere)
    band: Band = FULL

    def __post_init__(self) -> None:
        v = (self.vars,) if isinstance(self.vars, str) else tuple(self.vars)
        object.__setattr__(self, "vars", v)

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(field_of(v) for v in self.vars))

    def narrowed(self, band: Band) -> "Sel":
        return Sel(self.vars, self.region, self.band & band)

    def restricted(self, region: Region) -> "Sel":
        return Sel(self.vars, self.region & region, self.band)

    def may_overlap(self, o: "Sel") -> bool:
        """conservative: False only when the two provably cannot share state.

        region intersection is deliberately not decided here -- regions are
        symbolic until materialization -- so this screens on variables and band
        and errs towards claiming overlap.
        """
        return bool(set(self.vars) & set(o.vars)) and self.band.intersects(o.band)


def sel(*vars_: str, region: Region | None = None, band: Band = FULL) -> Sel:
    return Sel(tuple(vars_), region or Everywhere(), band)


def within(field: str, *components: str, region: Region | None = None, band: Band = FULL) -> Sel:
    """selector over components of one field: ``within("neural", "exc.activity")``."""
    return Sel(tuple(f"{field}.{c}" for c in components), region or Everywhere(), band)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolutionRule:
    """one clause of r(q); rules evaluate in order, first match wins."""
    region: Region
    spacing_mm: float
    band: Band = FULL


@dataclass(frozen=True)
class Resolution:
    """r(q) and B(q) together.

    resolution earns its cost only where coarse-graining fails to commute with
    the dynamics -- a nonlinearity, heterogeneity within a coarse cell, or a
    topology whose edges do not survive coarsening (ARCHITECTURE.md §1).
    """
    rules: tuple[ResolutionRule, ...] = ()
    default_mm: float = 2.0
    default_band: Band = Band(0.0, 100.0)


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


class Provenance(str, Enum):
    PHYSICS = "physics"
    LITERATURE = "literature"
    ATLAS = "atlas"
    FIT = "fit"
    DISTILLED = "distilled"
    WEAK = "weak"
    SPECULATIVE = "speculative"


class Tying(str, Enum):
    """how an implementation's parameters are shared across materialized sites.

    tying determines how many effective parameters exist and therefore what data
    could move them off their prior.  it does not determine what exists: a
    per-site parameterization over 10^5 positions is a legitimate declaration
    even when no dataset can distinguish its entries -- the posterior simply
    stays near the prior and the structure comes out smooth.
    """
    GLOBAL = "global"
    PER_PARTITION = "per_partition"
    EMBEDDING = "embedding"
    PER_SITE = "per_site"


@dataclass(frozen=True)
class Prior:
    dist: str                    # normal | lognormal | halfnormal | uniform | beta | dirichlet
    loc: float = 0.0
    scale: float = 1.0
    units: str = ""
    provenance: Provenance = Provenance.WEAK
    source: str = ""
    note: str = ""


def lognormal(median: float, spread: float, **kw) -> Prior:
    """positive parameter known to within a multiplicative factor.

    the natural prior for every rate constant, conductance and time constant in
    the inventory: all are positive, and the literature reports them to within a
    factor rather than an increment.
    """
    return Prior("lognormal", math.log(median), math.log(spread), **kw)


def normal(mean: float, sd: float, **kw) -> Prior:
    return Prior("normal", mean, sd, **kw)


def uniform(lo: float, hi: float, **kw) -> Prior:
    return Prior("uniform", lo, hi, **kw)


def weak(median: float = 1.0, spread: float = 10.0, **kw) -> Prior:
    """explicitly uninformative: the dynamics exist but the science does not pin them."""
    kw.setdefault("provenance", Provenance.WEAK)
    return lognormal(median, spread, **kw)


def speculative(median: float = 1.0, spread: float = 30.0, **kw) -> Prior:
    """the functional form itself is a guess, not only its parameters."""
    kw.setdefault("provenance", Provenance.SPECULATIVE)
    return lognormal(median, spread, **kw)


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Validity:
    """the regime in which a process's form is meaningful.

    every process reachable from a target is materialized, so resolution and
    bandwidth are the only levers -- and processes will routinely be asked to run
    outside the regime their form was written for.  the materializer records
    every violation in provenance rather than silently complying.
    """
    min_spacing_mm: float = 0.0
    max_spacing_mm: float = math.inf
    band: Band = FULL
    note: str = ""

    def violations(self, spacing_mm: float, band: Band) -> list[str]:
        out = []
        if spacing_mm < self.min_spacing_mm:
            out.append(f"materialized at {spacing_mm} mm, finer than the {self.min_spacing_mm} mm "
                       "floor at which this form is meaningful")
        if spacing_mm > self.max_spacing_mm:
            out.append(f"materialized at {spacing_mm} mm, coarser than the {self.max_spacing_mm} mm "
                       "ceiling at which this form is meaningful")
        if not self.band.covers(band):
            out.append(f"materialized over {band}, outside the validated {self.band}")
        return out
