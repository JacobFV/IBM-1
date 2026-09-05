"""assembling p(theta) for a materialized model.

    p(theta | D) proportional to p(theta) prod_d p(D_d | theta)

this module is the first factor.  nothing here fits anything; it collects the
priors every selected implementation already declared and lays them out as one
parameter space with an index, so that `ibm.forge.fit` has a vector to move and a
log-density to move it against.

three things follow from ARCHITECTURE.md and shape everything below.

**a prior is not an initializer.**  §4 says different initialization methods
simply produce different priors: physics, literature, a fit, or `weak()`.  so the
provenance of every entry survives into the parameter space and out the other
side, and a report can say which parameters the corpus could ever have moved.
the failure this prevents is a model whose smooth, plausible cortical map is
entirely the prior and is presented as a finding.

**tying is what determines how many parameters exist.**  §4 again: theta may be
shared globally, per anatomical partition, as a function of a learned embedding,
or held per position, and that choice determines what data could move it off its
prior.  it does *not* determine what exists -- a per-site parameterization over
10^5 cortical positions is a legitimate declaration even when no dataset can
distinguish its entries, and the right outcome is that the posterior stays near
the prior and the structure comes out smooth.  so `Tying` expands here, honestly,
and the expansion is reported rather than quietly capped.

**everything is fitted in an unconstrained space.**  every rate constant,
conductance and time constant in the inventory is positive and reported to within
a multiplicative factor, which is why `ibm.vocabulary.lognormal` is the house
prior.  optimizing those directly hits the boundary at zero; optimizing their
logs does not.  the transform is declared per distribution family, with its log
jacobian, so the density in the sampling space is the density the fit actually
sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ibm.registry import REGISTRY, Implementation, Process
from ibm.vocabulary import Anat, Prior, Provenance, Tying

_EPS = 1e-30
_LOG2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transform:
    """a map from an unconstrained real vector to the parameter's own support.

    named rather than implicit because the log jacobian is easy to forget and its
    absence is invisible: a MAP estimate without it is biased towards the
    boundary of the support, and a variational fit without it reports a posterior
    that integrates to something other than one.
    """

    name: str
    to_natural: Any
    to_unconstrained: Any
    log_det_jacobian: Any


def _identity() -> Transform:
    return Transform("identity", lambda u: u, lambda x: x, lambda u: np.zeros_like(u))


def _exp() -> Transform:
    return Transform("exp", np.exp, lambda x: np.log(np.maximum(x, _EPS)), lambda u: u)


def _logit(lo: float, hi: float) -> Transform:
    span = hi - lo
    return Transform(
        f"logit[{lo:g},{hi:g}]",
        lambda u: lo + span / (1.0 + np.exp(-u)),
        lambda x: np.log(np.clip((x - lo) / span, _EPS, 1 - 1e-12)
                         / (1.0 - np.clip((x - lo) / span, _EPS, 1 - 1e-12))),
        lambda u: np.log(span) - u - 2.0 * np.log1p(np.exp(-u)),
    )


def transform_for(p: Prior) -> Transform:
    if p.dist in ("lognormal", "halfnormal", "gamma", "exponential"):
        return _exp()
    if p.dist == "uniform":
        return _logit(p.loc, p.scale)
    if p.dist in ("beta", "dirichlet"):
        return _logit(0.0, 1.0)
    return _identity()


def log_prior(p: Prior, x: np.ndarray) -> np.ndarray:
    """log p(theta) in the *natural* parameterization.

    only the families `ibm.vocabulary` actually builds are implemented, and an
    unknown family returns an improper flat density rather than a guess -- a
    silently wrong prior is worse than an explicitly absent one, and the
    parameter space reports which entries fell through.
    """
    x = np.asarray(x, float)
    if p.dist == "normal":
        return -0.5 * (((x - p.loc) / max(p.scale, _EPS)) ** 2 + _LOG2PI) - math.log(max(p.scale, _EPS))
    if p.dist == "lognormal":
        lx = np.log(np.maximum(x, _EPS))
        return (-0.5 * (((lx - p.loc) / max(p.scale, _EPS)) ** 2 + _LOG2PI)
                - math.log(max(p.scale, _EPS)) - lx)
    if p.dist == "halfnormal":
        return np.where(x > 0, -0.5 * (x / max(p.scale, _EPS)) ** 2
                        - math.log(max(p.scale, _EPS)) + 0.5 * math.log(2.0 / math.pi), -np.inf)
    if p.dist == "uniform":
        span = max(p.scale - p.loc, _EPS)
        return np.where((x >= p.loc) & (x <= p.scale), -math.log(span), -np.inf)
    if p.dist == "beta":
        a, b = max(p.loc, _EPS), max(p.scale, _EPS)
        y = np.clip(x, _EPS, 1 - 1e-12)
        return ((a - 1) * np.log(y) + (b - 1) * np.log1p(-y)
                - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))
    return np.zeros_like(x)


def median_of(p: Prior) -> float:
    if p.dist == "lognormal":
        return math.exp(p.loc)
    if p.dist == "uniform":
        return 0.5 * (p.loc + p.scale)
    if p.dist == "halfnormal":
        return 0.6745 * p.scale
    if p.dist == "beta":
        return p.loc / max(p.loc + p.scale, _EPS)
    return p.loc


def sd_of(p: Prior) -> float:
    """the prior's standard deviation in its own natural units.

    the companion to `median_of`, and it exists for the same reason that one
    does: ARCHITECTURE.md 4 says the induced state distribution depends on p(x)
    **and** p(theta), and everything downstream of a build was reading the median
    and throwing the width away.  a first-order push-forward of parameter
    uncertainty needs exactly one number per parameter, and this is it.

    for a lognormal -- the house prior, because every rate constant and time
    constant in the inventory is positive and reported to within a factor -- the
    sd is `median sqrt(exp(s^2) - 1) exp(s^2/2)`, which for the `weak(1, 10)`
    priors most of the inventory carries is many times the median.  that is not a
    bug in the arithmetic; it is what "the science does not pin this" means, and a
    propagation that reports a huge width off a `weak()` prior is reporting the
    truth about the prior rather than a failure of the dynamics.
    """
    if p.dist == "normal":
        return float(p.scale)
    if p.dist == "lognormal":
        v = float(p.scale) ** 2
        return float(math.exp(p.loc + 0.5 * v) * math.sqrt(max(math.expm1(v), 0.0)))
    if p.dist == "halfnormal":
        return float(p.scale) * math.sqrt(max(1.0 - 2.0 / math.pi, 0.0))
    if p.dist == "uniform":
        return abs(float(p.scale) - float(p.loc)) / math.sqrt(12.0)
    if p.dist == "beta":
        a, b = max(float(p.loc), _EPS), max(float(p.scale), _EPS)
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))
    # an unknown family gets zero rather than a guess: a fabricated width would
    # be propagated as though it were declared, and the parameter space already
    # reports which entries fell through.
    return 0.0


def sample_prior(p: Prior, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    if p.dist == "normal":
        return p.loc + p.scale * rng.standard_normal(shape)
    if p.dist == "lognormal":
        return np.exp(p.loc + p.scale * rng.standard_normal(shape))
    if p.dist == "halfnormal":
        return np.abs(p.scale * rng.standard_normal(shape))
    if p.dist == "uniform":
        return rng.uniform(p.loc, p.scale, shape)
    if p.dist == "beta":
        return rng.beta(max(p.loc, _EPS), max(p.scale, _EPS), shape)
    return np.full(shape, median_of(p))


# ---------------------------------------------------------------------------
# one parameter, expanded by its tying policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterBlock:
    """one declared parameter of one implementation, expanded over its sites.

    `n` is the count after tying.  it is the number that answers "what could data
    move here": one global conductance is identifiable from a handful of
    recordings, one per cortical position over 10^5 positions is not, and both
    are legitimate declarations of what exists.  reporting `n` beside the
    provenance is what stops the second being mistaken for the first.
    """

    process: str
    implementation: str
    name: str
    prior: Prior
    tying: Tying
    n: int
    #: what each entry is indexed by: partition labels, site ids, embedding
    #: dimensions, or `("global",)`.  carried so a posterior can be reported
    #: against something a person recognizes rather than against an offset.
    index: tuple[str, ...] = ()
    offset: int = 0

    @property
    def key(self) -> str:
        return f"{self.process}:{self.implementation}.{self.name}"

    @property
    def slice(self) -> slice:
        return slice(self.offset, self.offset + self.n)

    @property
    def identifiable(self) -> bool:
        """a soft flag, not a gate.

        per-site and embedding tyings are routinely under-determined by the whole
        corpus, and that is allowed.  what is not allowed is presenting the
        resulting smoothness as a finding, so this is what `describe` prints and
        what `ibm.forge.fit` copies into provenance.
        """
        return self.tying in (Tying.GLOBAL, Tying.PER_PARTITION)


# ---------------------------------------------------------------------------
# the space
# ---------------------------------------------------------------------------


@dataclass
class ParameterSpace:
    """every parameter a materialized model has, laid out as one vector.

    flat, because the optimizers and samplers in `fit.py` want a vector, and
    blocked, because everything a human asks of it -- which process, which
    provenance, what tying, could this ever have been identified -- is per block.
    the flat layout is an implementation detail of fitting in the same way that
    `State.pack_means` is an implementation detail of solving.
    """

    blocks: tuple[ParameterBlock, ...] = ()
    missing_family: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return sum(b.n for b in self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def __getitem__(self, key: str) -> ParameterBlock:
        for b in self.blocks:
            if b.key == key or b.name == key:
                return b
        raise KeyError(f"no parameter {key!r} in this space")

    # -- vectors ---------------------------------------------------------

    def median(self) -> np.ndarray:
        v = np.zeros(self.size)
        for b in self.blocks:
            v[b.slice] = median_of(b.prior)
        return v

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        v = np.zeros(self.size)
        for b in self.blocks:
            v[b.slice] = sample_prior(b.prior, (b.n,), rng)
        return v

    def log_prior(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, float)
        return float(sum(np.sum(log_prior(b.prior, theta[b.slice])) for b in self.blocks))

    # -- unconstrained space ---------------------------------------------

    def to_unconstrained(self, theta: np.ndarray) -> np.ndarray:
        u = np.empty_like(np.asarray(theta, float))
        for b in self.blocks:
            u[b.slice] = transform_for(b.prior).to_unconstrained(theta[b.slice])
        return u

    def to_natural(self, u: np.ndarray) -> np.ndarray:
        theta = np.empty_like(np.asarray(u, float))
        for b in self.blocks:
            theta[b.slice] = transform_for(b.prior).to_natural(u[b.slice])
        return theta

    def log_prior_unconstrained(self, u: np.ndarray) -> float:
        """log p(theta(u)) + log |d theta / du|.

        the jacobian term is not optional.  without it a MAP fit of a lognormal
        rate constant drifts towards zero for no reason other than the coordinate
        change, and the resulting "fitted" time constant is an artefact of the
        optimizer's parameterization.
        """
        u = np.asarray(u, float)
        tot = 0.0
        for b in self.blocks:
            t = transform_for(b.prior)
            x = t.to_natural(u[b.slice])
            tot += float(np.sum(log_prior(b.prior, x)))
            tot += float(np.sum(t.log_det_jacobian(u[b.slice])))
        return tot

    # -- unpacking -------------------------------------------------------

    def unpack(self, theta: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """theta as `{process:implementation: {param: values}}`, which is what an f wants."""
        out: dict[str, dict[str, np.ndarray]] = {}
        for b in self.blocks:
            out.setdefault(f"{b.process}:{b.implementation}", {})[b.name] = theta[b.slice]
        return out

    # -- reporting -------------------------------------------------------

    def describe(self) -> str:
        rows = [(b.process, b.implementation, b.name, b.prior.dist, b.tying.value,
                 f"{b.n:,}", b.prior.provenance.value, "" if b.identifiable else "under")
                for b in self.blocks]
        head = ("process", "impl", "param", "dist", "tying", "n", "prov", "id?")
        w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
             for i, h in enumerate(head)]
        out = ["  ".join(h.ljust(x) for h, x in zip(head, w)),
               "  ".join("-" * x for x in w)]
        out += ["  ".join(c.ljust(x) for c, x in zip(r, w)) for r in rows]
        weak = sum(b.n for b in self.blocks
                   if b.prior.provenance in (Provenance.WEAK, Provenance.SPECULATIVE))
        under = sum(b.n for b in self.blocks if not b.identifiable)
        out.append(f"{self.size:,} parameters over {len(self.blocks)} blocks; "
                   f"{weak:,} ({weak / max(self.size, 1):.0%}) sit on weak or speculative "
                   f"priors, {under:,} ({under / max(self.size, 1):.0%}) are per-site or "
                   "embedding-tied and will mostly stay where the prior put them")
        if self.missing_family:
            out.append("no density implemented for: " + ", ".join(self.missing_family))
        return "\n".join(out)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def expand(tying: Tying, impl: Implementation, *, n_sites: int = 1,
           partitions: Sequence[str] = (), embedding_dim: int = 16
           ) -> tuple[int, tuple[str, ...]]:
    """how many entries one declared parameter becomes, and what indexes them.

    the four cases are exactly §4's four, and none of them is capped.  a per-site
    expansion over a fine materialization is enormous and is allowed to be: the
    declaration says what exists, and the posterior staying near the prior is the
    correct outcome, not a failure to be avoided by silently sharing parameters
    the model said were not shared.
    """
    if tying is Tying.GLOBAL:
        return 1, ("global",)
    if tying is Tying.PER_PARTITION:
        labels = tuple(partitions) or ("unpartitioned",)
        return len(labels), labels
    if tying is Tying.EMBEDDING:
        return embedding_dim, tuple(f"e{i}" for i in range(embedding_dim))
    return max(n_sites, 1), tuple(f"site{i}" for i in range(max(n_sites, 1)))


def partitions_for(impl: Implementation, process: Process | None) -> tuple[str, ...]:
    """labels a PER_PARTITION tying expands over.

    read from whichever anatomical partitioning system the process's selectors
    actually name.  a process whose region is `Anat("cortical_layers", "iv")` is
    tied over cortical layers and nothing else; a process that names no
    partitioning system has nothing to be per-partition over, and gets one
    entry with a label that says so rather than a silent collapse to global.
    """
    if process is None:
        return ()
    systems: list[str] = []
    for s in process.inputs + process.outputs:
        r = s.region
        for part in (getattr(r, "parts", ()) or (r,)):
            if isinstance(part, Anat):
                systems.append(part.system)
    labels: list[str] = []
    for name in dict.fromkeys(systems):
        a = REGISTRY.anatomies.get(name)
        if a is not None:
            labels += [f"{name}/{x}" for x in a.labels]
    return tuple(labels)


def assemble(model: Any = None, *, implementations: Iterable[Implementation] | None = None,
             sites: Mapping[str, int] | None = None, embedding_dim: int = 16
             ) -> ParameterSpace:
    """collect p(theta) for a materialized model, or for a set of implementations.

    the model argument is optional so that the whole ontology's prior can be
    assembled and printed before anything is materialized -- which is the honest
    way to answer "how much of this is actually pinned by the literature", and
    the answer is usually sobering.
    """
    impls = list(implementations) if implementations is not None else _selected(model)
    site_counts = dict(sites or _site_counts(model))
    blocks: list[ParameterBlock] = []
    missing: set[str] = set()
    offset = 0
    known = {"normal", "lognormal", "halfnormal", "uniform", "beta"}
    for impl in sorted(impls, key=lambda i: (i.process, i.name)):
        proc = REGISTRY.processes.get(impl.process)
        parts = partitions_for(impl, proc)
        n_sites = 1
        if proc is not None:
            n_sites = max((site_counts.get(v, 1) for s in proc.outputs for v in s.vars),
                          default=1)
        for name, prior in impl.params.items():
            if prior.dist not in known:
                missing.add(f"{impl.process}:{impl.name}.{name} ({prior.dist})")
            n, index = expand(impl.tying, impl, n_sites=n_sites, partitions=parts,
                              embedding_dim=embedding_dim)
            blocks.append(ParameterBlock(impl.process, impl.name, name, prior, impl.tying,
                                         n, index, offset))
            offset += n
    return ParameterSpace(tuple(blocks), tuple(sorted(missing)))


def _selected(model: Any) -> list[Implementation]:
    """implementations this materialization actually chose.

    §7 says the materialized model records which f was selected per process.  a
    model that does not expose that is treated as having selected every declared
    implementation, which over-counts the parameter space -- and the over-count is
    the safe direction, because it reports more unidentified parameters than
    there are rather than fewer.
    """
    if model is None:
        return list(REGISTRY.implementations.values())
    for attr in ("implementations", "selected", "impls"):
        v = getattr(model, attr, None)
        if v:
            out = []
            for x in (v.values() if isinstance(v, dict) else v):
                out.append(x if isinstance(x, Implementation)
                           else REGISTRY.implementations.get(str(x)))
            return [x for x in out if x is not None]
    return list(REGISTRY.implementations.values())


def _site_counts(model: Any) -> dict[str, int]:
    if model is None:
        return {}
    try:
        from ibm.runtime.state import Layout
        return {b.component: b.n_sites for b in Layout.of(model)}
    except Exception:
        return {}


__all__ = [
    "ParameterBlock", "ParameterSpace", "Transform", "assemble", "expand", "log_prior",
    "median_of", "partitions_for", "sample_prior", "sd_of", "transform_for",
]
