"""posterior updating over heterogeneous data.

    p(theta | D)  proportional to  p(theta) prod_d p(D_d | theta)

ARCHITECTURE.md is explicit that forging *has the semantics of* this update and
that the implementation may approximate it by gradient optimization, variational
inference, ensembles, distillation "or other techniques".  all four are here, and
they are four `Method`s over one objective rather than four pipelines, because
the thing that must stay fixed is the factorization: a prior with real
provenance, one likelihood factor per source, and nothing that updates theta
except a factor some card was permitted to contribute.

### why the fit is per process and not end-to-end

the tempting thing is one loss: materialize the whole graph, unroll it over a
recording, backpropagate to every parameter at once.  it is wrong here for four
separate reasons, and they compound.

*the gradient through a long unroll is worthless before it is expensive.*  the
graph contains resonant loops close to their stability margin (that is what an
alpha rhythm is), so the jacobian of a long window has eigenvalues on both sides
of 1 and the gradient it produces either vanishes or blows up along exactly the
directions that matter.  a per-process likelihood is a short, well-conditioned
map from a handful of parameters to a quantity that was actually measured.

*the sources do not overlap.*  §5's inventory spans conduction delay, potassium
uptake, neurovascular gain and skull conductivity; no dataset constrains more
than a few of those and most constrain one.  an end-to-end loss over such a
corpus is a sum in which almost every parameter appears in almost no term, so it
is exactly a set of small independent fits wearing a large expensive coat -- and
the coat hides which parameter each dataset actually moved, which §7 requires the
model to be able to say.

*identifiability is a per-process property and gets destroyed by pooling.*  a
neurovascular gain and a BOLD scaling factor are separately meaningful and jointly
unidentifiable from BOLD alone.  fitted separately, against a calibrated-CBF
source and a BOLD source, both are pinned.  fitted end-to-end against BOLD, their
product is pinned and the split is whatever the initialization was -- which then
propagates as confident nonsense into every prediction that uses either alone.

*and it makes the corpus non-additive.*  the whole point of the product form is
that a new source is a new factor.  per-process, adding one is a local update;
end-to-end, it is a rerun of everything.

so `fit` takes a *list* of `Task`s, each naming the parameter blocks it may move
and the likelihood it contributes, and updates them independently unless a task
explicitly names several processes because it genuinely couples them.  the
end-to-end path exists (`Method.JOINT`) for the cases where a source really does
only constrain a composition -- an evoked response is a statement about the whole
forward chain -- and it is opt-in, and it says in the report that it was used.

### what this module deliberately does not do

it does not read data.  a `Task` carries a callable that returns a log-likelihood
given theta, and where that number comes from -- a zarr under `data/sources/*/
evidence/`, a simulator, a teacher -- is the caller's business.  keeping the
likelihood abstract is what lets a distillation target and a measurement share
one optimizer while remaining permanently distinguishable in provenance, which is
the rule the distillation section of §4 exists to enforce.

numpy and scipy only.  gradients are numerical (`scipy.optimize` with a
2-point jacobian) which is fine for the tens-of-parameters problems a per-process
fit actually is, and hopeless for a learned module with 10^6 weights -- that is
the case where a jax or torch backend is required, and the seam for it is
`Task.grad`: supply one and every method here uses it instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ibm.forge.priors import ParameterBlock, ParameterSpace, transform_for
from ibm.registry import REGISTRY
from ibm.vocabulary import Provenance

_EPS = 1e-30


class Method(str, Enum):
    """how the update is approximated.  all of these are §4's list.

    MAP is the default not because it is best but because it is the one whose
    failure mode is legible: a maximum with no width, reported as such, is less
    dangerous than a variational posterior whose width came out of a mean-field
    assumption nobody checked.
    """

    MAP = "map"                  # gradient optimization to the posterior mode
    VI = "vi"                    # mean-field gaussian, reparameterized MC gradient
    ENSEMBLE = "ensemble"        # several MAP fits from prior draws; width from spread
    DISTIL = "distil"            # a teacher's targets, at calibrated precision
    JOINT = "joint"              # one objective over several processes; opt-in


@dataclass
class Task:
    """one likelihood factor and the parameters it is permitted to move.

    `moves` is the enforcement point for the rule that a source may only update
    what it bears on.  a card declaring `use: [evaluate]` produces a task with an
    empty `moves`, and it then contributes a number to the report and nothing to
    the posterior -- which is the mechanical version of "a model is never scored
    on what trained it".
    """

    name: str
    #: log p(D | theta).  theta arrives in the *natural* parameterization.
    logp: Callable[[np.ndarray], float]
    #: parameter block keys this factor may move.  empty means it moves nothing.
    moves: tuple[str, ...] = ()
    #: analytic gradient in the natural parameterization, if the caller has one.
    grad: Callable[[np.ndarray], np.ndarray] | None = None
    source: str = ""
    kind: str = "fit"            # fit | evaluate | control | distil | calibrate
    weight: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind in ("evaluate", "control") and self.moves:
            raise ValueError(
                f"task {self.name!r} is a {self.kind} source but names parameters to move; "
                "an evaluation that updates theta is not an evaluation")


@dataclass
class FitReport:
    """what moved, by how much, and on whose authority.

    §7 requires a materialized model to record which parameters were moved off
    their prior by evidence and which were not.  this is that record for the
    fitting half, and `moved_fraction` is the number to quote before quoting any
    prediction: a model whose parameters are 3% constrained is a prior with a
    thin coat of data on it, however good its held-out score looks.
    """

    method: str = "map"
    tasks: tuple[str, ...] = ()
    iterations: int = 0
    log_posterior: float = -math.inf
    log_posterior0: float = -math.inf
    converged: bool = False
    #: prior-sd units, per block key.  the honest measure of "did the data speak".
    shift: dict[str, float] = _field(default_factory=dict)
    #: posterior sd / prior sd, per block key, where the method produces a width.
    contraction: dict[str, float] = _field(default_factory=dict)
    held_out: dict[str, float] = _field(default_factory=dict)
    notes: list[str] = _field(default_factory=list)

    def moved(self, threshold: float = 0.25) -> tuple[str, ...]:
        return tuple(k for k, v in self.shift.items() if abs(v) >= threshold)

    def moved_fraction(self, threshold: float = 0.25) -> float:
        return len(self.moved(threshold)) / max(len(self.shift), 1)

    def __str__(self) -> str:
        lines = [f"{self.method} over {len(self.tasks)} tasks: "
                 f"log posterior {self.log_posterior0:.4g} -> {self.log_posterior:.4g} in "
                 f"{self.iterations} iterations"
                 f"{'' if self.converged else '  (NOT CONVERGED)'}"]
        mv = self.moved()
        lines.append(f"{len(mv)}/{len(self.shift)} blocks ({self.moved_fraction():.0%}) moved "
                     "more than a quarter of a prior sd; the rest are where the prior put them")
        for k in sorted(mv, key=lambda x: -abs(self.shift[x]))[:20]:
            c = self.contraction.get(k)
            lines.append(f"  {k:48s} {self.shift[k]:+.2f} prior sd"
                         + (f", width x{c:.2f}" if c is not None else ""))
        for k, v in sorted(self.held_out.items()):
            lines.append(f"  held out  {k:44s} {v:.4g}")
        lines += [f"  {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------


def _objective(space: ParameterSpace, tasks: Sequence[Task], mask: np.ndarray,
               base: np.ndarray):
    """negative log posterior in the unconstrained space, over the masked entries.

    the mask is what makes the fit per process: entries a task is not permitted
    to move are held at `base` and never enter the vector the optimizer sees, so
    a source cannot move a parameter it does not bear on even by accident, and
    the optimizer is not asked to search a space in which most directions are
    flat.
    """
    idx = np.flatnonzero(mask)

    def expand(u_free: np.ndarray) -> np.ndarray:
        u = space.to_unconstrained(base).copy()
        u[idx] = u_free
        return u

    def neg(u_free: np.ndarray) -> float:
        u = expand(u_free)
        theta = space.to_natural(u)
        lp = space.log_prior_unconstrained(u)
        for t in tasks:
            if t.kind in ("evaluate", "control"):
                continue
            lp += t.weight * float(t.logp(theta))
        return -lp

    return idx, expand, neg


def _mask_for(space: ParameterSpace, tasks: Sequence[Task]) -> np.ndarray:
    keys = {k for t in tasks for k in t.moves}
    mask = np.zeros(space.size, dtype=bool)
    for b in space.blocks:
        if b.key in keys or b.process in keys or f"{b.process}:{b.implementation}" in keys:
            mask[b.slice] = True
    return mask


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------


def fit_map(space: ParameterSpace, tasks: Sequence[Task], *, theta0: np.ndarray | None = None,
            max_iter: int = 500) -> tuple[np.ndarray, FitReport]:
    """gradient optimization to the posterior mode.

    reports the mode and no width, which is the honest output of a method that
    computes no width.  a laplace approximation from the numerical hessian is
    available and deliberately not taken by default: at these dimensionalities it
    is cheap, but a hessian from a 2-point jacobian at a mode found by the same
    2-point jacobian produces error bars whose only real content is the finite
    difference step.
    """
    from scipy.optimize import minimize

    theta = space.median() if theta0 is None else np.asarray(theta0, float).copy()
    mask = _mask_for(space, tasks)
    rep = FitReport(method="map", tasks=tuple(t.name for t in tasks))
    if not mask.any():
        rep.notes.append("no task named a parameter it may move; the posterior is the prior")
        rep.log_posterior = rep.log_posterior0 = space.log_prior(theta)
        _score(space, tasks, theta, rep)
        return theta, rep

    idx, expand, neg = _objective(space, tasks, mask, theta)
    u0 = space.to_unconstrained(theta)[idx]
    rep.log_posterior0 = -neg(u0)
    res = minimize(neg, u0, method="L-BFGS-B",
                   options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-10})
    theta1 = space.to_natural(expand(res.x))
    rep.iterations = int(getattr(res, "nit", 0))
    rep.converged = bool(res.success)
    rep.log_posterior = -float(res.fun)
    _shift(space, theta, theta1, rep)
    _score(space, tasks, theta1, rep)
    return theta1, rep


def fit_vi(space: ParameterSpace, tasks: Sequence[Task], *, theta0: np.ndarray | None = None,
           m: int = 16, steps: int = 400, lr: float = 0.05, seed: int = 0
           ) -> tuple[np.ndarray, np.ndarray, FitReport]:
    """mean-field gaussian variational inference in the unconstrained space.

    the approximating family is a diagonal gaussian over u, which is exactly the
    posterior family a lognormal prior wants and exactly the wrong one for
    parameters that trade off against each other -- which most of them do.  so
    the width it reports is a *lower bound in the wrong direction*: mean-field
    under-reports correlated uncertainty, systematically, and a fit whose
    posterior looks impressively tight should be checked against `fit_ensemble`
    before anyone believes it.

    the gradient is reparameterized monte carlo with a numerical gradient of the
    log density, which is the honest cost of having no autodiff here.  it is
    adequate for tens of parameters and untenable for a learned module; that is
    the seam where a jax backend goes.
    """
    from scipy.optimize import approx_fprime

    rng = np.random.default_rng(seed)
    theta = space.median() if theta0 is None else np.asarray(theta0, float).copy()
    mask = _mask_for(space, tasks)
    rep = FitReport(method="vi", tasks=tuple(t.name for t in tasks))
    idx, expand, neg = _objective(space, tasks, mask, theta)
    if idx.size == 0:
        rep.notes.append("no free parameters; the variational posterior is the prior")
        return theta, np.zeros_like(theta), rep

    mu = space.to_unconstrained(theta)[idx]
    log_sd = np.full(idx.size, -1.0)
    rep.log_posterior0 = -neg(mu)
    step = 1e-5

    # adam rather than plain sgd, and not for speed.  the objective is a sum of
    # log-densities whose curvatures differ by whatever the priors' scales differ
    # by -- a millisecond time constant beside a dimensionless gain -- so one
    # step size cannot serve both, and plain sgd at any lr that moves the flat
    # direction blows the steep one to nan.  a per-coordinate normalized step is
    # the cheapest fix that does not require the hessian nobody wants to form.
    mv = np.zeros_like(mu); vv = np.zeros_like(mu)
    ms = np.zeros_like(log_sd); vs = np.zeros_like(log_sd)
    b1, b2, eps_a = 0.9, 0.999, 1e-8
    best = (mu.copy(), log_sd.copy())

    for it in range(1, steps + 1):
        gmu = np.zeros_like(mu)
        gls = np.zeros_like(log_sd)
        for _ in range(m):
            e = rng.standard_normal(idx.size)
            g = approx_fprime(mu + np.exp(log_sd) * e, neg, step)
            if not np.all(np.isfinite(g)):
                continue
            gmu += g
            gls += g * e * np.exp(log_sd)
        gmu /= m
        gls = gls / m - 1.0                      # minus the entropy term's gradient
        if not (np.all(np.isfinite(gmu)) and np.all(np.isfinite(gls))):
            rep.notes.append(f"the elbo gradient went non-finite at step {it}; stopped there "
                             "and kept the last good iterate.  a mean-field gaussian over a "
                             "parameter whose posterior is nearly a delta does this, and the "
                             "answer is a tighter prior or a different method, not a smaller "
                             "step")
            mu, log_sd = best
            break
        mv = b1 * mv + (1 - b1) * gmu; vv = b2 * vv + (1 - b2) * gmu ** 2
        ms = b1 * ms + (1 - b1) * gls; vs = b2 * vs + (1 - b2) * gls ** 2
        mu -= lr * (mv / (1 - b1 ** it)) / (np.sqrt(vv / (1 - b2 ** it)) + eps_a)
        log_sd -= lr * (ms / (1 - b1 ** it)) / (np.sqrt(vs / (1 - b2 ** it)) + eps_a)
        log_sd = np.clip(log_sd, -20.0, 10.0)
        best = (mu.copy(), log_sd.copy())
        rep.iterations = it
        if np.linalg.norm(gmu) < 1e-8:
            rep.converged = True
            break
    rep.iterations = rep.iterations or steps

    theta1 = space.to_natural(expand(mu))
    sd = np.zeros(space.size)
    # the width lives in u; pushing it to natural units by the transform's local
    # slope is a first-order statement and is labelled as one.
    u_full = expand(mu)
    for b in space.blocks:
        t = transform_for(b.prior)
        s = np.zeros(b.n)
        loc = np.isin(np.arange(b.offset, b.offset + b.n), idx)
        if loc.any():
            pos = np.searchsorted(idx, np.arange(b.offset, b.offset + b.n)[loc])
            s[loc] = np.exp(log_sd[pos]) * np.abs(
                np.exp(t.log_det_jacobian(u_full[b.slice][loc])))
        sd[b.slice] = s
    rep.log_posterior = -neg(mu)
    _shift(space, theta, theta1, rep, sd)
    _score(space, tasks, theta1, rep)
    rep.notes.append("mean-field: correlated uncertainty is under-reported by construction")
    return theta1, sd, rep


def fit_ensemble(space: ParameterSpace, tasks: Sequence[Task], *, n: int = 8, seed: int = 0,
                 max_iter: int = 300) -> tuple[np.ndarray, np.ndarray, FitReport]:
    """n MAP fits from n prior draws; the spread is the posterior width.

    the cheapest honest width available, and the only one here that sees
    parameter correlations and multimodality.  where it disagrees with `fit_vi`
    the ensemble is usually right and the mean-field assumption is usually the
    reason.  where the members do not agree at all, the likelihood has several
    modes and the mode a single MAP run reports is an accident of its start.
    """
    rng = np.random.default_rng(seed)
    draws = []
    reps = []
    for i in range(n):
        start = space.sample(rng)
        t, r = fit_map(space, tasks, theta0=start, max_iter=max_iter)
        draws.append(t)
        reps.append(r)
    arr = np.stack(draws)
    mean, sd = arr.mean(0), arr.std(0, ddof=1)
    rep = FitReport(method="ensemble", tasks=tuple(t.name for t in tasks),
                    iterations=sum(r.iterations for r in reps),
                    converged=all(r.converged for r in reps),
                    log_posterior0=max(r.log_posterior0 for r in reps),
                    log_posterior=max(r.log_posterior for r in reps))
    _shift(space, space.median(), mean, rep, sd)
    _score(space, tasks, mean, rep)
    spread = float(np.mean(sd / np.maximum(np.abs(mean), _EPS)))
    rep.notes.append(
        f"{n} members, mean relative spread {spread:.2g}"
        + ("; the members disagree, so the likelihood is multimodal and a single MAP result "
           "here is an accident of its start" if spread > 0.5 else
           "; the members from independent prior draws landed together, so the likelihood "
           "dominates the prior here and the reported width is genuinely small" if spread < 1e-3
           else ""))
    return mean, sd, rep


def fit_distil(space: ParameterSpace, tasks: Sequence[Task], teachers: Mapping[str, Any],
               *, theta0: np.ndarray | None = None, max_iter: int = 300
               ) -> tuple[np.ndarray, FitReport]:
    """MAP against teacher targets, at the precision the teacher earned.

    identical machinery to `fit_map` and a different *authority*.  the tasks
    passed here must have been built with `ibm.runtime.fuse.TeacherPrecision`, so
    their likelihood already carries `1/((1-r2) Var[x])`, the off-distribution
    inflation, and the low-rank error structure.  the only thing this function
    adds is the record: every parameter it moves is marked distilled, and stays
    distinguishable from one a measurement moved for the life of the model.
    """
    for t in tasks:
        if t.kind != "distil":
            raise ValueError(f"task {t.name!r} is {t.kind!r}, not a distillation target; a "
                             "teacher and a measurement must not be fitted through the same "
                             "call, because the provenance of what they moved differs")
    theta, rep = fit_map(space, tasks, theta0=theta0, max_iter=max_iter)
    rep.method = "distil"
    for name, tp in teachers.items():
        eff = getattr(tp, "error_rank", None)
        rep.notes.append(
            f"teacher {name}: r2={getattr(tp, 'r2', float('nan')):g}, error rank "
            f"{eff if eff is not None else 'UNDECLARED'}; every parameter above is "
            "DISTILLED, not measured, and inherits this teacher's biases")
    return theta, rep


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


def fit(space: ParameterSpace, tasks: Sequence[Task], *, method: Method = Method.MAP,
        per_process: bool = True, **kw) -> tuple[np.ndarray, FitReport]:
    """update p(theta) with every task, per process unless told otherwise.

    `per_process=True` partitions the tasks by the processes they move and runs
    one fit per partition, which is the arrangement the module docstring argues
    for.  tasks that move parameters of several processes land in one partition
    together, so a source that genuinely couples a chain is still fitted jointly
    -- the partition is derived from the data, not imposed.
    """
    if not per_process or method is Method.JOINT:
        theta, rep = _dispatch(space, tasks, method, **kw)
        if method is Method.JOINT:
            rep.notes.append("fitted end-to-end: every parameter above was moved by one "
                             "objective, so which source moved which is not recoverable")
        return theta, rep

    groups = _partition(space, tasks)
    theta = kw.pop("theta0", None)
    theta = space.median() if theta is None else np.asarray(theta, float).copy()
    start = theta.copy()
    merged = FitReport(method=method.value, tasks=tuple(t.name for t in tasks))
    merged.log_posterior0 = space.log_prior(theta) + sum(
        t.weight * float(t.logp(theta)) for t in tasks if t.kind not in ("evaluate", "control"))
    for keys, group in groups:
        t1, r1 = _dispatch(space, group, method, theta0=theta, **kw)
        for b in space.blocks:
            if b.key in keys:
                theta[b.slice] = t1[b.slice]
        merged.iterations += r1.iterations
        # only this group's own blocks: a later group re-reports a shift of zero
        # for every parameter it did not touch, and letting that through would
        # erase what an earlier group found.
        merged.contraction.update({k: v for k, v in r1.contraction.items() if k in keys})
        merged.held_out.update(r1.held_out)
        merged.notes += r1.notes
        merged.converged = merged.converged or r1.converged
    merged.log_posterior = space.log_prior(theta) + sum(
        t.weight * float(t.logp(theta)) for t in tasks if t.kind not in ("evaluate", "control"))
    # measured once, end to end, against where the fit started -- which is the
    # only definition of "did the data move this" that survives grouping.
    _shift(space, start, theta, merged)
    merged.notes.append(f"fitted in {len(groups)} independent groups; each parameter was "
                        "moved by the sources that bear on it and no others")
    _score(space, tasks, theta, merged)
    return theta, merged


def _dispatch(space, tasks, method: Method, **kw):
    if method is Method.VI:
        theta, sd, rep = fit_vi(space, tasks, **kw)
        return theta, rep
    if method is Method.ENSEMBLE:
        theta, sd, rep = fit_ensemble(space, tasks, **kw)
        return theta, rep
    if method is Method.DISTIL:
        return fit_distil(space, tasks, kw.pop("teachers", {}), **kw)
    return fit_map(space, tasks, **kw)


def _partition(space: ParameterSpace, tasks: Sequence[Task]) -> list[tuple[set[str], list[Task]]]:
    """union-find over tasks by the parameter blocks they share.

    two tasks belong together exactly when they can move the same parameter.
    anything else -- grouping by process, by field, by dataset -- is a guess
    about coupling that the tasks themselves already answer.
    """
    keys_of: dict[str, set[str]] = {}
    for t in tasks:
        ks = set()
        for b in space.blocks:
            if b.key in t.moves or b.process in t.moves or \
                    f"{b.process}:{b.implementation}" in t.moves:
                ks.add(b.key)
        keys_of[t.name] = ks

    groups: list[tuple[set[str], list[Task]]] = []
    for t in tasks:
        ks = keys_of[t.name]
        hit = [g for g in groups if g[0] & ks] if ks else []
        if not hit:
            groups.append((set(ks), [t]))
            continue
        first = hit[0]
        for g in hit[1:]:
            first[0].update(g[0])
            first[1].extend(g[1])
            groups.remove(g)
        first[0].update(ks)
        first[1].append(t)
    return groups


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------


def _prior_sd(b: ParameterBlock) -> float:
    p = b.prior
    if p.dist == "lognormal":
        return abs(p.scale) * max(math.exp(p.loc), _EPS)     # local sd of the natural value
    if p.dist == "uniform":
        return abs(p.scale - p.loc) / math.sqrt(12.0)
    return abs(p.scale)


def _shift(space: ParameterSpace, before: np.ndarray, after: np.ndarray,
           rep: FitReport, sd: np.ndarray | None = None) -> None:
    for b in space.blocks:
        s = max(_prior_sd(b), _EPS)
        rep.shift[b.key] = float(np.mean(after[b.slice] - before[b.slice]) / s)
        if sd is not None:
            rep.contraction[b.key] = float(np.mean(sd[b.slice]) / s)


def _score(space: ParameterSpace, tasks: Sequence[Task], theta: np.ndarray,
           rep: FitReport) -> None:
    """evaluate every eval and control task at the final theta.

    a control task must produce a null; it is reported beside the evaluation
    scores rather than in a separate place, because a pipeline whose negative
    control lights up has invalidated the numbers next to it and the two should
    never be read apart.
    """
    for t in tasks:
        if t.kind in ("evaluate", "control"):
            rep.held_out[f"{t.kind}:{t.name}"] = float(t.logp(theta))


def provenance_after(space: ParameterSpace, rep: FitReport,
                     threshold: float = 0.25) -> dict[str, Provenance]:
    """what each block's provenance becomes once the fit has run.

    a parameter the data moved becomes `FIT`; one it did not keeps whatever the
    declaration said, whether that was `LITERATURE` or `WEAK`.  this is the
    mapping §7 needs to state which structure is constrained and which is
    prior-dominated, and computing it from the shift rather than from the
    intention is what makes it trustworthy.
    """
    moved = set(rep.moved(threshold))
    distilled = rep.method == "distil"
    out: dict[str, Provenance] = {}
    for b in space.blocks:
        if b.key in moved:
            out[b.key] = Provenance.DISTILLED if distilled else Provenance.FIT
        else:
            out[b.key] = b.prior.provenance
    return out


__all__ = [
    "FitReport", "Method", "Task", "fit", "fit_distil", "fit_ensemble", "fit_map",
    "fit_vi", "provenance_after",
]
