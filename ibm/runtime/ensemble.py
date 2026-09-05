"""ensemble propagation, and the projection back onto a declared form.

    p(x_{t+dt}) = P_f( p(x_t), p(theta) )

for a linear f this push-forward is exact and closed-form -- a transfer function
scales the mean and squares onto the power, which is what
`SpectralGaussian.apply_transfer` does -- and nothing in this module is needed.
for a nonlinear or learned f it is not, and ARCHITECTURE.md's general mechanism
is sampling followed by moment matching.  that is this module.

two things are worth being blunt about.

**the projection is a loss, and it is the point.**  moment matching keeps the
first two moments and discards everything else: bimodality, skew, the heavy tail
a saturating nonlinearity puts on a driven population.  ibm-1 projects back onto
each component's *declared* form because the declaration is the model's
hypothesis class -- and a runtime that quietly upgraded a block to a particle
cloud whenever the dynamics got interesting would have made the declaration a
suggestion.  so the projection happens, and `NonGaussianity` measures what it
cost, so a caller can see when the form has stopped being adequate rather than
inferring it from a prediction that is confidently wrong.

**the ensemble width is a real cost and it scales with how learned the graph is.**
an entirely analytic materialization needs no members at all.  one whose
neurovascular coupling is a fitted network needs enough to resolve that network's
output distribution, and the cost multiplies the site budget and the spectral
budget that were already multiplying each other.  `advise` states the trade
rather than hiding it behind a default.

uncertain parameters ride along.  §4 says theta is uncertain and the push-forward
is over both, so a member is a draw of state *and* a draw of theta; propagating
the mean parameters with a sampled state understates the width by exactly the
amount the parameters were uncertain, which for a `weak()` prior is most of it.

a jax backend replaces the python loop over members with a single `vmap` and
nothing else in this file changes -- the moment matching is already vectorized
over members, and the sampling already goes through the forms' own `sample`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ibm.fields.uncertainty.scalar import ScalarGaussian
from ibm.fields.uncertainty.spectral import SpectralGaussian, TemporalBasis
from ibm.runtime.state import Block, Layout, State

_EPS = 1e-30


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NonGaussianity:
    """what moment matching threw away, per component.

    skew and excess kurtosis of the ensemble, reduced over sites and frequency.
    they are cheap, they are the first two things a gaussian cannot represent,
    and a value of order 1 in either means the declared form has stopped
    describing this variable -- typically because a saturating nonlinearity is
    being driven past its linear range, which is exactly when a materialization
    is doing something interesting and exactly when its error bars stop meaning
    what they say.
    """

    component: str
    members: int
    skew: float
    excess_kurtosis: float
    #: monte-carlo standard error of the mean, relative to the ensemble sd.  if
    #: this is not small, the width being reported is the ensemble's noise rather
    #: than the state's uncertainty, and more members are needed before the
    #: skew and kurtosis above mean anything.
    mc_error: float

    @property
    def adequate(self) -> bool:
        return abs(self.skew) < 0.5 and abs(self.excess_kurtosis) < 1.0

    def __str__(self) -> str:
        flag = "" if self.adequate else "   <- gaussian projection is straining here"
        return (f"{self.component:32s} m={self.members:<5d} skew {self.skew:+.2f}  "
                f"exkurt {self.excess_kurtosis:+.2f}  mc {self.mc_error:.3f}{flag}")


def _diagnose(cid: str, members: np.ndarray) -> NonGaussianity:
    x = np.asarray(members)
    x = np.concatenate([x.real, x.imag], axis=-1) if np.iscomplexobj(x) else x
    m = x.shape[0]
    mu = x.mean(0)
    d = x - mu
    sd = np.sqrt(np.maximum((d ** 2).mean(0), _EPS))
    z = d / sd
    return NonGaussianity(
        component=cid, members=m,
        skew=float(np.mean((z ** 3).mean(0))),
        excess_kurtosis=float(np.mean((z ** 4).mean(0)) - 3.0),
        mc_error=float(1.0 / math.sqrt(max(m, 1))),
    )


# ---------------------------------------------------------------------------
# the ensemble
# ---------------------------------------------------------------------------


@dataclass
class Ensemble:
    """m draws of the state, held per component rather than as m whole `State`s.

    per component because that is the shape everything downstream wants: the
    nonlinearity is evaluated on a `(m, sites, n)` array in one call, the moment
    match reduces over axis 0, and neither ever needs a member to exist as a
    coherent `State` object.  materializing m states would also cost m copies of
    the provenance and the layout, which are identical across members by
    construction.
    """

    layout: Layout
    members: dict[str, np.ndarray] = _field(default_factory=dict)
    #: one draw of theta per member.  a member is a draw of state *and* of
    #: parameters; see the module docstring.
    theta: list[dict[str, Any]] = _field(default_factory=list)
    seed: int = 0

    @property
    def m(self) -> int:
        for v in self.members.values():
            return int(v.shape[0])
        return len(self.theta)

    # -- construction ----------------------------------------------------

    @classmethod
    def draw(cls, state: State, m: int, rng: np.random.Generator | None = None,
             components: Sequence[str] | None = None,
             theta_prior: Callable[[np.random.Generator], dict[str, Any]] | None = None
             ) -> "Ensemble":
        """sample m members from the state's own beliefs.

        sampling goes through each form's `sample`, not through a generic
        gaussian: the spectral form's sampler respects `relation`, so a
        phase-locked belief produces phase-locked members.  drawing them from psd
        alone would randomize the phase of every evoked response the moment it
        met a nonlinearity, which is the one place it matters most.
        """
        rng = rng or np.random.default_rng(0)
        out: dict[str, np.ndarray] = {}
        for b in state.layout:
            if components is not None and b.component not in components:
                continue
            if b.component not in state:
                continue
            out[b.component] = state[b.component].sample(m, rng)
        theta = [theta_prior(rng) for _ in range(m)] if theta_prior else []
        return cls(state.layout, out, theta)

    def time(self, cid: str, basis: TemporalBasis) -> np.ndarray:
        """member trajectories `(m, sites, n)` for a spectral block.

        the only representation a pointwise nonlinearity can be evaluated in: it
        is local in time and dense in frequency, so there is no shortcut through
        the coefficients however band-limited the coupling is.
        """
        z = self.members[cid]
        if self.layout[cid].uncertainty != "spectral":
            return np.repeat(np.asarray(z)[..., None], basis.n, axis=-1)
        return basis.synthesize(z)

    # -- reprojection ----------------------------------------------------

    def moment_match(self, basis: TemporalBasis | None = None
                     ) -> tuple[dict[str, Any], list[NonGaussianity]]:
        """project every member set back onto its component's declared form.

        the projection is exact in the sense that it is the best gaussian by
        moment agreement, and lossy in every other sense; `NonGaussianity` is
        returned alongside so the loss is not silent.
        """
        beliefs: dict[str, Any] = {}
        diag: list[NonGaussianity] = []
        for cid, mem in self.members.items():
            block = self.layout[cid]
            diag.append(_diagnose(cid, mem))
            if block.uncertainty == "spectral":
                b = block.basis or basis
                if b is None:
                    continue
                z = np.asarray(mem)
                if not np.iscomplexobj(z):
                    beliefs[cid] = SpectralGaussian.moment_match(b, z)
                    continue
                mu = z.mean(0)
                d = z - mu
                c = z.shape[0] / max(z.shape[0] - 1, 1)
                beliefs[cid] = SpectralGaussian(
                    b, mu, (np.abs(d) ** 2).mean(0).real * c, (d * d).mean(0) * c)
            else:
                beliefs[cid] = ScalarGaussian.moment_match(np.asarray(mem).real)
        return beliefs, diag

    def into(self, state: State, *, note: bool = True) -> State:
        """write the matched moments back, recording where the projection bit."""
        s = state.copy()
        beliefs, diag = self.moment_match(state.layout.basis)
        for cid, b in beliefs.items():
            s.beliefs[cid] = b
        if note:
            for d in diag:
                if not d.adequate:
                    s.note(d.component,
                           f"moment-matched from {d.members} members with skew {d.skew:+.2f} "
                           f"and excess kurtosis {d.excess_kurtosis:+.2f}: the declared "
                           "gaussian form no longer describes this variable's distribution")
                else:
                    s.note(d.component, f"moment-matched from {d.members} members")
        return s


# ---------------------------------------------------------------------------
# propagation
# ---------------------------------------------------------------------------


def propagate(state: State, f: Callable[[dict[str, np.ndarray], dict[str, Any]], Any], *,
              m: int, basis: TemporalBasis, rng: np.random.Generator | None = None,
              reads: Sequence[str] | None = None,
              theta_prior: Callable[[np.random.Generator], dict[str, Any]] | None = None
              ) -> tuple[dict[str, Any], list[NonGaussianity]]:
    """push an ensemble through a nonlinear or learned f and match moments.

    `f` is called once per member with time-domain trajectories and that member's
    theta draw, and returns a mapping from component id to `(sites, n)`.  once
    per member rather than once on a stacked array because a learned module may
    carry per-member parameters, and batching over members is the caller's
    optimization to make -- a jax backend does it with `vmap` and this loop
    disappears.
    """
    rng = rng or np.random.default_rng(0)
    ens = Ensemble.draw(state, m, rng, components=reads, theta_prior=theta_prior)
    xs = {cid: ens.time(cid, basis) for cid in ens.members}
    outs: dict[str, list[np.ndarray]] = {}
    for i in range(m):
        member = {cid: v[i] for cid, v in xs.items()}
        theta = ens.theta[i] if ens.theta else {}
        y = f(member, theta)
        if not isinstance(y, dict):
            raise TypeError("a propagated f must return {component_id: (sites, n)}; a bare "
                            "array cannot say which state it is pressure on")
        for cid, v in y.items():
            outs.setdefault(cid, []).append(np.asarray(v))
    stacked = Ensemble(state.layout,
                       {cid: np.stack(v) for cid, v in outs.items()})
    for cid in list(stacked.members):
        if cid in state.layout and state.layout[cid].uncertainty == "spectral":
            stacked.members[cid] = basis.analyze(stacked.members[cid])[..., : basis.k]
    return stacked.moment_match(basis)


def reproject_nonlinear(state: State, couplings: Iterable[Any], basis: TemporalBasis, *,
                        m: int = 32, rng: np.random.Generator | None = None) -> list[NonGaussianity]:
    """widen the belief of every component a nonlinear coupling writes, in place.

    called by `ibm.runtime.step.advance` once the window's *mean* has been
    solved.  the split is deliberate: the mean is a boundary-value solve and
    wants the exact linear inverse, while the width is a push-forward and wants
    samples, and doing both together would have meant m boundary-value solves per
    window for a quantity that is second-order.

    what it computes is the width the nonlinearity induces around the solved
    mean, which is a linearization-free version of the usual delta-method step
    and is correct to the extent that the nonlinearity does not move the mean
    much across the ensemble -- the same condition under which the solved mean
    was the right thing to solve for.  where it fails, the returned diagnostics
    say so.
    """
    rng = rng or np.random.default_rng(0)
    targets = {c.writes for c in couplings if not getattr(c, "linear", True)}
    targets = {t for t in targets if t in state and state.layout[t].uncertainty == "spectral"}
    if not targets:
        return []
    reads: set[str] = set()
    for c in couplings:
        if getattr(c, "linear", True):
            continue
        reads.update(getattr(c, "reads", ()) or ())
    reads &= set(state.layout.components)

    ens = Ensemble.draw(state, m, rng, components=sorted(reads | targets))
    xs = {cid: ens.time(cid, basis) for cid in ens.members}
    acc: dict[str, np.ndarray] = {t: np.zeros((m, state.layout[t].n_sites, basis.n))
                                  for t in targets}
    for c in couplings:
        if getattr(c, "linear", True) or c.writes not in acc or c.fn is None:
            continue
        for i in range(m):
            member = {cid: xs[cid][i] for cid in (c.reads or ()) if cid in xs}
            y = c.fn(member, c.theta)      # fn(x, theta), per ibm.processes.base
            y = y[c.writes] if isinstance(y, dict) else y
            acc[c.writes][i] += np.atleast_2d(np.asarray(y, float))

    diag: list[NonGaussianity] = []
    for cid, mem in acc.items():
        z = basis.analyze(mem)[..., : basis.k]
        d = _diagnose(cid, z)
        diag.append(d)
        belief = state[cid]
        var = (np.abs(z - z.mean(0)) ** 2).mean(0).real * (m / max(m - 1, 1))
        # the *mean* stays where the solve put it.  the ensemble's mean is a noisy
        # estimate of the same quantity and replacing a converged solve with it
        # would trade a residual of 1e-8 for one of 1/sqrt(m).
        state.beliefs[cid] = replace(belief, psd=belief.psd + var)
        state.note(cid, f"nonlinear width from {m} members; skew {d.skew:+.2f}, "
                        f"excess kurtosis {d.excess_kurtosis:+.2f}"
                        + ("" if d.adequate else " -- the gaussian projection is straining"))
    return diag


# ---------------------------------------------------------------------------
# what the ensemble costs
# ---------------------------------------------------------------------------


def advise(layout: Layout, couplings: Iterable[Any], *, target_mc_error: float = 0.05
           ) -> tuple[int, str]:
    """how many members this materialization needs, and why that is a real cost.

    the width is `1/sqrt(m)` in relative terms, so a 5% standard error on a
    reported sd is ~400 members -- for the *nonlinear fraction of the graph
    only*, which is the number worth reporting.  a fully analytic model needs
    zero and a fully learned one pays the ensemble on top of the site budget and
    the spectral budget that were already multiplying each other, which is the
    §1 accounting made concrete.
    """
    cs = list(couplings)
    nonlinear = [c for c in cs if not getattr(c, "linear", True)]
    written = {c.writes for c in nonlinear}
    if not written:
        return 0, ("every coupling is linear or a pure delay, so the push-forward is exact "
                   "and closed-form: no ensemble is needed at all")
    m = int(math.ceil(1.0 / max(target_mc_error, 1e-3) ** 2))
    cost = sum(layout[c].cost for c in written if c in layout)
    frac = len(nonlinear) / max(len(cs), 1)
    return m, (f"{len(nonlinear)}/{len(cs)} couplings ({frac:.0%}) are nonlinear or learned "
               f"and write {len(written)} components ({cost:,} numbers).  m={m} members for "
               f"{target_mc_error:.0%} relative error on the reported widths costs "
               f"{m * cost:,} numbers of working set -- the price of the learned fraction, "
               "and the reason an analytic f is worth keeping wherever the science supplies one")


__all__ = ["Ensemble", "NonGaussianity", "advise", "propagate", "reproject_nonlinear"]
