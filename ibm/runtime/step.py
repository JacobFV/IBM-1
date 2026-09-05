"""advancing a materialized model over a window.

there is no `march` in this file, and that is the whole point.  a state variable
is a belief about a trajectory over a window carried in a spectral basis
(ARCHITECTURE.md §1), so "advance" does not mean "take a small step from t to
t+dt".  the entire window is unknown at once, the dynamics couple every
coefficient in it, and what has to be solved is a *boundary-value problem over
the window*: find the spectral state whose implied derivative equals the pressure
every process exerts on it, subject to matching the window that came before.

the split that makes that affordable is the one §1 argues for.

*linear time-invariant and pure-delay contributions never enter the time domain.*
they are diagonal in this basis -- a transfer function and a phase ramp -- so
they are applied exactly, at any window length, with no per-edge history buffer
and no delay-bounded timestep.  they are also the part of the operator that can
be *inverted* exactly, which is what turns the fixed point below from a crawl
into a solve.

*nonlinear contributions round-trip.*  a pointwise nonlinearity is local in time
and dense in frequency, so it must be synthesized, evaluated, and re-analyzed
(`TemporalBasis.roundtrip`), and it does not preserve gaussianity.  the mean
iterates on the round-tripped mean; the uncertainty is reprojected onto the
declared form by moment matching over an ensemble (`ibm.runtime.ensemble`), which
is the one place non-gaussian structure is discarded, deliberately and visibly.

*scalar-form components are not in the window basis at all.*  a component whose
belief is one number and one variance over the window has no spectrum to solve
for, so it advances by ordinary integration over the hop -- exactly, since its
self-coupling is a single pole.  blood volume does not want a krylov space.

the solver is damped picard first and newton-krylov second, in that order and not
the other way round.  picard here is not a naive fixed point: the exact inverse of
the diagonal linear operator is available for free, so each sweep is
`z <- (1-a) z + a (i omega - A(omega))^-1 N(z)`, which converges immediately for
anything that is mostly LTI -- and most of the §5 inventory is.  newton-krylov
exists for the cases that are not: strong nonlinear feedback, a stiff algebraic
constraint, a resonant loop near its stability margin.  it is matrix-free -- gmres
on a finite-difference jvp over `State.pack_means` -- because the jacobian of a
whole materialized window is never worth forming.

### the causality refusal

windows are stitched by overlap, and the overlap has to be at least as long as
the model's longest conduction delay.  this is not a quality knob.  the temporal
laplacian is **cyclic**: `exp(-i omega tau)` is a circular shift, so a delay
longer than the region of the window already pinned by the previous solution
wraps around and delivers the window's own future as its past.  nothing raises an
error; the spectrum stays finite and plausible; the model has simply been handed
an acausal loop it will happily equilibrate.  so a plan whose overlap is shorter
than the longest delay is refused here, at plan time, rather than diagnosed later
from an oscillation nobody can source.

### where a jax backend goes

three places, all narrow.  `_drift` is the only function that touches array
values in a loop, and is a `jax.tree_map` over blocks once beliefs are pytrees.
`_newton_correction` is `jax.scipy.sparse.linalg.gmres` with `jax.jvp` in place of
the finite difference, which also makes the jvp exact rather than a difference
quotient.  `State.pack_means` disappears into `ravel_pytree`.  nothing else in
this module knows it is running on numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ibm.fields.uncertainty.scalar import ScalarGaussian
from ibm.fields.uncertainty.spectral import SpectralGaussian, TemporalBasis
from ibm.registry import REGISTRY, Form
from ibm.runtime.state import Block, Layout, State, _attr
from ibm.vocabulary import FULL, Band

_EPS = 1e-30


class CausalityViolation(Exception):
    """a window plan that would let the future act on the past.

    raised rather than warned because the failure is silent by construction: a
    wrapped delay produces a perfectly ordinary-looking spectrum, and the only
    evidence of it is that the answer is wrong.
    """


class SolverFailed(Exception):
    """the window did not converge, and the caller is not being handed a guess."""


# ---------------------------------------------------------------------------
# the window plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowPlan:
    """how a run is cut into overlapping windows.

    `overlap` is a fraction of the window, matching `ibm.materialize.request.Window`,
    because the request is where a human chose it and the runtime should not have
    a second, differently spelled opinion about it.  `hop` is what actually
    advances; the overlap is paid for twice on purpose.
    """

    basis: TemporalBasis
    overlap: float = 0.25
    n_windows: int = 1
    #: how tightly the overlap is held, and deliberately below 1.
    #:
    #: a driven linear system on a *cyclic* window has a unique solution: the
    #: homogeneous solutions of `i omega z = A z` exist only exactly at
    #: resonance, so there is generically no freedom left for a boundary
    #: condition to consume.  the dynamics and the carried tail are therefore two
    #: conditions the window cannot both satisfy, and imposing the tail as a hard
    #: projection makes the sweep alternate between them forever instead of
    #: converging -- classic alternating projection onto two sets that do not
    #: intersect.
    #:
    #: at weight below 1 the sweep is an *averaged* operator, whose fixed point
    #: is the least-squares compromise between the dynamics and the continuity
    #: being imposed on them.  that compromise is the honest object: §1 says
    #: continuity between windows is imposed rather than inherited, and the
    #: residual left at the joint is the price of imposing it.  `StepReport`
    #: carries that price rather than hiding it.
    match_weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be a fraction in [0, 1)")
        if self.n_windows < 1:
            raise ValueError("a plan runs at least one window")

    @property
    def overlap_n(self) -> int:
        return int(round(self.overlap * self.basis.n))

    @property
    def hop_n(self) -> int:
        return max(self.basis.n - self.overlap_n, 1)

    @property
    def overlap_s(self) -> float:
        return self.overlap_n * self.basis.dt

    @property
    def hop_s(self) -> float:
        return self.hop_n * self.basis.dt

    @property
    def duration_s(self) -> float:
        return self.basis.duration_s + (self.n_windows - 1) * self.hop_s

    # -- the refusal -----------------------------------------------------

    def causality_problems(self, max_memory_s: float) -> list[str]:
        """everything wrong with this plan given the graph's longest memory."""
        out: list[str] = []
        if max_memory_s > self.overlap_s:
            out.append(
                f"longest conduction memory is {max_memory_s * 1e3:.3g} ms but the window "
                f"overlap is only {self.overlap_s * 1e3:.3g} ms.  the temporal laplacian is "
                "cyclic, so a delay longer than the pinned overlap wraps and reads this "
                "window's future as its past -- a silent causality violation.  either raise "
                f"overlap to at least {max_memory_s / self.basis.duration_s:.3f} of the "
                "window, or lengthen the window")
        if max_memory_s > 0.5 * self.basis.duration_s:
            out.append(
                f"longest conduction memory is {max_memory_s * 1e3:.3g} ms against a "
                f"{self.basis.duration_s * 1e3:.3g} ms window: over half the window is "
                "boundary, so almost nothing here is dynamics")
        return out

    def refuse_if_acausal(self, max_memory_s: float) -> "WindowPlan":
        p = self.causality_problems(max_memory_s)
        if p:
            raise CausalityViolation("; ".join(p))
        return self

    def for_memory(self, max_memory_s: float, headroom: float = 2.0) -> "WindowPlan":
        """the same plan widened until it is legal.

        offered rather than applied automatically: widening the overlap costs
        real work per window and the caller should see it happen.
        """
        want = min(0.9, headroom * max_memory_s / max(self.basis.duration_s, _EPS))
        return replace(self, overlap=max(self.overlap, want))

    @classmethod
    def of(cls, window: Any, n_windows: int = 1) -> "WindowPlan":
        """derive a plan from `ibm.materialize.request.Window`, or from a basis."""
        if isinstance(window, TemporalBasis):
            return cls(window, 0.25, n_windows)
        basis = _attr(window, "basis", default=None)
        basis = basis() if callable(basis) else basis
        if basis is None:
            n, dt = _attr(window, "n", default=512), _attr(window, "dt", default=2e-3)
            basis = TemporalBasis(int(n), float(dt))
        return cls(basis, float(_attr(window, "overlap", default=0.25) or 0.0), n_windows)

    def describe(self) -> str:
        return (f"window n={self.basis.n} dt={self.basis.dt:g}s "
                f"({self.basis.duration_s * 1e3:.0f} ms, k={self.basis.k}, "
                f"nyquist {self.basis.nyquist_hz:g} Hz)\n"
                f"overlap {self.overlap:.2f} = {self.overlap_n} samples "
                f"({self.overlap_s * 1e3:.0f} ms), hop {self.hop_s * 1e3:.0f} ms, "
                f"{self.n_windows} windows -> {self.duration_s:g}s")


# ---------------------------------------------------------------------------
# couplings: the runtime's view of a materialized process
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coupling:
    """one materialized edge of the process graph, normalized for the solver.

    `ibm.materialize` owns what a materialized process *is*; this is the small
    subset of it the solver needs, and one process generally becomes several
    couplings -- one per (read component -> written component) edge -- because
    the split that matters here is which contributions are diagonal and which
    are not, not which declaration they came from.

    `memory_s` is the field that keeps the model honest.  it is the longest time
    into the past this coupling reads: a pure delay's tau, a dispersive tract's
    mean plus a few standard deviations, a synapse's settling time.  it is what
    the window plan is checked against, and defaulting it to the delay alone is
    an underestimate the caller should correct where the impulse response is long.
    """

    process: str
    writes: str
    reads: tuple[str, ...] = ()
    form: Form = Form.LTI
    #: H(omega) on the retained components: (k,), (n_in, k) or (n_out, k).
    #:
    #: an implementation declares this as `transfer(basis, **theta) -> (..., k)`
    #: -- it takes a `TemporalBasis` and returns H sampled on that basis's
    #: retained components, never a bare omega array and never curried.  the
    #: runtime owns the window and the retained bandwidth, so the basis is what
    #: crosses the boundary; see the calling convention at the top of
    #: `ibm.processes.base`.  a callable is evaluated here, against whichever
    #: basis is in hand, so that a solve on a truncated band gets H truncated to
    #: the same band rather than a stale array of the wrong length.
    transfer: np.ndarray | Callable[..., np.ndarray] | None = None
    #: spatial mixing from the topology, (n_out, n_in).  never enters the time
    #: domain -- it acts on the site axis at every frequency independently.
    mix: np.ndarray | None = None
    delay_s: float | np.ndarray = 0.0
    gain: float | np.ndarray = 1.0
    #: a time-domain rate law: fn(dict[cid, (n_sites, n)]) -> (n_out, n).
    fn: Callable[..., Any] | None = None
    #: gamma, for Form.CONSTRAINT.  the stiff limit of pressure is a question of
    #: how it is solved, not of what it is (§4), and this is that number.
    stiffness: float = 0.0
    #: soft region membership on the written sites.  a partition boundary is a
    #: gradient, so this scales the gain and is never thresholded.
    weights: np.ndarray | None = None
    theta: dict[str, Any] = _field(default_factory=dict)
    memory_s: float = 0.0
    note: str = ""

    # -- structure -------------------------------------------------------

    @property
    def linear(self) -> bool:
        return self.form in (Form.LTI, Form.CONSTRAINT)

    @property
    def self_diagonal(self) -> bool:
        """does this contribute to the operator that can be inverted exactly?

        only if it is linear, reads exactly the component it writes, and does not
        mix sites.  a self-coupling *through* a topology is still linear and
        still exact, but inverting it would mean inverting an (n_sites, n_sites)
        matrix per frequency, which is the thing this design exists to avoid; it
        goes on the right-hand side instead and picard handles it.
        """
        return self.form is Form.LTI and self.reads == (self.writes,) and self.mix is None

    @property
    def stiff(self) -> bool:
        """an algebraic relation, solved rather than stepped through.

        `x_O = g(x_I)` is the gamma -> infinity limit of `dot x_O += -gamma (x_O
        - g(x_I))` (§4), so it is not a different kind of thing and does not get
        a different mechanism: the `-gamma x_O` half joins the diagonal operator
        that is inverted exactly, and the `+gamma g(x_I)` half is an ordinary
        drive.  the quotient is finite and well conditioned however large gamma
        is, which is the whole reason the limit can be taken by *solving* rather
        than by integrating a stiff ODE nobody could take a step of.
        """
        return self.form is Form.CONSTRAINT

    @classmethod
    def from_implementation(cls, impl: Any, writes: str, reads: Sequence[str], *,
                            theta: Mapping[str, Any] | None = None, **kw) -> "Coupling":
        """build a coupling from a registered `Implementation`.

        the point of routing through here is the calling convention: an
        implementation's `transfer` is `transfer(basis, **theta) -> (..., k)` and
        its `fn` is `fn(x, theta) -> {component: pressure}`, and a coupling built
        by hand that gets either wrong fails only once a real window is solved.
        `transfer` is stored uncalled, so the same coupling is correct on a full
        window and on a band-truncated one.
        """
        return cls(process=str(getattr(impl, "process", "?")), writes=writes,
                   reads=tuple(reads), form=getattr(impl, "form", Form.LTI),
                   transfer=getattr(impl, "transfer", None), fn=getattr(impl, "fn", None),
                   theta=dict(theta or {}), **kw)

    def longest_memory_s(self) -> float:
        tau = float(np.max(np.abs(np.asarray(self.delay_s)))) if np.size(self.delay_s) else 0.0
        return max(self.memory_s, tau)

    # -- evaluation ------------------------------------------------------

    def _H(self, basis: TemporalBasis, n_in: int, n_out: int) -> np.ndarray | None:
        """transfer times delay ramp times gain, broadcast onto (.., k)."""
        parts: list[np.ndarray] = []
        if self.transfer is not None:
            H = (self.transfer(basis, **self.theta) if callable(self.transfer)
                 else self.transfer)
            parts.append(np.asarray(H, dtype=np.complex128))
        tau = np.asarray(self.delay_s, dtype=float)
        if np.any(tau != 0.0):
            w = basis.omega
            parts.append(np.exp(-1j * (tau[..., None] * w if tau.ndim else tau * w)))
        if self.form is Form.CONSTRAINT and self.stiffness:
            parts.append(np.full(basis.k, complex(self.stiffness)))
        g = np.asarray(self.gain)
        if g.ndim == 0:
            if g != 1.0:
                parts.append(np.full(basis.k, complex(g)))
        else:
            parts.append(g.astype(np.complex128)[..., None] if g.shape[-1] != basis.k
                         else g.astype(np.complex128))
        if not parts:
            return None
        out = parts[0]
        for p in parts[1:]:
            out = out * p
        return out

    def diagonal(self, basis: TemporalBasis, block: Block) -> np.ndarray:
        """A(omega): this coupling's share of the exactly invertible operator.

        for a self-diagonal LTI coupling that is H itself.  for a stiff algebraic
        constraint it is `-gamma`, frequency-flat, regardless of what the
        constraint reads -- the relaxation rate towards `g(x_I)` is not a
        property of the input.
        """
        if self.stiff:
            a = np.full((block.n_sites, basis.k), complex(-self.stiffness))
            return a * self._w(block)[:, None]
        H = self._H(basis, block.n_sites, block.n_sites)
        a = np.zeros(basis.k, dtype=np.complex128) if H is None else np.asarray(
            H, dtype=np.complex128)
        a = np.broadcast_to(a, (block.n_sites, basis.k)) if a.ndim == 1 else a
        return a * self._w(block)[:, None]

    def _w(self, block: Block) -> np.ndarray:
        if self.weights is None:
            return np.ones(block.n_sites)
        return np.asarray(self.weights, float).reshape(-1)

    def spectral_drift(self, state: State, basis: TemporalBasis) -> np.ndarray:
        """this coupling's contribution to dz/dt of the component it writes.

        LTI and pure delay are applied with `SpectralGaussian.apply_transfer`
        semantics on the mean; nonlinear forms round-trip through the time
        domain.  the two are kept in one function because the caller must not be
        able to accidentally route a nonlinearity down the exact path.
        """
        out_block = state.layout[self.writes]
        n_out = out_block.n_sites
        w = self._w(out_block)

        if self.linear:
            acc = np.zeros((n_out, basis.k), dtype=np.complex128)
            for cid in self.reads:
                z = _coefficients(state, cid, basis)
                H = self._H(basis, z.shape[0], n_out)
                if H is not None:
                    Hn = np.asarray(H)
                    if Hn.ndim == 2 and Hn.shape[0] == z.shape[0]:
                        z = z * Hn
                        H = None
                y = self.mix @ z if self.mix is not None else _fit_sites(z, n_out)
                if H is not None:
                    y = y * np.asarray(H)
                acc = acc + y
            # for a stiff constraint this is the `+gamma g(x_I)` half; `diagonal`
            # holds the `-gamma x_O` half, so the limit is solved, not stepped.
            return acc * w[:, None]

        if self.fn is None:
            return np.zeros((n_out, basis.k), dtype=np.complex128)

        xs = {cid: basis.synthesize(_coefficients(state, cid, basis)) for cid in self.reads}
        y = self.fn(xs, **self.theta) if self.theta else self.fn(xs)
        y = y[self.writes] if isinstance(y, dict) else y
        y = np.atleast_2d(np.asarray(y, float))
        z = basis.analyze(_fit_sites(y, n_out))
        return z[..., : basis.k] * w[:, None]

    def scalar_drift(self, state: State) -> tuple[np.ndarray, np.ndarray]:
        """(self rate a, external drive n) for a scalar-form written component.

        a scalar block has no spectrum, so what a coupling contributes to it is
        its DC behaviour: H(0) for a linear form, the time-average of f for a
        nonlinear one.  reading a spectral input at DC is the registered
        spectral -> scalar conversion and is lossy by declaration; it is the right
        conversion exactly when the target genuinely has one timescale, which is
        the reason its form was declared scalar in the first place.
        """
        block = state.layout[self.writes]
        n_out, w = block.n_sites, self._w(block)
        zero = np.zeros(n_out)

        if self.linear:
            g = self._H(_dc_basis(), n_out, n_out)
            dc = complex(np.asarray(g).reshape(-1)[0]) if g is not None else 1.0 + 0j
            if self.stiff:
                # -gamma x_O + gamma g(x_I): both halves, since a scalar block is
                # integrated rather than solved and there is no operator to invert.
                drive = np.zeros(n_out)
                for cid in self.reads:
                    x = _dc_value(state, cid)
                    y = self.mix @ x if self.mix is not None else _fit_sites(x, n_out)
                    drive = drive + dc.real * np.asarray(y, float).reshape(n_out)
                return np.full(n_out, -self.stiffness) * w, drive * w
            if self.self_diagonal:
                return np.full(n_out, dc.real) * w, zero
            drive = np.zeros(n_out)
            for cid in self.reads:
                x = _dc_value(state, cid)
                y = self.mix @ x if self.mix is not None else _fit_sites(x, n_out)
                drive = drive + dc.real * np.asarray(y, float).reshape(n_out)
            return zero, drive * w

        if self.fn is None:
            return zero, zero
        xs = {cid: _dc_value(state, cid)[:, None] for cid in self.reads}
        y = self.fn(xs, **self.theta) if self.theta else self.fn(xs)
        y = y[self.writes] if isinstance(y, dict) else y
        y = np.asarray(y, float).reshape(n_out, -1).mean(-1)
        return zero, y * w


def _dc_basis() -> TemporalBasis:
    """a one-coefficient basis, so `_H` can be asked for H(0) without a window."""
    return TemporalBasis(2, 1.0, kmax=1)


def _coefficients(state: State, cid: str, basis: TemporalBasis) -> np.ndarray:
    """the spectral mean of any block, lifting a scalar one into DC.

    the lift is the registered `scalar -> spectral` conversion: lossless in that
    nothing is discarded, but it asserts the variable has no power above DC
    within the window, which is a claim and not a neutral embedding.
    """
    b = state.layout[cid]
    belief = state[cid]
    if b.uncertainty == "spectral":
        z = np.asarray(belief.mean)
        return z[..., : basis.k] if z.shape[-1] >= basis.k else np.pad(
            z, [(0, 0)] * (z.ndim - 1) + [(0, basis.k - z.shape[-1])])
    z = np.zeros((b.n_sites, basis.k), dtype=np.complex128)
    z[:, 0] = np.asarray(belief.mean, float)
    return z


def _dc_value(state: State, cid: str) -> np.ndarray:
    b = state.layout[cid]
    belief = state[cid]
    if b.uncertainty == "spectral":
        return np.asarray(belief.mean[..., 0]).real
    return np.asarray(belief.mean, float)


def _fit_sites(x: np.ndarray, n_out: int) -> np.ndarray:
    """broadcast an input's site axis onto the output's, or refuse.

    the only implicit spatial map allowed: one-to-one, or a single site fanning
    out to all of them.  anything else is a topology, and a topology that has not
    been supplied as `mix` has not been materialized -- which is a build error,
    not something to guess at here.
    """
    x = np.asarray(x)
    if x.shape[0] == n_out:
        return x
    if x.shape[0] == 1:
        return np.broadcast_to(x, (n_out,) + x.shape[1:])
    raise ValueError(
        f"coupling reads {x.shape[0]} sites and writes {n_out} with no mixing matrix; "
        "the topology this edge runs over was not materialized")


def couplings_of(model: Any, basis: TemporalBasis) -> tuple[Coupling, ...]:
    """normalize whatever `ibm.materialize` built into couplings.

    the same adaptor stance as `Layout.of`: accept the plausible spellings, fail
    loudly when none is present, and disappear the moment materialize hands back
    `Coupling` objects directly.
    """
    raw = _attr(model, "couplings", "edges", "materialized_processes", "processes",
                default=None)
    if raw is None:
        raise TypeError(
            f"{type(model).__name__} exposes no couplings: expected `.couplings`, `.edges` "
            "or `.processes`.  ibm.runtime cannot step a model that has not said what "
            "dynamics it materialized.")
    out: list[Coupling] = []
    for c in raw:
        if isinstance(c, Coupling):
            out.append(c)
            continue
        pid = _attr(c, "process", "id", default="?")
        writes = _attr(c, "writes", "output", "target", default=None)
        if writes is None:
            raise TypeError(f"materialized process {pid!r} names no written component")
        reads = _attr(c, "reads", "inputs", "sources", default=())
        form = _attr(c, "form", default=Form.LTI)
        out.append(Coupling(
            process=str(pid), writes=str(writes),
            reads=tuple(str(r) for r in ([reads] if isinstance(reads, str) else reads)),
            form=form if isinstance(form, Form) else Form(str(form)),
            # left callable when it is one: `transfer(basis, **theta)` is the
            # declared convention, and evaluating it lazily is what lets one
            # coupling be reused across windows of different retained bandwidth.
            transfer=_attr(c, "transfer", "H"),
            mix=_attr(c, "mix", "weights_matrix", "operator"),
            delay_s=_attr(c, "delay_s", "delay", default=0.0) or 0.0,
            gain=_attr(c, "gain", default=1.0),
            fn=_attr(c, "fn", "f"),
            stiffness=float(_attr(c, "stiffness", "gamma", default=0.0) or 0.0),
            weights=_attr(c, "weights", "region_weights"),
            theta=dict(_attr(c, "theta", "params", default={}) or {}),
            memory_s=float(_attr(c, "memory_s", "memory", default=0.0) or 0.0),
            note=str(_attr(c, "note", default="") or ""),
        ))
    return tuple(out)


def longest_memory(couplings: Iterable[Coupling]) -> float:
    return max((c.longest_memory_s() for c in couplings), default=0.0)


# ---------------------------------------------------------------------------
# solver settings and reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Solve:
    """how hard to try, and when to stop pretending picard is working.

    `damping` below 1 is not timidity.  the picard sweep here is a contraction
    only when the non-diagonal part is small relative to the inverted diagonal
    one; a strongly recurrent circuit near its stability margin is exactly where
    that fails, and undamped it oscillates rather than diverging, which is worse
    because it looks like a rhythm.
    """

    damping: float = 0.5
    #: relative tolerance, against the *scale of the residual's own terms* --
    #: `||i omega z||` and `||sum_p f_p||` -- rather than against the initial
    #: residual.  the difference matters from the second window onwards: a window
    #: started from the previous one's converged answer already has a tiny
    #: initial residual, and measuring progress against it demands a relative
    #: drop that has nothing left to drop.
    tol: float = 1e-8
    #: absolute floor, so a state that is genuinely near zero converges instead
    #: of chasing round-off.
    atol: float = 1e-10
    #: relative movement of the iterate below which the solve is finished even
    #: though the residual has not reached `tol`.  this is not a fudge: with a
    #: hard overlap match the window is over-determined -- 128 dynamical
    #: equations plus 32 boundary values -- so the residual has an irreducible
    #: floor equal to the mismatch between the dynamics and the continuity being
    #: imposed on them.  §1 says continuity between windows is *imposed rather
    #: than inherited*, and that floor is the price of imposing it.  the honest
    #: thing is to stop when the iterate stops moving and report the floor, not
    #: to iterate forever against an equation that has no exact solution.
    x_tol: float = 1e-10
    max_iter: int = 80
    #: try newton-krylov once picard has stopped making progress.  off gives a
    #: cheaper, more predictable step and an honest failure instead of a costly one.
    newton: bool = True
    stall_after: int = 12
    stall_ratio: float = 0.98
    #: how many newton solves one window is allowed.  capped because a
    #: matrix-free solve costs one residual evaluation per krylov vector, and a
    #: window that has taken three of them without converging is telling you the
    #: model is ill-posed, not that it needs a fourth.
    max_newton: int = 3
    krylov_restart: int = 40
    krylov_maxiter: int = 200
    #: ensemble width for reprojecting nonlinear contributions.  0 propagates the
    #: mean only and leaves the uncertainty where it was, which is a *lie of
    #: omission* and is recorded in provenance as one.
    ensemble: int = 0
    seed: int = 0


@dataclass
class StepReport:
    """what the solve actually did.  §7 requires provenance, and this is its
    dynamical half: a window that limped to its tolerance by newton-krylov after
    picard stalled is not the same result as one that converged in three sweeps."""

    window: int = 0
    iterations: int = 0
    residual: float = math.inf
    residual0: float = math.inf
    method: str = "picard"
    converged: bool = False
    #: why the loop stopped, when it was not tolerance: "continuity" when the
    #: residual floor is the imposed joint, "newton" when even a newton
    #: direction did not help, "iterations" when it simply ran out.  a caller
    #: deciding whether to believe a window needs to tell these apart -- the
    #: first is a modelling fact and the other two are numerical failures.
    limited_by: str = ""
    newton_calls: int = 0
    #: how far apart the two windows were at the joint before continuity was
    #: imposed, in state units.  small means the stitch is invisible; large means
    #: the two windows disagree about what happened and the hop is too long for
    #: the dynamics in it.
    joint_gap: float = 0.0
    notes: list[str] = _field(default_factory=list)

    def note(self, m: str) -> None:
        self.notes.append(m)

    def __str__(self) -> str:
        drop = (self.residual / self.residual0) if self.residual0 else 0.0
        return (f"window {self.window}: {self.method}, {self.iterations} iters, "
                f"residual {self.residual:.3e} ({drop:.1e} of initial), "
                f"{'converged' if self.converged else 'NOT CONVERGED'}"
                + (f", {self.newton_calls} newton solves" if self.newton_calls else "")
                + (f", joint gap {self.joint_gap:.3e}" if self.joint_gap else "")
                + (f", limited by {self.limited_by}" if self.limited_by else "")
                + ("\n  " + "\n  ".join(self.notes) if self.notes else ""))


# ---------------------------------------------------------------------------
# the drift, the residual, and the exactly invertible part
# ---------------------------------------------------------------------------


def _targets(layout: Layout, couplings: Sequence[Coupling]) -> tuple[str, ...]:
    """the spectral blocks this solve is allowed to move.

    a component no coupling writes is *exogenous* -- a stimulus, a device drive,
    a material constant -- and the registry already refuses a graph in which one
    is read, not written, not clamped and not declared exogenous.  so the solver
    must leave it exactly where it was: a coil current the experimenter set is
    not an unknown, and including it in the residual would both drive it to zero
    and leave a floor `||i omega z||` that no iteration can ever remove, which
    reads from the outside as a solver that stalls for no reason.
    """
    written = {c.writes for c in couplings}
    return tuple(b.component for b in layout
                 if b.uncertainty == "spectral" and b.component in written)


def _self_operator(layout: Layout, couplings: Sequence[Coupling],
                   basis: TemporalBasis) -> dict[str, np.ndarray]:
    """A(omega) per spectral block: the diagonal linear operator, summed.

    this is the only thing that gets inverted anywhere in this module, and it is
    inverted exactly and elementwise.  everything that does not fit in it -- site
    mixing, cross-component coupling, every nonlinearity -- lives on the
    right-hand side, where picard and, failing that, gmres deal with it.
    """
    out: dict[str, np.ndarray] = {
        cid: np.zeros((layout[cid].n_sites, basis.k), dtype=np.complex128)
        for cid in _targets(layout, couplings)}
    for c in couplings:
        if c.writes not in out or not (c.self_diagonal or c.stiff):
            continue
        out[c.writes] = out[c.writes] + c.diagonal(basis, layout[c.writes])
    return out


def _drift(state: State, couplings: Sequence[Coupling],
           basis: TemporalBasis) -> dict[str, np.ndarray]:
    """sum_p f_p, in spectral coefficients, for every spectral block.

    pressure composes by summation (§4), so this really is a sum and there is
    nowhere for a process to declare precedence.  clamps override afterwards, in
    `ibm.runtime.intervene`, which is why they are a separate module.
    """
    acc = {cid: np.zeros((state.layout[cid].n_sites, basis.k), dtype=np.complex128)
           for cid in _targets(state.layout, couplings) if cid in state}
    for c in couplings:
        if c.writes not in acc:
            continue
        acc[c.writes] = acc[c.writes] + c.spectral_drift(state, basis)
    return acc


def _residual(state: State, couplings: Sequence[Coupling],
              basis: TemporalBasis) -> dict[str, np.ndarray]:
    """r = i omega z - sum_p f_p(z).  zero is the answer.

    differentiation is exactly multiplication by i omega in this basis, which is
    the whole reason the window can be posed as an algebraic system at all.
    """
    d = _drift(state, couplings, basis)
    w = 1j * basis.omega
    return {cid: w * _coefficients(state, cid, basis) - v for cid, v in d.items()}


def _norm(r: dict[str, np.ndarray]) -> float:
    tot = 0.0
    for v in r.values():
        tot += float(np.sum(np.abs(v) ** 2))
    return math.sqrt(tot)


def _scale(state: State, couplings: Sequence[Coupling], basis: TemporalBasis) -> float:
    """the size of the residual's own terms, which is what tolerance is relative to.

    the residual is `i omega z - sum_p f_p`, and both halves are derivatives with
    physical magnitude.  a tolerance measured against them is a statement about
    how well the dynamics balance; one measured against the *initial* residual is
    a statement about how good the initial guess was, which from the second
    window onward is "excellent" and therefore useless.
    """
    w = 1j * basis.omega
    lhs = {cid: w * _coefficients(state, cid, basis)
           for cid in _targets(state.layout, couplings) if cid in state}
    return max(_norm(lhs), _norm(_drift(state, couplings, basis)))


# ---------------------------------------------------------------------------
# window-to-window continuity
# ---------------------------------------------------------------------------


@dataclass
class Carry:
    """the tail of the solved window, in the time domain, per component.

    time domain and not spectral, deliberately.  the next window has a different
    phase origin, so carrying coefficients would require a rotation that is only
    correct if the two windows are the same length and perfectly aligned; the
    trajectory samples are the thing that is actually shared.  §1 says continuity
    between windows is imposed rather than inherited, and this is the object that
    imposes it.
    """

    samples: dict[str, np.ndarray] = _field(default_factory=dict)
    n: int = 0
    t0_s: float = 0.0

    @classmethod
    def tail(cls, state: State, n: int, t0_s: float = 0.0) -> "Carry":
        out: dict[str, np.ndarray] = {}
        for b in state.layout:
            if b.uncertainty != "spectral" or b.component not in state:
                continue
            x = state.mean_time(b.component)
            out[b.component] = x[:, x.shape[1] - n:] if n else x[:, :0]
        return cls(out, n, t0_s)


def _taper(m: int, weight: float) -> np.ndarray:
    """the matching weight across the overlap: strongest at the joint, released by its end.

    a flat weight would assert the previous window's tail equally at the joint
    and 60 ms later, which is wrong in both directions -- it over-constrains the
    part of the overlap where this window's own dynamics should already have
    taken over, and it under-constrains the first sample, which is the only place
    continuity actually has to hold.  a raised cosine is the cheapest ramp with a
    continuous derivative at both ends, which matters because a kink in the
    weight becomes a kink in the trajectory and a kink in the trajectory is
    broadband in exactly the basis this state is carried in.
    """
    if m <= 1:
        return np.full(max(m, 0), weight)
    return weight * 0.5 * (1.0 + np.cos(np.pi * np.arange(m) / m))


def _match_overlap(state: State, carry: Carry | None, basis: TemporalBasis,
                   weight: float) -> float:
    """pull this window's head towards the previous window's tail, and report the gap.

    a *weighted* pull rather than an assignment, for the reason spelled out on
    `WindowPlan.match_weight`: a hard projection and the dynamics are two
    conditions a cyclic window cannot both satisfy, and alternating between them
    never converges.  the returned number is the remaining discontinuity at the
    joint before the pull, in the same units as the state, which is the quantity
    a reader needs in order to know whether the stitch is invisible or whether
    the two windows genuinely disagree about what happened.
    """
    if carry is None or carry.n <= 0 or weight <= 0.0:
        return 0.0
    m = min(carry.n, basis.n)
    w = _taper(m, weight)
    gap = 0.0
    for cid, tail in carry.samples.items():
        if cid not in state or state.layout[cid].uncertainty != "spectral":
            continue
        belief = state[cid]
        x = basis.synthesize(belief.mean)
        head = tail[:, tail.shape[1] - m:]
        if head.shape[0] != x.shape[0]:
            continue
        gap = max(gap, float(np.max(np.abs(x[:, 0] - head[:, 0]))))
        x[:, :m] = (1.0 - w) * x[:, :m] + w * head
        state.beliefs[cid] = replace(belief, mean=basis.analyze(x)[..., : basis.k])
    return gap


# ---------------------------------------------------------------------------
# scalar-form advance
# ---------------------------------------------------------------------------


def _advance_scalar(state: State, couplings: Sequence[Coupling], dt_s: float) -> None:
    """ordinary integration, over the hop, for every scalar-form block.

    exact for the linear part rather than euler: a scalar block's self-coupling
    is a single pole, so `x' = x e^{aT} + (n/a)(e^{aT} - 1)` is closed form and
    costs the same as a forward step while being unconditionally stable.  blood
    volume relaxing with a two-second time constant under a ten-second hop is the
    ordinary case, and euler would have produced a sign flip there.

    variance follows the same map -- a linear map scales variance by its square,
    and the drive contributes its own -- which is the exact gaussian push-forward
    for a linear f and the reason nothing needs an ensemble here.
    """
    a = {b.component: np.zeros(b.n_sites) for b in state.layout if b.uncertainty == "scalar"}
    n = {c: np.zeros_like(v) for c, v in a.items()}
    for c in couplings:
        if c.writes not in a:
            continue
        ai, ni = c.scalar_drift(state)
        a[c.writes] = a[c.writes] + ai
        n[c.writes] = n[c.writes] + ni
    for cid, ai in a.items():
        if cid not in state:
            continue
        belief = state[cid]
        e = np.exp(ai * dt_s)
        small = np.abs(ai * dt_s) < 1e-8
        ramp = np.where(small, dt_s, (e - 1.0) / np.where(small, 1.0, ai))
        state.beliefs[cid] = ScalarGaussian(belief.mean * e + n[cid] * ramp,
                                            belief.var * e * e)


# ---------------------------------------------------------------------------
# newton-krylov fallback
# ---------------------------------------------------------------------------


def _newton_correction(state: State, couplings: Sequence[Coupling], basis: TemporalBasis,
                       solve: Solve, report: StepReport) -> State:
    """one matrix-free newton step: gmres on J dz = -r.

    the jacobian of a materialized window is (sites x k) squared per block and is
    never formed.  the jvp is a finite-difference directional derivative of the
    residual, which is a difference quotient and therefore only as accurate as
    the step -- a jax backend replaces this one function with `jax.jvp` and the
    quotient becomes exact.

    real-valued throughout.  `pack_means` splits complex coefficients into real
    and imaginary halves precisely so this krylov space is a real one: gmres over
    a complex field is available, but a real directional derivative fed into a
    complex krylov space is the classic way to get a silently wrong subspace.
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    comps = [cid for cid in _targets(state.layout, couplings) if cid in state]
    if not comps:
        return state

    def flat_residual(vec: np.ndarray) -> np.ndarray:
        s = state.unpack_means(vec, comps)
        r = _residual(s, couplings, basis)
        return np.concatenate([np.concatenate([r[c].real.ravel(), r[c].imag.ravel()])
                               for c in comps])

    x0 = state.pack_means(comps)
    r0 = flat_residual(x0)
    scale = np.linalg.norm(x0) or 1.0

    def jvp(v: np.ndarray) -> np.ndarray:
        eps = 1e-7 * scale / max(np.linalg.norm(v), _EPS)
        return (flat_residual(x0 + eps * v) - r0) / eps

    op = LinearOperator((r0.size, x0.size), matvec=jvp, dtype=float)
    try:
        dz, info = gmres(op, -r0, restart=solve.krylov_restart,
                         maxiter=solve.krylov_maxiter, rtol=1e-4, atol=0.0)
    except TypeError:                       # scipy < 1.12 spells it `tol`
        dz, info = gmres(op, -r0, restart=solve.krylov_restart,
                         maxiter=solve.krylov_maxiter, tol=1e-4, atol=0.0)
    report.newton_calls += 1
    if info != 0:
        report.note(f"gmres returned info={info}; taking the partial correction anyway, "
                    "because a partial newton direction still beats a stalled picard sweep")
    return state.unpack_means(x0 + dz, comps)


# ---------------------------------------------------------------------------
# one window
# ---------------------------------------------------------------------------


def solve_window(state: State, couplings: Sequence[Coupling], basis: TemporalBasis, *,
                 solve: Solve = Solve(), carry: Carry | None = None,
                 match_weight: float = 1.0, clamps: Any = (),
                 window: int = 0) -> tuple[State, StepReport]:
    """solve the boundary-value problem over one window.

    the loop is: sweep picard using the exact inverse of the diagonal linear
    operator, project onto the overlap constraint, apply clamps, measure.  if the
    residual stops falling, hand the current iterate to newton-krylov once and
    resume.  the clamps go *inside* the loop rather than after it because a hard
    clamp is part of the boundary-value problem -- a variable an experimenter has
    pinned is not free for the solver to move, and applying the clamp only at the
    end would let every other variable equilibrate against a value that is about
    to be overwritten.
    """
    from ibm.runtime.intervene import apply_clamps

    s = state.copy()
    report = StepReport(window=window)
    A = _self_operator(s.layout, couplings, basis)
    w = 1j * basis.omega
    inv = {cid: _guarded_inverse(w - a) for cid, a in A.items()}
    singular = {cid: np.abs(w - a) <= 1e-12 for cid, a in A.items()}
    for cid, m in singular.items():
        if bool(np.any(m)):
            s.note(cid, f"{int(np.sum(m))} coefficient(s) have no self-coupling and zero "
                        "frequency: DC is a solvability condition here, not a solve, and it "
                        "was left where the prior put it")

    _match_overlap(s, carry, basis, match_weight)
    apply_clamps(s, clamps, basis)
    r = _residual(s, couplings, basis)
    report.residual0 = report.residual = _norm(r)
    prev = report.residual
    ref = _scale(s, couplings, basis)
    done = lambda x: x <= solve.tol * ref + solve.atol
    stalled_noted = False

    for it in range(1, solve.max_iter + 1):
        before = {cid: s[cid].mean.copy() for cid in A if cid in s}
        d = _drift(s, couplings, basis)
        for cid, target in d.items():
            block = s.layout[cid]
            z = _coefficients(s, cid, basis)
            rest = target - A[cid] * z                       # the non-invertible remainder
            z_new = rest * inv[cid]
            z_new = np.where(singular[cid], z, z_new)
            z = (1.0 - solve.damping) * z + solve.damping * z_new
            s.beliefs[cid] = replace(s[cid], mean=z[:, : block.k] if block.k else z)

        report.joint_gap = _match_overlap(s, carry, basis, match_weight)
        apply_clamps(s, clamps, basis)

        r = _residual(s, couplings, basis)
        report.residual = _norm(r)
        report.iterations = it
        ref = max(ref, _scale(s, couplings, basis))
        if done(report.residual):
            report.converged = True
            break
        move = _norm({cid: s[cid].mean - before[cid] for cid in before})
        size = max(_norm({cid: s[cid].mean for cid in before}), 1.0)
        if move <= solve.x_tol * size:
            report.converged = True
            if report.residual > solve.tol * ref + solve.atol:
                report.note(
                    f"the iterate stopped moving with residual {report.residual:.3e} against "
                    f"a scale of {ref:.3e}: that floor is the mismatch between the dynamics "
                    "and the continuity imposed at the overlap, which §1 says is imposed "
                    "rather than inherited -- it is the price of the joint, not a failed solve")
            break
        stalled = (it >= solve.stall_after
                   and report.residual > solve.stall_ratio * prev)
        if stalled and carry is not None and report.joint_gap > 0.0:
            # the dynamics and the carried tail are inconsistent, so the residual
            # has a floor and no amount of newton will remove it.  say so, with
            # the number, and stop: the floor IS the answer to "what did imposing
            # continuity cost", and iterating against it produces nothing but heat.
            report.limited_by = "continuity"
            report.note(
                f"residual plateaued at {report.residual:.3e} against a scale of "
                f"{ref:.3e} ({report.residual / max(ref, _EPS):.0%}) with a joint gap of "
                f"{report.joint_gap:.3e}.  a driven cyclic window has a unique solution, so "
                "the dynamics and the previous window's tail are two conditions it cannot "
                "both satisfy; this floor is the price of imposing continuity (§1), not a "
                "failure to solve.  a large fraction here means the hop is too long for the "
                "dynamics in it -- shorten it or widen the overlap")
            break
        if stalled and solve.newton and report.newton_calls < solve.max_newton:
            report.method = "picard+newton"
            if not stalled_noted:
                report.note(f"picard stalled at iteration {it} (residual "
                            f"{report.residual:.3e} against {prev:.3e}); this is what a "
                            "strongly recurrent or stiff coupling looks like from inside a "
                            "fixed-point sweep")
                stalled_noted = True
            s = _newton_correction(s, couplings, basis, solve, report)
            _match_overlap(s, carry, basis, match_weight)
            apply_clamps(s, clamps, basis)
            r = _residual(s, couplings, basis)
            report.residual = _norm(r)
            if done(report.residual):
                report.converged = True
                break
        elif stalled and report.newton_calls >= solve.max_newton:
            report.limited_by = "newton"
            report.note(f"stopped after {report.newton_calls} newton solves without reaching "
                        "tolerance; a window that resists this is ill-posed rather than "
                        "under-solved -- check the loop's stability margin")
            break
        prev = report.residual

    if not report.converged:
        report.limited_by = report.limited_by or "iterations"
        if report.limited_by != "continuity":
            report.note("did not reach tolerance; the state returned is the last iterate and "
                        "should be treated as prior-dominated wherever the residual is large")
    return s, report


def _guarded_inverse(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) <= 1e-12, 0.0, 1.0 / np.where(np.abs(x) <= 1e-12, 1.0, x))


# ---------------------------------------------------------------------------
# a run of windows
# ---------------------------------------------------------------------------


def advance(state: State, couplings: Sequence[Coupling], plan: WindowPlan, *,
            solve: Solve = Solve(), clamps: Any = (),
            evidence: Any = (), on_window: Callable[[int, State, StepReport], None] | None = None
            ) -> tuple[State, list[StepReport]]:
    """run the plan, window by window, refusing an acausal one first.

    the refusal happens here and not inside `solve_window` because it is a
    property of the *plan against the graph*, not of any one window: a plan that
    is legal for a cortico-cortical model becomes illegal the moment a
    thalamocortical loop with a 40 ms round trip is materialized into it, and the
    place to discover that is before the first solve rather than after the last.
    """
    plan.refuse_if_acausal(longest_memory(couplings))

    from ibm.runtime.ensemble import reproject_nonlinear
    from ibm.runtime.fuse import fuse
    from ibm.runtime.intervene import apply_clamps

    reports: list[StepReport] = []
    carry: Carry | None = None
    s = state
    for i in range(plan.n_windows):
        s, rep = solve_window(s, couplings, plan.basis, solve=solve, carry=carry,
                              match_weight=plan.match_weight, clamps=clamps, window=i)
        # scalar blocks were never part of the boundary-value problem, so they are
        # integrated afterwards, over the hop and against the window just solved.
        _advance_scalar(s, couplings, plan.hop_s)
        apply_clamps(s, clamps, plan.basis)
        if solve.ensemble > 0:
            reproject_nonlinear(s, couplings, plan.basis, m=solve.ensemble,
                                rng=np.random.default_rng(solve.seed + i))
        else:
            for c in couplings:
                if not c.linear and c.writes in s:
                    s.note(c.writes, f"{c.process}: nonlinear f advanced on the mean only; "
                                     "the belief's width was not reprojected (Solve.ensemble=0)")
        if evidence:
            s = fuse(s, evidence)
        reports.append(rep)
        if on_window is not None:
            on_window(i, s, rep)
        carry = Carry.tail(s, plan.overlap_n, t0_s=carry.t0_s + plan.hop_s if carry else 0.0)
    return s, reports


def step(model: Any, state: State | None = None, *, plan: WindowPlan | None = None,
         solve: Solve = Solve(), clamps: Any = (), evidence: Any = ()
         ) -> tuple[State, list[StepReport]]:
    """the one-call entry point: take a materialized model and run it.

    everything it does is available separately, and is separate because a caller
    fitting parameters wants `solve_window` on a state it owns rather than a
    convenience that allocates.
    """
    layout = Layout.of(model)
    if plan is None:
        plan = WindowPlan.of(_attr(model, "window", "request", default=None) or layout.basis
                             or TemporalBasis(512, 2e-3))
    couplings = couplings_of(model, plan.basis)
    s = state if state is not None else State.prior(layout)
    return advance(s, couplings, plan, solve=solve, clamps=clamps, evidence=evidence)


__all__ = [
    "Carry", "CausalityViolation", "Coupling", "Solve", "SolverFailed", "StepReport",
    "WindowPlan", "advance", "couplings_of", "longest_memory", "solve_window", "step",
]
