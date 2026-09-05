"""the width of the belief, pushed through the graph the mean was solved on.

    p(x_{t+dt}) = P_f( p(x_t), p(theta) )

ARCHITECTURE.md 4 says two things about that map and the runtime implemented
neither.  it says the induced distribution depends on p(x) **and** p(theta), and
it says that for a linear f the projection back onto the declared form is
*exact*.  `SpectralGaussian.apply_transfer` is that exact projection -- it scales
psd by |H|^2 and relation by H^2, which is the closed-form push-forward of a
circular gaussian through an LTI filter -- and until this module existed nothing
in the solve path ever called it.  `solve_window` moved means; every psd came
back out of a window bit-identical to the one that went in.  the consequence was
not a small error.  it was a model that carried a mean trajectory and a
decorative covariance, and every epistemic claim resting on that covariance --
evidence fusion, prior-dominated versus evidence-constrained provenance,
calibrated distillation precision, the tier system -- was resting on a number
that no dynamics had ever touched.

### what the push-forward actually is here

`solve_window` does not apply a transfer function; it solves a boundary-value
problem.  at its fixed point every spectral target obeys

    z_O = (i omega - A_O)^-1 sum_c M_c z_{I_c}

where `A_O` is the exactly-invertible self operator `_self_operator` builds and
`M_c` is one coupling's site-mixing times its transfer.  so the map from inputs
to output is `G_c = inv_O M_c`, and *that* is the H whose square belongs on the
psd.  applying each coupling's own `H` to the psd and stopping there would be
wrong by exactly the factor the solve exists to compute.

the same fixed point therefore has to be run on second moments, and it is: the
sweep below is picard on `psd` with the identical operator, damped the same way,
and it converges in one pass per graph depth wherever the graph is a DAG.  where
it is not -- a recurrent loop -- it converges iff the loop's *power* gain is
below one at every frequency, and where that fails it says so and stops rather
than reporting a plausible-looking divergent spectrum.

### the independence bookkeeping, which is the part that can lie

variance does not compose the way precision does.  4 is explicit that pressure
and evidence compose by different rules, and a driven component's width is
pressure: two couplings writing one component contribute *amplitudes* that add
before they are squared, not precisions that add after.  getting that backwards
in either direction is a real error with a sign:

*within one input component the composition is coherent, and it is computed
coherently.*  if two couplings both read `neural.exc.activity` and write
`neural.exc.ampa`, the contribution is `|M_a + M_b|^2 P` and not
`(|M_a|^2 + |M_b|^2) P`.  the cross term is computed exactly, from the hadamard
product of the two mixing matrices -- which is usually *empty*, because the
distance bins of one topology partition its edges, so the pairs that survive are
only the ones where two genuinely different pathways touch the same pair of
sites.  the count of surviving pairs and the fraction of power they carry is
reported rather than assumed small.

*across distinct input components the composition is assumed independent, and
that assumption is not free.*  `neural.exc.ampa` and `neural.exc.nmda` are both
driven by the same activity, so they are strongly correlated, and a component
reading both gets a width that is too small by twice their covariance.  the
declared form carries no cross-component covariance and cannot: it is
(components x sites x k) squared.  so this is a real approximation in the
over-sharpening direction, it is named in the report, and the only honest fix is
a form that carries the block covariance.

*across sites within one component the composition is independent, and that is a
declaration rather than an approximation.*  `SpectralGaussian.psd` is diagonal in
the site axis, so the form itself asserts that two cortical columns' fluctuations
are uncorrelated.  a topology that averages over 160 partners therefore *narrows*
the belief the way averaging independent things narrows it.  that is the declared
hypothesis class behaving as declared, and if it is wrong the fix is a `factor`
term, not a fudge here.

### parameter uncertainty

for a coupling whose transfer is `H(omega; theta)` and whose theta is uncertain,

    Var[H(theta) x] = |H_0|^2 P + sum_p sigma_p^2 |dH/dtheta_p|^2 (|mu|^2 + P)

to first order, and the second term is what this module injects.  4's
factorization `x ~ p(x), theta ~ p(theta)` is what makes the two terms add: the
parameters and the state are a priori independent, so the induced variance of the
product is the sum and there is no cross term at this order.  `dH/dtheta` comes
from a central finite difference on the *registered* transfer, so it costs two
evaluations per (coupling, parameter) and requires nothing to be differentiable
by hand.

parameters shared across couplings -- one conduction velocity behind twelve
distance bins -- are differentiated *coherently*, by perturbing every coupling
that names the parameter at once and taking the difference of the summed drive.
differentiating each bin separately and adding the squares would have dropped the
cross terms and understated a shared parameter's influence by an order of
magnitude on a graph binned this finely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ibm.fields.uncertainty.spectral import SpectralGaussian, TemporalBasis
from ibm.registry import Form

_EPS = 1e-30


# ---------------------------------------------------------------------------
# settings and reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Propagate:
    """how the width is pushed, and what is allowed to be approximated.

    every field here is a decision that changes the answer, which is why none of
    them is buried in a call site.  `coherent=False` in particular is not a
    speed knob with a small cost -- it is the difference between adding
    amplitudes and adding powers, and on a graph where one pathway is binned into
    twelve it is the difference between a width and twelve times a width.
    """

    #: sweeps of the second-moment fixed point.  a DAG needs one per level; a
    #: recurrent graph needs enough for the loop's power gain to converge, and
    #: not converging is a finding rather than a reason to raise this.
    max_iter: int = 40
    tol: float = 1e-8
    damping: float = 1.0
    #: compose couplings that share an input component coherently.  see the
    #: module docstring: off is the wrong rule, kept only so the size of the
    #: error can be measured on a real graph.
    coherent: bool = True
    #: how many coupling pairs the cross-term search may examine per target.
    #: the search itself is O(pairs) sparsity intersections, which is cheap; the
    #: budget exists so a pathological fan-in reports being truncated instead of
    #: quietly taking an hour.
    max_pairs: int = 20_000
    #: propagate the first-order contribution of p(theta).
    parameters: bool = True
    #: relative step for the central difference on theta.
    theta_step: float = 1e-4
    #: the largest sigma_p / |theta_p| the first-order term is allowed to use.
    #:
    #: this is the honest limit of the method and not a safety margin.  the
    #: expansion `Var[H(theta) x] = |H_0|^2 P + sum_p sigma_p^2 |dH/dtheta_p|^2
    #: (|mu|^2 + P)` is first order in sigma, and most of the inventory's
    #: parameters carry `weak(median, 10)` -- a lognormal whose natural-units sd
    #: is two hundred times its median.  a derivative evaluated at the median
    #: says nothing whatever about a distribution that wide, and reporting
    #: `sigma^2 |dH/dtheta|^2` for it would be a large, precise and meaningless
    #: number.  so the sd is truncated here, the truncation is counted in the
    #: report, and the correct answer for a genuinely weak prior is stated
    #: rather than approximated: it needs sampling over p(theta), which is what
    #: `ibm.runtime.ensemble.propagate`'s `theta_prior` argument is for.
    theta_relative_cap: float = 0.5
    #: a psd is never driven below this multiple of the prior it replaces.  zero
    #: by default and deliberately: a driven component whose drive is certain
    #: really does have no width, and clipping that away would hide the most
    #: informative failure this module can report.
    floor_fraction: float = 0.0
    #: stop and report if total power grows by more than this over the sweep.
    blowup: float = 1e12


@dataclass
class TargetReport:
    """where one component's posterior width came from."""

    component: str
    #: fraction of the posterior power contributed by each input component.
    from_input: dict[str, float] = _field(default_factory=dict)
    #: fraction contributed by the first-order theta term.
    from_theta: float = 0.0
    #: fraction contributed by coherent cross terms between couplings sharing an
    #: input.  a large negative number means two pathways partially cancel and an
    #: incoherent sum would have overstated the width.
    from_cross: float = 0.0
    #: sum over retained coefficients of prior and posterior psd, mean over sites.
    prior_power: float = 0.0
    posterior_power: float = 0.0
    channels: int = 0
    cross_pairs: int = 0

    @property
    def ratio(self) -> float:
        return self.posterior_power / max(self.prior_power, _EPS)

    def __str__(self) -> str:
        src = ", ".join(f"{k} {v:.0%}" for k, v in
                        sorted(self.from_input.items(), key=lambda kv: -kv[1])[:3])
        return (f"{self.component:34s} prior {self.prior_power:11.4g} -> posterior "
                f"{self.posterior_power:11.4g}  x{self.ratio:11.4g}   "
                f"[{src}{'' if not self.from_theta else f', theta {self.from_theta:.0%}'}"
                f"{'' if abs(self.from_cross) < 5e-3 else f', cross {self.from_cross:+.0%}'}]")


@dataclass
class PropagationReport:
    """what the push-forward did, and what it had to assume to do it."""

    targets: dict[str, TargetReport] = _field(default_factory=dict)
    iterations: int = 0
    converged: bool = False
    diverged: bool = False
    #: components that were left at their prior because nothing writes them.
    exogenous: tuple[str, ...] = ()
    #: pairs of *distinct* input components feeding one target.  each is a place
    #: where independence was assumed and the posterior may be over-sharpened.
    correlated_inputs: list[tuple[str, str, str]] = _field(default_factory=list)
    notes: list[str] = _field(default_factory=list)
    seconds: float = 0.0

    def note(self, m: str) -> None:
        self.notes.append(m)

    def __str__(self) -> str:
        out = [f"linear push-forward: {self.iterations} sweeps, "
               f"{'converged' if self.converged else 'DIVERGED' if self.diverged else 'NOT converged'}"
               f", {self.seconds:.1f}s"]
        out += ["  " + str(t) for t in self.targets.values()]
        if self.correlated_inputs:
            out.append(f"  {len(self.correlated_inputs)} target/input pairs were composed as "
                       "INDEPENDENT although they share an ancestor; the posterior width is a "
                       "lower bound there")
        out += ["  " + n for n in self.notes]
        return "\n".join(out)


# ---------------------------------------------------------------------------
# sparse / dense helpers
# ---------------------------------------------------------------------------


def _issparse(m: Any) -> bool:
    try:
        from scipy.sparse import issparse
    except Exception:                                    # pragma: no cover
        return False
    return bool(issparse(m))


def _identity_mix(n_out: int, n_in: int):
    """the operator `_fit_sites` applies implicitly when a coupling has no `mix`.

    made explicit rather than special-cased everywhere below: the psd map needs
    |mix|^2 and pairwise hadamard products, and writing a second broadcasting
    path for the no-mix case is how the two paths drift apart.
    """
    from scipy.sparse import identity, csr_matrix
    if n_in == n_out:
        return identity(n_out, format="csr")
    if n_in == 1:
        return csr_matrix(np.ones((n_out, 1)))
    raise ValueError(
        f"a coupling reads {n_in} sites and writes {n_out} with no mixing matrix; the "
        "topology this edge runs over was not materialized")


def _absq(m: Any):
    """|m|^2 elementwise, preserving sparsity."""
    if _issparse(m):
        out = m.multiply(m.conjugate())
        out = out.real if np.iscomplexobj(m) else out
        return out.tocsr()
    return np.abs(np.asarray(m)) ** 2


def _sq(m: Any):
    """m^2 elementwise (no conjugate) -- the map the *relation* transforms under."""
    if _issparse(m):
        return m.multiply(m).tocsr()
    return np.asarray(m) ** 2


def _hadamard(a: Any, b: Any, conj_b: bool = False):
    bb = b.conjugate() if conj_b else b
    if _issparse(a):
        return a.multiply(bb).tocsr()
    if _issparse(bb):
        return bb.multiply(a).tocsr()
    return np.asarray(a) * np.asarray(bb)


def _nnz(m: Any) -> int:
    return int(m.nnz) if _issparse(m) else int(np.count_nonzero(m))


def _pad(x: np.ndarray, k: int) -> np.ndarray:
    """a block carries only its own band; the solve runs on the window's.

    zero above the block's width, matching `_coefficients`: a coefficient a
    component does not carry has no power, and padding with anything else would
    invent it.
    """
    if x.shape[-1] >= k:
        return x[..., :k]
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, k - x.shape[-1])])


# ---------------------------------------------------------------------------
# channels: one coupling's linear map, normalized
# ---------------------------------------------------------------------------


@dataclass
class _Channel:
    process: str
    reads: str
    mix: Any                      # (n_out, n_in), never None
    H: np.ndarray | None          # (k,), (n_in, k) or (n_out, k)
    pre: bool                     # H acts on the input site axis, before mix
    w: np.ndarray                 # (n_out,) soft region weights
    mix_absq: Any = None
    mix_sq: Any = None

    def prepare(self) -> "_Channel":
        self.mix_absq = _absq(self.mix)
        self.mix_sq = _sq(self.mix)
        return self

    def _split(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(factor applied before the mix, factor applied after it).

        `None` rather than a scalar one, so that a channel with no transfer at
        all costs no multiply and nothing downstream has to guess whether a
        broadcast is against the input site axis or the output one.
        """
        if self.H is None:
            return None, None
        return (self.H, None) if self.pre else (None, self.H)


def _channels(couplings: Sequence[Any], layout: Any, basis: TemporalBasis,
              targets: Sequence[str]) -> dict[str, list[_Channel]]:
    """every coupling's contribution to a target, minus the self operator's share.

    a self-diagonal LTI coupling is *already* inside `(i omega - A)^-1`; entering
    it here as well would apply it twice.  a stiff constraint is the opposite
    case -- its `-gamma x_O` half lives in `A` and its `+gamma g(x_I)` half is an
    ordinary drive -- so its reads become channels and its self term does not.
    """
    out: dict[str, list[_Channel]] = {t: [] for t in targets}
    for c in couplings:
        if c.writes not in out or not c.linear or c.self_diagonal:
            continue
        block = layout[c.writes]
        n_out = block.n_sites
        w = np.ones(n_out) if c.weights is None else np.asarray(c.weights, float).reshape(-1)
        for cid in c.reads:
            if cid not in layout:
                continue
            n_in = layout[cid].n_sites
            H = c._H(basis, n_in, n_out)
            pre = False
            if H is not None:
                Hn = np.asarray(H)
                pre = Hn.ndim == 2 and Hn.shape[0] == n_in and n_in != n_out
                if Hn.ndim == 2 and Hn.shape[0] == n_in and n_in == n_out:
                    pre = True          # `spectral_drift` checks the input axis first
                H = Hn
            mix = c.mix if c.mix is not None else _identity_mix(n_out, n_in)
            out[c.writes].append(_Channel(c.process, cid, mix, H, pre, w).prepare())
    return out


# ---------------------------------------------------------------------------
# the second-moment map
# ---------------------------------------------------------------------------


def _push(ch: _Channel, x: np.ndarray, sq: bool) -> np.ndarray:
    """one channel's diagonal contribution: |M|^2 P, or M^2 R for the relation.

    `sq` picks which of the two the array is.  the psd transforms by the squared
    modulus and the relation by the plain square, which is exactly the difference
    between `apply_transfer`'s `|H|^2` and its `H^2` -- and it is the whole
    reason a filter with phase creates phase preference rather than destroying it.
    """
    pre, post = ch._split()
    y = x if pre is None else ((pre ** 2 if sq else np.abs(pre) ** 2) * x)
    y = (ch.mix_sq if sq else ch.mix_absq) @ y
    if post is not None:
        y = y * (post ** 2 if sq else np.abs(post) ** 2)
    return y * (ch.w[:, None] ** 2)


def _push_cross(a: _Channel, b: _Channel, m2: Any, x: np.ndarray, sq: bool) -> np.ndarray:
    """the coherent cross term between two couplings that share an input.

    `2 Re(M_a conj(M_b)) P` for the power and `2 M_a M_b R` for the relation,
    with the site part already intersected into `m2`.  it is signed: two
    pathways can cancel, and an incoherent sum would then have reported a width
    the model does not have.
    """
    pa, qa = a._split()
    pb, qb = b._split()
    pre = None
    for f in (pa, None if pb is None else (pb if sq else np.conj(pb))):
        if f is not None:
            pre = f if pre is None else pre * f
    y = m2 @ (x if pre is None else pre * x)
    if qa is not None:
        y = y * qa
    if qb is not None:
        y = y * (qb if sq else np.conj(qb))
    y = y * (a.w[:, None] * b.w[:, None])
    return 2.0 * (y if sq else y.real)


def _pairs(chans: Sequence[_Channel], budget: int) -> tuple[list[tuple[int, int, Any]], int]:
    """the coupling pairs whose site supports actually intersect.

    the reason this is affordable at all: `binned_topology_couplings` splits one
    topology by distance *quantile*, so its bins partition the edge set and every
    within-family hadamard product is empty.  what survives is the handful of
    places two different pathways -- local recurrence and the lateral arbor, say
    -- write the same pair of sites, and those are exactly the terms that must
    not be dropped.
    """
    out: list[tuple[int, int, Any]] = []
    looked = 0
    for i in range(len(chans)):
        for j in range(i + 1, len(chans)):
            if chans[i].reads != chans[j].reads:
                continue
            looked += 1
            if looked > budget:
                return out, looked
            m2 = _hadamard(chans[i].mix, chans[j].mix, conj_b=True)
            if _nnz(m2) == 0:
                continue
            out.append((i, j, m2))
    return out, looked


# ---------------------------------------------------------------------------
# parameter uncertainty
# ---------------------------------------------------------------------------


def _dH(c: Any, name: str, basis: TemporalBasis, n_in: int, n_out: int,
        step: float) -> np.ndarray | None:
    """dH/dtheta by central difference on the registered transfer.

    a finite difference and not an analytic derivative because `transfer` is
    whatever an implementation declared -- a product of poles, a resonator, a
    dispersive kernel -- and the ontology never promised it would be
    differentiable by anything but arithmetic.  the step is relative, so a rate
    constant of 1e4 and a time constant of 1e-3 get the same conditioning.
    """
    v = c.theta.get(name)
    if v is None or not np.isscalar(v) or not np.isfinite(float(v)) or float(v) == 0.0:
        return None
    h = step * abs(float(v))
    try:
        up = replace(c, theta={**c.theta, name: float(v) + h})._H(basis, n_in, n_out)
        dn = replace(c, theta={**c.theta, name: float(v) - h})._H(basis, n_in, n_out)
    except Exception:
        return None
    if up is None or dn is None:
        return None
    d = (np.asarray(up) - np.asarray(dn)) / (2.0 * h)
    return None if not np.any(np.isfinite(d)) or not np.any(d) else d


def _theta_injection(state: Any, couplings: Sequence[Any], layout: Any,
                     basis: TemporalBasis, targets: Sequence[str],
                     settings: Propagate, report: PropagationReport
                     ) -> dict[str, np.ndarray]:
    """sum_p sigma_p^2 |d(drive)/dtheta_p|^2, before the inverse operator.

    grouped by (target, parameter name) so that one velocity behind twelve
    distance bins is differentiated once, coherently, across all twelve.  the
    alternative -- a derivative per coupling, squared and added -- drops the
    cross terms and understates a shared parameter by roughly the number of bins.
    """
    inj = {t: np.zeros((layout[t].n_sites, basis.k)) for t in targets}
    names: dict[tuple[str, str], list[Any]] = {}
    for c in couplings:
        if c.writes not in inj or not c.linear or not callable(c.transfer):
            continue
        for name in getattr(c, "theta_sd", {}) or {}:
            if name in c.theta:
                names.setdefault((c.writes, name), []).append(c)
    if not names:
        report.note("no coupling declared a `theta_sd`, so p(theta) contributed nothing: "
                    "every parameter was treated as known exactly at its prior median, "
                    "which is the assumption ARCHITECTURE.md 4 explicitly refuses")
        return inj
    counted, capped, worst = 0, 0, 0.0
    for (tgt, name), cs in names.items():
        n_out = layout[tgt].n_sites
        sd = max(float(np.mean([abs(float(c.theta_sd[name])) for c in cs])), 0.0)
        med = float(np.mean([abs(float(c.theta[name])) for c in cs
                             if np.isscalar(c.theta.get(name))] or [0.0]))
        if sd <= 0.0:
            continue
        if med > 0.0 and sd > settings.theta_relative_cap * med:
            capped += 1
            worst = max(worst, sd / med)
            sd = settings.theta_relative_cap * med
        chans: list[_Channel] = []
        for c in cs:
            w = np.ones(n_out) if c.weights is None else np.asarray(c.weights, float).reshape(-1)
            for cid in c.reads:
                if cid not in layout or cid not in state:
                    continue
                n_in = layout[cid].n_sites
                d = _dH(c, name, basis, n_in, n_out, settings.theta_step)
                if d is None:
                    continue
                pre = d.ndim == 2 and d.shape[0] == n_in
                mix = c.mix if c.mix is not None else _identity_mix(n_out, n_in)
                chans.append(_Channel(c.process, cid, mix, d, pre, w).prepare())
        if not chans:
            continue
        counted += 1
        # the coherent mean part: |sum_c dM_c/dtheta mu_I|^2, one complex sum.
        acc = np.zeros((n_out, basis.k), dtype=np.complex128)
        for ch in chans:
            mu = _pad(np.asarray(state[ch.reads].mean), basis.k)
            pre, post = ch._split()
            y = ch.mix @ (mu if pre is None else pre * mu)
            acc = acc + (y if post is None else y * post) * ch.w[:, None]
        contrib = np.abs(acc) ** 2
        # and the state's own width times the same derivative, per the expansion
        # in the module docstring.  incoherent across channels here, which is the
        # same approximation the main sweep makes across input components.
        for ch in chans:
            p = _pad(np.asarray(state[ch.reads].total_psd()), basis.k)
            contrib = contrib + _push(ch, p, sq=False)
        inj[tgt] = inj[tgt] + (sd ** 2) * contrib
    report.note(f"p(theta): {counted} (component, parameter) groups carried a declared prior "
                f"width and were differentiated by central difference on the registered "
                f"transfer")
    if capped:
        report.note(f"p(theta): {capped} of those had sigma/median above "
                    f"{settings.theta_relative_cap:g} (worst {worst:.4g}) and were truncated "
                    "to it.  a first-order term cannot describe a lognormal that wide, and "
                    "the width those parameters really imply is LARGER than what is reported "
                    "below -- unboundedly so for a `weak()` prior, which needs sampling over "
                    "p(theta) rather than a derivative at its median")
    return inj


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


def propagate_linear(state: Any, couplings: Sequence[Any], basis: TemporalBasis, *,
                     settings: Propagate = Propagate()) -> PropagationReport:
    """push psd and relation through every LTI and pure-delay coupling, in place.

    called by `ibm.runtime.step.advance` once the window's mean has converged,
    for the same reason `reproject_nonlinear` is: the mean is a boundary-value
    solve and wants the exact inverse, the width is a push-forward and wants the
    map that solve implies.  doing them together would mean re-deriving the
    operator inside the sweep for a quantity that does not feed back into it.

    the state's psd is *replaced*, not accumulated onto.  a component every one
    of whose drivers is materialized has no width of its own left to keep: its
    prior said what it would be before anything drove it, and after the window it
    is whatever the drive made it.  where that comes out smaller than the prior,
    the model is telling you the component is pinned by its inputs; where it comes
    out larger, the chain is lossy or the parameters are loose.  both are answers.
    """
    import time
    from ibm.runtime.step import _self_operator, _targets

    t0 = time.time()
    layout = state.layout
    report = PropagationReport()
    targets = [t for t in _targets(layout, couplings) if t in state]
    written = {c.writes for c in couplings}
    report.exogenous = tuple(b.component for b in layout
                             if b.uncertainty == "spectral" and b.component not in written)
    if not targets:
        return report

    A = _self_operator(layout, couplings, basis)
    w = 1j * basis.omega
    inv2 = {t: np.abs(np.where(np.abs(w - A[t]) <= 1e-12, np.inf, w - A[t])) ** -2.0
            for t in targets}
    inv_sq = {t: np.where(np.abs(w - A[t]) <= 1e-12, 0.0, 1.0 / (w - A[t])) ** 2
              for t in targets}

    chans = _channels(couplings, layout, basis, targets)
    cross: dict[str, list[tuple[int, int, Any]]] = {}
    if settings.coherent:
        for t in targets:
            cross[t], looked = _pairs(chans[t], settings.max_pairs)
            if looked > settings.max_pairs:
                report.note(f"{t}: the cross-term search hit its budget of "
                            f"{settings.max_pairs} pairs and was truncated; the width "
                            "reported for it drops some coherent terms")
    else:
        cross = {t: [] for t in targets}
        report.note("coherent=False: couplings sharing an input were composed as though "
                    "independent, which is the wrong rule for pressure and is here only so "
                    "the size of the error can be read off")

    for t in targets:
        seen = [ch.reads for ch in chans[t]]
        for i, a in enumerate(sorted(set(seen))):
            for b in sorted(set(seen))[i + 1:]:
                report.correlated_inputs.append((t, a, b))

    inj = (_theta_injection(state, couplings, layout, basis, targets, settings, report)
           if settings.parameters else {t: np.zeros((layout[t].n_sites, basis.k))
                                        for t in targets})

    prior = {t: np.asarray(state[t].total_psd()).copy() for t in targets}
    psd = {t: _pad(prior[t], basis.k).copy() for t in targets}
    rel = {t: _pad(np.asarray(state[t].relation), basis.k).copy()
           if state[t].relation is not None
           else np.zeros((layout[t].n_sites, basis.k), dtype=np.complex128) for t in targets}
    # the exogenous components never move, so their power is read once.
    fixed_psd = {b.component: _pad(np.asarray(state[b.component].total_psd()), basis.k)
                 for b in layout if b.uncertainty == "spectral"
                 and b.component in state and b.component not in targets}
    fixed_rel = {c: (_pad(np.asarray(state[c].relation), basis.k)
                     if state[c].relation is not None else np.zeros_like(v, dtype=np.complex128))
                 for c, v in fixed_psd.items()}

    parts: dict[str, dict[str, float]] = {t: {} for t in targets}
    cross_share = {t: 0.0 for t in targets}
    total0 = sum(float(v.sum()) for v in psd.values())

    for it in range(1, settings.max_iter + 1):
        new_p, new_r = {}, {}
        for t in targets:
            n_out = layout[t].n_sites
            p = np.zeros((n_out, basis.k))
            r = np.zeros((n_out, basis.k), dtype=np.complex128)
            per: dict[str, float] = {}
            for ch in chans[t]:
                pin = psd[ch.reads] if ch.reads in psd else fixed_psd.get(ch.reads)
                rin = rel[ch.reads] if ch.reads in rel else fixed_rel.get(ch.reads)
                if pin is None:
                    continue
                d = _push(ch, pin, sq=False)
                p = p + d
                per[ch.reads] = per.get(ch.reads, 0.0) + float(d.sum())
                if rin is not None:
                    r = r + _push(ch, rin, sq=True)
            xc = 0.0
            for i, j, m2 in cross[t]:
                a, b = chans[t][i], chans[t][j]
                pin = psd[a.reads] if a.reads in psd else fixed_psd.get(a.reads)
                rin = rel[a.reads] if a.reads in rel else fixed_rel.get(a.reads)
                if pin is None:
                    continue
                d = _push_cross(a, b, m2, pin, sq=False)
                p = p + d
                xc += float(d.sum())
                per[a.reads] = per.get(a.reads, 0.0) + float(d.sum())
                if rin is not None:
                    r = r + _push_cross(a, b, m2, rin, sq=True)
            p = p + inj[t]
            if float(inj[t].sum()) > 0.0:
                per["p(theta)"] = float(inj[t].sum())
            p = np.maximum(p, 0.0) * inv2[t]
            r = r * inv_sq[t]
            if settings.floor_fraction > 0.0:
                p = np.maximum(p, settings.floor_fraction * _pad(prior[t], basis.k))
            new_p[t] = (1.0 - settings.damping) * psd[t] + settings.damping * p
            new_r[t] = (1.0 - settings.damping) * rel[t] + settings.damping * r
            parts[t] = per
            cross_share[t] = xc
        move = sum(float(np.abs(new_p[t] - psd[t]).sum()) for t in targets)
        size = max(sum(float(np.abs(new_p[t]).sum()) for t in targets), _EPS)
        psd, rel = new_p, new_r
        report.iterations = it
        if size > settings.blowup * max(total0, _EPS):
            report.diverged = True
            report.note(
                f"the second-moment sweep diverged at iteration {it}: total power grew by "
                f"{size / max(total0, _EPS):.3g}.  a linear graph whose power gain around a "
                "loop exceeds one at any frequency has no stationary width, and the mean "
                "solve does not see this because it is driven and the width is not")
            break
        if move <= settings.tol * size:
            report.converged = True
            break
    else:
        report.note(f"the second-moment sweep did not converge in {settings.max_iter} "
                    "sweeps; the widths below are the last iterate")

    if not report.diverged:
        for t in targets:
            block = layout[t]
            k = block.k or basis.k
            belief = state[t]
            state.beliefs[t] = replace(belief, psd=psd[t][:, :k],
                                       relation=rel[t][:, :k], factor=None)
            tot = max(sum(v for kk, v in parts[t].items() if kk != "p(theta)"), _EPS)
            grand = max(float(psd[t].sum()), _EPS)
            tr = TargetReport(
                component=t,
                from_input={kk: v / max(sum(parts[t].values()), _EPS)
                            for kk, v in parts[t].items() if kk != "p(theta)"},
                from_theta=parts[t].get("p(theta)", 0.0) / max(sum(parts[t].values()), _EPS),
                from_cross=cross_share[t] / tot,
                prior_power=float(_pad(prior[t], basis.k).mean(0).sum()),
                posterior_power=float(psd[t].mean(0).sum()),
                channels=len(chans[t]), cross_pairs=len(cross[t]))
            report.targets[t] = tr
            state.note(t, f"psd pushed through {len(chans[t])} linear channels "
                          f"({len(cross[t])} coherent cross terms); "
                          f"x{tr.ratio:.3g} on the prior")
    report.seconds = time.time() - t0
    return report


# ---------------------------------------------------------------------------
# band-resolved reporting
# ---------------------------------------------------------------------------


def band_table(prior: Any, posterior: Any, basis: TemporalBasis,
               bands: Mapping[str, tuple[float, float]] | None = None,
               components: Sequence[str] | None = None) -> str:
    """prior vs posterior psd per component per band, and the ratio.

    band-resolved and not scalar because a scalar ratio cannot distinguish the
    two things worth distinguishing: a chain that is lossy everywhere, and a
    filter that has moved power from one band to another while conserving it.
    """
    bands = bands or {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
                      "beta": (13.0, 30.0), "gamma": (30.0, 100.0)}
    comps = components or [c for c in prior.layout.components
                           if prior.layout[c].uncertainty == "spectral" and c in posterior]
    head = ("component", "band", "prior psd", "posterior psd", "ratio")
    rows: list[tuple[str, ...]] = []
    for cid in comps:
        b = posterior.layout[cid].basis or basis
        f = b.freqs_hz
        p0 = np.asarray(prior[cid].total_psd())
        p1 = np.asarray(posterior[cid].total_psd())
        n = min(p0.shape[-1], p1.shape[-1], f.size)
        for name, (lo, hi) in bands.items():
            m = (f[:n] >= lo) & (f[:n] < hi)
            if not m.any():
                continue
            a = float(p0[..., :n][..., m].mean())
            c = float(p1[..., :n][..., m].mean())
            rows.append((cid, name, f"{a:.5g}", f"{c:.5g}",
                         f"{c / max(a, _EPS):.4g}"))
    if not rows:
        return "no spectral component carried a comparable band"
    wd = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)]
    out = ["  ".join(h.ljust(x) for h, x in zip(head, wd)),
           "  ".join("-" * x for x in wd)]
    out += ["  ".join(c.ljust(x) for c, x in zip(r, wd)) for r in rows]
    return "\n".join(out)


__all__ = ["Propagate", "PropagationReport", "TargetReport", "band_table",
           "propagate_linear"]
