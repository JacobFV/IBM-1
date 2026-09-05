"""gaussian evidence fusion, in information form.

pressure moves state, evidence constrains it, and the two compose by different
rules (ARCHITECTURE.md §4).  `State.add` is the first rule; this module is the
second.  keeping them in separate files is not tidiness -- it is the only thing
stopping a measurement from being folded in as though it were a force, which
would let a confident observation accelerate a variable instead of pinning it.

    J = Sigma^-1,   h = J mu,   J' = J + dJ,   h' = h + dh

the arithmetic is trivial; the honesty is not.  three things here are harder than
the two-line update suggests.

**the circular diagonal part is exact and cheap.**  a belief whose covariance is
diagonal in the spectral basis is a stationary process, and evidence about it
composes coefficient by coefficient.  an fMRI run contributes enormous precision
below 0.25 Hz and *literally none* above it, and the band-restricted increment
says so without a special case.

**a non-zero relation is a 2x2 solve, not a scalar one.**  every eigenvalue away
from DC and nyquist is doubly degenerate and its eigenspace is the (cos, sin)
pair at that frequency.  the belief there is a gaussian on a plane: `psd` is its
trace and `relation` is the anisotropy, i.e. phase preference.  fusing two such
beliefs by adding reciprocal psds -- which is what `SpectralGaussian.evidence`
does, and says it does -- averages an evoked response's phase-locking away.  the
exact update is the 2x2 precision addition on that plane, and it lives here
because this is where the batching is known.

**a low-rank increment is not a diagonal one.**  the `factor` term of a spectral
belief contributes `F F^H` across frequencies, and evidence carrying its own
low-rank structure updates through woodbury rather than elementwise.  this is the
same machinery the distillation section below needs, for the same reason: a
correlated error is a rank-deficient constraint.

### distillation precision

a teacher supplies a value where no measurement exists.  that value is evidence
and carries a precision, calibrated from the teacher's own reported accuracy:

    dJ = 1 / ((1 - r2) Var[x])

an encoder reporting r2 = 0.1 -- respectable for stimulus-to-brain prediction --
contributes about a tenth of the precision a perfect measurement would.
distilling it at unit precision is the fastest way to make a model hold a
teacher's biases as firmly as its own measurements.

two corrections are mandatory and both are implemented here rather than left to
the caller.

*reported accuracy holds on the benchmark distribution.*  off it, the variance
must be inflated, and by how much is itself uncertain -- so the inflation is a
process parameter with a prior (`ibm.vocabulary.Prior`), not a constant.  what
enters the update is a draw or the prior's median, and which one it was is
recorded.

*a teacher's errors covary across everything it writes.*  a model writing 10^4
cortical positions does not supply 10^4 independent constraints; its residuals
share structure.  so the increment is **low-rank**: the residual covariance is
modelled as `diag((1-rho) v) + B B^T` with `B` spanning the shared error
structure, and its inverse -- by woodbury -- is a diagonal minus a low-rank
correction.  `effective_constraints` reports what that leaves, and the number is
routinely two orders of magnitude below the count of values written.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field, replace
from typing import Any, Iterable, Sequence

import numpy as np

from ibm.fields.uncertainty.scalar import ScalarGaussian
from ibm.fields.uncertainty.spectral import SpectralGaussian, TemporalBasis
from ibm.registry import REGISTRY
from ibm.vocabulary import FULL, Band, Prior, Provenance

_EPS = 1e-30


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """one gaussian constraint on one component.

    `precision` is `dJ` and not a variance, because precision is what composes.
    the difference matters at the edges: zero precision is "this says nothing",
    which is the correct and expressible statement for a band an instrument
    cannot see, whereas infinite variance is a number nobody can store.

    `kind` is carried all the way through to provenance.  §7 requires that a
    prediction resting on prior-dominated structure not be presented like one
    resting on constrained structure, and a distilled value is prior-dominated
    with extra steps.  a measured and a distilled value never carry the same
    weight by default, and after fusion they are still distinguishable.
    """

    component: str
    mean: np.ndarray
    #: diagonal precision increment, broadcast onto the block's shape.
    precision: np.ndarray
    sites: np.ndarray | None = None
    band: Band = FULL
    #: dJ's non-circular part, for spectral blocks, in the same (trace,
    #: anisotropy) parameterization the belief uses for its covariance -- so
    #: `precision` is `trace(dJ)` on the eigenplane and this is
    #: `(dJxx - dJyy) + 2i dJxy`.  it is the phase preference the *evidence*
    #: carries: an evoked-response template has one, a band power measurement
    #: does not, and |relation| <= precision for the increment to be a real
    #: positive-semidefinite constraint.
    relation: np.ndarray | None = None
    #: low-rank part of dJ: `factor_sign * V V^H`, V of shape (..., k, q).  a
    #: negative sign is how a correlated *error* enters -- see `TeacherPrecision`.
    factor: np.ndarray | None = None
    factor_sign: float = 1.0
    #: precision a *single* value of this kind would contribute on its own, if
    #: its error were independent of everything else the source writes.  it is
    #: the yardstick `effective_constraints` measures against, and carrying it is
    #: what lets "10^4 distilled values are worth 1.1 measurements" be a number
    #: rather than a slogan.  0 means the source did not say.
    single_precision: float = 0.0
    kind: str = "measured"                 # measured | imposed | distilled | surrogate
    source: str = ""
    note: str = ""

    @property
    def rank(self) -> int:
        return 0 if self.factor is None else int(self.factor.shape[-1])

    def effective_constraints(self) -> float:
        """how many independent measurements this evidence is actually worth.

        the quantity is `1^T dJ 1` -- the information this increment carries about
        anything the values have in common, which is what a shared parameter is
        -- divided by the precision one independent value would have contributed
        alone.  for a diagonal increment that is just the count.  for a correlated
        one it saturates: at **rank 1**, with a fraction `rho` of the residual
        variance shared, N values are worth exactly `N / ((1 - rho) + N rho)`,
        which tends to `1/rho` however large N gets.  a teacher writing 10^4
        cortical positions whose errors covary 90% supplies about *one*
        measurement's worth of information about a global parameter, and that is
        the whole content of ARCHITECTURE.md's warning that N distilled values
        are not N constraints.  `scripts/verify_distillation_precision.py`
        reproduces the curve against a dense inverse.

        the closed form is a rank-1 statement and does not survive `error_rank`
        above 1, which is the trap a card author walks into.  `_shared_error_basis`
        models a rank-q teacher's errors as *smooth* rather than uniformly
        shared, so residuals decorrelate with separation and a global parameter
        averages over several independent patches instead of seeing one: at
        rho = 0.9 and N = 10^4 the count is 1.1 at rank 1 and several hundred at
        rank 4.  declare rank 1 unless a measured correlation length says
        otherwise, and read the rank column of that script before choosing.
        """
        d = np.asarray(self.precision, float).reshape(-1)
        tot = float(np.sum(d))
        if self.factor is not None:
            v = np.asarray(self.factor).reshape(d.size, -1)
            col = v.sum(0)
            tot += self.factor_sign * float(np.sum(np.abs(col) ** 2))
        ref = self.single_precision or (float(np.max(d)) if d.size else 0.0)
        return tot / ref if ref > 0 else 0.0

    def describe(self) -> str:
        return (f"{self.component} <- {self.kind}"
                + (f" from {self.source}" if self.source else "")
                + f", band {self.band}, rank {self.rank}, "
                f"{self.effective_constraints():.1f} effective constraints"
                + (f"  ({self.note})" if self.note else ""))


# ---------------------------------------------------------------------------
# the scalar case
# ---------------------------------------------------------------------------


def fuse_scalar(belief: ScalarGaussian, ev: Evidence) -> ScalarGaussian:
    """J' = J + dJ on a scalar block.  exact, and there is nothing else to say.

    a scalar belief has no eigenplane and no cross-frequency structure, so the
    two hard cases below simply do not arise -- which is the same reason the form
    exists.
    """
    idx = _index(ev, belief.mean.shape[0])
    j = 1.0 / np.maximum(belief.var, _EPS)
    h = j * belief.mean
    dj = np.broadcast_to(np.asarray(ev.precision, float), (idx.size,))
    dh = dj * np.broadcast_to(np.asarray(ev.mean, float), (idx.size,))
    j2, h2 = j.copy(), h.copy()
    j2[idx] += dj
    h2[idx] += dh
    if ev.factor is not None:
        v = np.asarray(ev.factor, float).reshape(idx.size, -1)
        j_full = np.zeros((belief.mean.shape[0], v.shape[1]))
        j_full[idx] = v
        # h' = h + dJ mu, and dJ is diagonal PLUS the low-rank term.  dropping
        # the second half here -- while keeping it in J -- is the mistake that
        # makes a correlated teacher pull the posterior mean past the value it
        # asserted, which is not an approximation but an inconsistency.
        m_full = np.zeros(belief.mean.shape[0])
        m_full[idx] = np.broadcast_to(np.asarray(ev.mean, float), (idx.size,))
        h2 = h2 + ev.factor_sign * (j_full @ (j_full.T @ m_full))
        var = _woodbury_diag(1.0 / np.maximum(j2, _EPS), j_full, ev.factor_sign)
        mean = _woodbury_solve(j2, j_full, ev.factor_sign, h2)
        return ScalarGaussian(mean, np.maximum(var, _EPS))
    return ScalarGaussian(h2 / np.maximum(j2, _EPS), 1.0 / np.maximum(j2, _EPS))


# ---------------------------------------------------------------------------
# the spectral case
# ---------------------------------------------------------------------------


def fuse_spectral(belief: SpectralGaussian, ev: Evidence,
                  problems: list[str] | None = None) -> SpectralGaussian:
    """fuse into a spectral belief: diagonal exactly, eigenplane exactly, factor by woodbury.

    the three paths are taken in order of what the evidence actually carries, so
    a plain band-limited power measurement never pays for a 2x2 solve it does not
    need, and an evoked-response constraint never silently loses its phase.
    """
    basis = belief.basis
    k = basis.band_indices(ev.band)
    sites = _index(ev, belief.mean.shape[0])
    if k.size == 0 or sites.size == 0:
        return belief

    mean = belief.mean.copy()
    psd = belief.psd.copy()
    rel = None if belief.relation is None else belief.relation.copy()
    sl = np.ix_(sites, k)

    p0 = np.maximum(belief.total_psd()[sl], _EPS)
    r0 = np.zeros_like(p0, dtype=np.complex128) if belief.relation is None else belief.relation[sl]
    mu0 = belief.mean[sl]

    dp = np.broadcast_to(np.asarray(ev.precision, float), p0.shape)
    mu1 = np.broadcast_to(np.asarray(ev.mean), p0.shape).astype(np.complex128)
    dr = (np.zeros_like(r0) if ev.relation is None
          else np.broadcast_to(np.asarray(ev.relation), p0.shape).astype(np.complex128))

    circular = not np.any(np.abs(r0) > 0) and not np.any(np.abs(dr) > 0)
    if circular:
        # the whole eigenplane is isotropic on both sides, so the 2x2 solve
        # degenerates to a scalar one and there is no reason to pay for it.
        j0 = 1.0 / p0
        j1 = dp
        j = j0 + j1
        mean[sl] = (j0 * mu0 + j1 * mu1) / j
        psd[sl] = 1.0 / j
        if rel is not None:
            rel[sl] = 0.0
    else:
        m, p, r = _eigenplane_fuse(mu0, p0, r0, mu1, dp, dr)
        mean[sl] = m
        psd[sl] = p
        rel = np.zeros_like(belief.mean) if rel is None else rel
        rel[sl] = r

    out = SpectralGaussian(basis, mean, psd, rel, belief.factor)
    if ev.factor is not None:
        out = _apply_low_rank(out, ev, sites, k, problems)
    return out


def _eigenplane_fuse(mu0, p0, r0, mu1, jp1, jr1):
    """the exact 2x2 precision addition on one (cos, sin) eigenplane.

    a complex coefficient z = a + i b is a real 2-vector on the degenerate
    eigenspace, and a symmetric 2x2 matrix on that plane is carried throughout
    ibm-1 as a (trace, anisotropy) pair:

        M(t, s) = [[ (t + Re s)/2 ,  Im s / 2     ],
                   [  Im s / 2    , (t - Re s)/2  ]]

    the belief supplies its *covariance* that way -- `psd` is t and `relation` is
    s -- and the evidence supplies its *precision increment* the same way, which
    is why `Evidence.precision` and `Evidence.relation` are named for dJ and not
    for a variance.  invert the prior, add the increment, invert back, and read
    the moments off.

    this is what makes an evoked response fusable at all.  `psd` alone is a
    circle, and two circles fuse to a circle no matter how sharply phase-locked
    either belief was -- which is exactly the failure the scalar update in
    `SpectralGaussian.evidence` warns about.  the anisotropy is the phase
    preference, and it survives only if the off-diagonal entry is carried through
    both inversions.
    """
    def parts(t, s):
        return (0.5 * (t + s.real), 0.5 * (t - s.real), 0.5 * s.imag)

    def inv2(xx, yy, xy):
        det = xx * yy - xy * xy
        det = np.where(np.abs(det) < _EPS, _EPS, det)
        return yy / det, xx / det, -xy / det

    a0, b0, c0 = inv2(*parts(p0, r0))                     # prior precision
    a1, b1, c1 = parts(np.maximum(jp1, 0.0), jr1)         # evidence precision, as given
    ja, jb, jc = a0 + a1, b0 + b1, c0 + c1

    # h = J mu on the plane, then mu' = J'^-1 h'.
    hx = a0 * mu0.real + c0 * mu0.imag + a1 * mu1.real + c1 * mu1.imag
    hy = c0 * mu0.real + b0 * mu0.imag + c1 * mu1.real + b1 * mu1.imag
    sxx, syy, sxy = inv2(ja, jb, jc)
    mx = sxx * hx + sxy * hy
    my = sxy * hx + syy * hy
    # the result is the inverse of a positive-definite matrix, so |r'| <= p' by
    # construction; clipping here would only mask an increment that was not a
    # valid constraint to begin with.
    return (mx + 1j * my, sxx + syy, (sxx - syy) + 2j * sxy)


def _apply_low_rank(belief: SpectralGaussian, ev: Evidence, sites: np.ndarray,
                    k: np.ndarray, problems: list[str] | None = None) -> SpectralGaussian:
    """woodbury update for a low-rank precision increment.

        (D + s V V^H)^-1 = D^-1 - s D^-1 V (I + s V^H D^-1 V)^-1 V^H D^-1

    applied per site across the retained frequencies, because a teacher's or an
    instrument's correlated error is correlated *across frequency* -- a
    haemodynamic model that is wrong about a subject's coupling gain is wrong at
    every frequency in the same direction, and a diagonal increment cannot say so.
    the result is folded back into `psd`, which loses the off-diagonal part: this
    form carries a diagonal covariance plus an explicit factor, and the
    correction is a rank-q object that no longer fits either slot exactly.  what
    is kept is its diagonal, and the loss is recorded by the caller.
    """
    v = np.asarray(ev.factor)
    if v.ndim == 1:
        v = v[:, None]
    if v.ndim == 2 and v.shape[0] != sites.size:
        v = np.broadcast_to(v, (sites.size,) + v.shape)
    elif v.ndim == 2:
        v = v[:, :, None] if v.shape[1] == k.size else np.broadcast_to(v[:, None, :], (sites.size, k.size, v.shape[1]))
    psd = belief.psd.copy()
    d = np.maximum(psd[np.ix_(sites, k)], _EPS)          # current covariance diagonal
    q = v.shape[-1]
    updated = d.copy()
    for i in range(sites.size):
        vi = np.asarray(v[i], dtype=np.complex128)
        if vi.shape[0] != k.size:
            vi = np.broadcast_to(vi.reshape(1, -1), (k.size, q))
        di = d[i]
        # the belief carries a *covariance* diagonal and the evidence a
        # *precision* increment, so the identity applied is
        #     (D^-1 + s V V^H)^-1 = D - s D V (I + s V^H D V)^-1 V^H D
        # -- D and not D^-1 inside M.  getting that backwards inverts which
        # frequencies the correlated part actually constrains.
        try:
            m = np.eye(q) + ev.factor_sign * np.real(vi.conj().T @ (vi * di[:, None]))
        except np.linalg.LinAlgError:
            continue
        if float(np.min(np.linalg.eigvalsh(m))) <= 0.0 and problems is not None:
            # a negative low-rank increment large enough to make the combined
            # precision indefinite is not a numerical accident: the evidence has
            # claimed to *remove* more certainty than the belief had, which means
            # the shared-error model was mis-specified.  the variance is floored
            # so nothing downstream sees a negative one, and the over-claim is
            # reported rather than absorbed.
            problems.append(
                f"low-rank increment from {ev.source or ev.kind!r} is indefinite at site "
                f"{int(sites[i])}: it subtracts more precision than the belief holds, so the "
                "result was floored.  the teacher's declared error rank or correlated "
                "fraction is too aggressive for this variable")
        raw = di - ev.factor_sign * di * di * np.real(
            np.einsum("kq,qk->k", vi, np.linalg.solve(m, vi.conj().T)))
        if problems is not None and bool(np.any(raw <= 0.0)):
            problems.append(
                f"{int(np.sum(raw <= 0.0))} coefficient(s) at site {int(sites[i])} came out "
                f"with a non-positive variance under {ev.source or ev.kind!r} and were floored")
        updated[i] = np.maximum(raw, _EPS)
    psd[np.ix_(sites, k)] = updated
    return replace(belief, psd=psd)


# ---------------------------------------------------------------------------
# woodbury helpers for the scalar / flat case
# ---------------------------------------------------------------------------


def _woodbury_diag(dinv: np.ndarray, v: np.ndarray, sign: float) -> np.ndarray:
    """diag((D + s V V^T)^-1) with D diagonal, given D^-1.

        (D + s V V^T)^-1 = D^-1 - s D^-1 V M^-1 V^T D^-1,   M = I + s V^T D^-1 V

    so the correction to entry i is `s * dinv_i^2 * (V M^-1 V^T)_ii` -- two
    factors of D^-1 and not three.  written out because the three-factor version
    is dimensionally plausible, numerically close whenever D is near identity,
    and wrong everywhere else.
    """
    q = v.shape[1]
    m = np.eye(q) + sign * (v.T @ (v * dinv[:, None]))
    try:
        y = np.linalg.solve(m, v.T)                       # M^-1 V^T, (q, n)
    except np.linalg.LinAlgError:
        return dinv
    corr = np.einsum("ij,ji->i", v, y)                    # (V M^-1 V^T)_ii
    return np.maximum(dinv - sign * dinv * dinv * corr, _EPS)


def _woodbury_solve(d: np.ndarray, v: np.ndarray, sign: float, h: np.ndarray) -> np.ndarray:
    """(D + s V V^T)^-1 h, D diagonal."""
    dinv = 1.0 / np.maximum(d, _EPS)
    q = v.shape[1]
    m = np.eye(q) + sign * (v.T @ (v * dinv[:, None]))
    try:
        y = np.linalg.solve(m, v.T @ (dinv * h))
    except np.linalg.LinAlgError:
        return dinv * h
    return dinv * h - sign * dinv * (v @ y)


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


def fuse(state, evidence: Evidence | Iterable[Evidence], *, note: bool = True):
    """fold evidence into a state, in place on a copy, recording what it was.

    order does not matter, which is the point of the information form: precision
    addition is commutative and associative, so a run that receives eeg before
    fmri lands in the same place as one that receives them the other way round.
    that is not true of anything that fuses by averaging trajectories, and it is
    why a heterogeneous corpus can be folded in as it arrives.
    """
    evs = [evidence] if isinstance(evidence, Evidence) else list(evidence)
    s = state.copy()
    for ev in evs:
        if ev.component not in s:
            continue
        block = s.layout[ev.component]
        belief = s[ev.component]
        problems: list[str] = []
        if block.uncertainty == "spectral":
            s.beliefs[ev.component] = fuse_spectral(belief, ev, problems)
        else:
            s.beliefs[ev.component] = fuse_scalar(belief, ev)
        if note:
            s.note(ev.component, ev.describe())
            for p in problems[:4]:
                s.note(ev.component, p)
    return s


# ---------------------------------------------------------------------------
# calibrated distillation precision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherPrecision:
    """what a teacher is allowed to assert, and how much of it.

    everything a source card's `distillation` block declares, made arithmetic.
    the card's schema requires `reported_accuracy` and `precision_model` for
    exactly this reason: a teacher writes evidence, so it must state the
    precision that evidence carries before it is permitted to write any.
    """

    r2: float
    #: prior over the variance inflation applied off the benchmark distribution.
    #: a prior and not a constant, because how far off-distribution a use is, and
    #: how badly the teacher degrades there, are both unknown.  `None` means the
    #: card declared none, which is itself a finding and is reported as one.
    ood_inflation: Prior | None = None
    #: rank of the shared error structure.  1 is the minimum honest value -- a
    #: teacher has at least a global bias -- and `None` means the card did not
    #: say, in which case a diagonal increment would be assumed and must not be.
    error_rank: int | None = 1
    #: fraction of residual variance that is shared rather than independent.
    correlated_fraction: float = 0.8
    on_benchmark: str = ""
    verified: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.r2 < 1.0:
            raise ValueError(f"r2 must be in [0, 1): {self.r2}.  r2 = 1 is a teacher claiming "
                             "to be a perfect measurement, which no teacher is")
        if not 0.0 <= self.correlated_fraction <= 1.0:
            raise ValueError("correlated_fraction is a fraction")

    def inflation(self, rng: np.random.Generator | None = None) -> float:
        """the off-distribution variance inflation: a draw, or the prior's median.

        drawn when an ensemble is being propagated, median otherwise, because a
        point estimate of a parameter whose whole role is to be uncertain is a
        compromise and should be visible as one.
        """
        p = self.ood_inflation
        if p is None:
            return 1.0
        if p.dist == "lognormal":
            return float(math.exp(p.loc + (p.scale * rng.standard_normal() if rng else 0.0)))
        if p.dist == "normal":
            return max(1.0, float(p.loc + (p.scale * rng.standard_normal() if rng else 0.0)))
        if p.dist == "uniform":
            return float(rng.uniform(p.loc, p.scale) if rng else 0.5 * (p.loc + p.scale))
        return 1.0

    def residual_variance(self, var_x: np.ndarray, rng=None) -> np.ndarray:
        """(1 - r2) Var[x], inflated.  the denominator of the calibrated precision."""
        return np.maximum(np.asarray(var_x, float), _EPS) * (1.0 - self.r2) * self.inflation(rng)

    def evidence(self, component: str, mean: np.ndarray, var_x: np.ndarray, *,
                 sites: np.ndarray | None = None, band: Band = FULL,
                 rng: np.random.Generator | None = None,
                 basis: TemporalBasis | None = None) -> Evidence:
        """the increment this teacher may contribute, low-rank by construction.

        the residual covariance is modelled as

            R = diag((1 - rho) v) + B B^T,   B B^T carrying rho v

        and dJ = R^-1, which by woodbury is `diag(1/((1-rho) v))` minus a rank-q
        correction.  so the increment is returned as a diagonal precision plus a
        *negative* factor, and `effective_constraints` on the result is the
        number that matters: for rho = 0.8 and rank 1 over 10^4 sites it is
        single digits, not 10^4.

        a caller who wants the naive diagonal figure can set
        `correlated_fraction=0` and will see it, along with the note saying what
        was assumed.
        """
        v = np.asarray(self.residual_variance(var_x, rng), float)
        shape = np.shape(np.asarray(mean))
        v = np.broadcast_to(v, shape) if v.shape != shape else v
        rho = self.correlated_fraction
        q = max(int(self.error_rank or 1), 1)
        indep = np.maximum(v * (1.0 - rho), _EPS).reshape(-1)
        n = indep.size

        factor = None
        dj = 1.0 / indep
        if rho > 0.0 and n > 0:
            # R = diag((1-rho) v) + B B^T, and the increment is R^-1, which by
            # woodbury is `diag(1/d) - U U^T` with U = D^-1 B (I + B^T D^-1 B)^-1/2.
            # writing B itself into `factor` -- the tempting shortcut -- inverts
            # the sign of the effect and makes a correlated teacher look *more*
            # informative than an independent one, which is precisely backwards.
            B = _shared_error_basis(n, q) * np.sqrt(np.maximum(v.reshape(-1) * rho, 0.0))[:, None]
            # rows of B now satisfy sum_j B_ij^2 = rho v_i exactly, so
            # diag(R) = (1-rho) v + rho v = v: the residual variance the teacher's
            # reported accuracy actually implies, split rather than inflated.
            db = B / indep[:, None]
            M = np.eye(q) + B.T @ db
            try:
                L = np.linalg.cholesky(M)
                U = np.linalg.solve(L, db.T).T
            except np.linalg.LinAlgError:
                U = db / math.sqrt(max(float(np.trace(M)) / q, 1.0))
            factor = U.reshape(shape + (q,))
        dj = dj.reshape(shape)

        note = (f"r2={self.r2:g}"
                + (f" on {self.on_benchmark}" if self.on_benchmark else "")
                + (", unverified" if not self.verified else "")
                + f", inflation prior {'declared' if self.ood_inflation else 'MISSING'}"
                + f", error rank {self.error_rank if self.error_rank is not None else 'UNDECLARED'}"
                + f", {rho:.0%} of residual variance treated as shared")
        return Evidence(component=component, mean=np.asarray(mean), precision=dj,
                        sites=sites, band=band, factor=factor, factor_sign=-1.0,
                        single_precision=float(1.0 / max(float(np.mean(v)), _EPS)),
                        kind="distilled", source=self.source, note=note)

    @classmethod
    def from_card(cls, card: dict, source: str = "") -> "TeacherPrecision | None":
        """read a source card's `distillation` block.

        returns None where the card declares none, because a card with
        `use: [distil]` and no distillation block is a card that has not said
        what its evidence is worth and must not be used to write any.
        """
        d = (card or {}).get("distillation")
        if not d:
            return None
        acc = d.get("reported_accuracy") or {}
        val = acc.get("value")
        metric = acc.get("metric", "unresolved")
        if val is None:
            return None
        r2 = variance_explained(metric, float(val))
        if r2 is None:
            return None
        pm = d.get("precision_model") or {}
        rank = pm.get("error_rank")
        rank = int(rank) if isinstance(rank, int) else None
        ood = pm.get("ood_inflation_prior")
        prior = None
        if isinstance(ood, str) and ood.strip():
            # the card carries prose; a lognormal centred at 2x with a wide
            # spread is the weakest defensible reading of "it gets worse and we
            # do not know by how much", and it is marked WEAK so nobody mistakes
            # it for a fitted figure.
            prior = Prior("lognormal", math.log(2.0), math.log(3.0),
                          provenance=Provenance.WEAK, note=ood)
        # a card that measured its own sharing fraction must have that number
        # reach the arithmetic.  reading `error_rank` and not `correlated_fraction`
        # -- which is what this did -- silently substituted the 0.8 default for a
        # measured 0.05, i.e. threw away the only empirical figure in the block
        # and left no trace that it had.  null stays at the default, because the
        # schema is explicit that an unmeasured fraction is to be treated as high
        # rather than as zero.
        rho = pm.get("correlated_fraction")
        rho = cls.correlated_fraction if rho is None else min(max(float(rho), 0.0), 1.0)
        return cls(r2=min(max(r2, 0.0), 0.999), ood_inflation=prior, error_rank=rank,
                   correlated_fraction=rho,
                   on_benchmark=str(acc.get("on_benchmark", "")),
                   verified=bool(acc.get("verified", False)), source=source)


def variance_explained(metric: str, value: float) -> float | None:
    """turn a published benchmark figure into the r2 the precision formula needs.

    this is the step where a distillation pipeline most easily lies to itself,
    because the two numbers are both fractions between a half and one and the
    conversion is a one-line coercion away.  a balanced accuracy of 0.81 is not
    81% of a state variable's variance explained.  it is not any variance
    explained at all: it is a decision rate, and reading it as an r2 hands a
    teacher roughly twice the precision it earned.

    what is defensible is the classical signal-detection route.  put a gaussian
    latent decision variable behind the benchmark's label with equal variances
    under the two classes, and a published figure fixes its separation d':

        auc  = Phi(d' / sqrt2)              ->  d' = sqrt2 Phi^-1(auc)
        bac  = Phi(d' / 2)                  ->  d' = 2 Phi^-1(bac)

    and the squared point-biserial correlation between that latent variable and
    a balanced binary label is

        r2 = d'^2 / (d'^2 + 4)

    the two routes are independent readings of the same table and they agree to
    about 0.01 on every card in data/sources, which is the only check available
    that the model behind them is not badly wrong.

    two limits are stated because they bound what the result may be used for.

    *this is an r2 on the benchmark's latent variable, not on a state variable.*
    "is this recording abnormal" is a coarse clinical summary; the thing a
    teacher is asked to infill is a field value.  the teacher explains at most
    this fraction of the label's variance and an unknown, smaller fraction of the
    state variable's, so what comes out here is an upper bound.  the gap is what
    `ood_inflation` is for, and a card that leans on this conversion should say
    so in its notes.

    *a raw multi-class accuracy cannot be converted at all.*  the probit route
    needs the chance level, and `accuracy` in the source schema carries no arity
    -- 0.645 is excellent on four classes and near chance on two.  guessing
    binary would silently halve or double the precision, so this returns None and
    the teacher is refused rather than approximated.  record an AUROC, which is
    arity-free, or an r2 on the target stream itself.
    """
    from scipy.special import ndtri

    m = (metric or "").strip().lower()
    if m in ("r2", "explained_variance"):
        return min(max(float(value), 0.0), 0.999)
    if m == "pearson_r":
        return min(max(float(value) ** 2, 0.0), 0.999)
    if m == "auc":
        v = min(max(float(value), 0.5 + 1e-6), 1.0 - 1e-9)
        d = math.sqrt(2.0) * float(ndtri(v))
        return min(d * d / (d * d + 4.0), 0.999)
    # "accuracy" and "unresolved" both land here: not convertible, so not used.
    return None


def distillation_precision(r2: float, var_x: np.ndarray, inflation: float = 1.0) -> np.ndarray:
    """dJ = 1 / ((1 - r2) Var[x]), the nominal figure and nothing else.

    exposed because it is the formula ARCHITECTURE.md states, and because seeing
    it beside `TeacherPrecision.evidence` makes the size of the two mandatory
    corrections obvious.  using this directly on a teacher writing many values is
    the mistake the low-rank increment exists to prevent.
    """
    return 1.0 / np.maximum((1.0 - float(r2)) * np.asarray(var_x, float) * inflation, _EPS)


# ---------------------------------------------------------------------------


def _shared_error_basis(n: int, q: int) -> np.ndarray:
    """q orthonormal directions the teacher's errors are assumed to share.

    the first is constant -- the one thing every teacher's residuals certainly
    share is a global bias -- and the rest are the next discrete cosine modes.
    the choice is a claim, so it is worth stating: a model's errors over a
    cortical sheet are smooth, not white, because whatever it got wrong about a
    region it got wrong about that region's neighbours too.  low-frequency modes
    are the cheapest expression of that, they are deterministic (so two runs
    agree), and they are distinct (so the rank is real rather than q copies of
    one direction wearing a hat).

    normalized by *row*, which is the part that is easy to get wrong and changes
    the answer completely.  the shared block of the residual covariance must have
    diagonal `rho * v_i` -- that is what "a fraction rho of this value's residual
    variance is shared" means -- so each row of the basis carries unit norm.
    normalizing by column instead makes each value's shared variance fall as 1/n,
    which quietly turns a correlated teacher back into an independent one exactly
    as n grows, i.e. precisely where the correction was supposed to bite.

    what row normalization does *not* do is keep the sharing global once q > 1.
    the implied correlation between residuals i and j is `rho * cos(angle between
    rows i and j)`: rho for neighbours, falling with separation, and identically
    zero between the first and last index for any q >= 2, because the row at 0 is
    (1, 1, 1, ...) and the row at n-1 is (1, -1, 1, ...).  that is the intended
    smoothness and it is defensible, but it is a correlation *length* and not the
    "fraction shared across everything it writes" that the source-card schema
    describes.  only q = 1 makes `correlated_fraction` mean what the schema says.
    """
    j = np.arange(n)[:, None]
    k = np.arange(q)[None, :]
    b = np.cos(np.pi * (j + 0.5) * k / max(n, 1))
    return b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), _EPS)


def _index(ev: Evidence, n: int) -> np.ndarray:
    return np.arange(n) if ev.sites is None else np.asarray(ev.sites, int).reshape(-1)


def from_observation(obs_id: str, mean: np.ndarray, var: np.ndarray, *,
                     sites: np.ndarray | None = None, source: str = "") -> Evidence:
    """build evidence from a registered observation.

    the observation carries the band over which its likelihood has any precision
    at all, and reading it from the registry rather than from the caller is what
    stops an fMRI run being fused above 0.25 Hz because somebody passed FULL.
    """
    o = REGISTRY.observations.get(obs_id)
    if o is None:
        raise KeyError(f"no registered observation {obs_id!r}")
    cid = o.observes.vars[0]
    return Evidence(component=cid, mean=np.asarray(mean),
                    precision=1.0 / np.maximum(np.asarray(var, float), _EPS),
                    sites=sites, band=o.band & o.observes.band,
                    kind="measured", source=source or obs_id)


__all__ = [
    "Evidence", "TeacherPrecision", "distillation_precision", "from_observation",
    "fuse", "fuse_scalar", "fuse_spectral", "variance_explained",
]
