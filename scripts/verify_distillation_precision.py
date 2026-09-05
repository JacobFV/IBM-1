"""does the calibrated distillation precision actually do what the doc says it does.

ARCHITECTURE.md's "distillation precision" section makes three claims, and every
one of them is the kind of claim that is easy to write and easy to get backwards.
this script checks each numerically against a dense reference, because the whole
argument rests on a sign.

**the residual covariance is diagonal plus low-rank, so the precision is diagonal
MINUS low-rank.**  the covariance `R = diag((1-rho) v) + B B^T` is the natural
object -- a teacher's errors share structure, and shared structure adds variance
-- but it is the *precision* that composes when evidence is fused, and inverting
a diagonal-plus-low-rank matrix by woodbury flips the sign of the correction.
writing `B` straight into the increment, which is the shortcut everyone reaches
for, makes a correlated teacher look *more* informative than an independent one.
part 1 checks `TeacherPrecision.evidence` against `np.linalg.inv(R)` so that the
sign is a measured fact and not an assertion.

**N correlated values are not N constraints.**  part 2 evaluates
`Evidence.effective_constraints` -- the information the increment carries about
anything the values have in common, `1^T dJ 1`, in units of one independent
value's precision -- against the closed form `N / ((1-rho) + N rho)`, and prints
the curve the doc summarises in a single sentence.

**getting it wrong costs something specific.**  part 3 fuses one teacher's output
into one prior twice, calibrated and naive, and reports how far apart the two
posteriors land.  the answer is asymmetric in a way worth knowing: the per-site
marginals barely move, because the low-rank subtraction is spread over n sites
and takes almost nothing from any one of them.  it is the *shared* direction --
the global mean, i.e. exactly the kind of parameter a materialization would fit
across a cortical sheet -- where a naive fusion claims four orders of magnitude
more information than it holds.  a model built that way is not slightly
overconfident; it is confidently wrong about the one thing the teacher's
correlated error was hiding.

run:  ./.venv/bin/python scripts/verify_distillation_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# run from anywhere: the repo root is the package root and there is no install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibm.fields.uncertainty.scalar import ScalarGaussian
from ibm.runtime.fuse import (
    Evidence,
    TeacherPrecision,
    _shared_error_basis,
    fuse_scalar,
)
from ibm.vocabulary import Prior, Provenance


def residual_covariance(v: np.ndarray, rho: float, q: int) -> np.ndarray:
    """R = diag((1-rho) v) + B B^T, built exactly as `TeacherPrecision.evidence` builds it.

    reconstructed here rather than returned by the module because the point of a
    reference implementation is that it does not share code with the thing it is
    checking.  the one piece deliberately shared is `_shared_error_basis`: which
    directions the errors are assumed to share is a modelling choice, not an
    arithmetic one, and disagreeing about it would test nothing.
    """
    n = v.size
    d = np.diag(np.maximum(v * (1.0 - rho), 1e-30))
    if rho <= 0.0:
        return d
    b = _shared_error_basis(n, q) * np.sqrt(np.maximum(v * rho, 0.0))[:, None]
    return d + b @ b.T


def dense_increment(ev: Evidence, n: int) -> np.ndarray:
    """the increment `ev` describes, written out as a full n x n matrix."""
    dj = np.diag(np.broadcast_to(np.asarray(ev.precision, float).reshape(-1), (n,)).copy())
    if ev.factor is not None:
        u = np.asarray(ev.factor, float).reshape(n, -1)
        dj = dj + ev.factor_sign * (u @ u.T)
    return dj


# ---------------------------------------------------------------------------
# part 1 -- woodbury against a dense inverse
# ---------------------------------------------------------------------------


def part1_woodbury_vs_dense(rng: np.random.Generator) -> float:
    """`dJ` from the module vs `inv(R)` from numpy, over rho and rank.

    the variances are drawn rather than held constant on purpose.  with `v`
    uniform, a row-normalized basis and a column-normalized one give the same
    answer, and the row/column distinction is the single most consequential
    detail in `_shared_error_basis` -- so a uniform test would pass either way
    and prove nothing.
    """
    print("=" * 78)
    print("part 1: woodbury increment vs dense inv(R)")
    print("=" * 78)
    print(f"{'n':>6} {'rho':>6} {'q':>3} {'max|dJ-inv(R)|/|inv(R)|':>24} {'max|dJ R - I|':>16}")
    worst = 0.0
    for n in (64, 512, 2048):
        for rho in (0.0, 0.5, 0.9, 0.99):
            for q in (1, 2, 8):
                if rho == 0.0 and q > 1:
                    continue
                var_x = np.exp(rng.normal(0.0, 0.7, n))          # heterogeneous on purpose
                tp = TeacherPrecision(r2=0.3, error_rank=q, correlated_fraction=rho)
                v = tp.residual_variance(var_x)
                r = residual_covariance(v, rho, q)
                ref = np.linalg.inv(r)
                ev = tp.evidence("x", np.zeros(n), var_x)
                got = dense_increment(ev, n)
                rel = float(np.max(np.abs(got - ref)) / max(float(np.max(np.abs(ref))), 1e-300))
                ident = float(np.max(np.abs(got @ r - np.eye(n))))
                worst = max(worst, rel)
                print(f"{n:>6} {rho:>6.2f} {q:>3} {rel:>24.3e} {ident:>16.3e}")
    print(f"\nworst relative disagreement over the sweep: {worst:.3e}")
    return worst


def part1b_fuse_scalar_vs_dense(rng: np.random.Generator) -> None:
    """the same check one level up: `fuse_scalar`'s posterior vs a dense solve.

    `fuse_scalar` does two things the increment alone does not -- it adds the
    low-rank part to `h` as well as to `J`, and it reads the posterior variance
    off `_woodbury_diag`.  the first is what stops a correlated teacher pulling
    the mean past the value it asserted; the second has a `D^-1` count that is
    dimensionally plausible at two, three or four factors and correct at exactly
    two.  neither is visible in part 1.
    """
    print("\n" + "=" * 78)
    print("part 1b: fuse_scalar posterior vs dense (J0 + dJ)^-1")
    print("=" * 78)
    print(f"{'n':>6} {'rho':>6} {'q':>3} {'max|mean err|':>16} {'max|var err|':>16}")
    for n in (64, 512):
        for rho in (0.0, 0.6, 0.95):
            for q in (1, 4):
                if rho == 0.0 and q > 1:
                    continue
                prior_var = np.exp(rng.normal(0.0, 0.5, n))
                prior = ScalarGaussian(rng.normal(0.0, 1.0, n), prior_var)
                teacher_mean = rng.normal(0.5, 1.0, n)
                tp = TeacherPrecision(r2=0.25, error_rank=q, correlated_fraction=rho)
                ev = tp.evidence("x", teacher_mean, np.ones(n))
                post = fuse_scalar(prior, ev)

                j = np.diag(1.0 / prior_var) + dense_increment(ev, n)
                h = prior.mean / prior_var + dense_increment(ev, n) @ teacher_mean
                sigma = np.linalg.inv(j)
                mu = sigma @ h
                print(f"{n:>6} {rho:>6.2f} {q:>3} "
                      f"{float(np.max(np.abs(post.mean - mu))):>16.3e} "
                      f"{float(np.max(np.abs(post.var - np.diag(sigma)))):>16.3e}")


# ---------------------------------------------------------------------------
# part 2 -- what a teacher writing 10^4 values is actually worth
# ---------------------------------------------------------------------------


def part2_effective_constraints() -> None:
    """the headline curve, measured rather than quoted -- and the caveat it needs.

    two different questions hide behind "how much does this teacher tell us", and
    the doc's sentence answers only the first.

    `1^T dJ 1 / (1/v)` is information about a parameter *shared* across the sites
    -- a global gain, a cortex-wide offset, anything the values have in common.
    that is the quantity that saturates at `1/rho`, and it is the quantity the
    architecture warns about.

    `tr(dJ) / (1/v)` is the sum of the *conditional* precisions: what the teacher
    pins down about site i once every other site is already known.  it goes *up*
    with rho, by `1/(1-rho)`, and that is not a contradiction -- splitting the
    residual into a shared part and an independent part makes the independent
    part smaller, so conditioning away the shared part leaves a sharper
    constraint.  a correlated teacher is more informative about differences and
    much less informative about levels.  reporting only the first column would
    overstate the correction; reporting only the second would hide it entirely.

    the rank column is where a card author can hurt themselves, so it is printed.
    """
    print("\n" + "=" * 78)
    print("part 2: effective constraints from n = 10^4 correlated teacher values")
    print("=" * 78)
    n = 10_000
    var_x = np.ones(n)
    print(f"{'rho':>6} {'q':>3} {'shared: 1^T dJ 1':>18} {'rank-1 closed form':>20} "
          f"{'conditional: tr(dJ)':>21}")
    for rho in (0.0, 0.5, 0.9, 0.99):
        for q in (1, 4, 16):
            if rho == 0.0 and q > 1:
                continue
            tp = TeacherPrecision(r2=0.3, error_rank=q, correlated_fraction=rho)
            ev = tp.evidence("x", np.zeros(n), var_x)
            shared = ev.effective_constraints()
            closed = n / ((1.0 - rho) + n * rho) if rho > 0 else float(n)
            d = np.asarray(ev.precision, float).reshape(-1)
            trace = float(np.sum(d))
            if ev.factor is not None:
                u = np.asarray(ev.factor, float).reshape(n, -1)
                trace += ev.factor_sign * float(np.sum(u * u))
            print(f"{rho:>6.2f} {q:>3} {shared:>18.4f} {closed:>20.4f} "
                  f"{trace / ev.single_precision:>21.1f}")

    print("\nat rank 1 the shared column matches n / ((1-rho) + N rho) exactly, which")
    print("tends to 1/rho however large n gets: at rho = 0.9, 10^4 values are worth")
    print("1.11 independent measurements about anything they have in common.")
    part2b_rank_caveat(n)


def part2b_rank_caveat(n: int) -> None:
    """what `error_rank > 1` actually asserts, which is not what the schema says.

    the schema describes `correlated_fraction` as "the fraction rho of the
    teacher's error variance that is SHARED across the values it writes".  at
    rank 1 that is literally true: every pair of residuals correlates at rho.
    above rank 1 it is not, and the difference is large enough to matter to a
    card author choosing a number.

    `_shared_error_basis` builds q discrete cosine modes and normalizes each
    *row* to unit norm, so the correlation between residual i and residual j is
    `rho * cos(angle between rows i and j)` -- rho for neighbours, decaying with
    separation, and for q >= 2 exactly zero between the first and last site,
    because the row at 0 is (1, 1, 1, ...) and the row at n-1 is (1, -1, 1, ...).
    that is a deliberate model of *smooth* error, and it is a defensible one: a
    teacher wrong about one region is wrong about its neighbours.  but it is a
    correlation *length*, not a global sharing fraction, and a global parameter
    averages over ~q independent patches instead of seeing one.

    the consequence is printed below and it is not small.  it also runs the wrong
    way round from the obvious intuition: raising q *lowers* the effective count
    over this range rather than raising it, because at small q the row norms are
    strongly heterogeneous near the edges of the index and that heterogeneity,
    not the correlation length, is what leaves the constant direction
    unconstrained.  a card that declares rank 4 and rho 0.9 is not declaring "one
    measurement's worth"; it is declaring several hundred.  rank 1 is the only
    value for which the declared fraction means what the schema says it means, so
    it is the value the source cards use.
    """
    print("\nrank caveat -- correlation implied by _shared_error_basis at rho = 0.9:")
    print(f"{'q':>4} {'corr(i, i+1)':>14} {'corr(mid, mid+1000)':>21} "
          f"{'corr(0, n-1)':>14} {'effective constraints':>22}")
    rho = 0.9
    for q in (1, 2, 4, 16, 64):
        b = _shared_error_basis(n, q)
        tp = TeacherPrecision(r2=0.3, error_rank=q, correlated_fraction=rho)
        ev = tp.evidence("x", np.zeros(n), np.ones(n))
        print(f"{q:>4} {rho * float(b[0] @ b[1]):>14.4f} "
              f"{rho * float(b[n // 2] @ b[n // 2 + 1000]):>21.4f} "
              f"{rho * float(b[0] @ b[-1]):>14.4f} "
              f"{ev.effective_constraints():>22.1f}")


# ---------------------------------------------------------------------------
# part 3 -- the cost of getting it wrong
# ---------------------------------------------------------------------------


def _shared_posterior_var(prior_var: np.ndarray, ev: Evidence, n: int) -> float:
    """Var of the posterior on the mean of x, (1/n) 1^T x, without forming Sigma.

        Var = (1/n^2) 1^T (D + s V V^T)^-1 1,   D = diag(J0 + diag(dJ))

    computed by woodbury for the same reason the update is: n is 10^4 and the
    dense inverse is 800 MB.  this is the functional the correction was written
    for, so it is the one worth reporting.
    """
    d = 1.0 / prior_var + np.asarray(ev.precision, float).reshape(-1)
    dinv = 1.0 / d
    one = np.ones(n)
    val = float(one @ (dinv * one))
    if ev.factor is not None:
        u = np.asarray(ev.factor, float).reshape(n, -1)
        m = np.eye(u.shape[1]) + ev.factor_sign * (u.T @ (u * dinv[:, None]))
        y = np.linalg.solve(m, u.T @ (dinv * one))
        val -= ev.factor_sign * float((dinv * one) @ (u @ y))
    return max(val, 0.0) / (n * n)


def part3_calibrated_vs_naive(rng: np.random.Generator) -> None:
    """one teacher, one prior, fused two ways, and the gap between the answers.

    the naive fusion is not a strawman.  "the teacher wrote a value, treat it as
    a measurement" is the default behaviour of every distillation loop that does
    not stop to ask what the value is worth, and unit precision against a unit
    prior is exactly what an L2 distillation loss with weight 1 implements.

    two teacher outputs are fused rather than one, because the size of the error
    depends on what the teacher said and quoting only the flattering case would
    be the same failure this script exists to catch.  a *flat* assertion lies
    entirely in the direction the shared error occupies, so the calibrated
    posterior barely moves and the naive one moves halfway; a *varying* assertion
    is mostly orthogonal to it, so both move, and the disagreement is smaller but
    still large.  the honest headline is the pair.
    """
    print("\n" + "=" * 78)
    print("part 3: calibrated vs naive fusion of the same teacher output")
    print("=" * 78)
    n = 10_000
    r2, rho, q = 0.30, 0.90, 1

    prior = ScalarGaussian(np.zeros(n), np.ones(n))       # Var[x] = 1, mean 0
    ood = Prior("lognormal", np.log(2.0), np.log(3.0), provenance=Provenance.WEAK,
                note="off the benchmark distribution the variance at least doubles")
    calibrated = TeacherPrecision(r2=r2, ood_inflation=ood, error_rank=q,
                                  correlated_fraction=rho, source="teacher")
    diag_only = TeacherPrecision(r2=r2, ood_inflation=ood, error_rank=q,
                                 correlated_fraction=0.0, source="teacher")

    print(f"\nsetup: n = {n} values, Var[x] = 1, prior mean 0 sd 1, r2 = {r2}, "
          f"ood inflation {calibrated.inflation():.2f}x (prior median),")
    print(f"       rho = {rho}, error rank {q}.  prior sd on (1/n) sum x_i is "
          f"{1 / np.sqrt(n):.4e}.")

    for label, teacher in (("flat: teacher asserts x = 1 at every site",
                            np.ones(n)),
                           ("varying: teacher asserts x_i ~ N(1, 1)",
                            rng.normal(1.0, 1.0, n))):
        ev_cal = calibrated.evidence("x", teacher, np.ones(n))
        ev_diag = diag_only.evidence("x", teacher, np.ones(n))
        ev_naive = Evidence(component="x", mean=teacher, precision=np.ones(n),
                            single_precision=1.0, kind="distilled", source="teacher",
                            note="unit precision per value: no r2, no inflation, no sharing")

        print(f"\n--- {label} ---")
        print(f"{'fusion':>24} {'eff. constr.':>13} {'rms post mean':>14} "
              f"{'rms post sd':>12} {'info added on (1/n)sum x':>26}")
        out = {}
        for name, ev in (("naive (unit precision)", ev_naive),
                         ("calibrated, diagonal", ev_diag),
                         ("calibrated, low-rank", ev_cal)):
            post = fuse_scalar(prior, ev)
            gvar = _shared_posterior_var(prior.var, ev, n)
            gain = 1.0 / gvar - float(n)                  # posterior minus prior precision
            out[name] = (post, gvar, gain)
            print(f"{name:>24} {ev.effective_constraints():>13.2f} "
                  f"{float(np.sqrt(np.mean(post.mean ** 2))):>14.4f} "
                  f"{float(np.mean(np.sqrt(post.var))):>12.4f} {gain:>26.1f}")

        p_naive = out["naive (unit precision)"][0]
        p_cal = out["calibrated, low-rank"][0]
        g_naive = out["naive (unit precision)"][2]
        g_cal = out["calibrated, low-rank"][2]
        d_mean = p_cal.mean - p_naive.mean
        print(f"  posterior mean, rms disagreement       : "
              f"{float(np.sqrt(np.mean(d_mean ** 2))):.4f}  "
              f"(max {float(np.max(np.abs(d_mean))):.4f})")
        spread = min(float(np.std(p_cal.mean)), float(np.std(p_naive.mean)))
        corr = ("undefined (a flat assertion gives a flat posterior)" if spread < 1e-12
                else f"{float(np.corrcoef(p_cal.mean, p_naive.mean)[0, 1]):.4f}")
        print(f"  posterior mean, correlation naive/cal   : {corr}")
        print(f"  posterior sd ratio, calibrated / naive  : "
              f"{float(np.mean(np.sqrt(p_cal.var)) / np.mean(np.sqrt(p_naive.var))):.4f}")
        print(f"  information claimed about the shared direction: "
              f"naive {g_naive:.1f} vs calibrated {g_cal:.2f}  "
              f"-> naive overstates by {g_naive / max(g_cal, 1e-12):.0f}x")

    print("\nand the sign check: flip factor_sign from -1 to +1 and refuse nothing else.")
    teacher = np.ones(n)
    ev_cal = calibrated.evidence("x", teacher, np.ones(n))
    flipped = Evidence(component="x", mean=teacher, precision=ev_cal.precision,
                       factor=ev_cal.factor, factor_sign=+1.0,
                       single_precision=ev_cal.single_precision, kind="distilled")
    g_cal = 1.0 / _shared_posterior_var(prior.var, ev_cal, n) - n
    g_flip = 1.0 / _shared_posterior_var(prior.var, flipped, n) - n
    print(f"  effective constraints, correct sign : {ev_cal.effective_constraints():>12.2f}")
    print(f"  effective constraints, flipped sign : {flipped.effective_constraints():>12.2f}")
    print(f"  shared-direction information, correct: {g_cal:>11.2f}")
    print(f"  shared-direction information, flipped: {g_flip:>11.2f}")
    print(f"  the flip is a factor of {g_flip / max(g_cal, 1e-12):.0f} in the wrong direction, and it")
    print("  overshoots the naive count as well: correlation read as corroboration.")


def main() -> None:
    rng = np.random.default_rng(20260904)
    part1_woodbury_vs_dense(rng)
    part1b_fuse_scalar_vs_dense(rng)
    part2_effective_constraints()
    part3_calibrated_vs_naive(rng)


if __name__ == "__main__":
    main()
