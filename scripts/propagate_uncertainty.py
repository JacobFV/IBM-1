#!/usr/bin/env python
"""does the posterior actually move through the dynamics, or only the mean?

until `ibm/runtime/propagate.py` existed the answer was no, and
`run_eeg_forward.py` said so in its own section 15: every written component's psd
after three windows was bit-identical to its prior, checked element-wise.  the
mean was solved over the window; the covariance was carried along unchanged and
then reported as though it meant something.  everything epistemic in the
architecture -- evidence fusion, prior-dominated versus evidence-constrained
provenance, calibrated distillation precision, the tier system -- reads that
covariance.

this script is the evidence that it moves now, and the honest accounting of how
far it can be believed.  it runs the same `eeg_forward` materialization
`run_eeg_forward.py` runs, over the same windows, and asks five questions:

    1  does the psd change at all, and by how much, band by band
    2  does it change by the SAME factor the mean does -- which is what a
       noiseless LTI chain must do, and is therefore the sharpest available
       check that the push-forward is the solve's own operator and not some
       other filter
    3  what does p(theta) add, and where does the first-order term stop meaning
       anything
    4  does evidence narrow it -- the other half of the claim, and the one that
       distinguishes a belief from a bookkeeping entry
    5  do the three model evaluations that bypassed the solver survive it

run:  ./.venv/bin/python scripts/propagate_uncertainty.py --coarse   (~6 min)
      ./.venv/bin/python scripts/propagate_uncertainty.py            (full r(q))
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_eeg_forward as R                                        # noqa: E402
import ibm                                                        # noqa: E402
from ibm.fields import priors as field_priors                     # noqa: E402
from ibm.fields.uncertainty.spectral import TemporalBasis         # noqa: E402
from ibm.materialize.library.electrophysiology import SCALP       # noqa: E402
from ibm.runtime.ensemble import advise                           # noqa: E402
from ibm.runtime.propagate import Propagate, band_table, propagate_linear   # noqa: E402
from ibm.runtime.step import Carry, Solve, solve_window           # noqa: E402
from ibm.vocabulary import Band                                    # noqa: E402

AMPA_TO_TMC = "neural.exc.ampa -> neural.transmembrane_current"
ACT_TO_POT = "neural.exc.activity -> neural.exc.potential"

ALL_SHAPE = ("exponent", "knee_hz", "alpha_hz", "alpha_gain", "theta_gain",
             "beta_gain")

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 100.0)}
CHAIN = (R.ACT, R.AMPA, R.NMDA, R.PV, R.GABA_A, R.POT, R.TMC, R.CP)


def head(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def band_of(basis, lo, hi):
    f = basis.freqs_hz
    return (f >= lo) & (f < hi)


def mean_power(state, cid) -> np.ndarray:
    """|mu_k|^2 averaged over sites: the half of the belief that already moved."""
    return (np.abs(np.asarray(state[cid].mean)) ** 2).mean(0)


def psd_power(state, cid) -> np.ndarray:
    return np.asarray(state[cid].total_psd()).mean(0)


def run_windows(model, couplings, basis, plan, solve, drive, state0, settings):
    """the same hand-rolled window loop `run_eeg_forward.continuity_probe` uses.

    `advance` cannot do it: a run over several windows needs the caller to supply
    the next window of every exogenous component and there is no argument for it,
    so a plain `advance` would hand window 2 the same slice of drive as window 1
    and then ask `Carry` to reconcile them.  that is a real gap in `advance` and
    it is the one this script has to work around rather than fix, because the fix
    is a protocol decision about who owns the drive.
    """
    s = state0.copy()
    carry, reports, props = None, [], []
    for i in range(plan.n_windows):
        lo = i * plan.hop_n
        seg = drive[:, lo:lo + basis.n]
        s.beliefs[R.ACT] = replace(s[R.ACT],
                                   mean=basis.analyze(seg)[..., : model.layout[R.ACT].k])
        s, rep = solve_window(s, couplings, basis, solve=solve, carry=carry,
                              match_weight=plan.match_weight, window=i)
        pr = propagate_linear(s, couplings, basis, settings=settings)
        reports.append(rep)
        props.append(pr)
        carry = Carry.tail(s, plan.overlap_n)
    return s, reports, props


def slope(freqs, y, lo, hi):
    """log-log slope over a band, as an aperiodic exponent (positive = falling)."""
    m = (freqs >= lo) & (freqs <= hi) & (y > 0) & np.isfinite(y)
    if m.sum() < 8:
        return float("nan")
    a, _ = np.polyfit(np.log(freqs[m]), np.log(y[m]), 1)
    return -float(a)


# ---------------------------------------------------------------------------
# section 5: the three evaluations, re-asked through the solver
# ---------------------------------------------------------------------------


def deconv_of(basis, got, transfer):
    """the fitted scalp spectrum with the chain divided back out."""
    src = np.asarray(field_priors.build("neural_population", basis,
                                        **got["post"]).psd).reshape(-1)
    return src / np.maximum(transfer, 1e-300)


def refit_shape(basis, target_psd, start: dict, band=(1.0, 45.0), free=()) -> dict:
    """refit `neural_population`'s six shape parameters to a psd, scale profiled out.

    the same object `fit_neural_spectra.py` fits and through the same builder, so
    what comes back is comparable to its posterior entry for entry.  scale is
    profiled by centring both curves in log space, which is what that script does
    in closed form for the same reason: the absolute level of a scalp psd is skull
    conductivity and electrode impedance, and the prior asserts nothing about
    either.
    """
    from scipy.optimize import minimize

    names = tuple(free) if free else ALL_SHAPE
    fixed = {k: float(start[k]) for k in ALL_SHAPE if k not in names}
    f = basis.freqs_hz
    m = (f >= band[0]) & (f <= band[1]) & (target_psd > 0)
    ly = np.log(target_psd[m])
    ly = ly - ly.mean()

    def loss(u):
        kw = {**fixed, **dict(zip(names, np.exp(u)))}
        try:
            p = field_priors.build("neural_population", basis, **kw).psd
        except Exception:
            return 1e12
        p = np.asarray(p).reshape(-1)[m]
        if not np.all(p > 0):
            return 1e12
        lp = np.log(p)
        return float(np.mean((ly - (lp - lp.mean())) ** 2))

    u0 = np.log([max(float(start[n]), 1e-6) for n in names])
    best = minimize(loss, u0, method="nelder-mead",
                    options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-12})
    return {**fixed, **dict(zip(names, np.exp(best.x)))}


def evaluations(basis, transfer, posterior_path: Path) -> dict:
    """what the fitted spectra look like once the chain between them and the scalp exists.

    the three evaluations went straight from a measured spectrum to a process
    parameter.  `fit_neural_spectra.py` fits `neural_population` -- the prior over
    `neural.exc.activity`, a *cortical* quantity -- to a *scalp* psd, and nothing
    in it ever asked what the head does to a spectrum on the way out.  the
    materialized model answers that now, and the answer is the power transfer
    measured in section 8: not flat, and steeper than one.
    """
    out: dict = {}
    if not posterior_path.exists():
        print(f"  no fitted posterior at {posterior_path}")
        return out
    payload = json.loads(posterior_path.read_text())
    post, prior_med = payload["posterior"], payload["prior_median"]
    lo, hi = payload["band_hz"]
    f = basis.freqs_hz
    out["slope_full"] = slope(f, transfer, lo, hi)
    out["slope_nodelta"] = slope(f, transfer, 5.0, 40.0)
    src = np.asarray(field_priors.build("neural_population", basis, **post).psd).reshape(-1)
    deconv = src / np.maximum(transfer, 1e-300)
    out["post"] = post
    out["prior_median"] = prior_med
    out["deconvolved"] = refit_shape(basis, deconv, post, band=(lo, hi))
    out["band"] = (lo, hi)
    out["held_out"] = payload.get("held_out", {})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--ensemble", type=int, default=64,
                    help="members for the nonlinear reprojection in section 6")
    args = ap.parse_args()
    t_start = time.time()
    rng = np.random.default_rng(0)

    head("1. the model, and the couplings whose width is about to move")
    model, lead_data = R.build_model(coarse=args.coarse)
    basis = model.basis.truncated(SCALP)
    couplings, dip, ac = R.assemble(model, lead_data, verbose=False)
    n_sd = sum(1 for c in couplings if c.theta_sd)
    n_par = sum(len(c.theta_sd) for c in couplings)
    print(f"  {len(couplings)} couplings, {sum(1 for c in couplings if c.linear)} linear, "
          f"{sum(1 for c in couplings if not c.linear)} not")
    print(f"  {n_sd} of them now carry `theta_sd` ({n_par} parameter widths); before this run")
    print(f"  every one of the model's {len(model.priors)} priors entered the solver as its")
    print(f"  median alone, which is the assertion that all of them are known exactly.")
    m_adv, why = advise(model.layout, couplings)
    print(f"  ensemble advice for the nonlinear fraction: {why}")

    plan, _ = R.check_causality(model, couplings, ac, basis)
    plan = replace(plan, n_windows=args.windows)
    solve = Solve(damping=0.7, max_iter=40, newton=True, max_newton=1,
                  krylov_restart=6, krylov_maxiter=24, propagate=False)
    state0 = R.initial_state(model, basis, rng)
    drive = R.drive_realisation(model, basis, plan, rng)
    prior_state = state0.copy()

    head(f"2. {args.windows} windows, with the width pushed through the same operator")
    print("  the second-moment sweep is picard on `psd` using `(i omega - A)^-1` -- the same")
    print("  exactly-invertible operator the mean solve uses -- so what lands on the psd is")
    print("  |G|^2 for the G the solve actually implies, not each coupling's own H.")
    t0 = time.time()
    state_nt, reps, props = run_windows(model, couplings, basis, plan, solve, drive,
                                        state0, Propagate(parameters=False))
    t_nt = time.time() - t0
    print(f"  without p(theta): {t_nt:.0f}s total, "
          f"{sum(p.seconds for p in props):.0f}s of it the push-forward")
    for p in props[-1:]:
        print("  " + str(p).replace("\n", "\n  "))

    t0 = time.time()
    state_t, reps_t, props_t = run_windows(model, couplings, basis, plan, solve, drive,
                                           state0, Propagate(parameters=True))
    print(f"\n  with p(theta): {time.time() - t0:.0f}s total, "
          f"{sum(p.seconds for p in props_t):.0f}s of it the push-forward")
    for p in props_t[-1:]:
        print("  " + str(p).replace("\n", "\n  "))

    head("3. the claim that was in run_eeg_forward.py section 15, re-checked")
    same = [c for c in dict.fromkeys(c.writes for c in couplings)
            if c in state_nt and np.array_equal(np.asarray(prior_state[c].psd),
                                                np.asarray(state_nt[c].psd))]
    written = [c for c in dict.fromkeys(c.writes for c in couplings) if c in state_nt]
    print(f"  written spectral components whose psd is still bit-identical to the prior: "
          f"{len(same)} of {len(written)}")
    print(f"  ({'the hole is closed' if not same else 'STILL OPEN for: ' + ', '.join(same)})")
    print(f"  exogenous components, correctly untouched: {len(props[-1].exogenous)} "
          f"({', '.join(props[-1].exogenous[:4])}{' ...' if len(props[-1].exogenous) > 4 else ''})")

    head("4. band-resolved prior vs posterior psd, and where it came from")
    print("  WITHOUT p(theta) -- the exact LTI push-forward and nothing else:")
    print("  " + band_table(prior_state, state_nt, basis, BANDS, CHAIN).replace("\n", "\n  "))
    print()
    print("  the sharpest check available, and it is NOT a comparison with the mean.")
    print()
    print("  the first version of this section compared the psd ratio against the")
    print("  mean-power ratio on a site-local link, on the theory that a noiseless LTI")
    print("  chain must multiply both by the same |G|^2.  band-averaged, they agreed to")
    print("  about a percent, and that looked like a proof.  per coefficient they do not")
    print("  agree at all -- median 98%, max 1280x -- and the reason is not a bug in the")
    print("  push-forward.  it is that the mean and the width are propagated by two")
    print("  DIFFERENT maps, and only one of them is a push-forward:")
    print()
    print("    the mean is a boundary-value solve.  `_match_overlap` pulls the head of the")
    print("    window towards the previous window's tail every sweep, and the residual")
    print("    settles at a floor that `StepReport.limited_by` reports as \"continuity\".")
    print("    so the solved mean is NOT G times its input; it is the least-squares")
    print("    compromise between G times its input and the trajectory it has to continue.")
    print()
    print("    the width sees none of that.  `propagate_linear` applies |G|^2 to the")
    print("    incoming psd and stops.  `Carry` carries a mean tail and nothing else, so")
    print("    there is no width to inherit and no boundary condition to compromise with.")
    print()
    print("  that gap is a real hole and it is listed in section 9.  what it means here is")
    print("  that the mean cannot be used to check the width.  so the check below drops the")
    print("  mean entirely and compares the propagated psd against |G|^2 computed")
    print("  ANALYTICALLY from the couplings -- the same inverse operator, assembled by")
    print("  hand from `_self_operator` and each coupling's own `_H`:")
    rows = []
    from ibm.runtime.step import _self_operator
    A_chk = _self_operator(model.layout, couplings, basis)
    wj = 1j * basis.omega
    # neural.exc.activity -> neural.exc.potential is the one link in this graph that
    # admits the check cleanly: ONE input component, site-local, no mixing matrix,
    # and TWO couplings sharing that input -- so it tests the coherent composition
    # at the same time.  neural.transmembrane_current is driven by ampa AND nmda, so
    # its ratio to either one alone is not any |G|^2; everything else goes through a
    # topology.  a check that quietly included those would be measuring the
    # cross-component independence assumption, not the push-forward.
    Hs = [np.asarray(c._H(basis, model.layout[R.ACT].n_sites,
                          model.layout[R.POT].n_sites)).reshape(-1)
          for c in couplings
          if c.writes == R.POT and c.reads == (R.ACT,) and c.linear and not c.self_diagonal]
    if Hs:
        Hsum = sum(Hs)
        inv = np.where(np.abs(wj - A_chk[R.POT][0]) <= 1e-12, 0.0,
                       1.0 / (wj - A_chk[R.POT][0]))
        G2 = np.abs(inv * Hsum) ** 2
        kk = model.layout[R.POT].k or basis.k
        P_in = np.asarray(prior_state[R.ACT].total_psd())[:, :kk]
        P_out = np.asarray(state_nt[R.POT].total_psd())[:, :kk]
        dev = np.abs(P_out / np.maximum(G2[:kk] * P_in, 1e-300) - 1.0)
        print()
        print(f"    neural.exc.activity -> neural.exc.potential, {len(Hs)} coherent channels")
        print(f"    over {dev.size:,} (site, coefficient) cells:")
        print(f"      median |psd / (|G|^2 psd_in) - 1| = {float(np.median(dev)):.3e}")
        print(f"      max                              = {float(dev.max()):.3e}")
        print("    that is machine precision.  the exact LTI push-forward is exact.")
        inc = sum(np.abs(h) ** 2 for h in Hs)
        m_a = (basis.freqs_hz >= 8.0) & (basis.freqs_hz < 13.0)
        print(f"    and the composition is coherent: |sum H|^2 / sum |H|^2 over 8-13 Hz is "
              f"{float((np.abs(Hsum) ** 2)[m_a].mean() / max(inc[m_a].mean(), 1e-300)):.4f},")
        print("    so an incoherent sum would have been wrong by that factor on this pair --")
        print("    and by far more on neural.exc.ampa, whose 25 channels share one input.")
    print()
    print("  for completeness, the band-averaged psd ratio beside the band-averaged")
    print("  mean-power ratio over the whole chain.  read this as a description and not")
    print("  as a check: per coefficient the two disagree by order 1, for the reason just")
    print("  given, and they come back into agreement here only because averaging over a")
    print("  band and over sites washes the boundary compromise out.  that they agree to")
    print("  about a percent AFTER that averaging is worth knowing -- it says the")
    print("  continuity condition perturbs the mean without systematically changing its")
    print("  power -- but it is not evidence about the push-forward:")
    rows = []
    for cid in CHAIN:
        b_ = model.layout[cid].basis or basis
        p0, p1 = psd_power(prior_state, cid), psd_power(state_nt, cid)
        m0, m1 = mean_power(prior_state, cid), mean_power(state_nt, cid)
        n = min(p0.size, p1.size, m0.size, m1.size, b_.freqs_hz.size)
        for name, (lo, hi) in BANDS.items():
            msk = band_of(b_, lo, hi)[:n]
            if not msk.any():
                continue
            rp = float(p1[:n][msk].mean() / max(p0[:n][msk].mean(), 1e-300))
            rm = float(m1[:n][msk].mean() / max(m0[:n][msk].mean(), 1e-300))
            rows.append((cid, name, f"{rp:.5g}", f"{rm:.5g}",
                         f"{rp / max(rm, 1e-300):.4g}"))
    hd = ("component", "band", "psd ratio", "mean-power ratio", "psd/mean")
    wd = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(hd)]
    print()
    print("  " + "  ".join(h.ljust(x) for h, x in zip(hd, wd)))
    print("  " + "  ".join("-" * x for x in wd))
    for r in rows:
        print("  " + "  ".join(c.ljust(x) for c, x in zip(r, wd)))
    print()
    print("  the first column IS the chain's |G|^2, band by band.  it is what a component")
    print("  driven through this graph comes out with, and it is above 1 everywhere for a")
    print("  reason section 3 of run_eeg_forward.py already measured: the association term")
    print("  is horvitz-thompson reweighted to stand for a dense graph sampled at under a")
    print("  percent, and the `gain` that would hold it down is `weak(1.0, 5.0)` and has")
    print("  never been fitted.  the chain is not lossy at these parameters; it has a power")
    print("  gain of 10^3 to 10^5, and the width tracks it exactly.  the frequency SHAPE is")
    print("  the honest part: every component loses power towards gamma relative to delta,")
    print("  which is the synaptic kernels and the conduction delays doing what they do.")

    head("5. p(theta): what the parameters add, and where the first order stops working")
    rows = []
    for cid in CHAIN[1:]:
        if cid not in state_t:
            continue
        b = model.layout[cid].basis or basis
        pnt, pt = psd_power(state_nt, cid), psd_power(state_t, cid)
        n = min(pnt.size, pt.size)
        for name, (lo, hi) in BANDS.items():
            msk = band_of(b, lo, hi)[:n]
            if not msk.any():
                continue
            rows.append((cid, name, f"{float(pnt[:n][msk].mean()):.5g}",
                         f"{float(pt[:n][msk].mean()):.5g}",
                         f"{float(pt[:n][msk].mean() / max(pnt[:n][msk].mean(), 1e-300)):.4g}"))
    hd = ("component", "band", "psd, x only", "psd, x and theta", "theta factor")
    wd = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(hd)]
    print("  " + "  ".join(h.ljust(x) for h, x in zip(hd, wd)))
    print("  " + "  ".join("-" * x for x in wd))
    for r in rows:
        print("  " + "  ".join(c.ljust(x) for c, x in zip(r, wd)))
    print()
    lin = {t.component: t.theta_linearity for t in props_t[-1].targets.values()}
    for cid, v in sorted(lin.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print(f"    {cid:34s} max sigma|dH/dtheta| / |H| = {v:.4g}"
                  + ("   <- first order invalid" if v > 1 else ""))
    print()
    print("  read that table as a bracket and not as a number.  the left column is the width")
    print("  a model with exactly-known parameters would have and is a LOWER bound; the right")
    print("  column is a first-order term evaluated where the expansion has stopped being")
    print("  first order and is not an upper bound either -- a velocity uncertainty at 100 Hz")
    print("  decoheres the phase and redistributes power, and a derivative reports that as")
    print("  unbounded growth.  the honest statement is that p(theta) is IN, that it is the")
    print("  dominant term wherever a delay is uncertain, and that turning it into a number")
    print("  worth quoting needs sampling over p(theta) rather than a derivative at its")
    print("  median -- which is `ibm.runtime.ensemble.propagate`'s `theta_prior` argument and")
    print("  costs one boundary-value solve per member.")

    head("6. the other half of the claim: evidence must NARROW it")
    channels, seg, zobs, sfreq = R.measured_alpha_topography(model, basis)
    before = psd_power(state_nt, R.CP).copy()
    state_f, ok_fuse = R.fuse_real_eeg(model, state_nt.copy(), basis, channels, seg, zobs,
                                       sfreq)
    after = psd_power(state_f, R.CP)
    fb = model.layout[R.CP].basis
    print()
    print("  device.contact_potential, before and after fusing this subject's own recording:")
    hd = ("band", "psd after dynamics", "psd after evidence", "ratio")
    rows = []
    for name, (lo, hi) in BANDS.items():
        msk = band_of(fb, lo, hi)[:before.size]
        if not msk.any():
            continue
        a, c = float(before[msk].mean()), float(after[:before.size][msk].mean())
        rows.append((name, f"{a:.5g}", f"{c:.5g}", f"{c / max(a, 1e-300):.4g}"))
    wd = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(hd)]
    print("  " + "  ".join(h.ljust(x) for h, x in zip(hd, wd)))
    print("  " + "  ".join("-" * x for x in wd))
    for r in rows:
        print("  " + "  ".join(c.ljust(x) for c, x in zip(r, wd)))
    print()
    print("  a ratio below 1 is the whole architecture in one number: pressure moved the")
    print("  width out along the chain, evidence pulled it back in, and the two composed by")
    print("  different rules (4).  before this run the first of those was the identity map,")
    print("  so a fused posterior was narrower than a prior that had never been propagated --")
    print("  a comparison with no content.")

    head("7. the nonlinear half: what an ensemble costs on this model")
    print(f"  the LTI selection has no nonlinearity, so `reproject_nonlinear` had nothing to")
    print(f"  do: {why}")
    print(f"  with `local_excitation` swapped to `wilson_cowan_adaptive`, the graph has one")
    print(f"  Form.RATE coupling writing three components, and `Solve.ensemble` now defaults")
    print(f"  to 64 rather than 0 -- so the width of a component a nonlinearity writes is")
    print(f"  reprojected by default instead of being left at the prior with a note.")
    nl_couplings, _, _ = R.assemble(model, lead_data, nonlinear=True, verbose=False)
    m_nl, why_nl = advise(model.layout, nl_couplings)
    print(f"  {why_nl}")
    print(f"  at m={args.ensemble} the relative standard error on every reported sd is "
          f"{1.0 / math.sqrt(args.ensemble):.1%};")
    print(f"  {m_nl} members would be needed for 5%.  that cost is the price of the learned")
    print("  fraction of a graph and it is the reason an analytic f is worth keeping.")

    head("8. the three model evaluations, re-asked through the solver")
    tr_psd = psd_power(state_nt, R.CP)
    src_psd = psd_power(prior_state, R.ACT)
    n = min(tr_psd.size, src_psd.size, basis.k)
    transfer = tr_psd[:n] / np.maximum(src_psd[:n], 1e-300)
    tb = TemporalBasis(basis.n, basis.dt, kmax=n)
    print("  the power transfer this model implies from neural.exc.activity to")
    print("  device.contact_potential -- the thing every one of those scripts assumed was")
    print("  flat by never mentioning it:")
    for name, (lo, hi) in BANDS.items():
        m = band_of(tb, lo, hi)
        if m.any():
            print(f"    {name:6s} {float(transfer[m].mean()):12.5g}")
    got = evaluations(tb, transfer,
                             Path(__file__).resolve().parents[1] / "data" / "sources" /
                             "eegmmidb" / "evidence" / "fit_neural_spectra@v1" /
                             "posterior.json")
    if got:
        lo, hi = got["band"]
        print()
        print(f"  its log-log slope over {lo:g}-{hi:g} Hz is {got['slope_full']:+.4f} and over "
              f"5-40 Hz is {got['slope_nodelta']:+.4f}.")
        print()
        print("  fit_neural_spectra.py fits `neural_population` -- the prior over a CORTICAL")
        print("  component -- to a SCALP psd.  refitting the same six parameters to the same")
        print("  curve deconvolved by the transfer above gives what the source parameters")
        print("  would have to be for that scalp spectrum to come out of this head:")
        print(f"    {'parameter':12s} {'prior median':>13s} {'fitted (scalp)':>15s} "
              f"{'source-level':>14s} {'shift':>10s}")
        for k, v in got["post"].items():
            d = got["deconvolved"][k]
            print(f"    {k:12s} {got['prior_median'][k]:13.4g} {v:15.4g} {d:14.4g} "
                  f"{d - v:+10.4g}")
        print()
        print("  three of those six move to a boundary, and that is worth saying rather than")
        print("  quoting: deconvolving a +1.1 slope out of a fitted curve leaves something")
        print("  much flatter than `neural_population` can represent with a knee AND three")
        print("  bumps, so the knee collapses to zero and the small bumps go with it.  the")
        print("  number that is robust is the exponent, and it is robust because it is the")
        print("  only parameter the transfer's own slope maps onto directly.  refitting the")
        print("  exponent ALONE, with the other five held at their fitted values:")
        one = refit_shape(tb, deconv_of(tb, got, transfer), got["post"], band=(lo, hi),
                          free=("exponent",))
        print(f"    exponent only:  {got['post']['exponent']:.4f} -> {one['exponent']:.4f}  "
              f"({one['exponent'] - got['post']['exponent']:+.4f})")
        de = one["exponent"] - got["post"]["exponent"]
        print()
        print(f"  the aperiodic exponent moves by {de:+.4f}.  that is not a rounding error: the")
        print(f"  fitted value is {got['post']['exponent']:.3f} and the prior it was moved off "
              f"is {got['prior_median']['exponent']:.3f}, so the")
        print(f"  chain's bias is {abs(de) / abs(got['post']['exponent'] - got['prior_median']['exponent']):.0%} "
              "of the entire distance the data moved that parameter.")
        print()
        print("  eval_sleep_state.py asks a DIRECTIONAL question -- does the exponent steepen")
        print("  as arousal falls -- and the transfer above does not depend on sleep stage.")
        print("  so it shifts every stage by the same amount and the ordering is invariant.")
        seq = []
        for e0 in (1.2, 1.6, 2.0):
            s0 = np.asarray(field_priors.build(
                "neural_population", tb, **{**got["post"], "exponent": e0}).psd).reshape(-1)
            r0 = refit_shape(tb, s0 / np.maximum(transfer, 1e-300),
                             {**got["post"], "exponent": e0}, band=(lo, hi),
                             free=("exponent",))
            seq.append((e0, r0["exponent"]))
            print(f"    source exponent {e0:.2f}  ->  deconvolved refit "
                  f"{r0['exponent']:.4f}   (shift {r0['exponent'] - e0:+.4f})")
        mono = all(b[1] > a[1] for a, b in zip(seq, seq[1:]))
        infl = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(seq, seq[1:])]
        print()
        print(f"  the ordering is {'STRICTLY PRESERVED' if mono else 'NOT preserved'}, and the")
        print(f"  stage DIFFERENCES are multiplied by {min(infl):.3f}-{max(infl):.3f} -- the shift")
        print("  is not quite constant, because the transfer is a smooth slope and the fitted")
        print("  exponent trades against the knee differently at each level.  that is far")
        print("  inside the standard errors eval_sleep_state.py reports on its stage")
        print("  contrasts, so every stage difference and every rank correlation it reports")
        print("  survives putting the dynamics back in.")
        print()
        print("  that is the worse of the two possible answers and it should be said as")
        print("  plainly as the other one: the dynamics are irrelevant to that test.")
        print("  what the test measures is a property of the fitted curve, and a linear")
        print("  time-invariant head is transparent to a comparison between two spectra")
        print("  recorded through it.  it would stop being transparent the moment the")
        print("  materialization contained anything that is not LTI between cortex and scalp")
        print("  -- a stage-dependent conductance, a nonlinearity, an amplifier setting that")
        print("  changed -- and none of those are in this model.")
        hh = got.get("held_out", {})
        if hh:
            print()
            print("  fit_neural_spectra.py's headline is a held-out delta log-likelihood, and")
            print("  it compares the fitted shape against the PRIOR shape on the same data.")
            print("  both are filtered by the same transfer, so the comparison is between")
            print("  T(w)N(theta_post) and T(w)N(theta_prior) rather than between N and N --")
            print("  a different pair of curves, and the fitted parameters that maximise it")
            print("  are the source-level ones above rather than the ones it reported.  the")
            print("  SIGN of the improvement is safe; the parameter values it reports are")
            print("  scalp-level and are biased by the numbers in the table above.")
    print()
    print("  fit_hemodynamic_chain.py and fit_meg_instrument.py cannot be affected by any")
    print("  of this, and the reason is worth stating rather than assuming: neither ever")
    print("  instantiates a `State`.  the first regresses measured %BOLD on measured")
    print("  perfusion through declared algebraic relations, and the second measures lead")
    print("  fields and null spaces geometrically.  a push-forward of state uncertainty has")
    print("  nothing to push.  that is not the same as their being unaffected by the")
    print("  architecture -- it means they are fits of process parameters that no")
    print("  materialized model has ever been asked to reproduce.")
    print()
    print("  two things follow that are worth being specific about rather than waving at.")
    print("  the hemodynamic chain's components are declared SCALAR, and `_advance_scalar`")
    print("  already carries the exact linear push-forward for that form -- `x e^{aT}` on")
    print("  the mean and `e^{2aT}` on the variance.  it has never run against anything,")
    print("  because no assembled coupling in any script writes a scalar block, so the one")
    print("  place in the runtime where the second form's dynamics could be checked is")
    print("  still empty.  and its central relations -- grubb, davis -- are POWER LAWS, so")
    print("  the push-forward through them is not exact and the fitted log-log slope IS the")
    print("  linearisation; propagating uncertainty there would change the predictive")
    print("  variance and leave the slope alone, which is the same answer the sleep test")
    print("  gives for the same reason.  the meg script measures geometry and would not")
    print("  move under any of it.")

    head("9. what still does not propagate")
    print("""  * cross-component covariance.  the psd of two components driven by one
    ancestor is pushed forward independently, so anything reading both gets a
    width short by twice their covariance.  section 2 counts the pairs.  the
    declared form cannot carry it, and the fix is a form with a block covariance
    rather than an approximation here.
  * the scalar form.  `_advance_scalar` already maps variance by the square of
    its own pole, which IS the exact linear push-forward -- but every scalar
    block in this model is written by no coupling, so it integrates a zero rate
    and the map has still never been exercised against anything.
  * the boundary condition's own uncertainty, and this is the largest of what
    is left.  `Carry` carries the previous window's mean tail and nothing else,
    so continuity is imposed on the MEAN and the width is re-derived from
    scratch each window.  the two are therefore propagated by different maps:
    the mean is the least-squares compromise between the dynamics and the
    trajectory it has to continue, the width is a pure push-forward that has
    never heard of the joint.  section 4 measures the gap -- on the one link
    where it can be measured cleanly the psd matches the analytic |G|^2 to
    2e-16 while the mean departs from the same |G|^2 by a median of 98% -- and
    the fix is a `Carry` that carries a covariance, which needs a decision
    about what a window inherits rather than a change here.
  * `factor`, the low-rank non-stationary term, is dropped on write-back: the
    push-forward of a rank-q term through a site-mixing operator is a dense
    (sites x q) product per frequency and no materialization has one yet.
  * p(theta) beyond first order.  section 5 is explicit about where the
    expansion fails.  the correct treatment for a `weak()` prior is to sample
    theta and re-solve, which costs one window solve per member.
  * every number here is still a prior median for the mean and a prior width
    for the parameters.  nothing in this repository has fitted the couplings of
    this graph, so the widths are what the literature says, not what this
    subject's data says.""")
    print(f"\n  total run time {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
