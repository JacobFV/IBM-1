"""known answers for the hippocampal module.

    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/gate_hippocampus.py

Verdicts are fixed here, in the code, before they run.  A failure is recorded as FAILED and
is not re-run with a kinder bar (CLAUDE.md).

  H0  BOUNDED.  Every rate stays in [0, 1] at three timesteps under extreme input.
  H1  IDEMPOTENCE.  Same generator, twice, bit-identical.
  H0b ALIVE.  Every population must be running within a factor of three of the sparsity it
      is DECLARED to run at.  Without this the first run of these gates reported five
      numbers about a hippocampus sitting at 0.0005 Hz -- and H2 PASSED, because two
      all-zero vectors are uncorrelated and "less correlated than the input" was true in
      the most useless possible way.  Every gate after this one is VOID if it fails.
  H2  SEPARATION.  Two entorhinal patterns correlated at ~0.8 must come back LESS correlated
      in the dentate.  Reported as both numbers and the drop; the dentate's expansion is the
      only reason this can happen, so it is the gate on the expansion being real.
  H3  COMPLETION, scored as SPECIFICITY.  Cue each stored pattern in turn with a 40%
      fragment: the retrieved state's best-matching pattern must be the one that was cued,
      for at least 5 of 6.  The control is a SHUFFLED recurrent matrix (permuted once,
      outside), which must fall to chance.

      This gate was rewritten after its first run, and the reason is worth keeping.  It
      originally required that an UNSTORED pattern not be completed to anything -- and the
      model duly completed one to overlap 1.000, which read as a failure.  It is not: an
      autoassociative network cued with a novel pattern falling into a stored attractor is
      the defining behaviour, not a bug, and CA1's comparison against entorhinal input is
      where novelty is supposed to be detected.  The control was mis-specified.  What the
      single-pattern overlap CANNOT see is a network with one global attractor, which scores
      +1.000 on every cue; only asking whether cue k retrieves pattern k can.

      The bar of 5/6 was chosen after seeing 5/6 on the exploratory patterns, so it is
      judged here on a FRESH pattern set and fresh cue seeds, with the exploratory draw
      excluded (CLAUDE.md: a post-hoc bar is pre-registered and tested on new seeds).
  H4  THETA.  CA1 must carry a 3-8 Hz rhythm with prominence > 0 -- prominence, because
      `peak_frequency` returns a number whether or not there is a peak.
  H4b THETA IS A LOOP.  Cutting the hippocampal return limb to the septum must WEAKEN it.
      If theta is unchanged, it was an imposed drive and the module's docstring is lying.
  H5  THETA-GAMMA.  Gamma amplitude in CA1 must be coupled to theta phase, MI > 0, with a
      relabelling control (phase series rolled by a large lag) that must collapse.
  H6  RIPPLES.  With the septal tone withdrawn, CA1 must show a 140-200 Hz rhythm with
      prominence > 0, and that rhythm must be ABSENT in the theta state.  One circuit, two
      regimes: the state contrast is the gate, not the band alone.
  H7  SENSITIVITY.  Every declared constant swept +-50%, re-measuring the theta frequency and
      the completion overlap.  Constants the module claims set something must move it;
      everything inert over a 3x sweep is listed.  Five times in two days an inert parameter
      has meant a mechanism that was not wired in.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm import spectral as SP                                          # noqa: E402
from ibm.hippocampus import Hippocampus, HippocampalPriors              # noqa: E402

DT = 0.001
FS = 1.0 / DT
THETA_BAND = (3.0, 8.0)
GAMMA_BAND = (30.0, 100.0)
RIPPLE_BAND = (140.0, 200.0)


def build(seed=0, priors=None):
    return Hippocampus(seed=seed, priors=priors)


def cue_pattern(pat, frac, generator):
    """a fragment: `frac` of the pattern's active units kept, the rest zeroed."""
    keep = (torch.rand(pat.shape, generator=generator) < frac).float().to(pat.device)
    return pat * keep


def settle(h, cue, steps=400, noise_gen=None, septal_tone=0.35):
    """drive CA3 with a cue for 60 ms, then let it run free and report where it lands."""
    st = h.init_state(1)
    st["ca3"] = cue.clone()
    for t in range(steps):
        z = None
        if noise_gen is not None:
            z = torch.randn(1, h.n_ca3, generator=noise_gen)
        drive = cue if t < 60 else None
        if drive is not None:
            st["ca3"] = torch.maximum(st["ca3"], drive)
        st = h.step(st, DT, ec=None, septal_tone=septal_tone, noise=z)
    return st["ca3"]


def gate_bounded(h):
    worst = 0.0
    for dt in (0.0005, 0.002, 0.01):
        for amp in (0.0, 3.0):
            g = torch.Generator().manual_seed(5)
            st = h.init_state(1)
            ec = torch.full((1, h.n_ec), amp)
            for _ in range(int(1.0 / dt)):
                z = torch.randn(1, h.n_ca3, generator=g)
                st = h.step(st, dt, ec=ec, noise=z)
            for k in ("dg", "ca3", "ca1", "sub", "i_dg", "i_ca3", "i_ca1", "ms_E"):
                v = st[k]
                worst = max(worst, float(max((v.min() * -1).clamp_min(0).max(),
                                             (v - 1).clamp_min(0).max())))
    return {"ok": worst <= 0.0, "worst_excursion": worst}


def gate_idempotent(h):
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    ec = torch.zeros(1, h.n_ec)
    ec[0, :20] = 1.0
    a, _ = h.rollout(600, DT, ec=ec, noise_gen=g1)
    b, _ = h.rollout(600, DT, ec=ec, noise_gen=g2)
    same = all(torch.equal(a[k], b[k]) for k in a)
    return {"ok": same, "max_abs_diff": max(float((a[k] - b[k]).abs().max()) for k in a)}


def corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm() + 1e-9))


def gate_alive(h):
    """H0b.  Is the circuit running where it is declared to run?"""
    pr = h.pr
    g = torch.Generator().manual_seed(3)
    ec = (torch.rand(1, h.n_ec, generator=g) < pr.rate_ec).float()
    st = h.init_state(1)
    for _ in range(800):
        st = h.step(st, DT, ec=ec, septal_tone=1.0,
                    noise=torch.randn(1, h.n_ca3, generator=g))
    want = {"dg": pr.rate_dg, "ca3": pr.rate_ca3, "ca1": pr.rate_ca1}
    out, ok = {}, True
    for k, target in want.items():
        got = float((st[k] > 0.2).float().mean())
        lo, hi = target / pr.rate_tolerance, target * pr.rate_tolerance
        inside = lo <= got <= hi
        ok &= inside
        out[k] = {"active_fraction": got, "declared": target,
                  "window": [lo, hi], "inside": inside}
    return {"ok": ok, "populations": out,
            "septum_mean": float(st["ms_E"].mean()),
            "rule": "every population within a factor of 3 of its declared sparsity"}


def gate_separation(h):
    g = torch.Generator().manual_seed(3)
    base = (torch.rand(h.n_ec, generator=g) < 0.25).float()
    flip = (torch.rand(h.n_ec, generator=g) < 0.12).float()
    other = torch.clamp(base + flip - 2 * base * flip, 0, 1)      # ~12% of bits flipped
    ec_in = corr(base, other)
    outs = []
    for pat in (base, other):
        st = h.init_state(1)
        for _ in range(250):
            st = h.step(st, DT, ec=pat[None, :], septal_tone=1.0)
        outs.append(st["dg"][0].clone())
    dg_out = corr(*outs)
    sp = h.sparsity(outs[0][None, :])
    # the dentate must actually be CODING.  Without this clause a silent dentate passes:
    # two all-zero vectors have correlation 0, which is duly "less correlated than the
    # input".  That is exactly what happened on the first run of this gate.
    alive = 0.002 <= sp <= 0.25
    return {"ok": bool(dg_out < ec_in - 0.05 and alive),
            "ec_input_corr": ec_in, "dg_output_corr": dg_out, "drop": ec_in - dg_out,
            "dg_sparsity": sp, "dg_is_coding": alive,
            "rule": "the dentate code must be less correlated than its input by > 0.05 AND "
                    "the dentate must be active (0.2%-25% of units), because a silent "
                    "dentate is trivially uncorrelated"}


def gate_completion(h_builder):
    """H3.  Does cue k retrieve pattern k?  Fresh patterns and fresh cue seeds."""
    PSEED, CUE0, NOISE0 = 4242, 900, 950          # NOT the exploratory draws (7, 21, 22)
    g = torch.Generator().manual_seed(PSEED)
    h = h_builder()
    pats = h.make_patterns(6, active=0.08, generator=g)
    h.store(pats)

    def retrieve(model, k, frac=0.4):
        gg = torch.Generator().manual_seed(CUE0 + k)
        cue = pats[k][None, :] * (torch.rand(1, model.n_ca3, generator=gg) < frac).float()
        ng = torch.Generator().manual_seed(NOISE0 + k)
        out = settle(model, cue, noise_gen=ng, septal_tone=0.1)
        ov = model.overlap(out)[0]
        return int(ov.argmax()), float(ov[k]), float(ov.max()), \
            float((out > 0.2).float().mean())

    rows, hits = {}, 0
    for k in range(6):
        win, own, best, act = retrieve(h, k)
        hits += int(win == k)
        rows[f"cue_{k}"] = {"retrieved": win, "overlap_with_cued": own,
                            "best_overlap": best, "active_fraction": act}
    # control: the same weights, permuted ONCE, outside both arms
    h_shuf = h_builder()
    h_shuf.stored = h.stored.clone()
    perm = torch.randperm(h.n_ca3, generator=torch.Generator().manual_seed(99))
    h_shuf.W_rec = h.W_rec[perm][:, perm].clone()
    hits_shuf = sum(int(retrieve(h_shuf, k)[0] == k) for k in range(6))
    return {"ok": hits >= 5 and hits_shuf <= 2,
            "specificity": f"{hits}/6", "control_shuffled_specificity": f"{hits_shuf}/6",
            "by_cue": rows,
            "rule": "cue k must retrieve pattern k for >= 5 of 6, on a pattern set and cue "
                    "seeds not used to choose the parameters; the shuffled-weight control "
                    "must manage at most 2 of 6"}


def theta_measure(h, ms_feedback=True, septal_tone=1.0, seed=17, steps=12000,
                  ec_rate=None):
    """Measured WITH an entorhinal input, because theta is a rhythm of a circuit that is
    running.  The first version passed `ec=None` and measured a hippocampus in the dark at
    0.245 Hz, where a prominence of -0.21 says nothing about the rhythm and everything about
    there being no activity to carry one."""
    g = torch.Generator().manual_seed(seed)
    rate = h.pr.rate_ec if ec_rate is None else ec_rate
    ec = (torch.rand(1, h.n_ec, generator=torch.Generator().manual_seed(seed + 1))
          < rate).float()
    tr, _ = h.rollout(steps, DT, ec=ec, septal_tone=septal_tone, noise_gen=g,
                      ms_feedback=ms_feedback, record=("ca1", "ca3"))
    x = tr["ca1"].mean(-1)
    freqs, psd = SP.welch_psd(x, FS, nperseg=4096)
    return {"peak_hz": float(SP.peak_frequency(psd, freqs, *THETA_BAND).mean()),
            "prominence": float(SP.peak_prominence(psd, freqs, *THETA_BAND).mean()),
            "mean_rate_hz": float(tr["ca1"].mean() * 100)}, tr


def gate_theta(h):
    m, _ = theta_measure(h)
    return {"ok": m["prominence"] > 0.0, **m,
            "rule": "prominence > 0 in 3-8 Hz; the frequency is reported beside it, never "
                    "alone"}


def gate_theta_is_a_loop(h):
    intact, _ = theta_measure(h, ms_feedback=True)
    cut, _ = theta_measure(h, ms_feedback=False)
    drop = intact["prominence"] - cut["prominence"]
    return {"ok": drop > 0.05, "intact": intact, "feedback_cut": cut, "drop": drop,
            "rule": "cutting the hippocampal return limb must cost more than 0.05 decades "
                    "of theta prominence; if it costs nothing, theta was an imposed drive"}


def gate_theta_gamma(h):
    g = torch.Generator().manual_seed(23)
    ec = (torch.rand(1, h.n_ec, generator=torch.Generator().manual_seed(24))
          < h.pr.rate_ec).float()
    tr, _ = h.rollout(12000, DT, ec=ec, septal_tone=1.0, noise_gen=g, record=("ca1",))
    x = tr["ca1"].mean(-1)
    mi = float(SP.pac_mi(x, FS, THETA_BAND, GAMMA_BAND).mean())
    # A PHASE-RANDOMISED surrogate: same power spectrum, no cross-frequency structure.
    # The first version of this control spliced a rolled copy onto the second half of the
    # signal, which returned a number IDENTICAL to the real one to every decimal -- a
    # control that cannot differ is not a control.  The phases are drawn from an explicit
    # generator, once.
    gg = torch.Generator().manual_seed(77)
    X = torch.fft.rfft(x, dim=-1)
    ph = torch.rand(X.shape, generator=gg) * 2 * math.pi
    ph[..., 0] = 0.0
    surrogate = torch.fft.irfft(X.abs() * torch.exp(1j * ph), n=x.shape[-1], dim=-1)
    mi_ctrl = float(SP.pac_mi(surrogate, FS, THETA_BAND, GAMMA_BAND).mean())
    return {"ok": mi > 0.0 and mi > mi_ctrl * 1.5, "mi": mi,
            "mi_phase_randomised_control": mi_ctrl,
            "rule": "MI > 0 and at least 1.5x a phase-randomised surrogate with the same "
                    "power spectrum"}


def gate_ripples(h):
    out = {}
    for name, tone in (("theta_state", 1.0), ("quiet_state", 0.0)):
        g = torch.Generator().manual_seed(29)
        # the quiet state is quiet, not dark: a weaker entorhinal input, not none.  Ripples
        # happen in a resting animal, not in an absent one.
        rate = h.pr.rate_ec if tone > 0.5 else 0.4 * h.pr.rate_ec
        ec = (torch.rand(1, h.n_ec, generator=torch.Generator().manual_seed(31)) < rate).float()
        tr, _ = h.rollout(12000, DT, ec=ec, septal_tone=tone, noise_gen=g, record=("ca1",))
        x = tr["ca1"].mean(-1)
        freqs, psd = SP.welch_psd(x, FS, nperseg=2048)
        out[name] = {
            "ripple_prominence": float(SP.peak_prominence(psd, freqs, *RIPPLE_BAND).mean()),
            "ripple_peak_hz": float(SP.peak_frequency(psd, freqs, *RIPPLE_BAND).mean()),
            "theta_prominence": float(SP.peak_prominence(psd, freqs, *THETA_BAND).mean()),
            "mean_rate_hz": float(tr["ca1"].mean() * 100)}
    ok = (out["quiet_state"]["ripple_prominence"] > 0.0
          and out["quiet_state"]["ripple_prominence"]
          > out["theta_state"]["ripple_prominence"] + 0.1)
    return {"ok": ok, "states": out,
            "rule": "ripple prominence > 0 in the quiet state AND more than 0.1 decades "
                    "above the theta state"}


def gate_sensitivity(h_builder):
    base = HippocampalPriors()
    fields = [f for f in base.__dataclass_fields__ if not f.startswith("d_") and f != "sigma"]

    def measure(pr):
        h = h_builder(pr)
        g = torch.Generator().manual_seed(31)
        ec = (torch.rand(1, h.n_ec, generator=torch.Generator().manual_seed(32))
              < h.pr.rate_ec).float()
        tr, _ = h.rollout(6000, DT, ec=ec, septal_tone=1.0, noise_gen=g, record=("ca1",))
        x = tr["ca1"].mean(-1)
        freqs, psd = SP.welch_psd(x, FS, nperseg=2048)
        return (float(SP.peak_frequency(psd, freqs, *THETA_BAND).mean()),
                float(SP.peak_prominence(psd, freqs, *THETA_BAND).mean()))

    f0, p0 = measure(base)
    rows = []
    for name in fields:
        v = getattr(base, name)
        kw = {k: getattr(base, k) for k in base.__dataclass_fields__}
        flo, plo = measure(HippocampalPriors(**{**kw, name: v * 0.5}))
        fhi, phi = measure(HippocampalPriors(**{**kw, name: v * 1.5}))
        rows.append({"param": name, "value": v, "hz_span": abs(fhi - flo),
                     "prominence_span": abs(phi - plo)})
    rows.sort(key=lambda r: -(r["hz_span"] + r["prominence_span"]))
    claimed = {r["param"]: r["hz_span"] for r in rows
               if r["param"] in ("tau_ms_a", "g_ms_a", "w_ms_EE")}
    inert = [r["param"] for r in rows
             if r["hz_span"] < 0.1 and r["prominence_span"] < 0.05]
    return {"ok": any(v > 0.3 for v in claimed.values()),
            "baseline_hz": f0, "baseline_prominence": p0,
            "claimed_septal_clock_hz_span": claimed, "inert": inert,
            "rule": "at least one of the septal constants the module claims sets theta must "
                    "move it by > 0.3 Hz over a 3x sweep",
            "sweep": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/gate_hippocampus.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rec = {"script": "scripts/gate_hippocampus.py", "dt": DT, "gates": {}}

    def save():
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=str)

    save()
    h = build()
    order = [("H0_bounded", lambda: gate_bounded(h)),
             ("H0b_alive", lambda: gate_alive(h)),
             ("H1_idempotent", lambda: gate_idempotent(h)),
             ("H2_separation", lambda: gate_separation(h)),
             ("H3_completion", lambda: gate_completion(lambda pr=None: build(priors=pr))),
             ("H4_theta", lambda: gate_theta(h)),
             ("H4b_theta_is_a_loop", lambda: gate_theta_is_a_loop(h)),
             ("H5_theta_gamma", lambda: gate_theta_gamma(h)),
             ("H6_ripples", lambda: gate_ripples(h)),
             ("H7_sensitivity", lambda: gate_sensitivity(lambda pr=None: build(priors=pr)))]
    ok_all = True
    for name, fn in order:
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001 -- recorded, never silently skipped
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        rec["gates"][name] = r
        save()
        ok_all &= bool(r.get("ok"))
        if name == "H0b_alive" and not r.get("ok"):
            rec["verdict"] = ("VOID: the circuit is not running at its declared sparsity, "
                              "so no downstream gate's number means anything")
            save()
            print("\nVOID -- H0b failed.  Stopping: a gate run on a dead circuit produces "
                  "numbers that look like results.", flush=True)
            return 1
        head = {k: v for k, v in r.items() if k not in ("sweep", "states", "by_cue")}
        print(f"[{'PASS' if r.get('ok') else 'FAIL'}] {name}: {head}", flush=True)
        if name == "H3_completion" and "by_cue" in r:
            for k, v in r["by_cue"].items():
                print(f"      {k}: retrieved {v['retrieved']}  "
                      f"overlap with cued {v['overlap_with_cued']:+.3f}  "
                      f"best {v['best_overlap']:+.3f}  active {v['active_fraction']:.3f}",
                      flush=True)
        if name == "H6_ripples" and "states" in r:
            for k, v in r["states"].items():
                print(f"      {k}: ripple {v['ripple_prominence']:+.3f} at "
                      f"{v['ripple_peak_hz']:.1f} Hz, theta {v['theta_prominence']:+.3f}, "
                      f"rate {v['mean_rate_hz']:.2f} Hz", flush=True)
        if name == "H7_sensitivity" and "sweep" in r:
            for row in r["sweep"][:6]:
                print(f"      {row['param']:14s} hz span {row['hz_span']:6.2f}  "
                      f"prominence span {row['prominence_span']:6.3f}", flush=True)
            print(f"      inert: {', '.join(r['inert']) or 'none'}", flush=True)
    rec["all_gates_ok"] = ok_all
    save()
    print(f"\n{'ALL GATES PASS' if ok_all else 'SOME GATES FAILED'} -- wrote {a.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
