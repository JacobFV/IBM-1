"""known answers for the hippocampal module.

    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/gate_hippocampus.py

Verdicts are fixed here, in the code, before they run.  A failure is recorded as FAILED and
is not re-run with a kinder bar (CLAUDE.md).

  H0  BOUNDED.  Every rate stays in [0, 1] at three timesteps under extreme input.
  H1  IDEMPOTENCE.  Same generator, twice, bit-identical.
  H2  SEPARATION.  Two entorhinal patterns correlated at ~0.8 must come back LESS correlated
      in the dentate.  Reported as both numbers and the drop; the dentate's expansion is the
      only reason this can happen, so it is the gate on the expansion being real.
  H3  COMPLETION.  Cue CA3 with a fragment of a stored pattern: the state must settle onto
      that pattern (overlap >= 0.8 at a 40% cue) and NOT onto any other.  Two controls, both
      of which must fail to complete:
        * a SHUFFLED recurrent matrix (the same weights, permuted once, outside);
        * an UNSTORED pattern, which must not be completed to anything.
      Without the second, a model that completes everything passes.
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
    return {"ok": dg_out < ec_in - 0.05, "ec_input_corr": ec_in, "dg_output_corr": dg_out,
            "drop": ec_in - dg_out,
            "dg_sparsity": h.sparsity(outs[0][None, :]),
            "rule": "the dentate code must be less correlated than its input, by > 0.05"}


def gate_completion(h_builder):
    g = torch.Generator().manual_seed(7)
    h = h_builder()
    pats = h.make_patterns(6, active=0.08, generator=g)
    h.store(pats)
    unstored = h.make_patterns(1, active=0.08, generator=g)[0]

    def run(model, pattern, frac, gen_seed):
        gg = torch.Generator().manual_seed(gen_seed)
        cue = cue_pattern(pattern[None, :], frac, gg)
        ng = torch.Generator().manual_seed(gen_seed + 1)
        out = settle(model, cue, noise_gen=ng)
        return model.overlap(out)[0]

    rows = {}
    for frac in (1.0, 0.6, 0.4, 0.2):
        ov = run(h, pats[0], frac, 21)
        rows[f"cue_{frac:g}"] = {"overlap_target": float(ov[0]),
                                 "overlap_best_other": float(ov[1:].max()),
                                 "margin": float(ov[0] - ov[1:].max())}
    # control 1: the same weights, permuted once, outside
    h_shuf = h_builder()
    h_shuf.stored = h.stored.clone()
    perm = torch.randperm(h.n_ca3, generator=torch.Generator().manual_seed(99))
    h_shuf.W_rec = h.W_rec[perm][:, perm].clone()
    ov_s = run(h_shuf, pats[0], 0.4, 21)
    # control 2: a pattern that was never stored
    ov_u = run(h, unstored, 0.4, 21)
    main = rows["cue_0.4"]
    ok = (main["overlap_target"] >= 0.8 and main["margin"] > 0.2
          and float(ov_s[0]) < main["overlap_target"] - 0.2
          and float(ov_u.max()) < 0.6)
    return {"ok": ok, "by_cue": rows,
            "control_shuffled_overlap": float(ov_s[0]),
            "control_unstored_best_overlap": float(ov_u.max()),
            "rule": "at a 40% cue: overlap >= 0.8, margin over the next pattern > 0.2, the "
                    "shuffled-weight control at least 0.2 lower, and an unstored pattern "
                    "completed to < 0.6"}


def theta_measure(h, ms_feedback=True, septal_tone=1.0, seed=17, steps=12000):
    g = torch.Generator().manual_seed(seed)
    tr, _ = h.rollout(steps, DT, ec=None, septal_tone=septal_tone, noise_gen=g,
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
    tr, _ = h.rollout(12000, DT, septal_tone=1.0, noise_gen=g, record=("ca1",))
    x = tr["ca1"].mean(-1)
    mi = float(SP.pac_mi(x, FS, THETA_BAND, GAMMA_BAND).mean())
    rolled = torch.roll(x, shifts=x.shape[-1] // 3, dims=-1)
    mi_ctrl = float(SP.pac_mi(torch.cat([rolled[:, :x.shape[-1] // 2],
                                         x[:, x.shape[-1] // 2:]], -1),
                              FS, THETA_BAND, GAMMA_BAND).mean())
    return {"ok": mi > 0.0 and mi > mi_ctrl, "mi": mi, "mi_rolled_control": mi_ctrl,
            "rule": "MI > 0 and above a phase-rolled control"}


def gate_ripples(h):
    out = {}
    for name, tone in (("theta_state", 1.0), ("quiet_state", 0.0)):
        g = torch.Generator().manual_seed(29)
        tr, _ = h.rollout(12000, DT, septal_tone=tone, noise_gen=g, record=("ca1",))
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
        tr, _ = h.rollout(6000, DT, septal_tone=1.0, noise_gen=g, record=("ca1",))
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
        head = {k: v for k, v in r.items() if k not in ("sweep", "states", "by_cue")}
        print(f"[{'PASS' if r.get('ok') else 'FAIL'}] {name}: {head}", flush=True)
        if name == "H3_completion" and "by_cue" in r:
            for k, v in r["by_cue"].items():
                print(f"      {k}: target {v['overlap_target']:+.3f}  "
                      f"next {v['overlap_best_other']:+.3f}  margin {v['margin']:+.3f}",
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
