"""s1.regime: fit the nonlinear parameters against measured N3 spectra.

the gate the whole cognitive half of the curriculum waits on.  prior medians put
the cortical loop at a single ~4.8 Hz fixed point, and a system with one fixed
point has no basins, no transitions and nothing for a cognitive object to be.  the
multistable and oscillatory regions are roughly 0.05% of the swept parameter
space, so theta has to be FIT to land in one -- it will not arrive there by
default.

what makes this fittable rather than a search: the slow oscillation is a
measurable feature of real N3 sleep, and in this model its frequency is set by a
single declared parameter.  so the objective is the distance between the model's
own SO frequency and the one in the recordings, and the gate is whether the fitted
point also has more than one invariant set.

targets come from sleep-edfx scored N3, not from literature: the SO band is quoted
as 0.5-1.0 Hz almost everywhere, and the actual peak in these recordings is what
theta should match.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

def n3_slow_peak(edf_dir: str, n_files: int = 8) -> tuple[float, float, int]:
    """the slow-oscillation peak in scored N3, measured rather than assumed."""
    import mne
    mne.set_log_level("ERROR")
    psgs = sorted(glob.glob(os.path.join(edf_dir, "*-PSG.edf")))[:n_files]
    peaks = []
    for p in psgs:
        hyp = p.replace("-PSG.edf", "-Hypnogram.edf")
        cand = glob.glob(p.rsplit("-PSG", 1)[0][:-1] + "*-Hypnogram.edf")
        hyp = hyp if os.path.exists(hyp) else (cand[0] if cand else None)
        if not hyp: continue
        try:
            raw = mne.io.read_raw_edf(p, preload=False)
            ann = mne.read_annotations(hyp)
            raw.set_annotations(ann)
            ch = [c for c in raw.ch_names if "Fpz" in c or "EEG" in c][:1]
            if not ch: continue
            raw.pick(ch).load_data().filter(0.3, 4.0)
            sf = raw.info["sfreq"]
            n3 = [(o, d) for o, d, desc in
                  zip(ann.onset, ann.duration, ann.description)
                  if "3" in desc or "4" in desc]
            if not n3: continue
            segs = []
            for o, d in n3[:12]:
                a_, b_ = int(o * sf), int((o + min(d, 60)) * sf)
                if b_ <= raw.n_times: segs.append(raw.get_data(start=a_, stop=b_)[0])
            if not segs: continue
            x = np.concatenate(segs)
            f = np.fft.rfftfreq(len(x), 1 / sf)
            P = np.abs(np.fft.rfft(x)) ** 2
            m = (f >= 0.3) & (f <= 2.0)
            peaks.append(float(f[m][np.argmax(P[m])]))
        except Exception:
            continue
    if not peaks: return (float("nan"), float("nan"), 0)
    return float(np.median(peaks)), float(np.std(peaks)), len(peaks)


def model_so(tau_a: float, w_ee: float, a_gain: float, T=30.0, dt=2e-5) -> tuple[float, float]:
    """the model's own slow oscillation: frequency and swing."""
    p = dict(tau_m=0.015, e_rest=-65.0, v_half=-55.0, slope=4.0, r_max=100.0,
             e_rev=-70.0, g_leak=1.0, tau_ampa=0.005, tau_gaba=0.008, tau_i=0.010)
    s = np.array([-65.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    n = int(T / dt); out = np.empty(n)
    for i in range(n):
        v, r_e, a, g_e, r_i, g_i = s
        r_inf = p["r_max"] / (1.0 + np.exp(-(v - p["v_half"]) / p["slope"]))
        dv = (-(v - p["e_rest"]) + g_e - a) / p["tau_m"]
        gi = max(g_i, 0.0)
        dv += -(v - p["e_rev"]) * gi / (p["tau_m"] * p["g_leak"])   # conductance form
        s = s + dt * np.array([dv, (r_inf - r_e) / 5e-3, (a_gain * r_inf - a) / tau_a,
                               (w_ee * r_e - g_e) / p["tau_ampa"],
                               (r_inf - r_i) / p["tau_i"], (0.0 - g_i) / p["tau_gaba"]])
        out[i] = s[0]
    tail = out[n // 2:]
    x = tail - tail.mean()
    f = np.fft.rfftfreq(len(x), dt); P = np.abs(np.fft.rfft(x)) ** 2
    k = np.argmax(P[1:]) + 1
    return float(f[k]), float(tail.max() - tail.min())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edf", default="data/sources/sleep-edfx/raw/1.0.0/sleep-cassette")
    ap.add_argument("--out", default="out/regime_fit.json")
    a = ap.parse_args()

    tgt, sd, n = n3_slow_peak(a.edf)
    print(f"measured N3 slow peak: {tgt:.3f} Hz (sd {sd:.3f}, n={n} recordings)", flush=True)
    if not np.isfinite(tgt):
        print("no N3 measured; aborting rather than fitting to a literature number")
        return

    best = None
    print(f"\n  {'tau_a':>7} {'w_ee':>6} {'a_gain':>7} {'f_model':>9} {'swing':>8} {'|err|':>8}")
    for tau_a in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        for w_ee in (0.30, 0.385, 0.45):
            for a_gain in (0.15, 0.20, 0.25):
                f, sw = model_so(tau_a, w_ee, a_gain)
                if sw < 5.0: continue            # no oscillation, not a candidate
                err = abs(f - tgt)
                if best is None or err < best[0]:
                    best = (err, tau_a, w_ee, a_gain, f, sw)
                print(f"  {tau_a:7.2f} {w_ee:6.3f} {a_gain:7.2f} {f:8.3f}  {sw:7.1f} {err:8.3f}",
                      flush=True)
    if best is None:
        print("\nno oscillatory point found"); return
    err, tau_a, w_ee, a_gain, f, sw = best
    print(f"\nFITTED: tau_adaptation_s={tau_a}  w_ee={w_ee}  adaptation_gain={a_gain}")
    print(f"  model SO {f:.3f} Hz vs measured {tgt:.3f} Hz   (|err| {err:.3f} Hz, "
          f"{err/max(sd,1e-9):.1f} measured sd)")
    print(f"  swing {sw:.1f} mV")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"measured_hz": tgt, "measured_sd": sd, "n_recordings": n,
               "tau_adaptation_s": tau_a, "w_ee": w_ee, "adaptation_gain": a_gain,
               "model_hz": f, "swing_mv": sw, "abs_err_hz": err}, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
