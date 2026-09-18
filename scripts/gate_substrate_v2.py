"""the v2 substrate's gates, as pre-registered in docs/LOG.md (2026-09-18).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/gate_substrate_v2.py

G0 idempotence   G1 bounded   G2 >1 invariant set (+ monostable control, must be 1)
G3 metastability G4 heterogeneity expressed (+ permuted-h control)
G5 memory beyond the input window (+ monostable arm, reported not gated)

runs on the CPU on purpose: the GB10's GPU shares system memory and a training run holds
~65 GB of it (CLAUDE.md, Jobs).  every draw comes from an explicit generator, and the
summary is written before anything that could raise, with a json default that degrades a
stray array to its shape instead of losing the record (CLAUDE.md, Jobs).

thresholds are the ones in the log entry and do not move after a run.
"""
from __future__ import annotations

import argparse, json, math, os, time
import numpy as np
import torch

from ibm.substrate import Priors, build_sheet

OUT = "out/gate_substrate_v2.json"


def jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f"<ndarray {o.shape}>"
    if torch.is_tensor(o):
        return o.tolist() if o.numel() <= 64 else f"<tensor {tuple(o.shape)}>"
    return str(o)


def save(res, path=OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=2, default=jdefault)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


@torch.no_grad()
def run(field, T, dt, batch, start="rest", start_gen=None, noise_gen=None,
        drive_fn=None, rec_every=None, rec_from=0.0):
    """integrate `T` seconds; return (final state, recorded E (B, R, N) or None)."""
    st = field.init_state(batch, random=(start == "random"), generator=start_gen)
    W = field.edge_weights()
    steps = int(round(T / dt))
    rec = []
    zero = torch.zeros(batch, field.n)
    for t in range(steps):
        d = drive_fn(t * dt) if drive_fn is not None else zero
        z = torch.randn(batch, field.n, generator=noise_gen) if noise_gen is not None else None
        st = field.step(st, d, dt, W=W, noise=z)
        if rec_every and t * dt >= rec_from and t % rec_every == 0:
            rec.append(st["E"].clone())
        if t % 2000 == 0:
            guard("run")
    return st, (torch.stack(rec, 1) if rec else None)


def rss_gb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return 0.0


# NO GRADIENTS ANYWHERE IN THIS SCRIPT.  the first launch ran G1's loop -- 30,000 steps at
# batch 32 -- outside `no_grad`, with `embed` and `pair_slope` still requiring grad, so
# autograd kept every intermediate: it reached 52.8 GB in ninety seconds, next to a 65 GB
# training run on a machine whose GPU shares the same 121 GB (CLAUDE.md), and was killed
# with 8 GB to spare.  a gate never trains, so the whole of main is under no_grad, and
# MEM_CEILING_GB aborts the run outright rather than letting a mistake like that reach the
# machine.
MEM_CEILING_GB = 12.0


def guard(where: str):
    g = rss_gb()
    if g > MEM_CEILING_GB:
        raise MemoryError(f"{where}: RSS {g:.1f} GB > ceiling {MEM_CEILING_GB} GB -- aborting")


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    t_all = time.time()
    res = {"preregistered": "docs/LOG.md 2026-09-18", "n": a.n, "k": a.k, "dt": a.dt,
           "seed": a.seed, "priors": Priors().__dict__, "gates": {}}

    def field_with(scale_wEE=1.0):
        pr = Priors()
        if scale_wEE != 1.0:
            pr.w_EE = tuple(v * scale_wEE for v in pr.w_EE)
        return build_sheet(a.n, a.k, seed=a.seed, priors=pr, learn_hetero=False)

    f = field_with()
    mono = field_with(0.3)
    res["unknown_regions"] = f.unknown_regions
    res["n_regions"] = len(f.region_list)
    res["column_bistable_fraction"] = float(f.column_bistable().float().mean())
    res["column_bistable_fraction_monostable_control"] = float(mono.column_bistable().float().mean())
    save(res, a.out)

    # ---------------------------------------------------------------- G0
    g = torch.Generator().manual_seed(a.seed + 1)
    s0 = f.init_state(4, random=True, generator=g)
    d0 = torch.rand(4, f.n, generator=g) * 0.5
    z0 = torch.randn(4, f.n, generator=g)
    W = f.edge_weights()
    o1 = f.step(s0, d0, a.dt, W=W, noise=z0)
    o2 = f.step(s0, d0, a.dt, W=W, noise=z0)
    step_eq = all(torch.equal(o1[k], o2[k]) for k in ("E", "I", "a", "x", "eta"))
    r1, _ = run(f, 1.0, a.dt, 4, "random", torch.Generator().manual_seed(9))
    r2, _ = run(f, 1.0, a.dt, 4, "random", torch.Generator().manual_seed(9))
    roll_eq = all(torch.equal(r1[k], r2[k]) for k in ("E", "I", "a", "x"))
    res["gates"]["G0_idempotence"] = {"step_equal": step_eq, "rollout_equal": roll_eq,
                                      "PASS": bool(step_eq and roll_eq)}
    save(res, a.out)
    if not (step_eq and roll_eq):
        res["VOID"] = "G0 failed: nothing else is read"
        save(res, a.out)
        print("G0 FAILED -- void")
        return 1
    print("G0", res["gates"]["G0_idempotence"])

    # ---------------------------------------------------------------- G1
    t0 = time.time()
    gp = torch.Generator().manual_seed(a.seed + 2)
    pulses = [(float(torch.rand(1, generator=gp)) * 28.0, torch.rand(32, f.n, generator=gp) < 0.1)
              for _ in range(40)]
    def pulse_drive(t):
        d = torch.zeros(32, f.n)
        for t0_, m in pulses:
            if t0_ <= t < t0_ + 0.2:
                d = d + m.float() * 1.5
        return d
    lo = {"E": 1e9, "I": 1e9, "x": 1e9, "a": 1e9}
    hi = {"E": -1e9, "I": -1e9, "x": -1e9, "a": -1e9}
    st = f.init_state(32, random=True, generator=torch.Generator().manual_seed(a.seed + 3))
    Wf = f.edge_weights()
    ng = torch.Generator().manual_seed(a.seed + 4)
    finite = True
    for t in range(int(30.0 / a.dt)):
        st = f.step(st, pulse_drive(t * a.dt), a.dt, W=Wf, noise=torch.randn(32, f.n, generator=ng))
        if t % 2000 == 0:
            guard("G1")
        if t % 50 == 0:
            for k in lo:
                lo[k] = min(lo[k], float(st[k].min())); hi[k] = max(hi[k], float(st[k].max()))
            finite = finite and all(bool(torch.isfinite(st[k]).all()) for k in lo)
    ga_max = float(f.site("g_a").max())
    ok = (finite and lo["E"] >= 0 and hi["E"] <= 1 and lo["I"] >= 0 and hi["I"] <= 1
          and lo["x"] >= 0 and hi["x"] <= 1 and lo["a"] >= 0 and hi["a"] <= ga_max + 1e-6)
    res["gates"]["G1_bounded"] = {"min": lo, "max": hi, "g_a_max": ga_max, "finite": finite,
                                  "PASS": bool(ok), "seconds": round(time.time() - t0, 1)}
    save(res, a.out)
    print("G1", res["gates"]["G1_bounded"])

    # ---------------------------------------------------------------- G2
    def count_sets(field, label):
        t0 = time.time()
        stt, rec = run(field, 23.0, a.dt, 32, "random", torch.Generator().manual_seed(a.seed + 5),
                       rec_every=10, rec_from=20.0)
        pat = rec.mean(1)                                    # (32, N) time-averaged pattern
        swing = (rec.max(1).values - rec.min(1).values).max(1).values   # per run
        reps = []
        for i in range(pat.shape[0]):
            p = pat[i]
            same = False
            for r in reps:
                q = pat[r]
                if p.std() < 1e-6 or q.std() < 1e-6:
                    same = bool(p.std() < 1e-6 and q.std() < 1e-6 and (p.mean() - q.mean()).abs() < 1e-3)
                else:
                    c = float(torch.corrcoef(torch.stack([p, q]))[0, 1])
                    same = c >= 0.9
                if same:
                    break
            if not same:
                reps.append(i)
        return {"n_sets": len(reps), "mean_E_per_set": [float(pat[r].mean()) for r in reps],
                "frac_up_per_set": [float((pat[r] > 0.5).float().mean()) for r in reps],
                "max_swing_last_3s": float(swing.max()), "stationary_runs": int((swing < 1e-3).sum()),
                "seconds": round(time.time() - t0, 1), "label": label}
    g2 = count_sets(f, "declared priors")
    g2c = count_sets(mono, "w_EE x 0.3 control")
    control_ok = g2c["n_sets"] == 1
    res["gates"]["G2_invariant_sets"] = {"field": g2, "control": g2c, "control_ok": control_ok,
                                         "PASS": bool(control_ok and g2["n_sets"] >= 2),
                                         "VOID": (not control_ok)}
    save(res, a.out)
    print("G2", {k: v for k, v in res["gates"]["G2_invariant_sets"].items() if k not in ("field", "control")},
          "field sets", g2["n_sets"], "control sets", g2c["n_sets"])

    # ---------------------------------------------------------------- G3 + G4 (one noisy run)
    t0 = time.time()
    _, recE = run(f, 60.0, a.dt, 1, "rest", noise_gen=torch.Generator().manual_seed(a.seed + 6),
                  rec_every=5, rec_from=2.0)
    E = recE[0].numpy()                                      # (R, N), 5 ms samples
    rid = f.region_id.numpy()
    R = len(f.region_list)
    reg = np.stack([E[:, rid == r].mean(1) for r in range(R)], 1)     # (R_t, 68)
    code = (reg > 0.5)
    keys = [c.tobytes() for c in code]
    trans = [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]]
    dwell_s = np.diff([0] + trans + [len(keys)]) * 0.005
    from collections import Counter
    visits = Counter()
    prev = None
    for kk in keys:
        if kk != prev:
            visits[kk] += 1
        prev = kk
    revisited = sum(1 for v in visits.values() if v >= 2)
    g3 = {"transitions": len(trans), "median_dwell_s": float(np.median(dwell_s)),
          "mean_dwell_s": float(dwell_s.mean()), "distinct_macrostates": len(visits),
          "revisited_ge2": revisited,
          "mean_regions_up": float(code.sum(1).mean()),
          "frac_time_any_region_up": float(code.any(1).mean()),
          "seconds": round(time.time() - t0, 1)}
    g3["PASS"] = bool(g3["transitions"] >= 10 and g3["median_dwell_s"] > 0.05 and revisited >= 3)
    res["gates"]["G3_metastability"] = g3
    save(res, a.out)
    print("G3", g3)

    # G4: per-site autocorrelation time (1/e crossing), against h and against permuted h
    X = E - E.mean(0, keepdims=True)
    var = (X ** 2).mean(0)
    live = var > 1e-8
    L = min(len(X) // 2, 2000)                               # up to 10 s of lag
    Fx = np.fft.rfft(X[:, live], n=2 * len(X), axis=0)
    ac = np.fft.irfft(Fx * np.conj(Fx), axis=0)[:L] / (var[live] * len(X))
    below = ac < math.exp(-1)
    tau = np.where(below.any(0), below.argmax(0), L) * 0.005
    h = f.h.numpy()[live]
    perm = np.random.default_rng(a.seed + 11).permutation(len(h))
    rho = spearman(tau, h)
    rho_ctl = spearman(tau, h[perm])
    g4 = {"sites_with_variance": int(live.sum()), "rho_tau_vs_h": rho, "rho_control_permuted_h": rho_ctl,
          "tau_s_quartiles": np.percentile(tau, [25, 50, 75]).tolist(),
          "tau_s_by_h_band": {f"{lo_}-{hi_}": float(np.median(tau[(h >= lo_) & (h < hi_)])) if ((h >= lo_) & (h < hi_)).any() else None
                              for lo_, hi_ in [(0, .3), (.3, .6), (.6, .8), (.8, 1.01)]}}
    g4["PASS"] = bool(rho > 0.3 and abs(rho_ctl) < 0.1)
    res["gates"]["G4_heterogeneity"] = g4
    save(res, a.out)
    print("G4", g4)

    # ---------------------------------------------------------------- G5
    def memory(field, label):
        t0 = time.time()
        gg = np.random.default_rng(a.seed + 12)
        R_ = len(field.region_list)
        sets = [gg.choice(R_, 4, replace=False) for _ in range(8)]
        mask = torch.stack([torch.from_numpy(np.isin(field.region_id.numpy(), s)).float() for s in sets])  # (8, N)
        trials = 20
        lab = torch.arange(8).repeat_interleave(trials)                  # (160,)
        drive_on = mask[lab] * 1.5
        zero = torch.zeros_like(drive_on)
        def dfn(t):
            return drive_on if t < 0.2 else zero
        stt, _ = run(field, 1.2, a.dt, len(lab), "rest", noise_gen=torch.Generator().manual_seed(a.seed + 13),
                     drive_fn=dfn)
        Xr = torch.cat([stt["E"], stt["a"], stt["x"]], 1)             # the whole readable state
        Xe = stt["E"]
        def loo(X):
            X = X.double()
            correct = 0
            for i in range(len(lab)):
                keep = torch.ones(len(lab), dtype=torch.bool); keep[i] = False
                cents = torch.stack([X[keep & (lab == c)].mean(0) for c in range(8)])
                pred = int((cents - X[i]).norm(dim=1).argmin())
                correct += int(pred == int(lab[i]))
            return correct
        kE, kS = loo(Xe), loo(Xr)
        n = len(lab)
        return {"label": label, "acc_E": kE / n, "ci_E": wilson(kE, n),
                "acc_full_state": kS / n, "ci_full_state": wilson(kS, n),
                "chance": 1 / 8, "readout_s_after_offset": 1.0, "seconds": round(time.time() - t0, 1)}
    g5 = memory(f, "declared priors")
    g5m = memory(mono, "w_EE x 0.3 (monostable), reported not gated")
    res["gates"]["G5_memory"] = {"field": g5, "monostable": g5m,
                                 "bypass": "chance by construction: the input is zero at readout",
                                 "PASS": bool(g5["ci_E"][0] > 1 / 8)}
    res["seconds_total"] = round(time.time() - t_all, 1)
    save(res, a.out)
    print("G5", g5, "\n   monostable", g5m)
    print("\nPASS:", {k: v.get("PASS") for k, v in res["gates"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
