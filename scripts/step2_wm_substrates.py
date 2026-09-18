"""research step 2: which substrate carries the working-memory array through the delay?

pre-registered in docs/LOG.md (2026-09-18, "PRE-REGISTRATION: research step 2" and its
AMENDMENT).  reads out/wm_features_ds008037.npz (scripts/wm_features_ds008037.py).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/step2_wm_substrates.py

arms: v2 (declared, noise 0.04), v2 at noise 0.12 (secondary), v2-mono (w_EE x 0.3), ESN
(1,024 leaky tanh, leak and spectral radius chosen on a validation split of TRAINING
subjects), v1 (CorticalDynamics run continuously), bypass (drive at readout = 0), ceiling
(the stimulus descriptors S).  readout: ridge, alpha by 5-fold CV grouped by subject within
training.  score: held-out R^2 per test subject over the 124 delay-EEG features; bootstrap
over test subjects; paired differences.

cpu only, no_grad, memory-guarded.  the summary is written as each arm finishes.
"""
from __future__ import annotations

import json, math, os, sys, time
import numpy as np
import torch

IN_NPZ = os.environ.get("STEP2_IN", "out/wm_features_ds008037.npz")
OUT = os.environ.get("STEP2_OUT", "out/step2_wm_substrates.json")
DT = 2e-3
T_ARRAY, T_RET, WIN = 0.5, 0.5, (0.3, 1.0)      # array 0-0.5 s; retention from 0.5 s
N, K, SEED = 1024, 32, 0
DRIVE_V2, DRIVE_V1_SCALE = 0.6, 20.0
TRIALS_PER_SUB = 120
KAPPA_DRIVE, KAPPA_HIST, NBINS = 4.0, 4.0, 12


def jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f"<ndarray {o.shape}>"
    if torch.is_tensor(o):
        return o.tolist() if o.numel() <= 64 else f"<tensor {tuple(o.shape)}>"
    return str(o)


def rss_gb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return 0.0


# ------------------------------------------------------------------ data
def load():
    z = np.load(IN_NPZ, allow_pickle=False)
    sub, ss = z["sub"], z["set_size"]
    # the amendment's rule: the first TRIALS_PER_SUB clean trials per subject, by position
    keep = np.zeros(len(sub), bool)
    for s in np.unique(sub):
        idx = np.flatnonzero(sub == s)[:TRIALS_PER_SUB]
        keep[idx] = True
    colours = [np.r_[t, nt[np.isfinite(nt)]] for t, nt in zip(z["target_deg"], z["nontarget_deg"])]
    return {"F": z["F"], "sub": sub, "ss": ss, "target": z["target_deg"], "reported": z["reported_deg"],
            "colours": colours, "keep": keep}


def descriptors(d):
    """S: set-size one-hot (3), 12-bin von Mises colour histogram (12), target cos/sin (2)."""
    ss = d["ss"]
    oh = np.stack([ss == 2, ss == 4, ss == 6], 1).astype(np.float32)
    ctr = np.arange(NBINS) * 2 * np.pi / NBINS
    H = np.zeros((len(ss), NBINS), np.float32)
    for i, cs in enumerate(d["colours"]):
        th = np.deg2rad(cs)[:, None]
        H[i] = np.exp(KAPPA_HIST * (np.cos(th - ctr[None]) - 1)).sum(0)
    tg = np.deg2rad(d["target"])
    return np.concatenate([oh, H, np.stack([np.cos(tg), np.sin(tg)], 1)], 1)


def split(subs):
    u = np.array(sorted(set(subs)))
    perm = np.random.default_rng(0).permutation(len(u))
    ntr = int(round(0.7 * len(u)))
    return set(u[perm[:ntr]]), set(u[perm[ntr:]])


# ------------------------------------------------------------------ the drive
def colour_drive(field_or_sites, colours, n_sites, visual_idx, seed=SEED):
    """(B, N) array-period drive: visual sites tuned to a preferred colour (drawn ONCE)."""
    phi = np.random.default_rng(seed + 31).uniform(0, 2 * np.pi, len(visual_idx))
    D = np.zeros((len(colours), n_sites), np.float32)
    for i, cs in enumerate(colours):
        th = np.deg2rad(cs)[:, None]
        D[i, visual_idx] = np.exp(KAPPA_DRIVE * (np.cos(th - phi[None]) - 1)).sum(0)
    return D


# ------------------------------------------------------------------ arms
@torch.no_grad()
def run_v2(D, priors, seed=SEED, batch=512):
    from ibm.substrate import build_sheet
    f = build_sheet(N, K, seed=seed, priors=priors, learn_hetero=False, long_topology="tract")
    W = f.edge_weights()
    g = torch.Generator().manual_seed(seed + 77)
    n_arr, n_all = int(T_ARRAY / DT), int((T_RET + WIN[1]) / DT)
    w0 = int((T_RET + WIN[0]) / DT)
    out = np.zeros((len(D), N), np.float32)
    for b in range(0, len(D), batch):
        d = torch.from_numpy(D[b:b + batch]) * DRIVE_V2
        st = f.init_state(len(d))
        acc = torch.zeros(len(d), N); cnt = 0
        zero = torch.zeros_like(d)
        for t in range(n_all):
            st = f.step(st, d if t < n_arr else zero, DT, W=W,
                        noise=torch.randn(len(d), N, generator=g))
            if t >= w0:
                acc += st["E"]; cnt += 1
        out[b:b + batch] = (acc / cnt).numpy()
        if rss_gb() > 14:
            raise MemoryError(f"RSS {rss_gb():.1f} GB")
    return out


def visual_sites_v2(seed=SEED):
    from ibm.substrate import build_sheet
    f = build_sheet(N, K, seed=seed, learn_hetero=False, long_topology="tract")
    vis = ("pericalcarine", "cuneus", "lingual", "lateraloccipital")
    return np.flatnonzero([r.split(".", 1)[-1] in vis for r in f.region_names])


@torch.no_grad()
def run_esn(D, rho, leak, seed=SEED, batch=1024):
    g = np.random.default_rng(seed + 55)
    dens = 0.01
    Wr = (g.random((N, N)) < dens) * g.standard_normal((N, N))
    ev = np.max(np.abs(np.linalg.eigvals(Wr)))
    Wr = torch.from_numpy((Wr * (rho / ev)).astype(np.float32))
    b0 = torch.from_numpy((g.standard_normal(N) * 0.1).astype(np.float32))
    n_arr, n_all = int(T_ARRAY / DT), int((T_RET + WIN[1]) / DT)
    w0 = int((T_RET + WIN[0]) / DT)
    out = np.zeros((len(D), N), np.float32)
    for b in range(0, len(D), batch):
        u = torch.from_numpy(D[b:b + batch])
        x = torch.zeros(len(u), N)
        acc = torch.zeros(len(u), N); cnt = 0
        for t in range(n_all):
            inp = u if t < n_arr else torch.zeros_like(u)
            x = (1 - leak) * x + leak * torch.tanh(x @ Wr.T + inp + b0)
            if t >= w0:
                acc += x; cnt += 1
        out[b:b + batch] = (acc / cnt).numpy()
    return out


@torch.no_grad()
def run_v1(D, seed=SEED, batch=512):
    sys.path.insert(0, "scripts")
    import pretrain_video_loop as P
    dyn = P.CorticalDynamics(N, 64, K, "cpu", geometry="surface", graph_seed=seed)
    W = dyn.edge_weights()
    # v1's own sites; the drive goes to ITS visual sites, tuned the same way
    import ibm.cortical_sheet as CS
    reg = CS.regions_at(dyn.pos.cpu().numpy())
    vis = np.flatnonzero([CS.REGIONS[i].split(".", 1)[-1] in ("pericalcarine", "cuneus", "lingual",
                                                               "lateraloccipital") for i in reg])
    return vis, dyn, W


@torch.no_grad()
def run_v1_rollout(D1, dyn, W, batch=512):
    n_arr, n_all = int(T_ARRAY / DT), int((T_RET + WIN[1]) / DT)
    w0 = int((T_RET + WIN[0]) / DT)
    out = np.zeros((len(D1), N), np.float32)
    for b in range(0, len(D1), batch):
        d = torch.from_numpy(D1[b:b + batch]) * DRIVE_V1_SCALE
        s = dyn.init_state(len(d), "cpu")
        acc = torch.zeros(len(d), N); cnt = 0
        zero = torch.zeros_like(d)
        for t in range(n_all):
            s = dyn.step(s, d if t < n_arr else zero, DT, W)
            if t >= w0:
                acc += s[1]; cnt += 1
        r = (acc / cnt)
        out[b:b + batch] = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).numpy()
    return out


# ------------------------------------------------------------------ readout & score
def ridge_fit(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xs.T @ (Y - Y.mean(0)))
    return mu, sd, W, Y.mean(0)


def ridge_pred(m, X):
    mu, sd, W, ym = m
    return ((X - mu) / sd) @ W + ym


def choose_alpha(X, Y, groups, alphas=(1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7)):
    # the grid reaches 1e7: 1,024 collinear site activities need far more shrinkage than 17
    # descriptors, and a grid that stops at 1e5 would hand the high-dimensional arms their
    # optimum only when it happens to be small.  the smoke test (synthetic data) showed
    # every arm pinned at the grid's LOW end, which on real data would be overfitting.
    ug = np.array(sorted(set(groups)))
    folds = [f for f in np.array_split(np.random.default_rng(1).permutation(ug), 5) if len(f)]
    best = None
    for a in alphas:
        err = 0.0
        for f in folds:
            te = np.isin(groups, f)
            if te.all() or not te.any():
                continue
            m = ridge_fit(X[~te], Y[~te], a)
            err += float(((ridge_pred(m, X[te]) - Y[te]) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, a)
    return best[1]


def per_subject_r2(Yp, Y, subs):
    out = {}
    for s in sorted(set(subs)):
        m = subs == s
        out[s] = float(1 - ((Y[m] - Yp[m]) ** 2).sum() / ((Y[m]) ** 2).sum())   # floor = 0
    return out


def boot(vals, n=4000, seed=2):
    v = np.array(vals); g = np.random.default_rng(seed)
    bs = v[g.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(bs.std())


def score(name, X, d, tr, te, res):
    Y, subs = d["F"], d["sub"]
    a = choose_alpha(X[tr], Y[tr], subs[tr])
    m = ridge_fit(X[tr], Y[tr], a)
    r2 = per_subject_r2(ridge_pred(m, X[te]), Y[te], subs[te])
    mean, se = boot(list(r2.values()))
    res["arms"][name] = {"alpha": a, "r2_mean": mean, "r2_se": se, "per_subject": r2}
    print(f"  {name:10s} R^2 {mean:+.5f}  (se {se:.5f}, alpha {a:g})", flush=True)
    return r2


def paired(r2a, r2b):
    ks = sorted(set(r2a) & set(r2b))
    return boot([r2a[k] - r2b[k] for k in ks])


# ------------------------------------------------------------------ main
@torch.no_grad()
def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    from ibm.substrate import Priors
    d = load()
    trs, tes = split(d["sub"])
    res = {"preregistered": "docs/LOG.md 2026-09-18 research step 2 + AMENDMENT",
           "n_trials_all": int(len(d["sub"])), "train_subjects": sorted(trs), "test_subjects": sorted(tes),
           "arms": {}, "gates": {}}
    save = lambda: json.dump(res, open(OUT, "w"), indent=1, default=jdefault)
    S = descriptors(d)
    tr_all = np.isin(d["sub"], list(trs)); te_all = np.isin(d["sub"], list(tes))
    # ceiling and bypass on ALL retained trials (cheap)
    r2_ceiling = score("ceiling", S, d, tr_all, te_all, res)
    r2_bypass = score("bypass", np.zeros((len(S), 1), np.float32), d, tr_all, te_all, res)
    save()
    # dynamical arms on the capped set
    cap = d["keep"]
    dc = {k: (v[cap] if isinstance(v, np.ndarray) else [c for c, kk in zip(v, cap) if kk]) for k, v in d.items()}
    tr = np.isin(dc["sub"], list(trs)); te = np.isin(dc["sub"], list(tes))
    res["n_trials_capped"] = int(cap.sum())
    vis = visual_sites_v2()
    D = colour_drive(None, dc["colours"], N, vis)
    arms = {}
    t0 = time.time()
    arms["v2"] = run_v2(D, Priors()); print(f"v2 ran ({time.time()-t0:.0f}s)", flush=True)
    r2 = {"v2": score("v2", arms["v2"], dc, tr, te, res)}; save()
    pm = Priors(); pm.w_EE = tuple(v * 0.3 for v in pm.w_EE)
    arms["v2_mono"] = run_v2(D, pm); r2["v2_mono"] = score("v2_mono", arms["v2_mono"], dc, tr, te, res); save()
    p12 = Priors(); p12.sigma = 0.12
    arms["v2_s012"] = run_v2(D, p12); r2["v2_s012"] = score("v2_s012", arms["v2_s012"], dc, tr, te, res); save()
    # ESN: leak and spectral radius chosen on a validation split OF TRAINING SUBJECTS
    trsub = np.array(sorted(trs)); vperm = np.random.default_rng(3).permutation(len(trsub))
    vsubs = set(trsub[vperm[:len(trsub) // 4]])
    va = tr & np.isin(dc["sub"], list(vsubs)); fit = tr & ~va
    best = None
    for rho in (0.8, 0.9, 0.95, 0.99):
        for leak in (0.05, 0.2):
            Xe = run_esn(D, rho, leak)
            a = choose_alpha(Xe[fit], dc["F"][fit], dc["sub"][fit])
            m = ridge_fit(Xe[fit], dc["F"][fit], a)
            v = float(np.mean(list(per_subject_r2(ridge_pred(m, Xe[va]), dc["F"][va], dc["sub"][va]).values())))
            print(f"  esn tune rho {rho} leak {leak}: val R^2 {v:+.5f}", flush=True)
            if best is None or v > best[0]:
                best = (v, rho, leak, Xe)
    res["esn_tuning"] = {"rho": best[1], "leak": best[2], "val_r2": best[0]}
    r2["esn"] = score("esn", best[3], dc, tr, te, res); save()
    vis1, dyn, W1 = run_v1(None)
    D1 = colour_drive(None, dc["colours"], N, vis1)
    arms["v1"] = run_v1_rollout(D1, dyn, W1); r2["v1"] = score("v1", arms["v1"], dc, tr, te, res); save()
    # capped-set ceiling and bypass, for PAIRED comparisons on the same trials
    r2["ceiling_c"] = score("ceiling_capped", descriptors(dc), dc, tr, te, res)
    r2["bypass_c"] = score("bypass_capped", np.zeros((len(dc["sub"]), 1), np.float32), dc, tr, te, res)

    # ---- gates, in the pre-registered order
    cm, cs = res["arms"]["ceiling_capped"]["r2_mean"], res["arms"]["ceiling_capped"]["r2_se"]
    res["gates"]["VOID"] = {"ceiling_r2": cm, "se": cs, "PASS": bool(cm > 2 * cs)}
    for nm, (a, b) in {"V1_v2_minus_bypass": ("v2", "bypass_c"), "V2_v2_minus_esn": ("v2", "esn"),
                       "V3_v2_minus_mono": ("v2", "v2_mono"), "v1_minus_bypass": ("v1", "bypass_c"),
                       "ceiling_minus_v2": ("ceiling_c", "v2")}.items():
        m_, se_ = paired(r2[a], r2[b])
        res["gates"][nm] = {"diff": m_, "se": se_, "sqrt2_se_bar": math.sqrt(2) * se_}
    save()
    print(json.dumps(res["gates"], indent=1, default=jdefault), flush=True)

    # ---- behavioural signature (reported, not gated): colour-histogram fidelity by set size
    H = descriptors(dc)[:, 3:15]
    beh = {}
    for nm in ("v2", "v2_mono", "esn_best", "v1"):
        X = best[3] if nm == "esn_best" else arms[nm]
        m = ridge_fit(X[tr], H[tr], choose_alpha(X[tr], H[tr], dc["sub"][tr]))
        Hp = ridge_pred(m, X[te])
        fid = {}
        for s_ in (2, 4, 6):
            mm = dc["ss"][te] == s_
            fid[s_] = float(np.mean([np.corrcoef(Hp[mm][i], H[te][mm][i])[0, 1] for i in range(int(mm.sum()))]))
        beh[nm] = fid
    err = {}
    for s_ in (2, 4, 6):
        mm = dc["ss"][te] == s_
        dd = np.deg2rad(dc["reported"][te][mm] - dc["target"][te][mm])
        err[s_] = float(np.degrees(np.mean(np.abs(np.angle(np.exp(1j * dd))))))
    res["behavioural_signature"] = {"histogram_fidelity_by_set_size": beh,
                                    "human_mean_abs_error_deg_by_set_size": err}
    save()
    print(json.dumps(res["behavioural_signature"], indent=1), flush=True)


if __name__ == "__main__":
    main()
