#!/usr/bin/env python3
"""does visceral afference reaching cortex buy anything?  the controls first.

The materialization is `InteroceptiveLoop`: fifteen afferent channels from IHM-1's
native systemic state, grouped into seven (trunk, fibre class) conduction groups
whose delays span 9.2 ms to 507.8 ms over IHM's measured routes, entering the
cortical sheet and read out as three physiological quantities at a horizon.

**Nothing here is reported without its control.**  This repo's ledger is twenty
entries of a quantity computed correctly and compared against the wrong thing, so
the arms are:

  baselines, trivial      predict zero; predict the training mean; PERSISTENCE
  baselines, fair         ridge and an MLP on the same afferent vector, with no
                          dynamics at all -- the two that answer "does the
                          cortex add anything", which no trivial baseline can
  arms, trained           the learned kernel; a PERMUTED kernel trained the same
                          way; the slow unmyelinated arm removed and retrained
  arms, post hoc          the kernel severed, and each conduction group withheld,
                          on the already-trained weights

**Persistence is given more than the model is, and that is deliberate.**  It
predicts the target's value at the input instant.  For `discomfort` and
`sensation` that value is a function of the afferent vector the model does see,
so the comparison is fair.  For `endurance_h` it is not: endurance is computed
from liver and muscle glycogen against metabolic rate, none of which is an
afferent channel, so persistence is handed a quantity the model must infer.  A
model that beats persistence on endurance has extracted substrate state from gut
and metabolic afference; a model that loses to it has not, and the asymmetry is
why that has to be said rather than scored.

**The sanity gates run first and the script exits if they fail.**  A metric is
checked against cases whose answers are known: the mean predictor must score
exactly 0.000000 against the mean baseline, the persistence predictor exactly
0.000000 against persistence, and the zero predictor exactly 0.000000 against
zero.  Ledger entry 4 is a metric that reported its own ceiling for three hours;
the cost of these three lines is nothing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(HERE, "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)


# ---------------------------------------------------------------------------
# the corpus, and the pairs
# ---------------------------------------------------------------------------

class InteroCorpus:
    """the recorded visceral trajectories, as (afference at t, scalars at t+h).

    The horizon is in SECONDS, not samples.  Two of the seven sequences were
    recorded at 30 s cadence and five at 5 s, so a horizon in samples would mean
    "60 s ahead" for one group and "360 s ahead" for the other while printing one
    number -- which is the shape of ledger entry 11, where a builder assumed a
    constant that was not.
    """

    def __init__(self, root: str, horizon_s: float):
        self.root, self.horizon_s = root, horizon_s
        m = json.load(open(os.path.join(root, "meta.json")))
        self.meta = m
        X = np.load(os.path.join(root, "afferent.npy"))
        Y = np.load(os.path.join(root, "scalars.npy"))
        S = np.load(os.path.join(root, "sensation.npy"))
        seq = np.load(os.path.join(root, "sequence.npy"))
        tr = np.load(os.path.join(root, "is_train.npy"))
        te = np.load(os.path.join(root, "is_test.npy"))

        # the join is checked, not trusted: the channel table lives in the other
        # repository and a renamed channel would wire a port to a rate nobody
        # sends while the encoder still ran.
        import ibm.interoception as IO
        IO.check(m)
        self.channels = m["channels"]

        T = np.concatenate([Y, S], 1).astype(np.float32)
        self.target_names = list(m["scalars"]) + \
            [f"sensation_{i}" for i in range(S.shape[1])]

        i_in, i_out, split = [], [], []
        for s in m["sequences"]:
            k = int(round(horizon_s / s["dt_s"]))
            if k < 1:
                raise ValueError(f"horizon {horizon_s}s is under one sample of "
                                 f"{s['run']}/{s['protocol']} (dt {s['dt_s']}s)")
            idx = np.nonzero(seq == s["index"])[0]
            a, b = idx[:-k], idx[k:]
            for u, v in zip(a, b):
                if tr[u] and tr[v]:
                    i_in.append(u); i_out.append(v); split.append(0)
                elif te[u] and te[v]:
                    i_in.append(u); i_out.append(v); split.append(1)
        self.i_in = np.asarray(i_in); self.i_out = np.asarray(i_out)
        split = np.asarray(split)
        self.train = np.nonzero(split == 0)[0]
        self.test = np.nonzero(split == 1)[0]

        # standardisation from the TRAIN pairs only.
        xin = X[self.i_in]
        self.x_mu = xin[self.train].mean(0)
        self.x_sd = np.where(xin[self.train].std(0) > 0,
                             xin[self.train].std(0), 1.0)
        yout = T[self.i_out]
        self.y_mu = yout[self.train].mean(0)
        self.y_sd = np.where(yout[self.train].std(0) > 0,
                             yout[self.train].std(0), 1.0)

        self.X = ((xin - self.x_mu) / self.x_sd).astype(np.float32)
        self.Y_raw = yout.astype(np.float32)          # target at t+h, raw units
        self.P_raw = T[self.i_in].astype(np.float32)  # target at t, raw -- persistence
        self.Y = ((self.Y_raw - self.y_mu) / self.y_sd).astype(np.float32)

    @property
    def n_out(self) -> int:
        return self.Y.shape[1]

    def describe(self) -> str:
        return (f"{len(self.i_in)} pairs at horizon {self.horizon_s:.0f}s "
                f"({len(self.train)} train / {len(self.test)} test) over "
                f"{len(self.meta['sequences'])} sequences; "
                f"{len(self.channels)} channels -> {self.n_out} targets "
                f"({', '.join(self.target_names)})")


# ---------------------------------------------------------------------------
# baselines and skill
# ---------------------------------------------------------------------------

def per_target_mse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return ((pred - truth) ** 2).mean(0)


def baselines(c: InteroCorpus, idx: np.ndarray) -> dict:
    """the three trivial predictors, in RAW units, per target.

    Raw units and not standardised, because standardising by the training mean
    makes "predict zero" and "predict the mean" the same predictor by
    construction and quietly deletes a baseline.  That baseline is the one that
    caught the worst error in this repo's history.
    """
    truth = c.Y_raw[idx]
    return {
        "zero": per_target_mse(np.zeros_like(truth), truth),
        "mean": per_target_mse(np.broadcast_to(c.y_mu.astype(np.float32),
                                               truth.shape), truth),
        "persistence": per_target_mse(c.P_raw[idx], truth),
    }


def skill(mse: np.ndarray, base: dict) -> dict:
    return {f"skill_vs_{k}": np.where(v > 0, 1.0 - mse / v, np.nan).tolist()
            for k, v in base.items()}


def sanity_gates(c: InteroCorpus, idx: np.ndarray) -> dict:
    """check the metric against three cases whose answers are known.

    A predictor that IS a baseline must score exactly zero skill against that
    baseline.  If it does not, the metric is wrong and every number after it is
    decoration.
    """
    truth = c.Y_raw[idx]
    base = baselines(c, idx)
    got = {}
    for name, pred in (("zero", np.zeros_like(truth)),
                       ("mean", np.broadcast_to(c.y_mu.astype(np.float32),
                                                truth.shape)),
                       ("persistence", c.P_raw[idx])):
        s = np.asarray(skill(per_target_mse(pred, truth), base)[f"skill_vs_{name}"])
        worst = float(np.nanmax(np.abs(s)))
        got[name] = worst
        if worst > 1e-9:
            raise SystemExit(
                f"SANITY GATE FAILED: the {name} predictor scores skill "
                f"{worst:.3e} against the {name} baseline, and must score 0. "
                f"the metric is wrong; nothing below it means anything.")
    return got


def report_row(name: str, mse: np.ndarray, base: dict, targets: list[str],
               agg: float) -> str:
    s = skill(mse, base)
    parts = []
    for j, t in enumerate(targets):
        parts.append(f"{t}={s['skill_vs_mean'][j]:+.3f}/"
                     f"{s['skill_vs_persistence'][j]:+.3f}")
    return f"  {name:22s} agg {agg:+.4f}   " + "  ".join(parts)


# ---------------------------------------------------------------------------
# the dynamics-free controls
# ---------------------------------------------------------------------------

def ridge_control(c: InteroCorpus, lam: float = 1.0) -> np.ndarray:
    """closed-form linear map from the afferent vector to the target.

    The floor a cortex has to clear.  If the loop does not beat a ridge on the
    same fifteen numbers, the dynamics contributed nothing that a matrix could
    not, whatever the loss curve did.
    """
    Xtr = np.c_[c.X[c.train], np.ones(len(c.train), np.float32)]
    Ytr = c.Y[c.train]
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    Xte = np.c_[c.X[c.test], np.ones(len(c.test), np.float32)]
    return (Xte @ W) * c.y_sd + c.y_mu


def mlp_control(c: InteroCorpus, dev, steps: int, hidden: int, lr: float,
                seed: int) -> np.ndarray:
    """the same encoder capacity, no cortex, no delays.

    Matched in width and depth to one conduction group's encoder plus the head,
    so a gap between this and the loop is about the sheet and not about
    parameter count.
    """
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(len(c.channels), hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, 256), nn.GELU(),
                        nn.Linear(256, c.n_out)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    X = torch.from_numpy(c.X).to(dev); Y = torch.from_numpy(c.Y).to(dev)
    tr = torch.from_numpy(c.train).to(dev)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        i = tr[torch.from_numpy(rng.integers(0, len(c.train), 256)).to(dev)]
        loss = F.mse_loss(net(X[i]), Y[i])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        te = torch.from_numpy(c.test).to(dev)
        p = net(X[te]).cpu().numpy()
    return p * c.y_sd + c.y_mu


# ---------------------------------------------------------------------------
# the cortical arms
# ---------------------------------------------------------------------------

def build(c: InteroCorpus, a, dev, seed: int):
    torch.manual_seed(seed)
    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev,
                             long_range=a.long_range).to(dev)
    if a.init and os.path.exists(a.init):
        d = torch.load(a.init, map_location="cpu")
        e = d["dyn.embed"]
        if e.shape[0] == a.sites:
            dyn.embed.data.copy_(e.to(dev))
    model = P.InteroceptiveLoop(dyn, c.channels, n_out=c.n_out,
                                hidden=a.hidden, read_sites=a.read_sites).to(dev)
    return dyn, model


@torch.no_grad()
def evaluate(c: InteroCorpus, model, dev, batch: int, **kw) -> np.ndarray:
    model.eval()
    X = torch.from_numpy(c.X[c.test]).to(dev)
    out = []
    for i in range(0, len(X), batch):
        p, _ = model(X[i:i + batch], checkpoint_every=0, **kw)
        out.append(p.cpu().numpy())
    model.train()
    return np.concatenate(out) * c.y_sd + c.y_mu


def train_arm(c: InteroCorpus, a, dev, seed: int, kernel: str, drop: tuple,
              label: str):
    dyn, model = build(c, a, dev, seed)
    opt = torch.optim.AdamW(list(dyn.parameters()) + list(model.parameters()),
                            lr=a.lr, weight_decay=1e-4)
    X = torch.from_numpy(c.X).to(dev); Y = torch.from_numpy(c.Y).to(dev)
    tr = torch.from_numpy(c.train).to(dev)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(a.steps):
        i = tr[torch.from_numpy(
            rng.integers(0, len(c.train), a.batch)).to(dev)]
        pred, s = model(X[i], kernel=kernel, drop=drop)
        loss = F.mse_loss(pred, Y[i]) + 1e-1 * P.viability_penalty(s[0])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(dyn.parameters()) + list(model.parameters()), 1.0)
        opt.step()
        if step % max(1, a.steps // 8) == 0:
            print(f"    {label} step {step:4d}  train mse {loss.item():.5f}  "
                  f"{time.time()-t0:5.0f}s", flush=True)
    return dyn, model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/derived/intero-corpus")
    ap.add_argument("--horizon-s", type=float, default=60.0)
    ap.add_argument("--init", default="ckpt/ibm1_implicit.pt")
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--read-sites", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--control-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/intero_ablation.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = InteroCorpus(a.corpus, a.horizon_s)
    print(c.describe(), flush=True)

    base = baselines(c, c.test)
    gates = sanity_gates(c, c.test)
    print(f"\nsanity gates PASSED (worst deviation "
          f"{max(gates.values()):.1e}; each baseline predictor scores exactly "
          f"0 against itself)")
    print("\nbaseline MSE on the held-out split, raw units:")
    for j, t in enumerate(c.target_names):
        print(f"  {t:14s} zero {base['zero'][j]:12.5f}  "
              f"mean {base['mean'][j]:12.5f}  "
              f"persistence {base['persistence'][j]:12.5f}")

    # the aggregate is the mean over targets of the STANDARDISED error, so a
    # target whose raw variance is 30 (endurance, in hours) does not swamp one
    # whose raw variance is 0.004 (discomfort, a fraction).
    def agg(pred_raw):
        z = (pred_raw - c.y_mu) / c.y_sd
        zt = (c.Y_raw[c.test] - c.y_mu) / c.y_sd
        zm = ((np.broadcast_to(c.y_mu.astype(np.float32), zt.shape) - c.y_mu)
              / c.y_sd)
        return float(1.0 - ((z - zt) ** 2).mean() / ((zm - zt) ** 2).mean())

    results = {"config": vars(a), "corpus": c.describe(),
               "targets": c.target_names,
               "sanity_gates": gates,
               "baselines_raw": {k: v.tolist() for k, v in base.items()},
               "arms": {}}

    def record(name, pred, note=""):
        mse = per_target_mse(pred, c.Y_raw[c.test])
        A = agg(pred)
        results["arms"][name] = {"mse_raw": mse.tolist(), "agg_skill_vs_mean": A,
                                 "note": note, **skill(mse, base)}
        print(report_row(name, mse, base, c.target_names, A), flush=True)
        return A

    print("\n  arm                    aggregate   per target: "
          "skill vs mean / vs persistence")
    record("ridge_no_dynamics", ridge_control(c),
           "closed-form linear on the same 15 channels; no cortex")
    record("mlp_no_dynamics",
           mlp_control(c, dev, a.control_steps, a.hidden, a.lr, a.seed),
           "matched-capacity encoder; no cortex, no conduction delays")

    # --- the trained cortical arm -----------------------------------------
    print(f"\n  training the cortical arms ({a.steps} steps each, "
          f"seed {a.seed} matched)", flush=True)
    dyn, model = train_arm(c, a, dev, a.seed, "trained", (), "trained")
    print(f"\n{model.describe()}\n", flush=True)
    record("cortex_trained", evaluate(c, model, dev, a.eval_batch),
           "the learned association kernel")

    # --- post-hoc ablations on those weights -------------------------------
    record("cortex_severed_posthoc",
           evaluate(c, model, dev, a.eval_batch, kernel="severed"),
           "kernel zeroed on the trained weights: what the trained solution "
           "actually depends on")
    record("cortex_permuted_posthoc",
           evaluate(c, model, dev, a.eval_batch, kernel="permuted"),
           "kernel rows shuffled on the trained weights")
    for g in model.group_keys:
        record(f"drop_{g[0]}_{g[1]}",
               evaluate(c, model, dev, a.eval_batch, drop=(g,)),
               f"the {g[0]} {g[1]} conduction group withheld "
               f"({1000*model.delays_s[g]:.0f} ms)")
    slow = tuple(g for g in model.group_keys if g[1] == "c")
    fast = tuple(g for g in model.group_keys if g[1] != "c")
    record("drop_all_c_fibres", evaluate(c, model, dev, a.eval_batch, drop=slow),
           f"every unmyelinated group withheld ({len(slow)} of "
           f"{len(model.group_keys)}): the slow chemical and nociceptive arm of "
           f"visceral afference removed, the fast mechanical arm kept")
    record("drop_all_myelinated",
           evaluate(c, model, dev, a.eval_batch, drop=fast),
           f"every myelinated group withheld ({len(fast)} of "
           f"{len(model.group_keys)}): the converse")
    del dyn, model
    if dev == "cuda":
        torch.cuda.empty_cache()

    # --- retrained arms ----------------------------------------------------
    _, m2 = train_arm(c, a, dev, a.seed, "permuted", (), "permuted")
    record("cortex_permuted_retrained",
           evaluate(c, m2, dev, a.eval_batch, kernel="permuted"),
           "a PERMUTED kernel trained from the same seed for the same steps.  "
           "with a tonic drive and a whole-sheet readout this is the arm that "
           "separates a cortical result from a readout result, and ledger 20 "
           "says to expect it to match")
    del m2
    if dev == "cuda":
        torch.cuda.empty_cache()

    _, m3 = train_arm(c, a, dev, a.seed, "trained", slow, "no_c_fibres")
    record("cortex_trained_without_c",
           evaluate(c, m3, dev, a.eval_batch, drop=slow),
           "retrained with every unmyelinated group removed -- the fair version "
           "of the vagotomy, since a post-hoc drop only says the trained "
           "solution used the group, not that the group carries anything a "
           "model could not do without")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")

    # --- what it means -----------------------------------------------------
    A = results["arms"]
    ct = A["cortex_trained"]["agg_skill_vs_mean"]
    rd = A["ridge_no_dynamics"]["agg_skill_vs_mean"]
    ml = A["mlp_no_dynamics"]["agg_skill_vs_mean"]
    pm = A["cortex_permuted_retrained"]["agg_skill_vs_mean"]
    sv = A["cortex_severed_posthoc"]["agg_skill_vs_mean"]
    print("\nwhat this says:")
    print(f"  the cortical loop reaches {ct:+.4f} against predicting the mean; "
          f"a ridge on the same fifteen numbers reaches {rd:+.4f} and an MLP "
          f"{ml:+.4f}.")
    if ct <= max(rd, ml) + 0.01:
        print("  the loop does NOT beat a dynamics-free map on the same input. "
              "the afference carries the signal; the cortex is not adding to "
              "it here.")
    print(f"  severing the kernel post hoc moves the aggregate to {sv:+.4f} "
          f"({ct - sv:+.4f}); a permuted kernel retrained reaches {pm:+.4f} "
          f"({ct - pm:+.4f} behind the trained one).")
    if abs(ct - pm) < 0.01:
        print("  trained and permuted are indistinguishable, so the kernel's "
              "STRUCTURE is not carrying this term -- the readout is.  that is "
              "ledger entry 20 on a second pathway, and it is a result about "
              "the architecture, not about interoception.")


if __name__ == "__main__":
    main()
