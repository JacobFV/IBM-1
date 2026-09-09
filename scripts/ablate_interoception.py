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

    def __init__(self, root: str, horizon_s: float, split: str = "protocol"):
        self.root, self.horizon_s, self.split = root, horizon_s, split
        m = json.load(open(os.path.join(root, "meta.json")))
        self.meta = m
        X = np.load(os.path.join(root, "afferent.npy"))
        Y = np.load(os.path.join(root, "scalars.npy"))
        S = np.load(os.path.join(root, "sensation.npy"))
        seq = np.load(os.path.join(root, "sequence.npy"))
        if split == "protocol":
            tr = np.load(os.path.join(root, "is_train_protocol.npy"))
            te = np.load(os.path.join(root, "is_test_protocol.npy"))
        elif split == "time":
            tr = np.load(os.path.join(root, "is_train.npy"))
            te = np.load(os.path.join(root, "is_test.npy"))
        else:
            raise ValueError(f"unknown split {split!r}")

        # the join is checked, not trusted: the channel table lives in the other
        # repository and a renamed channel would wire a port to a rate nobody
        # sends while the encoder still ran.
        import ibm.interoception as IO
        IO.check(m)
        self.channels = m["channels"]

        T = np.concatenate([Y, S], 1).astype(np.float32)
        self.target_names = list(m["scalars"]) + \
            [f"sensation_{i}" for i in range(S.shape[1])]
        # WHICH TARGETS ARE FUNCTIONS OF THE INPUT, and it changes which
        # baseline is the one that bites.
        #
        # `discomfort` is a fixed fibre-class-weighted sum of the fifteen
        # afferent rates and `sensation_k` is a fixed linear projection of them,
        # so a model that sees the afference at t can compute both at t exactly.
        # Predicting them at t+h against "predict the mean" therefore scores
        # ~0.99 for a linear map and means nothing: the ridge is inverting its
        # own input.  PERSISTENCE is the baseline for those.
        #
        # `endurance_h` is not: it is liver and muscle glycogen against
        # metabolic rate, and none of the three is an afferent channel.  It has
        # to be inferred, which is the interesting question -- and persistence
        # is the wrong baseline for it in the other direction, because it moves
        # by 0.007 h over 60 s, so persistence MSE is 4.5e-05 and every skill
        # ratio against it is a division by measurement noise.  MEAN is the
        # baseline for that one.
        self.derived_from_afference = [False] + [True] * (T.shape[1] - 1)
        self.baseline_of_record = ["mean"] + ["persistence"] * (T.shape[1] - 1)

        i_in, i_out, which = [], [], []
        for s in m["sequences"]:
            k = int(round(horizon_s / s["dt_s"]))
            if k < 1:
                raise ValueError(f"horizon {horizon_s}s is under one sample of "
                                 f"{s['run']}/{s['protocol']} (dt {s['dt_s']}s)")
            idx = np.nonzero(seq == s["index"])[0]
            a, b = idx[:-k], idx[k:]
            for u, v in zip(a, b):
                if tr[u] and tr[v]:
                    i_in.append(u); i_out.append(v); which.append(0)
                elif te[u] and te[v]:
                    i_in.append(u); i_out.append(v); which.append(1)
        self.i_in = np.asarray(i_in); self.i_out = np.asarray(i_out)
        which = np.asarray(which)
        self.train = np.nonzero(which == 0)[0]
        self.test = np.nonzero(which == 1)[0]

        # standardisation from the TRAIN pairs of THIS split only.
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
        held = self.meta.get("held_out_protocols", [])
        return (f"{len(self.i_in)} pairs at horizon {self.horizon_s:.0f}s, "
                f"{self.split} split "
                f"({len(self.train)} train / {len(self.test)} test) over "
                f"{len(self.meta['sequences'])} sequences"
                + (f", holding out {', '.join(held)}"
                   if self.split == "protocol" else "") +
                f"; {len(self.channels)} channels -> {self.n_out} targets "
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
    """check the metric against cases whose answers are known.

    Two kinds, and both have to pass before any arm is reported.

    *self-skill is exactly zero.*  A predictor that IS a baseline must score 0
    against that baseline.  If it does not, the metric is wrong and every number
    after it is decoration.

    *an exactly-linear target must print 1.000.*  `discomfort` is a fixed
    weighted sum of the fifteen afferent rates, so a ridge from the afference to
    `discomfort` AT THE SAME INSTANT has an exact solution and must recover it.
    This is the "chance prints 1.0x" check for a regression pipeline: a known
    answer computed the same way every other number here is computed.  It also
    makes the degeneracy visible rather than flattering -- the reason
    `discomfort` and `sensation` are scored against persistence and not against
    the mean is precisely that this gate passes.
    """
    truth = c.Y_raw[idx]
    base = baselines(c, idx)
    got = {}
    for name, pred in (("zero", np.zeros_like(truth)),
                       ("mean", np.broadcast_to(c.y_mu.astype(np.float32),
                                                truth.shape)),
                       ("persistence", c.P_raw[idx])):
        sk = np.asarray(skill(per_target_mse(pred, truth), base)[f"skill_vs_{name}"])
        worst = float(np.nanmax(np.abs(sk)))
        got[f"self_skill_{name}"] = worst
        if worst > 1e-9:
            raise SystemExit(
                f"SANITY GATE FAILED: the {name} predictor scores skill "
                f"{worst:.3e} against the {name} baseline, and must score 0. "
                f"the metric is wrong; nothing below it means anything.")

    # the exactly-linear target.  X is standardised, discomfort is affine in the
    # raw rates, so an affine map exists and a ridge must find it.
    j = c.target_names.index("discomfort")
    Xtr = np.c_[c.X[c.train], np.ones(len(c.train), np.float32)]
    P0 = c.P_raw[:, j]                        # discomfort at the INPUT instant
    ytr = P0[c.train]
    W = np.linalg.solve(Xtr.T @ Xtr + 1e-6 * np.eye(Xtr.shape[1], dtype=np.float32),
                        Xtr.T @ ytr)
    Xte = np.c_[c.X[idx], np.ones(len(idx), np.float32)]
    r2 = float(1.0 - ((Xte @ W - P0[idx]) ** 2).mean() /
               ((P0[idx] - ytr.mean()) ** 2).mean())
    got["linear_target_r2"] = r2
    if r2 < 0.999:
        raise SystemExit(
            f"SANITY GATE FAILED: a ridge from the afference to `discomfort` at "
            f"the same instant scores R^2 {r2:.6f} and must score 1.000 -- "
            f"discomfort is an exact affine function of those fifteen rates. "
            f"the corpus, the standardisation or the pairing is wrong.")
    return got


def change_guard(c: InteroCorpus, pred: np.ndarray, idx: np.ndarray) -> dict:
    """how big is the predicted CHANGE, and does it point the right way?

    Ledger entry 14: an objective reported skill +0.0003 over persistence and
    was announced as beating it, and what the model had actually learned was to
    emit zero -- which IS persistence.  The loss could not tell the difference
    and neither could the skill.  What separated them was measuring the
    predicted residual's size and direction.

    So for every arm: the ratio of the predicted change's magnitude to the true
    change's, and the cosine between them.  A ratio near 0 means the arm has
    reproduced persistence whatever its skill says.
    """
    true_d = c.Y_raw[idx] - c.P_raw[idx]
    pred_d = pred - c.P_raw[idx]
    out = {}
    for j, t in enumerate(c.target_names):
        a, b = pred_d[:, j], true_d[:, j]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        out[t] = {"magnitude_ratio": na / nb if nb > 0 else float("nan"),
                  "cosine": float(a @ b / (na * nb)) if na * nb > 0 else float("nan")}
    return out


def headline(c: InteroCorpus, mse: np.ndarray, base: dict) -> dict:
    """two numbers, because there are two questions and one aggregate is a lie.

    *state*: can the cortex read UNOBSERVED body state off the afference?
    `endurance_h` is glycogen against metabolic rate and none of those is a
    channel, so this is inference and the baseline is the training mean.

    *trajectory*: does the afference say where the visceral state is GOING?
    `discomfort` and `sensation` are exact functions of the afference at the
    same instant -- the sanity gate proves it -- so scoring them against the
    mean measures nothing but the model's ability to invert its own input.  The
    baseline that bites is persistence, and beating it requires extrapolating
    the afferent trajectory rather than reading it.

    A single aggregate over all five would mix a division by 4.5e-05 with a
    division by 1.6, which is how a number stops meaning anything.
    """
    sk_mean = 1.0 - mse / base["mean"]
    sk_pers = 1.0 - mse / base["persistence"]
    st = [j for j, d in enumerate(c.derived_from_afference) if not d]
    tj = [j for j, d in enumerate(c.derived_from_afference) if d]
    return {"state_skill_vs_mean": float(np.mean(sk_mean[st])),
            "trajectory_skill_vs_persistence": float(np.mean(sk_pers[tj]))}


def report_row(name: str, c: InteroCorpus, mse: np.ndarray, base: dict,
               h: dict, guard: dict) -> str:
    sk_mean = 1.0 - mse / base["mean"]
    sk_pers = 1.0 - mse / base["persistence"]
    parts = []
    for j, t in enumerate(c.target_names):
        v = sk_pers[j] if c.derived_from_afference[j] else sk_mean[j]
        parts.append(f"{t.replace('sensation_', 's')}={v:+.3f}")
    dmag = np.mean([guard[t]["magnitude_ratio"]
                    for j, t in enumerate(c.target_names)
                    if c.derived_from_afference[j]])
    return (f"  {name:26s} state {h['state_skill_vs_mean']:+.4f}  "
            f"traj {h['trajectory_skill_vs_persistence']:+.4f}  |dz| "
            f"{dmag:5.2f}  " + " ".join(parts))


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
                seed: int, batch: int = 256) -> np.ndarray:
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
        i = tr[torch.from_numpy(rng.integers(0, len(c.train), batch)).to(dev)]
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
    # `model` holds `dyn` as a submodule, so model.parameters() already contains
    # every kernel parameter.  concatenating the two lists put each of them in
    # the optimizer TWICE, and AdamW then applies its update twice per step --
    # a silently doubled learning rate on the one tensor the whole experiment is
    # about.  torch warns; the warning is easy to scroll past.  filter the way
    # `train_curriculum.py` does.
    head = [p for n, p in model.named_parameters() if not n.startswith("dyn.")]
    params = list(dyn.parameters()) + head
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
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
        torch.nn.utils.clip_grad_norm_(params, 1.0)
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
    ap.add_argument("--split", default="protocol", choices=("protocol", "time"),
                    help="protocol holds out whole recorded runs and is the "
                         "default; time is contiguous within each sequence.  "
                         "the time split is ill-posed for endurance_h -- its "
                         "held-out window is a 1.6 h band at the top of a 27 h "
                         "range, where a model accurate to 2.8%% scores -2.07 "
                         "against the mean")
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
    c = InteroCorpus(a.corpus, a.horizon_s, a.split)
    print(c.describe(), flush=True)

    base = baselines(c, c.test)
    gates = sanity_gates(c, c.test)
    print(f"\nsanity gates PASSED")
    print(f"  every baseline predictor scores exactly 0 against itself "
          f"(worst {max(v for k, v in gates.items() if k.startswith('self')):.1e})")
    print(f"  the exactly-linear target prints "
          f"{gates['linear_target_r2']:.6f}, and must print 1.000000 -- "
          f"`discomfort` is an affine function of the fifteen rates, so a ridge "
          f"has to recover it.  that it does is also WHY discomfort and "
          f"sensation are scored against persistence and not the mean.")
    print("\nbaseline MSE on the held-out split, raw units:")
    for j, t in enumerate(c.target_names):
        print(f"  {t:14s} zero {base['zero'][j]:12.6f}  "
              f"mean {base['mean'][j]:12.6f}  "
              f"persistence {base['persistence'][j]:12.6f}   "
              f"baseline of record: {c.baseline_of_record[j]}"
              + ("   (derived from the afference)"
                 if c.derived_from_afference[j] else
                 "   (NOT an afferent channel; must be inferred)"))
    # whether persistence is a usable baseline for endurance depends on the
    # SPLIT, and the number decides it rather than the prose.  on a contiguous
    # time split the held-out endurance window is nearly flat, persistence MSE
    # is 4.5e-05 against a variance of 1.58, and a skill ratio against it is a
    # division by measurement noise.  on the protocol split the held-out runs
    # contain an exercise bout, endurance moves 6 h inside it, and persistence
    # becomes an ordinary and rather strong baseline.  Same quantity, same
    # formula, two different meanings -- which is the entire shape of this
    # repo's corrections ledger, so it is checked and printed rather than
    # assumed.
    ratio = base["persistence"][0] / base["mean"][0]
    print(f"\n  endurance: persistence MSE {base['persistence'][0]:.3e} against "
          f"a held-out variance of {base['mean'][0]:.3f}, ratio {ratio:.2e}.")
    if ratio < 1e-3:
        print("    persistence is near-exact here, so a skill ratio against it "
              "is a division by noise; endurance is reported against the MEAN.")
    else:
        print("    endurance moves enough on this split for persistence to be "
              "an ordinary baseline; both ratios are reported for it, and the "
              "headline stays the mean because that is the inference question.")

    results = {"config": vars(a), "corpus": c.describe(),
               "targets": c.target_names,
               "derived_from_afference": c.derived_from_afference,
               "baseline_of_record": c.baseline_of_record,
               "sanity_gates": gates,
               "baselines_raw": {k: v.tolist() for k, v in base.items()},
               "arms": {}}

    def record(name, pred, note=""):
        mse = per_target_mse(pred, c.Y_raw[c.test])
        h = headline(c, mse, base)
        guard = change_guard(c, pred, c.test)
        results["arms"][name] = {"mse_raw": mse.tolist(), **h,
                                 "change_guard": guard,
                                 "note": note, **skill(mse, base)}
        print(report_row(name, c, mse, base, h, guard), flush=True)
        return h

    print("\n  arm                        state = endurance skill vs MEAN; "
          "traj = mean skill vs PERSISTENCE over the four afference-derived\n"
          "                             targets; |dz| = predicted change "
          "magnitude / true change magnitude (ledger 14: a model that\n"
          "                             emits persistence scores 0 skill and "
          "|dz| ~ 0, and the loss cannot tell you which)\n")
    record("ridge_no_dynamics", ridge_control(c),
           "closed-form linear on the same 15 channels; no cortex")
    record("mlp_no_dynamics",
           mlp_control(c, dev, a.control_steps, a.hidden, a.lr, a.seed),
           f"matched-capacity encoder, no cortex and no conduction delays, "
           f"trained to convergence ({a.control_steps} steps x 256 = "
           f"{a.control_steps*256:,} samples)")
    # MATCHED SAMPLES, not matched convergence.  the cortical arm costs 590 ms
    # of simulated cortex per forward and can only afford a few hundred steps,
    # so comparing it to a control that saw a hundred times the data would be
    # comparing budgets and calling it architecture.  this arm sees exactly what
    # the cortical arms see.
    record("mlp_matched_samples",
           mlp_control(c, dev, a.steps, a.hidden, a.lr, a.seed, batch=a.batch),
           f"the same encoder on the same number of samples the cortical arms "
           f"get ({a.steps} steps x {a.batch} = {a.steps*a.batch:,})")

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
    def st(k): return A[k]["state_skill_vs_mean"]
    def tj(k): return A[k]["trajectory_skill_vs_persistence"]
    print("\nwhat this says:")
    print(f"  STATE (can the afference report unobserved substrate?)  "
          f"cortex {st('cortex_trained'):+.4f}   ridge "
          f"{st('ridge_no_dynamics'):+.4f}   MLP {st('mlp_no_dynamics'):+.4f}")
    print(f"  TRAJECTORY (does it predict where the state is going?)  "
          f"cortex {tj('cortex_trained'):+.4f}   ridge "
          f"{tj('ridge_no_dynamics'):+.4f}   MLP {tj('mlp_no_dynamics'):+.4f}")
    best_free = max(st('ridge_no_dynamics'), st('mlp_no_dynamics'))
    if st('cortex_trained') <= best_free + 0.01:
        print("  the loop does NOT beat a dynamics-free map on the same input. "
              "whatever the afference carries, the cortex is not adding to it "
              "here, and the honest report is the afference's skill and not the "
              "loop's.")
    d_sev = st('cortex_trained') - st('cortex_severed_posthoc')
    d_perm = st('cortex_trained') - st('cortex_permuted_retrained')
    print(f"  severing the kernel post hoc costs {d_sev:+.4f} on state; a "
          f"permuted kernel RETRAINED reaches {st('cortex_permuted_retrained'):+.4f} "
          f"({d_perm:+.4f} behind trained).")
    if abs(d_perm) < 0.02:
        print("  trained and permuted are indistinguishable, so the kernel's "
              "STRUCTURE is not carrying this term -- the readout is.  that is "
              "ledger entry 20 reproduced on a second pathway, and it is a "
              "result about the architecture rather than about interoception.")
    d_c = st('cortex_trained') - st('cortex_trained_without_c')
    print(f"  removing every unmyelinated group and retraining costs {d_c:+.4f} "
          f"on state and "
          f"{tj('cortex_trained') - tj('cortex_trained_without_c'):+.4f} on "
          f"trajectory.  those five C-fibre channels are the slow chemical and "
          f"nociceptive arm, and they are the half of visceral afference that "
          f"arrives 168-508 ms after the mechanical half.")


if __name__ == "__main__":
    main()
