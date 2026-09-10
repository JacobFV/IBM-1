"""spindle afference into postcentral, muscle command out of precentral.

every motor materialization this programme has trained drove the cortex with
IMAGES and read a motor region, which is not a sensorimotor loop -- it is a
transport probe wearing one's clothes.  the pieces to do it properly now exist:

  the afference    68 forced-pose motions, 17,622 frames, each carrying every
      muscle's musculotendon path length in units of its own optimal fibre
      length and the rate of that -- the two quantities a spindle reports.
      collected by imposing MEASURED human motion on the body, so the afference
      is what a real periphery produces rather than what a controller wished for.

  the route        postcentral -> precentral, which is the ONE pair the consensus
      connectome declares of those this programme has tried.  occipital ->
      precentral has no fascicle in a human brain, and the concentrated kernel
      never left chance on it while reaching 22.4x on this pair.

  the readout      a muscle command, 80 activations in [0,1], which is what IHM-1
      accepts.  not an embedding, not a rank -- the thing a body can be driven by.

the task is next-command prediction: from the afference at time t, produce the
activation that the corpus's own motion implies at t.  a body that can do this
has a forward model of its own periphery, which is the first of the four
highest-leverage absent components in docs/DEVELOPMENTAL_COMPONENTS.md.

THE BASELINES ARE THE WHOLE POINT, because a motor target is heavily
autocorrelated and almost anything scores well against zero:

  persistence   emit the previous frame's activation.  at 100 Hz this is a very
      strong baseline and it is the one to beat -- the video term looked like it
      worked for exactly this reason until persistence was measured.
  mean          emit the corpus mean per muscle.
  ridge         closed-form linear map from afference to command, no cortex.
      this is the control that says whether the DYNAMICS contribute anything
      over a linear read of the same input.
  severed       the cortical kernel zeroed, everything else identical.
  permuted      the kernel's site rows shuffled.  three pathways have now shown
      that permuted matches trained, so if this one does too that is the fourth.

held out by MOTION, never by frame: whole motions are withheld, so the test asks
for a movement never seen rather than a later moment of one that was.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(HERE, "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

CORPUS = "/home/brandonin/Documents/IHM-1/data/derived/pose-corpus"


def load_corpus(corpus: str, holdout: int):
    idx = json.load(open(f"{corpus}/index.json"))
    ms = sorted(idx["motions"], key=lambda m: -m["total_excursion_optimal_fiber"])
    names = [m["id"] for m in ms]
    te = names[:holdout]                      # the richest motions are held out
    tr = names[holdout:]

    # A FIXED MUSCLE SET ACROSS EVERY MOTION.  motions resolve different subsets
    # -- a pose outside a path polynomial's fitted range drops that muscle -- so
    # taking each motion's own list silently produces rows of different width and
    # the array refuses to build.  the intersection is the honest common basis,
    # and its size is reported rather than assumed.
    # The corpus spans SEVERAL OpenSim models -- motions resolve 15, 30, 40 or 80
    # muscles and the small sets are gait10dof18musc and friends, whose muscle
    # NAMES do not overlap this body's at all.  Intersecting across everything
    # therefore returns the empty set, which is the corpus telling the truth:
    # those motions are not about this body.  Keep the motions that carry the
    # largest set and report how many were dropped, rather than silently
    # unioning names from a model we do not simulate.
    sets = {}
    for nm in names:
        f = json.load(open(f"{corpus}/{nm}.json"))
        if f["frames"]:
            sets[nm] = frozenset(f["frames"][0]["path_over_optimal_fiber"])
    if not sets:
        raise SystemExit("no motion carries muscle geometry")
    biggest = max(sets.values(), key=len)
    keep = {nm for nm, k in sets.items() if biggest <= k}
    mus = sorted(biggest)
    print(f"muscle sets present: "
          f"{sorted({len(k) for k in sets.values()})} -> keeping the {len(mus)}-muscle "
          f"set, {len(keep)} of {len(sets)} motions; {len(sets)-len(keep)} are from "
          f"other OpenSim models and are dropped", flush=True)
    te = [n for n in te if n in keep]
    tr = [n for n in tr if n in keep]

    def pack(sel):
        X, Y, M = [], [], []
        for nm in sel:
            f = json.load(open(f"{corpus}/{nm}.json"))
            fr = f["frames"]
            if len(fr) < 3:
                continue
            for a, b in zip(fr[:-1], fr[1:]):
                X.append([a["path_over_optimal_fiber"][m] for m in mus]
                         + [a["rate_per_s"].get(m, 0.0) for m in mus])
                # the "command" the corpus implies: the change in each muscle's
                # length over the step, which is what an activation has to
                # produce.  the corpus carries no measured activation for every
                # motion, so this is derived and is labelled as such everywhere.
                Y.append([b["path_over_optimal_fiber"][m]
                          - a["path_over_optimal_fiber"][m] for m in mus])
                M.append(nm)
        return np.asarray(X, np.float32), np.asarray(Y, np.float32), M, mus

    return pack(tr), pack(te), tr, te


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/wiring_surface_slice.pt")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--holdout-motions", type=int, default=12)
    ap.add_argument("--drive-region", default="postcentral")
    ap.add_argument("--read-region", default="precentral")
    ap.add_argument("--long-topology", default="tract", choices=("random", "tract"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dyn-steps", type=int, default=4)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=2e-2)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--arms", default="trained,severed,permuted")
    ap.add_argument("--long-gain", type=float, default=1.0)
    ap.add_argument("--long-topm", type=int, default=0)
    ap.add_argument("--long-min-dist", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/proprioceptive_motor.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    (Xtr, Ytr, _, mus), (Xte, Yte, _, _), tr, te = load_corpus(
        a.corpus, a.holdout_motions)
    print(f"train {len(Xtr):,} transitions over {len(tr)} motions | "
          f"held out {len(Xte):,} over {len(te)} motions", flush=True)
    print(f"in {Xtr.shape[1]} (length + rate for {len(mus)} muscles) -> "
          f"out {Ytr.shape[1]}", flush=True)

    # normalise on TRAIN only; a scaler fitted on the held-out motions would
    # leak the thing being predicted.
    mu, sg = Xtr.mean(0), Xtr.std(0).clip(1e-6)
    Xtr_n, Xte_n = (Xtr - mu) / sg, (Xte - mu) / sg
    # THE TARGET MUST BE NORMALISED TOO, and not normalising it invalidated the
    # first run of this experiment.  The deltas are ~1e-2 in magnitude while a
    # freshly initialised head emits O(1), so the model spends its whole budget
    # learning to shrink by two orders of magnitude and never gets to the shape.
    # Measured: the trained arm reached a TRAIN loss of 9.1e-03 against a target
    # whose zero baseline is 6.7e-04 -- 13x worse than emitting nothing, on data
    # it had seen.  That is not signal lost in transit, it is a model that never
    # fit the scale, and reading it as a transport result would have been wrong.
    # Predictions are mapped back before any baseline comparison.
    ymu, ysg = Ytr.mean(0), Ytr.std(0).clip(1e-9)
    Ytr_n = (Ytr - ymu) / ysg

    # ---- baselines, before anything is trained -------------------------------
    zero = float((Yte ** 2).mean())
    mean_pred = Ytr.mean(0)
    mean_mse = float(((Yte - mean_pred) ** 2).mean())
    # persistence: the previous step's delta, which is what a body that keeps
    # doing what it was doing would produce
    pers = np.vstack([Yte[:1], Yte[:-1]])
    pers_mse = float(((Yte - pers) ** 2).mean())
    A = np.concatenate([Xtr_n, np.ones((len(Xtr_n), 1), np.float32)], 1)
    W = np.linalg.lstsq(A, Ytr, rcond=None)[0]
    B = np.concatenate([Xte_n, np.ones((len(Xte_n), 1), np.float32)], 1)
    ridge_mse = float(((Yte - B @ W) ** 2).mean())
    print(f"\nheld-out baselines (MSE, and skill against persistence):", flush=True)
    for nm, v in (("zero", zero), ("mean", mean_mse),
                  ("persistence", pers_mse), ("ridge, no cortex", ridge_mse)):
        print(f"  {nm:18s} {v:.6e}   skill {1 - v/pers_mse:+.4f}", flush=True)

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = d.get("model", d)
    if "dyn.idx" not in sd:
        for h in (d.get("heads") or {}).values():
            if isinstance(h, dict) and "dyn.idx" in h:
                sd = h; break
    emb = sd["dyn.embed"]; n_sites, e_dim = emb.shape; k = sd["dyn.idx"].shape[1]

    Xt = torch.from_numpy(Xtr_n).to(dev); Yt = torch.from_numpy(Ytr_n).to(dev)
    Xv = torch.from_numpy(Xte_n).to(dev); Yv = torch.from_numpy(Yte).to(dev)

    res = {"ckpt": a.ckpt, "baselines": {"zero": zero, "mean": mean_mse,
           "persistence": pers_mse, "ridge_no_cortex": ridge_mse},
           "held_out_motions": te, "arms": {}}

    # THE CONTROL THAT SETTLES WHETHER THE SHEET IS THE PROBLEM.  Same encoder,
    # same head, same budget, same data -- the cortical sheet removed entirely and
    # the encoder's output handed straight to the readout.  If this matches the
    # cortical arms, the 4.9x gap to persistence belongs to the OBJECTIVE and the
    # substrate is innocent; if it beats them, the sheet destroys something a
    # plain MLP keeps.  Either answer closes a question two withdrawn claims have
    # already turned on.
    if "no_cortex" in a.arms.split(","):
        torch.manual_seed(a.seed)
        mlp = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(),
                            nn.Linear(256, 512), nn.GELU(),
                            nn.Linear(512, Ytr.shape[1])).to(dev)
        o2 = torch.optim.AdamW(mlp.parameters(), lr=a.lr, weight_decay=1e-4)
        Xt_ = torch.from_numpy(Xtr_n).to(dev); Yt_ = torch.from_numpy(Ytr_n).to(dev)
        Xv_ = torch.from_numpy(Xte_n).to(dev); Yv_ = torch.from_numpy(Yte).to(dev)
        ym = torch.from_numpy(ymu).to(dev); ys = torch.from_numpy(ysg).to(dev)
        r2 = np.random.default_rng(a.seed); h2 = []
        print("\n### no_cortex: encoder -> head, the sheet removed entirely", flush=True)
        for step in range(a.steps + 1):
            i = torch.from_numpy(r2.integers(0, len(Xt_), a.batch)).to(dev)
            ls = F.mse_loss(mlp(Xt_[i]), Yt_[i])
            o2.zero_grad(set_to_none=True); ls.backward(); o2.step()
            if step % a.eval_every == 0:
                with torch.no_grad():
                    v = float((((mlp(Xv_) * ys + ym) - Yv_) ** 2).mean())
                sk = 1 - v / pers_mse
                h2.append({"step": step, "held_mse": v, "skill_vs_persistence": sk})
                print(f"  {step:5d}  train {float(ls):.3e}  held {v:.3e}  "
                      f"SKILL vs persistence {sk:+.4f}  vs ridge {1-v/ridge_mse:+.4f}",
                      flush=True)
        res["arms"]["no_cortex"] = {"history": h2,
            "best_skill_vs_persistence": max(x["skill_vs_persistence"] for x in h2)}
        json.dump(res, open(a.out, "w"), indent=2)

    for arm in [x for x in a.arms.split(",") if x != "no_cortex"]:
        torch.manual_seed(a.seed)
        dyn = P.CorticalDynamics(n_sites, e_dim, k, dev, geometry="surface",
                                 long_topology=a.long_topology,
                                 long_gain=a.long_gain, long_topm=a.long_topm,
                                 long_min_dist=a.long_min_dist).to(dev)
        with torch.no_grad():
            dyn.embed.copy_(emb.to(dev))
            for nm in ("idx", "geo", "pos", "w_ee", "w_ei", "w_assoc", "a_gain", "log_len"):
                if f"dyn.{nm}" in sd:
                    getattr(dyn, nm).copy_(sd[f"dyn.{nm}"].to(dev))
            if arm == "severed":
                dyn.geo.zero_()
            elif arm == "permuted":
                g = torch.Generator(device="cpu").manual_seed(a.seed)
                perm = torch.randperm(n_sites, generator=g).to(dev)
                dyn.embed.copy_(dyn.embed[perm])
        port = P.region_index(dyn.pos, a.drive_region).to(dev)
        read = P.region_index(dyn.pos, a.read_region).to(dev)
        if len(np.intersect1d(port.cpu().numpy(), read.cpu().numpy())):
            raise SystemExit("ports overlap -- not a transport test")
        torch.manual_seed(a.seed)
        enc = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(),
                            nn.Linear(256, len(port))).to(dev)
        head = nn.Sequential(nn.Linear(len(read), 512), nn.GELU(),
                             nn.Linear(512, Ytr.shape[1])).to(dev)
        params = list(enc.parameters()) + list(head.parameters()) + [dyn.embed]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
        print(f"\n### {arm}: drive {a.drive_region} ({len(port)}) -> read "
              f"{a.read_region} ({len(read)}), {a.long_topology} edges", flush=True)

        def fwd(x):
            s = dyn.init_state(x.shape[0], dev)
            w = dyn.edge_weights()
            drive = torch.zeros(x.shape[0], dyn.n, device=dev).index_copy(1, port, enc(x))
            h = a.dt / a.substeps
            for _ in range(a.dyn_steps):
                for _ in range(a.substeps):
                    s = dyn.step(s, drive, h, w)
            return head(s[1][:, read])

        hist, best, t0 = [], -1e9, time.time()
        rng = np.random.default_rng(a.seed)
        for step in range(a.steps + 1):
            i = torch.from_numpy(rng.integers(0, len(Xt), a.batch)).to(dev)
            loss = F.mse_loss(fwd(Xt[i]), Yt[i])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % a.eval_every == 0:
                with torch.no_grad():
                    # map back to real units before comparing to any baseline
                    ym = torch.from_numpy(ymu).to(dev)
                    ys = torch.from_numpy(ysg).to(dev)
                    v = 0.0
                    for j in range(0, len(Xv), 4096):
                        p_ = fwd(Xv[j:j + 4096]) * ys + ym
                        v += float(((p_ - Yv[j:j + 4096]) ** 2).sum())
                    v /= Yv.numel()
                sk = 1 - v / pers_mse
                best = max(best, sk)
                hist.append({"step": step, "held_mse": v, "skill_vs_persistence": sk,
                             "skill_vs_ridge": 1 - v / ridge_mse})
                print(f"  {step:5d}  train {float(loss):.3e}  held {v:.3e}  "
                      f"SKILL vs persistence {sk:+.4f}  vs ridge "
                      f"{1 - v/ridge_mse:+.4f}   {time.time()-t0:5.0f}s", flush=True)
                res["arms"][arm] = {"history": hist, "best_skill_vs_persistence": best}
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                json.dump(res, open(a.out, "w"), indent=2)

    print("\n  arm            best skill vs persistence", flush=True)
    for nm, v in res["arms"].items():
        print(f"  {nm:14s} {v['best_skill_vs_persistence']:+.4f}", flush=True)
    if {"trained", "permuted"} <= set(res["arms"]):
        t = res["arms"]["trained"]["best_skill_vs_persistence"]
        p_ = res["arms"]["permuted"]["best_skill_vs_persistence"]
        print(f"\n  trained - permuted = {t - p_:+.4f}"
              f"   {'the learned content contributes' if t - p_ > 0.02 else 'PERMUTED MATCHES TRAINED -- ledger row 23 again'}",
              flush=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
