"""self-supervised video continuation across many films, split BY FILM.

two things were wrong with every video result before this one, and the corpus is
what fixed the first:

*eleven minutes of one film*.  train and test frames came from the same film, so
a held-out frame was a near-duplicate of a training frame -- the split measured
memorisation, not generalisation.  the via-EEG arms overfit 9x on it and could
not be evaluated at all.  here entire FILMS are held out, so the test asks
whether the model continues footage it has never seen.

*no persistence baseline*.  the 40k-step run reported recon falling 0.540 ->
0.011, a 50x improvement, and was **25% worse than emitting frame t unchanged**
(ledger row 13).  at horizon 8 on 25 fps film, frame t+H closely resembles frame
t, so a zero baseline flatters any model that reproduces the input.  persistence
is the only baseline continuation has to clear, and it is reported every eval
here.

the corpus grows while this trains -- the acquisition writes one pair of arrays
per film -- so the film list is read once at startup and the run states which
films it actually used.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/derived/pd-film")
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--dyn-steps", type=int, default=8)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--target", choices=("direct", "residual"), default="residual",
                    help="what the decoder predicts.  'direct' emits frame t+H and "
                         "is minimised by blur, which loses to persistence by "
                         "construction because persistence keeps the edges -- "
                         "measured at skill -1.08.  'residual' emits the CHANGE, so "
                         "a zero output IS persistence and the model starts at "
                         "skill 0 and can only improve on it")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout-films", type=int, default=2)
    ap.add_argument("--shuffle-targets", action="store_true",
                    help="pair each frame t with a RANDOM frame t+H from elsewhere "
                         "in the film.  the image statistics, the encoder's input "
                         "distribution and the gradient magnitudes are unchanged; "
                         "only the temporal correspondence is destroyed.  if the "
                         "cortical kernel still transfers to the EEG task under "
                         "this, the transfer is a property of running gradients "
                         "through the dynamics and not of learning to predict")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also save an unconditional checkpoint every N steps.  the "
                         "`best` gate selects on skill against persistence, and "
                         "skill does NOT predict what this run is actually for: "
                         "wiring that transfers.  video_via_eeg_learned scored "
                         "-5.80 on its own task and transferred 90.1%%, while "
                         "video_multifilm scored -0.66 and transferred worse than "
                         "random.  snapshots let transfer be measured as a function "
                         "of steps rather than inferred from the wrong metric")
    ap.add_argument("--ckpt", default="ckpt/video_multifilm.pt")
    ap.add_argument("--out", default="out/video_multifilm.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    films = sorted(glob.glob(f"{a.corpus}/*_frames.npy"))
    if len(films) < a.holdout_films + 2:
        print(f"only {len(films)} films; need at least {a.holdout_films+2}"); return
    tr_f, te_f = films[:-a.holdout_films], films[-a.holdout_films:]
    TR = [np.load(f, mmap_mode="r") for f in tr_f]
    TE = [np.load(f, mmap_mode="r") for f in te_f]
    n_tr = sum(len(x) for x in TR); n_te = sum(len(x) for x in TE)
    print(f"train {len(TR)} films, {n_tr:,} frames ({n_tr/25/3600:.2f} h)", flush=True)
    for f in tr_f:
        print(f"    {os.path.basename(f)}", flush=True)
    print(f"HELD-OUT {len(TE)} films, {n_te:,} frames ({n_te/25/3600:.2f} h)", flush=True)
    for f in te_f:
        print(f"    {os.path.basename(f)}", flush=True)

    def draw(pool, m, rng):
        """a batch of (frame t, frame t+H) pairs from a random film each time."""
        xs, ys = [], []
        for _ in range(m):
            v = pool[rng.integers(len(pool))]
            i = int(rng.integers(0, len(v) - a.horizon - 1))
            j = (int(rng.integers(0, len(v) - 1)) if a.shuffle_targets
                 else i + a.horizon)
            xs.append(np.asarray(v[i])); ys.append(np.asarray(v[j]))
        f = lambda t: (torch.from_numpy(np.stack(t)).to(dev)
                       .permute(0, 3, 1, 2).float() / 127.5) - 1.0
        return f(xs), f(ys)

    rng_te = np.random.default_rng(7)
    with torch.no_grad():
        xb, yb = draw(TE, 256, np.random.default_rng(7))
        zero_mse = float((yb ** 2).mean())
        persist_mse = float(F.mse_loss(xb, yb))
    print(f"\nheld-out baselines: zero {zero_mse:.5f}  PERSISTENCE {persist_mse:.5f}"
          f"  (the one to beat)\n", flush=True)

    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
    model = P.VideoLoop(dyn).to(dev)
    tot = sum(p.numel() for p in model.parameters())
    print(f"params {tot:,} ({dyn.embed.numel():,} association) | target={a.target}",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    log = {"config": vars(a), "n_params": tot, "train_films": tr_f,
           "holdout_films": te_f, "zero_mse": zero_mse,
           "persistence_mse": persist_mse, "steps": []}
    best, t0 = -1e9, time.time()
    rng = np.random.default_rng(0)
    for step in range(a.steps + 1):
        x, y = draw(TR, a.batch, rng)
        out, s = model(x, a.dyn_steps, a.dt)
        # residual: the decoder emits the CHANGE and it is added to frame t, so the
        # zero output is exactly persistence.  the model cannot do worse than the
        # trivial baseline by blurring -- it has to spend capacity on what moves.
        pred = (x + out) if a.target == "residual" else out
        loss = F.mse_loss(pred, y) + 1e-1 * P.viability_penalty(s[0])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % a.eval_every == 0:
            with torch.no_grad():
                xt, yt = draw(TE, 128, np.random.default_rng(1000 + step))
                ot, st = model(xt, a.dyn_steps, a.dt)
                pt = (xt + ot) if a.target == "residual" else ot
                held = float(F.mse_loss(pt, yt))
                # skill at +/-0.002 of persistence is consistent with two very
                # different models: one that learned the motion, and one that
                # learned to emit ZERO -- which IS persistence.  the loss cannot
                # separate them, so the residual's size and direction are reported
                # beside it.  measured at step 750 of the first residual run:
                # ratio 0.039, cosine -0.0016, i.e. fully degenerate while the
                # log line read "BEATS PERSISTENCE".
                res_t = yt - xt
                res_p = ot if a.target == "residual" else (ot - xt)
                ratio = float(res_p.pow(2).mean().sqrt() /
                              res_t.pow(2).mean().sqrt().clamp_min(1e-9))
                cos = float(F.cosine_similarity(res_p.flatten(1),
                                                res_t.flatten(1)).mean())
                p_ = float(F.mse_loss(xt, yt))
                r_eff = P.effective_rank(st[1][:, ::max(dyn.n // 512, 1)].float())
            skill = 1 - held / p_
            log["steps"].append({"step": step, "train": float(loss.detach()),
                                 "held": held, "persistence": p_,
                                 "skill_vs_persistence": skill,
                                 "skill_vs_zero": 1 - held / zero_mse,
                                 "residual_ratio": ratio, "residual_cos": cos,
                                 "r_eff": r_eff, "sec": round(time.time() - t0, 1)})
            flag = ""
            if skill > best and ratio > 0.15 and cos > 0.1:
                best = skill
                flag = "  <- best"
                os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step,
                            "skill_vs_persistence": skill, "config": vars(a)}, a.ckpt)
            # a "win" that comes from emitting nothing is not a win.
            if a.snapshot_every and step % a.snapshot_every == 0 and step:
                snap = a.ckpt.replace(".pt", f".step{step:06d}.pt")
                torch.save({"model": model.state_dict(), "step": step,
                            "skill_vs_persistence": skill, "config": vars(a)}, snap)
            mark = ("  BEATS PERSISTENCE" if (skill > 0 and ratio > 0.15 and cos > 0.1)
                    else "  [degenerate: emits ~nothing]" if ratio < 0.15 else "")
            print(f"{step:6d}  train {float(loss):.5f}  held {held:.5f}  "
                  f"persist {p_:.5f}  SKILL {skill:+.4f}  res {ratio:.3f} "
                  f"cos {cos:+.3f}  r_eff {r_eff:5.2f}  "
                  f"{time.time()-t0:5.0f}s{flag}{mark}", flush=True)

    log["best_skill"] = best
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"best skill against persistence: {best:+.4f}", flush=True)


if __name__ == "__main__":
    main()
