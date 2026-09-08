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
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout-films", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=250)
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
            xs.append(np.asarray(v[i])); ys.append(np.asarray(v[i + a.horizon]))
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
    print(f"params {tot:,} ({dyn.embed.numel():,} association)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    log = {"config": vars(a), "n_params": tot, "train_films": tr_f,
           "holdout_films": te_f, "zero_mse": zero_mse,
           "persistence_mse": persist_mse, "steps": []}
    best, t0 = -1e9, time.time()
    rng = np.random.default_rng(0)
    for step in range(a.steps + 1):
        x, y = draw(TR, a.batch, rng)
        pred, s = model(x, a.dyn_steps, a.dt)
        loss = F.mse_loss(pred, y) + 1e-1 * P.viability_penalty(s[0])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % a.eval_every == 0:
            with torch.no_grad():
                xt, yt = draw(TE, 128, np.random.default_rng(1000 + step))
                pt, st = model(xt, a.dyn_steps, a.dt)
                held = float(F.mse_loss(pt, yt))
                p_ = float(F.mse_loss(xt, yt))
                r_eff = P.effective_rank(st[1][:, ::max(dyn.n // 512, 1)].float())
            skill = 1 - held / p_
            log["steps"].append({"step": step, "train": float(loss.detach()),
                                 "held": held, "persistence": p_,
                                 "skill_vs_persistence": skill,
                                 "skill_vs_zero": 1 - held / zero_mse,
                                 "r_eff": r_eff, "sec": round(time.time() - t0, 1)})
            flag = ""
            if skill > best:
                best = skill
                flag = "  <- best"
                os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step,
                            "skill_vs_persistence": skill, "config": vars(a)}, a.ckpt)
            mark = "  BEATS PERSISTENCE" if skill > 0 else ""
            print(f"{step:6d}  train {float(loss):.5f}  held {held:.5f}  "
                  f"persist {p_:.5f}  SKILL {skill:+.4f}  r_eff {r_eff:5.2f}  "
                  f"{time.time()-t0:5.0f}s{flag}{mark}", flush=True)

    log["best_skill"] = best
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"best skill against persistence: {best:+.4f}", flush=True)


if __name__ == "__main__":
    main()
