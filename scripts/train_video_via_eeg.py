"""video continuation through the sensor projection, against two explicit baselines.

the materialisation the programme has been missing: the neural readout is not a
second head beside the video task, it is the ONLY path the video task has.  the
decoder never sees the cortical state -- it sees `n_sensors` channels through a
rank-limited lead field, sampled as the dynamics evolve, and has to continue the
film from that.

the reason this matters is m-multi.  there the MEG head sat at held-out skill
-0.003 for 29,000 steps while the video branch trained perfectly well beside it,
because nothing made the neural prediction necessary for anything.  here it is
necessary by construction.

**three arms, and only the comparison means anything:**

  direct    the ordinary VideoLoop, cortical state read with no bottleneck.
            reached recon 0.011 at 40k steps.  the upper bound.
  learned   this loop, learned lead field.
  random    this loop, lead field FROZEN at its random initialisation.

the third arm is the one that keeps the result honest.  a good `learned` number
on its own shows only that 64 x 8 floats are enough to carry a 64x64x3 frame --
a fact about the width of the bottleneck, not about the readout.  the gap between
`learned` and `random` is what the lead field is worth.

what this does NOT claim: that the projection is EEG.  no corpus pairs this film
with a recording, so nothing supervises those channels against measured data.
the claim under test is narrower and still worth testing -- that what a sensor
array can observe of the cortex suffices to continue the stimulus.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--dyn-steps", type=int, default=8)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--sensors", type=int, default=64)
    ap.add_argument("--sensor-times", type=int, default=8)
    ap.add_argument("--lead-rank", type=int, default=32)
    ap.add_argument("--control", choices=("learned", "random"), default="learned")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--frames", default="data/derived/koyaanisqatsi-full/frames_64x64.npy")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt", default="ckpt/video_via_eeg.pt")
    ap.add_argument("--out", default="out/video_via_eeg.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    fr = np.load(a.frames, mmap_mode="r")
    n = len(fr)
    ntr = int(n * 0.8)
    GAP = 250                       # frames; the film is continuous and adjacent
                                    # frames are near-duplicates across a split
    print(f"{n:,} frames | train 0-{ntr:,} | test {ntr+GAP:,}-{n:,} | "
          f"horizon {a.horizon} | bottleneck {a.sensors}x{a.sensor_times} = "
          f"{a.sensors*a.sensor_times} floats vs {64*64*3:,} in a frame | "
          f"arm={a.control}", flush=True)

    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
    model = P.VideoViaEEGLoop(dyn, n_sensors=a.sensors, n_times=a.sensor_times,
                              lead_rank=a.lead_rank, control=a.control).to(dev)
    tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params {tot:,}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=1e-4)

    def batch(lo, hi, m):
        i = np.random.randint(lo, hi - a.horizon, m)
        x = torch.from_numpy(np.ascontiguousarray(fr[i])).to(dev)
        y = torch.from_numpy(np.ascontiguousarray(fr[i + a.horizon])).to(dev)
        f = lambda t: (t.permute(0, 3, 1, 2).float() / 127.5) - 1
        return f(x), f(y)

    # the baselines a recon number has to be read against.  persistence is the one
    # that matters: predicting frame t for frame t+H is free, and any model that
    # does not beat it has learned nothing about motion.
    with torch.no_grad():
        xb, yb = batch(ntr + GAP, n, 256)
        zero_mse = float((yb ** 2).mean())
        persist_mse = float(F.mse_loss(xb, yb))
    print(f"baselines on held-out: zero {zero_mse:.5f}  persistence {persist_mse:.5f}",
          flush=True)

    log = {"config": vars(a), "n_params": tot, "zero_mse": zero_mse,
           "persistence_mse": persist_mse, "steps": []}
    best, t0 = 1e9, time.time()
    for step in range(a.steps + 1):
        x, y = batch(0, ntr, a.batch)
        pred, sensors, s = model(x, a.dyn_steps, a.dt)
        loss = F.mse_loss(pred, y) + 1e-1 * P.viability_penalty(s[0])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                        if p.requires_grad], 1.0)
        opt.step()

        if step % a.eval_every == 0:
            with torch.no_grad():
                xt, yt = batch(ntr + GAP, n, 128)
                pt, st_, _ = model(xt, a.dyn_steps, a.dt)
                held = float(F.mse_loss(pt, yt))
                r_eff = P.effective_rank(st_.flatten(1).float())
            skill_p = 1 - held / persist_mse
            skill_z = 1 - held / zero_mse
            log["steps"].append({"step": step, "train": float(loss.detach()),
                                 "held": held, "skill_vs_persistence": skill_p,
                                 "skill_vs_zero": skill_z, "sensor_r_eff": r_eff,
                                 "sec": round(time.time() - t0, 1)})
            flag = ""
            if held < best:
                best = held
                flag = "  <- best"
                os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step,
                            "held": held, "config": vars(a)}, a.ckpt)
            print(f"{step:6d}  train {float(loss):.5f}  held {held:.5f}  "
                  f"SKILL vs persistence {skill_p:+.4f}  vs zero {skill_z:+.4f}  "
                  f"sensor r_eff {r_eff:5.2f}  {time.time()-t0:5.0f}s{flag}",
                  flush=True)

    log["best_held"] = best
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"best held-out {best:.5f}  = skill {1-best/persist_mse:+.4f} against "
          f"persistence", flush=True)


if __name__ == "__main__":
    main()
