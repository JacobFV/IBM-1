"""does the video loop predict MOTION, or only exploit that frame t+H resembles t?

every video result in this programme is quoted at horizon 8, and at 25 fps that
is 320 ms -- a lag over which most footage barely changes, so persistence is a
very strong baseline and beating it is close to impossible.  the multifilm run
reaches skill -0.66 there, which reads as "nearly as good as copying" and says
nothing about whether the model has learned motion at all.

the discriminating measurement is the SHAPE of skill against horizon, because the
two hypotheses come apart as the lag grows:

  a model that only reproduces its input   tracks persistence down as the lag
      grows -- both degrade together and skill stays flat and negative, because
      the model IS persistence plus noise.

  a model that has learned scene dynamics   loses more slowly than persistence
      does, so skill RISES with horizon and may cross zero at some lag.  that
      crossing, if it exists, is the first evidence of motion prediction in this
      repo and it is invisible at the single horizon everything is quoted at.

persistence MSE necessarily grows with H, so this is not a fair-baseline trick in
the model's favour: it is the only comparison under which "predicts motion" and
"copies the input" make different predictions.

both are reported in absolute MSE as well as skill, because a rising skill that
comes from persistence collapsing rather than the model holding up is a different
and much weaker claim, and the absolute columns show which it is.

whole FILMS are held out by the multifilm run, so this asks the model to continue
footage it has never seen, not a later reel of a film it memorised.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

import numpy as np
import torch

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/video_multifilm.pt")
    ap.add_argument("--corpus", default="data/derived/pd-film")
    ap.add_argument("--horizons", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-json", default="",
                    help="the run's own json, which NAMES its holdout films")
    ap.add_argument("--out", default="out/video_horizon.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = d["model"] if "model" in d else d
    cfg = d.get("config", {})
    n_sites, embed = sd["dyn.embed"].shape
    k = sd["dyn.idx"].shape[1]
    ds = cfg.get("dyn_steps", 8)
    dt = cfg.get("dt", 5e-3)
    trained_H = cfg.get("horizon", 8)
    # the config records target=None for this run, so a .get() default of
    # "residual" silently selected the DIRECT interpretation.  make the
    # fallback explicit and say which was used.
    target = cfg.get("target") or "residual"

    dyn = P.CorticalDynamics(n_sites, embed, k, dev,
                             long_range=cfg.get("long_range", 0.25)).to(dev)
    # read_sites is a property of the checkpoint, not of today's default -- the
    # v6 weights are (256, 3750) against a current default of 4096.
    rs = sd["from_cortex.weight"].shape[1] if "from_cortex.weight" in sd else 4096
        # a checkpoint without `read_idx` predates its registration as a buffer,
    # and every such checkpoint was trained with the ANTERIOR readout.  the
    # shapes match either way, so guessing wrong is silent and costs a factor
    # of 112 on video_v6.  detect it rather than assume.
    ro = "linspace" if "read_idx" in sd else "anterior"
    print(f"readout convention: {ro}", flush=True)
    model = P.VideoLoop(dyn, read_sites=rs, readout=ro).to(dev)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"{a.ckpt}: step {d.get('step','?')}, {n_sites:,} sites, "
          f"trained at horizon {trained_H}, target '{target}'", flush=True)

    # the SAME held-out films the run held out, taken the same way (the last
    # `holdout_films` of the sorted list) so this is not a fresh split that might
    # include footage the model trained on.
    # TAKE THE HOLDOUT FROM THE RUN THAT DEFINED IT, never from a reconstruction.
    #
    # the corpus GROWS between runs -- this checkpoint trained on 5 films and the
    # directory now holds 30 -- so `sorted(glob(...))[-2:]` returns two films that
    # run never designated, and may return films it trained on.  taking that slice
    # gave persistence 0.0413 against the run's own logged 0.0440, and the model
    # read 0.88 against a logged 0.102: a completely different measurement wearing
    # the right shape.  the run wrote its holdout by name; read it.
    side = a.run_json or os.path.splitext(a.ckpt.replace("ckpt/", "out/"))[0] + ".json"
    held = None
    if os.path.exists(side):
        meta = json.load(open(side))
        names = meta.get("holdout_films") or []
        if names:
            held = [n if os.path.exists(n) else os.path.join(a.corpus,
                    os.path.basename(n)) for n in names]
            print(f"holdout read from {side} ({len(held)} films named by the run)",
                  flush=True)
            if meta.get("persistence_mse"):
                print(f"  the run logged persistence {meta['persistence_mse']:.5f} "
                      f"-- H=8 below must match it or the split is wrong",
                      flush=True)
    if held is None:
        raise SystemExit(
            f"{side} does not name the holdout films. refusing to guess a split "
            f"from the current directory: the corpus grows between runs, so a "
            f"reconstructed slice measures a different experiment.")
    missing = [h for h in held if not os.path.exists(h)]
    if missing:
        raise SystemExit(f"holdout films named by the run are missing: {missing}")
    films = sorted(glob.glob(f"{a.corpus}/*_frames.npy"))
    print(f"held-out films ({len(held)} of {len(films)}):", flush=True)
    for f in held:
        print(f"  {os.path.basename(f)}", flush=True)

    arrs = [np.load(f, mmap_mode="r") for f in held]
    horizons = [int(x) for x in a.horizons.split(",")]
    maxH = max(horizons)

    @torch.no_grad()
    def one(H: int):
        rng = np.random.default_rng(a.seed)          # SAME frames at every H
        mse_m, mse_p = [], []
        for _ in range(a.batches):
            arr = arrs[rng.integers(len(arrs))]
            if len(arr) <= maxH + 2:
                continue
            i = rng.integers(0, len(arr) - maxH - 2, a.batch)
            x = np.stack([arr[j] for j in i]).astype(np.float32) / 127.5 - 1
            y = np.stack([arr[j + H] for j in i]).astype(np.float32) / 127.5 - 1
            xt = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
            yt = torch.from_numpy(y).permute(0, 3, 1, 2).to(dev)
            out, _ = model(xt, n_steps=ds, dt=dt)   # (prediction, state)
            pred = xt + out if target == "residual" else out
            mse_m.append(float(((pred - yt) ** 2).mean()))
            mse_p.append(float(((xt - yt) ** 2).mean()))     # persistence
        return float(np.mean(mse_m)), float(np.mean(mse_p))

    print(f"\n  {'H':>5s} {'lag ms':>8s} {'model MSE':>11s} {'persist MSE':>12s} "
          f"{'skill':>9s}", flush=True)
    res, crossed = {}, None
    for H in horizons:
        m, p_ = one(H)
        s = 1 - m / p_ if p_ > 0 else float("nan")
        res[H] = {"model_mse": m, "persistence_mse": p_, "skill": s}
        flag = "  <- trained here" if H == trained_H else ""
        if s > 0 and crossed is None:
            crossed = H
            flag += "  <- BEATS PERSISTENCE"
        print(f"  {H:5d} {1000*H/25:8.0f} {m:11.5f} {p_:12.5f} {s:+9.4f}{flag}",
              flush=True)

    ss = [res[H]["skill"] for H in horizons]
    rising = ss[-1] > ss[0]
    print(f"\n  skill at H={horizons[0]}: {ss[0]:+.4f}   "
          f"at H={horizons[-1]}: {ss[-1]:+.4f}   "
          f"{'RISING' if rising else 'flat or falling'} with horizon", flush=True)
    if crossed:
        verdict = (f"PREDICTS MOTION beyond persistence from H={crossed} "
                   f"({1000*crossed/25:.0f} ms). the model degrades more slowly "
                   f"than copying does, which copying cannot explain.")
    elif rising:
        verdict = ("degrades more slowly than persistence but never overtakes it. "
                   "consistent with having learned SOME scene structure; not "
                   "sufficient to claim motion prediction.")
    else:
        verdict = ("tracks persistence down at every lag -- the model is "
                   "persistence plus noise, and the horizon-8 number was not "
                   "hiding a motion model.")
    print(f"  -> {verdict}", flush=True)

    res_out = {"ckpt": a.ckpt, "step": d.get("step"), "trained_horizon": trained_H,
               "target": target, "held_out_films": [os.path.basename(f) for f in held],
               "by_horizon": res, "first_positive_horizon": crossed,
               "verdict": verdict}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res_out, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
