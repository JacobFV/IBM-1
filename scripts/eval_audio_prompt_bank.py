"""re-measure `ckpt/audio_contrastive_v1.pt` on held-out pools, and on the exact
bank the prompt bar serves.

the number stored in the checkpoint is the training loop's BEST eval, and a best
over 40 evaluations is a selection over lucky draws even when each evaluation is
itself an 8-pool average (docs/LOG.md row 9).  the prompt endpoint quotes an
accuracy at the user, so it quotes one measured here at the checkpoint, once,
against fixed seeds, rather than the training loop's maximum.

it also measures the ONE pool the endpoint actually serves as its retrieval bank,
separately and labelled as a single pool -- a single pool of 200 has sd ~2.8
points, so that figure is reported beside the 8-pool mean and never instead of it.

the bar is the measured dynamics-free control on this corpus and split: 7.12% +/-
1.71 at a 1 s window (4.62% +/- 0.99 at 200 ms).  this head is compared against it
here rather than against chance alone.
"""
from __future__ import annotations

import importlib.util, json, os
import numpy as np, torch

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

D = "data/derived/libribrain-paired"
CKPT = "ckpt/audio_contrastive_v1.pt"
GAP = 2500                    # the builder's 10 s guard band
SERVED_SEED = 20260909        # the pool the endpoint serves
POOL_SEEDS = [20260909 + i for i in range(8)]
CONTROL = {"top1": 0.0712, "sd": 0.0171, "window": "1 s",
           "what": "two convnets with no dynamics in the path, same corpus and "
                   "split, 8 held-out pools of 200"}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg, sd = d["config"], d["model"]
    stim = np.load(f"{D}/cochleagram_250hz.npy", mmap_mode="r")
    neur = np.load(f"{D}/meg_250hz.npy", mmap_mode="r")
    n = min(len(stim), len(neur)); ntr = int(n * 0.8); C = neur.shape[1]
    win, ctx, pool = cfg["win"], cfg["ctx"], cfg["pool"]

    n_sites, embed = sd["dyn.embed"].shape
    dyn = P.CorticalDynamics(n_sites, embed, sd["dyn.idx"].shape[1], dev,
                             long_range=cfg["long_range"]).to(dev)
    model = P.AudioContrastiveLoop(dyn, n_sensors=C, n_bands=stim.shape[1],
                                   stim_len=ctx, n_times=win).to(dev)
    model.load_state_dict(sd); model.eval()

    def coch(i):
        return np.stack([np.asarray(stim[q - ctx:q]) for q in i]).astype(np.float32)

    def meg(i):
        return np.stack([np.clip(np.asarray(neur[q:q + win]).astype(np.float32),
                                 -6, 6).T for q in i])

    @torch.no_grad()
    def score(j):
        x = torch.from_numpy(coch(j)).to(dev).permute(0, 2, 1)
        y = torch.from_numpy(meg(j)).to(dev)
        z, _ = model.embed_audio(x, substeps=cfg["substeps"], dt=cfg["dt"],
                                n_steps=cfg["n_steps"])
        sim = z @ model.embed_meg(y).T
        lbl = torch.arange(len(j), device=dev)
        srt = sim.sort(1, descending=True).values
        return (float((sim.argmax(1) == lbl).float().mean()),
                float((sim.topk(5, 1).indices == lbl[:, None]).any(1).float().mean()),
                float(srt[:, 0].median()), float((srt[:, 0] - srt[:, 1]).median()))

    lo, hi = ntr + GAP, n - win - 1
    per_pool = []
    for s in POOL_SEEDS:
        j = np.random.default_rng(s).integers(lo, hi, pool)
        t1, t5, msim, mmar = score(j)
        per_pool.append({"seed": int(s), "top1": t1, "top5": t5,
                         "median_top1_cos": msim, "median_margin": mmar})
        print(f"  seed {s}: top-1 {100*t1:5.2f}%  top-5 {100*t5:5.2f}%  "
              f"med cos {msim:+.4f}  med margin {mmar:+.4f}", flush=True)

    a = np.array([p["top1"] for p in per_pool])
    served = next(p for p in per_pool if p["seed"] == SERVED_SEED)
    out = {
        "ckpt": CKPT, "train_step": d["step"],
        "checkpoint_stored_top1": d.get("top1"), "checkpoint_stored_sd": d.get("top1_sd"),
        "checkpoint_stored_note": "the training loop's BEST eval over 41 evaluations; "
                                  "a maximum over draws, not an unbiased estimate",
        "corpus": "data/derived/libribrain-paired (v3, the 2026-09-07 rebuild; "
                  "v1 and v2 were misaligned by up to +/-3.4 s and every result "
                  "measured on them is void)",
        "split": "contiguous, last 20% with a 10 s (2,500 sample) guard band",
        "pool": pool, "chance": 1.0 / pool,
        "window_ms": 1000 * win / 250, "context_frames": ctx,
        "top1_mean_8_pools": float(a.mean()), "top1_sd_8_pools": float(a.std()),
        "top1_x_chance": float(a.mean() * pool),
        "served_bank_seed": SERVED_SEED,
        "served_bank_top1_single_pool": served["top1"],
        "served_bank_top5_single_pool": served["top5"],
        "served_bank_note": "ONE pool.  a single pool of 200 has sd ~2.8 points "
                            "(docs/LOG.md row 9); read the 8-pool mean, not this",
        "per_pool": per_pool,
        "dynamics_free_control": CONTROL,
        "verdict": None,
    }
    m, s_ = a.mean(), a.std()
    gap = m - CONTROL["top1"]
    pooled = float(np.hypot(s_, CONTROL["sd"]))
    out["verdict"] = (
        f"{100*m:.2f}% +/- {100*s_:.2f} against a dynamics-free control at "
        f"{100*CONTROL['top1']:.2f}% +/- {100*CONTROL['sd']:.2f}: a gap of "
        f"{100*gap:+.2f} points, {abs(gap)/pooled:.2f} pooled sd.  the auditory "
        f"head does NOT beat its dynamics-free control, and this endpoint must "
        f"not present an audio result as if it were the visual one.")
    print("\n" + out["verdict"], flush=True)
    os.makedirs("out", exist_ok=True)
    json.dump(out, open("out/audio_prompt_bank.json", "w"), indent=2)
    print("wrote out/audio_prompt_bank.json", flush=True)


if __name__ == "__main__":
    main()
