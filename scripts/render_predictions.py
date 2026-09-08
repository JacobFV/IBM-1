"""render predicted vs true continuation, five tracks stacked, as a clip.

the model predicts frame t+H from frame t.  each row is a different starting
point in the film, so one clip shows five independent continuations at once and
you can see whether the failures are shared (a property of the model) or
per-scene (a property of the content).

left column is truth, right is prediction, and they are the same scale and
contrast -- no per-panel normalization, because normalizing each panel to its own
range is how a washed-out prediction is made to look sharp.
"""
from __future__ import annotations
import argparse, importlib.util, numpy as np, torch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--audio-frames", default="")
ap.add_argument("--rows", type=int, default=5)
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--out", default="out/predictions.gif")
a = ap.parse_args()

sp = importlib.util.spec_from_file_location("pt", "scripts/pretrain_video_loop.py")
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
sd = d.get("model") or d.get("av")
# the publish path used to rewrite the checkpoint without `config`, so every
# checkpoint this loop published before 36a66a2 -- all 68 on HuggingFace -- lacks
# it.  the field is not lost: `sidecar()` writes the same dict to the .json
# beside the weights.  read that when the checkpoint does not carry its own,
# rather than silently defaulting `modality` to "av" and building the wrong model.
cfg = d.get("config") or {}
if not cfg:
    import json
    side = a.ckpt[:-3] + ".json" if a.ckpt.endswith(".pt") else ""
    if side and __import__("os").path.exists(side):
        cfg = json.load(open(side)).get("config", {})
        print(f"config recovered from {side}", flush=True)
    else:
        print("WARNING: no config in the checkpoint and no sidecar beside it; "
              "falling back to defaults, which may build the wrong model",
              flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
n_sites, embed = sd["dyn.embed"].shape; k = sd["dyn.idx"].shape[1]
H = cfg.get("horizon", 8); ds = cfg.get("dyn_steps", 8); dt = cfg.get("dt", 5e-3)

frames = np.load(a.frames, mmap_mode="r")
coch = np.load(a.audio_frames, mmap_mode="r") if a.audio_frames else None
dyn = P.CorticalDynamics(n_sites, embed, k, dev, long_range=cfg.get("long_range", .25))
mod = cfg.get("modality", "av")
model = (P.AudioVisualLoop(dyn, n_bands=coch.shape[-1]) if mod == "av"
         else P.VideoLoop(dyn)).to(dev)
model.load_state_dict(sd); model.eval()
print(f"{a.ckpt}: {mod}, {n_sites:,} sites, horizon {H}", flush=True)

n_out = int(a.seconds * a.fps)
# five starting points spread across the film, each advancing one frame per tick
lim = min(len(frames), len(coch) if coch is not None else len(frames)) - H - n_out - 2
starts = np.linspace(8, lim, a.rows).astype(int)

def to_u8(x):
    return np.clip((x + 1.0) * 127.5, 0, 255).astype(np.uint8)

panels = []
with torch.no_grad():
    for t in range(n_out):
        idx = starts + t
        xv = torch.from_numpy(np.ascontiguousarray(frames[idx])).to(dev)
        xv = (xv.permute(0, 3, 1, 2).float() / 127.5) - 1.0
        truth = np.ascontiguousarray(frames[idx + H])
        if mod == "av":
            xa = torch.from_numpy(np.stack([coch[j-8:j] for j in idx])).float().to(dev)
            pv, _, _ = model(xv, xa, ds, dt)
        else:
            pv, _ = model(xv, ds, dt)
        pred = to_u8(pv.permute(0, 2, 3, 1).cpu().numpy())
        # one column of truth, one of prediction, a 2 px rule between
        rule = np.full((truth.shape[1], 2, 3), 90, np.uint8)
        rows = [np.concatenate([truth[i], rule, pred[i]], axis=1) for i in range(a.rows)]
        gap = np.full((2, rows[0].shape[1], 3), 90, np.uint8)
        stack = rows[0]
        for r in rows[1:]:
            stack = np.concatenate([stack, gap, r], axis=0)
        panels.append(stack)
        if t % 50 == 0: print(f"  {t}/{n_out}", flush=True)

import imageio.v2 as imageio
import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
imageio.mimsave(a.out, panels, fps=a.fps, loop=0)
print(f"wrote {a.out}  {panels[0].shape}  {len(panels)} frames "
      f"({len(panels)/a.fps:.1f}s)  left=truth right=prediction(t+{H})", flush=True)
