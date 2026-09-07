"""evaluate the paired head on a CONTIGUOUS held-out block, against baselines.

training reported skill +0.997 on random indices.  that is not a plausible number
for stimulus-to-MEG encoding -- the literature gets 5-15% of variance -- so either
the model found something real or it is reading the answer.  a contiguous split
with a guard band is what tells them apart.
"""
import argparse, numpy as np, torch, torch.nn.functional as F, importlib.util
from ibm.evaluate import time_splits, baseline_scores, skill

sp = importlib.util.spec_from_file_location("pt","scripts/pretrain_video_loop.py")
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--stim", required=True); ap.add_argument("--neural", required=True)
ap.add_argument("--batches", type=int, default=40)
a = ap.parse_args()

d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
sd = d["model"]
if "config" not in d:
    raise SystemExit("checkpoint has no config: refusing to evaluate against "
                     "hard-coded defaults, which would silently use dyn_steps=6 "
                     "where the trainer's own default is 8")
cfg = d["config"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
n_sites, embed = sd["dyn.embed"].shape; k = sd["dyn.idx"].shape[1]
stim = np.load(a.stim, mmap_mode="r"); neur = np.load(a.neural, mmap_mode="r")
sc = np.load(a.neural.replace("meg_250hz","meg_scale"))

dyn = P.CorticalDynamics(n_sites, embed, k, dev, long_range=cfg.get("long_range",.25))
m = P.PairedNeuralLoop(dyn, n_bands=stim.shape[-1], n_sensors=neur.shape[-1]).to(dev)
m.load_state_dict(sd); m.eval()
ctx, bs = 125, cfg.get("batch", 16)
sp_ = time_splits(min(len(stim), len(neur)), gap=2500)   # 10 s guard at 250 Hz
print(f"{n_sites:,} sites | splits: " +
      ", ".join(f"{k2} {v.lo:,}-{v.hi:,}" for k2, v in sp_.items()))

@torch.no_grad()
def run(split):
    tot=n=0.0; ys=[]; ps=[]
    r = np.random.default_rng(0)
    for _ in range(a.batches):
        i = r.integers(split.lo+ctx, split.hi-2, size=bs)
        x = torch.from_numpy(np.stack([stim[j-ctx:j] for j in i])).float().to(dev)
        yn = np.clip((np.ascontiguousarray(neur[i])-sc[0])/sc[1], -6, 6).astype(np.float32)
        y = torch.from_numpy(yn).to(dev)
        p,_ = m(x, cfg.get("dyn_steps",6), cfg.get("dt",5e-3))
        tot += float(F.mse_loss(p,y)); n += 1
        ys.append(yn); ps.append(p.cpu().numpy())
    Y=np.concatenate(ys); b=baseline_scores(Y)
    return tot/n, b

for name in ("train","test"):
    mse, b = run(sp_[name])
    s = skill(mse, b)
    print(f"  {name:5s} mse {mse:8.5f}   baseline(zero) {b['zero']:7.4f}   "
          f"skill_vs_zero {s['skill_vs_zero']:+7.3f}")
