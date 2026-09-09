"""does the video loop beat persistence, or has it been reproducing frame t?

the clips look plausible and the recon loss fell 50x, but a recon loss is not a
result -- CLAUDE.md's first rule.  the baseline that matters for continuation is
PERSISTENCE: predicting frame t for frame t+H costs nothing and is very strong on
a film at 25 fps.  a model that does not beat it has learned no motion, and a
clip from it would look right for the wrong reason.
"""
import importlib.util, numpy as np, torch, torch.nn.functional as F
sp = importlib.util.spec_from_file_location("p", "scripts/pretrain_video_loop.py")
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

d = torch.load("ckpt/video_v6.pt", map_location="cpu", weights_only=False)
cfg, sd = d["config"], d["model"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
n_sites, embed = sd["dyn.embed"].shape; k = sd["dyn.idx"].shape[1]
H = cfg["horizon"]; ds = cfg["dyn_steps"]; dt = cfg["dt"]
dyn = P.CorticalDynamics(n_sites, embed, k, dev, long_range=cfg["long_range"]).to(dev)
rs = sd["from_cortex.weight"].shape[1] if "from_cortex.weight" in sd else 4096
model = P.VideoLoop(dyn, read_sites=rs).to(dev); model.load_state_dict(sd); model.eval()

fr = np.load(cfg["frames"], mmap_mode="r"); n = len(fr); ntr = int(n*0.8); GAP=250
rng = np.random.default_rng(0)
mse_m = mse_p = mse_z = 0.0; nb = 0
with torch.no_grad():
    for _ in range(20):
        i = rng.integers(ntr+GAP, n-H, 64)
        x = torch.from_numpy(np.ascontiguousarray(fr[i])).to(dev)
        y = torch.from_numpy(np.ascontiguousarray(fr[i+H])).to(dev)
        f = lambda t: (t.permute(0,3,1,2).float()/127.5)-1
        x, y = f(x), f(y)
        pred, _ = model(x, ds, dt)
        mse_m += float(F.mse_loss(pred, y)); mse_p += float(F.mse_loss(x, y))
        mse_z += float((y**2).mean()); nb += 1
m, p_, z = mse_m/nb, mse_p/nb, mse_z/nb
print(f"held-out frames, horizon {H}, {nb*64} samples")
print(f"  model       MSE {m:.5f}")
print(f"  persistence MSE {p_:.5f}   skill {1-m/p_:+.4f}")
print(f"  zero        MSE {z:.5f}   skill {1-m/z:+.4f}")
print()
print("  -> " + ("BEATS persistence" if m < p_ else
      "WORSE THAN PERSISTENCE: the clip looks right because frame t+8 resembles "
      "frame t, not because the model predicts motion"))
