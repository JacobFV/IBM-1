"""at what horizon does continuation stop being trivial?

the contrastive run opened with persistence at 83.79% -- frame t's own embedding
retrieves frame t+8 among 64 in-film negatives five times out of six, before any
training.  horizon 8 at 25 fps is 0.32 s, and over that interval a film barely
changes, so the task was nearly saturated by the baseline.

H was inherited from the original video loop and never checked.  this measures
what it should be: how distinguishable frame t+H is from frame t, as H grows.
no training involved -- pixel distance and a random projection both answer it,
and a random projection is the right proxy because it is what an untrained
encoder is.
"""
import glob, numpy as np, torch, torch.nn.functional as F

films = sorted(glob.glob("data/derived/pd-film/*_frames.npy"))[-2:]
TE = [np.load(f, mmap_mode="r") for f in films]
dev = "cuda" if torch.cuda.is_available() else "cpu"
g = torch.Generator(device="cpu").manual_seed(0)
W = torch.randn(64*64*3, 128, generator=g).to(dev) / (64*64*3) ** 0.5
B = 64

print(f"{'H':>5s} {'seconds':>8s} {'pixel MSE':>10s} {'persist top-1':>14s}")
for H in (8, 25, 50, 100, 200, 400):
    accs, mses = [], []
    rng = np.random.default_rng(3)
    for _ in range(12):
        v = TE[rng.integers(len(TE))]
        if len(v) <= H + 2: continue
        i = rng.integers(0, len(v) - H - 1, B)
        f = lambda idx: ((torch.from_numpy(np.stack([np.asarray(v[j]) for j in idx]))
                          .to(dev).float() / 127.5) - 1.0).flatten(1)
        x, y = f(i), f(i + H)
        mses.append(float(((x - y) ** 2).mean()))
        zx = F.normalize(x @ W, dim=-1); zy = F.normalize(y @ W, dim=-1)
        lb = torch.arange(len(i), device=dev)
        accs.append(float(((zx @ zy.T).argmax(1) == lb).float().mean()))
    print(f"{H:5d} {H/25:8.2f} {np.mean(mses):10.4f} {100*np.mean(accs):13.1f}%")
print(f"\nchance = {100/B:.2f}%.  a horizon where persistence is near 100% cannot")
print("measure prediction; the model would only have to match the baseline.")
