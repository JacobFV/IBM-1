"""does wiring learned by next-frame video training help the EEG task?

the architecture's central bet is that most of the model is shared, so a corpus
that constrains the substrate should constrain every materialisation.  next-frame
video is the volume term -- 40 hours of it, needing no neural pairing -- so if it
shapes the cortical wiring usefully, that is the cheapest source of structure the
programme has.

`dyn.embed` IS the wiring: w_ij = M[parcel] x exp(-d/l) x sigma(<e_i,e_j>), and
only that third factor trains.  both runs move it essentially all the way --
cosine with initialisation is +0.001 -- so magnitude says they reshaped it.
magnitude has misled this project twice (ledger row 6: a weight 4.4x baseline
that cost -0.06% to sever), so this asks by TRANSFER instead.

the visual-contrastive model is evaluated on the designated THINGS-EEG2 test set
with its association kernel replaced four ways:

  own        the wiring that model learned.  the upper bound.
  video      the wiring the next-frame run learned, transplanted.
  random     a fresh random kernel -- the `frozen` arm, already measured at 28.0%.
  zeroed     no association at all.

**if `video` beats `random`, next-frame training produces transferable cortical
structure** and the volume term is doing what the architecture claims. if it does
not, the wiring each task learns is task-specific and the shared-substrate bet is
not being paid on this pair.
"""
from __future__ import annotations

import argparse, importlib.util, json, os
import numpy as np
import torch

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)

D = "data/derived/things-paired"
ONSET, KEEP = 20, 50


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eeg-ckpt", default="ckpt/visual_contrastive_v2.pt")
    ap.add_argument("--video-ckpt", default="ckpt/video_fixedread.pt")
    ap.add_argument("--out", default="out/transfer_wiring.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(a.eeg_ckpt, map_location="cpu", weights_only=False)
    cfg, sd = d["config"], d["model"]
    vsd = torch.load(a.video_ckpt, map_location="cpu", weights_only=False)["model"]

    if sd["dyn.embed"].shape != vsd["dyn.embed"].shape:
        print(f"shape mismatch {sd['dyn.embed'].shape} vs {vsd['dyn.embed'].shape}; "
              "the two runs must share --sites and --embed to transplant"); return

    tr = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    ntr = int(min(len(tr), len(np.load(f"{D}/images_training.npy", mmap_mode="r"))) * 0.8)
    samp = np.asarray(tr[:ntr:7]).astype(np.float32)
    med = np.median(samp, 0)
    iqr = ((np.percentile(samp, 75, 0) - np.percentile(samp, 25, 0)) / 1.349).clip(1e-9)
    imgs = np.load(f"{D}/images_test.npy")
    ev = np.load(f"{D}/evoked_test_groupmean.npy")
    n = len(imgs)
    y = np.clip((ev.astype(np.float32) - med) / iqr, -6, 6)[..., ONSET:ONSET+KEEP:cfg["decimate"]]

    n_sites, embed = sd["dyn.embed"].shape
    k = sd["dyn.idx"].shape[1]
    dyn = P.CorticalDynamics(n_sites, embed, k, dev, long_range=cfg["long_range"]).to(dev)
    model = P.VisualContrastiveLoop(dyn, n_sensors=ev.shape[1], n_times=y.shape[-1]).to(dev)
    model.load_state_dict(sd)
    model.eval()

    x = torch.from_numpy(np.ascontiguousarray(imgs)).to(dev)
    x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1
    yt = torch.from_numpy(y).to(dev)
    lbl = torch.arange(n, device=dev)
    own = dyn.embed.data.clone()
    vid = vsd["dyn.embed"].to(dev)
    torch.manual_seed(0)
    rnd = (torch.randn_like(dyn.embed) * 0.02).to(dev)

    print(f"designated test set: {n} images, chance {100.0/n:.2f}%")
    print(f"transplanting `dyn.embed` only -- every other weight is the EEG model's\n")

    @torch.no_grad()
    def run(name, w):
        dyn.embed.data.copy_(w)
        z, _ = model.embed_image(x, substeps=cfg["substeps"], dt=cfg["dt"],
                                 n_steps=cfg.get("n_steps", 4))
        sim = z @ model.embed_eeg(yt).T
        rank = (sim > sim.gather(1, lbl[:, None])).sum(1)
        return {"top1": float((rank == 0).float().mean()),
                "top5": float((rank < 5).float().mean())}

    res = {}
    for name, w in (("own", own), ("video", vid), ("random", rnd),
                    ("zeroed", torch.zeros_like(own))):
        res[name] = run(name, w)
        print(f"  {name:8s} top-1 {100*res[name]['top1']:5.2f}%  "
              f"top-5 {100*res[name]['top5']:5.2f}%", flush=True)
    dyn.embed.data.copy_(own)

    v, r = res["video"]["top1"], res["random"]["top1"]
    print(f"\n  video vs random: {100*(v-r):+.2f} points")
    verdict = ("next-frame training produces TRANSFERABLE cortical structure"
               if v > r + 0.03 else
               "the video wiring is no better than random here -- what each task "
               "learns is task-specific, and the shared-substrate bet is not paid "
               "on this pair")
    print(f"  -> {verdict}")
    res["verdict"] = verdict
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
