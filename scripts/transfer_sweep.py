"""which video runs produce transferable cortical wiring, and does it track their
own task performance?

transplanting the fixed-readout video kernel into the EEG model recovered 92% of
what the EEG-trained kernel is worth, from a corpus with no recordings in it.
that raises the question this sweeps: is wiring quality a function of how well
the video run did on VIDEO, or of something else -- data volume, the readout, the
objective?

the runs on disk differ in exactly the ways that matter:

  video_v6            11 min, one film, anterior readout, 40k steps
  video_multifilm     37.7 h, 28 films, anterior readout, 20k steps
  video_fixedread     37.7 h, 28 films, WHOLE-SHEET readout, best at 13k
  video_residual      residual objective, anterior readout -- DEGENERATE, emitted
                      1.9% of the true residual with no direction
  video_contrastive*  contrastive objective, anterior readout

the degenerate one is the interesting case.  it failed its own task completely.
if its wiring still transfers, wiring quality is decoupled from task performance
and the volume term does not need to succeed to be worth running.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, os
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
    ap.add_argument("--out", default="out/transfer_sweep.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(a.eeg_ckpt, map_location="cpu", weights_only=False)
    cfg, sd = d["config"], d["model"]

    tr = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    ntr = int(min(len(tr), len(np.load(f"{D}/images_training.npy", mmap_mode="r"))) * 0.8)
    s_ = np.asarray(tr[:ntr:7]).astype(np.float32)
    med = np.median(s_, 0)
    iqr = ((np.percentile(s_, 75, 0) - np.percentile(s_, 25, 0)) / 1.349).clip(1e-9)
    imgs = np.load(f"{D}/images_test.npy")
    ev = np.load(f"{D}/evoked_test_groupmean.npy")
    y = np.clip((ev.astype(np.float32) - med) / iqr, -6, 6)[..., ONSET:ONSET+KEEP:cfg["decimate"]]

    ns, emb = sd["dyn.embed"].shape
    k = sd["dyn.idx"].shape[1]
    dyn = P.CorticalDynamics(ns, emb, k, dev, long_range=cfg["long_range"]).to(dev)
    model = P.VisualContrastiveLoop(dyn, n_sensors=ev.shape[1],
                                    n_times=y.shape[-1]).to(dev)
    model.load_state_dict(sd)
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(imgs)).to(dev)
    x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1
    yt = torch.from_numpy(y).to(dev)
    lbl = torch.arange(len(imgs), device=dev)
    own = dyn.embed.data.clone()
    torch.manual_seed(0)
    rnd = (torch.randn_like(dyn.embed) * 0.02).to(dev)

    @torch.no_grad()
    def score(w):
        dyn.embed.data.copy_(w)
        z, _ = model.embed_image(x, substeps=cfg["substeps"], dt=cfg["dt"],
                                 n_steps=cfg.get("n_steps", 4))
        sim = z @ model.embed_eeg(yt).T
        return float((sim.argmax(1) == lbl).float().mean())

    base = score(rnd)
    top = score(own)
    print(f"designated test set, chance {100.0/len(imgs):.2f}%")
    print(f"  EEG-trained kernel (ceiling) {100*top:5.2f}%")
    print(f"  random kernel (floor)        {100*base:5.2f}%")
    print(f"\n{'video run':26s} {'top-1':>7s} {'recovered':>10s}")
    res = {"ceiling": top, "floor": base, "runs": {}}
    for f in sorted(glob.glob(os.environ.get("SWEEP_GLOB","ckpt/video*.pt"))):
        try:
            w = torch.load(f, map_location="cpu", weights_only=False)["model"]["dyn.embed"]
        except Exception as e:
            print(f"{os.path.basename(f):26s}  skip ({type(e).__name__})"); continue
        if w.shape[1] != own.shape[1]:
            print(f"{os.path.basename(f):26s}  skip (embed {w.shape[1]})"); continue
        if w.shape[0] != own.shape[0]:
            # a 150k kernel is not incomparable, only differently sampled.  carry
            # it to the evaluation resolution the same way fusion does, so a
            # higher-resolution run can be measured and weighted rather than
            # silently dropped -- which is what happened to the 150k run.
            fu = importlib.util.spec_from_file_location(
                "fuse", os.path.join(os.path.dirname(__file__), "fuse_implicit.py"))
            FU = importlib.util.module_from_spec(fu); fu.loader.exec_module(FU)
            w = FU.resample(w.float(), w.shape[0], own.shape[0])
        v = score(w.to(dev))
        rec = (v - base) / max(top - base, 1e-9)
        res["runs"][os.path.basename(f)] = {"top1": v, "recovered": rec}
        print(f"{os.path.basename(f):26s} {100*v:6.2f}% {100*rec:9.1f}%", flush=True)
    dyn.embed.data.copy_(own)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
