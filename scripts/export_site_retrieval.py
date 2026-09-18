"""export real THINGS-EEG2 retrieval trials to the site.

the site claims 63.5% top-1 over the designated test set.  a panel that showed the
stimulus beside "the predicted image" would, on a hit, show the same photograph twice,
and a reader could not tell whether the model picked it out of 200 or was handed it.
so this exports the RANKING: the stimulus, its measured evoked response, and the model's
top five out of the whole 200-image pool, with where the true image actually landed.

the pipeline is lifted from scripts/eval_things_test.py deliberately -- same checkpoint,
same training-derived normalisation, same port and sheet read from the checkpoint's own
config.  it REFUSES to write unless it reproduces that script's published top-1, because
a panel drawn from a subtly different pipeline would be illustrating a number the
programme never measured.

run it on the CPU -- it is 200 images through one small model, and the GPU is usually
someone's training run, on a machine where GPU memory is system memory:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/export_site_retrieval.py
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("ptrain", os.path.join(HERE, "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

D = "data/derived/things-paired"
ONSET, KEEP = 20, 50
ROOT = Path(__file__).resolve().parents[1]
OUTJS = ROOT / "site" / "data" / "retrieval.js"
OUTIMG = ROOT / "site" / "media" / "retrieval"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="ckpt/visual_contrastive_v2.pt")
    ap.add_argument("--expect-top1", type=float, default=0.635,
                    help="the published figure this must reproduce")
    ap.add_argument("--thumb", type=int, default=160,
                    help="thumbnail edge in px; drawn at <=74 css px, so 160 covers 2x screens")
    ap.add_argument("--n-trials", type=int, default=23,
                    help="how many test trials to export, taken in dataset order")
    a = ap.parse_args()

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg, sd = d["config"], d["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # normalisation from TRAINING only -- fitting it on the test set leaks it
    tr = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    ntr = int(min(len(tr), len(np.load(f"{D}/images_training.npy", mmap_mode="r"))) * 0.8)
    samp = np.asarray(tr[:ntr:7]).astype(np.float32)
    med = np.median(samp, 0)
    iqr = ((np.percentile(samp, 75, 0) - np.percentile(samp, 25, 0)) / 1.349).clip(1e-9)

    imgs = np.load(f"{D}/images_test.npy")
    ev = np.load(f"{D}/evoked_test_groupmean.npy")
    paths = np.load(f"{D}/image_paths_test.npy")
    n = len(imgs)
    dec = cfg["decimate"]
    y = np.clip((ev.astype(np.float32) - med) / iqr, -6, 6)[..., ONSET:ONSET + KEEP:dec]
    T, C = y.shape[-1], ev.shape[1]

    dyn = P.dynamics_from_state_dict(
        sd, dev, long_range=cfg["long_range"], geometry=cfg.get("geometry", "sphere"),
        long_topology=cfg.get("long_topology", "random")).to(dev)
    model = P.VisualContrastiveLoop(dyn, n_sensors=C, n_times=T,
                                    port_region=cfg.get("port_region")).to(dev)
    # `port_idx` was added to VisualContrastiveLoop AFTER these checkpoints were saved,
    # so a strict load of visual_contrastive_v2.pt now fails outright -- the script that
    # published 63.5% stopped being able to reproduce it.  the buffer is safe to rebuild:
    # with no `port_region` in the config the port is the deterministic [:n//8] slice and
    # port_idx is just arange(n//8), carrying nothing learned.  loading non-strictly is
    # therefore exact -- but ONLY for that key, so anything else missing still fails loud.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    allowed = {"port_idx"}
    if set(missing) - allowed or unexpected:
        raise RuntimeError(f"state_dict mismatch beyond {allowed}: "
                           f"missing={sorted(set(missing) - allowed)} unexpected={sorted(unexpected)}")
    model.eval()

    x = torch.from_numpy(np.ascontiguousarray(imgs)).to(dev)
    x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1
    yt = torch.from_numpy(y).to(dev)
    with torch.no_grad():
        z, _ = model.embed_image(x, substeps=cfg["substeps"], dt=cfg["dt"],
                                 n_steps=cfg.get("n_steps", 4))
        ze = model.embed_eeg(yt)
        sim = z @ ze.T                       # sim[i, j] = image i against eeg j

    lbl = torch.arange(n, device=dev)
    # the direction eval_things_test.py publishes: given an IMAGE, rank the EEGs
    rank_img = (sim > sim.gather(1, lbl[:, None])).sum(1)
    top1_img = float((rank_img == 0).float().mean())
    # the direction this panel shows: given an EEG, rank the IMAGES.
    # this figure is DEVICE-DEPENDENT by exactly one trial: the site's 60.0% (120/200) came
    # from a GPU pass, and the CPU pass reads 60.5% (121/200).  measured on the CPU, test
    # image 70's true similarity beats its best competitor by +2.4e-5 on similarities of
    # ~0.24 -- a tie to within float reduction order -- and the next-closest margin is
    # 8x larger.  image 70 is outside the first 23, so no exported row depends on it.
    # the image->eeg figure checked below reproduces 63.50% on both devices.
    simT = sim.T
    rank_eeg = (simT > simT.gather(1, lbl[:, None])).sum(1)
    top1_eeg = float((rank_eeg == 0).float().mean())

    print(f"{a.ckpt}: step {d['step']}, {n} images, chance {100.0/n:.2f}%")
    print(f"  image->eeg top-1 {100*top1_img:.2f}%   (published figure)")
    print(f"  eeg->image top-1 {100*top1_eeg:.2f}%   (what this panel shows)")
    if abs(top1_img - a.expect_top1) > 0.005:
        print(f"  !! does not reproduce the published {100*a.expect_top1:.1f}% -- refusing to write",
              file=sys.stderr)
        return 2

    # ---- which trials: the first N of the test set IN DATASET ORDER, and nothing else ----
    # this used to take "the first two rank-1 trials and the first trial ranked 2-5".  that
    # rule was stated and was not chosen by eye, but it was still a SELECTION ON THE
    # OUTCOME: it read the ranks first and kept the trials with the ranks it wanted, so
    # three slots showed two hits out of three whatever the model's real rate was.  with
    # enough slots to show a spread, the unbiased thing is to take them in order, read NO
    # rank before choosing, and let each row's rank line say where the true image landed
    # -- including rows where it is not in the top five at all.
    order = torch.argsort(simT, dim=1, descending=True).cpu().numpy()
    r = rank_eeg.cpu().numpy()
    if not 1 <= a.n_trials <= n:
        print(f"  !! --n-trials must be in 1..{n}", file=sys.stderr)
        return 2
    chosen = list(range(a.n_trials))
    k1 = int((r[chosen] == 0).sum())
    print(f"  first {a.n_trials} trials in dataset order: {k1} at rank 1, "
          f"{int((r[chosen] < 5).sum())} in the top five "
          f"(a sample of {a.n_trials}, not an estimate of the {100*top1_eeg:.1f}%)")

    OUTIMG.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    # one file per TEST IMAGE, not per slot: the same photograph turns up as the stimulus
    # of one row and a candidate in several others, and 23 rows x 6 slots written per slot
    # would ship most of them several times over.  the directory is this script's alone,
    # so stale files from an earlier export are cleared rather than left to ship unused.
    for old in OUTIMG.glob("*.jpg"):
        old.unlink()
    written: dict[int, str] = {}

    def thumb(idx: int) -> str:
        if idx in written:
            return written[idx]
        src = ROOT / str(paths[idx])
        im = Image.open(src).convert("RGB") if src.exists() else Image.fromarray(imgs[idx])
        im = im.resize((a.thumb, a.thumb), Image.LANCZOS)
        rel = f"media/retrieval/test{idx:03d}.jpg"
        im.save(ROOT / "site" / rel, quality=84, optimize=True)
        written[idx] = rel
        return rel

    def concept(idx: int) -> str:
        return Path(str(paths[idx])).parent.name.split("_", 1)[-1].replace("_", " ")

    trials = []
    for k, i in enumerate(chosen):
        top5 = [int(j) for j in order[i][:5]]
        trials.append({
            "seen": {"src": thumb(i), "label": concept(i)},
            "rank": int(r[i]) + 1,
            "top5": [{"src": thumb(j), "label": concept(j), "isTrue": j == i}
                     for j in top5],
            # the measured response, decimated for drawing: a handful of posterior
            # channels is what the reader can actually parse
            "eeg": [[round(float(v), 3) for v in ev[i, c, ONSET:ONSET + KEEP]]
                    for c in range(0, C, 8)],
        })

    payload = {
        "ckpt": a.ckpt, "step": int(d["step"]), "n": n, "chance": 1.0 / n,
        "top1": top1_eeg, "top1_published": top1_img,
        "selection": f"the first {a.n_trials} test trials in dataset order, chosen before any rank was read",
        "trials": trials,
    }
    OUTJS.write_text(
        "/* generated by scripts/export_site_retrieval.py -- do not edit by hand.\n"
        f"   real trials from the designated THINGS-EEG2 test set, {a.ckpt} step {d['step']}. */\n"
        "window.IBM_RETRIEVAL = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    for t in trials:
        print(f"  trial {t['seen']['label']:22} true image at rank {t['rank']}")
    print(f"{OUTJS}: {len(trials)} trials, {OUTJS.stat().st_size/1e3:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
