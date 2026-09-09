"""does it matter WHERE the drive enters the sheet?

`optic_nerve` and `visual_eeg` share a corpus, a target and an encoder and differ
in one thing: the first drives the sites `cortical_regions` labels occipital, the
second drives `drive[:, :dyn.n // 8]` -- an arbitrary eighth of a seeded point
cloud that a comment calls occipital.  comparing their training curves does not
answer the question, because they started from different places: visual_eeg was
warm-started at 26.75% and optic_nerve from scratch.

so this asks it the way that admits an answer.  take ONE trained model and move
its port, changing nothing else:

  as-trained   the port the model learned with
  shifted      the same NUMBER of sites, contiguous, starting elsewhere
  scattered    the same number, drawn uniformly at random across the sheet
  antipodal    the occipital sites' opposite hemisphere on the sphere

if the anatomy is doing work, an anatomically-placed port should beat a scattered
one of identical size.  if all four score the same, the port assignment is
decoration and the "declared pathway" framing should be dropped rather than
defended -- which is the outcome this repo has had to accept twice already for
magnitude claims.
"""
from __future__ import annotations

import argparse, importlib.util, json, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("ptrain", os.path.join(HERE, "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

D = "data/derived/things-paired"
ONSET, KEEP = 20, 50


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/ibm1_curriculum16.pt")
    ap.add_argument("--term", default="optic_nerve")
    ap.add_argument("--pools", type=int, default=8)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--out", default="out/ablate_port.json")
    a = ap.parse_args()

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    heads = d.get("heads", {})
    if a.term not in heads:
        raise SystemExit(f"{a.term} not in checkpoint; have {sorted(heads)[:6]}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sites, embed = d["sites"], d["embed"]

    tr = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    imgs = np.load(f"{D}/images_training.npy", mmap_mode="r")
    n = min(len(tr), len(imgs)); ntr = int(n * 0.8)
    s_ = np.asarray(tr[:ntr:7]).astype(np.float32)
    med = np.median(s_, 0)
    iqr = ((np.percentile(s_, 75, 0) - np.percentile(s_, 25, 0)) / 1.349).clip(1e-9)

    def eeg(i):
        y = (np.asarray(tr[i]).astype(np.float32) - med) / iqr
        return np.clip(y, -6, 6)[..., ONSET:ONSET + KEEP:2]

    T = eeg(np.arange(4)).shape[-1]
    dyn = P.CorticalDynamics(sites, embed, 48, dev).to(dev)
    dyn.embed.data.copy_(d["dyn.embed"].to(dev))
    model = P.CranialNerveLoop(dyn, nerve="optic", lobe="occipital",
                               n_sensors=tr.shape[1], n_times=T).to(dev)
    model.load_state_dict(heads[a.term]); model.eval()

    native = model.port.clone()
    k = len(native)
    g = torch.Generator().manual_seed(0)
    start = (int(native.max()) + sites // 3) % max(1, sites - k)
    ports = {
        "as-trained (occipital)": native,
        "shifted (contiguous elsewhere)": torch.arange(start, start + k),
        "scattered (uniform random)": torch.randperm(sites, generator=g)[:k].sort().values,
        "antipodal": ((native + sites // 2) % sites).sort().values,
    }

    res = {}
    print(f"{a.ckpt} :: {a.term}, step {d.get('step')}")
    print(f"port is {k:,} of {sites:,} sites; chance {100.0/a.pool:.2f}%\n")
    print(f"{'port':34s} {'top-1':>8s} {'x chance':>9s}")
    for name, idx in ports.items():
        model.port = idx.to(dev)
        accs = []
        with torch.no_grad():
            rng = np.random.default_rng(4242)
            for _ in range(a.pools):
                j = rng.integers(ntr, n - 1, a.pool)
                x = torch.from_numpy(np.ascontiguousarray(imgs[j])).to(dev)
                x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1
                y = torch.from_numpy(eeg(j)).to(dev)
                z, _ = model.embed_stimulus(x)
                sim = z @ model.embed_eeg(y).T
                accs.append(float((sim.argmax(1) ==
                            torch.arange(len(j), device=dev)).float().mean()))
        m, sd = float(np.mean(accs)), float(np.std(accs))
        res[name] = {"top1": m, "sd": sd}
        print(f"{name:34s} {100*m:7.2f}% {m*a.pool:8.1f}x   +/-{100*sd:.2f}")
    model.port = native.to(dev)

    base = res["as-trained (occipital)"]["top1"]
    sc = res["scattered (uniform random)"]["top1"]
    gap = base - sc
    csd = (res["as-trained (occipital)"]["sd"] ** 2 +
           res["scattered (uniform random)"]["sd"] ** 2) ** 0.5
    print(f"\n  anatomical - scattered: {100*gap:+.2f} points "
          f"({gap/max(csd,1e-9):+.2f} sd)")
    print("  -> " + ("the port placement carries the result"
                     if gap > 2 * csd else
                     "PLACEMENT IS NOT DOING THE WORK -- a scattered port of the "
                     "same size scores the same, so the anatomy is decoration here"))
    res["verdict_gap_sd"] = gap / max(csd, 1e-9)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
