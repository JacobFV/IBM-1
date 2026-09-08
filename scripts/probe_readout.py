"""does the cortical readout preserve the information the drive put in?

three video objectives failed on one branch, and the contrastive run said why in
a way worth checking directly: the state after 8 dynamics steps retrieved frame
t+50 *worse* than frame t's own embedding did.  that is consistent with the
dynamics being unpredictive, and equally consistent with the READOUT destroying
whatever the drive carried before prediction is even attempted.

so this asks the question whose answer is known.  encode a frame, drive the
cortex, integrate N steps, read the anterior sites, and try to identify WHICH
frame drove it, among in-film distractors.  at N=0 this must be near perfect --
the drive is a linear map of the encoding and nothing has happened yet.  if
accuracy collapses as N grows, the substrate is erasing its input, and no
objective downstream can recover it.

the encoder is UNTRAINED on purpose.  a trained one would confound "the readout
preserves information" with "the encoder learned to survive the readout"; a
random projection measures the channel itself.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, os
import numpy as np
import torch
import torch.nn.functional as F

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/derived/pd-film")
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", default="out/probe_readout.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    films = sorted(glob.glob(f"{a.corpus}/*_frames.npy"))[-2:]
    TE = [np.load(f, mmap_mode="r") for f in films]
    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev).to(dev)
    model = P.VideoLoop(dyn).to(dev).eval()
    read_lo = -dyn.n // 8

    print(f"identify the driving frame from the cortical readout, "
          f"{a.batch} in-film candidates, chance {100/a.batch:.2f}%")
    print(f"encoder is UNTRAINED -- this measures the channel, not a model\n")
    print(f"{'steps':>6s} {'sim ms':>7s} {'top-1':>8s} {'r_eff':>7s} {'|v|max':>8s}")

    res = {}
    for N in (0, 1, 2, 4, 8, 16, 32):
        accs, res_es, vmaxes = [], [], []
        for r in range(a.reps):
            rng = np.random.default_rng(100 + r)
            v = TE[rng.integers(len(TE))]
            i = rng.integers(0, len(v) - 1, a.batch)
            x = ((torch.from_numpy(np.stack([np.asarray(v[j]) for j in i]))
                  .to(dev).permute(0, 3, 1, 2).float() / 127.5) - 1.0)
            with torch.no_grad():
                b = x.shape[0]
                drive = torch.zeros(b, dyn.n, device=dev)
                drive[:, :model.n_in] = model.to_cortex(model.enc(x))
                s = dyn.init_state(b, dev)
                w = dyn.edge_weights()
                for _ in range(N):
                    s = dyn.step(s, drive, a.dt, w)
                # read the same sites the video loop reads
                z = F.normalize(s[1][:, read_lo:] if N > 0
                                else drive[:, :model.n_in], dim=-1)
                # the reference is the drive itself: can the readout be matched
                # back to the input that produced it?
                d0 = F.normalize(drive[:, :model.n_in], dim=-1)
                sim = z @ d0.T if N == 0 else z @ z.T * 0 + (z @ d0.T)
                lb = torch.arange(b, device=dev)
                accs.append(float((sim.argmax(1) == lb).float().mean()))
                res_es.append(P.effective_rank(
                    (s[1] if N > 0 else drive)[:, ::max(dyn.n // 512, 1)].float()))
                vmaxes.append(float(s[0].abs().max()) if N > 0 else 0.0)
        m = float(np.mean(accs))
        res[N] = {"top1": m, "r_eff": float(np.mean(res_es)),
                  "v_absmax": float(np.mean(vmaxes))}
        print(f"{N:6d} {N*a.dt*1000:7.1f} {100*m:7.2f}% {np.mean(res_es):7.2f} "
              f"{np.mean(vmaxes):8.1f}", flush=True)

    a0, a8 = res[0]["top1"], res[8]["top1"]
    print(f"\n  N=0 (drive itself)     {100*a0:.2f}%")
    print(f"  N=8 (what video used)  {100*a8:.2f}%   retained {100*a8/max(a0,1e-9):.1f}%")
    if a8 < 0.2 * a0:
        print("\n  -> THE READOUT ERASES THE DRIVE.  by the time the video loop reads "
              "the state, the frame that produced it is no longer identifiable, so "
              "no downstream objective could have worked.")
    else:
        print("\n  -> the readout preserves the drive; the failure is downstream of it.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
