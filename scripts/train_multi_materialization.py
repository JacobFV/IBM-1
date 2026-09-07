"""train several materializations of one implicit model at once.

this is the architecture's own claim made operational.  ARCHITECTURE.md says
there is no standard materialization: an explicit model is a lazy projection of
the implicit one, and different projections share whatever they both use.
STATE.md 7c draws the consequence -- the posterior over one theta given several
corpora factorizes,

    grad log p(theta | D)  =  grad log p(theta)  +  SUM_d grad log p(D_d | theta)

-- so materializations are not competing training regimes, they are more terms in
the same product.  this script IS that sum: one `CorticalDynamics`, several heads,
and gradients accumulated onto the shared substrate before a single optimizer
step.

what makes it more than a convenience: the terms constrain each other in ways
neither does alone.

*the paired term anchors.*  a self-supervised loop can hit its loss with a
decorative cortex, because encoder and decoder are both learned.  the paired term
has a MEASURED target, so it prices any cortical state that does not produce real
MEG.

*the self-supervised terms supply volume.*  four hours of paired MEG is a lot of
paired MEG and almost no data by pretraining standards.  video and audio are
unbounded.

*the shared theta is the only channel between them.*  the video head never sees
MEG and the MEG head never sees a frame; anything they teach each other has to
pass through the association kernel, which is exactly the object we want taught.

weighting is by 1/phi rather than by corpus size, following the same argument
forge_joint makes: a term's contribution should reflect how much independent
information it carries, and a long recording is not many independent samples.
"""
from __future__ import annotations

import argparse
import os
import os, json, time, math
import numpy as np, torch, torch.nn.functional as F

import importlib.util
spec = importlib.util.spec_from_file_location(
    "ptrain", __file__.replace("train_multi_materialization.py", "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", type=int, default=150_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--dyn-steps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--viability-weight", type=float, default=1e-1)
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-audio", required=True)
    ap.add_argument("--paired-stim", required=True)
    ap.add_argument("--paired-neural", required=True)
    ap.add_argument("--w-av", type=float, default=1.0)
    ap.add_argument("--w-paired", type=float, default=1.0)
    ap.add_argument("--ckpt", default="ckpt/multi.pt")
    ap.add_argument("--upload-every", type=int, default=2000)
    ap.add_argument("--out", default="out/multi.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    frames = np.load(a.video, mmap_mode="r")
    vcoch = np.load(a.video_audio, mmap_mode="r")
    pstim = np.load(a.paired_stim, mmap_mode="r")
    pneur = np.load(a.paired_neural, mmap_mode="r")
    # see the note in pretrain_video_loop: the stored MEG is scaled by a std that
    # artifacts dominate, so it is rescaled at load by a recorded median/IQR.
    _sc = a.paired_neural.replace("meg_250hz", "meg_scale")
    megsc = np.load(_sc) if os.path.exists(_sc) else None
    print(f"video {frames.shape} + {vcoch.shape} | paired {pstim.shape} -> {pneur.shape}",
          flush=True)

    # ONE substrate.  every head below indexes the same embedding table.
    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
    av = P.AudioVisualLoop(dyn, n_bands=vcoch.shape[-1]).to(dev)
    pr = P.PairedNeuralLoop(dyn, n_bands=pstim.shape[-1],
                            n_sensors=pneur.shape[-1]).to(dev)

    shared = sum(p.numel() for p in dyn.parameters())
    head_av = sum(p.numel() for p in av.parameters()) - shared
    head_pr = sum(p.numel() for p in pr.parameters()) - shared
    print(f"shared substrate {shared:,} | av head {head_av:,} | paired head {head_pr:,} "
          f"| total {shared+head_av+head_pr:,}", flush=True)

    params = list(dyn.parameters()) + \
             [p for n, p in av.named_parameters() if not n.startswith("dyn.")] + \
             [p for n, p in pr.named_parameters() if not n.startswith("dyn.")]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)

    H, vctx, pctx = a.horizon, 8, 125
    log = {"config": vars(a), "shared": shared, "steps": []}
    t0 = time.time()

    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)

        # -- term 1: the audio-visual self-supervised materialization ------
        lim = min(len(frames), len(vcoch)) - H - 2
        i = np.random.randint(vctx, lim, size=a.batch)
        xv = torch.from_numpy(np.ascontiguousarray(frames[i])).to(dev)
        yv = torch.from_numpy(np.ascontiguousarray(frames[i + H])).to(dev)
        xv = (xv.permute(0, 3, 1, 2).float() / 127.5) - 1.0
        yv = (yv.permute(0, 3, 1, 2).float() / 127.5) - 1.0
        xa = torch.from_numpy(np.stack([vcoch[j - vctx:j] for j in i])).float().to(dev)
        ya = torch.from_numpy(np.ascontiguousarray(vcoch[i + H])).float().to(dev)
        pv, pa, sv = av(xv, xa, a.dyn_steps, a.dt)
        l_av = F.mse_loss(pv, yv) + 0.5 * F.mse_loss(pa, ya)
        viab_av = P.viability_penalty(sv[0])
        (a.w_av * (l_av + a.viability_weight * viab_av)).backward()

        # -- term 2: the paired stimulus -> measured MEG materialization ----
        lim = min(len(pstim), len(pneur)) - 2
        j = np.random.randint(pctx, lim, size=a.batch)
        xp = torch.from_numpy(np.stack([pstim[q - pctx:q] for q in j])).float().to(dev)
        yn = np.ascontiguousarray(pneur[j]).astype(np.float32)
        if megsc is not None:
            yn = np.clip((yn - megsc[0]) / megsc[1], -6, 6)
        yp = torch.from_numpy(yn).to(dev)
        pp, sp = pr(xp, a.dyn_steps, a.dt)
        l_pr = F.mse_loss(pp, yp)
        viab_pr = P.viability_penalty(sp[0])
        (a.w_paired * (l_pr + a.viability_weight * viab_pr)).backward()

        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % 25 == 0 or step == a.steps - 1:
            with torch.no_grad():
                r_av = P.effective_rank(sv[1][:, ::max(dyn.n // 512, 1)].float())
                r_pr = P.effective_rank(sp[1][:, ::max(dyn.n // 512, 1)].float())
                xm = av.cross_modal_weight()
            rec = {"step": step, "av": float(l_av), "paired": float(l_pr),
                   "r_eff_av": r_av, "r_eff_paired": r_pr, "cross_modal": xm,
                   "grad_norm": float(gn), "sec": round(time.time() - t0, 1)}
            log["steps"].append(rec)
            print(f"{step:5d}  av {float(l_av):.4f}  meg {float(l_pr):.4f}  "
                  f"r_av {r_av:5.2f}  r_meg {r_pr:5.2f}  xmod {xm:.4f}  "
                  f"{time.time()-t0:6.0f}s", flush=True)
            if not math.isfinite(float(l_av) + float(l_pr)):
                print("DIVERGED", flush=True); break

        if a.ckpt and a.upload_every and step and step % a.upload_every == 0:
            # save before anything that can raise; see the note in the sibling script
            torch.save({"dyn": dyn.state_dict(), "av": av.state_dict(),
                        "paired": pr.state_dict(), "step": step}, a.ckpt)
            from ibm.release import CheckpointName, sidecar, upload
            nm = CheckpointName(modality="multi", sites=a.sites, embed=a.embed,
                                degree=a.k, objective="av+meg",
                                viability_weight=a.viability_weight, step=step)
            torch.save({"dyn": dyn.state_dict(), "av": av.state_dict(),
                        "paired": pr.state_dict(), "step": step, "name": str(nm)}, a.ckpt)
            meta = sidecar(nm, geometry="spherical shell, area-matched to 202,437 mm^2",
                           n_params=shared + head_av + head_pr, n_assoc=dyn.embed.numel(),
                           metrics=log["steps"][-1], config=vars(a))
            try:
                print("  uploaded", upload(a.ckpt, meta), flush=True)
            except Exception as e:
                print(f"  upload failed: {e}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(log, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
