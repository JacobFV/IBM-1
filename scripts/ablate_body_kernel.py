#!/usr/bin/env python3
"""does a kernel TRAINED on the body carry motor content its permutation does not?

IHM-1 measured that the IBM kernel's site rows can be permuted -- destroying the
learned association while preserving every marginal statistic -- and the body
still recovers the same push.  their conclusion was that the stance policy does
not depend on what the kernel learned, only on the E/I network being a
well-conditioned filter.  that result is correct, and it was measured on a kernel
trained ONLY on vision, audio and EEG retrieval.  nothing in that corpus is
motor, so a permutation of it had no motor content to destroy.

this asks the question their control leaves open, on a kernel that HAS been
trained on a motor corpus: `body_stance` reached skill +0.42 against predicting
the training mean, on commands recorded from an engineered LQR holding a body
upright under perturbation.

four arms, one held-out split, the same decoder architecture refit on each so
that no arm keeps a decoder tuned to a different kernel:

  trained    the kernel body_stance learned
  permuted   the same kernel, site rows shuffled -- IHM-1's control exactly
  random     a fresh kernel at matched scale
  frozen     the kernel at its initialisation

the prediction that makes this falsifiable: if the motor corpus wrote something
into the kernel, trained beats permuted.  if it did not, they tie and IHM-1's
finding extends to kernels trained on the body -- which would be a stronger and
more interesting negative than the one they have.
"""
import argparse, importlib.util, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("ptrain", os.path.join(HERE, "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="ckpt/body_stance_v2.pt")
    ap.add_argument("--corpus", default="data/derived/body-corpus")
    ap.add_argument("--refit-steps", type=int, default=1500)
    ap.add_argument("--out", default="out/ablate_body_kernel.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    kernel = d["dyn.embed"]; sites = d["sites"]; embed = d["embed"]
    X = torch.from_numpy(np.load(f"{a.corpus}/state.npy"))
    Y = torch.from_numpy(np.load(f"{a.corpus}/command.npy"))
    muscles = json.load(open(f"{a.corpus}/meta.json"))["muscles"]
    n = len(X); ntr = int(n * .8); gap = max(1, n // 50)
    tr, te = slice(0, ntr), slice(ntr + gap, n)
    Xtr, Ytr, Xte, Yte = X[tr].to(dev), Y[tr].to(dev), X[te].to(dev), Y[te].to(dev)
    mean_mse = float(((Yte - Ytr.mean(0)) ** 2).mean())

    # the ridge is the ceiling: the LQR is u = -Kx, so a linear fit is the best
    # any model can do on this corpus, and it is what "good" means here.
    A = torch.cat([Xtr - Xtr.mean(0), torch.ones(len(Xtr), 1, device=dev)], 1)
    B = torch.cat([Xte - Xtr.mean(0), torch.ones(len(Xte), 1, device=dev)], 1)
    W = torch.linalg.solve(A.T @ A + 1e-3 * torch.eye(A.shape[1], device=dev),
                           A.T @ (Ytr - Ytr.mean(0)))
    ridge = float(((Yte - (B @ W + Ytr.mean(0))) ** 2).mean())

    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    arms = {
        "trained": kernel.clone(),
        "permuted": kernel[torch.randperm(kernel.shape[0], generator=g)],
        "random": torch.randn(kernel.shape, generator=g) * float(kernel.std()),
        "frozen_init": torch.randn(kernel.shape, generator=torch.Generator().manual_seed(1)) * 0.02,
    }

    print(f"{a.ckpt}: {sites} sites, step {d.get('step')}")
    print(f"held-out {len(Xte)} steps | predict-mean {mean_mse:.8f} | "
          f"ridge {ridge:.8f} (skill {1-ridge/mean_mse:+.4f})\n")
    print(f"{'arm':14s} {'held MSE':>12s} {'skill vs mean':>14s} {'of ridge':>10s}")
    res = {"mean_mse": mean_mse, "ridge_mse": ridge,
           "ridge_skill": 1 - ridge / mean_mse, "arms": {}}
    for name, emb in arms.items():
        # refit the decoder for EVERY arm.  keeping the trained decoder would
        # measure "does this decoder like this kernel", not "does this kernel
        # carry motor content".
        torch.manual_seed(0)
        dyn = P.CorticalDynamics(sites, embed, 48, dev).to(dev)
        dyn.embed.data.copy_(emb.to(dev))
        m = P.SensorimotorLoop(dyn, muscles=muscles,
                               afferent_channels=X.shape[1]).to(dev)
        opt = torch.optim.Adam([p for q, p in m.named_parameters()
                                if not q.startswith("dyn.")], lr=3e-3)
        rng = np.random.default_rng(0)
        for _ in range(a.refit_steps):
            i = torch.from_numpy(rng.integers(0, ntr, 128)).to(dev)
            pred, _ = m(Xtr[i], n_steps=8, substeps=2)
            loss = F.mse_loss(pred, Ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p_, _ = m(Xte, n_steps=8, substeps=2)
            mse = float(F.mse_loss(p_, Yte))
        sk = 1 - mse / mean_mse
        res["arms"][name] = {"mse": mse, "skill_vs_mean": sk,
                             "fraction_of_ridge": sk / max(res["ridge_skill"], 1e-9)}
        print(f"{name:14s} {mse:12.8f} {sk:+14.4f} {100*sk/max(res['ridge_skill'],1e-9):9.1f}%")

    t, p_m = res["arms"]["trained"]["mse"], res["arms"]["permuted"]["mse"]
    gap_pct = 100 * (p_m - t) / p_m
    print(f"\n  trained vs permuted: {gap_pct:+.2f}% MSE")
    verdict = ("the body-trained kernel CARRIES motor content its permutation does not"
               if p_m > t * 1.05 else
               "trained and permuted tie -- a kernel trained on the body still does "
               "not carry motor content, and IHM-1's finding extends to it")
    print(f"  -> {verdict}")
    res["verdict"] = verdict; res["trained_vs_permuted_pct"] = gap_pct
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
