"""video continuation as RETRIEVAL, because regression collapses to the mean.

every video run in this programme has failed the same way and the failure is not
a bug.  under MSE against frame t+H, the loss-optimal output for an unpredictable
target is its CONDITIONAL MEAN, and for natural film the mean change over 320 ms
is ~0.  so the model correctly learns to emit nothing: the 37.7 h run reaches
skill +0.0009 with a residual magnitude of 0.004 and its own diagnostic prints
`[degenerate: emits ~nothing]`.  the via-EEG route does emit something and lands
at -0.92.  more training cannot fix either -- zero IS the optimum being sought.

this repo has already solved exactly this shape once, and recorded it:
`control_speech_to_meg.py` opens with it -- "the visual term had exactly that
shape and was rescued by changing the OBJECTIVE: waveform regression peaked at
skill +0.011 while contrastive retrieval reached 42x chance on the same pairs."

so ask for DISCRIMINATION instead of reconstruction.  given frame t, does the
cortical state identify which of N candidates is the true continuation?  a
constant output scores chance, so the degenerate solution is worth nothing and
the model has to represent something about what happens next.

THE BASELINE THAT MAKES OR BREAKS THIS.  retrieving frame t+H from a pool of
random frames is trivial -- t+H looks like t, so raw appearance solves it and the
number would mean nothing.  two things fix that, and both are required:

  hard negatives   the pool is drawn from the SAME FILM within a window of the
      anchor, so every candidate shares the scene, the lighting and the palette,
      and appearance alone cannot separate them.

  the pixel baseline   raw frame-t against candidate cosine, no model at all, on
      the identical pools.  this is the persistence analogue for retrieval and it
      is what the model has to beat.  reporting top-1 against chance alone would
      repeat the mistake this file exists to avoid.

whole FILMS are held out, so this asks for continuation of footage never seen.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(HERE, "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/derived/pd-film")
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--dyn-steps", type=int, default=8)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--window", type=int, default=600,
                    help="negatives are drawn within +/- this many frames of the "
                         "anchor, from the same film -- hard negatives that share "
                         "the scene, so appearance alone cannot solve the task")
    ap.add_argument("--eval-pools", type=int, default=8,
                    help="a single pool has sd of several points and has misled "
                         "this project three times")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--holdout-films", type=int, default=2)
    ap.add_argument("--ckpt", default="ckpt/video_contrastive_cont.pt")
    ap.add_argument("--out", default="out/video_contrastive_cont.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    films = sorted(glob.glob(f"{a.corpus}/*_frames.npy"))
    if len(films) <= a.holdout_films:
        raise SystemExit(f"{len(films)} films is not enough to hold out "
                         f"{a.holdout_films}")
    tr_f, te_f = films[:-a.holdout_films], films[-a.holdout_films:]
    TR = [np.load(f, mmap_mode="r") for f in tr_f]
    TE = [np.load(f, mmap_mode="r") for f in te_f]
    n_tr = sum(len(v) for v in TR)
    print(f"train {len(TR)} films, {n_tr:,} frames ({n_tr/25/3600:.2f} h)", flush=True)
    print(f"held out whole: {[os.path.basename(f) for f in te_f]}", flush=True)
    print(f"pool {a.pool} (chance {100/a.pool:.2f}%), hard negatives within "
          f"+/-{a.window} frames of the anchor, same film", flush=True)

    def draw(pool_list, m, rng, window: int):
        """one anchor and m-1 hard negatives, all from ONE film near the anchor.

        the true continuation sits at index 0 of the candidates; the caller
        never sees that, the labels do.
        """
        v = pool_list[rng.integers(len(pool_list))]
        lo, hi = 0, len(v) - a.horizon - 1
        if hi - lo < 4 * m:
            v = max(pool_list, key=len); lo, hi = 0, len(v) - a.horizon - 1
        anchors = rng.integers(lo, hi, m)
        cands = np.empty(m, dtype=np.int64)
        for b, i in enumerate(anchors):
            cands[b] = i + a.horizon
        # negatives for sample b are the OTHER rows' targets, so every candidate
        # is a real continuation of some anchor in the same film -- an in-batch
        # contrastive set with hard negatives by construction.
        w = min(window, (hi - lo) // 2)
        base = rng.integers(lo + w, hi - w) if hi - lo > 2 * w else (lo + hi) // 2
        anchors = np.clip(base + rng.integers(-w, w, m), lo, hi - 1)
        cands = anchors + a.horizon
        f = lambda idx: (torch.from_numpy(
            np.stack([np.asarray(v[j]) for j in idx])).to(dev)
            .permute(0, 3, 1, 2).float() / 127.5) - 1.0
        return f(anchors), f(cands)

    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev).to(dev)
    model = P.VideoLoop(dyn).to(dev)
    # the cortical readout becomes an EMBEDDING head; the decoder is unused here
    head = nn.Sequential(nn.Linear(model.read_sites, 512), nn.GELU(),
                         nn.Linear(512, a.dim)).to(dev)
    tgt = nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
        nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(), nn.Flatten(),
        nn.Linear(128 * 8 * 8, 512), nn.GELU(), nn.Linear(512, a.dim)).to(dev)
    temp = nn.Parameter(torch.tensor(0.07, device=dev))
    params = ([p for n, p in model.named_parameters() if not n.startswith("dec.")]
              + list(head.parameters()) + list(tgt.parameters()) + [temp])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    print(f"trainable {sum(p.numel() for p in params):,} "
          f"({dyn.embed.numel():,} association)", flush=True)

    def embed_anchor(x):
        s = model.dyn.init_state(x.shape[0], x.device)
        w = model.dyn.edge_weights()
        drive = torch.zeros(x.shape[0], model.dyn.n, device=x.device)
        drive = drive.index_copy(
            1, torch.arange(model.n_in, device=x.device),
            model.to_cortex(model.enc(x)))
        for _ in range(a.dyn_steps):
            s = model.dyn.step(s, drive, a.dt, w)
        return F.normalize(head(s[1][:, model.read_idx.to(x.device)]), dim=-1), s

    @torch.no_grad()
    def evaluate(step):
        """model top-1 and the RAW PIXEL baseline on the identical pools."""
        accs, pix = [], []
        rng = np.random.default_rng(90_000 + step)
        for _ in range(a.eval_pools):
            x, y = draw(TE, a.pool, rng, a.window)
            z, _ = embed_anchor(x)
            zt = F.normalize(tgt(y), dim=-1)
            lbl = torch.arange(len(x), device=dev)
            accs.append(float(((z @ zt.T).argmax(1) == lbl).float().mean()))
            # no model at all: does frame t simply LOOK most like its own t+H?
            xf = F.normalize(x.flatten(1), dim=-1)
            yf = F.normalize(y.flatten(1), dim=-1)
            pix.append(float(((xf @ yf.T).argmax(1) == lbl).float().mean()))
        return (float(np.mean(accs)), float(np.std(accs)),
                float(np.mean(pix)), float(np.std(pix)))

    chance = 1.0 / a.pool
    log = {"config": vars(a), "chance": chance, "train_films": tr_f,
           "holdout_films": te_f, "steps": []}
    best, t0 = -1e9, time.time()
    rng = np.random.default_rng(0)
    for step in range(a.steps + 1):
        x, y = draw(TR, a.batch, rng, a.window)
        z, s = embed_anchor(x)
        zt = F.normalize(tgt(y), dim=-1)
        logits = z @ zt.T / temp.clamp(0.01, 1.0)
        lbl = torch.arange(len(x), device=dev)
        loss = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl)) \
            + 1e-1 * P.viability_penalty(s[0])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

        if step % a.eval_every == 0:
            m, sd, pm, psd = evaluate(step)
            r = float(P.effective_rank(s[1])) if hasattr(P, "effective_rank") else float("nan")
            rec = {"step": step, "loss": float(loss), "top1": m, "sd": sd,
                   "pixel_top1": pm, "pixel_sd": psd,
                   "x_chance": m / chance, "vs_pixel": m - pm}
            log["steps"].append(rec)
            flag = ""
            if m > pm + 2 * (sd + psd) / 2:
                flag = "  <- BEATS the pixel baseline"
            print(f"{step:6d}  loss {float(loss):.4f}  top-1 {100*m:5.2f}%+/-{100*sd:4.2f} "
                  f"({m/chance:5.1f}x chance) | PIXEL {100*pm:5.2f}%+/-{100*psd:4.2f} "
                  f"| model-pixel {100*(m-pm):+6.2f}pt   {time.time()-t0:5.0f}s{flag}",
                  flush=True)
            if m - pm > best:
                best = m - pm
                os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
                tmp = a.ckpt + ".tmp"
                torch.save({"model": model.state_dict(), "head": head.state_dict(),
                            "tgt": tgt.state_dict(), "config": vars(a),
                            "step": step, "top1": m, "pixel_top1": pm}, tmp)
                os.replace(tmp, a.ckpt)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(log, open(a.out, "w"), indent=2)
    print(f"\nbest margin over the pixel baseline: {100*best:+.2f} points", flush=True)


if __name__ == "__main__":
    main()
