#!/usr/bin/env python3
"""Does the MSE objective DESTROY a good readout, or merely fail to find one?

Established today, all on one fixed held-out draw: the signal is in the corpus (ridge on the
stimulus +0.0466), it survives the dynamics (ridge on the cortical state +0.0400), and it is
reachable at the head's OWN rank-64 lead-field constraint (+0.0400, identical to full rank). The
trained head reaches +0.0006. A 67x gap with every architectural explanation measured and
excluded.

That leaves two readings of the objective, and they are not the same problem:

  MSE MERELY FAILS TO FIND IT. The optimum is fine and the optimisation never gets there from a
  random start. Then the fix is cheap: initialise the readout by ridge and train from there.
  MSE ACTIVELY DESTROYS IT. The gradient is dominated by variance no stimulus can predict, so a
  head placed ON the good solution is driven off it. Then initialisation cannot help and the
  objective has to change, which is what `VisualContrastiveLoop`'s docstring has argued from the
  start.

This installs the ridge solution INTO the head's own `lead_u`/`lead_v`, checks it reproduces the
ridge's number, and then trains with the ordinary MSE objective while watching held-out
correlation. Degradation from +0.0400 is the second reading; stability or improvement is the
first.

THE INSTALLATION IS ITS OWN KNOWN ANSWER, and a strong one. The head computes
`lead_v(lead_u(state))`, a bias-free linear map from `read_sites` to sensors. The ridge is fitted
raw -- no centring, no standardisation -- precisely so it IS such a map, then factored by SVD to
rank `lead_rank`: `W ~= A @ B` gives `lead_v.weight = A`, `lead_u.weight = B`. Running the head
with those weights must reproduce the ridge's own held-out predictions to numerical precision. If
it does not, the installation is wrong and nothing after it means anything. This cannot pass by
accident: a transposed, mis-scaled or mis-factored install changes the predictions completely.

The dynamics are FROZEN throughout. Only the readout trains, which is the most favourable case
for the objective -- if MSE degrades a good readout with the rest of the model held still, it
would not do better with everything moving.

PREDICTED, before the run: held-out correlation falls substantially from its installed value
within a few hundred steps, because the MSE gradient is dominated by the unpredictable part of the
target and the head's measured behaviour is to emit 12-19x too much amplitude. If instead it holds
or rises, the objective is not the problem, the optimisation was, and ridge initialisation is the
fix for the paired term.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--paired-stim", required=True)
    ap.add_argument("--paired-neural", required=True)
    ap.add_argument("--sites", type=int, default=150_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--lead-rank", type=int, default=64)
    ap.add_argument("--dyn-steps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=5e-3)
    ap.add_argument("--ctx", type=int, default=125)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--exclude", default="1619681:1627268")
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--gpu-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="out/mse_destroys_ridge.json")
    a = ap.parse_args()

    import pretrain_video_loop as P
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(a.paired_stim, mmap_mode="r")
    Y = np.load(a.paired_neural, mmap_mode="r")
    sc = a.paired_neural.replace("meg_250hz", "meg_scale")
    megsc = np.load(sc) if Path(sc).exists() else None
    n = min(len(X), len(Y)) - 2
    lo = int(n * (1.0 - a.holdout))

    def target(j):
        # the trailing astype is NOT redundant: megsc is float64, so `(y - megsc[0]) / megsc[1]`
        # promotes a float32 y back to float64 and the array silently changes dtype. that cost a
        # run -- `p @ t` raised on Float against Double several steps later, far from the cause.
        y = np.ascontiguousarray(Y[np.asarray(j)]).astype(np.float32)
        out = np.clip((y - megsc[0]) / megsc[1], -6, 6) if megsc is not None else y
        return np.ascontiguousarray(out, dtype=np.float32)

    excl = [tuple(int(v) for v in r.split(":")) for r in a.exclude.split(",") if r.strip()]
    pool = np.arange(a.ctx, lo)
    for x0, x1 in excl:
        pool = pool[~((pool >= x0) & (pool < x1))]
    tr = np.sort(np.random.default_rng(7).choice(pool, size=min(a.n_train, len(pool)), replace=False))
    g = np.random.default_rng(20260911)
    ev = np.sort(np.concatenate([g.integers(lo, n, size=a.batch) for _ in range(a.batches)]))

    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
    pr = P.PairedNeuralLoop(dyn, n_bands=X.shape[-1], n_sensors=Y.shape[-1],
                            lead_rank=a.lead_rank).to(dev)
    sd = torch.load(a.ckpt, map_location=dev)
    dyn.load_state_dict(sd["dyn"]); pr.load_state_dict(sd["paired"])
    for p_ in dyn.parameters():
        p_.requires_grad_(False)          # the dynamics are frozen; only the readout trains
    dyn.eval()

    read_idx = pr.read_idx.to(dev)
    print(f"train {len(tr):,} rows | eval {len(ev):,} rows from [{lo:,}, {n:,})")
    print(f"head reads {len(read_idx):,} sites, lead rank {a.lead_rank}, dynamics FROZEN\n")

    def states(j):
        """The cortical state the head reads, for these rows. No grad: the dynamics are frozen."""
        out = []
        with torch.no_grad():
            for s in range(0, len(j), a.gpu_batch):
                q = j[s:s + a.gpu_batch]
                xp = torch.from_numpy(np.stack([X[t - a.ctx:t] for t in q])).float().to(dev)
                b = xp.shape[0]
                drive = torch.zeros(b, dyn.n, device=dev)
                drive[:, pr.off:pr.off + pr.port] = pr.to_cortex(pr.enc(xp))
                st = dyn.init_state(b, dev); w = dyn.edge_weights()
                for _ in range(a.dyn_steps):
                    st = dyn.step(st, drive, a.dt, w)
                out.append(st[1][:, read_idx])
        return torch.cat(out)

    print("computing cortical states ...", flush=True)
    Str, Sev = states(tr), states(ev)
    Ttr = torch.from_numpy(target(tr)).to(dev)
    Tev = torch.from_numpy(target(ev)).to(dev)
    zero = float((Tev ** 2).mean())
    se = 1.0 / np.sqrt(Tev.numel())

    def corr(Pd, T):
        Pd, T = Pd.float(), T.float()          # never let a dtype mismatch reach the dot product
        p, t = (Pd - Pd.mean(0)).flatten(), (T - T.mean(0)).flatten()
        return float(p @ t / max(p.norm() * t.norm(), torch.tensor(1e-30, device=p.device)))

    # ---- the ridge, fitted RAW so that it IS a bias-free linear map ----------------
    # ALPHA IS CHOSEN ON A VALIDATION SPLIT CARVED FROM THE TRAINING ROWS, never on the
    # evaluation draw. The first version hard-coded alpha=1e4, carried over from a fit on
    # STANDARDISED features -- and on unstandardised features that is a completely different
    # amount of regularisation. It read +0.0132 where the standardised fit reads +0.0400, so the
    # experiment would have started from a readout four times worse than the one it is about to
    # ask MSE to preserve. A constant is not portable across a change of units.
    nval = max(200, len(tr) // 10)
    A64, B64 = Str[:-nval].double(), Ttr[:-nval].double()
    Av, Bv = Str[-nval:].double(), Ttr[-nval:].double()
    G = A64.T @ A64; RHS = A64.T @ B64
    eye = torch.eye(A64.shape[1], device=dev, dtype=torch.float64)
    best = None
    for al in (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4):
        Wa = torch.linalg.solve(G + al * eye, RHS)
        cv = corr((Av @ Wa).float(), Bv.float())
        print(f"  alpha {al:8.0e}  validation corr {cv:+.4f}" + ("   <-- best" if best is None or cv > best[0] else ""))
        if best is None or cv > best[0]:
            best = (cv, al, Wa)
    _, alpha, W = best
    print(f"  chosen alpha {alpha:.0e} on {nval} held-back TRAINING rows\n")

    Pr_full = (Sev.double() @ W).float()
    # W is (read_sites, sensors) and the head computes state @ (lead_v.weight @ lead_u.weight).T,
    # so W.T = lead_v.weight @ lead_u.weight. With W = U diag(S) Vt:
    #   lead_v.weight = Vt[:k].T * S[:k]   (sensors, k)
    #   lead_u.weight = U[:, :k].T         (k, read_sites)
    # The first version had these two swapped and the shape check caught it immediately, which is
    # the good case -- a silently transposable pair would not have.
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(a.lead_rank, S.numel())
    Wk = (U[:, :k] * S[:k]) @ Vt[:k]
    Pr_rank = (Sev.double() @ Wk).float()
    print(f"ridge on the frozen cortical state: full rank corr {corr(Pr_full, Tev):+.4f}, "
          f"rank-{k} {corr(Pr_rank, Tev):+.4f}")

    # ---- install it as the head's own lead field -----------------------------------
    with torch.no_grad():
        pr.lead_u.weight.copy_(U[:, :k].T.contiguous().float())              # (k, read_sites)
        pr.lead_v.weight.copy_((Vt[:k].T * S[:k]).contiguous().float())      # (sensors, k)

    def head_predict(j, S_cached=None):
        St = states(j) if S_cached is None else S_cached
        with torch.no_grad():
            return pr.lead_v(pr.lead_u(St))

    P_installed = head_predict(ev, Sev)
    c_installed = corr(P_installed, Tev)
    err = float((P_installed - Pr_rank).abs().max() / Pr_rank.abs().max())
    print(f"\nKNOWN ANSWER: the head with these weights must reproduce the rank-{k} ridge.")
    print(f"  head {c_installed:+.6f} against ridge {corr(Pr_rank, Tev):+.6f}   "
          f"max relative difference {err:.2e}   "
          f"{'PASS' if err < 1e-4 else 'FAIL -- the installation is wrong'}")
    if err >= 1e-4:
        sys.exit("known answer FAILED; nothing after this is interpretable")

    def evaluate():
        with torch.no_grad():
            Pd = pr.lead_v(pr.lead_u(Sev))
            mse = float(((Pd - Tev) ** 2).mean())
            return dict(correlation=corr(Pd, Tev), skill_vs_zero=1.0 - mse / zero,
                        amplitude_ratio=float(Pd.std() / Tev.std()))

    hist = [dict(step=0, **evaluate())]
    print(f"\nstep 0 (installed): corr {hist[0]['correlation']:+.4f}  "
          f"skill {hist[0]['skill_vs_zero']:+.4f}  amplitude {hist[0]['amplitude_ratio']:.2f}x")
    print(f"\nnow training the READOUT ONLY with the ordinary MSE objective, "
          f"{a.steps} steps at lr {a.lr}:\n")

    opt = torch.optim.AdamW([pr.lead_u.weight, pr.lead_v.weight], lr=a.lr, weight_decay=1e-4)
    rng = np.random.default_rng(3)
    for step in range(1, a.steps + 1):
        idx = rng.integers(0, len(tr), size=a.batch)
        s_b, t_b = Str[idx], Ttr[idx]
        loss = torch.nn.functional.mse_loss(pr.lead_v(pr.lead_u(s_b)), t_b)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            e = evaluate(); hist.append(dict(step=step, **e))
            print(f"  step {step:4d}  train mse {float(loss):.4e}   held-out corr "
                  f"{e['correlation']:+.4f}  skill {e['skill_vs_zero']:+.4f}  "
                  f"amplitude {e['amplitude_ratio']:.2f}x")

    c0, c1 = hist[0]["correlation"], hist[-1]["correlation"]
    destroyed = c1 < c0 - 5 * se
    print(f"\n  installed {c0:+.4f}  ->  after {a.steps} steps {c1:+.4f}   "
          f"({(c1-c0)/se:+.1f} sd)")
    print("\n  VERDICT: " + (
        f"MSE DESTROYS the solution. A head placed ON a readout worth {c0:+.4f} is driven off it\n"
        f"  to {c1:+.4f} by its own training objective, with the dynamics frozen and only the\n"
        f"  readout moving -- the most favourable case there is. Initialisation cannot fix this;\n"
        f"  the objective has to change." if destroyed else
        f"MSE does NOT destroy the solution -- it held at {c1:+.4f} from {c0:+.4f}. The objective\n"
        f"  is not what loses the signal; the optimisation never reached this point from a random\n"
        f"  start. Ridge-initialising the readout is then the fix for the paired term."))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(
        alpha=float(alpha), lead_rank=k, n_train=len(tr), n_eval=len(ev), steps=a.steps, lr=a.lr,
        standard_error=float(se), install_relative_error=err,
        ridge_full_rank=corr(Pr_full, Tev), ridge_rank_k=corr(Pr_rank, Tev),
        history=hist, mse_destroys_solution=bool(destroyed)), indent=2) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
