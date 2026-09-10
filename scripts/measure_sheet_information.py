"""why does the same sheet help retrieval and hurt regression?

measured, on real surface geometry, this is not a small difference:

  visual retrieval, disjoint readout   bypassing the dynamics retains 11.7%
  proprioceptive regression            removing the dynamics is 15.7% BETTER

same substrate, opposite sign.  and it is not amplitude -- concentration, which
buys 209x transport, changes neither.

the hypothesis this measures: a lossy nonlinear map can preserve NEIGHBOURHOOD
STRUCTURE while destroying LINEAR RECOVERABILITY, and the two tasks ask for
different ones.  retrieval needs only to know which candidate is nearest, so a
transform that scrambles values while keeping neighbours intact costs nothing and
its nonlinearity may even help.  regression needs the values back, and every bit
of that loss is error in the answer.

so drive the sheet with real proprioceptive afference and ask two questions of
the same readout:

  linear recoverability   R^2 of reconstructing the INPUT from the readout by
      least squares.  this is what a regression head has to work with.
  neighbourhood retention what fraction of each point's k nearest neighbours in
      the input are still among its k nearest in the readout.  this is what a
      retrieval head has to work with.

if retention stays high while R^2 collapses, the puzzle is answered and the
prescription follows: read the sheet with a metric head, not a regression head.

RESULT: THE HYPOTHESIS IS REFUSED, AND THE FIRST RUN OF IT WAS WRONG.

  representation                  linear R^2  neighbours kept   common mode
  identity (the input itself)         1.0000           1.0000          0.0x
  random projection                   1.0000           0.9800          0.0x
  the sheet, severed                  0.0000           0.0061     173667.4x
  the sheet, intact                   0.9998           0.8952       3499.6x

Both survive.  The sheet is close to an invertible change of coordinates on this
afference -- and separately measured, it *improves* the condition number by 435x
(1.95e+08 -> 4.48e+05).  So it discards essentially nothing, and none of the three
candidate explanations (information loss, metric distortion, ill-conditioning)
accounts for the regression deficit.

The first run reported retention 0.124 and I wrote "structure is destroyed"
before noticing that near-invertibility and destroyed structure cannot both be
true.  `retention` normalised WITHOUT centring, and the readout carries a common
mode 3,500x its own per-sample spread.  See the docstring on `retention`.

What the common mode DOES explain is trainability, which is a different claim and
the one that survived: a freshly initialised Linear reading this sheet sees a
constant of magnitude 8.24 with the signal at 0.0024 riding on it, while the
no_cortex control's head reads a normalised encoder output.  The two arms were
never matched on that.  `train_proprioceptive_motor.py --readout-norm batchnorm`
is the test.

three controls, because a readout can look good for reasons that are not the
sheet:

  identity      the input itself, mapped through nothing.  R^2 must be 1.000 and
      retention must be 1.000, or the metric is broken before it is applied.
  random        a fixed random projection to the same width as the readout.  a
      random map preserves distances well (Johnson-Lindenstrauss) and preserves
      linear structure perfectly, so it separates "the sheet is doing something"
      from "any projection would".
  severed       the sheet with its association zeroed.
"""
from __future__ import annotations

import argparse, importlib.util, json, os
import numpy as np
import torch

HERE = os.path.dirname(__file__)
sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(HERE, "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)

CORPUS = "/home/brandonin/Documents/IHM-1/data/derived/pose-corpus"


def afference(corpus: str, n: int, seed: int):
    idx = json.load(open(f"{corpus}/index.json"))
    names = [m["id"] for m in idx["motions"]]
    sets = {}
    for nm in names:
        f = json.load(open(f"{corpus}/{nm}.json"))
        if f["frames"]:
            sets[nm] = frozenset(f["frames"][0]["path_over_optimal_fiber"])
    big = max(sets.values(), key=len)
    keep = [nm for nm in names if big <= sets[nm]]
    mus = sorted(big)
    X = []
    for nm in keep:
        for fr in json.load(open(f"{corpus}/{nm}.json"))["frames"]:
            X.append([fr["path_over_optimal_fiber"][m] for m in mus]
                     + [fr["rate_per_s"].get(m, 0.0) for m in mus])
    X = np.asarray(X, np.float32)
    rng = np.random.default_rng(seed)
    if len(X) > n:
        X = X[rng.choice(len(X), n, replace=False)]
    X = (X - X.mean(0)) / X.std(0).clip(1e-6)
    return X, mus


def r2_linear(A: np.ndarray, B: np.ndarray) -> float:
    """R^2 of predicting A from B by least squares, held out on a split half."""
    n = len(A) // 2
    Btr = np.concatenate([B[:n], np.ones((n, 1), np.float32)], 1)
    Bte = np.concatenate([B[n:], np.ones((len(B) - n, 1), np.float32)], 1)
    W = np.linalg.lstsq(Btr, A[:n], rcond=None)[0]
    resid = ((A[n:] - Bte @ W) ** 2).sum()
    total = ((A[n:] - A[:n].mean(0)) ** 2).sum()
    return float(1 - resid / total)


def retention(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """fraction of each point's k nearest neighbours in A still nearest in B.

    CENTRE BEFORE NORMALISING.  the first run of this measurement did not, and
    it reported 0.124 where the answer is 0.895 -- a sevenfold error that pointed
    at exactly the wrong conclusion.  the sheet's readout carries a common mode
    ~3,500x larger than the per-sample variation (mean magnitude 8.24, spread
    0.0024), so cosine similarity on the raw readout is dominated by an offset
    every sample shares.  every point looks like every other point, the ranking
    among them is noise, and the metric reports destroyed structure for a
    representation whose structure is intact.

    a representation is free to sit anywhere in space.  what a retrieval head
    reads is the arrangement of points relative to each other, which is what
    remains after the shared offset is removed.  the identity control does NOT
    catch this: the input is already standardised, so centring is a no-op on it
    and the gate passes at 1.000 either way.
    """
    def nn(M):
        M = M - M.mean(0, keepdims=True)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        S = M @ M.T
        np.fill_diagonal(S, -np.inf)
        return np.argpartition(-S, k, axis=1)[:, :k]
    a, b = nn(A), nn(B)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


def common_mode(B: np.ndarray) -> float:
    """ratio of the shared offset to the per-sample variation about it.

    reported alongside every representation because a large value is what makes
    an uncentred distance metric meaningless, and it is invisible in the
    retention number itself.
    """
    spread = float(np.abs(B.std(0)).mean())
    return float(np.abs(B.mean(0)).mean() / max(spread, 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/wiring_surface_slice.pt")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--drive-region", default="postcentral")
    ap.add_argument("--read-region", default="precentral")
    ap.add_argument("--dyn-steps", type=int, default=4)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=2e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/sheet_information.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, mus = afference(a.corpus, a.n, a.seed)
    print(f"{len(X):,} afference vectors, {X.shape[1]} channels "
          f"(length + rate for {len(mus)} muscles)", flush=True)

    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = d.get("model", d)
    if "dyn.idx" not in sd:
        for h in (d.get("heads") or {}).values():
            if isinstance(h, dict) and "dyn.idx" in h:
                sd = h; break
    emb = sd["dyn.embed"]; n_sites, e_dim = emb.shape; k_edges = sd["dyn.idx"].shape[1]

    torch.manual_seed(a.seed)
    dyn = P.CorticalDynamics(n_sites, e_dim, k_edges, dev, geometry="surface",
                             long_topology="tract").to(dev)
    with torch.no_grad():
        dyn.embed.copy_(emb.to(dev))
        for nm in ("idx", "geo", "pos", "w_ee", "w_ei", "w_assoc", "a_gain", "log_len"):
            if f"dyn.{nm}" in sd:
                getattr(dyn, nm).copy_(sd[f"dyn.{nm}"].to(dev))
    port = P.region_index(dyn.pos, a.drive_region).to(dev)
    read = P.region_index(dyn.pos, a.read_region).to(dev)
    print(f"drive {a.drive_region} ({len(port)}) -> read {a.read_region} "
          f"({len(read)}), disjoint\n", flush=True)

    torch.manual_seed(a.seed)
    proj = torch.randn(X.shape[1], len(port), device=dev) / X.shape[1] ** 0.5
    Xt = torch.from_numpy(X).to(dev)
    geo0 = dyn.geo.clone()

    @torch.no_grad()
    def through_sheet(sever: bool):
        dyn.geo.copy_(geo0)
        if sever:
            dyn.geo.zero_()
        out = []
        for i in range(0, len(Xt), 256):
            x = Xt[i:i + 256]
            s = dyn.init_state(len(x), dev)
            w = dyn.edge_weights()
            drive = torch.zeros(len(x), dyn.n, device=dev).index_copy(1, port, x @ proj)
            h = a.dt / a.substeps
            for _ in range(a.dyn_steps):
                for _ in range(a.substeps):
                    s = dyn.step(s, drive, h, w)
            out.append(s[1][:, read].cpu().numpy())
        dyn.geo.copy_(geo0)
        return np.concatenate(out).astype(np.float32)

    torch.manual_seed(a.seed + 1)
    rand_proj = (Xt @ torch.randn(X.shape[1], len(read), device=dev)
                 / X.shape[1] ** 0.5).cpu().numpy().astype(np.float32)

    reps = {
        "identity (the input itself)": X,
        "random projection": rand_proj,
        "the sheet, severed": through_sheet(True),
        "the sheet, intact": through_sheet(False),
    }

    print(f"  {'representation':30s} {'linear R^2':>11s} {'neighbours kept':>16s}"
          f" {'common mode':>13s}", flush=True)
    res = {"n": len(X), "k": a.k, "channels": X.shape[1], "reps": {}}
    for nm, R in reps.items():
        r2 = r2_linear(X, R)
        ret = retention(X, R, a.k)
        cm = common_mode(R)
        res["reps"][nm] = {"linear_r2": r2, "neighbour_retention": ret,
                           "common_mode_ratio": cm}
        print(f"  {nm:30s} {r2:11.4f} {ret:16.4f} {cm:12.1f}x", flush=True)

    idn = res["reps"]["identity (the input itself)"]
    if abs(idn["linear_r2"] - 1) > 1e-3 or abs(idn["neighbour_retention"] - 1) > 1e-9:
        print("\n  GATE FAILED: the identity does not score 1.000 on both. "
              "the metric is wrong before it was applied to anything.", flush=True)
    else:
        print("\n  gate: identity scores 1.000 on both, as it must", flush=True)

    sheet = res["reps"]["the sheet, intact"]
    print(f"\n  the sheet keeps {100*sheet['neighbour_retention']:.1f}% of "
          f"neighbours while linear recoverability is {sheet['linear_r2']:.3f}",
          flush=True)
    if sheet["neighbour_retention"] > 0.5 > sheet["linear_r2"]:
        print("  -> STRUCTURE SURVIVES, VALUES DO NOT.  that is why the same sheet "
              "helps retrieval and hurts regression, and it says read it with a "
              "metric head rather than a regression head.", flush=True)
    elif sheet["linear_r2"] > 0.5 and sheet["neighbour_retention"] > 0.5:
        print("  -> BOTH SURVIVE.  the sheet is close to an invertible change of "
              "coordinates on this afference, so neither the retrieval advantage "
              "nor the regression deficit is explained by what it discards.  the "
              "hypothesis this script was written to test is refused.", flush=True)
    elif sheet["linear_r2"] > 0.5:
        print("  -> values largely survive; the regression deficit is NOT explained "
              "by the sheet destroying linear structure.", flush=True)
    else:
        print("  -> neither survives well; the sheet is lossy for both and the "
              "retrieval result needs another explanation.", flush=True)

    print(f"\n  common mode: the sheet readout sits "
          f"{sheet['common_mode_ratio']:.0f}x its own per-sample spread away from "
          f"the origin.  any distance metric applied to it WITHOUT centring "
          f"measures that offset and not the representation.", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
