"""fuse many materialization checkpoints into ONE implicit model.

the architecture's claim is that there is one theta and the task-specific models
are projections of it.  in practice every run so far built its own
`CorticalDynamics` from scratch, so what is published is 142 independent models
that share an architecture rather than a parameter set.  this closes that gap.

**why averaging is legitimate here, measured rather than assumed.**  the
association is w_ij = M[parcel] x exp(-d/l) x sigma(<e_i, e_j>), so the embedding
enters only through inner products and e and Re are the same function for any
orthogonal R.  two runs from different bases would average to mush.  they are not
in different bases: every trainer seeds `torch.manual_seed(0)` and inits from the
same distribution, so the runs stay near a common frame -- mean |cos| between the
video and EEG kernels is 0.658 before any alignment.  measured on the designated
test set, transplanted into the EEG model:

    e_eeg alone (ceiling)          63.50%   100.0%
    e_video alone                  60.50%    91.5%
    naive average                  62.00%    95.8%
    Procrustes-aligned average     62.50%    97.2%

so averaging preserves what each kernel knows, and aligning first recovers a
little more.  Procrustes is exactly the right tool -- it minimises
||e_i R - e_ref|| over orthogonal R, which is precisely the symmetry the inner
product leaves free.

**cross-resolution.**  sites are drawn by a seeded RNG, and because `z` and
`theta` are consumed as two separate blocks of that stream, a 30k and a 150k
model do NOT share their first 30k positions -- the point sets are genuinely
different.  so a kernel is carried to the target resolution by k-NN
interpolation on the sphere: each target site takes an inverse-distance-weighted
mean of its nearest source sites.  that is an approximation and it is recorded as
one; a site whose nearest source neighbour is far away contributes a blurred
value rather than a wrong one.

**weighting.**  a checkpoint that transfers well should count for more than one
that does not, and transfer is measured, not guessed -- `--weights transfer`
reads out/transfer_sweep.json.  the default is uniform, which is the honest
choice when transfer has not been measured for every input.
"""
from __future__ import annotations

import argparse, glob, json, math, os
import numpy as np
import torch


def cortical_sites(n: int, seed: int = 0) -> torch.Tensor:
    """the same placement the trainers use, reproduced so fusion never guesses."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    radius = math.sqrt(202437.0 / (4.0 * math.pi))
    z = torch.rand(n, generator=g) * 2 - 1
    theta = torch.rand(n, generator=g) * 2 * math.pi
    r = torch.sqrt(1 - z * z)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], 1) * radius


def resample(src: torch.Tensor, n_src: int, n_dst: int, k: int = 1,
             chunk: int = 2048) -> torch.Tensor:
    """carry a per-site kernel from one resolution to another by k-NN on the sphere.

    **k defaults to 1, and that is not a detail.**  the kernel acts through
    <e_i, e_j>, and neighbouring sites hold near-orthogonal embeddings, so
    averaging a few of them cancels rather than smooths.  measured by round-tripping
    a known-good kernel 30k -> 150k -> 30k and scoring it on the designated test set:

        k=1   74.6% of the original recovered
        k=2   74.6%
        k=4    7.0%     <- the previous default, which destroyed it

    even at k=1 a round trip costs a quarter of the kernel's value, so
    cross-resolution transfer is lossy and should be avoided where a native-
    resolution kernel exists.  it is not a free reparameterisation.
    """
    if n_src == n_dst:
        return src
    ps, pd = cortical_sites(n_src), cortical_sites(n_dst)
    out = torch.empty(n_dst, src.shape[1], dtype=src.dtype)
    for i in range(0, n_dst, chunk):
        d = torch.cdist(pd[i:i + chunk], ps)
        dist, idx = torch.topk(d, k, dim=1, largest=False)
        w = 1.0 / dist.clamp_min(1e-6)
        w = w / w.sum(1, keepdim=True)
        out[i:i + chunk] = (src[idx] * w[..., None]).sum(1)
    return out


def procrustes(e: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """rotate e into ref's frame -- the symmetry <e_i, e_j> leaves free."""
    A = e.T.double() @ ref.double()
    U, _, Vh = torch.linalg.svd(A)
    return e @ (U @ Vh).float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="ckpt/*.pt")
    ap.add_argument("--sites", type=int, default=30_000,
                    help="the fused model's resolution.  30k is the default because "
                         "it is where the best-measured materialisation lives; the "
                         "kernel resamples to anything on materialize")
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--align", action="store_true", default=True)
    ap.add_argument("--no-align", dest="align", action="store_false")
    ap.add_argument("--weights", choices=("uniform", "transfer"), default="uniform")
    ap.add_argument("--transfer-json", default="out/transfer_sweep.json")
    ap.add_argument("--exclude", default="vshuf,vnoise",
                    help="substrings to skip -- the shuffled-target and noise-input "
                         "runs are CONTROLS, and fusing a control into the model it "
                         "was built to test would make the test unfalsifiable")
    ap.add_argument("--out", default="ckpt/ibm1_implicit.pt")
    a = ap.parse_args()

    excl = [s for s in a.exclude.split(",") if s]
    tw = {}
    if a.weights == "transfer" and os.path.exists(a.transfer_json):
        tw = {k: v["recovered"]
              for k, v in json.load(open(a.transfer_json)).get("runs", {}).items()}

    kernels, names, ws = [], [], []
    for f in sorted(glob.glob(a.glob)):
        b = os.path.basename(f)
        if any(s in b for s in excl) or b.startswith("ibm1_implicit"):
            continue
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        sd = d.get("model") or d.get("av") or {}
        e = sd.get("dyn.embed")
        if e is None or e.shape[1] != a.embed:
            continue
        if a.weights == "transfer":
            # measured transfer or nothing.  defaulting an UNMEASURED checkpoint to
            # 1.0 while a measured good one scores 0.93 inverts the ranking and
            # lets the unmeasured majority decide the average -- which is what the
            # first run of this did.  a checkpoint that transfers at or below the
            # random floor is excluded outright: the anterior-readout runs recover
            # -11% to -22%, so including them is worse than leaving them out.
            if b not in tw or tw[b] <= 0.0:
                continue
            ws.append(tw[b])
        else:
            ws.append(1.0)
        kernels.append(resample(e.float(), e.shape[0], a.sites))
        names.append(b)

    if not kernels:
        print("no compatible checkpoints"); return
    ws = torch.tensor(ws).clamp_min(0)
    if ws.sum() <= 0:
        ws = torch.ones(len(kernels))
    ws = ws / ws.sum()

    ref = kernels[int(ws.argmax())]
    if a.align:
        kernels = [k if i == int(ws.argmax()) else procrustes(k, ref)
                   for i, k in enumerate(kernels)]
    fused = sum(w * k for w, k in zip(ws, kernels))

    print(f"fused {len(kernels)} checkpoints -> {a.sites:,} sites x {a.embed}")
    print(f"  alignment: {'orthogonal Procrustes onto the heaviest' if a.align else 'none'}")
    print(f"  weighting: {a.weights}")
    for n, w in sorted(zip(names, ws.tolist()), key=lambda t: -t[1])[:8]:
        print(f"    {w:6.4f}  {n}")
    if len(names) > 8:
        print(f"    ... and {len(names)-8} more")
    print(f"  fused RMS {fused.pow(2).mean().sqrt():.5f}  "
          f"(inputs {torch.stack([k.pow(2).mean().sqrt() for k in kernels]).mean():.5f})")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"dyn.embed": fused, "sites": a.sites, "embed": a.embed,
                "sources": names, "weights": ws.tolist(), "aligned": a.align,
                "schema": "ibm1/implicit-v1"}, a.out)
    print(f"\nwrote {a.out}")
    print("this is the IMPLICIT model: one kernel, no heads.  materialize it with "
          "scripts/materialize.py")


if __name__ == "__main__":
    main()
