"""materialize a task-specific model from the implicit one.

`ckpt/ibm1_implicit.pt` is the implicit model: ONE association kernel over the
cortical sheet, no heads.  every materialisation this programme trains --
image->EEG retrieval, cochleagram->MEG, video continuation, the evoked forward
model -- is that kernel plus a small task head, and the heads are the cheap part
(at 250k sites the shared substrate was 32.0M against heads of 9.9M and 3.0M).

so a developer should not download five models.  they should download the kernel
and materialize what they need.  that is what this does:

    python scripts/materialize.py --target visual_eeg --sites 30000

the kernel RESAMPLES to whatever resolution is asked for, by k-NN on the sphere,
because sites are drawn from a seeded stream whose point sets differ between
resolutions -- a 30k and a 150k model do not share positions.  asking for a
resolution the kernel was not fused at is therefore an approximation, and it is
reported as one rather than performed silently.

**what a fresh materialisation is and is not.**  the kernel carries the learned
cortico-cortical connectivity, which is the part that transfers: measured on the
designated THINGS-EEG2 test set, transplanting the fused kernel into a trained
EEG model recovers 88.7% of what that model's own kernel is worth, against 0% for
a random kernel.  the HEAD is not carried and cannot be -- it is task-specific by
construction -- so a materialised model is initialised, not trained.  it needs its
head fitted before it predicts anything.
"""
from __future__ import annotations

import argparse, importlib.util, json, os
import torch

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)
fu = importlib.util.spec_from_file_location(
    "fuse", os.path.join(os.path.dirname(__file__), "fuse_implicit.py"))
F = importlib.util.module_from_spec(fu)
fu.loader.exec_module(F)

TARGETS = {
    "visual_eeg":  ("VisualContrastiveLoop", "image -> cortex -> embedding, aligned "
                    "with 64-channel EEG.  63.5% top-1 on the THINGS-EEG2 "
                    "designated test set, 127x chance"),
    "audio_meg":   ("AudioContrastiveLoop", "cochleagram -> cortex -> embedding, "
                    "aligned with 306-channel MEG.  6.75% top-1, 13.5x chance, "
                    "against a dynamics-free ceiling of 7.12%"),
    "video":       ("VideoLoop", "frame t -> cortex -> frame t+H.  best skill "
                    "-0.149 against persistence; this materialisation does not "
                    "beat its trivial baseline and is published as such"),
    "evoked":      ("VisualEvokedLoop", "image -> cortex -> the evoked response as "
                    "a time series, read through a rank-limited lead field"),
    "video_via_eeg": ("VideoViaEEGLoop", "frame -> cortex -> sensor projection -> "
                      "next frame, with the decoder seeing ONLY the projection"),
}


def load_implicit(path: str, sites: int, embed: int):
    d = torch.load(path, map_location="cpu")
    if d.get("schema") != "ibm1/implicit-v1":
        raise ValueError(f"{path} is not an implicit model (schema {d.get('schema')})")
    e = d["dyn.embed"]
    if e.shape[1] != embed:
        raise ValueError(f"kernel embed dim {e.shape[1]} != requested {embed}")
    if e.shape[0] != sites:
        print(f"resampling kernel {e.shape[0]:,} -> {sites:,} sites by k-NN on the "
              f"sphere.  the point sets differ between resolutions, so this is an "
              f"approximation", flush=True)
        e = F.resample(e, e.shape[0], sites)
    return e, d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--implicit", default="ckpt/ibm1_implicit.pt")
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cls_name, what = TARGETS[a.target]
    e, meta = load_implicit(a.implicit, a.sites, a.embed)
    dev = "cpu"
    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range)
    dyn.embed.data.copy_(e)

    cls = getattr(P, cls_name)
    model = cls(dyn)
    n_tot = sum(p.numel() for p in model.parameters())
    n_ker = dyn.embed.numel()
    print(f"\nmaterialized `{a.target}` -> {cls_name}")
    print(f"  {what}")
    print(f"  sites {a.sites:,}  embed {a.embed}  degree {a.k}")
    print(f"  parameters {n_tot:,}  of which kernel {n_ker:,} "
          f"({100*n_ker/n_tot:.0f}%)")
    print(f"  kernel fused from {len(meta['sources'])} checkpoints"
          f"{', Procrustes-aligned' if meta.get('aligned') else ''}")
    print(f"\n  THE HEAD IS UNTRAINED.  the kernel carries the learned "
          f"cortico-cortical\n  connectivity; the task head is task-specific and "
          f"has to be fitted.")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "target": a.target,
                    "config": {"sites": a.sites, "embed": a.embed, "k": a.k,
                               "long_range": a.long_range},
                    "implicit_sources": meta["sources"]}, a.out)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
