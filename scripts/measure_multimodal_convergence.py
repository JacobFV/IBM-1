"""do sight, hearing and touch converge anywhere the motor strip can read?

the somato-motor claim the programme wants to make is that one sheet integrates
vision, audition and somatosensation into a motor command.  every ingredient for
that exists: an optic nerve into occipital cortex at 43x chance, a cochlear nerve
into temporal at 7.5x, a declared postcentral port for body afference, and a
precentral port IHM-1 reads descending commands from.

what has never been measured is whether the three ever MEET.  they enter at three
disjoint regions and the command leaves from a fourth, so integration is entirely
a claim about transport across the sheet -- and transport is exactly what the
propagation result says is missing (0.03-0.07% of a drive reaches a disjoint
region, scaling linearly with association gain rather than compounding).

so this script does not train anything.  it measures, on a trained checkpoint,
how much of the precentral state each modality actually explains, and it is built
so the answer cannot be flattered:

*drop-one attribution*  -- drive all three, then re-run with one silenced, and
    attribute to that modality the variance of the difference at precentral.
    this is an ABLATION, not a magnitude.  CLAUDE.md is emphatic about the
    difference and this project has been burned by it: a weight 4.4x the random
    baseline cost -0.06% to sever.

*the readout is precentral ONLY*  -- the single most expensive error in the motor
    work was a whole-sheet readout that sampled the driven region directly and
    reported 3,134x the variance of everything else as if it were transport.  a
    readout that can see the port it is measuring transport FROM measures nothing.

*two controls that must print known answers* -- driving nothing must give exactly
    zero at precentral, and reading out AT a driven port must recover essentially
    all of that port's variance.  a metric that cannot pass a case whose answer
    is known is not evidence about a case whose answer is not.

the number to read is `transport`: precentral variance from a modality divided by
that modality's variance at its own entry port.  1.0 would mean the motor strip
sees the modality as well as the cortex that receives it.  the propagation result
predicts ~1e-3 or worse, and if that is what prints, the honest conclusion is
that the somato-motor materialization is not blocked on training data or on the
loss -- it is blocked on the sheet not conducting.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np
import torch

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)

#: modality -> the cortex it enters.  these are the ports the rest of the
#: programme already uses, not new ones invented here: the optic nerve loop
#: drives occipital, the cochlear loop drives temporal, and IHM-1 routes body
#: afference to brain-{l,r}h-postcentral and reads commands from precentral.
ENTRY = {"sight": "occipital", "hearing": "temporal", "touch": "postcentral"}
READOUT = "precentral"


def _load_kernel(ckpt: str, device):
    """rebuild CorticalDynamics from whichever checkpoint shape this is.

    three shapes exist in ckpt/ and a reader that understands one of them
    silently skips the others -- transfer_sweep.py did exactly that and left the
    curriculum run out of its own evaluation.  so fail loudly on an unknown
    shape rather than returning an untrained kernel that looks trained.
    """
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("model", "kernel"):
        if key in d and isinstance(d[key], dict) and "dyn.embed" in d[key]:
            sd, embed = d[key], d[key]["dyn.embed"]
            break
    else:
        if "dyn.embed" in d:
            sd, embed = d, d["dyn.embed"]
        else:
            raise KeyError(
                f"{ckpt}: no 'dyn.embed' at top level or under model/kernel; "
                f"keys are {sorted(k for k in d if not k.startswith('_'))}")
    n_sites, embed_dim = embed.shape
    # THE GRAPH IS NOT AT THE TOP LEVEL of an implicit-v1 checkpoint.
    #
    # that format saves {"dyn.embed", "sites", "embed", "heads", ...} and the
    # advertised artifact is the bare embedding.  but 12 of every site's 48 edges
    # are long-range partners drawn with `torch.randint` from the GLOBAL rng at
    # construction, and they are not a function of the seed alone -- the same
    # seed on cpu and on cuda draws different partners (measured: coincidence
    # 0.00062 against a chance of 0.00050, i.e. unrelated).
    #
    # so a reader that constructs CorticalDynamics fresh and copies only
    # dyn.embed is evaluating learned weights on a topology those weights never
    # saw, silently discarding a quarter of the connectivity.  the learned factor
    # is sigma(<e_i, e_j>) over specific (i, j) pairs; redraw the pairs and every
    # long-range weight is meaningless.
    #
    # the graph IS in the file, inside any head's state_dict, because each head
    # holds `dyn` as a submodule.  read it from there rather than redrawing.
    idx = sd.get("dyn.idx", d.get("dyn.idx"))
    src = "top level"
    if idx is None and isinstance(d.get("heads"), dict):
        for name, head in d["heads"].items():
            if isinstance(head, dict) and "dyn.idx" in head:
                idx, sd, src = head["dyn.idx"], head, f"heads[{name}]"
                break
    if idx is None:
        raise KeyError(
            f"{ckpt}: no 'dyn.idx' at top level or in any head. the long-range "
            f"graph cannot be recovered and constructing one would randomise 25% "
            f"of the connectivity -- refusing rather than reporting a number "
            f"measured on the wrong topology.")
    print(f"  graph recovered from {src}", flush=True)
    k = idx.shape[1]
    dyn = P.CorticalDynamics(n_sites, embed_dim, k, device).to(device)
    # the graph is part of the trained object: the long-range partners were drawn
    # once at construction and the embeddings were learned against THAT graph.
    # rebuilding with a fresh draw would silently evaluate the learned weights on
    # a different topology.
    with torch.no_grad():
        dyn.embed.copy_(embed.to(device))
        dyn.idx.copy_(idx.to(device))
        for name in ("geo", "pos"):
            src = sd.get(f"dyn.{name}", d.get(f"dyn.{name}"))
            if src is not None:
                getattr(dyn, name).copy_(src.to(device))
        for name in ("w_ee", "w_ei", "w_assoc", "a_gain", "log_len"):
            src = sd.get(f"dyn.{name}", d.get(f"dyn.{name}"))
            if src is not None:
                getattr(dyn, name).copy_(src.to(device))
    dyn.eval()
    return dyn, d.get("step", -1)


@torch.no_grad()
def run(dyn, drives: dict, ports: dict, read: torch.Tensor, *, batch: int,
        n_steps: int, substeps: int, dt: float, device) -> torch.Tensor:
    """settle the sheet under a set of per-modality drives; return precentral rates."""
    s = dyn.init_state(batch, device)
    w = dyn.edge_weights()
    h = dt / substeps
    drive = torch.zeros(batch, dyn.n, device=device)
    for name, v in drives.items():
        drive[:, ports[name]] += v
    for _ in range(n_steps):
        for _ in range(substeps):
            s = dyn.step(s, drive, h, w)
    return s[1][:, read]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/ibm1_curriculum16.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dyn-steps", type=int, default=32,
                    help="a drive that never crosses the sheet gives every arm "
                         "the same answer to 6 decimals; 2 was not enough")
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    # the anisotropy knobs, so this metric can be run against a modified sheet
    # without a second copy of it.  defaults reproduce the kernel as trained;
    # see CorticalDynamics.__init__ for what they mean and why.
    ap.add_argument("--tanh-slope", type=float, default=None)
    ap.add_argument("--long-gain", type=float, default=None)
    ap.add_argument("--long-topm", type=int, default=None)
    ap.add_argument("--out", default="out/multimodal_convergence.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    dyn, step = _load_kernel(a.ckpt, dev)
    for attr, val in (("tanh_slope", a.tanh_slope), ("long_gain", a.long_gain),
                      ("long_topm", a.long_topm)):
        if val is not None:
            setattr(dyn, attr, type(getattr(dyn, attr))(val))
    print(f"{a.ckpt}: step {step}, {dyn.n} sites, k={dyn.k}, "
          f"{dyn.n_far} long-range edges per site", flush=True)
    print(f"  kernel: tanh_slope={dyn.tanh_slope} long_gain={dyn.long_gain} "
          f"long_topm={dyn.long_topm}", flush=True)
    _w = dyn.edge_weights()
    print(f"  |w| mean {float(_w.abs().mean()):.4e}  "
          f"row L1 mean {float(_w.abs().sum(-1).mean()):.4e}  "
          f"row L1/max mean {float((_w.abs().sum(-1)/_w.abs().max(-1).values).mean()):.2f}"
          f"  (48 = perfectly flat, 1 = one edge carries the row)", flush=True)

    ports = {m: P.region_index(dyn.pos, lobe).to(dev) for m, lobe in ENTRY.items()}
    read = P.region_index(dyn.pos, READOUT).to(dev)
    for m, lobe in ENTRY.items():
        print(f"  {m:8s} -> {lobe:12s} {len(ports[m]):5d} sites", flush=True)
    print(f"  readout  <- {READOUT:12s} {len(read):5d} sites", flush=True)
    if len(read) == 0:
        raise SystemExit(f"{READOUT} is empty at {dyn.n} sites -- nothing to read")
    for m, p in ports.items():
        overlap = len(np.intersect1d(p.cpu().numpy(), read.cpu().numpy()))
        if overlap:
            raise SystemExit(
                f"{m}'s port overlaps the {READOUT} readout in {overlap} sites. "
                f"transport measured through an overlapping readout is not "
                f"transport -- it is the drive read back.")

    # independent random drives, one per modality.  random rather than real
    # stimuli on purpose: this asks what the SHEET conducts, and a real stimulus
    # would confound conduction with whatever structure the encoders impose.
    g = torch.Generator(device=dev).manual_seed(a.seed)
    sig = {m: a.amp * torch.randn(a.batch, len(ports[m]), device=dev, generator=g)
           for m in ENTRY}

    kw = dict(batch=a.batch, n_steps=a.dyn_steps, substeps=a.substeps,
              dt=a.dt, device=dev)
    full = run(dyn, sig, ports, read, **kw)

    # ---- control 1: no drive at all must give exactly zero variance ---------
    quiet = run(dyn, {}, ports, read, **kw)
    v_quiet = float(quiet.var(0).mean())
    print(f"\n  [control] undriven precentral variance {v_quiet:.3e}"
          f"   (must be 0; it is the same state every trial)", flush=True)
    if v_quiet > 1e-12:
        print("  WARNING: the undriven sheet is not identical across the batch. "
              "every attribution below is contaminated by that.", flush=True)

    # ---- control 2: read AT a driven port; must recover ~all of its variance -
    at_port = run(dyn, {"sight": sig["sight"]}, ports, ports["sight"], **kw)
    at_port0 = run(dyn, {}, ports, ports["sight"], **kw)
    v_at = float((at_port - at_port0).var(0).mean())
    print(f"  [control] variance AT the driven occipital port {v_at:.3e}"
          f"   (the scale transport is measured against)", flush=True)
    if v_at <= 0:
        raise SystemExit("driving a port produced no variance at that port -- "
                         "the drive is not reaching the sheet at all")

    # ---- drop-one attribution ----------------------------------------------
    print(f"\n  {'modality':10s} {'at own port':>13s} {'at precentral':>15s} "
          f"{'transport':>12s}", flush=True)
    res = {"ckpt": a.ckpt, "step": step, "sites": dyn.n,
           "tanh_slope": dyn.tanh_slope, "long_gain": dyn.long_gain,
           "long_topm": dyn.long_topm,
           "undriven_var": v_quiet, "modalities": {}}
    for m in ENTRY:
        held = {k: v for k, v in sig.items() if k != m}
        without = run(dyn, held, ports, read, **kw)
        # the variance this modality ADDS at precentral, over trials
        v_pre = float((full - without).var(0).mean())
        # the same modality's variance at its own entry port, as the denominator
        own = run(dyn, {m: sig[m]}, ports, ports[m], **kw)
        own0 = run(dyn, {}, ports, ports[m], **kw)
        v_own = float((own - own0).var(0).mean())
        t = v_pre / v_own if v_own > 0 else float("nan")
        res["modalities"][m] = {"entry": ENTRY[m], "var_at_port": v_own,
                                "var_at_precentral": v_pre, "transport": t}
        print(f"  {m:10s} {v_own:13.3e} {v_pre:15.3e} {t:12.3e}", flush=True)

    ts = [v["transport"] for v in res["modalities"].values()]
    best = max(ts)
    res["best_transport"] = best

    # ---- does driving all three do more than the sum of the parts? ----------
    # integration, if it exists, is superadditive: the sheet should make a
    # combination mean something the parts do not.  measured against the
    # linear-sum null, which is what a sheet that merely superposes would give.
    lin = sum(run(dyn, {m: sig[m]}, ports, read, **kw) - quiet for m in ENTRY)
    v_lin = float(lin.var(0).mean())
    v_full = float((full - quiet).var(0).mean())
    res["var_precentral_full"] = v_full
    res["var_precentral_linear_sum"] = v_lin
    res["superadditivity"] = v_full / v_lin if v_lin > 0 else float("nan")
    print(f"\n  precentral variance, all three driven   {v_full:.3e}", flush=True)
    print(f"  same, as a linear sum of the parts      {v_lin:.3e}", flush=True)
    print(f"  ratio (1.0 = the sheet only superposes) {res['superadditivity']:.4f}",
          flush=True)

    print(f"\n  best transport across the three modalities: {best:.3e}", flush=True)
    if best < 1e-2:
        verdict = ("NO CONVERGENCE -- the motor strip does not see the senses. "
                   "the somato-motor materialization is blocked on the sheet "
                   "not conducting, not on data or on the loss.")
    elif best < 0.1:
        verdict = ("WEAK CONVERGENCE -- a signal arrives but attenuated by more "
                   "than 10x; a readout could use it only if trained against it "
                   "directly, and any such result needs a permuted-kernel control.")
    else:
        verdict = ("CONVERGENCE -- the senses reach the motor strip at a usable "
                   "amplitude. this must still be gated on an ablation before it "
                   "is claimed as integration.")
    print(f"  -> {verdict}", flush=True)
    res["verdict"] = verdict

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
