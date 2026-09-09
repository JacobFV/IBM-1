"""why did end-to-end training fail where a head fit on frozen features succeeds?

the disjoint-readout ablation established two things that sit awkwardly together:
a head FIT ON FROZEN FEATURES reads 48.4x chance from a precentral-only readout,
so the information arrives and is linearly usable -- and yet every end-to-end
attempt to train a sensorimotor materialization through the sheet has failed.

the standing hypothesis, so far untested, is that this is an OPTIMISATION problem
downstream of the transport problem: the loss gradient reaching the encoder has
to travel back through the same ~1e-3 attenuation the signal travelled forward
through, so it arrives as small as the signal does and the encoder never learns.
if that is right, the gradient norm at the encoder should scale with TRANSPORT,
and the concentrated kernel should train end-to-end where the unmodified one
cannot.

this measures the gradient directly rather than inferring it from training
curves.  identical batch, identical loss, identical initialisation, identical
random seed -- the ONLY difference between the arms is the kernel configuration,
so any difference in gradient norm is attributable to it.

three gates, because a gradient-norm comparison is easy to fool:

*severed*   -- with the association zeroed, the encoder gradient must be exactly
    zero.  nothing connects the drive to a disjoint readout, so any nonzero norm
    means gradient is arriving by a path that is not the sheet, and every other
    number here would be measuring that path instead.

*at the port* -- the gradient of a readout taken AT the driven region must be
    large in every arm.  this separates "the sheet does not transmit gradient"
    from "the loss is flat", which produce identical encoder norms and mean
    completely different things.

*matched forward signal* -- the forward transport ratio is reported beside the
    gradient ratio.  the hypothesis is that they MATCH.  a gradient ratio that
    substantially exceeds the transport ratio would falsify the simple
    backward-attenuation story rather than confirm it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)


def load_kernel(ckpt: str, device):
    """the graph is part of the trained object -- read it, never redraw it."""
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = d
    if "dyn.idx" not in sd:
        for head in (d.get("heads") or {}).values():
            if isinstance(head, dict) and "dyn.idx" in head:
                sd = head
                break
    if "dyn.idx" not in sd:
        raise KeyError(f"{ckpt}: no association graph; refusing to redraw it")
    embed = sd["dyn.embed"] if "dyn.embed" in sd else d["dyn.embed"]
    n, e = embed.shape
    k = sd["dyn.idx"].shape[1]
    dyn = P.CorticalDynamics(n, e, k, device).to(device)
    with torch.no_grad():
        dyn.embed.copy_(embed.to(device))
        for name in ("idx", "geo", "pos"):
            if f"dyn.{name}" in sd:
                getattr(dyn, name).copy_(sd[f"dyn.{name}"].to(device))
        for name in ("w_ee", "w_ei", "w_assoc", "a_gain", "log_len"):
            if f"dyn.{name}" in sd:
                getattr(dyn, name).copy_(sd[f"dyn.{name}"].to(device))
    return dyn, d.get("step", -1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/ibm1_curriculum16.pt")
    ap.add_argument("--configs",
                    default="base:2.0,1.0,0,1.0,0.0;aniso4d:2.0,8.0,4,1.0,120.0",
                    help="name:tanh_slope,long_gain,long_topm,local_gain,long_min_dist")
    ap.add_argument("--drive-region", default="occipital")
    ap.add_argument("--read-region", default="precentral")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=4)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=2e-2)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/encoder_gradients.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dyn, step = load_kernel(a.ckpt, dev)
    port = P.region_index(dyn.pos, a.drive_region).to(dev)
    read = P.region_index(dyn.pos, a.read_region).to(dev)
    inter = len(np.intersect1d(port.cpu().numpy(), read.cpu().numpy()))
    if inter:
        raise SystemExit(f"{a.drive_region} and {a.read_region} share {inter} "
                         f"sites; gradient through an overlapping readout is not "
                         f"gradient through the sheet")
    print(f"{a.ckpt}: step {step}, {dyn.n} sites | drive {a.drive_region} "
          f"({len(port)}) -> read {a.read_region} ({len(read)}), disjoint",
          flush=True)

    def fresh_heads():
        """identical initialisation for every arm -- the kernel is the ONLY
        difference, so the seed is reset immediately before construction."""
        torch.manual_seed(a.seed)
        enc = nn.Sequential(nn.Linear(64, 256), nn.GELU(),
                            nn.Linear(256, len(port))).to(dev)
        head = nn.Sequential(nn.Linear(len(read), 256), nn.GELU(),
                             nn.Linear(256, a.dim)).to(dev)
        tgt = nn.Sequential(nn.Linear(64, 256), nn.GELU(),
                            nn.Linear(256, a.dim)).to(dev)
        return enc, head, tgt

    g = torch.Generator(device=dev).manual_seed(a.seed)
    x = torch.randn(a.batch, 64, device=dev, generator=g)
    y = torch.randn(a.batch, 64, device=dev, generator=g)

    def run(cfg, read_at_port: bool, sever: bool):
        enc, head, tgt = fresh_heads()
        for name, v in cfg.items():
            setattr(dyn, name, v)
        geo0 = dyn.geo.clone()
        if sever:
            dyn.geo.zero_()
        idx = port if read_at_port else read
        if read_at_port:
            torch.manual_seed(a.seed)
            head = nn.Sequential(nn.Linear(len(port), 256), nn.GELU(),
                                 nn.Linear(256, a.dim)).to(dev)
        drive = torch.zeros(a.batch, dyn.n, device=dev)
        drive = drive.index_copy(1, port, enc(x))
        s = dyn.init_state(a.batch, dev)
        w = dyn.edge_weights()
        h = a.dt / a.substeps
        for _ in range(a.n_steps):
            for _ in range(a.substeps):
                s = dyn.step(s, drive, h, w)
        z = F.normalize(head(s[1][:, idx]), dim=-1)
        zt = F.normalize(tgt(y), dim=-1)
        logits = z @ zt.T / 0.07
        lbl = torch.arange(a.batch, device=dev)
        loss = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl))
        enc.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in enc.parameters()
                                  if p.grad is not None)))
        fwd = float(s[1][:, idx].detach().std(0).mean())
        dyn.geo.copy_(geo0)
        return gn, fwd, float(loss)

    cfgs = {}
    for spec in a.configs.split(";"):
        name, f = spec.split(":")
        v = [float(t) for t in f.split(",")]
        cfgs[name] = {"tanh_slope": v[0], "long_gain": v[1],
                      "long_topm": int(v[2]),
                      "local_gain": v[3] if len(v) > 3 else 1.0,
                      "long_min_dist": v[4] if len(v) > 4 else 0.0}

    print(f"\n  {'config':10s} {'|grad| enc':>12s} {'fwd sd':>11s} "
          f"{'loss':>8s}   (readout = {a.read_region})", flush=True)
    res = {"ckpt": a.ckpt, "step": step, "configs": {}}
    for name, cfg in cfgs.items():
        gn, fwd, ls = run(cfg, read_at_port=False, sever=False)
        gs, _, _ = run(cfg, read_at_port=False, sever=True)
        gp, fp, _ = run(cfg, read_at_port=True, sever=False)
        res["configs"][name] = {"grad_disjoint": gn, "fwd_sd_disjoint": fwd,
                                "loss": ls, "grad_severed": gs,
                                "grad_at_port": gp, "fwd_sd_at_port": fp}
        print(f"  {name:10s} {gn:12.4e} {fwd:11.4e} {ls:8.4f}", flush=True)
        print(f"  {'':10s} severed {gs:.3e} (must be 0)   "
              f"at-port grad {gp:.3e}, fwd sd {fp:.3e}", flush=True)

    names = list(cfgs)
    if len(names) == 2:
        b, c = res["configs"][names[0]], res["configs"][names[1]]
        gr = c["grad_disjoint"] / b["grad_disjoint"] if b["grad_disjoint"] else float("nan")
        fr = c["fwd_sd_disjoint"] / b["fwd_sd_disjoint"] if b["fwd_sd_disjoint"] else float("nan")
        res["gradient_ratio"], res["forward_ratio"] = gr, fr
        print(f"\n  gradient ratio {names[1]}/{names[0]}  {gr:8.2f}x", flush=True)
        print(f"  forward  ratio {names[1]}/{names[0]}  {fr:8.2f}x", flush=True)
        if not (gr == gr and fr == fr):
            verdict = "one arm produced no signal; ratios undefined"
        elif abs(gr / fr - 1) < 0.5:
            verdict = ("gradient scales WITH transport, as the hypothesis "
                       "predicts. the backward pass is attenuated by the same "
                       "factor as the forward pass, so end-to-end training "
                       "through the unmodified sheet was starved of gradient "
                       "and the concentrated kernel should train where it could "
                       "not. this is a mechanism, not yet a demonstration -- "
                       "only an actual end-to-end run shows it trains.")
        elif gr > fr:
            verdict = ("gradient grows FASTER than forward transport, which the "
                       "simple backward-attenuation story does not predict")
        else:
            verdict = ("gradient grows more SLOWLY than forward transport -- "
                       "attenuation alone does not account for it, and the "
                       "optimisation story needs another term")
        print(f"  -> {verdict}", flush=True)
        res["verdict"] = verdict

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
