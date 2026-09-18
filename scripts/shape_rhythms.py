"""shape the cortical field toward the rhythms it is specified to have.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/shape_rhythms.py \
        --states wake-eyes-open,nrem3 --steps 300 --out out/shape_rhythms.json

WHAT THIS IS, AND WHY IT IS NOT A TEST
--------------------------------------
This programme is a construction (CLAUDE.md, "Construction, not search").  `docs/RHYTHMS.md`
says what the brain must do spectrally; this script makes that a **pressure during gradient
descent**, one term in the objective beside the anatomical ones, so the substrate is pulled
toward the declared spectrum instead of being measured afterwards to see where it landed.

Nothing here decides whether the substrate "deserves to exist".  What it decides is whether
the pressure REACHES: a spectral term that cannot move the model is a term that will sit in
every future objective doing nothing, and that is worth finding out on its own.

THE OBJECTIVE
-------------
    L = sum_r w_r * hinge(measured_r, declared interval_r)      the rhythms (ibm/rhythms.py)
      + w_bg * hinge(aperiodic exponent, declared interval)     the 1/f background
      + w_h  * health                                            the guards below
      + w_a  * anatomy                                           keep the declared priors

`hinge` is zero anywhere inside a declared interval and squared outside it, because a
declared band IS an interval; pulling toward its midpoint would invent a precision the
physiology does not have.

THE GUARDS, and why the objective is not just the spectrum
----------------------------------------------------------
A spectrum is cheap to satisfy the wrong way.  Every one of these was put in because the
cheapest way to move a spectral term is a regime nobody wants:

  * **rate**.  The mean population rate must stay in 1-20 Hz.  A cortex saturated at
    E = 1 has a beautiful flat spectrum and is a seizure.
  * **saturation**.  The fraction of (site, time) with E > 0.9 is penalised directly.
  * **silence**.  A dead sheet has no peaks to be wrong, so a floor sits under the rate.
  * **anatomy**.  The heterogeneity residuals are already bounded to +-35% of their prior
    by construction, and are additionally pulled toward zero: the spectrum may bend the
    declared hierarchy, it may not replace it.

PRE-REGISTERED GATES (docs/LOG.md, 18 September 2026).  Recorded as FAILED if they fail;
instruments may change after a failure, thresholds may not.

  G0  ENTRAINMENT, a known answer for the whole pipeline.  Drive V1 at 10 Hz and the V1
      trace's measured peak must come back 10.0 +- 0.5 Hz.  This is the catalogue's own
      `ssvep` row used as a calibration: drive at f, get f.  If it fails, the sheet or the
      instrument is broken and every other number in the run is void.
  G1  IDEMPOTENCE.  The same evaluation, with the same generator, twice, bit-identical.
      CLAUDE.md's rule: a non-idempotent instrument produces confident wrong conclusions
      from sound reasoning.
  G2  BASELINE.  Every target's measured value BEFORE training, recorded, so an
      improvement is measured against where the substrate already was rather than against
      nothing.
  G3  REACH.  After training, each target's measured value is inside its declared interval.
      Reported per target; a target that does not arrive is named.
  G4  RELABEL CONTROL.  The same training with the site-to-station map PERMUTED once (one
      draw, made outside and passed in).  If the permuted arm reaches the same loss, the
      pressure is not anatomical -- it is "make the whole sheet oscillate", which any sheet
      can do -- and G3 means nothing.  The control must end HIGHER than the shaped arm.

Memory: CPU by default.  On the GB10 a GPU job is a machine-wide memory risk (CLAUDE.md),
so `--device cuda` is opt-in and one at a time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm import spectral as SP                                          # noqa: E402
from ibm.rhythms import BACKGROUND, RHYTHM, rhythm_targets, station_sites  # noqa: E402
from ibm.substrate import R_MAX, build_sheet                            # noqa: E402


# --------------------------------------------------------------------------------------
# behavioural states as protocols.  a state is a CONTEXT the substrate is put in -- a drive
# and a neuromodulatory gain -- never a flag the model reads.  `seconds` is per state
# because the slowest target in it sets the window: you cannot resolve 1 Hz in 2 seconds.
# --------------------------------------------------------------------------------------
PROTOCOLS = {
    "wake-eyes-open": dict(drive_stations=("v1", "v_extra"), drive=0.30, tonic=0.22,
                           m_beta=1.0, m_sigma=1.0, seconds=6.0, burn_s=1.5),
    "wake-rest":      dict(drive_stations=(), drive=0.0, tonic=0.22,
                           m_beta=1.0, m_sigma=1.0, seconds=8.0, burn_s=2.0),
    "wake-task":      dict(drive_stations=("v_extra", "ips"), drive=0.25, tonic=0.22,
                           m_beta=1.05, m_sigma=0.9, seconds=6.0, burn_s=1.5),
    "listening":      dict(drive_stations=("a1", "stg"), drive=0.30, tonic=0.22,
                           m_beta=1.0, m_sigma=1.0, seconds=6.0, burn_s=1.5),
    "nrem3":          dict(drive_stations=(), drive=0.0, tonic=0.16,
                           m_beta=0.92, m_sigma=1.30, seconds=10.0, burn_s=2.0),
    "nrem2":          dict(drive_stations=(), drive=0.0, tonic=0.18,
                           m_beta=0.95, m_sigma=1.20, seconds=10.0, burn_s=2.0),
}

# a target is only scorable if the window holds enough cycles of its slowest component.
# five is the bar: fewer, and the estimate is one or two lumps of a periodogram and its
# gradient is noise.
MIN_CYCLES = 5.0


def hinge(x: torch.Tensor, target):
    """0 inside the declared interval, squared distance outside it.

    A float target means a point (a measured centre frequency, e.g. the 13.45 Hz spindle),
    and is scored as a squared error.  A (lo, hi) target is an interval, which is what a
    declared band is, and anywhere inside it costs nothing.
    """
    if target is None:
        return x * 0.0
    if isinstance(target, (int, float)):
        return (x - float(target)) ** 2
    lo, hi = float(target[0]), float(target[1])
    below = torch.clamp(lo - x, min=0.0)
    above = torch.clamp(x - hi, min=0.0)
    return below ** 2 + above ** 2


def trace_of(trace, sites):
    """the population trace of a station: mean over its sites.  (B, T)."""
    return trace[..., sites].mean(-1)


def measure_target(t, trace, fs, nperseg):
    """the catalogue row's declared quantity, measured on this rollout.  Differentiable."""
    kind = t["kind"]
    if kind == "coherence":
        vals = []
        for i in range(len(t["groups"])):
            for j in range(i + 1, len(t["groups"])):
                x = trace_of(trace, t["groups"][i])
                y = trace_of(trace, t["groups"][j])
                vals.append(SP.band_coherence(x, y, fs, t["band"][0], t["band"][1],
                                              nperseg=nperseg))
        return torch.stack(vals).mean()
    x = trace_of(trace, t["sites"])
    if kind == "pac":
        return SP.pac_mi(x, fs, t["phase_band"], t["amp_band"]).mean()
    freqs, psd = SP.welch_psd(x, fs, nperseg=nperseg)
    lo, hi = t["band"]
    if kind == "relative_power":
        return SP.relative_band_power(psd, freqs, lo, hi).mean()
    if kind == "peak_frequency":
        return SP.peak_frequency(psd, freqs, lo, hi).mean()
    if kind == "peak_prominence":
        return SP.peak_prominence(psd, freqs, lo, hi).mean()
    raise ValueError(f"no measurement for kind {kind!r}")


def resolvable(t, seconds, fs, nperseg):
    """(ok, reason).  Can this window resolve the target's slowest component at all?"""
    lo = t.get("band", t.get("phase_band", (1.0, 1.0)))[0]
    if lo <= 0:
        return False, "band starts at 0 Hz"
    seg_s = nperseg / fs
    cycles = seg_s * lo
    if cycles < MIN_CYCLES:
        return False, (f"{lo:g} Hz needs {MIN_CYCLES:g} cycles = {MIN_CYCLES / lo:.1f}s "
                       f"per segment; this window gives {seg_s:.1f}s")
    return True, ""


class Shaper:
    """one substrate, a set of states, and the objective over them."""

    def __init__(self, a, device):
        self.a = a
        self.device = device
        self.field = build_sheet(a.sites, a.k, a.long_frac, seed=a.seed, device=device,
                                 long_topology=a.topology).to(device)
        self.names = self.field.region_names
        self.fs = 1.0 / a.dt
        # ONE permutation, drawn once, here, and handed to whoever needs it: the relabel
        # control must not draw its own (CLAUDE.md, "a shared generator").
        g = torch.Generator().manual_seed(a.seed + 104729)
        self.perm = torch.randperm(len(self.names), generator=g).tolist()

    def targets(self, state, relabel=False):
        names = self.names
        if relabel:
            names = [self.names[i] for i in self.perm]
        ts, skipped = rhythm_targets(state, names, only_expressible=True,
                                     min_sites=self.a.min_sites)
        p = PROTOCOLS[state]
        nperseg = self.nperseg(p)
        keep = []
        for t in ts:
            ok, why = resolvable(t, p["seconds"], self.fs, nperseg)
            (keep if ok else skipped).append(t if ok else (t["id"], why))
        return keep, skipped

    def nperseg(self, p):
        n = int(round(p["seconds"] * self.fs / self.a.segments))
        return max(64, 1 << int(math.floor(math.log2(max(64, n)))))

    def drive(self, state, T, B=1):
        p = PROTOCOLS[state]
        d = torch.full((B, T, self.field.n), float(p["tonic"]), device=self.device)
        if p["drive_stations"]:
            sites = station_sites(p["drive_stations"], self.names)
            d[:, :, sites] += p["drive"]
        return d

    def rollout(self, state, gen_seed, grad=True, extra_drive=None):
        """burn in without gradients, then a graded window.  The burn-in matters: the
        first second from rest is a transient, and its spectrum is the transient's."""
        p = PROTOCOLS[state]
        dt, fs = self.a.dt, self.fs
        nb = int(round(p["burn_s"] * fs))
        nt = int(round(p["seconds"] * fs))
        st = self.field.init_state(1, device=self.device)
        gen = torch.Generator().manual_seed(gen_seed)
        with torch.no_grad():
            d = self.drive(state, nb)
            _, st = self.field.rollout(d, st, dt, noise_gen=gen,
                                       m_beta=p["m_beta"], m_sigma=p["m_sigma"])
        st = self.field.detach(st)
        d = self.drive(state, nt)
        if extra_drive is not None:
            d = d + extra_drive
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            trace, st = self.field.rollout(d, st, dt, noise_gen=gen,
                                           m_beta=p["m_beta"], m_sigma=p["m_sigma"])
        return trace

    def loss(self, state, targets, gen_seed, grad=True):
        a = self.a
        p = PROTOCOLS[state]
        nperseg = self.nperseg(p)
        trace = self.rollout(state, gen_seed, grad=grad)
        parts, total = {}, torch.zeros((), device=self.device)
        for t in targets:
            v = measure_target(t, trace, self.fs, nperseg)
            pen = hinge(v, t["target"])
            total = total + a.w_rhythm * pen
            parts[t["id"]] = {"measured": float(v), "penalty": float(pen),
                              "target": t["target"], "kind": t["kind"]}
        # the 1/f background, on the mean cortical trace
        mean_trace = trace.mean(-1)
        freqs, psd = SP.welch_psd(mean_trace, self.fs, nperseg=nperseg)
        _off, expo = SP.aperiodic_fit(psd, freqs, BACKGROUND["fit_band"][0],
                                      min(BACKGROUND["fit_band"][1], self.fs / 2 - 1),
                                      exclude=BACKGROUND["exclude"])
        expo = expo.mean()
        pen = hinge(expo, BACKGROUND["exponent_target"])
        total = total + a.w_background * pen
        parts["aperiodic"] = {"measured": float(expo), "penalty": float(pen),
                              "target": BACKGROUND["exponent_target"], "kind": "exponent"}
        # health.  a spectrum reached by saturating the cortex is not the spectrum.
        rate = trace.mean() * R_MAX
        sat = (trace > 0.9).float().mean()
        h = hinge(rate, (a.rate_lo, a.rate_hi)) / 100.0 + (sat ** 2) * 100.0
        total = total + a.w_health * h
        parts["health"] = {"rate_hz": float(rate), "saturated_frac": float(sat),
                           "penalty": float(h)}
        # anatomy: the declared priors are the landscape; the spectrum may bend them.
        anat = sum((getattr(self.field, f"res_{n}") ** 2).mean()
                   for n in ("tau_E", "tau_a", "g_a", "tau_rec", "w_EE", "theta_E"))
        total = total + a.w_anatomy * anat
        parts["anatomy"] = {"residual_l2": float(anat)}
        return total, parts

    # ---------------------------------------------------------------- gates
    def gate_entrainment(self):
        """G0.  Drive V1 at 10 Hz; the V1 trace must peak at 10 Hz."""
        state = "wake-eyes-open"
        p = PROTOCOLS[state]
        nt = int(round(p["seconds"] * self.fs))
        sites = station_sites(("v1",), self.names)
        if len(sites) < self.a.min_sites:
            return {"ok": False, "why": f"v1 resolves to {len(sites)} sites"}
        t = torch.arange(nt, device=self.device) * self.a.dt
        wave = 0.15 * torch.sin(2 * math.pi * 10.0 * t)
        extra = torch.zeros(1, nt, self.field.n, device=self.device)
        extra[0, :, sites] = wave[:, None]
        with torch.no_grad():
            trace = self.rollout(state, self.a.seed + 1, grad=False, extra_drive=extra)
            x = trace_of(trace, sites)
            freqs, psd = SP.welch_psd(x, self.fs, nperseg=self.nperseg(p))
            f = float(SP.peak_frequency(psd, freqs, 4.0, 20.0).mean())
        return {"ok": abs(f - 10.0) <= 0.5, "measured_hz": f, "expected_hz": 10.0,
                "tolerance_hz": 0.5}

    def gate_idempotent(self, state, targets):
        """G1.  Two identical evaluations must agree bit for bit."""
        with torch.no_grad():
            a, _ = self.loss(state, targets, self.a.seed + 2, grad=False)
            b, _ = self.loss(state, targets, self.a.seed + 2, grad=False)
        return {"ok": bool(torch.equal(a, b)), "first": float(a), "second": float(b)}


def jdefault(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist() if o.numel() <= 64 else f"<tensor {tuple(o.shape)}>"
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f"<ndarray {o.shape}>"
    return str(o)


def train(sh, states, relabel, steps, out, tag):
    """one arm.  Returns its record; writes weights unconditionally at the last step."""
    a = sh.a
    per_state = {}
    for s in states:
        ts, skipped = sh.targets(s, relabel=relabel)
        per_state[s] = (ts, skipped)
        print(f"  [{tag}] {s}: {len(ts)} targets "
              f"({', '.join(t['id'] for t in ts) or 'none'}); {len(skipped)} skipped",
              flush=True)
    scorable = [s for s in states if per_state[s][0]]
    if not scorable:
        return {"tag": tag, "error": "no state has a scorable target"}
    opt = torch.optim.Adam([p for p in sh.field.parameters() if p.requires_grad], lr=a.lr)
    hist = []
    for step in range(steps):
        state = scorable[step % len(scorable)]
        ts, _ = per_state[state]
        opt.zero_grad(set_to_none=True)
        total, parts = sh.loss(state, ts, a.seed + 1000 + step, grad=True)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in sh.field.parameters() if p.requires_grad], a.clip)
        opt.step()
        rec = {"step": step, "state": state, "loss": float(total), "grad_norm": float(gn),
               "parts": parts}
        hist.append(rec)
        if step % a.log_every == 0 or step == steps - 1:
            named = "  ".join(f"{k}={v['measured']:.4g}" for k, v in parts.items()
                              if "measured" in v)
            print(f"  [{tag}] {step:4d} {state:15s} L={float(total):9.4f} "
                  f"|g|={float(gn):7.3f}  {named}", flush=True)
    # final measurement per state, no gradient, on a HELD-OUT noise draw: the training
    # draws are seeds 1000+; this one is not among them.
    final = {}
    for s in states:
        ts, skipped = per_state[s]
        if not ts:
            final[s] = {"targets": [], "skipped": skipped}
            continue
        with torch.no_grad():
            total, parts = sh.loss(s, ts, a.seed + 900001, grad=False)
        final[s] = {"loss": float(total), "parts": parts,
                    "skipped": [list(x) for x in skipped]}
    torch.save({"state_dict": sh.field.state_dict(), "tag": tag, "args": vars(a)},
               out.replace(".json", f"_{tag}.pt"))
    return {"tag": tag, "relabel": relabel, "final": final,
            "history": hist[-1:] if a.thin_history else hist}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", default="wake-eyes-open,nrem3")
    ap.add_argument("--sites", type=int, default=512)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--long-frac", type=float, default=0.25)
    ap.add_argument("--topology", default="tract", choices=("tract", "random"))
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--segments", type=float, default=3.0,
                    help="segments per window; nperseg = seconds*fs/segments, rounded down "
                         "to a power of two")
    ap.add_argument("--min-sites", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-rhythm", type=float, default=1.0)
    ap.add_argument("--w-background", type=float, default=1.0)
    ap.add_argument("--w-health", type=float, default=5.0)
    ap.add_argument("--w-anatomy", type=float, default=0.05)
    ap.add_argument("--rate-lo", type=float, default=1.0)
    ap.add_argument("--rate-hi", type=float, default=20.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--thin-history", action="store_true")
    ap.add_argument("--no-control", action="store_true",
                    help="skip G4. Recorded in the output as a missing control, not as a "
                         "passed one.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="out/shape_rhythms.json")
    a = ap.parse_args()

    states = [s.strip() for s in a.states.split(",") if s.strip()]
    bad = [s for s in states if s not in PROTOCOLS]
    if bad:
        print(f"no protocol for: {', '.join(bad)}.  known: {', '.join(PROTOCOLS)}")
        return 2
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # the cheap summary FIRST, before anything that can raise (CLAUDE.md).
    rec = {"script": "scripts/shape_rhythms.py", "args": vars(a),
           "preregistered": "docs/LOG.md 2026-09-18 rhythm shaping G0-G4",
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": os.environ.get(
               "IBM_GIT_SHA", "git-unknown"), "gates": {}, "arms": []}

    def save():
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=jdefault)

    save()
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)          # only for anything torch draws internally
    sh = Shaper(a, dev)
    print(f"sheet: {sh.field.n} sites, k={sh.field.k}, {sh.field.n_far} long-range, "
          f"topology={a.topology}, fs={sh.fs:g} Hz", flush=True)

    rec["gates"]["G0_entrainment"] = sh.gate_entrainment()
    print(f"G0 entrainment: {rec['gates']['G0_entrainment']}", flush=True)
    save()

    ts0, _ = sh.targets(states[0])
    rec["gates"]["G1_idempotence"] = sh.gate_idempotent(states[0], ts0) if ts0 else {
        "ok": False, "why": f"no scorable target in {states[0]}"}
    print(f"G1 idempotence: {rec['gates']['G1_idempotence']}", flush=True)
    save()

    base = {}
    for s in states:
        ts, skipped = sh.targets(s)
        if not ts:
            base[s] = {"targets": [], "skipped": [list(x) for x in skipped]}
            continue
        with torch.no_grad():
            total, parts = sh.loss(s, ts, a.seed + 3, grad=False)
        base[s] = {"loss": float(total), "parts": parts,
                   "skipped": [list(x) for x in skipped]}
    rec["gates"]["G2_baseline"] = base
    save()

    if not rec["gates"]["G0_entrainment"].get("ok") or not rec["gates"]["G1_idempotence"].get("ok"):
        rec["verdict"] = "VOID: a calibration gate failed; no training was run"
        save()
        print("VOID: G0 or G1 failed.  Not training -- a run on a broken instrument "
              "produces numbers that look like results.")
        return 1

    t0 = time.time()
    rec["arms"].append(train(sh, states, False, a.steps, a.out, "shaped"))
    save()
    if not a.no_control:
        sh2 = Shaper(a, dev)            # a fresh substrate, same seed, same starting point
        rec["arms"].append(train(sh2, states, True, a.steps, a.out, "relabelled"))
    else:
        rec["arms"].append({"tag": "relabelled", "skipped": "--no-control was passed"})
    rec["seconds"] = time.time() - t0

    # the verdict, by the rules written in the header BEFORE the run
    shaped = next(x for x in rec["arms"] if x["tag"] == "shaped")
    ctrl = next(x for x in rec["arms"] if x["tag"] == "relabelled")
    reach = {}
    for s, f in shaped.get("final", {}).items():
        for tid, p in f.get("parts", {}).items():
            if "measured" not in p or tid in ("health", "anatomy"):
                continue
            reach[f"{s}/{tid}"] = {"measured": p["measured"], "target": p["target"],
                                   "inside": p["penalty"] <= 0.0}
    rec["gates"]["G3_reach"] = reach
    if "final" in ctrl:
        ls = sum(f.get("loss", 0.0) for f in shaped["final"].values())
        lc = sum(f.get("loss", 0.0) for f in ctrl["final"].values())
        rec["gates"]["G4_relabel_control"] = {"ok": lc > ls, "shaped_loss": ls,
                                              "relabelled_loss": lc}
    else:
        rec["gates"]["G4_relabel_control"] = {"ok": False, "why": "control not run"}
    save()

    print("\nG3 reach:")
    for k, v in reach.items():
        print(f"  [{'IN ' if v['inside'] else 'OUT'}] {k}: {v['measured']:.4g} "
              f"vs {v['target']}")
    print(f"G4 relabel control: {rec['gates']['G4_relabel_control']}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
