"""known answers for `ibm/brain/thalamus.py`, the v3 thalamus.

    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python \
        scripts/gate_brain_thalamus.py --gates T0,T1,T2,T3,T4,T5
    OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python \
        scripts/gate_brain_thalamus.py --gates T6

Two invocations because T6 sweeps 33 constants twice and the pair runs longer than one
command should.  The JSON at `--out` is MERGED, not overwritten, and the exit status is
computed over every gate recorded in it -- so a run of T6 alone still fails if T4 failed
earlier and nothing has re-run it.

EVERY VERDICT BELOW IS FIXED HERE, IN THE CODE, BEFORE IT RUNS.  A gate that fails is
recorded FAILED and is never rescored (CLAUDE.md).  Instruments may change after a failure;
thresholds may not.

  T0  BOUNDED.  Every state variable in [0, 1] at dt = 1, 5 and 20 ms with relay drives of
      -2, 0 and +2.  PASS iff the worst excursion outside the box is exactly 0.
  T1  IDEMPOTENCE.  The same rollout, same generator, twice.  PASS iff bit-identical.
      Not a known answer -- a known answer tests the value, this tests whether the thing is
      a function at all (CLAUDE.md).
  T2  REBOUND, the mechanism, in three arms.  A 400 ms hyperpolarising pulse, then release.
      (a) the relay peak must arrive AFTER the release and exceed twice the mean rate
          DURING the pulse.  A filter of the input cannot do this.
      (b) ARMED WITH NO PUSH: the same pulse with the tonic drive at zero must NOT produce
          a burst -- peak < 0.10 and < 2x its own pre-pulse level.  A window current is
          availability TIMES activation; a cell that fires when armed and unpushed has the
          activation term inverted, which is what `ibm/thalamus.py` shipped first.
      (c) NO SELF-OSCILLATION: with the relay -> reticular projection at weight 0, the
          11-16 Hz prominence must fall by at least 1.0 decade against the intact loop.
          v2's inverted version rang at 7 Hz with the shell silenced to 0.45 Hz.
  T3  SPINDLE BAND, against the known answer.  At the module's declared NREM2 polarisation
      (`thalamus.POLARISATION["nrem2"]`, which is also `drives()`) the isolated
      reticular-relay loop must peak in 11-16 Hz with `peak_prominence` > 0.  Measured on
      three noise seeds; the gate is on the MEAN.  A frequency without a prominence is not
      evidence (CLAUDE.md), so both are reported, and the difference from v2's 13.626 Hz
      and from the 13.45 Hz measured on held-out sleepers is reported explicitly.  A large
      difference is a finding, not automatically a failure.
  T4  SELECTIVE GATING, in three arms, on the thalamus alone.  Two modalities are driven
      with independent signals at the declared WAKE polarisation and transmission is a
      TRANSFER SLOPE -- not a correlation, which is scale-free and cannot see a gain change
      at all, and which is how v2's gating gate failed the first time.
      (a) gate the LGN sector (its drive set to +0.60 against the waking -0.35):
          LGN's slope must fall to <= 0.5x its ungated value.
      (b) MGN's slope must stay within +-25% of its ungated value.
      (c) POSITIVE CONTROL, and (b) is void without it.  The same two nuclei sharing ONE
          reticular population (each relay driving it at half weight, so the shared sector
          receives what a single sector does) must lose BOTH modalities when that one
          population is gated -- both ratios <= 0.5.  Without this arm (b) is a control
          that cannot fail, because two nuclei with separate sectors share nothing.
  T5  HIGHER-ORDER RELAY.  `ctx.pericalcarine.E` and `ctx.inferiorparietal.E` have NO
      direct projection in `ibm/brain/cortex.py` (asserted by `higher_order_pairs()`).  A
      signal is injected into the first and the transfer slope measured at the second.
      PASS iff |intact| >= 4 x max(|pulvinar cut|, |surrogate null|), where the surrogate
      null is the same measurement against an independent signal the circuit never saw --
      the noise floor of the estimator.
      A fourth arm, DIRECT FIBRE, replaces the pulvinar with a direct A -> B projection of
      the same weight.  It decides where a failure lives: if the direct fibre delivers no
      more than the cut arm, no pathway of any kind can carry a signal between two areas of
      this cortex and the failure is not the thalamus's.  It is reported either way and it
      is not part of the verdict.
  T6  SENSITIVITY.  Every one of the 33 declared constants swept +-50% and T3's frequency
      re-measured.  The two constants this module CLAIMS are the clock -- `tau_relay` and
      `tau_gabaa` -- must each move it by more than 1 Hz.  Everything moving less than
      0.2 Hz over the 3x sweep is listed as INERT, because a parameter that changes nothing
      is this programme's cheapest detector of a mechanism that is not wired in.  The
      constants that cannot possibly act on an isolated nucleus (every cortical weight and
      every cortical delay) are listed separately as inert BY CONSTRUCTION OF THE TEST,
      which is what v2's `w_CT_R` and `w_CT_T` were.
      T6 also records `pinned_state`: the range of every state variable over a run, and
      every variable whose range never reaches 1e-3.  That is CLAUDE.md's next step after a
      flat sweep -- look for the variable that is pinned -- run unconditionally so the
      answer sits in the artefact beside the sweep that raised the question.  It is a
      report, not part of the verdict.

WHAT THIS SCRIPT DOES NOT CLAIM.  T3, T4 and T6 run on the thalamus alone, with the tonic
drive standing in for a brainstem that is not written yet.  That is an input, not a dial:
`ibm/brain/neuromodulators.py` already declares the `nm.ppt` edges that will produce it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm import spectral as SP                                   # noqa: E402
from ibm.brain import cortex as CX                               # noqa: E402
from ibm.brain import thalamus as TH                             # noqa: E402
from ibm.circuit import Circuit                                  # noqa: E402

DT = 0.001
SPINDLE_BAND = (11.0, 16.0)
V2_SPINDLE_HZ = 13.626          # ibm/thalamus.py's T3, the known answer
MEASURED_SPINDLE_HZ = 13.45     # scripts/fit_sleep_resonance.py, held-out sleepers
GATED_SECTOR_DRIVE = 0.60       # what a hard layer-6 drive onto one sector looks like


# --------------------------------------------------------------------------------------
# sub-networks.  A gate runs the smallest declared sub-network that can carry its claim:
# the whole brain is 92 populations and 877 projections and a 20 s rollout of it is half an
# hour of python.  Nothing here edits a declaration; it SELECTS from pops() and the
# projections whose two ends are both selected.
# --------------------------------------------------------------------------------------
def thal_pops(nucs):
    ids = set()
    for x in nucs:
        ids |= {TH.relay_id(x), TH.trn_id(x), TH.gabaa_id(x), TH.gabab_id(x)}
    return ids


def build(nucs=(), areas=(), extra_projs=(), drop_projs=(), edit=None, seed=0):
    keep = thal_pops(nucs) | {f"ctx.{a}.{k}" for a in areas for k in ("E", "I")}
    pops = [p for p in TH.pops() + CX.pops() if p.id in keep]
    projs = [q for q in TH.internal() + TH.external() + CX.internal() + CX.external()
             if q.src in keep and q.dst in keep and q.key not in set(drop_projs)]
    if edit is not None:
        projs = [edit(q) for q in projs]
    return Circuit(pops, [q for q in projs if q is not None] + list(extra_projs), seed=seed)


def tonic(c, pol="nrem2"):
    """the tonic drive: cortex's own, plus the polarisation the brainstem would supply.

    The relay's level is the module's declared `drives()` plus the polarisation's OFFSET
    from NREM2, so a higher-order nucleus -- which is driven by cortex and declares a small
    tonic drive for that reason -- is depolarised by waking without being handed a
    first-order nucleus's whole stand-in driver on top of its real one.
    """
    p = TH.POLARISATION[pol]
    dpol = p["relay"] - TH.POLARISATION["nrem2"]["relay"]
    decl = TH.drives()
    d = {k: v for k, v in CX.drives().items() if k in c.pops}
    for pid in c.pops:
        if pid.endswith(".relay"):
            d[pid] = decl[pid] + dpol
        elif ".trn." in pid:
            d[pid] = p["trn"]
    return d


def spectrum(x, nperseg=4096):
    freqs, psd = SP.welch_psd(x, 1.0 / DT, nperseg=min(nperseg, x.shape[-1]))
    f = float(SP.peak_frequency(psd, freqs, 5.0, 25.0).mean())
    prom = float(SP.peak_prominence(psd, freqs, SPINDLE_BAND[0], SPINDLE_BAND[1],
                                    fit_lo=1.0, fit_hi=45.0).mean())
    return f, prom


def slope(x, y):
    """the TRANSFER SLOPE, which is what transmission means.  A correlation is scale-free
    and cannot see a gain change; that is how v2's T5 failed."""
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (y * y).sum())


def isolated(nuc="lgn", drop_relay_trn=False, seed=0):
    def edit(q):
        if drop_relay_trn and q.src == TH.relay_id(nuc) and q.dst == TH.trn_id(nuc):
            return dataclasses.replace(q, weight=0.0)
        return q
    return build(nucs=[nuc], edit=edit, seed=seed)


def run_isolated(nuc="lgn", steps=16000, seed=5, drop_relay_trn=False, nperseg=4096):
    c = isolated(nuc, drop_relay_trn=drop_relay_trn)
    g = torch.Generator().manual_seed(seed)
    tr, _ = c.rollout(steps, DT, drive=tonic(c, "nrem2"), noise_gen=g, burn=1000,
                      record=[TH.relay_id(nuc), TH.trn_id(nuc)])
    x = tr[TH.relay_id(nuc)].mean(-1)
    f, prom = spectrum(x, nperseg)
    return {"peak_hz": f, "spindle_prominence": prom,
            "relay_hz": float(x.mean() * 100), "relay_sd_hz": float(x.std() * 100),
            "trn_hz": float(tr[TH.trn_id(nuc)].mean() * 100)}


# --------------------------------------------------------------------------------------
# T0 bounded
# --------------------------------------------------------------------------------------
def gate_bounded():
    """every rate over every step, and the other bounded variables in the final state.

    `eta` is an Ornstein-Uhlenbeck current and `a` an adaptation variable: neither lives in
    [0, 1] by construction, so their ranges are REPORTED rather than gated.  Gating a
    variable that was never claimed to be bounded would be a control that cannot fail.
    """
    worst, cases = 0.0, []
    for dt in (0.001, 0.005, 0.020):
        for dr in (-2.0, 0.0, 2.0):
            c = isolated("lgn")
            d = {k: (v + dr if k.endswith(".relay") else v)
                 for k, v in tonic(c, "nrem2").items()}
            g = torch.Generator().manual_seed(11)
            tr, st = c.rollout(int(3.0 / dt), dt, drive=d, noise_gen=g,
                               record=list(c.pops))
            bad = 0.0
            for pid in c.pops:
                v = tr[pid]
                bad = max(bad, float((-v).clamp_min(0).max()),
                          float((v - 1).clamp_min(0).max()))
                for k in ("x", "u", "inh"):
                    if k in st[pid]:
                        w = st[pid][k]
                        bad = max(bad, float((-w).clamp_min(0).max()),
                                  float((w - 1).clamp_min(0).max()))
            worst = max(worst, bad)
            cases.append({"dt": dt, "relay_drive_offset": dr, "worst": bad,
                          "relay_hz": float(tr[TH.relay_id("lgn")].mean() * 100),
                          "adapt_range": [float(st[TH.relay_id("lgn")]["a"].min()),
                                          float(st[TH.relay_id("lgn")]["a"].max())],
                          "eta_range": [float(st[TH.relay_id("lgn")]["eta"].min()),
                                        float(st[TH.relay_id("lgn")]["eta"].max())]})
    return {"ok": worst <= 0.0, "worst_excursion": worst,
            "checked": "every population's r at every step, and x/u/inh in the final "
                       "state, against [0, 1]; a and eta reported, not gated",
            "cases": cases}


# --------------------------------------------------------------------------------------
# T1 idempotence
# --------------------------------------------------------------------------------------
def gate_idempotent():
    out = []
    for _ in range(2):
        c = isolated("lgn")
        g = torch.Generator().manual_seed(7)
        tr, _ = c.rollout(3000, DT, drive=tonic(c, "nrem2"), noise_gen=g, burn=200,
                          record=[TH.relay_id("lgn"), TH.trn_id("lgn")])
        out.append(torch.cat([tr[k] for k in sorted(tr)], -1))
    same = bool(torch.equal(out[0], out[1]))
    return {"ok": same, "bit_identical": same,
            "max_abs_diff": float((out[0] - out[1]).abs().max())}


# --------------------------------------------------------------------------------------
# T2 rebound
# --------------------------------------------------------------------------------------
def _pulse_run(tonic_relay, steps=1600, t0=300, t1=700, amp=-1.2):
    c = isolated("lgn")
    base = tonic(c, "nrem2")

    def drive(i):
        d = dict(base)
        d[TH.relay_id("lgn")] = tonic_relay + (amp if t0 <= i < t1 else 0.0)
        return d

    tr, _ = c.rollout(steps, DT, drive=drive, noise_gen=None, burn=600,
                      record=[TH.relay_id("lgn")])
    x = tr[TH.relay_id("lgn")].mean(-1)[0]
    k = int(torch.argmax(x[t0:])) + t0
    return {"pre_pulse": float(x[:t0].mean()), "during_pulse": float(x[t0:t1].mean()),
            "peak": float(x[k]), "peak_step": k, "release_step": t1,
            "peak_ms_after_release": (k - t1) * DT * 1000.0}


def gate_rebound():
    a = _pulse_run(TH.POLARISATION["nrem2"]["relay"])
    a["ok"] = a["peak_step"] > a["release_step"] and a["peak"] > 2.0 * a["during_pulse"]
    b = _pulse_run(0.0)
    b["ok"] = b["peak"] < 0.10 and b["peak"] < 2.0 * max(b["pre_pulse"], 1e-6)
    intact = run_isolated("lgn")
    silenced = run_isolated("lgn", drop_relay_trn=True)
    drop = intact["spindle_prominence"] - silenced["spindle_prominence"]
    c = {"ok": drop >= 1.0, "intact": intact, "relay_to_trn_weight_zero": silenced,
         "prominence_drop_decades": drop,
         "note": "zeroing relay->trn also raises the relay's rate, and that confound is "
                 "reported rather than corrected: the claim is only that the ring is the "
                 "loop's"}
    return {"ok": bool(a["ok"] and b["ok"] and c["ok"]),
            "a_rebound_after_release": a, "b_armed_with_no_push": b,
            "c_no_self_oscillation": c}


# --------------------------------------------------------------------------------------
# T3 spindle band
# --------------------------------------------------------------------------------------
def gate_spindle():
    rows = [run_isolated("lgn", seed=s) for s in (5, 17, 29)]
    fs = torch.tensor([r["peak_hz"] for r in rows])
    ps = torch.tensor([r["spindle_prominence"] for r in rows])
    f, sd = float(fs.mean()), float(fs.std())
    prom = float(ps.mean())
    lo, hi = SPINDLE_BAND
    return {"ok": bool(lo <= f <= hi and prom > 0.0),
            "mean_peak_hz": f, "sd_over_seeds_hz": sd, "mean_prominence_decades": prom,
            "band": list(SPINDLE_BAND), "seeds": rows,
            "polarisation": TH.POLARISATION["nrem2"],
            "minus_v2_13.626_hz": f - V2_SPINDLE_HZ,
            "minus_measured_13.45_hz": f - MEASURED_SPINDLE_HZ}


# --------------------------------------------------------------------------------------
# T4 selective gating
# --------------------------------------------------------------------------------------
def _lumped_edit(shared, other):
    """one reticular population for two nuclei, receiving what a single sector receives.

    Both relays drive the shared sector at HALF weight, so the lumped arm is not gated more
    easily merely because its sector is driven twice as hard.  `thal.trn.<other>` is left in
    the network with no edges at all and is not recorded.
    """
    s_trn, o_trn = TH.trn_id(shared), TH.trn_id(other)

    def edit(q):
        if q.src == o_trn and q.dst == o_trn:
            return None                                   # the dead sector's self-loop
        if q.src == TH.relay_id(other) and q.dst == o_trn:
            return dataclasses.replace(q, dst=s_trn, weight=q.weight * 0.5)
        if q.src == o_trn:
            return dataclasses.replace(q, src=s_trn)      # its two synaptic pools
        if q.dst == o_trn:
            return None
        if q.src == TH.relay_id(shared) and q.dst == s_trn:
            return dataclasses.replace(q, weight=q.weight * 0.5)
        return q
    return edit


def _transmission(lumped: bool, gate_lgn: bool, steps=10000, seed=4):
    g = torch.Generator().manual_seed(21)
    sig = {n: (0.25 * torch.randn(1, steps, 1, generator=g)).repeat(1, 1, TH.N_RELAY)
           for n in ("lgn", "mgn")}
    c = build(nucs=["lgn", "mgn"],
              edit=_lumped_edit("lgn", "mgn") if lumped else None)
    base = tonic(c, "wake")
    gated_sector = TH.trn_id("lgn")

    def drive(i):
        i = max(0, min(i, steps - 1))
        d = dict(base)
        for n in ("lgn", "mgn"):
            d[TH.relay_id(n)] = TH.POLARISATION["wake"]["relay"] + sig[n][:, i]
        if gate_lgn:
            d[gated_sector] = GATED_SECTOR_DRIVE
        return d

    gg = torch.Generator().manual_seed(seed)
    tr, _ = c.rollout(steps, DT, drive=drive, noise_gen=gg, burn=1000,
                      record=[TH.relay_id(n) for n in ("lgn", "mgn")])
    return {n: {"transfer_slope": slope(tr[TH.relay_id(n)].mean(-1)[0], sig[n][0, :, 0]),
                "relay_hz": float(tr[TH.relay_id(n)].mean() * 100)}
            for n in ("lgn", "mgn")}


def gate_selective():
    out = {}
    for lumped in (False, True):
        key = "lumped_one_shared_sector" if lumped else "sectored"
        ung = _transmission(lumped, False)
        gat = _transmission(lumped, True)
        out[key] = {"ungated": ung, "gated": gat,
                    "ratio": {n: gat[n]["transfer_slope"] / ung[n]["transfer_slope"]
                              for n in ("lgn", "mgn")}}
    s = out["sectored"]["ratio"]
    l = out["lumped_one_shared_sector"]["ratio"]
    a = s["lgn"] <= 0.5
    b = 0.75 <= s["mgn"] <= 1.25
    ctrl = l["lgn"] <= 0.5 and l["mgn"] <= 0.5
    return {"ok": bool(a and b and ctrl),
            "a_gated_modality_falls": {"ok": bool(a), "lgn_ratio": s["lgn"], "bar": "<= 0.5"},
            "b_other_modality_holds": {"ok": bool(b), "mgn_ratio": s["mgn"],
                                       "bar": "0.75 to 1.25"},
            "c_positive_control_lumped": {"ok": bool(ctrl), "ratios": l,
                                          "bar": "both <= 0.5",
                                          "note": "without this arm (b) is a control that "
                                                  "cannot fail"},
            "polarisation": TH.POLARISATION["wake"],
            "gated_sector_drive": GATED_SECTOR_DRIVE,
            "detail": out}


# --------------------------------------------------------------------------------------
# T5 higher-order relay
# --------------------------------------------------------------------------------------
def gate_higher_order(steps=16000):
    nuc, a_area, b_area = TH.higher_order_pairs()[0]
    A, B = f"ctx.{a_area}.E", f"ctx.{b_area}.E"
    g = torch.Generator().manual_seed(33)
    sig = (0.30 * torch.randn(1, steps, 1, generator=g)).repeat(1, 1, 64)
    null = (0.30 * torch.randn(1, steps, 1, generator=g)).repeat(1, 1, 64)
    direct = TH.Proj(A, B, weight=TH.C.w_tc_e, topology="sparse", p=0.25, delay_s=0.008,
                     note="the direct fibre this pair does NOT have, as a positive control")

    def arm(nucs, extra=()):
        c = build(nucs=nucs, areas=[a_area, b_area], extra_projs=extra)
        base = tonic(c, "wake")

        def drive(i):
            i = max(0, min(i, steps - 1))
            d = dict(base)
            d[A] = base[A] + sig[:, i]
            return d

        gg = torch.Generator().manual_seed(8)
        rec = [A, B] + ([TH.relay_id(nuc)] if nucs else [])
        tr, _ = c.rollout(steps, DT, drive=drive, noise_gen=gg, burn=1000, record=rec)
        r = {"source_slope": slope(tr[A].mean(-1)[0], sig[0, :, 0]),
             "target_slope": slope(tr[B].mean(-1)[0], sig[0, :, 0]),
             "target_slope_vs_surrogate": slope(tr[B].mean(-1)[0], null[0, :, 0]),
             "source_hz": float(tr[A].mean() * 100), "target_hz": float(tr[B].mean() * 100)}
        if nucs:
            r["relay_slope"] = slope(tr[TH.relay_id(nuc)].mean(-1)[0], sig[0, :, 0])
            r["relay_hz"] = float(tr[TH.relay_id(nuc)].mean() * 100)
        return r

    intact = arm([nuc])
    cut = arm([])
    fibre = arm([], extra=(direct,))
    floor = max(abs(cut["target_slope"]), abs(intact["target_slope_vs_surrogate"]),
                abs(cut["target_slope_vs_surrogate"]))
    ok = abs(intact["target_slope"]) >= 4.0 * floor
    return {"ok": bool(ok), "pair": [nuc, a_area, b_area],
            "no_direct_fibre_in_cortex": True,
            "intact": intact, "pulvinar_cut": cut, "direct_fibre_control": fibre,
            "noise_floor": floor, "ratio_intact_over_floor":
                abs(intact["target_slope"]) / max(floor, 1e-12),
            "bar": "|intact| >= 4 x max(|cut|, |surrogate null|)",
            "diagnosis": "the direct-fibre arm says whether ANY pathway can carry a signal "
                         "between two areas of this cortex; it is reported, not gated"}


# --------------------------------------------------------------------------------------
# T6 sensitivity
# --------------------------------------------------------------------------------------
#: constants that cannot act on an ISOLATED nucleus, because the test has no cortex and no
#: striatum.  They are inert by construction of the test, which is what v2's `w_CT_R` and
#: `w_CT_T` were, and they are listed apart from the ones that are inert in the model.
INERT_BY_CONSTRUCTION = ("w_ct_relay", "w_ct_trn", "w_tc_e", "w_tc_i", "w_ho_driver",
                         "w_cm_ctx", "w_cm_str", "d_relay_cortex", "d_cortex_trn",
                         "d_cortex_relay")


def _freq_with(name=None, value=None, steps=5000, seed=5):
    base = TH.C
    try:
        if name is not None:
            TH.C = dataclasses.replace(base, **{name: value})
        c = isolated("lgn")
        g = torch.Generator().manual_seed(seed)
        tr, _ = c.rollout(steps, DT, drive=tonic(c, "nrem2"), noise_gen=g, burn=500,
                          record=[TH.relay_id("lgn")])
        return spectrum(tr[TH.relay_id("lgn")].mean(-1), nperseg=2048)
    finally:
        TH.C = base


def _pinned_state(seconds=6.0, seed=5):
    """the range of every state variable over a run, and which ones never move.

    CLAUDE.md's instruction for a flat sweep: do not conclude the parameter does not
    matter, print the state variables and look for the one that is pinned.  This is that
    step, run unconditionally and recorded in the artefact, so the answer is in the JSON
    beside the sweep that raised the question instead of in someone's terminal.
    """
    c = isolated("lgn")
    st = c.init_state(1, DT)
    d = tonic(c, "nrem2")
    g = torch.Generator().manual_seed(seed)
    lo, hi = {}, {}
    for i in range(int(seconds / DT)):
        z = {p.id: torch.randn(1, p.n, generator=g) for p in c.pops.values() if p.sigma}
        st = c.step(st, DT, drive=d, noise=z)
        if i < int(1.0 / DT):
            continue
        for pid in c.pops:
            for k, v in st[pid].items():
                key = f"{pid}:{k}"
                m, M = float(v.min()), float(v.max())
                lo[key] = min(lo.get(key, m), m)
                hi[key] = max(hi.get(key, M), M)
    rows = {k: {"min": lo[k], "max": hi[k], "range": hi[k] - lo[k]} for k in lo}
    return {"variables": rows,
            "pinned_range_below_1e-3": sorted(k for k, v in rows.items()
                                              if v["range"] < 1e-3)}


def gate_sensitivity():
    f0, p0 = _freq_with()
    rows = []
    for name in [f.name for f in dataclasses.fields(TH.Constants)]:
        v = getattr(TH.C, name)
        flo, plo = _freq_with(name, v * 0.5)
        fhi, phi = _freq_with(name, v * 1.5)
        rows.append({"param": name, "value": v, "hz_at_half": flo, "hz_at_1p5x": fhi,
                     "span_hz": abs(fhi - flo), "prominence_at_half": plo,
                     "prominence_at_1p5x": phi})
        print(f"      {name:16s} {flo:6.2f} -> {fhi:6.2f} Hz   span {abs(fhi-flo):5.2f}",
              flush=True)
    rows.sort(key=lambda r: -r["span_hz"])
    clock = {r["param"]: r["span_hz"] for r in rows
             if r["param"] in ("tau_relay", "tau_gabaa")}
    inert = [r["param"] for r in rows
             if r["span_hz"] < 0.2 and r["param"] not in INERT_BY_CONSTRUCTION]
    by_construction = [r["param"] for r in rows
                       if r["span_hz"] < 0.2 and r["param"] in INERT_BY_CONSTRUCTION]
    return {"ok": all(v > 1.0 for v in clock.values()) and len(clock) == 2,
            "baseline_hz": f0, "baseline_prominence": p0,
            "claimed_clock_span_hz": clock,
            "rule": "tau_relay and tau_gabaa must each move the spindle frequency > 1 Hz",
            "inert_within_0.2hz": inert,
            "inert_by_construction_of_the_test": by_construction,
            "pinned_state": _pinned_state(),
            "sweep": rows}


# --------------------------------------------------------------------------------------
GATES = {"T0_bounded": gate_bounded, "T1_idempotent": gate_idempotent,
         "T2_rebound": gate_rebound, "T3_spindle_band": gate_spindle,
         "T4_selective_gating": gate_selective, "T5_higher_order_relay": gate_higher_order,
         "T6_sensitivity": gate_sensitivity}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", default=",".join(GATES))
    ap.add_argument("--out", default="out/gate_brain_thalamus.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    rec = {"script": "scripts/gate_brain_thalamus.py", "dt": DT, "gates": {}}
    if os.path.exists(a.out):
        try:
            with open(a.out) as fh:
                rec = json.load(fh)
            rec.setdefault("gates", {})
        except Exception:
            pass

    def save():
        # the cheap summary written BEFORE anything that can raise, and numpy/torch scalars
        # coerced rather than allowed to destroy the record (CLAUDE.md)
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1, default=str)

    # the assembly report belongs in the record: an orphan edge is a loop that has quietly
    # stopped existing, and it is the first thing to read when a gate surprises you
    import ibm.brain as B
    for label, names in (("assembly_cortex_thalamus", ["cortex", "thalamus"]),
                         ("assembly_all_available", None)):
        try:
            rec[label] = B.collect(names)[5]
        except Exception as e:
            rec[label] = {"error": repr(e)}
    rec["missing_afferents"] = TH.missing_afferents()
    save()

    for name in [g.strip() for g in a.gates.split(",") if g.strip()]:
        key = next((k for k in GATES if k.startswith(name)), None)
        if key is None:
            print(f"no such gate: {name}", flush=True)
            return 2
        t = time.time()
        r = GATES[key]()
        r["seconds"] = round(time.time() - t, 1)
        rec["gates"][key] = r
        save()
        head = {k: v for k, v in r.items()
                if k not in ("cases", "sweep", "detail", "seeds")}
        print(f"[{'PASS' if r.get('ok') else 'FAIL'}] {key}: "
              f"{json.dumps(head, default=str)[:900]}", flush=True)

    bad = [k for k, v in rec["gates"].items() if not v.get("ok")]
    rec["failed"] = bad
    rec["all_ok"] = not bad
    save()
    print(f"\nwrote {a.out}; recorded gates: {sorted(rec['gates'])}; FAILED: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
