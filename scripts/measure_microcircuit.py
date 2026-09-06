#!/usr/bin/env python
"""measure the cortical microcircuit, instead of quoting it.

`ibm/processes/neural.py` declares three processes over the local circuit --
`local_excitation`, `local_inhibition` and `laminar_propagation` -- and every
number in them that is not a receptor time constant is a `weak()` prior with a
note saying, correctly, that the connectivity behind it is "measured in mouse, in
slice, in a handful of areas".  that note is the honest position for a literature
value.  it is not the honest position any more, because dense EM reconstruction
with proofread axons now measures the same quantities directly, and this script
is the arithmetic that turns those reconstructions into numbers with spreads.

what is actually measurable, and from what
------------------------------------------
three sources, each answering a different question, and none of them answering
another's.

**MICrONS** (mouse visual cortex, ~1 mm^3, one animal).  the only source anywhere
with PROOFREAD AXONS: 564 cells whose axon has been traced by hand to the point
where its output within the volume is complete.  that word is what makes a
connection *probability* possible at all.  an automatically segmented axon breaks
every few tens of microns, so a connection probability computed from one is a
statement about the segmentation and not about the tissue; with a fully-extended
axon the numerator (targets this cell actually contacts) and the denominator
(targets whose soma is in the volume) are both complete, and the ratio means what
it says.  77 of the 564 are `axon_fully_extended`, and those 77 carry the primary
measurement.  the other two proofreading strategies are measured alongside them
and reported separately, because the gap between them IS the size of the
correction a non-proofread connectome would need.

**H01** (human temporal cortex, ~1 mm^3, one person, epilepsy surgery).  no
proofread axons to speak of -- 104 cells, and the released synapse table for them
carries no partner identity -- so it cannot give a connection probability that
means the same thing.  what it does give is a complete soma census with layer
labels, per-segment excitatory and inhibitory input counts over 46,637 segments,
and 100 of the 166 shards of the full synapse table.  those answer the questions
that do not need a complete axon: what fraction of neurons are inhibitory, how
inhibitory input is distributed over cells, how synapse density varies with depth
-- in HUMAN, which is the only reason the volume is worth carrying.

**Allen Cell Types** (2,333 patch-clamp recordings, 1,920 mouse and 413 human).
not connectivity at all.  it is here because three of the parameters this
exercise is aimed at -- `tau_membrane_s`, `tau_inh_membrane_s`, `v_rest_mv` --
are *cellular* rather than circuit quantities, they are measured in hundreds of
identified cells of both species, and the repository currently carries them as
literature point values with declared spreads that were never checked against a
distribution.

why the effective sample count is the hard part
-----------------------------------------------
`scripts/measure_tract_uncertainty.py` had 96 tractograms of one phantom and had
to answer "how many independent votes is that".  the same question here has a
worse answer.  one EM volume is one animal, one cortical area and one fixation,
and the 71,551 typed cells inside it are not 71,551 samples of anything: they share the
same segmentation model, the same proofreading team, the same tissue block and
the same six hundred microns of visual cortex.  so the script measures the
INTRACLASS correlation of the per-cell statistics across spatial blocks of the
volume and routes it through `ibm.runtime.fuse.TeacherPrecision.effective_
constraints`, exactly as the tractography measurement does, and reports what
77 proofread axons are actually worth.

the number that comes out is small, and reporting it is the point.  a microcircuit
prior derived from this is worth a handful of independent observations and must
carry a spread that says so -- which is still an enormous improvement on `weak()`,
because `weak()` is a factor of 10 with no measurement under it at all.

what this script will not do
----------------------------
it will not convert a mouse connection probability into a human one.  the two
volumes are measured separately and reported side by side, and the places where
they can be compared at all (inhibitory fraction, synapses per neuron, laminar
composition) are exactly the places where they disagree most.  see
`ibm/topologies/microcircuit_prior.py` for what is claimed to transfer and what
is not.

usage
-----
    ./.venv/bin/python scripts/measure_microcircuit.py            # everything
    ./.venv/bin/python scripts/measure_microcircuit.py --only microns
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MICRONS_ROOT = Path("/home/brandonin/Documents/IBM-1/data/sources/microns/public_data")
H01_ROOT = Path("/home/brandonin/Documents/IBM-1/data/sources/h01")
ALLEN_ROOT = Path("/home/brandonin/Documents/IBM-1/data/sources/allen-cell-types-patchseq")

TODAY = "2026-09-05"

#: MICrONS `cell_type` -> the population component of `ibm.fields.neural` it is
#: evidence about.  the four inhibitory classes are the reason this mapping is
#: worth writing down: BC is a perisomatic basket cell and is what `neural.pv` is
#: a population of, MC is a martinotti cell and is `neural.sst`, BPC is a bipolar
#: cell and is `neural.vip`.  NGC (neurogliaform) has NO component -- the field
#: declares three interneuron classes and neurogliaform cells are a fourth -- so
#: it is carried through the measurement under its own name and dropped at the
#: prior, rather than being folded into sst because it is also dendrite-targeting.
CLASS_OF = {
    "23P": "E", "4P": "E", "5P-IT": "E", "5P-ET": "E", "5P-NP": "E",
    "6P-IT": "E", "6P-CT": "E", "6P-U": "E", "WM-P": "E",
    "BC": "PV", "MC": "SST", "BPC": "VIP", "NGC": "NGC",
}
#: excitatory type -> layer.  this is a definition and not a measurement: the
#: classifier's labels ARE laminar ("23P" means a layer 2/3 pyramidal cell), so
#: using them is reading the annotation rather than inferring depth.  the
#: interneurons get no such label and are assigned by depth against boundaries
#: this script measures from the excitatory somata themselves.
LAYER_OF = {"23P": "L2/3", "4P": "L4", "5P-IT": "L5", "5P-ET": "L5",
            "5P-NP": "L5", "6P-IT": "L6", "6P-CT": "L6", "6P-U": "L6"}
LAYERS = ("L2/3", "L4", "L5", "L6")

#: intersomatic-distance bins, microns.  tight near zero because that is where
#: the probability is changing by a factor per bin, and open at the top because
#: the volume runs out at about 1.2 mm and the last bin is where truncation lives.
DIST_BINS = np.array([0., 25., 50., 75., 100., 150., 200., 250., 300., 400.,
                      500., 700., 1000.])


# ---------------------------------------------------------------------------
# shared arithmetic
# ---------------------------------------------------------------------------


def effective_n(n: int, rho: float) -> float:
    """`n` observations sharing a fraction `rho` of their error are worth how many.

    routed through `ibm.runtime.fuse` for the same reason
    `scripts/measure_tract_uncertainty.py` routes it there: that module is where
    the arithmetic is supposed to live, and a second implementation is a second
    place to get the sign backwards.  the ceiling is 1/rho however many cells the
    volume contains, which is the whole reason a million synapses in one cubic
    millimetre are not a million measurements.
    """
    if n <= 0 or not np.isfinite(rho):
        return 0.0
    rho = float(np.clip(rho, 0.0, 1.0 - 1e-9))
    try:
        from ibm.runtime.fuse import TeacherPrecision
        tp = TeacherPrecision(r2=0.0, correlated_fraction=rho, error_rank=1,
                              source="dense EM reconstruction, one volume")
        ev = tp.evidence("structural.synaptic_density", np.zeros(n), np.ones(n))
        return float(ev.effective_constraints())
    except Exception:
        return n / ((1.0 - rho) + n * rho)


def icc(values: np.ndarray, group: np.ndarray) -> tuple[float, int]:
    """fraction of the variance of `values` that is shared within a `group`.

    a one-way random-effects intraclass correlation, computed the plain way from
    the between- and within-group mean squares.  it is the same quantity a source
    card calls `correlated_fraction`, and here the groups are spatial blocks of
    one EM volume: two cells in the same block share a piece of tissue, a local
    segmentation quality and a proofreader, and the ICC is how much of their
    disagreement with the volume mean that accounts for.

    it is a LOWER bound on what a second animal would add.  everything that is
    constant across the whole volume -- the species, the cortical area, the
    fixation, the classifier -- has zero between-block variance by construction
    and is therefore invisible here.
    """
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    values, group = values[ok], np.asarray(group)[ok]
    if values.size < 4:
        return float("nan"), 0
    keys, inv = np.unique(group, return_inverse=True)
    k = len(keys)
    if k < 2:
        return float("nan"), k
    n = values.size
    counts = np.bincount(inv, minlength=k).astype(float)
    means = np.bincount(inv, weights=values, minlength=k) / np.maximum(counts, 1)
    grand = values.mean()
    ss_b = float(np.sum(counts * (means - grand) ** 2))
    ss_w = float(np.sum((values - means[inv]) ** 2))
    df_b, df_w = k - 1, n - k
    if df_w <= 0:
        return float("nan"), k
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    # the standard unbalanced n0.  with equal group sizes it is just the size.
    n0 = (n - float(np.sum(counts ** 2)) / n) / df_b
    var_b = max((ms_b - ms_w) / max(n0, 1e-9), 0.0)
    tot = var_b + ms_w
    return (var_b / tot if tot > 0 else 0.0), k


def exp_fit(d: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    """least squares on log p against d: p(d) = p0 exp(-d / lambda).

    weighted by the number of candidate pairs in each bin, because the bins at
    300 um carry twenty times the pairs of the bin at 25 um and an unweighted fit
    would let the noisiest, closest bin set the slope.  bins with no connections
    are dropped rather than floored -- a floor would invent a probability the data
    does not have and would flatten exactly the tail this is trying to measure.
    """
    m = (p > 0) & np.isfinite(p) & (w > 0)
    if m.sum() < 3:
        return {"lambda_um": None, "p0": None, "r2": None, "n_bins": int(m.sum())}
    x, y, ww = d[m], np.log(p[m]), w[m]
    A = np.vstack([np.ones_like(x), -x]).T
    W = np.diag(ww / ww.sum())
    beta = np.linalg.lstsq(W @ A, W @ y, rcond=None)[0]
    pred = A @ beta
    ss = float(np.sum(ww * (y - pred) ** 2))
    tot = float(np.sum(ww * (y - np.average(y, weights=ww)) ** 2))
    return {"lambda_um": float(1.0 / beta[1]) if beta[1] > 0 else None,
            "p0": float(math.exp(beta[0])), "r2": 1.0 - ss / tot if tot > 0 else None,
            "n_bins": int(m.sum())}


# ---------------------------------------------------------------------------
# MICrONS
# ---------------------------------------------------------------------------


def microns_input_ratio(neurons, axons) -> dict:
    """the fraction of a proofread cell's INPUT synapses that come from an inhibitory cell.

    the one quantity in this whole exercise that MICrONS and H01 both measure the
    same way, and therefore the only direct mouse-against-human comparison the two
    volumes support.  H01 reports it per segment from its released summary; here it
    is computed from the input-synapse table of the 564 cells whose dendrites were
    proofread, which is the closer analogue of a whole cell.

    it is biased and the direction is known.  a presynaptic partner counts only if
    it has a soma in the volume and a cell-type call, and inhibitory axons are far
    more locally complete than excitatory ones -- an excitatory input may well come
    from outside the millimetre and be unattributable, while an inhibitory one
    usually does not.  so this OVERSTATES the inhibitory share, which matters
    because the human figure it is being compared against is large.
    """
    import pyarrow.feather as pf

    path = MICRONS_ROOT / "syn_proofread_axons_all_in_microns_1078.feather"
    if not path.exists():
        return {}
    inp = pf.read_table(path, columns=["pre_pt_root_id", "post_pt_root_id"]).to_pandas()
    klass = dict(zip(neurons.pt_root_id.values, neurons.klass.values))
    post_ok = set(axons.pt_root_id.values) & set(klass)
    inp = inp[inp.post_pt_root_id.isin(post_ok)]
    pre_k = inp.pre_pt_root_id.map(klass)
    typed = pre_k.notna()
    inh = typed & (pre_k != "E")
    out = {"n_input_synapses": int(len(inp)),
           "typed_presynaptic_fraction": float(typed.mean()),
           "inhibitory_input_fraction_typed": float(inh.sum() / max(typed.sum(), 1))}
    per = {}
    for cls_ in ("E", "PV", "SST", "VIP"):
        sel = inp.post_pt_root_id.map(klass) == cls_
        t = typed & sel
        if t.sum() > 1000:
            per[cls_] = float((inh & sel).sum() / t.sum())
    out["by_postsynaptic_class"] = per
    return out


def load_microns():
    import pyarrow.feather as pf
    import pandas as pd

    auto = pf.read_table(MICRONS_ROOT / "cell_types_microns_1078_auto.feather").to_pandas()
    man = pf.read_table(MICRONS_ROOT / "cell_types_microns_1078_manual.feather").to_pandas()
    axons = pf.read_table(MICRONS_ROOT / "proofread_axons_microns_1078.feather").to_pandas()
    syn = pf.read_table(MICRONS_ROOT / "syn_proofread_axons_all_out_microns_1078.feather"
                        ).to_pandas()

    cells = auto[["pt_root_id", "cell_type", "classification_system",
                  "pt_position_x", "pt_position_y", "pt_position_z"]].copy()
    # the manual labels are the same cells relabelled by hand and they win where
    # they exist.  1,357 of 90,296 is not enough to change any aggregate, and
    # taking them anyway is the cheap half of a disagreement measurement: the
    # agreement rate between the two is reported, and it is the classifier's own
    # error rate on the population this whole measurement is stratified by.
    man_lab = man.set_index("pt_root_id").cell_type
    agree = cells.set_index("pt_root_id").cell_type.reindex(man_lab.index)
    both = agree.notna() & man_lab.notna()
    classifier_agreement = float((agree[both] == man_lab[both]).mean())
    cells["cell_type"] = cells.pt_root_id.map(man_lab).fillna(cells.cell_type)
    cells["klass"] = cells.cell_type.map(CLASS_OF)
    cells["layer_label"] = cells.cell_type.map(LAYER_OF)
    for c, n in (("x", "pt_position_x"), ("y", "pt_position_y"), ("z", "pt_position_z")):
        cells[c] = cells[n] / 1000.0        # nm -> um
    neurons = cells[cells.klass.notna()].reset_index(drop=True)
    return neurons, axons, syn, classifier_agreement, len(auto), int(both.sum())


def layer_boundaries(neurons) -> dict:
    """the depth of each laminar boundary, read off the somata rather than a paper.

    for each adjacent pair of excitatory types the boundary is the depth that
    minimizes the number of cells on the wrong side of it.  that is a real
    measurement of where the classifier's laminar labels separate, and it is what
    the interneurons are then assigned against -- so a martinotti cell is called
    layer 5 because it sits below the depth that best separates 4P from 5P cells
    in this volume, not because a literature boundary was imported into a frame
    it was never measured in.
    """
    out = {}
    for upper, lower, name in (("23P", "4P", "L2/3-L4"), ("4P", "5P-IT", "L4-L5"),
                               ("5P-IT", "6P-IT", "L5-L6")):
        du = neurons[neurons.cell_type == upper].y.values
        dl = neurons[neurons.cell_type == lower].y.values
        if du.size < 10 or dl.size < 10:
            continue
        grid = np.linspace(min(du.min(), dl.min()), max(du.max(), dl.max()), 4000)
        err = np.array([(du > g).sum() + (dl <= g).sum() for g in grid])
        j = int(np.argmin(err))
        out[name] = {"depth_um": float(grid[j]),
                     "misassigned": int(err[j]), "n": int(du.size + dl.size),
                     "separability": 1.0 - float(err[j]) / float(du.size + dl.size)}
    return out


def assign_layers(neurons, bounds: dict):
    """excitatory cells keep their label; everyone else is placed by depth."""
    b23_4 = bounds["L2/3-L4"]["depth_um"]
    b4_5 = bounds["L4-L5"]["depth_um"]
    b5_6 = bounds["L5-L6"]["depth_um"]
    y = neurons.y.values
    depth_layer = np.where(y < b23_4, "L2/3",
                           np.where(y < b4_5, "L4", np.where(y < b5_6, "L5", "L6")))
    lab = neurons.layer_label.values.astype(object)
    return np.where([isinstance(v, str) for v in lab], lab, depth_layer)


def microns_measure(neurons, axons, syn, extras: dict) -> dict:
    import pandas as pd

    bounds = layer_boundaries(neurons)
    neurons = neurons.copy()
    neurons["layer"] = assign_layers(neurons, bounds)

    pre = axons.merge(neurons, on="pt_root_id", how="inner")
    idx = {int(r): i for i, r in enumerate(neurons.pt_root_id.values)}
    xyz = neurons[["x", "y", "z"]].values
    klass = neurons.klass.values
    layer = neurons.layer.values
    ids = neurons.pt_root_id.values

    # one row per (pre, post) pair, with the synapse count.  autapses are dropped:
    # a cell contacting itself is a real thing and is not what "connection
    # probability between two populations" means.
    pairs = (syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size()
             .rename("nsyn").reset_index())
    pairs = pairs[pairs.pre_pt_root_id != pairs.post_pt_root_id]
    in_vol = pairs.post_pt_root_id.isin(idx)
    orphan_fraction = 1.0 - float(in_vol.mean())
    pairs = pairs[in_vol]

    edge_n = {}
    for a, b, n in zip(pairs.pre_pt_root_id.values, pairs.post_pt_root_id.values,
                       pairs.nsyn.values):
        edge_n[(int(a), int(b))] = int(n)

    volume = {"x_um": [float(xyz[:, 0].min()), float(xyz[:, 0].max())],
              "y_um": [float(xyz[:, 1].min()), float(xyz[:, 1].max())],
              "z_um": [float(xyz[:, 2].min()), float(xyz[:, 2].max())]}

    # spatial blocks for the intraclass correlation: a 3 x 3 grid in the two
    # tangential axes.  the depth axis is deliberately NOT split, because depth is
    # the thing the laminar measurement is about and blocking on it would move a
    # real laminar gradient into the "shared error" term.
    qx = np.quantile(xyz[:, 0], [1 / 3, 2 / 3])
    qz = np.quantile(xyz[:, 2], [1 / 3, 2 / 3])
    block_of = lambda p: int(np.searchsorted(qx, p[0]) * 3 + np.searchsorted(qz, p[2]))

    def curves(pre_rows):
        """numerator and denominator per (post class, distance bin) for a pre set."""
        nb = len(DIST_BINS) - 1
        classes = ["E", "PV", "SST", "VIP", "NGC"]
        num = {c: np.zeros(nb) for c in classes}
        den = {c: np.zeros(nb) for c in classes}
        per_cell = []
        for r in pre_rows.pt_root_id.values:
            i = idx[int(r)]
            dt = np.hypot(xyz[:, 0] - xyz[i, 0], xyz[:, 2] - xyz[i, 2])
            bi = np.digitize(dt, DIST_BINS) - 1
            ok = (bi >= 0) & (bi < nb) & (ids != r)
            hit = np.array([(int(r), int(t)) in edge_n for t in ids])
            for c in classes:
                sel = ok & (klass == c)
                np.add.at(den[c], bi[sel], 1.0)
                np.add.at(num[c], bi[sel & hit], 1.0)
            per_cell.append((int(r), int(hit[ok].sum()), int(ok.sum()),
                             block_of(xyz[i])))
        return num, den, per_cell

    strategies = {}
    for strat in ("axon_fully_extended", "axon_partially_extended", "axon_interareal"):
        rows = pre[pre.strategy_axon == strat]
        if rows.empty:
            continue
        out = {}
        for src_class in ("E", "PV", "SST", "VIP"):
            sub = rows[rows.klass == src_class]
            if len(sub) < 3:
                continue
            num, den, per_cell = curves(sub)
            centres = 0.5 * (DIST_BINS[:-1] + DIST_BINS[1:])
            entry = {"n_pre": int(len(sub)), "by_post_class": {}}
            for c, nn in num.items():
                dd = den[c]
                if dd.sum() < 100:
                    continue
                p = np.divide(nn, np.maximum(dd, 1))
                entry["by_post_class"][c] = {
                    "p_overall": float(nn.sum() / max(dd.sum(), 1)),
                    "n_connected": int(nn.sum()), "n_candidate_pairs": int(dd.sum()),
                    "p_within_100um": float(nn[:4].sum() / max(dd[:4].sum(), 1)),
                    "p_by_bin": [round(float(v), 6) for v in p],
                    "n_by_bin": [int(v) for v in dd],
                    "fit": exp_fit(centres, p, dd),
                }
            # the intraclass correlation of per-cell out-degree across blocks.
            deg = np.array([np.log(max(h, 0.5) / max(t, 1)) for _, h, t, _ in per_cell])
            blk = np.array([b for *_, b in per_cell])
            r, k = icc(deg, blk)
            entry["per_cell_log_p"] = {
                "mean": float(np.mean(deg)), "sd": float(np.std(deg, ddof=1))
                if len(deg) > 1 else None,
                "n_cells": int(len(deg)), "n_blocks": int(k),
                "icc_across_blocks": None if not np.isfinite(r) else float(r),
                "effective_cells": (None if not np.isfinite(r)
                                    else effective_n(len(deg), r)),
            }
            out[src_class] = entry
        strategies[strat] = out

    # synapses per connection, per class pair, over every proofread axon.  this
    # one does NOT need a complete axon: a connection that was found was found,
    # and how many synapses it carries is a property of the connection rather
    # than of how much of the axon was traced.
    pre_class = {int(r): k for r, k in zip(pre.pt_root_id.values, pre.klass.values)}
    post_class = {int(r): k for r, k in zip(ids, klass)}
    spc = {}
    for (a, b), n in edge_n.items():
        ca, cb = pre_class.get(a), post_class.get(b)
        if ca is None or cb is None:
            continue
        spc.setdefault(f"{ca}->{cb}", []).append(n)
    syn_per_conn = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                        "median": float(np.median(v)),
                        "frac_multi": float(np.mean(np.asarray(v) > 1)),
                        "p90": float(np.quantile(v, 0.9)), "max": int(max(v)),
                        "n_connections": len(v)}
                    for k, v in sorted(spc.items()) if len(v) >= 30}

    # laminar specificity.  restricted to excitatory pre and excitatory post,
    # because the L4 -> L2/3 -> L5 -> L6 motif `laminar_propagation` declares is a
    # statement about the principal-cell cascade and folding interneurons into it
    # would average a within-layer inhibitory loop into a translaminar gain.
    lam_num = {a: {b: 0.0 for b in LAYERS} for a in LAYERS}
    lam_den = {a: {b: 0.0 for b in LAYERS} for a in LAYERS}
    lam_pre_n = {a: 0 for a in LAYERS}
    pre_layer = {int(r): l for r, l in zip(pre.pt_root_id.values,
                                           assign_layers(pre, bounds))}
    exc_pre = pre[(pre.klass == "E")
                  & pre.strategy_axon.isin(["axon_fully_extended",
                                            "axon_partially_extended"])]
    is_e = klass == "E"
    for r in exc_pre.pt_root_id.values:
        r = int(r)
        la = pre_layer.get(r)
        if la not in lam_num:
            continue
        lam_pre_n[la] += 1
        hit = np.array([(r, int(t)) in edge_n for t in ids])
        for lb in LAYERS:
            sel = is_e & (layer == lb) & (ids != r)
            lam_den[la][lb] += float(sel.sum())
            lam_num[la][lb] += float((sel & hit).sum())
    laminar = {a: {b: {"p": (lam_num[a][b] / lam_den[a][b]) if lam_den[a][b] else None,
                       "n_connected": int(lam_num[a][b]),
                       "n_pairs": int(lam_den[a][b])}
                   for b in LAYERS} for a in LAYERS}

    # the same matrix with distance divided out, and this is the one that means
    # what "laminar specificity" is supposed to mean.  the raw matrix above cannot
    # separate "layer 6 cells rarely contact layer 4 cells" from "layer 6 somata
    # are 300 microns from layer 4 somata and connection probability falls by an
    # order of magnitude over that distance".  here each candidate pair is given
    # the probability the pre cell's own class-wide p(d) curve predicts at its
    # intersomatic distance, the expectations are summed per layer pair, and the
    # ratio of observed to expected is what is left after distance is accounted
    # for.  1.0 means "exactly as often as distance alone predicts"; the laminar
    # motif is the claim that some pairs are well above it.
    fe_e = (m_fe := strategies.get("axon_fully_extended", {}).get("E"))
    if fe_e and "E" in fe_e["by_post_class"]:
        p_of_bin = np.asarray(fe_e["by_post_class"]["E"]["p_by_bin"], float)
        obs = {a: {b: 0.0 for b in LAYERS} for a in LAYERS}
        exp = {a: {b: 0.0 for b in LAYERS} for a in LAYERS}
        nb = len(DIST_BINS) - 1
        for r in exc_pre.pt_root_id.values:
            r = int(r)
            la = pre_layer.get(r)
            if la not in obs:
                continue
            i = idx[r]
            d3 = np.linalg.norm(xyz - xyz[i], axis=1)
            bi = np.digitize(d3, DIST_BINS) - 1
            hit = np.array([(r, int(t)) in edge_n for t in ids])
            for lb in LAYERS:
                sel = is_e & (layer == lb) & (ids != r) & (bi >= 0) & (bi < nb)
                obs[la][lb] += float((sel & hit).sum())
                exp[la][lb] += float(p_of_bin[bi[sel]].sum())
        specificity = {a: {b: {"observed": int(obs[a][b]), "expected": exp[a][b],
                               "ratio": (obs[a][b] / exp[a][b]) if exp[a][b] > 0 else None}
                           for b in LAYERS} for a in LAYERS}
    else:
        specificity = {}

    def asym(a, b):
        pab = laminar[a][b]["p"]
        pba = laminar[b][a]["p"]
        if not pab or not pba:
            return None
        out = {"forward": pab, "backward": pba, "ratio": pab / pba,
               "log_ratio": math.log(pab / pba)}
        if specificity:
            sab = specificity[a][b]["ratio"]
            sba = specificity[b][a]["ratio"]
            if sab and sba:
                out.update({"forward_specificity": sab, "backward_specificity": sba,
                            "specificity_ratio": sab / sba})
        return out

    motif = {"L4->L2/3": asym("L4", "L2/3"), "L2/3->L5": asym("L2/3", "L5"),
             "L5->L6": asym("L5", "L6"), "L6->L4": asym("L6", "L4")}

    # the lateral extent question, asked the way a materialization has to ask it:
    # if a column node stands for a disc of cortex of radius R, what fraction of a
    # cell's outgoing connections does that disc contain?  this is the number that
    # says whether `microcircuit`'s within-site self-edge is a description or an
    # approximation, and at what site spacing it stops being either.
    lateral = {}
    for strat in ("axon_fully_extended", "axon_partially_extended"):
        rows = pre[pre.strategy_axon == strat]
        for src in ("E", "PV", "SST", "VIP"):
            sub = rows[rows.klass == src]
            if len(sub) < 3:
                continue
            dists = []
            for r in sub.pt_root_id.values:
                i = idx[int(r)]
                hit = np.array([(int(r), int(t)) in edge_n for t in ids])
                if not hit.any():
                    continue
                dists.append(np.hypot(xyz[hit, 0] - xyz[i, 0], xyz[hit, 2] - xyz[i, 2]))
            if not dists:
                continue
            dd = np.concatenate(dists)
            lateral[f"{strat}:{src}"] = {
                "n_pre": int(len(sub)), "n_connections": int(dd.size),
                "median_um": float(np.median(dd)), "p90_um": float(np.quantile(dd, .9)),
                "frac_within_100um": float(np.mean(dd < 100)),
                "frac_within_200um": float(np.mean(dd < 200)),
                "frac_within_300um": float(np.mean(dd < 300)),
                "frac_within_500um": float(np.mean(dd < 500))}

    # what proofreading is worth, measured.  the same quantity computed from
    # fully-extended and from partially-extended axons of the same class differs
    # by a factor that is entirely an artefact of how much axon was traced, and
    # reporting it is what lets a reader convert any published non-proofread
    # connection probability -- which is most of them -- into this one's units.
    proofreading = {}
    fe_g = strategies.get("axon_fully_extended", {})
    pe_g = strategies.get("axon_partially_extended", {})
    for src in set(fe_g) & set(pe_g):
        for post in set(fe_g[src]["by_post_class"]) & set(pe_g[src]["by_post_class"]):
            a = fe_g[src]["by_post_class"][post]["p_within_100um"]
            b = pe_g[src]["by_post_class"][post]["p_within_100um"]
            if a and b:
                proofreading[f"{src}->{post}"] = a / b

    census = {"n_neurons": int(len(neurons)),
              "by_class": {k: int(v) for k, v in
                           neurons.klass.value_counts().items()},
              "by_type": {k: int(v) for k, v in
                          neurons.cell_type.value_counts().items()},
              "by_layer": {k: int(v) for k, v in
                           pd.Series(neurons.layer).value_counts().items()},
              "inhibitory_fraction": float((neurons.klass != "E").mean()),
              "inhibitory_fraction_by_layer": {
                  l: float((neurons.klass[neurons.layer == l] != "E").mean())
                  for l in LAYERS},
              "classifier_agreement_with_manual": extras["agreement"],
              "n_manual_labels_compared": extras["n_manual"],
              "n_rows_auto_table": extras["n_auto"]}

    return {"volume_um": volume, "layer_boundaries": bounds, "census": census,
            "input_ratio": microns_input_ratio(neurons, pre),
            "orphan_target_fraction": orphan_fraction,
            "connection_probability": strategies,
            "synapses_per_connection": syn_per_conn,
            "lateral_extent": lateral,
            "proofreading_correction_p_fully_over_partially": proofreading,
            "laminar": {"p_matrix": laminar, "n_pre_by_layer": lam_pre_n,
                        "distance_controlled_specificity": specificity,
                        "canonical_motif": motif}}


# ---------------------------------------------------------------------------
# H01
# ---------------------------------------------------------------------------


def _h01_shard(path: str):
    """reduce one avro shard to the four columns any of this needs.

    a worker function at module scope because it has to be picklable.  the shard
    is 200 MB of nested avro and 1 million synapses; what survives is the pre and
    post segment ids and the synapse class, which is 16 bytes a row.
    """
    import fastavro
    # the postsynaptic compartment is carried alongside the synapse class because
    # it is what LETS the class be checked.  H01 documents synapse type 2 as
    # excitatory and 1 as inhibitory, and taking a released convention on trust is
    # how a sign gets inverted silently -- so the reduction keeps the compartment
    # label and the measurement cross-tabulates the two.  inhibitory synapses
    # target somata and axon initial segments and excitatory ones do not, so the
    # enrichment settles which code is which from the data itself.
    codes = {"DENDRITE": 0, "SOMA": 1, "AIS": 2, "AXON": 3}
    pre, post, typ, comp = [], [], [], []
    with open(path, "rb") as fh:
        for rec in fastavro.reader(fh):
            a = rec.get("pre_synaptic_site") or {}
            b = rec.get("post_synaptic_partner") or {}
            pa, pb = a.get("neuron_id"), b.get("neuron_id")
            if pa is None or pb is None:
                continue
            pre.append(pa)
            post.append(pb)
            typ.append(rec.get("type") or 0)
            comp.append(codes.get(b.get("class_label") or "", 4))
    return (np.asarray(pre, np.int64), np.asarray(post, np.int64),
            np.asarray(typ, np.int8), np.asarray(comp, np.int8))


def h01_measure(max_workers: int = 8) -> dict:
    import pandas as pd

    somas = pd.read_csv(H01_ROOT / "derived" / "data_20210601_c3_tables_somas.csv")
    props = json.load(gzip.open(H01_ROOT / "derived"
                                / "data_20210601_c3_segment_properties_info"))["inline"]
    tags = None
    per_seg = {}
    for p in props["properties"]:
        if p["type"] == "tags":
            tags = p["tags"]
            tag_values = p["values"]
        elif p["type"] == "number":
            per_seg[p["id"]] = np.asarray(p["values"], float)
    seg_ids = np.asarray([int(s) for s in props["ids"]], np.int64)

    # the depth axis, fitted rather than assumed.  H01's block was cut at an angle
    # to the pia, so neither x nor y is depth: the layer centroids run diagonally
    # through both.  the first principal direction of the seven layer centroids IS
    # the pia-to-white axis, and using it is the difference between a depth
    # profile and a projection of one.
    lay_order = ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6",
                 "White matter"]
    cent = np.array([[somas.loc[somas.layer == l, c].mean() for c in ("x", "y", "z")]
                     for l in lay_order])
    axis = cent[-1] - cent[0]
    # the voxel grid is 8 x 8 x 33 nm, so a raw difference is not a distance until
    # the anisotropy is taken out.
    vox = np.array([8.0, 8.0, 33.0]) / 1000.0        # um per voxel
    cent_um = cent * vox
    axis = cent_um[-1] - cent_um[0]
    axis = axis / np.linalg.norm(axis)
    soma_um = somas[["x", "y", "z"]].values * vox
    depth = (soma_um - cent_um[0]) @ axis
    somas = somas.assign(depth_um=depth)

    exc = {"PYRAMIDAL", "SPINY_STELLATE", "SPINY_ATYPICAL"}
    inh = {"INTERNEURON"}
    n_exc = int(somas.celltype.isin(exc).sum())
    n_inh = int(somas.celltype.isin(inh).sum())

    census = {
        "n_somas": int(len(somas)),
        "by_celltype": {k: int(v) for k, v in somas.celltype.value_counts().items()},
        "by_layer": {k: int(v) for k, v in somas.layer.value_counts().items()},
        "n_excitatory": n_exc, "n_inhibitory": n_inh,
        "inhibitory_fraction_of_classified_neurons": n_inh / max(n_exc + n_inh, 1),
        "inhibitory_fraction_by_layer": {
            l: float(somas.loc[somas.layer == l, "celltype"].isin(inh).sum()
                     / max(somas.loc[somas.layer == l, "celltype"].isin(exc | inh).sum(), 1))
            for l in lay_order},
        "layer_depth_um": {l: float(somas.loc[somas.layer == l, "depth_um"].mean())
                           for l in lay_order},
        "cortical_depth_span_um": float(np.linalg.norm(cent_um[-1] - cent_um[0])),
        "depth_axis_unit_vector": [float(v) for v in axis],
    }

    # per-segment excitatory and inhibitory input counts.  these are complete --
    # they are the released summary over the whole volume -- so the E/I input
    # ratio they give is a real human number and not a sample statistic.
    seg_tags = {int(s): set(t) for s, t in zip(props["ids"], tag_values)}
    name = {i: t for i, t in enumerate(tags)}
    def has(sid, label):
        return any(name[i] == label for i in seg_tags.get(sid, ()))
    nsie, nsii, nso = per_seg.get("NSIe"), per_seg.get("NSIi"), per_seg.get("NSO")
    ratio = nsii / np.maximum(nsie + nsii, 1)
    is_pyr = np.array([has(int(s), "pyramidal") for s in seg_ids])
    is_int = np.array([has(int(s), "interneuron") for s in seg_ids])
    ei = {"n_segments": int(len(seg_ids)),
          "inhibitory_input_fraction_all": float(np.mean(ratio)),
          "inhibitory_input_fraction_pyramidal": float(np.mean(ratio[is_pyr]))
          if is_pyr.any() else None,
          "inhibitory_input_fraction_interneuron": float(np.mean(ratio[is_int]))
          if is_int.any() else None,
          "n_pyramidal": int(is_pyr.sum()), "n_interneuron": int(is_int.sum()),
          "median_inputs_per_segment": float(np.median(nsie + nsii)),
          "median_outputs_per_segment": float(np.median(nso))}

    # the sampled synapse table.
    shard_dir = H01_ROOT / "synapses_exported"
    shards = sorted(str(p) for p in shard_dir.glob("export*"))
    result = {"census": census, "ei_input": ei,
              "synapse_sample": {"n_shards": len(shards), "n_shards_total": 166}}
    if not shards:
        return result

    pre_l, post_l, typ_l, comp_l = [], [], [], []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for a, b, t, c in ex.map(_h01_shard, shards):
            pre_l.append(a); post_l.append(b); typ_l.append(t); comp_l.append(c)
    pre = np.concatenate(pre_l); post = np.concatenate(post_l)
    typ = np.concatenate(typ_l); comp = np.concatenate(comp_l)
    del pre_l, post_l, typ_l, comp_l

    # what fraction of the volume's synapses did those shards actually contain?
    # the released per-segment output count NSO is over the WHOLE volume, so the
    # ratio of what was sampled to what NSO says is the sampling fraction --
    # measured rather than assumed to be n_shards / 166.  it is the only check
    # available that the shards are a random partition and not a spatial one.
    order = np.argsort(seg_ids)
    sk, sv = seg_ids[order], nso[order]
    cnt = np.bincount(np.searchsorted(sk, pre[np.isin(pre, sk)]),
                      minlength=sk.size).astype(float)
    known = sv > 20
    f_hat = float(cnt[known].sum() / max(sv[known].sum(), 1))
    per_seg_ratio = cnt[known] / sv[known]
    result["synapse_sample"].update({
        "n_synapses_sampled": int(pre.size),
        "sampling_fraction_measured": f_hat,
        "sampling_fraction_nominal": len(shards) / 166.0,
        "per_segment_sampling_ratio_sd": float(np.std(per_seg_ratio)),
        "per_segment_sampling_ratio_iqr": [float(np.quantile(per_seg_ratio, .25)),
                                           float(np.quantile(per_seg_ratio, .75))],
        "synapse_type_counts": {str(int(k)): int(v) for k, v in
                                zip(*np.unique(typ, return_counts=True))},
    })
    # which synapse code is inhibitory, settled from the tissue rather than from
    # the release note.  a perisomatic enrichment above 1 means that code's
    # synapses land on somata and axon initial segments more often than the other
    # code's, which is what inhibition does and excitation does not.
    soma_ais = np.isin(comp, (1, 2))
    frac = {int(k): float(soma_ais[typ == k].mean()) for k in np.unique(typ)}
    keys = sorted(frac, key=lambda k: -frac[k])
    inhibitory_code = keys[0] if len(keys) > 1 else None
    result["synapse_sample"].update({
        "perisomatic_fraction_by_type": {str(k): v for k, v in frac.items()},
        "inferred_inhibitory_type_code": inhibitory_code,
        "perisomatic_enrichment": (frac[keys[0]] / max(frac[keys[-1]], 1e-9)
                                   if len(keys) > 1 else None),
        "inhibitory_synapse_fraction": (float(np.mean(typ == inhibitory_code))
                                        if inhibitory_code is not None else None),
        "postsynaptic_compartment_counts": {
            n: int((comp == c).sum()) for n, c in
            (("dendrite", 0), ("soma", 1), ("ais", 2), ("axon", 3), ("other", 4))},
    })

    # soma-to-soma connectivity, for what it is worth -- and the docstring above
    # says what that is.  H01's axons are not proofread, so this is a LOWER bound
    # on connection probability by an unknown factor; what it can still show is
    # the SHAPE of the decay with intersomatic distance, which is the thing the
    # cross-species comparison actually needs.
    soma_seg = somas.dropna(subset=["c3_rep_strict"]).copy()
    soma_seg["seg"] = soma_seg.c3_rep_strict.astype(np.int64)
    seg2row = {int(s): i for i, s in enumerate(soma_seg.seg.values)}
    su = soma_seg[["x", "y", "z"]].values * vox
    scls = np.where(soma_seg.celltype.isin(exc), "E",
                    np.where(soma_seg.celltype.isin(inh), "I", "other"))
    m = np.array([s in seg2row for s in pre]) & np.array([s in seg2row for s in post])
    ia = np.array([seg2row[int(s)] for s in pre[m]])
    ib = np.array([seg2row[int(s)] for s in post[m]])
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    d = np.linalg.norm(su[ia] - su[ib], axis=1)
    result["soma_to_soma"] = {
        "n_synapses_between_somas": int(ia.size),
        "fraction_of_sampled_synapses": float(ia.size / max(pre.size, 1)),
        "n_unique_pairs": int(len({(int(a), int(b)) for a, b in zip(ia, ib)})),
        "intersomatic_distance_um": {
            "median": float(np.median(d)) if d.size else None,
            "p10": float(np.quantile(d, .1)) if d.size else None,
            "p90": float(np.quantile(d, .9)) if d.size else None,
            "frac_within_100um": float(np.mean(d < 100)) if d.size else None,
            "frac_within_250um": float(np.mean(d < 250)) if d.size else None},
        "by_class_pair": {f"{a}->{b}": int(np.sum((scls[ia] == a) & (scls[ib] == b)))
                          for a in ("E", "I") for b in ("E", "I")},
    }
    return result


# ---------------------------------------------------------------------------
# Allen Cell Types
# ---------------------------------------------------------------------------


def allen_measure() -> dict:
    """membrane kinetics of identified cells, human and mouse, as distributions.

    the reason this belongs in a microcircuit measurement: `tau_membrane_s`,
    `tau_inh_membrane_s` and `v_rest_mv` are declared LITERATURE in
    `ibm/processes/neural.py` with spreads chosen to look reasonable, and they are
    the only parameters in the three target processes that a per-cell recording
    can settle directly.  there is nothing to infer here -- the database reports
    tau per cell -- so the whole measurement is a grouping and a spread.

    the split that matters is not human against mouse.  it is fast-spiking
    against everything else: `tau_inh_membrane_s` is a claim about pv basket
    cells specifically, and the Pvalb-IRES-Cre line is what identifies them.
    """
    import collections
    recs = json.load(open(ALLEN_ROOT / "cell_types_specimen_details.json"))["msg"]

    def group(rows, key):
        v = np.array([r[key] for r in rows if r.get(key) is not None], float)
        v = v[np.isfinite(v)]
        if v.size < 5:
            return None
        return {"n": int(v.size), "median": float(np.median(v)),
                "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "log_sd": float(np.std(np.log(v[v > 0]), ddof=1)) if (v > 0).all()
                else None,
                "p10": float(np.quantile(v, .1)), "p90": float(np.quantile(v, .9))}

    human = [r for r in recs if r["donor__species"] == "Homo Sapiens"]
    mouse = [r for r in recs if r["donor__species"] == "Mus musculus"]
    spiny = lambda rows: [r for r in rows if r.get("tag__dendrite_type") == "spiny"]
    line = lambda rows, s: [r for r in rows if s in (r.get("line_name") or "")]

    groups = {
        "mouse_spiny": spiny(mouse), "human_spiny": spiny(human),
        "mouse_aspiny": [r for r in mouse if r.get("tag__dendrite_type") == "aspiny"],
        "human_aspiny": [r for r in human if r.get("tag__dendrite_type") == "aspiny"],
        "mouse_pvalb": line(mouse, "Pvalb"), "mouse_sst": line(mouse, "Sst"),
        "mouse_vip": line(mouse, "Vip"),
    }
    out = {"n_records": len(recs), "n_human": len(human), "n_mouse": len(mouse),
           "by_group": {}}
    for name, rows in groups.items():
        out["by_group"][name] = {
            "n": len(rows),
            "tau_ms": group(rows, "ef__tau"),
            "vrest_mv": group(rows, "ef__vrest"),
            "ri_mohm": group(rows, "ef__ri"),
            "adaptation": group(rows, "ef__adaptation"),
            "avg_firing_rate_hz": group(rows, "ef__avg_firing_rate"),
            "f_i_slope_hz_per_pa": group(rows, "ef__f_i_curve_slope"),
            "upstroke_downstroke_ratio": group(
                rows, "ef__upstroke_downstroke_ratio_long_square"),
        }
    out["by_layer_mouse_spiny_tau_ms"] = {}
    lay = collections.defaultdict(list)
    for r in spiny(mouse):
        lay[str(r.get("structure__layer"))].append(r)
    for k, rows in sorted(lay.items()):
        g = group(rows, "ef__tau")
        if g:
            out["by_layer_mouse_spiny_tau_ms"][k] = g
    # human against mouse, on the one parameter both measure the same way.
    hm, mm = out["by_group"]["human_spiny"]["tau_ms"], out["by_group"]["mouse_spiny"]["tau_ms"]
    if hm and mm:
        out["human_over_mouse_tau_ratio"] = hm["median"] / mm["median"]
    hv = out["by_group"]["human_spiny"]["vrest_mv"]
    mv = out["by_group"]["mouse_spiny"]["vrest_mv"]
    if hv and mv:
        out["human_minus_mouse_vrest_mv"] = hv["median"] - mv["median"]
    return out


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def write_evidence(source: str, recipe: str, version: int, payload: dict,
                   manifest: dict) -> Path:
    d = REPO / "data" / "sources" / source / "evidence" / f"{recipe}@{version}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "measured.json").write_text(json.dumps(payload, indent=1, sort_keys=False))
    import yaml
    (d / "manifest.yaml").write_text(
        "# what this recipe produced, from which raw source.  committed.\n"
        + yaml.safe_dump(manifest, sort_keys=False, width=100))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    choices=["all", "microns", "h01", "allen"])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = ap.parse_args()

    if args.only in ("all", "microns"):
        neurons, axons, syn, agreement, n_auto, n_manual = load_microns()
        m = microns_measure(neurons, axons, syn,
                            {"agreement": agreement, "n_auto": n_auto,
                             "n_manual": n_manual})
        m["sources"] = ["MICrONS minnie65 public release, materialization 1078; "
                        "proofread axon table, automatic and manual cell types, "
                        "and all output synapses of the proofread axons"]
        m["measured_at"] = TODAY
        m["species"] = "Mus musculus, visual cortex (V1/HVA), one animal, one volume"
        fe = m["connection_probability"].get("axon_fully_extended", {})
        d = write_evidence(
            "microns", "microcircuit_statistics", 1, m,
            {"recipe": "microcircuit_statistics", "version": 1,
             "produced_at": TODAY,
             "from_raw": {"microns": "public_data/*.feather, 342 MB, materialization 1078"},
             "outputs": ["measured.json"],
             "headline": {
                 "n_neurons_with_type": m["census"]["n_neurons"],
                 "inhibitory_fraction": round(m["census"]["inhibitory_fraction"], 4),
                 "n_fully_extended_axons": sum(v["n_pre"] for v in fe.values()),
                 "E_to_E_p_within_100um": round(
                     fe.get("E", {}).get("by_post_class", {}).get("E", {})
                     .get("p_within_100um", float("nan")), 5),
                 "synapses_per_connection_E_to_E": round(
                     m["synapses_per_connection"].get("E->E", {}).get("mean", float("nan")), 3),
                 "synapses_per_connection_PV_to_E": round(
                     m["synapses_per_connection"].get("PV->E", {}).get("mean", float("nan")), 3),
                 "E_lambda_um": round(
                     (fe.get("E", {}).get("by_post_class", {}).get("E", {})
                      .get("fit", {}) or {}).get("lambda_um") or float("nan"), 1),
                 "SST_to_E_lambda_um": round(
                     (fe.get("SST", {}).get("by_post_class", {}).get("E", {})
                      .get("fit", {}) or {}).get("lambda_um") or float("nan"), 1),
                 "E_output_within_200um": (
                     m["lateral_extent"].get("axon_fully_extended:E", {})
                     .get("frac_within_200um")),
                 "SST_output_within_200um": (
                     m["lateral_extent"].get("axon_fully_extended:SST", {})
                     .get("frac_within_200um")),
                 "laminar_L4_to_L23_specificity": (
                     (m["laminar"]["canonical_motif"].get("L4->L2/3") or {})
                     .get("forward_specificity")),
                 "laminar_L6_to_L4_specificity": (
                     (m["laminar"]["canonical_motif"].get("L6->L4") or {})
                     .get("forward_specificity")),
                 "effective_excitatory_axons": (
                     fe.get("E", {}).get("per_cell_log_p", {}).get("effective_cells")),
                 "n_volumes": 1, "n_animals": 1,
             },
             "read_it_as": "one mouse, one cubic millimetre of visual cortex.  the "
                           "connection probabilities are unbiased WITHIN the volume for "
                           "fully-extended axons and lower bounds for the other two "
                           "proofreading strategies; nothing here transfers to human "
                           "absolute density.",
             "notes": "see derive/microcircuit_statistics.yaml and "
                      "scripts/measure_microcircuit.py"})
        print(f"microns -> {d}")

    if args.only in ("all", "h01"):
        h = h01_measure(max_workers=args.workers)
        h["sources"] = ["H01 release 20210601: c3 soma table, c3 segment properties, "
                        "and 100 of 166 shards of the exported synapse table"]
        h["measured_at"] = TODAY
        h["species"] = ("Homo sapiens, temporal cortex, ONE person, resected during "
                        "epilepsy surgery")
        d = write_evidence(
            "h01", "human_microcircuit_census", 1, h,
            {"recipe": "human_microcircuit_census", "version": 1,
             "produced_at": TODAY,
             "from_raw": {"h01": "derived/ tables plus synapses_exported/ shards"},
             "outputs": ["measured.json"],
             "headline": {
                 "n_somas": h["census"]["n_somas"],
                 "inhibitory_fraction_of_classified_neurons":
                     round(h["census"]["inhibitory_fraction_of_classified_neurons"], 4),
                 "inhibitory_input_fraction_pyramidal":
                     h["ei_input"]["inhibitory_input_fraction_pyramidal"],
                 "n_synapses_sampled": h["synapse_sample"].get("n_synapses_sampled"),
                 "inhibitory_synapse_fraction":
                     h["synapse_sample"].get("inhibitory_synapse_fraction"),
                 "sampling_fraction_measured":
                     h["synapse_sample"].get("sampling_fraction_measured"),
             },
             "read_it_as": "one person with epilepsy, one cubic millimetre of temporal "
                           "cortex, and NO proofread axons -- so every connection "
                           "statistic here is a lower bound and only the census and the "
                           "per-segment input ratios are measurements in the ordinary "
                           "sense.",
             "notes": "see derive/human_microcircuit_census.yaml"})
        print(f"h01 -> {d}")

    if args.only in ("all", "allen"):
        a = allen_measure()
        a["sources"] = ["Allen Cell Types Database, ApiCellTypesSpecimenDetail, "
                        "2333 specimens"]
        a["measured_at"] = TODAY
        d = write_evidence(
            "allen-cell-types-patchseq", "population_kinetics", 1, a,
            {"recipe": "population_kinetics", "version": 1, "produced_at": TODAY,
             "from_raw": {"allen-cell-types-patchseq":
                          "cell_types_specimen_details.json, 2333 rows"},
             "outputs": ["measured.json"],
             "headline": {
                 "n_human": a["n_human"], "n_mouse": a["n_mouse"],
                 "mouse_spiny_tau_ms_median":
                     a["by_group"]["mouse_spiny"]["tau_ms"]["median"],
                 "human_spiny_tau_ms_median":
                     a["by_group"]["human_spiny"]["tau_ms"]["median"],
                 "mouse_pvalb_tau_ms_median":
                     a["by_group"]["mouse_pvalb"]["tau_ms"]["median"],
             },
             "read_it_as": "single-cell patch clamp in slice at 34 C.  a membrane time "
                           "constant measured at rest in a quiet slice is several times "
                           "LONGER than the effective one in the high-conductance state "
                           "of active cortex, which is the value the process parameter "
                           "actually means.",
             "notes": "see derive/population_kinetics.yaml"})
        print(f"allen -> {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
