#!/usr/bin/env python
"""step the materialized `eeg_forward` model, and ask whether what comes out is physics.

everything before this script fitted process parameters directly against measured
spectra.  that is a legitimate thing to do and it never touched `ibm.runtime`: no
materialized model had ever been handed to `step`, so the windowed harmonic-balance
solve, the phase-ramp delays, the lead field and the evidence path had all been
verified against each other and against dense linear algebra, and none of them had
ever been asked to produce a number about a head.

this script asks.  it builds `eeg_forward` on the mne `sample` subject, assembles
the process graph into `Coupling` objects, initialises state at the ontology's own
declared priors, and solves the window.  then it puts four claims to it, each of
which can fail and each of which is reported with the number that decided it:

    spectrum     the stepped neural state should carry a 1/f^beta background with
                 an alpha peak in 8-13 Hz, given a prior that puts one there.  the
                 exponent and the peak reported are the OUTPUT's, measured on the
                 solved state, not the input's.
    delays       a tract or association edge carries its conduction delay as
                 exp(-i omega tau).  the cross-spectral phase between two coupled
                 sites must then be a straight line in frequency whose slope is
                 -tau.  this is the sharpest test available of the spectral form
                 because it is exact rather than approximate: if the slope is not
                 the declared delay, the phase ramp is wrong, full stop.
    topography   pushing the stepped cortical state through the real BEM lead
                 field must produce a scalp map that is smooth, dipolar for a
                 focal source, and of an order of magnitude a scalp amplifier
                 would see.
    stability    the linearised recurrent loop must have a positive
                 `stability_margin`, and the nonlinear solve must stay bounded
                 over several windows rather than growing.

and it exercises the causality refusal -- a plan whose overlap is shorter than the
longest memory in the graph is refused, because the temporal laplacian is cyclic
and a memory that long wraps and reads the window's own future as its past -- and
fuses one real EEG segment from the same subject as evidence.

what this script does NOT do is tune anything until it looks right.  every theta
is the prior median the build recorded, and a failure reported with its diagnosis
is the most valuable thing available here.  there are failures below: the
multi-window solve stops at a residual floor, the lead field the build computed
connects nothing in the state graph, the declared window is too short for the
declared amplifier, and the closed loop diverges at prior gains.  each is
reported with the number that decided it.

run:  ./.venv/bin/python scripts/run_eeg_forward.py            (r(q) as
      `materialize_eeg_forward.py` reports it -- 13,647 column nodes at 3.25 mm,
      531 tissue voxels, 12,419 conductor cells, 389,510 state variables; about
      22 minutes, most of it in the three-window solve)
      ./.venv/bin/python scripts/run_eeg_forward.py --coarse   (6 mm sheet: the
      same head, the same BEM, a third of the nodes, about 4 minutes.  every
      conclusion below holds on both; the numbers quoted in the checks are the
      full-resolution ones)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ibm
from ibm.forge.priors import median_of, sd_of
from ibm.materialize import geometry as geo
from ibm.materialize.build import build
from ibm.materialize.cache import Cache, default_root, spec_hash
from ibm.materialize.library import MODELS
from ibm.materialize.library.electrophysiology import SCALP
from ibm.materialize.request import DeviceSpec, SubjectSpec, Window
from ibm.processes.base import (
    alpha_synapse, constant_gain, low_pass, pure_delay, series, stability_margin,
)
from ibm.processes.neural import synaptic_drive_transfer
from ibm.registry import REGISTRY, Form
from ibm.runtime.fuse import from_observation, fuse
from ibm.runtime.state import State
from ibm.fields.uncertainty.spectral import TemporalBasis
from ibm.runtime.step import (
    Carry, CausalityViolation, Coupling, Solve, WindowPlan, advance, longest_memory,
    solve_window,
)
from ibm.vocabulary import Difference, Near, OnSupport, Resolution, ResolutionRule

RIBBON_EXCLUSION_MM = 6.0

REGIONS = (
    ("cortex", OnSupport("cortical_surface")),
    ("subcortex", Difference(OnSupport("tissue"), Near("cortex", RIBBON_EXCLUSION_MM))),
    ("conductor", OnSupport("head_volume")),
)


def rule(region, mm: float, band=SCALP) -> ResolutionRule:
    return ResolutionRule(region, mm, band)


def head(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def resolution_for(coarse: bool) -> Resolution:
    """the r(q) this script runs at.

    the fine one is `materialize_eeg_forward.py`'s own coarsened rule, verbatim,
    so that what is stepped here is the model that script reports.  `--coarse`
    halves the sheet's node count and doubles the conductor's cell, which changes
    nothing about the dynamics and makes an iteration take a minute instead of
    ten -- it is for developing this script, not for believing its numbers.
    """
    if coarse:
        return Resolution((rule(OnSupport("head_volume"), 16.0),
                           rule(OnSupport("tissue"), 21.0),
                           rule(OnSupport("cortical_surface"), 6.0)),
                          default_mm=24.0, default_band=SCALP)
    return Resolution((rule(OnSupport("head_volume"), 8.0),
                       rule(OnSupport("tissue"), 10.5),
                       rule(OnSupport("cortical_surface"), 3.25)),
                      default_mm=12.0, default_band=SCALP)


def build_model(coarse: bool = False, window: Window | None = None, rebuild: bool = False):
    """build `eeg_forward` on the sample subject, or read back the one already built.

    the octrees, the poisson-disk sampling of the sheet and the geodesic graph
    cost minutes and are completely determined by the request, which is what
    `ibm.materialize.cache` exists for and which `build()` uses on its own.  the
    BEM solve is the one artefact it never sees, because a lead field is indexed
    by source position and therefore cannot be computed until the site table
    exists -- so it is memoised here instead, keyed on the positions it was
    solved over.
    """
    ibm.load_all(seal=True, strict=True)
    base = MODELS["eeg_forward"].request
    res = resolution_for(coarse)
    store = Path(default_root()) / "run_eeg_forward"
    store.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cache = Cache()
    paths = geo.sample_paths()
    geometry = geo.sample_subject()
    montage = geometry.get("sensor_array")
    coreg = geo.sample_coregistration(geometry=geometry)
    white = np.asarray(geometry.get("cortical_surface").vertices, float)
    dev = DeviceSpec("eeg", "sensor_array", geo.MONTAGE_FRAME, "observe",
                     n_elements=len(np.asarray(montage.xyz)),
                     positions=np.asarray(montage.xyz, float),
                     element_ids=tuple(montage.ids),
                     note=f"digitised montage from {paths.raw_fif.name}")
    subject = SubjectSpec(id="sample", frame=geo.ANATOMICAL_FRAME,
                          surface_frame=geo.ANATOMICAL_FRAME, template=None,
                          note="mne sample: individual T1, individual watershed BEM")
    request = replace(base, subject=subject, devices=(dev,), regions=REGIONS,
                      resolution=res, window=window or base.window,
                      budget=replace(base.budget, max_state_variables=600_000))
    print(f"  window      n={request.window.n} dt={request.window.dt:g}s "
          f"({request.window.duration_s:g}s)")
    probe = build(request, geometry=geometry, warp=coreg.warp,
                  anchors={"cortex": white}, strict=False, cache=cache)
    sheet = probe.sites["cortical_surface"]
    print(f"  {sheet.n:,} column nodes  ({time.time() - t0:.0f}s)")

    # the BEM solve is the one artefact `Cache` never sees -- it is computed
    # outside `build()` because a lead field is indexed by source position and so
    # cannot exist until the site table does -- so it is memoised here by the
    # positions it was solved over, which is exactly what determines it.
    lf_key = spec_hash("run_eeg_forward.leadfield",
                       (np.asarray(sheet.xyz, float), tuple(montage.ids)))
    lf_blob = store / f"{lf_key}.npz"
    if lf_blob.exists() and not rebuild:
        z = np.load(lf_blob)
        lead_data, lead_note = z["data"], str(z["note"])
        print(f"  lead field read back from {lf_blob.name}")
    else:
        lead = geo.sample_lead_field(np.asarray(sheet.xyz, float),
                                     sensor_ids=tuple(montage.ids), paths=paths)
        lead_data, lead_note = np.asarray(lead.data, float), lead.describe()
        np.savez_compressed(lf_blob, data=lead_data, note=lead_note)
        print(f"  BEM solved and cached  ({time.time() - t0:.0f}s)")

    topology_inputs = {
        "electromagnetic": {"lead_field": lead_data, "source_support": "cortical_surface"},
        "cortical_surface": {"radius_mm": 9.0, "k": 18},
        "cortical_association": {"max_degree": 64},
        "local": {},
        "microcircuit": {},
        "device_coupling": {"device_support": "sensor_array", "medium_support": "head_volume",
                            "reach_mm": 12.0, "direction": "record"},
    }
    model = build(request, geometry=geometry, warp=coreg.warp, anchors={"cortex": white},
                  strict=False, topology_inputs=topology_inputs, cache=cache)
    print(f"  materialized in {time.time() - t0:.0f}s")
    print(f"  {lead_note}")
    return model, lead_data




# ---------------------------------------------------------------------------
# the two conventions, and the one number that reconciles them
# ---------------------------------------------------------------------------

#: the relaxation rate that turns a declared transfer function into pressure.
#:
#: `ibm.processes.base` says at the top of the file, in bold, that an
#: implementation's transfer is `H(omega)` -- and every function in its library
#: is an *input to output* filter: `low_pass` is "the membrane, in one line",
#: `feedback` closes a loop and returns the closed-loop response, `pure_delay` is
#: the delay itself.  `ibm.runtime.step` solves `i omega z = sum_p f_p(z)`, so
#: what a coupling hands it is `dz/dt`.  those are not the same object and
#: nothing in the repository converts between them, because no materialized model
#: had ever been stepped.
#:
#: ARCHITECTURE.md §4 says what the conversion is.  an instantaneous algebraic
#: relation `x_O = g(x_I)` is the stiff limit of pressure,
#:
#:     dot x_O += -gamma (x_O - g(x_I)),    gamma -> infinity
#:
#: and "is a question of how it is solved, not of what it is".  so a declared LTI
#: transfer enters as `+gamma H(omega) x_I` from each source and `-gamma x_O`
#: once, which composes by summation exactly as §4 requires and leaves the
#: written component equal to `sum_i H_i(omega) x_i` divided by `1 + i omega /
#: gamma`.
#:
#: the number below is chosen from that error term and from nothing else.  at the
#: top of the retained band, 100 Hz, `omega / gamma` is 1.0e-2: the amplitude is
#: low by 0.005% and the phase lags by 0.57 degrees.  raising gamma makes the
#: exact-inverse diagonal stiffer and costs nothing in this solver, because the
#: diagonal is inverted in closed form; it is kept finite only so the error is a
#: number that can be quoted.
STIFF_RATE_HZ = 1e4
GAMMA = 2.0 * math.pi * STIFF_RATE_HZ


def stiff_limit(component: str, n_sites: int) -> Coupling:
    """the `-gamma x_O` half of §4's stiff limit, once per written component.

    it is a self-coupling that reads and writes the same component with no site
    mixing, so `Coupling.self_diagonal` is true and it lands in the operator
    `solve_window` inverts exactly.  every declared transfer writing this
    component supplies the `+gamma H x_I` half as an ordinary drive, and because
    those are drives they *sum* -- which is the whole reason the leak is declared
    once here rather than folded into each of them, where N writers would have
    produced N leaks and turned the sum into an average.
    """
    return Coupling(process=f"stiff-limit({component})", writes=component,
                    reads=(component,), form=Form.LTI,
                    transfer=lambda basis: np.full(basis.k, complex(-GAMMA)),
                    note="ARCHITECTURE.md 4: the gamma -> infinity limit of pressure")


# ---------------------------------------------------------------------------
# site bookkeeping
# ---------------------------------------------------------------------------


def merged_index(model, cid: str) -> tuple[np.ndarray, int]:
    """global site index -> index within the component's merged belief, or -1.

    a component split across two supports has one belief over the concatenation
    of its blocks in layout order (`Layout.__getitem__` returns the merged
    block), while an edge set indexes the model's *global* site table.  the two
    orderings differ -- global order is by support table, merged order is by
    layout block -- so an edge set cannot be turned into a mixing matrix without
    this map, and getting it wrong scrambles the cortex into the thalamus
    without any shape check noticing.
    """
    n_total = sum(t.n for t in model.sites.tables.values())
    out = np.full(n_total, -1, dtype=np.int64)
    at = 0
    for s in model.site_layout.supports_of(cid):
        t = model.sites[s]
        out[t.offset:t.offset + t.n] = np.arange(at, at + t.n)
        at += t.n
    return out, at


def edge_mix(model, edges, cid_in: str, cid_out: str, weight: np.ndarray,
             mask: np.ndarray | None = None, symmetric: bool = True):
    """one topology's edges as a sparse (n_out, n_in) operator on the site axis.

    sparse and not dense for the obvious reason -- 4.3e5 association edges over
    1.4e4 nodes is 0.2% fill and the dense matrix is 1.5 GiB -- and it never
    enters the time domain, so it acts at every frequency independently exactly
    as `Coupling.mix` says.

    `symmetric` adds the reverse of every stored edge.  an undirected edge set
    stores one fact per pair on purpose (`EdgeSet.symmetrized`), and a process
    that scatters over it has to see both directions or half the cortex receives
    nothing.
    """
    from scipy.sparse import coo_matrix

    src, dst = np.asarray(edges.src), np.asarray(edges.dst)
    w = np.asarray(weight, float)
    if mask is not None:
        src, dst, w = src[mask], dst[mask], w[mask]
    if symmetric and not edges.directed:
        src, dst, w = (np.concatenate([src, dst]), np.concatenate([dst, src]),
                       np.concatenate([w, w]))
    gin, n_in = merged_index(model, cid_in)
    gout, n_out = merged_index(model, cid_out)
    r, c = gout[dst], gin[src]
    keep = (r >= 0) & (c >= 0)
    return coo_matrix((w[keep], (r[keep], c[keep])), shape=(n_out, n_in)).tocsr(), int(keep.sum())


def theta_of(model, impl, fn=None) -> dict:
    """the prior median of every parameter of `impl` that its transfer accepts.

    the median and not a fitted value, and every one of them, so that nothing in
    this script is tuned: `ibm.forge` has never been run against this model, so
    the honest theta is p(theta)'s centre.

    `MaterializedModel.priors` is where the build recorded p(theta) -- but only
    for the implementation it *selected*, and it is keyed `process.param` with no
    room for two implementations of one process.  where this script uses an
    implementation build did not select (`device_coupling`, section 3) the
    priors it wants are simply not in that mapping, so the declaration's own
    `params` is the fallback.  reading the model first keeps a forged parameter
    winning wherever one exists.
    """
    return _theta(model, impl, fn, median_of)


def theta_sd_of(model, impl, fn=None) -> dict:
    """the same parameters' prior *widths*, in the same natural units.

    ARCHITECTURE.md 4 says the induced state distribution depends on p(x) and
    p(theta), and every number this script has ever reported came from the
    median alone -- which is the assertion that all 123 of this model's
    parameters are known exactly.  they are not: most of them are `weak()`, and
    what that declaration means is a lognormal spread of a factor of ten.
    `Coupling.theta_sd` carries these so `ibm.runtime.propagate` can put the
    first-order term on the psd instead of pretending the second factor is a
    delta.
    """
    return _theta(model, impl, fn, sd_of)


def _theta(model, impl, fn, reduce) -> dict:
    import inspect

    fn = fn or impl.transfer
    want = set(inspect.signature(fn).parameters)
    out = {name: reduce(pr) for name, pr in impl.params.items() if name in want}
    for key, prior in model.priors.items():
        proc, _, name = key.partition(".")
        if proc == impl.process and name in want:
            out[name] = reduce(prior)
    return out


# ---------------------------------------------------------------------------
# assembling the process graph
# ---------------------------------------------------------------------------

#: how many distance bins a topology-mediated coupling is split into.
#:
#: `Coupling` carries one frequency-independent `mix` and one transfer that is
#: either per-frequency or per-*input site*.  a tract or association kernel is
#: per *edge* -- `association_transfer` takes the edge's own `distance_mm` and
#: returns both its weight and its delay from it -- and there is no shape in the
#: dataclass that can hold `H_e(omega)` for 4.3e5 edges.  so the operator is
#: written as a sum of terms, one per distance bin, each of which is an exact
#: (mix x transfer) pair evaluated at that bin's mean distance.  the
#: approximation is the within-bin spread of the delay, and it is REPORTED in
#: milliseconds in section 3 rather than asserted to be small.  it is not small:
#: 12 quantile bins over an 8-168 mm association graph leave more than ten
#: milliseconds of delay spread in the top bin, because that distribution is
#: skewed and its top quantile is wide however finely it is cut.  bins cost real
#: time -- each is a separate `spectral_drift` over a (sites x k) array, and
#: tripling them tripled the window -- so the number here is a budget decision,
#: and section 9's delay check is run on *unbinned* couplings, one per edge, so
#: that what it measures is the phase ramp and not this approximation.
DISTANCE_BINS = 12


def binned_topology_couplings(model, edges, transfer, process: str, impl_name: str,
                              cid_in: str, cid_out: str, feature: str, param: str,
                              theta: dict, velocity_key: str, cv_key: str,
                              per_edge_scale=None, n_bins: int = DISTANCE_BINS,
                              note: str = "", theta_sd: dict | None = None):
    """one coupling per distance bin of a topology whose kernel varies per edge.

    `feature` is the edge column binned over and `param` is the transfer's own
    argument name for it, and they are not always the same word: the lateral
    process propagates along the sheet, so what its `distance_mm` argument means
    is the topology's `geodesic_mm` column and not its `distance_mm` one.
    """
    d = np.asarray(edges.features[feature], float)
    qs = np.quantile(d, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0] -= 1e-9
    out, spread_ms, kept = [], 0.0, 0
    v = float(theta.get(velocity_key, 1.0))
    cv = float(theta.get(cv_key, 0.0))
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (d > lo) & (d <= hi)
        if not m.any():
            continue
        d_rep = float(d[m].mean())
        scale = np.ones(len(d)) if per_edge_scale is None else np.asarray(per_edge_scale, float)
        mix, n = edge_mix(model, edges, cid_in, cid_out, scale, m)
        kept += n
        tau = d_rep * 1e-3 / max(v, 1e-9)
        spread_ms = max(spread_ms, float(d[m].max() - d[m].min()) * 1e-3 / max(v, 1e-9) * 1e3)
        out.append(Coupling(
            process=f"{process}[{impl_name}] {feature}~{d_rep:.0f}mm",
            writes=cid_out, reads=(cid_in,), form=Form.LTI,
            transfer=_stiff(transfer, {param: d_rep}), mix=mix,
            theta=dict(theta), theta_sd=dict(theta_sd or {}),
            # `memory_s` is "the longest time into the past this coupling
            # reads", and for this kernel that is two things added: the
            # dispersed delay out to three standard deviations, and the slow
            # receptor's own settling.  nmda decays with a ~100 ms time
            # constant, so three of those is 300 ms and dominates every
            # conduction delay on a head this size -- which is the sort of thing
            # a plan check exists to notice.
            memory_s=tau * (1.0 + 3.0 * cv) + 3.0 * float(theta.get("tau_nmda_decay_s", 0.1)),
            note=note))
    return tuple(out), spread_ms, kept


def _stiff(fn, fixed: dict):
    """`gamma * H(omega)`, with this bin's edge geometry baked in.

    the gamma belongs here rather than in `Coupling.gain` so that what the
    dataclass carries as its transfer is exactly the thing the solver needs:
    the pressure a source exerts, not the response it would produce.
    """
    def H(basis, **theta):
        return GAMMA * np.asarray(fn(basis, **fixed, **theta), dtype=np.complex128)
    return H


def _self(fn, fixed: dict | None = None):
    """the same, for a coupling that runs site-to-site with no topology."""
    fixed = fixed or {}

    def H(basis, **theta):
        return GAMMA * np.asarray(fn(basis, **fixed, **theta), dtype=np.complex128)
    return H


# ---------------------------------------------------------------------------
# the read -> write table
# ---------------------------------------------------------------------------

ACT = "neural.exc.activity"
AMPA = "neural.exc.ampa"
NMDA = "neural.exc.nmda"
POT = "neural.exc.potential"
ADAPT = "neural.exc.adaptation"
PV = "neural.pv.activity"
GABA_A = "neural.inh.gaba_a"
TMC = "neural.transmembrane_current"
CP = "device.contact_potential"

PAIRING_ARGUMENT = """  a process declares I and O as *sets* of components.  it does not declare which
  input drives which output, and the cross product is emphatically not the
  answer: `local_excitation` reads five components and writes nine, and
  `synaptic_drive_transfer` is a presynaptic-rate-to-postsynaptic-conductance
  kernel, so pairing it with the membrane potential input would be arithmetic on
  quantities that have nothing to do with each other.  nothing in the registry,
  the trace or the materialized model records the pairing, so `couplings_of`
  cannot be written generically and this table is written by hand, with the
  physics that fixes each row.  that gap is the first finding of this script."""


def assemble(model, lead_data, *, nonlinear: bool = False, verbose: bool = True):
    """turn the materialized model into the couplings `ibm.runtime.step` solves.

    every transfer function, every parameter and every edge weight below comes
    out of the model: the implementations are the ones `build` selected, theta is
    the prior median `build` recorded, and the mixing matrices are the edge sets
    `build`'s topology builders produced on this subject.  what this function
    supplies that the model does not is the pairing (see `PAIRING_ARGUMENT`) and
    the stiff-limit encoding (see `STIFF_RATE_HZ`).
    """
    from scipy.sparse import identity as sp_identity

    impl = model.implementations
    lines: list[str] = []
    cs: list[Coupling] = []

    def imp(process: str, name: str):
        """the named implementation of a process, whether or not build picked it.

        `build` scores implementations and picks one, and for `device_coupling`
        on a scalp montage it picked `acoustic_beam` -- an ultrasound
        transducer's beam pattern -- which is reported in section 3 and is the
        second finding.  where this script needs a different f it says so here
        rather than quietly using whatever was selected.
        """
        for i in REGISTRY.implementations.values():
            if i.process == process and i.name == name:
                return i
        raise KeyError(f"{process}/{name}")

    # -- 1. local recurrent excitation: rate -> conductance ------------------
    loc = model.edges["local"]
    n_neural = merged_index(model, ACT)[1]
    ex = imp("local_excitation", "conductance_lti")
    th = theta_of(model, ex)
    mix_local, n_loc = edge_mix(model, loc, ACT, AMPA, np.ones(loc.n_edges))
    # a patch excites itself, and the topology cannot say so: `local_radius`
    # carries pairs of positions and a position is not a pair.  recurrence
    # within one materialized position is the dominant term of local excitation
    # -- it is what "recurrent" means at a mean-field position -- so the
    # identity is added here and counted in the degree reported below.
    mix_local = mix_local + sp_identity(n_neural, format="csr")
    for out, note in ((AMPA, "ampa branch"), (NMDA, "nmda branch")):
        cs.append(Coupling(process="local_excitation[conductance_lti]", writes=out,
                           reads=(ACT,), form=Form.LTI, transfer=_self(ex.transfer),
                           mix=mix_local, theta=th,
                           memory_s=3.0 * th.get("tau_nmda_decay_s", 0.1),
                           note=f"local, {note}"))
    lines.append(f"  local_excitation[conductance_lti]   {ACT} -> {AMPA}, {NMDA}   "
                 f"over `local` ({n_loc:,} directed edges + {n_neural:,} self), mean degree "
                 f"{(n_loc + n_neural) / n_neural:.1f}")

    # -- 2. lateral intracortical propagation, over the geodesic ------------
    surf = model.edges["cortical_surface"]
    lat = imp("lateral_cortical_propagation", "geodesic_exponential_lti")
    th_lat = theta_of(model, lat)
    lat_cs, lat_spread, n_lat = binned_topology_couplings(
        model, surf, lat.transfer, "lateral_cortical_propagation",
        "geodesic_exponential_lti", ACT, AMPA, "geodesic_mm", "distance_mm", th_lat,
        "velocity_m_s", "velocity_cv", theta_sd=theta_sd_of(model, lat),
        note="horizontal layer 2/3 arbor; the distance is the geodesic, per the "
             "topology's own metric, and the delay follows from it")
    cs += list(lat_cs)
    lines.append(f"  lateral_cortical_propagation        {ACT} -> {AMPA}   over "
                 f"`cortical_surface` ({n_lat:,} directed edges, {len(lat_cs)} distance bins, "
                 f"within-bin delay spread <= {lat_spread:.2f} ms)")

    # -- 3. long-range association, under a distance prior ------------------
    assoc = model.edges["cortical_association"]
    ass = imp("cortical_association_propagation", "distance_prior_lti")
    th_ass = theta_of(model, ass)
    # the horvitz-thompson correction rides on the *mix*, not on the transfer.
    # `association_transfer` divides by an `inclusion_prob` that it takes as a
    # scalar; the builder sampled every edge with its own probability, so the
    # correction is per edge and belongs in the matrix.  the transfer therefore
    # gets the default 1.0 and the two do not double-count.
    ht = 1.0 / np.clip(np.asarray(assoc.features["inclusion_prob"], float), 1e-6, 1.0)
    for out in (AMPA, PV):
        acs, ass_spread, n_ass = binned_topology_couplings(
            model, assoc, ass.transfer, "cortical_association_propagation",
            "distance_prior_lti", ACT, out, "distance_mm", "distance_mm", th_ass,
            "velocity_m_s", "velocity_cv", per_edge_scale=ht,
            theta_sd=theta_sd_of(model, ass),
            note="geometric prior over UNOBSERVED connectivity: no tractogram informed "
                 "these edges on this subject")
        cs += list(acs)
    lines.append(f"  cortical_association_propagation    {ACT} -> {AMPA}, {PV}   over "
                 f"`cortical_association` ({n_ass:,} directed edges, {len(acs)} bins, "
                 f"within-bin delay spread <= {ass_spread:.2f} ms), horvitz-thompson "
                 f"reweighting x{ht.mean():.0f} on average")

    # -- 4. feedforward inhibition: pv rate -> gaba-a conductance -----------
    micro = model.edges["microcircuit"]
    inh = imp("local_inhibition", "gaba_conductance_lti")
    th_inh = theta_of(model, inh)
    mix_micro, n_micro = edge_mix(model, micro, PV, GABA_A, np.ones(micro.n_edges))
    cs.append(Coupling(process="local_inhibition[gaba_conductance_lti]", writes=GABA_A,
                       reads=(PV,), form=Form.LTI, transfer=_self(inh.transfer),
                       mix=mix_micro, theta=th_inh, theta_sd=theta_sd_of(model, inh),
                       memory_s=3.0 * (th_inh.get("tau_gaba_b_rise_s", 0.05)
                                       + th_inh.get("tau_gaba_b_decay_s", 0.15)),
                       note="pv -> gaba-a; the microcircuit graph on this subject is "
                            "within-site only, so this is site-local"))
    lines.append(f"  local_inhibition[gaba_conductance_lti]  {PV} -> {GABA_A}   over "
                 f"`microcircuit` ({n_micro:,} edges, all within-site)")

    # -- 5. conductance -> membrane potential -------------------------------
    if not nonlinear:
        ei = imp("local_excitation", "ei_loop_lti")
        th_ei = theta_of(model, ei)
        cs.append(Coupling(process="local_excitation[ei_loop_lti]", writes=POT, reads=(ACT,),
                           form=Form.LTI, transfer=_self(ei.transfer), theta=th_ei,
                           theta_sd=theta_sd_of(model, ei),
                           memory_s=3.0 * (th_ei.get("tau_membrane_s", 0.015)
                                           + th_ei.get("tau_gaba_a_s", 6e-3)
                                           + th_ei.get("tau_inh_membrane_s", 8e-3)),
                           note="the closed local E/I loop: the inhibitory return path is "
                                "inside this transfer's own feedback term, so carrying "
                                "gaba-a -> potential separately would double-count it"))
        tc = imp("thalamocortical_coupling", "alpha_resonator")
        th_tc = theta_of(model, tc)
        cs.append(Coupling(process="thalamocortical_coupling[alpha_resonator]", writes=POT,
                           reads=(ACT,), form=Form.LTI, transfer=_self(tc.transfer),
                           theta=th_tc, theta_sd=theta_sd_of(model, tc),
                           # a resonator rings for q cycles; three ring times is
                           # 3 q / (pi f0), which at q=4 and 10 Hz is 380 ms.
                           memory_s=3.0 * th_tc.get("q", 4.0) / (math.pi
                                                                 * th_tc.get("f0_hz", 10.0)),
                           note="site-local: `tractometric` built 0 edges on this subject, "
                                "so there is no materialized thalamus to run the loop "
                                "through and the resonance is applied where the loop's "
                                "cortical end is"))
        lines.append(f"  local_excitation[ei_loop_lti]       {ACT} -> {POT}   site-local, "
                     f"loop_gain {th_ei.get('loop_gain', float('nan')):.2f}")
        lines.append(f"  thalamocortical_coupling[alpha_resonator]  {ACT} -> {POT}   "
                     f"site-local, f0 {th_tc.get('f0_hz', float('nan')):.2f} Hz, "
                     f"q {th_tc.get('q', float('nan')):.2f}")
    else:
        wc = imp("local_excitation", "wilson_cowan_adaptive")
        # this implementation was not the one build selected, so `model.priors`
        # carries `conductance_lti`'s theta under the same `process.param` keys
        # and none of wilson-cowan's.  the declaration's own params are the only
        # place its priors exist.
        th_wc = {name: median_of(pr) for name, pr in wc.params.items()}
        fn = wc.fn
        for out in (POT, ACT, ADAPT):
            cs.append(Coupling(process="local_excitation[wilson_cowan_adaptive]", writes=out,
                               reads=(POT, ADAPT, AMPA, NMDA, ACT), form=Form.RATE,
                               fn=fn, theta=th_wc,
                               # the adaptation current integrates the rate with a
                               # half-second time constant, so this f reads a second
                               # and a half into its own past.  that is longer than
                               # any conduction delay in the model and it is what
                               # makes the closed loop unwindowable at 2.048 s.
                               memory_s=3.0 * th_wc.get("tau_adaptation_s", 0.5),
                               note="the sigmoid: the only nonlinearity in the file, and the "
                                    "one edge that closes the cortical loop"))
        lines.append(f"  local_excitation[wilson_cowan_adaptive]  {POT},{ADAPT},{AMPA},{NMDA},"
                     f"{ACT} -> {POT},{ACT},{ADAPT}   site-local rate law (Form.RATE)")

    # -- 6. synaptic conductance -> transmembrane current -------------------
    # I = g (V - E), linearised about the background driving force, which is what
    # the `gain` prior of `conductance_lti` already absorbs ("absorbs synapse
    # count, release probability and every unit convention").  the current is
    # what the lead field integrates, so it has to be sourced from the
    # conductances rather than from the rate -- `ibm.fields.neural` is explicit
    # that treating firing rate as a proxy is how a forward model stops being
    # physics.
    for src in (AMPA, NMDA):
        cs.append(Coupling(process="local_excitation[conductance_lti] -> current", writes=TMC,
                           reads=(src,), form=Form.LTI,
                           transfer=_self(constant_gain, {"gain": 1.0}), memory_s=0.0,
                           note="I = g (V - E) linearised about the background driving force"))
    lines.append(f"  local_excitation -> current         {AMPA}, {NMDA} -> {TMC}   "
                 "site-local, frequency-flat")

    # -- 7. the instrument --------------------------------------------------
    L, dip = lead_field_mix(model, lead_data)
    dc = imp("device_coupling", "quasistatic_lead_field")
    ei_dev = imp("device_coupling", "electrode_interface")
    th_dev = theta_of(model, ei_dev)

    tau_hp = th_dev.get("tau_highpass_s", 3.0)
    tau_lp = th_dev.get("tau_lowpass_s", 1.6e-3)

    def instrument(basis, **theta):
        # two declared implementations of one process in series: one is geometry
        # and one is hardware, and `device_coupling`'s own docstring says the
        # amplifier's band-pass sits in the same path as the lead field.
        #
        # the AC-coupling corner is NOT in this product, and section 5 is where
        # that is argued rather than here: its impulse response is
        # `-(1/tau) exp(-t/tau)` with tau = 3 s, so on the 2.048 s window
        # `eeg_forward` declares it wraps, and the plan check refuses the run
        # outright.  what is kept is the anti-alias low-pass, which is where the
        # observation's band ceiling comes from.
        return GAMMA * series(dc.transfer(basis, gain=1.0),
                              low_pass(basis, tau_lp, theta.get("gain", 1.0)))

    def instrument_ac(basis, **theta):
        return GAMMA * series(dc.transfer(basis, gain=1.0),
                              ei_dev.transfer(basis, **theta))

    cs.append(Coupling(process="device_coupling[quasistatic_lead_field x anti-alias]",
                       writes=CP, reads=(TMC,), form=Form.LTI, transfer=instrument,
                       mix=L, theta=th_dev, theta_sd=theta_sd_of(model, ei_dev),
                       memory_s=3.0 * tau_lp,
                       note="the subject's own three-layer BEM solution"))
    ac = Coupling(process="device_coupling[quasistatic_lead_field x electrode_interface]",
                  writes=CP, reads=(TMC,), form=Form.LTI, transfer=instrument_ac,
                  mix=L, theta=th_dev, theta_sd=theta_sd_of(model, ei_dev),
                  memory_s=3.0 * tau_hp,
                  note="the same, with the amplifier's AC-coupling corner restored")
    lines.append(f"  device_coupling                     {TMC} -> {CP}   "
                 f"{L.shape[0]} electrodes x {L.shape[1]:,} sources, dense BEM lead field, "
                 f"anti-alias corner {1.0 / (2 * math.pi * tau_lp):.4g} Hz")

    # -- 8. the stiff limit, once per written component ---------------------
    written = []
    for cid in dict.fromkeys(c.writes for c in cs):
        if nonlinear and cid in (POT, ACT, ADAPT):
            continue        # the rate law supplies its own relaxation
        cs.append(stiff_limit(cid, model.layout[cid].n_sites))
        written.append(cid)
    lines.append(f"  stiff-limit                         -gamma x_O on {len(written)} written "
                 f"components, gamma = 2 pi x {STIFF_RATE_HZ:g} Hz")

    if verbose:
        for line in lines:
            print(line)
    return tuple(cs), dip, ac


def lead_field_mix(model, lead_data):
    """the BEM lead field as an operator from current source density to microvolts.

    three conversions, all arithmetic, none of them fitted.

    the lead field is `V per A.m`: it wants a dipole *moment*.  what the model
    carries is `neural.transmembrane_current` in `nA/mm^3` -- a current source
    density -- so the moment of one column node is its own cortical area times
    the ribbon thickness times an effective dipole length, and the sheet's site
    table carries the area per node.  the thickness and the dipole length are the
    same 2.5 mm on this subject, which is its measured mean ribbon thickness.

    the free-orientation solution is projected onto the *cortical normal*,
    computed from the subject's own white-surface mesh, rather than taking the
    vector magnitude.  the magnitude is what `em_lead_field` uses to threshold
    the topology and it is the right object there -- it asks "can this sensor
    see this source at all" -- but using it as a gain asserts every dipole is
    optimally oriented, which deletes precisely the sulcal cancellation that is
    most of why EEG sees so little of what the cortex does.
    """
    L = np.asarray(lead_data, float)
    sheet = model.sites["cortical_surface"]
    nrm = vertex_normals(sheet)
    if L.ndim == 3:
        L = np.einsum("msc,sc->ms", L, nrm)
    area = np.asarray(sheet.columns["area_mm2"], float)
    thickness_mm, dipole_mm = 2.5, 2.5
    # nA/mm^3 * mm^2 * mm * mm = nA.mm = 1e-12 A.m; V -> uV is another 1e6.
    scale = area * thickness_mm * dipole_mm * 1e-12 * 1e6
    gmap, n_out = merged_index(model, CP)
    smap, n_in = merged_index(model, TMC)
    cols = smap[sheet.offset:sheet.offset + sheet.n]
    rows = gmap[model.sites["sensor_array"].offset:
                model.sites["sensor_array"].offset + model.sites["sensor_array"].n]
    from scipy.sparse import csr_matrix
    dense = np.zeros((n_out, n_in))
    dense[np.ix_(rows, cols)] = L * scale[None, :]
    return csr_matrix(dense), dict(scale_uV_per_nA_mm3=float(np.median(np.abs(L) * scale)),
                                   normal_projected=True)


def vertex_normals(table) -> np.ndarray:
    """the cortical normal at each column node, from the subject's own mesh.

    area-weighted face normals accumulated onto vertices, read at the vertex each
    poisson-disk node was seeded on.  it is the one geometric quantity the sheet
    carries that the volume cannot, and it is the reason `eeg_forward` spends its
    site budget on the sheet at all.
    """
    v = np.asarray(table.columns["mesh_vertices"], float)
    f = np.asarray(table.columns["mesh_faces"], int)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    acc = np.zeros_like(v)
    for c in range(3):
        np.add.at(acc, f[:, c], fn)
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    return acc[np.asarray(table.columns["vertex_of_site"], int)]


# ---------------------------------------------------------------------------
# the state a run starts from
# ---------------------------------------------------------------------------


def initial_state(model, basis, rng, sites_of_interest=None):
    """the declared prior, with one realisation of it drawn into the mean.

    `State.prior` now allocates each component at the builder its own
    declaration names -- `neural_population` for population state, which is 1/f
    with theta, alpha and beta bumps on it, and `device_broadband` for an
    electrode.  that prior's *mean* is zero, and deliberately: an ongoing rhythm
    is an isotropic gaussian on each eigenplane, known amplitude and uniform
    phase, and a nonzero mean would be a phase-locked evoked response the resting
    brain has no reason to have.

    but `solve_window` moves means and nothing else.  handed a belief whose mean
    is zero it propagates zero, exactly and forever, and every spectrum measured
    off the result is identically empty.  so the mean is initialised at one draw
    from the belief -- one resting trajectory the prior considers typical -- and
    what the spectral check below measures is the spectrum of the OUTPUT against
    the spectrum of that input.  the draw is per site and independent, because
    the prior declares no spatial covariance: spatial structure in ibm-1 lives in
    the topologies, not in the prior over a component.

    this is a workaround for a real gap and is not a fix for it.  §4 says a
    linear f's push-forward of the *uncertainty* is exact, and
    `SpectralGaussian.apply_transfer` is exactly that push-forward -- but
    `solve_window` never calls it, so for an all-LTI model every psd in the
    returned state is still the prior's.  section 15 says so again with numbers.
    """
    s = State.prior(model.layout)
    for cid in model.layout.components:
        b = model.layout[cid]
        if b.uncertainty != "spectral":
            continue
        draw = s[cid].sample(1, rng)[0]
        s.beliefs[cid] = replace(s[cid], mean=draw)
    return s


# ---------------------------------------------------------------------------
# the four checks
# ---------------------------------------------------------------------------


def psd_of(state, cid, basis, sites=None) -> np.ndarray:
    """mean power per retained coefficient over the selected sites."""
    z = np.asarray(state[cid].mean)
    if sites is not None:
        z = z[sites]
    return (np.abs(z) ** 2).mean(0)


def aperiodic_fit(freqs, psd, lo=1.0, hi=45.0, notch=(6.0, 16.0)):
    """log-log slope of the background, with the rhythm band held out.

    holding the band out rather than fitting a bump alongside it: the question
    here is what exponent the *background* came out at, and a bump left in the
    fit pulls the exponent towards it by exactly the amount that makes the bump
    look smaller.
    """
    m = (freqs >= lo) & (freqs <= hi) & ~((freqs >= notch[0]) & (freqs <= notch[1]))
    m &= psd > 0
    if m.sum() < 8:
        return float("nan"), float("nan")
    a, b = np.polyfit(np.log(freqs[m]), np.log(psd[m]), 1)
    return -float(a), float(b)


def peak_in(freqs, psd, beta, offset, lo=6.0, hi=16.0):
    """the largest excursion above the fitted background, and where it is."""
    bg = np.exp(offset) * np.maximum(freqs, 1e-9) ** (-beta)
    rel = psd / np.maximum(bg, 1e-300)
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any():
        return float("nan"), float("nan")
    i = np.flatnonzero(m)[int(np.argmax(rel[m]))]
    return float(freqs[i]), float(rel[i])




def joint_scale(state, couplings) -> float:
    """the rms of the trajectory the joint gap is a gap in.

    `StepReport.joint_gap` is in state units, and state units here are whatever
    the prior medians made them (section 3).  the number that means something is
    the gap divided by the size of the thing it is a gap in, and reporting the
    ratio is the only way two runs at different amplitudes can be compared at
    all.
    """
    tot, n = 0.0, 0
    for cid in dict.fromkeys(c.writes for c in couplings):
        if cid not in state or state.layout[cid].uncertainty != "spectral":
            continue
        x = state.mean_time(cid)
        tot += float((x ** 2).mean()); n += 1
    return math.sqrt(tot / max(n, 1))


def exogenous(model, couplings) -> tuple[str, ...]:
    """the spectral components no assembled coupling writes.

    `_targets` leaves them exactly where the prior put them, which is the right
    behaviour and is also what makes this model do anything at all: with the
    cortical loop open (section 11), the drive is whatever the ontology's prior
    says about the components nothing here writes.  they are the model's stand-in
    for the thalamic and neuromodulatory input this materialization has no
    anatomy for, and naming them is the difference between a driven model and a
    model that decays to zero.
    """
    written = {c.writes for c in couplings}
    return tuple(c for c in model.layout.components
                 if model.layout[c].uncertainty == "spectral" and c not in written)


def drive_realisation(model, basis, plan, rng, cid=None):
    """one continuous trajectory of the exogenous drive, long enough for the run.

    drawn on a basis as long as the whole run and then sliced, rather than drawn
    per window: a run of three overlapping windows is three views of ONE
    trajectory, and drawing independently per window would ask the continuity
    match to reconcile two unrelated realisations.  the psd it is drawn from is
    the component's own declared prior evaluated at the long window, so nothing
    about the spectrum changes -- only its length.
    """
    from ibm.fields import priors as _priors

    cid = cid or ACT
    long_n = basis.n + (plan.n_windows - 1) * plan.hop_n
    long_basis = TemporalBasis(long_n, basis.dt)
    comp = REGISTRY.components[cid]
    b = model.layout[cid]
    belief = _priors.get(comp.prior)(long_basis, shape=(b.n_sites,))
    return belief.sample_time(1, rng)[0]


def continuity_probe(model, couplings, basis, plan, solve, drive, state0):
    """the same windows again, with the exogenous drive actually advanced.

    `advance` has no hook for it.  it re-solves each window against whatever the
    state holds, and the components the solver may not move -- the exogenous ones
    -- therefore hold the SAME window of trajectory in window 2 as in window 1,
    while `Carry` asks window 2's head to match window 1's tail.  those are two
    different parts of a trajectory driven by the same input, so they genuinely
    disagree, and the residual floor `limited_by="continuity"` reports is that
    disagreement rather than anything about the dynamics.

    the loop below is `solve_window` plus `Carry` by hand, with the drive sliced
    forward by one hop each window, which is what a run over a real stimulus
    would have to do.  if the diagnosis is right the joint gap collapses.
    """
    head("7. the same run with the exogenous drive advanced")
    print("  `advance` re-solves each window against the state it is handed, and the")
    print("  components it may not move -- everything no coupling writes -- are not among")
    print("  the things it advances.  so window 2 is driven by window 1's input while")
    print("  `Carry` asks its head to match window 1's tail: same input, different part of")
    print("  the trajectory, and the two conditions are inconsistent by construction.")
    print("  below, the drive is sliced forward by one hop per window, by hand, because")
    print("  there is no hook in `advance` for doing it.")
    s = state0.copy()
    carry = None
    gaps, res, amps, reps = [], [], [], []
    k = model.layout[ACT].k
    for i in range(plan.n_windows):
        seg = drive[:, i * plan.hop_n: i * plan.hop_n + basis.n]
        s.beliefs[ACT] = replace(s[ACT], mean=basis.analyze(seg)[..., :k])
        s, rep = solve_window(s, couplings, basis, solve=solve, carry=carry,
                              match_weight=plan.match_weight, window=i)
        carry = Carry.tail(s, plan.overlap_n)
        gaps.append(rep.joint_gap / max(joint_scale(s, couplings), 1e-300))
        res.append(rep.residual / max(rep.residual0, 1e-300))
        amps.append(window_amplitude(s, couplings))
        reps.append(rep)
        print(f"    {rep}")
    print(f"  joint gap relative to the state's own rms {['%.3g' % g for g in gaps]}, "
          f"residual/initial {['%.2e' % r for r in res]}")
    print(f"  max |mean| per window {['%.4g' % a for a in amps]}")
    return s, gaps, amps, reps


def check_spectrum(model, state0, state1, basis, sheet_sites):
    head("8. check 1 -- spectrum:  1/f background with an alpha peak on the OUTPUT")
    rows = []
    for cid in (ACT, AMPA, POT):
        for label, st in (("input ", state0), ("output", state1)):
            p = psd_of(st, cid, basis, sheet_sites)
            f = model.layout[cid].basis.freqs_hz
            n = min(len(f), len(p))
            beta, off = aperiodic_fit(f[:n], p[:n])
            fpk, rel = peak_in(f[:n], p[:n], beta, off)
            rows.append((cid, label, beta, fpk, rel, float(p[:n].sum())))
    print(f"    {'component':30s} {'':6s} {'beta':>6s} {'peak Hz':>8s} {'x bg':>7s} "
          f"{'total power':>13s}")
    for cid, label, beta, fpk, rel, tot in rows:
        print(f"    {cid:30s} {label:6s} {beta:6.2f} {fpk:8.2f} {rel:7.2f} {tot:13.4g}")
    print(f"    ({ACT} is exogenous -- no assembled coupling writes it -- so its two rows")
    print("     are two slices of the same drive realisation and differ only by sampling.")
    print("     the conductances and the potential are what the solve produced.)")
    out = [r for r in rows if r[1] == "output" and r[0] == POT][0]
    ok_beta = 0.5 <= out[2] <= 4.0
    ok_alpha = 8.0 <= out[3] <= 13.0 and out[4] > 1.2
    print()
    print(f"    exponent of the stepped {POT}: beta = {out[2]:.2f}  "
          f"[{'PASS' if ok_beta else 'FAIL'}: a resting scalp/cortical spectrum sits "
          "between 1 and 3 (donoghue 2020)]")
    print(f"    peak of the stepped {POT}: {out[3]:.2f} Hz at {out[4]:.2f}x background  "
          f"[{'PASS' if ok_alpha else 'FAIL'}: 8-13 Hz and above background]")
    return ok_beta, ok_alpha


def check_delays(model, basis, rng):
    head("9. check 2 -- delays:  is the phase ramp the declared conduction delay?")
    print("  the sharpest test available, and the only exact one.  a delay is")
    print("  `exp(-i omega tau)` in this basis, so the cross-spectral phase between a")
    print("  source site and the conductance it drives must be a straight line in")
    print("  frequency whose slope is exactly -tau.  the probe below drives ONE cortical")
    print("  site and solves a window with one coupling per outgoing association edge --")
    print("  one per edge and not binned, so what is measured is the edge's own declared")
    print("  delay and not this script's binning.")
    assoc = model.edges["cortical_association"]
    ass = imp_of("cortical_association_propagation", "distance_prior_lti")
    th = theta_of(model, ass)
    src = np.asarray(assoc.src); dst = np.asarray(assoc.dst); d = np.asarray(
        assoc.features["distance_mm"], float)
    smap, n_neural = merged_index(model, ACT)
    # the site with the widest spread of outgoing distances, so the fit is tested
    # over the whole range of delays this topology carries rather than at one.
    counts = np.bincount(src, minlength=assoc.n_sites)
    i_glob = int(np.argmax(counts))
    sel = np.flatnonzero(src == i_glob)[:24]
    i_loc = int(smap[i_glob])

    cs = []
    for e in sel:
        m, _ = edge_mix(model, assoc, ACT, AMPA, np.ones(assoc.n_edges),
                        mask=(np.arange(assoc.n_edges) == e), symmetric=False)
        cs.append(Coupling(process=f"edge{e}", writes=AMPA, reads=(ACT,), form=Form.LTI,
                           transfer=_stiff(ass.transfer, {"distance_mm": float(d[e])}),
                           mix=m, theta=th, memory_s=0.2))
    cs.append(stiff_limit(AMPA, n_neural))

    s = State.prior(model.layout)
    z = np.zeros((n_neural, model.layout[ACT].k), dtype=np.complex128)
    z[i_loc, :] = 1.0                       # flat spectrum: every coefficient probed at once
    s.beliefs[ACT] = replace(s[ACT], mean=z)
    s.beliefs[AMPA] = replace(s[AMPA], mean=np.zeros_like(s[AMPA].mean))
    s, rep = solve_window(s, cs, basis, solve=Solve(damping=1.0, max_iter=8))
    print(f"    probe solve: {rep}")

    w = basis.omega
    v = float(th.get("velocity_m_s", 4.0))
    zo = np.asarray(s[AMPA].mean)
    fit_band = (basis.freqs_hz >= 1.0) & (basis.freqs_hz <= 40.0)
    errs, taus, decl, worst = [], [], [], 0.0
    for e in sel:
        j = int(smap[int(dst[e])])
        H = ass.transfer(basis, distance_mm=float(d[e]), **th) / (1.0 + 1j * w / GAMMA)
        got = zo[j][: basis.k]
        rel = np.abs(got - H) / np.maximum(np.abs(H), 1e-300)
        worst = max(worst, float(np.nanmax(rel[fit_band])))
        # divide out the synaptic kernel: what is left is the dispersed delay,
        # whose phase is exactly -omega tau.
        syn = synaptic_drive_transfer(
            basis, tau_ampa_s=th.get("tau_ampa_s", 3e-3),
            tau_nmda_decay_s=th.get("tau_nmda_decay_s", 0.1),
            nmda_fraction=th.get("nmda_fraction", 0.2))
        ph = np.unwrap(np.angle(got / (syn / (1.0 + 1j * w / GAMMA))))
        slope = np.polyfit(w[fit_band], ph[fit_band], 1)[0]
        taus.append(-slope)
        decl.append(float(d[e]) * 1e-3 / v)
        errs.append(abs(-slope - decl[-1]))
    taus, decl, errs = np.array(taus), np.array(decl), np.array(errs)
    print(f"    {len(sel)} association edges out of column node {i_glob}, "
          f"{d[sel].min():.0f}-{d[sel].max():.0f} mm at {v:g} m/s")
    print(f"    declared delays   {decl.min() * 1e3:.2f} - {decl.max() * 1e3:.2f} ms")
    print(f"    measured slopes   {taus.min() * 1e3:.2f} - {taus.max() * 1e3:.2f} ms")
    print(f"    |measured - declared|  median {np.median(errs) * 1e6:.3g} us, "
          f"max {errs.max() * 1e6:.3g} us")
    print(f"    pointwise |solved - declared H| / |H| over 1-40 Hz: max {worst:.3e}")
    ok = float(np.median(errs)) < 1e-4 and worst < 1e-6
    print(f"    [{'PASS' if ok else 'FAIL'}: the ramp is applied exactly; the residual is "
          "round-off plus the stiff-limit pole, not an approximation]")
    return ok


def imp_of(process: str, name: str):
    for i in REGISTRY.implementations.values():
        if i.process == process and i.name == name:
            return i
    raise KeyError(f"{process}/{name}")


def check_topography(model, state, basis, lead_data, dip, couplings):
    head("10. check 3 -- lead field and topography")
    sensors = model.sites["sensor_array"]
    xyz = np.asarray(sensors.xyz, float)
    fb = model.layout[CP].basis
    z = np.asarray(state[CP].mean)
    alpha = (fb.freqs_hz >= 8.0) & (fb.freqs_hz <= 13.0)
    # the transform is orthonormal, so the sum of |z|^2 over a band is the
    # time-domain variance that band contributes times n.  dividing by sqrt(n)
    # turns it back into microvolts RMS, which is the unit a scalp amplitude is
    # quoted in and the only one comparable with a recording.
    amp = np.sqrt((np.abs(z[:, alpha]) ** 2).sum(1) / basis.n)
    print(f"  the stepped model's own scalp map, {len(amp)} electrodes")
    print(f"    8-13 Hz amplitude   median {np.median(amp):.3g} uV rms, "
          f"range {amp.min():.3g} - {amp.max():.3g}")
    print(f"    lead-field scale    {dip['scale_uV_per_nA_mm3']:.3g} uV per nA/mm^3 at the "
          "median (source, sensor) pair,")
    print("                        via q = J . area . thickness . dipole length; the sheet's")
    print("                        own area per node, 2.5 mm ribbon, normal-projected")

    # -- spatial smoothness: the skull's centimetre kernel, measured -------
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    iu = np.triu_indices(len(amp), 1)
    corr_len = smoothness(d[iu], amp)
    print(f"    spatial correlation length of the simulated map: {corr_len:.0f} mm")

    # -- a focal source, and whether it is dipolar --------------------------
    #
    # solved rather than read off the lead field.  the two should agree, and that
    # they do is itself the check that the instrument coupling is being applied
    # by the solver the way the matrix says: a delta of current at one column
    # node is put into the state, the window is solved with the device coupling
    # and its stiff-limit leak, and what comes out at the 60 contacts is compared
    # against the lead field's own column.
    L, _ = lead_field_mix(model, lead_data)
    col = np.asarray(L.todense())
    best = int(np.argmax(np.abs(col).max(0)))       # the node the montage sees best
    dev = [c for c in couplings if c.writes == CP]
    n_tmc = merged_index(model, TMC)[1]
    s = State.prior(model.layout)
    zt = np.zeros((n_tmc, model.layout[TMC].k), dtype=np.complex128)
    zt[best, :] = 1.0
    s.beliefs[TMC] = replace(s[TMC], mean=zt)
    s.beliefs[CP] = replace(s[CP], mean=np.zeros_like(s[CP].mean))
    s, rep = solve_window(s, tuple(dev) + (stiff_limit(CP, model.layout[CP].n_sites),),
                          basis, solve=Solve(damping=1.0, max_iter=6))
    solved = np.asarray(s[CP].mean)
    ratio = solved[:, 1] / np.where(np.abs(col[:, best]) > 0, col[:, best], np.nan)
    print()
    print(f"    focal solve: {rep}")
    print(f"    solved map / lead-field column: spread across contacts "
          f"{np.nanstd(ratio) / max(abs(np.nanmean(ratio)), 1e-30):.2e} "
          "(the instrument coupling is applied as the matrix says)")
    v = solved[:, 1].real / abs(np.nanmean(ratio)) if np.isfinite(
        np.nanmean(ratio)) else col[:, best]
    pos, neg = float(v.max()), float(v.min())
    ext_p, ext_n = xyz[int(np.argmax(v))], xyz[int(np.argmin(v))]
    sep = float(np.linalg.norm(ext_p - ext_n))
    print()
    print("  a single column node driven alone (the focal-source test):")
    print(f"    extrema             {pos:+.3g} / {neg:+.3g} uV per nA/mm^3")
    print(f"    both signs present  {'yes' if pos > 0 and neg < 0 else 'NO'}   "
          f"|neg| / pos = {abs(neg) / max(pos, 1e-30):.2f}")
    print(f"    extremum separation {sep:.0f} mm across the scalp")
    print(f"    correlation length of the focal map: {smoothness(d[iu], np.abs(v)):.0f} mm")
    dipolar = pos > 0 and neg < 0 and abs(neg) / max(pos, 1e-30) > 0.1 and 40.0 < sep < 220.0
    print(f"    [{'PASS' if dipolar else 'FAIL'}: a single cortical dipole seen through a "
          "three-layer BEM should give one positive and one negative lobe, tens of "
          "centimetres apart]")
    return amp, corr_len, dipolar


def smoothness(dist, values):
    """the separation at which the spatial autocorrelation of a map falls to 1/e.

    a crude estimator and deliberately so: with 60 electrodes there is no honest
    way to fit a covariance function, and what is being compared between the
    simulated and the measured map is one number of the same crude kind computed
    the same way on both.
    """
    v = np.asarray(values, float)
    v = (v - v.mean()) / max(v.std(), 1e-30)
    n = len(v)
    iu = np.triu_indices(n, 1)
    prod = np.outer(v, v)[iu]
    edges = np.linspace(0.0, float(np.max(dist)), 25)
    mid, c = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() > 4:
            mid.append(0.5 * (lo + hi)); c.append(float(prod[m].mean()))
    mid, c = np.array(mid), np.array(c)
    below = np.flatnonzero(c < math.exp(-1.0))
    if not len(below) or below[0] == 0:
        return float(mid[-1]) if len(mid) else float("nan")
    i = int(below[0])
    # linear interpolation between the two binned correlations that straddle
    # 1/e, so the answer is not quantized to the bin width -- two maps that
    # differ by less than one bin were coming out identical.
    f = (c[i - 1] - math.exp(-1.0)) / max(c[i - 1] - c[i], 1e-30)
    return float(mid[i - 1] + f * (mid[i] - mid[i - 1]))


def measured_alpha_topography(model, basis):
    """the same 8-13 Hz map, from this subject's own recording.

    the comparison that is fair and the one that is not are worth separating.
    comparing the simulated *pattern* to the measured one is not fair: the model
    has no stimulus, no phase locking and a spatially independent prior draw for
    its cortical drive, so its pattern is a sample from a distribution whose mean
    is flat, and any agreement would be luck.  comparing the *smoothness* is
    fair, because that is a property of the volume conductor -- the skull's
    centimetre-wide kernel -- and the volume conductor is the same object in
    both.
    """
    import mne
    from scipy.signal import resample

    paths = geo.sample_paths()
    raw = mne.io.read_raw_fif(paths.raw_fif, preload=True, verbose="ERROR")
    # bads excluded: this recording marks one EEG contact bad, and a dead
    # electrode fused as evidence is a measurement of the amplifier.
    raw.pick("eeg", exclude="bads")
    ids = list(np.asarray(model.sites["sensor_array"].columns["element_ids"]))
    have = [c for c in ids if c in raw.ch_names]
    raw.pick(have)
    order = [raw.ch_names.index(c) for c in have]
    x = raw.get_data()[order] * 1e6                                # V -> uV
    sf = float(raw.info["sfreq"])
    n = basis.n
    start = int(10.0 * sf)
    seg = x[:, start:start + int(round(n * basis.dt * sf))]
    seg = resample(seg, n, axis=1)
    seg = seg - seg.mean(1, keepdims=True)
    return have, seg, basis.analyze(seg)[..., : basis.k], sf


def check_stability(model, couplings, basis, reports, nonlinear: bool):
    head("11. check 4 -- stability")
    th_ei = theta_of(model, imp_of("local_excitation", "ei_loop_lti"))
    loop = th_ei.get("loop_gain", 8.0) * series(
        alpha_synapse(basis, th_ei.get("tau_gaba_a_s", 6e-3)),
        low_pass(basis, th_ei.get("tau_inh_membrane_s", 8e-3)),
        pure_delay(basis, th_ei.get("synaptic_lag_s", 1.5e-3)))
    m_ei = stability_margin(loop)
    print(f"  ei_loop_lti's own return path, at the prior median theta:")
    print(f"    loop_gain {th_ei.get('loop_gain', float('nan')):.2f}, "
          f"min |1 + L(omega)| over 0.5-100 Hz = {m_ei:.3f}   "
          f"[{'PASS' if m_ei > 0.5 else 'MARGINAL' if m_ei > 0.1 else 'FAIL'}]")
    peak = float(basis.freqs_hz[int(np.argmin(np.abs(1.0 + loop)))])
    print(f"    the minimum sits at {peak:.1f} Hz, which is where the round trip is half a")
    print("    cycle -- pyramidal-interneuron gamma, from time constants and not fitted")

    cycles = graph_cycles(couplings)
    print()
    if cycles:
        print(f"  the assembled graph has {len(cycles)} cycle(s): "
              + "; ".join(" -> ".join(c) for c in cycles[:3]))
    else:
        print("  the assembled graph has NO cycle, and that is the finding rather than a")
        print("  convenience.  the edge that closes the cortical loop is potential -> rate,")
        print("  and the only f the ontology offers for it is the sigmoid in")
        print("  `wilson_cowan_excitatory` -- Form.RATE, the single nonlinearity in")
        print("  `ibm/processes/neural.py`.  build selected `conductance_lti` for")
        print("  `local_excitation` and every one of the 29 traced processes came out LTI,")
        print("  so under this selection the cortex is a feed-forward chain: whatever the")
        print("  loop gain of the real circuit is, this materialization cannot express it,")
        print("  and its linearised margin is trivially 1.")
    print()
    print("  boundedness over the run, per window (max |mean| over every written component):")
    for i, (r, amp) in enumerate(reports):
        print(f"    window {i}: {amp:.4g}")
    growth = reports[-1][1] / max(reports[0][1], 1e-300)
    ok = np.isfinite(growth) and growth < 10.0
    print(f"    growth over {len(reports)} windows: x{growth:.3f}   "
          f"[{'PASS' if ok else 'FAIL'}: bounded]")
    return m_ei, ok, cycles


def graph_cycles(couplings):
    """every cycle in the read -> write graph, up to length 6.

    a cycle is what a stability margin is a statement about, so whether there is
    one at all has to be checked rather than assumed: a graph with none has a
    margin of 1 for a reason that has nothing to do with the circuit.
    """
    adj: dict[str, set[str]] = {}
    for c in couplings:
        if c.process.startswith("stiff-limit"):
            continue
        for r in c.reads:
            adj.setdefault(r, set()).add(c.writes)
    out, seen = [], set()

    def walk(start, node, path):
        if len(path) > 6:
            return
        for nxt in adj.get(node, ()):
            if nxt == start:
                key = frozenset(path)
                if key not in seen:
                    seen.add(key)
                    out.append(path + [start])
            elif nxt not in path:
                walk(start, nxt, path + [nxt])

    for n in list(adj):
        walk(n, n, [n])
    return out


def check_causality(model, couplings, ac_instrument, basis):
    head("5. the causality refusal, fired on this model")
    print("  the refusal is not a quality knob.  the temporal laplacian is cyclic, so")
    print("  `exp(-i omega tau)` is a circular shift and any memory longer than the part of")
    print("  the window the previous solution has already pinned wraps round and delivers")
    print("  the window's own future as its past.  nothing raises, the spectrum stays")
    print("  finite and plausible, and the answer is wrong.")
    print()
    full = tuple(couplings) + (ac_instrument,)
    mem_full = longest_memory(full)
    mem = longest_memory(couplings)
    print("  the whole declared instrument chain, on the window `eeg_forward` declares:")
    print(f"    longest memory {mem_full * 1e3:.0f} ms, from {ac_instrument.process}")
    declared = WindowPlan.of(model.request.window)
    print(f"    plan from the request: {declared.describe()}".replace("\n", "\n    "))
    try:
        declared.refuse_if_acausal(mem_full)
        print("    IT PASSED.  that is a bug in the refusal, not a licence.")
        fired = False
    except CausalityViolation as exc:
        fired = True
        for part in str(exc).split("; "):
            print(f"    REFUSED  {part}")
    print()
    print("  and the diagnosis is the point.  the culprit is not an axon.  it is")
    print(f"  `electrode_interface`'s AC-coupling corner: a biopotential amplifier blocks")
    tau_hp = median_of(imp_of("device_coupling", "electrode_interface")
                       .params["tau_highpass_s"])
    print(f"  DC with a {tau_hp:g} s time constant, whose impulse response is")
    print("  `-(1/tau) exp(-t/tau)`, and three time constants of that is nine seconds of")
    print("  memory against a two-second window.  `eeg_forward`'s `Window(n=2048, dt=1e-3)`")
    print("  and `electrode_interface`'s `tau_highpass_s = 3.0` are inconsistent with each")
    print("  other, and one of them has to move: either the request carries an 8-second")
    print("  window -- four times the spectral coefficients, and the two budgets multiply --")
    print("  or the slow half of the amplifier is not in the model.  no amount of overlap")
    print(f"  fixes it: the plan's own suggestion is overlap {declared.for_memory(mem_full).overlap:.3f}, and overlap is")
    print("  a fraction below 1 by construction.")
    print()
    print("  the run below therefore keeps the anti-alias half of the amplifier and drops")
    print("  the AC corner, which is a stated modelling decision and not a workaround: the")
    print("  slow cortical potentials that corner attenuates are outside every claim made")
    print("  in sections 8 to 12.  with it dropped:")
    worst = max(couplings, key=lambda c: c.longest_memory_s())
    print(f"    longest memory {mem * 1e3:.0f} ms, from {worst.process}")
    tab = sorted({(c.process.split(" ")[0], c.longest_memory_s()) for c in couplings},
                 key=lambda kv: -kv[1])[:6]
    for name, t in tab:
        print(f"      {name:52s} {t * 1e3:7.1f} ms")
    print()
    bad = WindowPlan(basis, overlap=0.01, n_windows=1)
    print(f"  a plan with {bad.overlap:.0%} overlap ({bad.overlap_s * 1e3:.0f} ms):")
    try:
        bad.refuse_if_acausal(mem)
        print("    IT PASSED.  that is a bug in the refusal, not a licence.")
        fired2 = False
    except CausalityViolation as exc:
        fired2 = True
        for part in str(exc).split("; "):
            print(f"    REFUSED  {part}")
    good = WindowPlan(basis, overlap=declared.overlap or 0.25, n_windows=1)
    print()
    print(f"  the request's own plan, {good.overlap:.0%} overlap "
          f"({good.overlap_s * 1e3:.0f} ms):")
    try:
        good.refuse_if_acausal(mem)
        passed = True
        print(f"    accepted, and legal by {good.overlap_s / mem:.2f}x.  "
              f"{good.describe()}".replace("\n", "\n    "))
    except CausalityViolation as exc:
        passed = False
        print(f"    REFUSED  {exc}")
        good = good.for_memory(mem)
        print(f"    widened by `WindowPlan.for_memory` to overlap {good.overlap:.3f} "
              f"({good.overlap_s * 1e3:.0f} ms), which is legal and is what the run uses.")
        print("    that the REQUEST's own overlap is illegal for the graph the request")
        print("    traced is worth saying plainly: gaba-b's 150 ms decay alone is 450 ms of")
        print("    settling, and `eeg_forward` chose 25% of a 2.048 s window without")
        print("    anything checking it against the processes its target pulls in.")
        good.refuse_if_acausal(mem)
        passed = True
    print(f"  [{'PASS' if fired and fired2 and passed else 'FAIL'}: the refusal fires on the "
          "declared chain and on a short overlap, and a legal plan passes]")
    return good, fired and fired2 and passed


def fuse_real_eeg(model, state, basis, channels, seg, zobs, sfreq):
    head("12. one real observation, fused")
    print(f"  {seg.shape[0]} EEG channels x {seg.shape[1]} samples from "
          f"{geo.sample_paths().raw_fif.name}, resampled {sfreq:.1f} -> "
          f"{1.0 / basis.dt:.0f} Hz onto the model's own window")
    print(f"    recording RMS {np.sqrt((seg ** 2).mean()):.2f} uV")
    print("    no re-referencing is applied.  `device.contact_potential` is declared as the")
    print("    voltage 'relative to the system reference', and the recording carries its")
    print("    own; a montage change is a declared transformation this model does not")
    print("    carry, so the two are being compared in whatever reference each was in.")

    # the amplifier's noise floor, measured rather than assumed, and measured
    # ABOVE the band the model retains: between 120 and 165 Hz this recording is
    # above every cortical rhythm and still below its own 172 Hz anti-alias
    # corner.  under the orthonormal transform a white time-domain variance is
    # the same number per coefficient, which is what makes this a legitimate
    # per-coefficient precision.
    #
    # it is an UPPER bound and not the instrument's noise: temporalis and neck
    # EMG dominate the scalp spectrum from ~20 Hz upward, so what is measured
    # here is noise plus muscle.  an overestimated floor understates how far the
    # observation should move the posterior, which is the safe direction to be
    # wrong in and is stated rather than hidden.
    full = np.fft.rfft(seg, axis=-1, norm="ortho")
    ff = np.fft.rfftfreq(seg.shape[1], d=basis.dt)
    band = (ff >= 120.0) & (ff <= 165.0)
    var = float(np.mean(np.abs(full[:, band]) ** 2))
    print(f"    noise floor measured over {ff[band][0]:.0f}-{ff[band][-1]:.0f} Hz "
          f"(above B, below the recording's own anti-alias corner):")
    print(f"      {math.sqrt(var):.3g} uV per coefficient -- an UPPER bound, it still has "
          "temporalis emg in it")

    ids = list(np.asarray(model.sites["sensor_array"].columns["element_ids"]))
    sites = np.array([ids.index(c) for c in channels], int)
    obs_band = REGISTRY.observations["eeg"].band & REGISTRY.observations["eeg"].observes.band
    k = model.layout[CP].basis.band_indices(obs_band)
    # the evidence is carried on exactly the coefficients the observation has
    # precision over and no others.  the eeg band starts at the amplifier's AC
    # corner, so DC is not among them: a scalp recording says nothing about the
    # contact's mean, and handing it the full array would have asserted that it
    # does.
    ev = from_observation("eeg", zobs[:, k], np.full((len(sites), len(k)), var),
                          sites=sites, source="mne-sample sample_audvis_raw.fif")
    print(f"    {ev.describe()}")
    print(f"    carried on {len(k)} of the block's {model.layout[CP].k} coefficients "
          f"({basis.freqs_hz[k[0]]:.2f}-{basis.freqs_hz[k[-1]]:.1f} Hz)")

    before = state[CP]
    after = fuse(state, ev)
    a = after[CP]
    sl = np.ix_(sites, k)
    dmu = np.abs(a.mean[sl] - before.mean[sl])
    sd0 = np.sqrt(np.maximum(before.total_psd()[sl], 1e-300))
    shrink = a.psd[sl] / np.maximum(before.psd[sl], 1e-300)
    print()
    print(f"  the posterior over {CP}, on the {len(sites)} observed contacts and the "
          f"{len(k)} coefficients the eeg observation carries precision over:")
    print(f"    prior sd            median {np.median(sd0):.4g} uV")
    print(f"    |posterior - prior| median {np.median(dmu):.4g} uV "
          f"= {np.median(dmu / sd0):.2f} prior sd, max {np.max(dmu / sd0):.2f}")
    print(f"    variance ratio      median {np.median(shrink):.4f} "
          f"(1 = the evidence said nothing, 0 = it pinned the state)")
    print(f"    effective constraints {ev.effective_constraints():.0f} against "
          f"{dmu.size:,} fused numbers")
    outside = np.setdiff1d(np.arange(model.layout[CP].k), k)
    if len(outside):
        moved = np.abs(a.mean[:, outside] - before.mean[:, outside]).max()
        print(f"    outside the observation's band ({len(outside)} coefficients): "
              f"max |change| {moved:.3g} uV -- zero precision, zero movement, no special case")
    ok = float(np.median(dmu / sd0)) > 0.05 and float(np.median(shrink)) < 0.999
    print(f"  [{'PASS' if ok else 'FAIL'}: the posterior moved and narrowed]")
    return after, ok


def window_amplitude(state, couplings) -> float:
    m = 0.0
    for cid in dict.fromkeys(c.writes for c in couplings):
        if cid in state and state.layout[cid].uncertainty == "spectral":
            m = max(m, float(np.max(np.abs(state[cid].mean))))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", action="store_true",
                    help="a cheaper r(q) for iterating on this script")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--no-nonlinear", action="store_true",
                    help="skip the closed-loop Form.RATE probe in section 13")
    args = ap.parse_args()
    t_start = time.time()
    rng = np.random.default_rng(0)

    head("1. the model")
    model, lead_data = build_model(coarse=args.coarse, rebuild=args.rebuild)
    print(model.describe())
    if args.build_only:
        return 0

    # ------------------------------------------------------------------
    head("2. the two conventions this run had to reconcile")
    print("  `ibm/processes/base.py` opens by fixing its calling convention: an")
    print("  implementation's transfer is `transfer(basis, **theta) -> H(omega)`.  every")
    print("  function in its library is an INPUT-TO-OUTPUT filter -- `low_pass` is \"the")
    print("  membrane, in one line\", `feedback` returns a closed-loop response,")
    print("  `pure_delay` is the delay itself.")
    print()
    print("  `ibm/runtime/step.py` solves `i omega z = sum_p f_p(z)`, and")
    print("  `Coupling.spectral_drift` is documented as \"this coupling's contribution to")
    print("  dz/dt\".  what it wants from a coupling is PRESSURE, not a response.")
    print()
    print("  those are different objects and nothing in the repository converts between")
    print("  them, because no materialized model had ever been stepped.  read one way, a")
    print("  membrane low-pass acquires a second pole from the integrator; read the other,")
    print("  a self-coupling's H is used as a rate and `low_pass` becomes a decay rate of")
    print("  +1 at DC.  ARCHITECTURE.md 4 says what the conversion is: an algebraic")
    print("  relation x_O = g(x_I) is the stiff limit of pressure, -gamma (x_O - g(x_I)),")
    print("  and it 'is a question of how it is solved, not of what it is'.")
    print()
    print(f"  so every declared transfer enters as +gamma H(omega) x_I, and each written")
    print(f"  component carries one -gamma x_O leak.  gamma = 2 pi x {STIFF_RATE_HZ:g} Hz,")
    print("  chosen from its error term and nothing else:")
    for hz in (1.0, 10.0, 40.0, 100.0):
        r = 2 * math.pi * hz / GAMMA
        print(f"    at {hz:6.1f} Hz   amplitude low by {100 * r ** 2 / 2:7.4f}%, "
              f"phase lags {math.degrees(math.atan(r)):.3f} deg")
    print("  the drives sum and the leak is declared once, which is what keeps N writers")
    print("  onto one conductance a sum rather than an average.")

    # ------------------------------------------------------------------
    head("3. the process graph, and what could not be assembled from it")
    print(PAIRING_ARGUMENT)
    print()
    couplings, dip, ac_instrument = assemble(model, lead_data)
    print()
    print(f"  {len(couplings)} couplings over {len(set(c.writes for c in couplings))} written "
          f"components; {sum(1 for c in couplings if c.linear)} linear, "
          f"{sum(1 for c in couplings if not c.linear)} not")
    print()
    print("  the fan-in each path actually carries -- the mean row sum of its mixing")
    print(f"  matrix, i.e. how much drive one node receives into {AMPA}:")
    fan = {}
    for c in couplings:
        if c.mix is None or c.writes != AMPA:
            continue
        name = c.process.split("[")[0]
        fan[name] = fan.get(name, 0.0) + float(np.asarray(c.mix.sum(1)).ravel().mean())
    for name in sorted(fan, key=lambda x: -fan[x]):
        print(f"    {name:36s} {fan[name]:12.4g}")
    print("  the association term is three orders of magnitude above the local one, and")
    print("  that is arithmetic rather than physiology: the builder SAMPLED a dense")
    print("  geometric prior at well under a percent and `association_transfer` divides by")
    print("  the inclusion probability so that the sampled sum is unbiased for the dense")
    print("  graph, which makes each kept partner stand for ~160.  the `gain` that would")
    print("  hold it down is `weak(1.0, 5.0)` and has never been fitted.  every amplitude")
    print("  below inherits this, which is why section 10 calls its agreement with the")
    print("  recording an order of magnitude rather than a match.")
    print()
    print("  what the materialized graph could NOT be asked to do, and why:")
    em = model.edges["electromagnetic"]
    print(f"    * `em_generation` writes {', '.join(sorted(c for c in model.layout.components if c.startswith('electromagnetic')))}")
    print(f"      and every one of those blocks is on `head_volume`.  the `electromagnetic`")
    print(f"      edge set it runs over has {em.n_edges:,} edges from `cortical_surface` to")
    print(f"      `sensor_array` -- the lead field -- so its destinations carry no component")
    print(f"      this process writes.  the BEM solution was computed, sparsified into a")
    print(f"      topology, and connects nothing in the state graph.  the chain")
    print(f"      cortex -> head volume potential -> contact is broken at its first link.")
    print(f"    * this script therefore applies the lead field where it physically acts:")
    print(f"      {TMC} -> {CP}, which is em_generation . em_coupling . device_coupling")
    print(f"      composed, and says so rather than reporting a forward model it did not run.")
    print(f"    * `tractometric` built 0 edges: this subject has no tractogram bound, so")
    print(f"      `tract_propagation` and `thalamocortical_coupling` have no anatomy.  the")
    print(f"      alpha resonance is applied site-locally in section 8 and that is stated")
    print(f"      there; the delay check in section 9 uses `cortical_association` instead,")
    print(f"      whose delays are real geometry even though its edges are a distance prior.")
    print(f"    * build selected `acoustic_beam` -- an ultrasound transducer's beam pattern,")
    print(f"      with a time of flight and a carrier -- as the implementation of")
    print(f"      `device_coupling` for a 60-channel scalp EEG montage.  this script uses")
    print(f"      `quasistatic_lead_field` x `electrode_interface` instead and names both.")

    # ------------------------------------------------------------------
    head("4. the window")
    basis = model.basis.truncated(SCALP)
    print(f"  the solve runs on the band the request declared, not the window's full")
    print(f"  nyquist: model basis k={model.basis.k} (0-{model.basis.nyquist_hz:g} Hz) "
          f"truncated to k={basis.k} (0-{basis.freqs_hz[-1]:.1f} Hz), which is B.")
    print(f"  resolution {basis.freqs_hz[1]:.4f} Hz over {basis.duration_s:g} s")
    ks = sorted({model.layout[c].k for c in model.layout.components
                 if model.layout[c].uncertainty == "spectral"})
    print(f"  the blocks do not all carry the same width: k in {ks}.  a coefficient a block")
    print("  does not carry cannot have pressure applied to it, and until this run `_drift`")
    print("  did not know that -- see section 14.")

    plan, causal_ok = check_causality(model, couplings, ac_instrument, basis)
    plan = replace(plan, n_windows=args.windows)

    # ------------------------------------------------------------------
    head(f"6. the solve: {args.windows} windows of {basis.duration_s:g} s, "
         f"hop {plan.hop_s:g} s")
    state0 = initial_state(model, basis, rng)
    drive = drive_realisation(model, basis, plan, rng)
    state0.beliefs[ACT] = replace(state0[ACT],
                                  mean=basis.analyze(drive[:, :basis.n])[
                                      ..., : model.layout[ACT].k])
    exo = exogenous(model, tuple(couplings))
    print(f"  {len(exo)} spectral components are exogenous -- no assembled coupling writes")
    print(f"  them, so `_targets` leaves them at the prior and they are what drives the")
    print(f"  model: {', '.join(exo[:6])}{' ...' if len(exo) > 6 else ''}")
    print(f"  {ACT} among them is set from one continuous realisation of its own declared")
    print(f"  prior (`{REGISTRY.components[ACT].prior}`), {drive.shape[1]:,} samples long, "
          "sliced per window.")
    sheet_off = model.sites["cortical_surface"].offset
    smap, _ = merged_index(model, POT)
    sheet_sites = smap[sheet_off:sheet_off + model.sites["cortical_surface"].n]
    solve = Solve(damping=0.7, max_iter=40, newton=True, max_newton=1,
                  krylov_restart=6, krylov_maxiter=24)
    print(f"  Solve(damping={solve.damping}, tol={solve.tol:g}, max_iter={solve.max_iter}, "
          f"newton={solve.newton})")
    print(f"  krylov restart is {solve.krylov_restart} and not the default 40 for a reason")
    print(f"  worth stating: `pack_means` over the written components of this model is")
    print(f"  {sum(model.layout[c].n_sites * model.layout[c].k * 2 for c in set(c.writes for c in couplings)):,} "
          f"reals, so every krylov vector is "
          f"{sum(model.layout[c].n_sites * model.layout[c].k * 2 for c in set(c.writes for c in couplings)) * 8 / 2**30:.2f} GiB.")
    trace_amp: list[tuple[Any, float]] = []
    rel_a: list[float] = []
    t0 = time.time()

    def on_window(i, s, rep):
        trace_amp.append((rep, window_amplitude(s, couplings)))
        rel_a.append(rep.joint_gap / max(joint_scale(s, couplings), 1e-300))
        print(f"    {rep}")

    state1, reports = advance(state0, couplings, plan, solve=solve, on_window=on_window)
    print(f"  {time.time() - t0:.0f}s for {args.windows} windows")
    conv = all(r.converged for r in reports)
    print(f"  converged: {conv}   limited_by: "
          f"{[r.limited_by or '-' for r in reports]}   "
          f"iterations: {[r.iterations for r in reports]}   "
          f"newton solves: {[r.newton_calls for r in reports]}")
    print(f"  joint gaps: {['%.3g' % r.joint_gap for r in reports]}")
    print(f"  residual / initial: {['%.2e' % (r.residual / max(r.residual0, 1e-300)) for r in reports]}")

    state1b, gaps_b, amps_b, reps_b = continuity_probe(model, couplings, basis, plan, solve,
                                                       drive, state0)
    gap_a = max(rel_a) if rel_a else float("nan")
    gap_b = max(gaps_b)
    res_a = max(r.residual / max(r.residual0, 1e-300) for r in reports[1:]) if len(
        reports) > 1 else float("nan")
    res_b = (max(r.residual / max(r.residual0, 1e-300) for r in reps_b[1:])
             if len(reps_b) > 1 else float("nan"))
    print()
    print("  the same three windows, compared on the two numbers that mean something:")
    print(f"    joint gap / state rms     {gap_a:.4g} frozen   ->  {gap_b:.4g} advanced")
    print(f"    residual / initial        {res_a:.4g} frozen   ->  {res_b:.4g} advanced")
    print("  so `limited_by=\"continuity\"` in the run above was mostly reporting the price of")
    print("  freezing the drive and not the price of imposing continuity on the dynamics.")
    print("  the missing piece is in `advance`: a run over several windows needs the")
    print("  caller to supply the next window of every exogenous component, and there is")
    print("  no argument, callback or protocol for it.  `on_window` fires after the solve")
    print("  and before `Carry.tail`, so even mutating the state there would poison the")
    print("  carry with the next window's input.")

    # ------------------------------------------------------------------
    state1 = state1b            # the run with the drive advanced, per section 7
    trace_amp = list(zip(reps_b, amps_b))
    conv_note = (
        f"window 0 converged in {reps_b[0].iterations} picard sweeps "
        f"({reps_b[0].residual / max(reps_b[0].residual0, 1e-300):.1e} of its initial "
        "residual), 0 newton solves; "
        + ("later windows too" if all(r.converged for r in reps_b[1:]) else
           f"windows 1-{len(reps_b) - 1} stopped at limited_by="
           f"{reps_b[1].limited_by!r} with the residual at "
           f"{max(r.residual / max(r.residual0, 1e-300) for r in reps_b[1:]):.0%} of "
           "initial"))
    ok_beta, ok_alpha = check_spectrum(model, state0, state1, basis, sheet_sites)
    ok_delay = check_delays(model, basis, rng)
    amp, corr_len, ok_topo = check_topography(model, state1, basis, lead_data, dip,
                                              couplings)
    m_ei, ok_bounded, cycles = check_stability(model, couplings, basis, trace_amp, False)

    channels, seg, zobs, sfreq = measured_alpha_topography(model, basis)
    fb = model.layout[CP].basis
    alpha = (fb.freqs_hz >= 8.0) & (fb.freqs_hz <= 13.0)
    meas_amp = np.sqrt((np.abs(zobs[:, alpha[: zobs.shape[1]]]) ** 2).sum(1) / basis.n)
    xyz = np.asarray(model.sites["sensor_array"].xyz, float)
    ids = list(np.asarray(model.sites["sensor_array"].columns["element_ids"]))
    sel = np.array([ids.index(c) for c in channels], int)
    d = np.linalg.norm(xyz[sel][:, None] - xyz[sel][None], axis=2)
    iu = np.triu_indices(len(sel), 1)
    meas_len = smoothness(d[iu], meas_amp)
    print()
    print("  against this subject's own recording (section 10, continued):")
    print(f"    measured 8-13 Hz amplitude   median {np.median(meas_amp):.3g} uV rms "
          f"(range {meas_amp.min():.3g} - {meas_amp.max():.3g})")
    print(f"    simulated 8-13 Hz amplitude  median {np.median(amp):.3g} uV rms   "
          f"ratio {np.median(amp) / max(np.median(meas_amp), 1e-30):.2f}x")
    print(f"    correlation length  measured {meas_len:.0f} mm, simulated {corr_len:.0f} mm")
    print("    what is compared here is the SMOOTHNESS and the ORDER OF MAGNITUDE, not the")
    print("    pattern.  comparing the patterns would not be fair: the model has no")
    print("    stimulus, no phase locking, and a spatially independent draw from the prior")
    print("    for its cortical drive, so its map is one sample from a distribution whose")
    print("    mean is flat.  the smoothness is fair, because it is a property of the")
    print("    volume conductor and the volume conductor is the same head in both.")

    state2, ok_fuse = fuse_real_eeg(model, state1, basis, channels, seg, zobs, sfreq)

    # ------------------------------------------------------------------
    nonlinear_note = "not run"
    if not args.no_nonlinear:
        nonlinear_note = run_nonlinear(model, lead_data, basis, rng)

    # ------------------------------------------------------------------
    head("14. what was fixed in ibm/runtime, and why")
    # the band-mask fix, measured on this model rather than asserted.
    narrow = [c for c in model.layout.components
              if model.layout[c].uncertainty == "spectral"
              and 0 < model.layout[c].k < basis.k]
    lost, total = 0.0, 0.0
    for c in couplings:
        if c.writes not in narrow:
            continue
        d = c.spectral_drift(state1, basis)
        k = model.layout[c.writes].k
        lost += float(np.sum(np.abs(d[:, k:]) ** 2))
        total += float(np.sum(np.abs(d) ** 2))
    print(f"  measured on this run, for the fix in point 1 below: the blocks narrower than")
    print(f"  the solve are {', '.join(narrow)}")
    print(f"  ({', '.join(str(model.layout[c].k) for c in narrow)} coefficients against the "
          f"solve's {basis.k}), and the pressure that")
    print(f"  landed above their own width was {math.sqrt(lost / max(total, 1e-300)):.1%} of "
          "the pressure written to them --")
    print("  an irreducible residual of that size, in a solve whose tolerance is 1e-8.")
    print()
    print("""  seven things, all of them invisible until a materialized model was stepped.

  ibm/runtime/step.py

  1. `_drift` now zeroes pressure above each written block's own retained width.
     the solve writes back `z[:, :block.k]` but the residual was computed over
     the whole window basis, so for any block whose band is narrower than the
     solve's -- here `neural.exc.nmda`, `neural.exc.adaptation` and
     `neural.inh.gaba_b` at 0.5-50 Hz against a 0.5-100 Hz solve -- the drift
     between k=103 and k=205 was residual the iterate could never remove.  the
     window could not converge, at all, for a reason that reads from outside as
     a solver stalling.  a model with one band everywhere never sees it; a model
     that makes bandwidth a laziness axis per component sees it immediately.

  2. `_match_overlap` re-analyses at the block's width, not the window's.  it was
     writing a mean of `basis.k` coefficients back onto a belief whose psd had
     `block.k`, which survives until something asks for the total power.

  3. `Coupling.spectral_drift` and `scalar_drift` call a rate law as
     `fn(x, theta)`, positionally.  `ibm/processes/base.py` declares that
     convention at the top of the file as "fixed, and load-bearing"; step.py
     spread theta as keyword arguments, so no registered Form.RATE
     implementation could be called at all -- every one of them is written
     `def f(x, theta)`.

  4. `_newton_correction` packs the residual to each block's own width.  the
     residual is computed over the window basis and `pack_means` packs the state
     over each block's basis, so the moment one block is narrower than the solve
     -- again, the ordinary case as soon as B varies per component -- the two
     lengths differed and `LinearOperator` was handed a rectangular shape.  on
     this model that is 28,350,680 residual entries against 24,827,192 state
     entries, and scipy raises `expected square matrix` from three frames down.
     newton-krylov could never have run on a model with per-component bandwidth,
     and the only way to find that out was to make picard stall on one.

  ibm/runtime/ensemble.py

  5. `reproject_nonlinear` had the same `fn(x, **theta)` call, and the same fix.

  6. `reproject_nonlinear` also matched moments at the window's width and added
     the result onto a psd of the block's width, which is a numpy broadcast
     error the moment two components differ in bandwidth.  same family as 1 and
     4: three separate places assumed one band for the whole model.

  ibm/runtime/state.py

  7. `State.prior` allocates at the prior the component actually declares.  every
     component names a builder in `ibm.fields.priors` -- `neural_population` is
     1/f with theta, alpha and beta bumps and cites its sources -- and `prior()`
     was allocating a generic 1/f^beta for everything, throwing the ontology's
     own priors away at the last possible moment.  a run initialised that way
     cannot show an alpha peak because it was never given one.""")

    head("15. what is still untested")
    same = all(np.array_equal(state0[c].psd, state1[c].psd)
               for c in dict.fromkeys(c.writes for c in couplings) if c in state1)
    print(f"""  the honest list, after this run.

  * the uncertainty is never propagated.  every written component's psd after
    {args.windows} windows is bit-identical to its prior ({'confirmed' if same else 'NOT confirmed'}, checked
    element-wise).  ARCHITECTURE.md 4 says the push-forward of a
    linear f is exact and `SpectralGaussian.apply_transfer` is that push-forward,
    but `solve_window` moves means only and `reproject_nonlinear` widens the
    belief of components a NONLINEAR coupling writes.  an all-LTI materialization
    -- which is what build selected, 29 of 29 -- therefore returns every psd
    exactly as it found it, silently.  this is the largest remaining hole and it
    is a design decision that has not been taken, not a bug I could fix here.
  * `Form.CONSTRAINT` is never exercised.  build selected
    `capacitive_admittivity_lti` over `quasistatic_lead_field` for
    `em_generation`, so the one CONSTRAINT implementation in the inventory was
    not chosen, and this script encodes the stiff limit by hand instead of
    through `Coupling.stiff`.  the `-gamma` / `+gamma g(x_I)` split inside
    `Coupling` has still never run on a real model.
  * scalar-form blocks never move.  `_advance_scalar` runs on every window here
    and every scalar block in this model -- material, structural, metabolic,
    thermal, mechanical -- is written by no assembled coupling, so it integrates
    a zero rate.  the closed-form pole advance is untested against anything.
  * clamps and interventions: `apply_clamps` is called once per sweep with an
    empty tuple.  `ibm.runtime.intervene` is untouched.
  * the `factor` (low-rank) and `relation` (phase-preference) paths of
    `fuse_spectral` are untested: the eeg evidence here is circular and diagonal,
    so the 2x2 eigenplane solve and the woodbury path were both skipped.
  * multi-support dynamics.  every coupling above acts on the merged block, so
    the 531 `tissue` sites are stepped with the sheet's own edge sets; nothing
    tests a process that reads one support and writes the other.
  * every number here rests on prior medians.  `MaterializedModel.priors` has
    {len(model.priors)} entries and provenance says 0 of them have been moved by evidence.
    the spectrum, the amplitudes and the loop gains are what the literature
    priors say, not what this subject's data says.""")
    head("16. the verdict")
    rows = [
        ("solver converged", conv_note),
        ("check 1  spectrum", f"{'PASS' if ok_beta and ok_alpha else 'FAIL'}"),
        ("check 2  delays", f"{'PASS' if ok_delay else 'FAIL'}"),
        ("check 3  topography", f"{'PASS' if ok_topo else 'FAIL'}"),
        ("check 4  stability", f"{'PASS' if ok_bounded else 'FAIL'}"),
        ("causality refusal", f"{'PASS' if causal_ok else 'FAIL'}"),
        ("evidence fusion", f"{'PASS' if ok_fuse else 'FAIL'}"),
        ("closed loop (Form.RATE)", nonlinear_note),
    ]
    for name, verdict in rows:
        print(f"  {name:28s} {verdict}")
    print(f"\n  total run time {time.time() - t_start:.0f}s")
    return 0


def run_nonlinear(model, lead_data, basis, rng) -> str:
    """close the cortical loop with the one nonlinearity the ontology has.

    the LTI selection cannot close it (section 11), so this swaps
    `local_excitation` to `wilson_cowan_adaptive` -- Form.RATE, the sigmoid and
    the adaptation current, and the only implementation in
    `ibm/processes/neural.py` that returns a derivative rather than a filter.
    that makes the graph recurrent, gives `stability_margin` something to be a
    statement about, and is the only way anything here exercises the round-trip,
    the newton fallback and the ensemble reprojection.
    """
    head("13. the closed loop: the one nonlinearity, and what the solver does with it")
    cs, _, _ = assemble(model, lead_data, nonlinear=True, verbose=False)
    print(f"  {len(cs)} couplings, {sum(1 for c in cs if not c.linear)} of them Form.RATE")
    th = {name: median_of(pr) for name, pr
          in imp_of("local_excitation", "wilson_cowan_adaptive").params.items()}
    tau_m = float(th.get("tau_membrane_s", 0.015))
    slope = float(th.get("slope_mv", 4.0))
    r_max = float(th.get("r_max_hz", 100.0))
    drive = float(th.get("drive_gain", 1.0))
    v_half, v_rest = float(th.get("v_half_mv", -55.0)), float(th.get("v_rest_mv", -65.0))
    # the sigmoid's slope at the resting potential: the gain of the edge that
    # closes the loop, and the one number the linearisation turns on.
    x = math.exp(-(v_rest - v_half) / slope)
    r0 = r_max / (1.0 + x)
    dr = r_max * x / (slope * (1.0 + x) ** 2)
    print(f"  resting fixed point: v = {v_rest:g} mV -> r = {r0:.2f} Hz, "
          f"dr/dv = {dr:.3f} Hz/mV")

    # total synaptic weight one node receives, from the assembled mixes
    w_tot = 0.0
    for c in cs:
        if c.writes == AMPA and c.mix is not None:
            w_tot += float(np.asarray(c.mix.sum(1)).ravel().mean())
    ex = imp_of("local_excitation", "conductance_lti")
    th_ex = theta_of(model, ex)
    syn = ex.transfer(basis, **th_ex)
    w = basis.omega
    loop = (w_tot * syn / (1.0 + 1j * w / GAMMA)
            * drive / (1.0 + 1j * w * tau_m)
            * dr / (1.0 + 1j * w * 5e-3))
    # excitatory recurrence is POSITIVE feedback, so the closed-loop denominator
    # is 1 - L and the margin is min |1 - L|.  `stability_margin` computes
    # min |1 + L|, so it is handed -L.  getting this sign wrong turns a runaway
    # circuit into a stable-looking one, which `base.feedback` says in as many words.
    m = stability_margin(-loop)
    peak = float(basis.freqs_hz[int(np.argmin(np.abs(1.0 - loop)))])
    print(f"  mean total synaptic weight onto one node: {w_tot:.4g} "
          f"(local + lateral + association, horvitz-thompson corrected)")
    print(f"  |L(0)| = {abs(loop[0]):.4g}; min |1 - L| over the band = {m:.4g} at "
          f"{peak:.1f} Hz")
    print("  note what `stability_margin` does and does not say here.  it is min |1 + L|,")
    print("  the distance from the nyquist curve to -1, and with |L| in the thousands that")
    print("  distance is large for the wrong reason: the curve is far from -1 because it is")
    print("  far from everything, not because the loop is damped.  the number that decides")
    print("  stability for a positive-feedback loop is |L| against 1, and it is above it by")
    print(f"  {abs(loop[0]):.0f}x.  a margin quoted without the gain beside it is misleading,")
    print("  and `base.stability_margin`'s docstring says it is a statement about a fixed")
    print("  point, not about what the nonlinear system does instead.")
    if abs(loop[0]) > 1.0:
        print("  the loop gain is above unity at DC, so the linearised fixed point is")
        print("  UNSTABLE.  that is a statement about the prior medians, not about cortex:")
        print("  every `gain` in this chain is `weak(1.0, 5.0)` and the association term is")
        print("  reweighted by 1/inclusion_prob to stand for a dense graph that was sampled")
        print("  at ~0.6%, so the recurrent gain is whatever the geometry says it is with no")
        print("  data holding it down.  the nonlinear solve below is what actually happens.")

    print()
    mem = longest_memory(cs)
    print(f"  a multi-window plan is REFUSED before anything is solved: the graph's longest")
    print(f"  memory is now {mem * 1e3:.0f} ms, from the adaptation current's own time")
    print(f"  constant, against a {basis.duration_s * 1e3:.0f} ms window --")
    try:
        WindowPlan(basis, overlap=0.25, n_windows=2).refuse_if_acausal(mem)
        print("    IT PASSED, which would be a bug in the refusal.")
    except CausalityViolation as exc:
        for part in str(exc).split("; "):
            print(f"    REFUSED  {part}")
    print("  and no overlap fixes it, because the memory is most of the window.  that is a")
    print("  real result: selecting the one f in the inventory that can produce up/down")
    print("  alternation makes `eeg_forward`'s declared window too short to stitch, and the")
    print("  request has no way to know that -- the window is chosen against the target,")
    print("  not against the implementations the target's trace selects.")
    print()
    print("  so what follows is ONE window, solved through `solve_window` directly with no")
    print("  carry and no stitching, which is legal: the refusal is about the overlap")
    print("  between windows and there is no second window.  the cyclic wrap WITHIN the")
    print("  window is still there and is the reason the multi-window run is refused.")
    solve = Solve(damping=0.3, max_iter=25, newton=True, max_newton=1,
                  krylov_restart=4, krylov_maxiter=6, ensemble=8, seed=1)
    st = initial_state(model, basis, rng)
    t0 = time.time()
    try:
        st, rep = solve_window(st, cs, basis, solve=solve)
        print(f"    {rep}")
        v = st.mean_time(POT)
        r = st.mean_time(ACT)
        finite = bool(np.isfinite(v).all() and np.isfinite(r).all())
        print(f"  {time.time() - t0:.0f}s.  within the window: "
              f"{POT} spans {v.min():+.4g} to {v.max():+.4g} mV, "
              f"{ACT} spans {r.min():+.4g} to {r.max():+.4g} Hz")
        bounded = finite and abs(v).max() < 1e6
        print(f"  finite everywhere: {finite}   "
              f"[{'bounded' if bounded else 'DIVERGED'}]")
        if not bounded or abs(r).max() > 2.0 * th.get("r_max_hz", 100.0):
            print(f"  the rate leaves [0, {th.get('r_max_hz', 100.0):g}] Hz, and the sigmoid "
                  "is not what let it:")
            print("  `wilson_cowan_excitatory` writes `(r_sig(v) - r)/5e-3` -- the rate is a")
            print("  state variable integrating TOWARDS the saturating curve, not the curve")
            print("  itself -- so an unstable loop carries it wherever it likes.  the overflow")
            print("  warning above is `_sigmoid` saturating exactly as it should while the")
            print("  variable it feeds runs away.  with a DC loop gain of "
                  f"{abs(loop[0]):.0f} that is the")
            print("  correct answer for these parameters and not a solver failure: the same")
            print("  1/inclusion_prob reweighting that section 3 measured is in the loop now,")
            print("  and nothing has ever fitted the gain that would hold it down.")
        from ibm.runtime.ensemble import advise, reproject_nonlinear
        m, why = advise(model.layout, cs, target_mc_error=0.05)
        print(f"  ensemble advice: {why}")
        before = {c: st[c].psd.copy() for c in (POT, ACT, ADAPT)}
        diag = reproject_nonlinear(st, cs, basis, m=solve.ensemble,
                                   rng=np.random.default_rng(1))
        for d in diag:
            grew = float(np.mean(st[d.component].psd / np.maximum(before[d.component], 1e-300)))
            print(f"    {d}  -> psd x{grew:.3g}")
        note = (f"{rep.method}, {rep.iterations} iters, "
                f"{'converged' if rep.converged else 'NOT CONVERGED'}"
                f"{', ' + rep.limited_by if rep.limited_by else ''}, "
                f"{'bounded' if bounded else 'DIVERGED'}, "
                f"psd reprojected from {solve.ensemble} members")
    except Exception as exc:                                   # noqa: BLE001
        note = f"{type(exc).__name__}: {exc}"
        print(f"  the nonlinear run raised: {note}")
    print()
    print("  this is the only place in the whole run where a belief's WIDTH moves at all.")
    print("  everything in sections 6 to 12 is LTI, and `solve_window` propagates means.")
    return note


if __name__ == "__main__":
    raise SystemExit(main())
