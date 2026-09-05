#!/usr/bin/env python
"""do four heterogeneous datasets actually update ONE set of process parameters?

    p(theta | D)  proportional to  p(theta) prod_d p(D_d | theta)

that product is the sentence the whole architecture rests on.  §4 says forging
*has the semantics of* posterior updating and that "this common mechanism covers
hand-engineered initialization, atlas and literature initialization, statistical
fitting, pretrained learned processes, heterogeneous supervised forging,
distillation and subject-specific adaptation".  the middle one of those --
heterogeneous supervised forging -- has never been run.  `ibm/forge/fit.py` was
written for it: `Task` carries a `source` and a `weight`, `Method.JOINT` exists,
and `_partition` does union-find over the parameter blocks tasks share.  every
fit in this repository so far has handed it one dataset and one process's
parameters, which exercises the plumbing and not the claim.

the claim is worth stating in its strongest form before testing it, because the
strong form is what makes the architecture pay for itself.  `local_excitation`
appears in 20 of the 40 named models, and the 22 parameters of its four
implementations are the SAME parameters whether the materialization ends in a
scalp electrode, a BOLD voxel or an invasive array.  fifteen processes appear in
five or more models.  if that sharing is real, then a membrane time constant is
constrained by a scalp spectrum at 40 Hz AND by a haemodynamic fluctuation at
0.05 Hz, and the two compose by adding precisions with no hand-built bridge
between the modalities -- because a bridge is what a process graph IS.

if the sharing is nominal, three things happen instead, and each of them is
visible here.  a joint fit does no better than four separate fits.  a dataset
moves nothing outside its own band.  and the datasets disagree about a parameter
they are all supposed to be measuring, which is not a failure of the optimizer
but a falsification of either the shared-parameter assumption or of the process
that is claimed to link the two modalities.

    (a) TRANSFER      does the jointly fitted theta beat the per-dataset theta on
                      HELD-OUT subjects of each dataset?  if joint is worse on a
                      dataset than that dataset's own fit, the parameters are not
                      really shared and that is the finding.
    (b) CROSS-BAND    does ds004873's haemodynamic evidence at 0.01-0.35 Hz move
                      `local_excitation:ei_loop_lti.tau_membrane_s`, which
                      eegmmidb constrains through a corner at 10 Hz and a gamma
                      shoulder at 40-70?  measured two ways: the posterior with
                      and without each dataset, and the evidence in nats each
                      dataset's own likelihood carries about that one parameter.
    (c) CONFLICT      fit each dataset alone and ask whether any two of them are
                      mutually inconsistent about a shared parameter given their
                      own stated uncertainties.  this is the most important test
                      here.  a conflict localises WHERE the model is wrong, which
                      is the one thing a per-dataset fit can never do.
    (d) LEAVE-ONE-OUT fit on three, predict the fourth.  transfer across MODALITY
                      rather than across subjects.

### the one forward model, and why it is composed this way

every likelihood below is a statement about the SHAPE of a power spectrum, and
they all read the same registered transfer functions:

    local(f)  =  |H_ei(f)|^2  +  A |H_res(f)|^2
    scalp(f)  =  g_r f^-e local(f)                          eegmmidb, sleep-edfx, ds000117
    bold(f)   =  g_r [ f^-e local(f) |H_nvc(f)|^2 |H_wk(f)|^2  +  n_r ]     ds004873

`H_ei` is `ibm.processes.neural.ei_loop_transfer`, the registered implementation
`local_excitation:ei_loop_lti`; `H_res` is `alpha_resonance_transfer`, which is
`thalamocortical_coupling:alpha_resonator`; `H_nvc` is `vascular.nvc_transfer`
(`neurovascular_coupling:balloon_lti`) and `H_wk` is `windkessel_transfer`
(`vascular_flow:windkessel_lti`).  none of them is reimplemented here.  if they
were, the thing fitted and the thing a materialization would run could drift
apart and nothing would notice.

the thalamocortical path is added in PARALLEL and the vascular chain is applied
in SERIES, and both choices are forced rather than tasteful.  the E-I loop and
the thalamocortical loop are two pathways driven by the same cortical input and
summing at the same cortical output, so their powers add; cascading them, as a
single-process script can get away with, multiplies two second-order rolloffs
into an f^-10 high-frequency tail that no scalp recording has and that forces the
drive exponent onto the boundary of any grid (checked: it does exactly that).
the vascular chain, by contrast, genuinely is downstream -- the BOLD signal is
the cortical output passed through neurovascular coupling and a windkessel -- so
it multiplies.  the consequence is the thing test (b) exists to measure: the
BOLD likelihood is a function of every `local_excitation` parameter, through the
process graph, with no special-casing.  whether it carries any INFORMATION about
them is then a measurement rather than an assumption.

two nuisances are profiled per recording and identically for the prior and for
every posterior, so that no comparison below is a comparison of nuisances.
`g_r` is the overall scale, which for a scalp psd is skull conductivity,
electrode impedance and the reference montage and for a BOLD series is the echo
time and the receive coil -- profiled in closed form at its exact gamma maximum.
`e` is the exponent of the broadband drive, profiled on a grid; it is a property
of the input to the process graph, not of the graph, and letting it be shared
across modalities would manufacture the coupling this script is trying to detect.
`n_r`, a white floor as a fraction of the model's own peak, is profiled for the
BOLD source only, because above ~0.1 Hz a BOLD series is thermal and
physiological noise rather than haemodynamics and a model without a floor would
spend its whole likelihood there.  the scalp sources get no floor because they do
not need one (r^2 of log psd is already 0.77-0.97 without it).

### how the sources are weighted, and why it is not by size

the product form weights every factor at one.  that is correct when each factor
is a correct likelihood over independent data, and it is badly wrong here, in a
direction that would decide the experiment by itself.  the gamma likelihood of an
averaged periodogram treats a cohort as one spectrum observed R x M times, so a
source with 10^5 windows in a 24 Hz band reports a curvature 10^3 times that of a
source with 10^2 windows spanning 70 Hz -- not because it knows more about
cortex, but because it counted windows.  the limiting uncertainty on a POPULATION
parameter is between-subject variance, and that is set by the number of
independent heads.

so every factor is CALIBRATED to the dispersion it actually shows, and then
weighted at one:

    w_f  =  1 / (phi_f  x  recordings-per-subject  x  instruments)

`phi_f` is measured, not chosen.  each recording's own overall scale is removed,
the variance of log psd ACROSS subjects is taken per bin, and it is divided by
the `trigamma(a)` the estimator claims for that bin.  a whole night gives 420
windows and therefore claims a log-psd sd of 0.05; two people's wake spectra do
not agree to 5%, and the ratio is the factor by which that likelihood is
overconfident.  dividing by it turns a demonstrably wrong noise model into a
correctly specified factor, after which the product form's own weight of one is
the right weight.  this is exactly the mechanism the problem calls for: a source
with 10^5 windows in a narrow band has a tiny `trigamma(a)`, a huge phi, and is
deflated hard; a source with 10^2 windows has phi near one and is barely touched.
size stops buying influence and dispersion starts paying for it.

the second and third divisors are the double-count.  sleep-edfx holds two nights
of one person and ds000117 puts two instruments on one head; the product form
would count both as two independent observations of cortex, and a person is one
observation of cortex however many times you record them.

the naive w = 1 product is run as well and reported beside it.  it is not a straw
man: it is what the architecture literally says, and the difference between the
two is the size of the mistake that would otherwise be invisible.

### what is deliberately not here

no inverse solution and no lead field, so `mne-sample`'s FreeSurfer reconstruction
is not used and neither is any `em_generation` parameter.  the reason is not
effort.  in the quasi-static approximation every one of those parameters -- four
tissue conductivities, a lead-field gain, a permittivity whose corner sits at
59 kHz -- multiplies the spectrum by a frequency-flat constant, which is exactly
degenerate with the per-recording scale that has to be profiled anyway.  fitting
them against a spectrum would produce a posterior that is the prior wearing a
fit's provenance.  `scripts/fit_meg_instrument.py` is where the instrument is
tested, against the instrument.

the split is BY SUBJECT in every dataset, never by window and never by night or
run.  windows from one person share a skull, a montage and an individual alpha
peak; nights from one person share all three.

run:  ./.venv/bin/python scripts/forge_joint.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ibm
from ibm.forge.fit import Method, Task, fit, laplace_sd
from ibm.forge.priors import ParameterSpace, assemble
from ibm.forge.spectra import SpectralEvidence, evidence_from_signal, local_root
from ibm.processes.neural import alpha_resonance_transfer, ei_loop_transfer
from ibm.processes.vascular import nvc_transfer, windkessel_transfer
from ibm.registry import REGISTRY
from ibm.vocabulary import Band, Tying

# ---------------------------------------------------------------------------
# what is fitted: registered implementations, collapsed to one global value each
# ---------------------------------------------------------------------------

#: `implementation key -> the parameters of it this script moves`.  three kinds of
#: parameter are dropped and each drop is a statement.  `ei_loop_lti.gain` and
#: `balloon_lti.efficacy` are pure multipliers, exactly degenerate with the
#: per-recording scale the gamma likelihood profiles in closed form, so carrying
#: them would put a parameter in the optimizer whose posterior is its prior times
#: an amplifier.  `windkessel_lti.grubb_exponent` and the rest of
#: `mass_action_budget` are static relations no spectrum and no baseline here
#: identifies.  what is NOT dropped is `alpha_resonator.gain`: in the parallel
#: composition it is the thalamocortical path's power RELATIVE to the local loop's,
#: which is a shape and is identifiable.
FITTED: dict[str, tuple[str, ...]] = {
    "local_excitation:ei_loop_lti": (
        "tau_membrane_s", "tau_ampa_s", "tau_inh_membrane_s", "tau_gaba_a_s",
        "synaptic_lag_s", "loop_gain"),
    "thalamocortical_coupling:alpha_resonator": ("f0_hz", "q", "gain"),
    "neurovascular_coupling:balloon_lti": ("tau_signal_s", "tau_feedback_s", "onset_lag_s"),
    "vascular_flow:windkessel_lti": ("tau_transit_s", "tau_compliance_s",
                                     "autoregulation_gain"),
    "vascular_flow:poiseuille_network": ("baseline_cbf", "baseline_cbv"),
    "tissue_exchange:fick_oxygen_limitation": ("oxygen_extraction_fraction",
                                               "arterial_oxygen_content"),
    "tissue_exchange:linearized_exchange_lti": ("coupling_ratio_n",),
    "metabolism:mass_action_budget": ("cmro2_baseline",),
}

EI = "local_excitation:ei_loop_lti"
AR = "thalamocortical_coupling:alpha_resonator"
NVC = "neurovascular_coupling:balloon_lti"
WK = "vascular_flow:windkessel_lti"

#: the electrophysiological blocks -- the ones every source that sees a current
#: dipole is claimed to bear on.  named once, because "which parameters are
#: shared" is the question and it must not be answered twice in two places.
ELECTRO = tuple(f"{k}.{n}" for k in (EI, AR) for n in FITTED[k])
VASCULAR = tuple(f"{k}.{n}" for k in FITTED if k not in (EI, AR) for n in FITTED[k])

#: mL O2 at STP per micromole.  ds004873 reports cmro2 in umol/100g/min and
#: `mass_action_budget.cmro2_baseline` is declared in mL O2/100g/min.
ML_O2_PER_UMOL = 22.414e-3

#: the deposit's own README asks for these limits on every quantitative map
#: before analysis, "to avoid areas mainly influenced by CSF, large vessels, and
#: susceptibility-related artifact areas".  obeying the depositor's stated
#: exclusion is the difference between measuring grey matter and measuring a vein.
QLIMITS = {"cbf": (1.0, 200.0), "cbv": (0.2, 10.0), "oef": (0.05, 0.9),
           "cmro2": (1.0, 1e4)}
T2_GM = (50.0, 90.0)

#: parieto-occipital channels, as `fit_neural_spectra.py` picks them: alpha is a
#: posterior rhythm and the eyes-closed run is the eegmmidb card's alpha fixture.
POSTERIOR_CH = ("p3", "p1", "pz", "p2", "p4", "po3", "poz", "po4", "o1", "oz", "o2")

BOLD_TR_S = 1.2


def parameter_space() -> ParameterSpace:
    """one `ParameterSpace` over every process the four sources are claimed to share.

    assembled from `REGISTRY` through the ordinary `assemble` door, so this tests
    the declaration and cannot drift from it: a change to a declared prior changes
    what is fitted here.  every tying is replaced by GLOBAL and that replacement is
    a NARROWING of the declaration, not a correction of it.  `ei_loop_lti` declares
    PER_PARTITION -- one loop per cortical layer and area -- and a scalp average, a
    bipolar derivation and a grey-matter median identify exactly one of those.
    expanding the declared tying here would report a laminar map whose every entry
    is the prior, which is the failure `priors.py` names in its own docstring.
    """
    impls = []
    for key, names in FITTED.items():
        proc, name = key.split(":")
        src = REGISTRY.implementations[key]
        impls.append(replace(src, params={n: src.params[n] for n in names},
                             tying=Tying.GLOBAL))
    return assemble(implementations=impls)


def unpack(space: ParameterSpace, theta: np.ndarray) -> dict[str, dict[str, float]]:
    return {k: {n: float(v[0]) for n, v in d.items()}
            for k, d in space.unpack(theta).items()}


# ---------------------------------------------------------------------------
# the forward model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Axis:
    """the only attribute the registered transfer functions read off a basis.

    they take `basis` and use `basis.omega`, so handing them an object with just
    that is not a hack -- it is the actual interface -- and it means the model is
    evaluated on the band's own bins rather than on every bin up to nyquist.
    """

    omega: np.ndarray

    @classmethod
    def of(cls, freqs_hz: np.ndarray) -> "Axis":
        return cls(2.0 * np.pi * np.asarray(freqs_hz, float))


def _positive(d: dict[str, float], *names: str) -> bool:
    return all(d[n] > 0.0 for n in names)


def local_shape(ax: Axis, th: dict[str, dict[str, float]]) -> np.ndarray | None:
    """|H_ei|^2 + A |H_res|^2: the cortical output's spectrum, at unit drive.

    the sum rather than the product is argued in the module docstring and it is
    the only structural decision in this script that is not read straight off the
    registry.  it is also the one the data settle: cascaded, the two forms produce
    a rolloff four decades steeper than any recording here has.
    """
    ei, ar = th[EI], th[AR]
    if not _positive(ei, "tau_membrane_s", "tau_ampa_s", "tau_inh_membrane_s",
                     "tau_gaba_a_s", "synaptic_lag_s"):
        return None
    if not (ar["f0_hz"] > 0.5 and ar["q"] > 0.05 and ar["gain"] > 0.0):
        return None
    h = ei_loop_transfer(ax, tau_membrane_s=ei["tau_membrane_s"],
                         tau_ampa_s=ei["tau_ampa_s"],
                         tau_inh_membrane_s=ei["tau_inh_membrane_s"],
                         tau_gaba_a_s=ei["tau_gaba_a_s"],
                         loop_gain=ei["loop_gain"],
                         synaptic_lag_s=ei["synaptic_lag_s"], gain=1.0)
    hr = alpha_resonance_transfer(ax, f0_hz=ar["f0_hz"], q=ar["q"], gain=1.0)
    out = np.abs(h) ** 2 + ar["gain"] * np.abs(hr) ** 2
    return out if np.all(np.isfinite(out)) and np.all(out > 0) else None


def bold_shape(ax: Axis, th: dict[str, dict[str, float]]) -> np.ndarray | None:
    """the same cortical output, through neurovascular coupling and a windkessel.

    nothing is truncated and nothing is special-cased: the BOLD likelihood is a
    function of all nine `local_excitation` / `thalamocortical_coupling` blocks
    exactly as the process graph says it is.  whether it carries information about
    them at 0.01-0.35 Hz is then measured in (b) rather than assumed either way.
    """
    loc = local_shape(ax, th)
    if loc is None:
        return None
    nv, wk = th[NVC], th[WK]
    if not _positive(nv, "tau_signal_s", "tau_feedback_s", "onset_lag_s"):
        return None
    if not _positive(wk, "tau_transit_s", "tau_compliance_s"):
        return None
    hn = nvc_transfer(ax, efficacy=1.0, tau_signal_s=nv["tau_signal_s"],
                      tau_feedback_s=nv["tau_feedback_s"], onset_lag_s=nv["onset_lag_s"])
    hw = windkessel_transfer(ax, tau_transit_s=wk["tau_transit_s"],
                             tau_compliance_s=wk["tau_compliance_s"],
                             autoregulation_gain=wk["autoregulation_gain"])
    out = loc * np.abs(hn * hw) ** 2
    return out if np.all(np.isfinite(out)) and np.all(out > 0) else None


# ---------------------------------------------------------------------------
# a set of spectra, and the likelihood it contributes
# ---------------------------------------------------------------------------


@dataclass
class Cohort:
    """the psds of one split of one source, stacked so the likelihood is two matmuls.

    the nuisance profile is the reason this is a class and not a function.  for a
    model shape `s(f)`, a drive exponent `e` and a floor fraction `n`, the gamma
    likelihood's optimal scale is available in closed form, and substituting it
    back leaves a log-likelihood that is a matmul of the data against `log(model)`
    plus a matmul against `1/model`.  the whole (recording x nuisance-grid) table
    therefore costs two `(R,F) x (F,J)` products, which is what makes profiling per
    RECORDING affordable -- and per recording is the conservative choice, because
    letting every recording have its own drive slope removes the cheapest way for
    a fit to look good without having learned anything about the process graph.
    """

    name: str
    source: str
    subjects: tuple[str, ...]
    freqs: np.ndarray                    # (F,)
    psd: np.ndarray                      # (R, F)
    a: np.ndarray                        # (R, F), half the degrees of freedom
    e_grid: np.ndarray
    floor_grid: np.ndarray | None
    kind: str = "electro"                # electro | bold
    const: np.ndarray = field(init=False)
    fpow: np.ndarray = field(init=False)
    axis: Axis = field(init=False)

    def __post_init__(self) -> None:
        a, s = self.a, np.maximum(self.psd, 1e-300)
        from scipy.special import gammaln
        self.const = (a * np.log(a) - gammaln(a) + (a - 1.0) * np.log(s)).sum(-1)
        f = np.maximum(self.freqs, 1e-9)
        self.fpow = f[None, :] ** (-self.e_grid[:, None])
        self.axis = Axis.of(self.freqs)

    @property
    def n(self) -> int:
        return self.psd.shape[0]

    @property
    def n_bins(self) -> int:
        return self.psd.shape[1]

    @property
    def k_eff(self) -> float:
        """sum of `a` over recordings and bins: what the likelihood THINKS it knows.

        `a = dof/2` is exactly the fisher information an averaged periodogram bin
        carries about a log scale, so summing it is what the unweighted product
        would count this source as being worth.  it is reported next to the
        overdispersion because the two together are the whole weighting argument.
        """
        return float(self.a.sum())

    @property
    def overdispersion(self) -> float:
        """how much noisier these spectra are across SUBJECTS than the estimator claims.

        this is the number that decides the weighting, and it is measured rather
        than chosen.  the gamma likelihood treats a cohort as one spectrum observed
        R x M times, so it believes its own uncertainty is `trigamma(a)` per bin
        with `a` growing linearly in the window count.  a whole night gives 420
        windows and a claimed log-psd sd of 0.05; two people's wake spectra do not
        agree to 5%.  the ratio of what is observed to what is claimed is the
        factor by which that source's likelihood is overconfident, and dividing by
        it is a CALIBRATION of a demonstrably wrong noise model, not a taste
        judgement about which dataset should matter.

        it is exactly the mechanism the brief asks for: a source with 10^5 windows
        in a narrow band has a tiny `trigamma(a)`, so its overdispersion is large
        and it is deflated hard; a source with 10^2 windows has an overdispersion
        near one and is barely touched.  size stops buying influence and dispersion
        starts paying for it.

        computed model-free -- each recording's own overall scale is removed as its
        mean log psd across the band, and nothing else -- so it does not depend on
        the fit and cannot be tuned by it.  floored at one, because a likelihood is
        never made MORE confident here than its own estimator says.
        """
        from scipy.special import polygamma
        if self.n < 3:
            return 1.0
        y = np.log(np.maximum(self.psd, 1e-300))
        z = y - y.mean(-1, keepdims=True)
        v = z.var(0, ddof=1)
        expect = polygamma(1, np.maximum(self.a.mean(0), 1e-12))
        return float(max(np.median(v / np.maximum(expect, 1e-30)), 1.0))

    def models(self, shape: np.ndarray) -> np.ndarray:
        m = self.fpow * shape[None, :]                       # (E, F)
        if self.floor_grid is None:
            return m
        peak = m.max(-1, keepdims=True)
        return (m[:, None, :] + self.floor_grid[None, :, None] * peak[:, None, :]
                ).reshape(-1, m.shape[1])

    def per_recording(self, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """log p(psd_hat | shape) per recording, at each recording's own best nuisance.

        returns the log-likelihood, the chosen nuisance index and the profiled
        scale, because the fitted drive exponent has to be printable: a source
        whose exponent sits on the edge of the grid is a source whose model shape
        is wrong in a way no parameter of the model can absorb, and that must be
        visible rather than folded into a likelihood.
        """
        m = np.maximum(self.models(shape), 1e-300)           # (J, F)
        a, s = self.a, np.maximum(self.psd, 1e-300)
        tot = a.sum(-1)[:, None]                             # (R, 1)
        q = (a * s) @ (1.0 / m).T                            # (R, J)
        c = q / tot
        ll = self.const[:, None] - (a @ np.log(m).T) - tot * np.log(np.maximum(c, 1e-300)) - tot
        j = np.argmax(ll, axis=1)
        return ll[np.arange(self.n), j], j, c[np.arange(self.n), j]

    def loglik(self, shape: np.ndarray) -> float:
        return float(self.per_recording(shape)[0].sum())

    def r2(self, shape: np.ndarray) -> np.ndarray:
        m = np.maximum(self.models(shape), 1e-300)
        _, j, c = self.per_recording(shape)
        y = np.log10(np.maximum(self.psd, 1e-300))
        f = np.log10(c[:, None] * m[j])
        ss = ((y - y.mean(-1, keepdims=True)) ** 2).sum(-1)
        return 1.0 - ((y - f) ** 2).sum(-1) / np.maximum(ss, 1e-300)

    def exponents(self, shape: np.ndarray) -> np.ndarray:
        _, j, _ = self.per_recording(shape)
        if self.floor_grid is None:
            return self.e_grid[j]
        return self.e_grid[j // len(self.floor_grid)]


@dataclass
class Scalars:
    """one measured scalar per subject, against the prior that asserts it.

    the width is the TRAIN cohort's own between-subject sd, computed once and then
    held fixed for the held-out subjects.  it is not a free parameter and it is not
    the declared prior's width: a literature review's spread and a philips
    scanner's cohort spread are different objects, and using the second is what
    makes the likelihood a statement about these forty people.
    """

    name: str
    source: str
    subjects: tuple[str, ...]
    y: np.ndarray                        # (R,)
    sd: float
    #: theta-dict -> a scalar prediction, or one prediction per subject.  the
    #: per-subject form is what the fick relation needs: each person's measured
    #: flow and extraction predict that person's own metabolic rate, and collapsing
    #: it to a cohort mean would turn a per-subject prediction into a group mean
    #: and silently make the held-out score easier.
    predict: object

    @property
    def n(self) -> int:
        return int(self.y.size)

    @property
    def k_eff(self) -> float:
        return float(self.n)

    @property
    def overdispersion(self) -> float:
        """one, by construction: the width already IS the between-subject spread."""
        return 1.0

    def per_recording(self, mu) -> np.ndarray:
        m = np.broadcast_to(np.asarray(mu, float), self.y.shape)
        return -0.5 * (((self.y - m) / self.sd) ** 2 + math.log(2.0 * math.pi)) \
            - math.log(self.sd)

    def loglik(self, mu) -> float:
        return float(self.per_recording(mu).sum())


# ---------------------------------------------------------------------------
# a dataset: its factors, its weights, its task
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """one source, its train/test factors, and the parameter blocks it may move.

    `moves` is the enforcement point for the claim under test.  ds004873 names
    every `local_excitation` block, not because a BOLD experiment is a good way to
    measure a membrane time constant but because the process graph says its
    measurement passes through one.  if the graph is real, allowing that costs
    nothing and gains precision; if it is nominal, the block does not move and the
    sweep in (b) says by how many nats it failed to.
    """

    key: str
    source: str
    moves: tuple[str, ...]
    train: list
    test: list
    n_train_subjects: int
    n_test_subjects: int
    band: str
    note: str = ""
    #: one weight per factor, in the same order as `train`.
    weights: tuple[float, ...] = ()

    def shape_for(self, factor, th) -> np.ndarray | None:
        if isinstance(factor, Scalars):
            return None
        return bold_shape(factor.axis, th) if factor.kind == "bold" \
            else local_shape(factor.axis, th)

    def _one(self, factor, th) -> float:
        if isinstance(factor, Scalars):
            v = factor.predict(th)
            if v is None or not np.all(np.isfinite(v)):
                return -1e12
            return factor.loglik(v)
        sh = self.shape_for(factor, th)
        if sh is None:
            return -1e12
        v = factor.loglik(sh)
        return v if np.isfinite(v) else -1e12

    def logp(self, space: ParameterSpace, theta: np.ndarray, *, split: str = "train",
             weighted: bool = True) -> float:
        th = unpack(space, theta)
        fs = self.train if split == "train" else self.test
        w = self.weights if (weighted and split == "train") else (1.0,) * len(fs)
        return float(sum(wi * self._one(f, th) for wi, f in zip(w, fs)))

    def per_subject(self, space: ParameterSpace, theta: np.ndarray, *,
                    split: str = "test") -> dict[str, float]:
        """held-out log-likelihood summed per subject, UNWEIGHTED.

        the weight is a decision about how much a source should influence the
        posterior; a held-out score is a statement about how well a theta predicts
        bytes that were never seen, and putting the fitting weight into it would
        make the comparison a comparison of weights.  so the score is the raw
        likelihood, and the same raw likelihood scores every theta.
        """
        th = unpack(space, theta)
        fs = self.train if split == "train" else self.test
        out: dict[str, float] = {}
        for f in fs:
            if isinstance(f, Scalars):
                v = f.predict(th)
                per = f.per_recording(v) if v is not None and np.all(np.isfinite(v)) \
                    else np.full(f.n, -1e6)
            else:
                sh = self.shape_for(f, th)
                per = f.per_recording(sh)[0] if sh is not None else np.full(f.n, -1e6)
                per = np.where(np.isfinite(per), per, -1e6)
            for s, v in zip(f.subjects, per):
                out[s] = out.get(s, 0.0) + float(v)
        return out

    def task(self, space: ParameterSpace, *, weighted: bool = True) -> Task:
        return Task(name=f"{self.key}.train",
                    logp=lambda th: self.logp(space, th, weighted=weighted),
                    moves=self.moves, source=self.source, kind="fit",
                    weight=1.0, note=self.note)


def calibrated_weights(ds: Dataset, *, instruments: int = 1) -> tuple[float, ...]:
    """w_f = 1 / (overdispersion x recordings-per-subject x instruments).

    two corrections, both of them arithmetic rather than preference.

    the first is the overdispersion, measured on that factor's own train spectra:
    a likelihood that is demonstrably wrong about its noise by a factor phi is
    divided by phi, after which it is a correctly specified factor and the product
    form's own weight of one is the right weight for it.  this is what stops a
    whole-night recording from outvoting a forty-person quantitative-MRI cohort by
    having more windows in it.

    the second is the double-count.  sleep-edfx holds two nights of one person and
    ds000117 holds two instruments on one head; the product form would treat both
    as two independent observations of cortex and neither is.  dividing by the
    number of recordings per subject makes a person count once.
    """
    out = []
    for f in ds.train:
        rps = max(f.n / max(ds.n_train_subjects, 1), 1.0) if not isinstance(f, Scalars) else 1.0
        out.append(1.0 / (f.overdispersion * rps * max(instruments, 1)))
    return tuple(out)


# ---------------------------------------------------------------------------
# reading: eegmmidb
# ---------------------------------------------------------------------------


def _chan_index(names: list[str], wanted: tuple[str, ...]) -> list[int]:
    norm = [x.strip().strip(".").lower() for x in names]
    return [i for i, x in enumerate(norm) if x in wanted]


def read_eegmmidb(n_subjects: int, window_s: float, verbose: bool = True):
    """resting eyes-closed runs (R02) of the first `n_subjects` usable participants.

    one run per subject, deliberately.  a subject contributing both R01 and R02
    would enter the cohort likelihood twice at full weight while being one head,
    which is the exact accounting error the weighting section exists to prevent.
    eyes-closed is the run kept because it is the one with a thalamocortical
    resonance in it, and the resonator is a third of what the scalp sources are
    being asked to identify.

    four participants were recorded at a different sampling rate and are skipped
    rather than resampled: resampling changes the anti-alias rolloff, and this fit
    reaches 75 Hz where that rolloff lives.
    """
    import mne
    mne.set_log_level("ERROR")
    root = local_root("eegmmidb") / "1.0.0"
    out: dict[str, SpectralEvidence] = {}
    skipped: list[str] = []
    for p in sorted(root.glob("S[0-9][0-9][0-9]")):
        if len(out) >= n_subjects:
            break
        f = p / f"{p.name}R02.edf"
        if not f.is_file():
            continue
        raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        if abs(fs - 160.0) > 1e-6:
            skipped.append(f"{p.name} ({fs:g} Hz)")
            continue
        ch = _chan_index(raw.ch_names, POSTERIOR_CH)
        if len(ch) < len(POSTERIOR_CH):
            skipped.append(f"{p.name} (montage)")
            continue
        out[p.name] = evidence_from_signal(
            raw.get_data(picks=ch), fs, window_s=window_s, band=Band(1.0, 75.0),
            channels=tuple(raw.ch_names[i] for i in ch),
            source=f"eegmmidb/{p.name}/R02", note="eyes_closed")
    if verbose and skipped:
        print(f"  skipped {len(skipped)}: {', '.join(skipped[:6])}"
              + (" ..." if len(skipped) > 6 else ""))
    return out, root


# ---------------------------------------------------------------------------
# reading: sleep-edfx
# ---------------------------------------------------------------------------


def read_sleep(n_nights: int, window_s: float, max_epochs: int, band: Band,
               verbose: bool = True):
    """the scored WAKE epochs of the first `n_nights` sleep-cassette nights.

    W and not N2, and the choice decides what this experiment means.  arousal
    state genuinely changes cortical E/I -- that is most of what the sleep
    literature is about -- so pooling stages would force one theta across states
    the model has every right to describe differently, and would manufacture a
    conflict that says nothing about whether modalities share parameters.  scored
    wake in an overnight montage is the state that eegmmidb's resting runs and
    ds000117's task runs are also in, so a disagreement between them is a
    disagreement about the same cortex measured by three labs with three montages,
    which is the disagreement worth finding.

    N2 is read as well and fitted alone, as a POSITIVE CONTROL for the conflict
    detector in (c): if the detector cannot see W against N2 in the same people, a
    null between datasets means nothing.
    """
    from eval_sleep_state import evidence_by_stage, pairs
    folder = local_root("sleep-edfx") / "1.0.0" / "sleep-cassette"
    got: dict[str, dict[str, SpectralEvidence]] = {}
    skipped = 0
    for psg, hyp in pairs(folder)[:n_nights]:
        try:
            ev, _ = evidence_by_stage(psg, hyp, window_s=window_s, band=band,
                                      max_epochs=max_epochs, min_epochs=10)
        except Exception:
            skipped += 1
            continue
        if "W" in ev:
            got[psg.stem[:6]] = ev
        else:
            skipped += 1
    if verbose and skipped:
        print(f"  skipped {skipped} nights with no scorable wake or no hypnogram")
    return got, folder


def sleep_subject(night: str) -> str:
    """`SC4001` -> `SC400`.

    sleep-edfx numbers the person in characters 3-4 and the night in character 5,
    so two nights of one person differ in one digit and a split that ignored it
    would leak a skull across the split.
    """
    return night[:5]


# ---------------------------------------------------------------------------
# reading: ds000117
# ---------------------------------------------------------------------------


def read_ds000117(n_subjects: int, window_s: float, seconds: float, verbose: bool = True):
    """one continuous segment per participant, seen simultaneously by MEG and EEG.

    this is the only source held that puts two physically different instruments on
    one head at one instant, which is why it is here: the two sensor arrays are
    linear instantaneous mixtures of the same source currents, so whatever else
    differs between them the SHAPE of the spectrum they report must be explained by
    one `local_excitation`.  magnetometers rather than gradiometers, because a
    gradiometer's spatial derivative reweights the sources and the comparison is
    supposed to be about the instrument's physics and not about its baseline.

    the band stops at 45 Hz for both.  above that the magnetometer spectrum is
    flat -- the sensor noise floor -- and there is a 50 Hz line, and fitting a
    cortical rolloff to a dewar's noise floor would be fitting the dewar.  so this
    source cannot see gamma at all, which makes its constraint on the E-I loop
    weaker than eegmmidb's and is exactly the kind of band asymmetry the product
    form is supposed to handle.
    """
    import mne
    mne.set_log_level("ERROR")
    root = local_root("ds000117") / "1.1.0"
    out: dict[str, dict[str, SpectralEvidence]] = {}
    for p in sorted(root.glob("sub-*")):
        if len(out) >= n_subjects or not p.is_dir():
            continue
        f = (p / "ses-meg" / "meg" /
             f"{p.name}_ses-meg_task-facerecognition_run-01_meg.fif")
        if not f.is_file():
            continue
        raw = mne.io.read_raw_fif(f, preload=False, allow_maxshield=True, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        picks = {"meg": mne.pick_types(raw.info, meg="mag", eeg=False, exclude="bads"),
                 "eeg": mne.pick_types(raw.info, meg=False, eeg=True, exclude="bads")}
        if len(picks["meg"]) < 50 or len(picks["eeg"]) < 30:
            continue
        start = int(30.0 * fs)
        stop = min(start + int(seconds * fs), raw.n_times)
        if stop - start < int(0.5 * seconds * fs):
            continue
        got = {}
        for inst, pk in picks.items():
            x = raw.get_data(picks=pk, start=start, stop=stop)
            got[inst] = evidence_from_signal(
                x, fs, window_s=window_s, band=Band(2.0, 45.0),
                source=f"ds000117/{p.name}/{inst}", note=inst)
        out[p.name] = got
    return out, root


# ---------------------------------------------------------------------------
# reading: ds004873
# ---------------------------------------------------------------------------


def read_ds004873(n_subjects: int, window_s: float, verbose: bool = True):
    """per-subject grey-matter physiology and the BOLD residual spectrum, in T2 space.

    everything for one subject comes off ONE grid with ONE mask.  the alternative
    -- quantitative maps in MNI under the deposit's group mask, the epi in native
    space under another -- would mean the flow this script relates to the spectrum
    was measured in a different set of voxels from the spectrum, and for a ratio
    that is the difference between a physiological relation and a change of
    denominator.  the mask is the depositor's own advised validity limits on cbf,
    cbv and oef intersected with a T2 window that keeps grey matter and drops CSF
    and large vessels.

    the block design is projected out of the time series before the spectrum is
    taken, along with a quadratic trend.  it is not throwing away the neural part:
    a 30 s block puts a line at 1/60 Hz and its harmonics right through the band,
    and a smooth drive spectrum cannot produce a line, so leaving it in would make
    every haemodynamic time constant a fit to a stimulus schedule.  what remains is
    the ongoing fluctuation, which is what the neurovascular and windkessel forms
    are claims about.
    """
    import nibabel as nib
    root = local_root("ds004873")
    dv = root / "derivatives"
    fs = 1.0 / BOLD_TR_S
    recs: list[dict] = []
    skipped: list[str] = []
    for p in sorted(dv.glob("sub-p*")):
        if len(recs) >= n_subjects or not p.is_dir():
            continue
        s = p.name
        q = p / "qmri"

        def load(name: str):
            # the deposit ships its cmro2 maps as (x, y, z, 1) and everything else
            # as (x, y, z); the trailing singleton is a fact about their writer, not
            # about the physiology, and squeezing only trailing singletons keeps a
            # genuinely 4-d file from being silently flattened.
            f = q / name
            if not f.is_file():
                f = q / (name + ".gz")
            if not f.is_file():
                return None
            a = np.asarray(nib.load(f).dataobj, dtype=np.float32)
            while a.ndim > 3 and a.shape[-1] == 1:
                a = a[..., 0]
            return a

        t2 = load(f"{s}_space-T2_T2map.nii")
        bold_f = p / "func" / f"{s}_task-all_space-T2_desc-preproc_bold.nii.gz"
        if t2 is None or not bold_f.is_file():
            skipped.append(f"{s} (no T2 map or no bold)")
            continue
        maps: dict[tuple[str, str], np.ndarray] = {}
        ok_sub = True
        for cond in ("control", "calc"):
            for v in ("cbf", "cbv", "oef"):
                a = load(f"{s}_task-{cond}_space-T2_{v}.nii")
                if a is None:
                    ok_sub = False
                maps[(cond, v)] = a
            a = load(f"{s}_task-{cond}_space-T2_desc-orig_cmro2.nii")
            if a is not None:
                maps[(cond, "cmro2")] = a
        if not ok_sub or ("control", "cmro2") not in maps:
            skipped.append(f"{s} (a control map is missing)")
            continue

        ok = np.isfinite(t2) & (t2 > T2_GM[0]) & (t2 < T2_GM[1])
        for (cond, v), a in maps.items():
            if a is None or a.shape != ok.shape:
                continue
            lo, hi = QLIMITS[v]
            ok &= np.isfinite(a) & (a > lo) & (a < hi)
        if ok.sum() < 2000:
            skipped.append(f"{s} (mask emptied: {int(ok.sum())} voxels)")
            continue

        rec = {"subject": s, "n_voxels": int(ok.sum())}
        for (cond, v), a in maps.items():
            if a is not None and a.shape == ok.shape:
                rec[f"{v}_{cond}"] = float(np.median(a[ok]))

        # -- the BOLD residual spectrum ---------------------------------
        img = nib.load(bold_f)
        x = np.asarray(img.dataobj, dtype=np.float32)
        if x.shape[:3] != ok.shape:
            skipped.append(f"{s} (bold grid {x.shape[:3]} != qmri grid {ok.shape})")
            continue
        ts = x[ok].mean(0).astype(float)
        del x
        ts = _regress_design(ts, root / s / "func" / f"{s}_task-all_events.tsv")
        if ts is None:
            skipped.append(f"{s} (no events file)")
            continue
        # every window is kept: with three windows a median-absolute-deviation
        # rejection rule has no median to speak of, and a rule that cannot be
        # estimated must not be applied silently.
        rec["bold"] = evidence_from_signal(
            ts[None, :], fs, window_s=window_s, band=Band(0.013, 0.35),
            max_mad=1e9, source=f"ds004873/{s}/bold", note="task-regressed residual")
        recs.append(rec)
    if verbose and skipped:
        print(f"  skipped {len(skipped)}: {'; '.join(skipped[:6])}"
              + (" ..." if len(skipped) > 6 else ""))
    return recs, root


def _regress_design(ts: np.ndarray, events: Path) -> np.ndarray | None:
    if not events.is_file():
        return None
    rows = [ln.rstrip("\n").split("\t") for ln in events.read_text().splitlines()][1:]
    n = ts.size
    t = np.arange(n) * BOLD_TR_S
    z = (t - t.mean()) / max(t.std(), 1e-9)
    cols = [np.ones(n), z, z ** 2, z ** 3]
    for kind in sorted({r[2] for r in rows if len(r) >= 3}):
        b = np.zeros(n)
        for r in rows:
            if len(r) >= 3 and r[2] == kind:
                on, dur = float(r[0]), float(r[1])
                b[(t >= on) & (t < on + dur)] = 1.0
        cols += [b, np.gradient(b)]
    x = np.stack(cols, 1)
    beta, *_ = np.linalg.lstsq(x, ts, rcond=None)
    return ts - x @ beta


# ---------------------------------------------------------------------------
# statistics and reporting helpers
# ---------------------------------------------------------------------------


def paired(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """paired difference y - x, with the three numbers that have to travel together.

    an effect size with no p, a p with no effect size, and either with no win rate
    are all ways of reporting less than was measured.  the win rate is the one that
    survives a non-gaussian difference distribution intact.
    """
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = y - x
    if d.size < 2:
        return {"mean": float(d.mean()) if d.size else float("nan"), "sd": float("nan"),
                "sem": float("nan"), "t": float("nan"), "p": float("nan"),
                "dz": float("nan"), "win_rate": float("nan"), "n": int(d.size)}
    t, p = stats.ttest_rel(y, x)
    sd = float(d.std(ddof=1))
    return {"mean": float(d.mean()), "sd": sd, "sem": sd / math.sqrt(d.size),
            "t": float(t), "p": float(p),
            "dz": float(d.mean() / sd) if sd else float("nan"),
            "win_rate": float((d > 0).mean()), "n": int(d.size)}


def short(key: str) -> str:
    """`local_excitation:ei_loop_lti.tau_membrane_s` -> `ei_loop_lti.tau_membrane_s`.

    the process name is constant down every table below and the parameter name is
    the part that has to survive a column width; truncating from the left is how a
    table quietly stops saying which parameter it is about.
    """
    return key.split(":", 1)[1] if ":" in key else key


def prior_sd_u(block) -> float:
    """one prior sd, measured in the UNCONSTRAINED coordinate the fit works in.

    a sweep of "plus or minus three prior sd" has to be a sweep in the same
    coordinate the prior's width is quoted in, and for the house prior that
    coordinate is the log.  a lognormal's `scale` is already the sd of `log x`; a
    normal's is the sd of `x` and its transform is the identity, so the same
    number serves; a uniform is pushed through the logit and its width there is
    not a single number, so it gets the logistic's own sd rather than a fabricated
    one.
    """
    p = block.prior
    if p.dist in ("lognormal", "halfnormal", "gamma", "exponential"):
        return abs(float(p.scale))
    if p.dist == "normal":
        return abs(float(p.scale))
    if p.dist in ("uniform", "beta", "dirichlet"):
        return math.pi / math.sqrt(3.0)          # sd of a standard logistic in u
    return 1.0


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def split_by_subject(subjects, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    order = list(subjects)
    rng.shuffle(order)
    k = int(round(frac * len(order)))
    return sorted(order[:k]), sorted(order[k:])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eeg-subjects", type=int, default=60)
    ap.add_argument("--sleep-nights", type=int, default=60)
    ap.add_argument("--meg-subjects", type=int, default=16)
    ap.add_argument("--fmri-subjects", type=int, default=40)
    ap.add_argument("--sleep-epochs", type=int, default=60)
    ap.add_argument("--meg-seconds", type=float, default=240.0)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--polish", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    ibm.load_all(seal=True, strict=True)
    from ibm.materialize.library import MODELS

    payload: dict = {"argv": sys.argv[1:], "seed": args.seed}

    # -- 1. what is claimed to be shared ---------------------------------
    rule("1. the claim: how much of the ontology is shared, and by how many models")
    use: dict[str, int] = {}
    for m in MODELS.values():
        for p in m.request.processes:
            use[p] = use.get(p, 0) + 1
    n_par: dict[str, int] = {}
    for impl in REGISTRY.implementations.values():
        n_par[impl.process] = n_par.get(impl.process, 0) + len(impl.params)
    rows = sorted(use.items(), key=lambda kv: -kv[1])
    print(f"{'process':34s} {'models':>7s} {'impls':>6s} {'declared params':>16s}")
    print("-" * 68)
    shared5 = 0
    for p, c in rows:
        ni = sum(1 for i in REGISTRY.implementations.values() if i.process == p)
        print(f"{p:34s} {c:7d} {ni:6d} {n_par.get(p, 0):16d}")
        if c >= 5:
            shared5 += n_par.get(p, 0)
    print("-" * 68)
    print(f"{len(MODELS)} models over {len(REGISTRY.processes)} processes and "
          f"{len(REGISTRY.implementations)} implementations")
    print(f"{sum(1 for _, c in rows if c >= 5)} processes appear in 5 or more models and "
          f"carry {shared5} declared parameters between them;\n`local_excitation` alone is in "
          f"{use.get('local_excitation', 0)} of {len(MODELS)} models with "
          f"{n_par.get('local_excitation', 0)} parameters across its "
          f"{sum(1 for i in REGISTRY.implementations.values() if i.process == 'local_excitation')}"
          " implementations.  those\nnumbers are the SIZE of the claim; nothing below assumes "
          "any of them is true.")
    payload["ontology_sharing"] = {
        "models": len(MODELS), "processes": len(REGISTRY.processes),
        "implementations": len(REGISTRY.implementations),
        "per_process": {p: {"models": c, "params": n_par.get(p, 0)} for p, c in rows},
        "params_in_processes_used_by_5plus_models": shared5}

    # -- 2. one parameter space ------------------------------------------
    space = parameter_space()
    theta0 = space.median()
    rule("2. one ParameterSpace over the processes the four sources are claimed to share")
    print(space.describe())
    print("\nevery tying above was collapsed from the declaration to GLOBAL.  the declared\n"
          "tyings are PER_PARTITION and PER_SITE; a scalp average, a bipolar derivation, a\n"
          "magnetometer array and a grey-matter median identify one value each, and\n"
          "expanding the declaration here would report a map whose every entry is the prior.")

    # -- 3. data ----------------------------------------------------------
    rule("3. data")
    print("roots come from each card's data/sources/<id>/raw/.location.yaml; nothing here "
          "hardcodes a path.")

    print("\neegmmidb -- 64-ch scalp EEG at 160 Hz, resting eyes-closed (R02)")
    eeg_ev, eeg_root = read_eegmmidb(args.eeg_subjects, 4.0)
    print(f"  {eeg_root}")
    ex = eeg_ev[sorted(eeg_ev)[0]]
    print(f"  {len(eeg_ev)} subjects, one run each.  {ex}")

    sleep_band = Band(1.0, 25.0)
    print("\nsleep-edfx -- 2 EEG derivations at 100 Hz, expert-scored, WAKE epochs")
    sl_ev, sl_root = read_sleep(args.sleep_nights, 4.0, args.sleep_epochs, sleep_band)
    sl_nights = sorted(sl_ev)
    sl_subj = sorted({sleep_subject(n) for n in sl_nights})
    print(f"  {sl_root}")
    print(f"  {len(sl_nights)} nights from {len(sl_subj)} participants.  "
          f"{sl_ev[sl_nights[0]]['W']}")
    print("  band stops at 25 Hz: above it these recordings are flat, which is the "
          "archive's\n  own anti-alias floor and not cortex (checked: 30-50 Hz differs from "
          "20-30 Hz by\n  0.02 decades in W).")

    print("\nds000117 -- 102 magnetometers and 70 EEG channels, simultaneous, same head")
    meg_ev, meg_root = read_ds000117(args.meg_subjects, 4.0, args.meg_seconds)
    print(f"  {meg_root}")
    print(f"  {len(meg_ev)} participants x 2 instruments.  "
          f"{meg_ev[sorted(meg_ev)[0]]['meg']}")

    print("\nds004873 -- quantitative CBF/CBV/OEF/CMRO2 and a task BOLD series")
    fm_recs, fm_root = read_ds004873(args.fmri_subjects, 128 * BOLD_TR_S)
    print(f"  {fm_root}")
    print(f"  {len(fm_recs)} subjects with a complete control map set and a readable epi; "
          f"grey-matter\n  mask {np.median([r['n_voxels'] for r in fm_recs]):.0f} voxels "
          f"(median).  {fm_recs[0]['bold']}")
    n_calc = sum(1 for r in fm_recs if "cmro2_calc" in r)
    print(f"  {n_calc} of them also have a calc CMRO2 map, which is what the coupling ratio "
          "needs.")

    # -- 4. splits --------------------------------------------------------
    rule("4. splits: by subject, always")
    eeg_tr, eeg_te = split_by_subject(sorted(eeg_ev), args.train_frac, args.seed)
    sl_str, sl_ste = split_by_subject(sl_subj, args.train_frac, args.seed + 1)
    sl_tr = [n for n in sl_nights if sleep_subject(n) in sl_str]
    sl_te = [n for n in sl_nights if sleep_subject(n) in sl_ste]
    meg_tr, meg_te = split_by_subject(sorted(meg_ev), args.train_frac, args.seed + 2)
    fm_ids = [r["subject"] for r in fm_recs]
    fm_tr, fm_te = split_by_subject(fm_ids, args.train_frac, args.seed + 3)
    print(f"eegmmidb   train {len(eeg_tr):3d} / test {len(eeg_te):3d} subjects")
    print(f"sleep-edfx train {len(sl_str):3d} / test {len(sl_ste):3d} participants "
          f"({len(sl_tr)} / {len(sl_te)} nights)")
    print(f"ds000117   train {len(meg_tr):3d} / test {len(meg_te):3d} participants "
          "(both instruments follow the participant)")
    print(f"ds004873   train {len(fm_tr):3d} / test {len(fm_te):3d} subjects")
    print("\nno subject contributes windows to both sides of its own split, and sleep-edfx's\n"
          "two nights per person move together.")
    payload["splits"] = {"eegmmidb": {"train": eeg_tr, "test": eeg_te},
                         "sleep-edfx": {"train": sl_str, "test": sl_ste},
                         "ds000117": {"train": meg_tr, "test": meg_te},
                         "ds004873": {"train": fm_tr, "test": fm_te}}

    # -- build the cohorts ------------------------------------------------
    e_grid = np.arange(-2.0, 3.51, 0.1)
    floor_grid = np.array([0.0, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])

    def spectral(name, source, evs, subs, band, kind="electro", floors=None):
        b = evs[0].basis
        idx = b.band_indices(band)
        if source == "eegmmidb":                 # drop the mains band
            f = b.freqs_hz[idx]
            idx = idx[~((f >= 57.0) & (f < 63.0))]
        return Cohort(name, source, tuple(subs), b.freqs_hz[idx],
                      np.stack([e.psd[idx] for e in evs]),
                      np.stack([0.5 * e.dof[idx] for e in evs]),
                      e_grid, floors, kind)

    eeg_band = Band(1.0, 75.0)
    datasets: dict[str, Dataset] = {}

    datasets["eegmmidb"] = Dataset(
        key="eegmmidb", source="eegmmidb", moves=ELECTRO,
        train=[spectral("eegmmidb/train", "eegmmidb", [eeg_ev[s] for s in eeg_tr],
                        eeg_tr, eeg_band)],
        test=[spectral("eegmmidb/test", "eegmmidb", [eeg_ev[s] for s in eeg_te],
                       eeg_te, eeg_band)],
        n_train_subjects=len(eeg_tr), n_test_subjects=len(eeg_te),
        band="1-75 Hz, 57-63 excluded", note="resting eyes-closed, posterior montage")

    datasets["sleep-edfx"] = Dataset(
        key="sleep-edfx", source="sleep-edfx", moves=ELECTRO,
        train=[spectral("sleep/train", "sleep-edfx", [sl_ev[n]["W"] for n in sl_tr],
                        [sleep_subject(n) for n in sl_tr], sleep_band)],
        test=[spectral("sleep/test", "sleep-edfx", [sl_ev[n]["W"] for n in sl_te],
                       [sleep_subject(n) for n in sl_te], sleep_band)],
        n_train_subjects=len(sl_str), n_test_subjects=len(sl_ste),
        band="1-25 Hz", note="scored wake, Fpz-Cz and Pz-Oz")

    meg_band = Band(2.0, 45.0)
    datasets["ds000117"] = Dataset(
        key="ds000117", source="ds000117", moves=ELECTRO,
        train=[spectral(f"ds000117/train/{i}", "ds000117",
                        [meg_ev[s][i] for s in meg_tr], meg_tr, meg_band)
               for i in ("meg", "eeg")],
        test=[spectral(f"ds000117/test/{i}", "ds000117",
                       [meg_ev[s][i] for s in meg_te], meg_te, meg_band)
              for i in ("meg", "eeg")],
        n_train_subjects=len(meg_tr), n_test_subjects=len(meg_te),
        band="2-45 Hz", note="magnetometers and EEG, same head, same clock")

    # ds004873's factors: one spectrum and six per-subject scalars
    by_id = {r["subject"]: r for r in fm_recs}

    def fmri_factors(ids: list[str], sds: dict[str, float] | None):
        evs = [by_id[s]["bold"] for s in ids]
        fs = [spectral("ds004873/bold", "ds004873", evs, ids, Band(0.013, 0.35),
                       kind="bold", floors=floor_grid)]
        obs = {
            "cbf": ([by_id[s]["cbf_control"] for s in ids], ids,
                    lambda th: th["vascular_flow:poiseuille_network"]["baseline_cbf"]),
            "cbv": ([by_id[s]["cbv_control"] for s in ids], ids,
                    lambda th: 100.0 * th["vascular_flow:poiseuille_network"]["baseline_cbv"]),
            "oef": ([by_id[s]["oef_control"] for s in ids], ids,
                    lambda th: th["tissue_exchange:fick_oxygen_limitation"]
                    ["oxygen_extraction_fraction"]),
            "cmro2": ([by_id[s]["cmro2_control"] for s in ids], ids,
                      lambda th: th["metabolism:mass_action_budget"]["cmro2_baseline"]
                      / ML_O2_PER_UMOL),
        }
        # the fick relation, as a constraint on arterial oxygen content: each
        # subject's MEASURED cbf and oef predict that subject's MEASURED cmro2 only
        # if CaO2 is right.  it is not circular in the acquisition sense -- cbf is
        # pCASL and oef comes from r2' and a bolus -- and it is not independent
        # either, because the depositor derived cmro2 through this same relation.
        # that is why it is allowed to constrain one number and not the chain, and
        # why it is kept per subject rather than collapsed to a cohort mean.
        fick_ids = ids
        fick_y = np.array([by_id[s]["cmro2_control"] for s in fick_ids])
        fick_x = np.array([by_id[s]["cbf_control"] * by_id[s]["oef_control"]
                           for s in fick_ids])
        # the coupling ratio n, from calc against control
        n_ids = [s for s in ids if "cmro2_calc" in by_id[s]]
        n_y = np.array([
            ((by_id[s]["cbf_calc"] - by_id[s]["cbf_control"]) / by_id[s]["cbf_control"])
            / ((by_id[s]["cmro2_calc"] - by_id[s]["cmro2_control"])
               / by_id[s]["cmro2_control"])
            for s in n_ids]) if n_ids else np.zeros(0)
        keep = (np.isfinite(n_y) & (n_y > 0) & (n_y < 20)) if n_y.size \
            else np.zeros(0, bool)

        out_sds = {}
        for k, (y, sub, pred) in obs.items():
            y = np.asarray(y, float)
            sd = sds[k] if sds else float(max(y.std(ddof=1), 1e-6))
            out_sds[k] = sd
            fs.append(Scalars(f"ds004873/{k}", "ds004873", tuple(sub), y, sd, pred))
        resid = fick_y - fick_x * 0.2 / ML_O2_PER_UMOL
        sd_f = sds["fick"] if sds else float(max(resid.std(ddof=1), 1e-6))
        out_sds["fick"] = sd_f
        fs.append(Scalars("ds004873/fick", "ds004873", tuple(fick_ids), fick_y, sd_f,
                          lambda th, x=fick_x: x * th[
                              "tissue_exchange:fick_oxygen_limitation"]
                          ["arterial_oxygen_content"] / ML_O2_PER_UMOL))
        if keep.size and keep.sum() >= 3:
            yy = n_y[keep]
            sd_n = sds.get("n") if sds else None
            sd_n = sd_n if sd_n else float(max(yy.std(ddof=1), 1e-6))
            out_sds["n"] = sd_n
            fs.append(Scalars("ds004873/coupling_n", "ds004873",
                              tuple(np.array(n_ids)[keep]), yy, sd_n,
                              lambda th: th["tissue_exchange:linearized_exchange_lti"]
                              ["coupling_ratio_n"]))
        return fs, out_sds

    fm_train_f, fm_sds = fmri_factors(fm_tr, None)
    fm_test_f, _ = fmri_factors(fm_te, fm_sds)
    datasets["ds004873"] = Dataset(
        key="ds004873", source="ds004873", moves=ELECTRO + VASCULAR,
        train=fm_train_f, test=fm_test_f,
        n_train_subjects=len(fm_tr), n_test_subjects=len(fm_te),
        band="0.013-0.35 Hz spectrum + per-subject baselines",
        note="quantitative physiology and the task-regressed BOLD residual")

    for k, d in datasets.items():
        d.weights = calibrated_weights(d, instruments=2 if k == "ds000117" else 1)

    # the N2 positive control for (c): the same nights, a different arousal state
    n2_nights = [n for n in sl_nights if "N2" in sl_ev[n]]
    n2_ds = Dataset(
        key="sleep-edfx/N2", source="sleep-edfx", moves=ELECTRO,
        train=[spectral("sleep/N2", "sleep-edfx", [sl_ev[n]["N2"] for n in n2_nights],
                        [sleep_subject(n) for n in n2_nights], sleep_band)],
        test=[], n_train_subjects=len({sleep_subject(n) for n in n2_nights}),
        n_test_subjects=0, band="1-25 Hz", note="scored N2, the same people")
    n2_ds.weights = calibrated_weights(n2_ds)

    # -- 5. the shared-parameter table ------------------------------------
    rule("5. which parameters are shared by how many datasets")
    keys = [b.key for b in space.blocks]
    who = {k: [d.key for d in datasets.values() if k in d.moves] for k in keys}
    print(f"{'parameter':40s} {'n':>2s}  datasets that name it")
    print("-" * 78)
    for k in keys:
        w = who[k]
        print(f"{short(k):40s} {len(w):2d}  {', '.join(w) if w else '-'}")
    counts: dict[int, int] = {}
    for k in keys:
        counts[len(who[k])] = counts.get(len(who[k]), 0) + 1
    print("-" * 78)
    for n_ds in sorted(counts, reverse=True):
        print(f"  {counts[n_ds]:2d} parameters are named by {n_ds} of the 4 datasets")
    print("\nthe nine electrophysiological blocks are named by all four sources, ds004873\n"
          "included, because the process graph says a BOLD signal is a cortical output\n"
          "passed through a vascular chain.  naming them is what makes (b) a measurement.")
    payload["sharing"] = {k: who[k] for k in keys}

    # -- 6. weighting -----------------------------------------------------
    rule("6. weighting: calibrate each factor to the dispersion it actually shows")
    print(f"{'factor':26s} {'subj':>5s} {'recs':>5s} {'K claimed':>11s} {'phi':>9s} "
          f"{'weight':>10s} {'w*K':>11s}")
    print("-" * 78)
    wtab = {}
    for d in datasets.values():
        inst = 2 if d.key == "ds000117" else 1
        for f, w in zip(d.train, d.weights):
            phi = f.overdispersion
            wtab[f.name] = {"subjects": d.n_train_subjects, "recordings": f.n,
                            "k_claimed": f.k_eff, "phi": phi, "instruments": inst,
                            "weight": w, "k_effective": w * f.k_eff}
            print(f"{f.name:26s} {d.n_train_subjects:5d} {f.n:5d} {f.k_eff:11.4g} "
                  f"{phi:9.3g} {w:10.4g} {w * f.k_eff:11.4g}")
    print("-" * 78)
    naive = {d.key: sum(f.k_eff for f in d.train) for d in datasets.values()}
    used = {d.key: sum(w * f.k_eff for f, w in zip(d.train, d.weights))
            for d in datasets.values()}
    tn, tu = sum(naive.values()), sum(used.values())
    print(f"{'source':14s} {'naive share':>13s} {'calibrated share':>18s}")
    for k in datasets:
        print(f"{k:14s} {naive[k] / tn:12.1%}  {used[k] / tu:17.1%}")
    worst = max(naive, key=naive.get)
    print(f"\nthe uncorrected product would give {worst} {naive[worst] / tn:.0%} of the "
          f"corpus's curvature\non {datasets[worst].n_train_subjects} heads, purely because "
          "its recordings are long.  after calibration it holds\n"
          f"{used[worst] / tu:.0%}.  both weightings are fitted below and both are reported.")
    payload["weights"] = wtab
    payload["naive_share"] = {k: naive[k] / tn for k in datasets}
    payload["calibrated_share"] = {k: used[k] / tu for k in datasets}

    # -- 7. the fits ------------------------------------------------------
    rule("7. forging")

    def do_fit(keys_in: list[str], label: str, *, weighted: bool = True,
               theta0: np.ndarray | None = None):
        """one fit, restarted from its own answer until it stops improving.

        `fit_map` asks L-BFGS-B for `ftol=1e-12` on an objective whose gradient is
        a finite difference of a grid-profiled likelihood, and it routinely reports
        NOT CONVERGED because it cannot reach a tolerance that tight through
        numerical noise.  refitting from the previous answer is the cheap and
        honest way to tell "the optimizer gave up" from "the optimizer is at the
        mode": what is printed is the log-posterior gained by the LAST pass, and a
        gain of a small fraction of a nat means further passes buy nothing.
        """
        tasks = [datasets[k].task(space, weighted=weighted) for k in keys_in]
        th, rep, gain = theta0, None, float("inf")
        for _ in range(args.polish):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                th_new, rep = fit(space, tasks, method=Method.JOINT, theta0=th,
                                  max_iter=args.max_iter)
            gain = rep.log_posterior - rep.log_posterior0
            th = th_new
            if gain < 1e-3:                 # a restart that bought nothing; stop
                break
        print(f"\n--- {label} ---")
        print(rep)
        print(f"  the last of up to {args.polish} restarts-from-its-own-answer gained "
              f"{gain:.3g} nats; a gain\n  well under a nat means the point below is the "
              "mode and not where the optimizer tired")
        return th, rep, tasks

    alone: dict[str, np.ndarray] = {}
    alone_rep: dict[str, object] = {}
    alone_tasks: dict[str, list] = {}
    for k in datasets:
        alone[k], alone_rep[k], alone_tasks[k] = do_fit([k], f"{k} alone")

    theta_joint, rep_joint, tasks_joint = do_fit(list(datasets), "JOINT: all four sources")

    theta_naive, rep_naive, _ = do_fit(list(datasets), "JOINT, naive w = 1 product",
                                       weighted=False)

    # the optimizer's own reproducibility, from independent prior draws.  without
    # it every "the posterior moved by x sd" below is uninterpretable: a
    # numerically flat direction moves under a restart by as much as it moves under
    # deleting a dataset, and the two have to be told apart by measurement rather
    # than by hope.
    rng = np.random.default_rng(args.seed + 100)
    restarts = []
    for i in range(args.restarts):
        th_i, _, _ = do_fit(list(datasets), f"JOINT restart {i + 1} from a prior draw",
                            theta0=space.sample(rng))
        restarts.append(th_i)
    print(f"\n{args.restarts} restarts of the joint fit from independent prior draws, as the "
          "yardstick\nevery posterior shift below is measured against.")

    lodo: dict[str, np.ndarray] = {}
    for k in datasets:
        rest = [x for x in datasets if x != k]
        lodo[k], _, _ = do_fit(rest, f"leave out {k}  (fitted on {', '.join(rest)})")

    theta_n2, rep_n2 = None, None
    for _ in range(args.polish):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            theta_n2, rep_n2 = fit(space, [n2_ds.task(space)], method=Method.JOINT,
                                   theta0=theta_n2, max_iter=args.max_iter)
        if rep_n2.converged or rep_n2.log_posterior - rep_n2.log_posterior0 < 1e-3:
            break
    print("\n--- sleep-edfx N2 alone (the positive control for the conflict detector) ---")
    print(rep_n2)

    rule("7b. where the posterior landed")
    print(f"{'parameter':34s} {'prior':>9s} {'joint':>9s}  "
          + " ".join(f"{k[:9]:>9s}" for k in datasets))
    print("-" * 78)
    for b in space.blocks:
        row = f"{short(b.key):34s} {theta0[b.slice][0]:9.4g} {theta_joint[b.slice][0]:9.4g}  "
        row += " ".join(f"{alone[k][b.slice][0]:9.4g}" for k in datasets)
        print(row)
    print("-" * 78)
    print("the last four columns are what each source says on its own, with every other\n"
          "parameter left at its prior median.  a column that reproduces the prior column\n"
          "exactly is a source that could not see that parameter at all.")

    # -- 8. (a) transfer --------------------------------------------------
    rule("(a) TRANSFER: does the joint theta beat the per-dataset theta on held-out subjects?")
    print("scored per HELD-OUT SUBJECT, unweighted, at exactly the likelihood each source\n"
          "contributes.  three thetas on identical terms: the untouched prior, the source's\n"
          "own fit, and the joint fit.")
    transfer = {}
    for k, d in datasets.items():
        s_prior = d.per_subject(space, theta0)
        s_own = d.per_subject(space, alone[k])
        s_joint = d.per_subject(space, theta_joint)
        subs = sorted(s_prior)
        vp = np.array([s_prior[s] for s in subs])
        vo = np.array([s_own[s] for s in subs])
        vj = np.array([s_joint[s] for s in subs])
        dj_p, do_p, dj_o = paired(vp, vj), paired(vp, vo), paired(vo, vj)
        transfer[k] = {"n_subjects": len(subs), "ll_prior": float(vp.mean()),
                       "ll_own": float(vo.mean()), "ll_joint": float(vj.mean()),
                       "joint_vs_prior": dj_p, "own_vs_prior": do_p, "joint_vs_own": dj_o}
        print(f"\n{k}  ({len(subs)} held-out subjects, {d.band})")
        print(f"  mean held-out log-likelihood   prior {vp.mean():14.2f}   "
              f"own {vo.mean():14.2f}   joint {vj.mean():14.2f}")
        print(f"  own   - prior  {do_p['mean']:+12.2f} +/- {do_p['sem']:.2f} sem, "
              f"dz {do_p['dz']:+.2f}, p {do_p['p']:.3g}, wins {do_p['win_rate']:.0%}")
        print(f"  joint - prior  {dj_p['mean']:+12.2f} +/- {dj_p['sem']:.2f} sem, "
              f"dz {dj_p['dz']:+.2f}, p {dj_p['p']:.3g}, wins {dj_p['win_rate']:.0%}")
        print(f"  joint - own    {dj_o['mean']:+12.2f} +/- {dj_o['sem']:.2f} sem, "
              f"dz {dj_o['dz']:+.2f}, p {dj_o['p']:.3g}, wins {dj_o['win_rate']:.0%}"
              + ("   <-- joint BEATS the source's own fit"
                 if dj_o["mean"] > 0 and dj_o["p"] < 0.05 else
                 "   <-- joint is WORSE than the source's own fit"
                 if dj_o["mean"] < 0 and dj_o["p"] < 0.05 else
                 "   <-- no difference"))
        for f in d.test:
            if isinstance(f, Cohort) and f.kind in ("electro", "bold"):
                sh_j = d.shape_for(f, unpack(space, theta_joint))
                sh_p = d.shape_for(f, unpack(space, theta0))
                if sh_j is not None and sh_p is not None:
                    ex_j = f.exponents(sh_j)
                    print(f"    {f.name}: mean r^2 of log10 psd  prior "
                          f"{f.r2(sh_p).mean():.3f} -> joint {f.r2(sh_j).mean():.3f}; "
                          f"drive exponent {ex_j.mean():+.2f} "
                          f"[{ex_j.min():+.2f}, {ex_j.max():+.2f}]")
    payload["transfer"] = transfer

    # -- 9. (b) cross-band ------------------------------------------------
    rule("(b) CROSS-BAND: does a source move a parameter it can only see indirectly?")
    print("two measurements, not one.  first the posterior with and without each source;\n"
          "second -- the sharper one -- the EVIDENCE in nats each source's own likelihood\n"
          "carries about one parameter, by sweeping it +/- 3 prior sd with everything else\n"
          "held at the joint mode.  a source that cannot see a parameter produces a flat\n"
          "sweep, and flat is a number.")

    probe = [f"{EI}.tau_membrane_s", f"{EI}.tau_gaba_a_s", f"{EI}.loop_gain",
             f"{AR}.f0_hz", f"{NVC}.tau_signal_s", f"{NVC}.onset_lag_s"]
    sweep = {}
    print(f"\n{'parameter':34s} " + "  ".join(f"{k[:10]:>10s}" for k in datasets))
    print("-" * 78)
    u0 = space.to_unconstrained(theta_joint)
    for key in probe:
        b = space[key]
        offs = np.linspace(-3.0, 3.0, 13) * prior_sd_u(b)
        row = {}
        for k, d in datasets.items():
            vals = []
            for o in offs:
                u = u0.copy()
                u[b.slice] = u0[b.slice] + o
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    vals.append(d.logp(space, space.to_natural(u), weighted=True))
            v = np.array(vals)
            v = v[np.isfinite(v) & (v > -1e11)]
            row[k] = float(v.max() - v.min()) if v.size > 1 else float("nan")
        sweep[key] = row
        print(f"{short(key):34s} " + "  ".join(f"{row[k]:10.4g}" for k in datasets))
    print("-" * 78)
    print("nats of weighted log-likelihood swept over +/- 3 prior sd.  a value near zero is\n"
          "a source that carries NO information about that parameter, however loudly the\n"
          "process graph says it passes through it.  `onset_lag_s` is the built-in control:\n"
          "a pure delay has |H| = 1 identically, so its sweep is exactly zero in a power\n"
          "spectrum and any non-zero there would mean the sweep machinery is broken.")
    payload["sweep_nats"] = sweep

    print("\nand the posterior with and without each source, for the electrophysiological\n"
          "blocks, in units of the JOINT posterior's own marginal sd:")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sd_joint, ok_joint = laplace_sd(space, tasks_joint, theta_joint)
    if not ok_joint:
        print("  the joint hessian is NOT positive definite at the joint mode, so `laplace_sd`\n"
              "  floored its eigenvalues: the flat directions come back very wide, which makes\n"
              "  every shift below smaller than it would otherwise look.  the error is in the\n"
              "  conservative direction and the restart column is what calibrates it.")
    u_joint = space.to_unconstrained(theta_joint)
    u_restart = [space.to_unconstrained(t) for t in restarts]
    influence = {}
    print(f"\n{'parameter':30s} " + "  ".join(f"{'-' + k[:8]:>9s}" for k in datasets)
          + f" {'restart':>9s}")
    print("-" * 78)
    for b in space.blocks:
        if b.key not in ELECTRO:
            continue
        s = sd_joint[b.slice][0]
        row = {}
        for k in datasets:
            u_l = space.to_unconstrained(lodo[k])
            row[k] = float((u_joint[b.slice][0] - u_l[b.slice][0]) / s) if s > 0 \
                else float("nan")
        noise = float(np.max(np.abs([u[b.slice][0] - u_joint[b.slice][0]
                                     for u in u_restart])) / s) if (s > 0 and u_restart) \
            else float("nan")
        row["restart_noise"] = noise
        influence[b.key] = row
        print(f"{short(b.key):30s} " + "  ".join(f"{row[k]:+9.3f}" for k in datasets)
              + f" {noise:9.3f}")
    print("-" * 78)
    print("each source's column is (joint - joint-without-that-source) / joint sd: how far\n"
          "the posterior moves when that source is deleted from the product.  the last\n"
          "column is the largest move produced by merely RESTARTING the same joint fit from\n"
          "a prior draw, and it is the floor: a shift no larger than the restart column is\n"
          "the optimizer wandering along a flat direction, not a dataset speaking.")
    payload["leave_one_out_shift_sd"] = influence

    key_tau = f"{EI}.tau_membrane_s"
    b_tau = space[key_tau]
    print(f"\nthe headline question, spelled out.  "
          f"{key_tau}:")
    print(f"  joint posterior                     {theta_joint[b_tau.slice][0] * 1e3:8.3f} ms")
    print(f"  joint WITHOUT ds004873              "
          f"{lodo['ds004873'][b_tau.slice][0] * 1e3:8.3f} ms")
    print(f"  ds004873 alone                      {alone['ds004873'][b_tau.slice][0] * 1e3:8.3f} ms")
    print(f"  prior median                        {theta0[b_tau.slice][0] * 1e3:8.3f} ms")
    print(f"  move caused by deleting ds004873    "
          f"{influence[key_tau]['ds004873']:+8.3f} joint sd")
    print(f"  move caused by a bare restart       "
          f"{influence[key_tau]['restart_noise']:8.3f} joint sd  (the floor)")
    print(f"  nats ds004873 carries about it      {sweep[key_tau]['ds004873']:8.4g}")
    print(f"  nats eegmmidb carries about it      {sweep[key_tau]['eegmmidb']:8.4g}")

    # -- 10. (c) conflict --------------------------------------------------
    rule("(c) CONFLICT: do the sources disagree about a parameter they all claim to measure?")
    print("each source fitted alone, then every pair compared on every block both of them\n"
          "move, in the unconstrained (log) parameterization where a disagreement of 'a\n"
          "factor of two' is one number.  the width is the marginal laplace sd from that\n"
          "source's own full hessian -- marginal and not mean-field, because a membrane time\n"
          "constant and a loop gain trade off against each other and a mean-field width\n"
          "would call that trade a disagreement.")
    sds: dict[str, np.ndarray] = {}
    hess_ok: dict[str, bool] = {}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in datasets:
            sds[k], hess_ok[k] = laplace_sd(space, alone_tasks[k], alone[k])
        sd_n2, ok_n2 = laplace_sd(space, [n2_ds.task(space)], theta_n2)
    for k in datasets:
        if not hess_ok[k]:
            print(f"  NOTE: {k}'s hessian is not positive definite at its own mode.  its flat\n"
                  "        directions were floored to a wide width, so its share of every z "
                  "below is\n        UNDER-stated rather than over-stated.")

    def conflicts(pairs_in, tag: str):
        rows = []
        for (ka, ua, sa), (kb, ub, sb) in pairs_in:
            for b in space.blocks:
                if b.key not in ELECTRO:
                    continue
                s2 = sa[b.slice][0] ** 2 + sb[b.slice][0] ** 2
                if not np.isfinite(s2) or s2 <= 0:
                    continue
                z = (ua[b.slice][0] - ub[b.slice][0]) / math.sqrt(s2)
                rows.append({"a": ka, "b": kb, "param": b.key, "z": float(z),
                             "theta_a": float(space.to_natural(ua)[b.slice][0]),
                             "theta_b": float(space.to_natural(ub)[b.slice][0]),
                             "sd_a": float(sa[b.slice][0]), "sd_b": float(sb[b.slice][0])})
        rows.sort(key=lambda r: -abs(r["z"]))
        print(f"\n{tag}")
        print(f"{'parameter':34s} {'a':13s} {'b':13s} {'theta_a':>9s} {'theta_b':>9s} "
              f"{'z':>7s}")
        print("-" * 78)
        for r in rows:
            flag = " ***" if abs(r["z"]) > 3 else (" *" if abs(r["z"]) > 2 else "")
            print(f"{short(r['param']):34s} {r['a'][:13]:13s} {r['b'][:13]:13s} "
                  f"{r['theta_a']:9.4g} {r['theta_b']:9.4g} {r['z']:+7.2f}{flag}")
        n3 = sum(1 for r in rows if abs(r["z"]) > 3)
        n2 = sum(1 for r in rows if abs(r["z"]) > 2)
        print(f"{n3} of {len(rows)} comparisons exceed 3 sigma, {n2} exceed 2 sigma")
        return rows

    us = {k: space.to_unconstrained(alone[k]) for k in datasets}
    pair_list = []
    ks = list(datasets)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            pair_list.append(((ks[i], us[ks[i]], sds[ks[i]]),
                              (ks[j], us[ks[j]], sds[ks[j]])))
    rows_between = conflicts(pair_list, "BETWEEN SOURCES")

    rows_control = conflicts(
        [(("sleep-edfx/W", us["sleep-edfx"], sds["sleep-edfx"]),
          ("sleep-edfx/N2", space.to_unconstrained(theta_n2), sd_n2))],
        "POSITIVE CONTROL: the same nights, wake against N2.  arousal genuinely changes\n"
        "cortical E/I, so a detector that finds nothing here has no power and every null\n"
        "above is uninterpretable.")
    payload["conflicts"] = {"between_sources": rows_between, "control_w_vs_n2": rows_control,
                            "hessian_pd": {k: bool(v) for k, v in hess_ok.items()}}

    print("\nthe two posteriors in each comparison share one prior, which pulls them together,\n"
          "so every z above is CONSERVATIVE: the likelihood-only disagreement is larger.")

    # -- 11. leave-one-dataset-out ------------------------------------------
    rule("(d) LEAVE-ONE-DATASET-OUT: fit on three, predict the fourth")
    print("this is transfer across MODALITY, not across subjects.  the held-out subjects of\n"
          "the left-out source are scored at a theta fitted from three other sources that\n"
          "never saw that instrument.")
    lodo_res = {}
    for k, d in datasets.items():
        s_prior = d.per_subject(space, theta0)
        s_own = d.per_subject(space, alone[k])
        s_lodo = d.per_subject(space, lodo[k])
        subs = sorted(s_prior)
        vp = np.array([s_prior[s] for s in subs])
        vo = np.array([s_own[s] for s in subs])
        vl = np.array([s_lodo[s] for s in subs])
        a1, a2 = paired(vp, vl), paired(vo, vl)
        lodo_res[k] = {"ll_prior": float(vp.mean()), "ll_own": float(vo.mean()),
                       "ll_lodo": float(vl.mean()), "lodo_vs_prior": a1,
                       "lodo_vs_own": a2}
        own_gap = float(vo.mean() - vp.mean())
        lodo_res[k]["own_gap"] = own_gap
        print(f"\n{k}  (held out entirely; {len(subs)} test subjects)")
        print(f"  prior {vp.mean():14.2f}   fitted-on-the-other-three {vl.mean():14.2f}   "
              f"own fit {vo.mean():14.2f}")
        print(f"  lodo - prior {a1['mean']:+12.2f} +/- {a1['sem']:.2f}, p {a1['p']:.3g}, "
              f"wins {a1['win_rate']:.0%}")
        print(f"  lodo - own   {a2['mean']:+12.2f} +/- {a2['sem']:.2f}, p {a2['p']:.3g}, "
              f"wins {a2['win_rate']:.0%}")
        if abs(own_gap) < 1e-6:
            print("  its own data closes no gap either, so there is nothing here to "
                  "recover and the\n  fraction is undefined rather than large")
        elif a1["mean"] > 0:
            print(f"  recovered {a1['mean'] / own_gap:.0%} of the gap its own data closes")
        else:
            print("  the other three sources make this source's predictions WORSE than "
                  "the prior")
    payload["leave_one_dataset_out"] = lodo_res

    # -- 12. the naive comparison ------------------------------------------
    rule("weighting sensitivity: the naive w = 1 product")
    print(f"{'parameter':40s} {'calibrated':>12s} {'naive w=1':>12s} {'prior':>12s}")
    for b in space.blocks:
        if b.key in ELECTRO:
            print(f"{short(b.key):40s} {theta_joint[b.slice][0]:12.4g} "
                  f"{theta_naive[b.slice][0]:12.4g} {theta0[b.slice][0]:12.4g}")
    naive_tr = {}
    for k, d in datasets.items():
        s_n = d.per_subject(space, theta_naive)
        s_j = d.per_subject(space, theta_joint)
        subs = sorted(s_n)
        st = paired(np.array([s_n[s] for s in subs]), np.array([s_j[s] for s in subs]))
        naive_tr[k] = st
        print(f"  held out {k:14s} subject-equivalent - naive = {st['mean']:+12.2f} "
              f"+/- {st['sem']:.2f}, p {st['p']:.3g}")
    payload["naive_vs_weighted"] = naive_tr

    # -- 13. summary --------------------------------------------------------
    rule("summary")
    n_beat = sum(1 for k in datasets if transfer[k]["joint_vs_own"]["mean"] > 0
                 and transfer[k]["joint_vs_own"]["p"] < 0.05)
    n_worse = sum(1 for k in datasets if transfer[k]["joint_vs_own"]["mean"] < 0
                  and transfer[k]["joint_vs_own"]["p"] < 0.05)
    print(f"(a) transfer   joint beats the source's own fit on {n_beat} of "
          f"{len(datasets)} sources, loses on {n_worse}")
    cross = sweep[key_tau]["ds004873"]
    print(f"(b) cross-band ds004873 carries {cross:.4g} nats about "
          f"{key_tau.split('.')[-1]}, against "
          f"{sweep[key_tau]['eegmmidb']:.4g} from eegmmidb;\n"
          f"               deleting it moves the joint posterior by "
          f"{influence[key_tau]['ds004873']:+.3f} sd")
    n3 = sum(1 for r in rows_between if abs(r["z"]) > 3)
    c3 = sum(1 for r in rows_control if abs(r["z"]) > 3)
    print(f"(c) conflict   {n3} between-source comparisons exceed 3 sigma "
          f"({len(rows_between)} tested); the W-vs-N2 positive control finds {c3}")
    n_lodo = sum(1 for k in datasets if lodo_res[k]["lodo_vs_prior"]["mean"] > 0
                 and lodo_res[k]["lodo_vs_prior"]["p"] < 0.05)
    print(f"(d) lodo       three sources beat the prior on {n_lodo} of {len(datasets)} "
          "left-out modalities")
    noise = max(abs(influence[k]["restart_noise"]) for k in influence)
    biggest = max(max(abs(influence[k][d]) for d in datasets) for k in influence)
    print(f"    NOTE       restarting the joint fit from a prior draw moves the posterior by "
          f"up to\n               {noise:.0f} sd, against {biggest:.0f} sd for deleting a "
          "whole dataset.  the joint objective is\n               multimodal, so a posterior "
          "shift is not by itself evidence that a source spoke;\n               the swept "
          "nats are.")

    out = args.out or (Path(__file__).resolve().parents[1] / "data" / "joint" /
                       "forge_joint@v1" / "posterior.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["posterior"] = {
        "joint": {b.key: float(theta_joint[b.slice][0]) for b in space.blocks},
        "joint_naive": {b.key: float(theta_naive[b.slice][0]) for b in space.blocks},
        "prior_median": {b.key: float(theta0[b.slice][0]) for b in space.blocks},
        "alone": {k: {b.key: float(alone[k][b.slice][0]) for b in space.blocks}
                  for k in datasets},
        "leave_one_out": {k: {b.key: float(lodo[k][b.slice][0]) for b in space.blocks}
                          for k in datasets},
        "sleep_n2": {b.key: float(theta_n2[b.slice][0]) for b in space.blocks},
        "joint_sd_unconstrained": {b.key: float(sd_joint[b.slice][0]) for b in space.blocks},
    }
    payload["runtime_s"] = time.time() - t0
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwritten to {out}")
    print(f"{time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
