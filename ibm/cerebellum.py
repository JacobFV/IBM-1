"""the cerebellum: one microzone, its granule expansion, and the loop out and back.

WHY THIS EXISTS
---------------
`ibm/rhythms.py` declares three cerebellar loops -- `cb_local` (Purkinje and the
molecular-layer interneurons), `olivo` (olivo-cerebellar timing) and `ctc`
(cerebello-thalamo-cortical) -- and four rhythms filed under them: `purkinje_fast`
(160-250 Hz), `olivary_clock` (5-10 Hz), `cerebello_cortical` (13-30 Hz coherence) and
`physiological_tremor` (8-12 Hz).  Every one of them is marked `needs-cerebellum`.  There
was no cerebellum, so none of them could be scored, and the programme's stated end -- a
real anatomical human crawling in a 3-D world -- runs through a structure whose entire job
is to time and correct movement.

This module is that structure at the resolution the rest of the programme works at: rates,
not spikes; one microzone, not a hemisphere.

THE MODEL, and why each piece is here
-------------------------------------
Seven populations, each bounded in [0, 1], plus the synaptic species that set the timing:

    M    mossy-fibre terminals           the input, a relay and not a computation
    G    granule cells                   the EXPANSION.  n_granule >> n_mossy
    Go   Golgi cells                     feedback inhibition that keeps G sparse
    I    molecular-layer interneurons    basket and stellate cells
    P    Purkinje cells                  TONICALLY ACTIVE at rest, and inhibitory
    N    deep nucleus                    disinhibited by a Purkinje pause
    O    inferior olive                  electrically coupled; supplies climbing fibres

Three things in here are mechanisms rather than decoration, and each has a gate.

**1. The granule expansion is a decorrelator.**  Each granule cell takes `n_dendrite = 4`
mossy fibres -- the real number, and one of the most conserved facts in the brain -- from a
SPARSE, FIXED, randomly drawn projection, and fires only when enough of them are active
together (`theta_G` sits near 3 of 4).  Expansion alone does not decorrelate: a random
linear mixture preserves correlation almost exactly.  It is expansion **plus** the
coincidence threshold **plus** the Golgi feedback that holds the code sparse, which is
Marr-Albus, and gate C2 measures whether it actually happens rather than asserting it.

The projection is drawn ONCE from a `torch.Generator` the caller must supply.  This is
CLAUDE.md's standing trap and it has bitten this repository twice in one file in one day:
`CorticalDynamics.__init__` drew its long-range partners from the global RNG, so a `--seed`
flag that claimed to vary the initialisation silently redrew the topology, and cpu and cuda
drew unrelated graphs from the same seed.  `__init__` therefore ASSERTS it was handed a
generator rather than making one; a callee that can draw its own will eventually draw its
own.

**2. The climbing fibre is a teaching signal.**  `W_pf`, the parallel-fibre -> Purkinje
weight matrix, is a registered BUFFER and not a parameter -- gradient descent does not touch
it, experience does.  This follows `ibm/substrate.py`'s `P` buffer exactly, for the same
reason given there: like every other thing that weights an edge it has to be saved with the
model, and like every other thing written by experience it must be outside the optimiser.

The rule is the classical one.  A parallel fibre active shortly before a climbing-fibre
event is DEPRESSED (LTD); a parallel fibre active without one is potentiated (LTP), which
is a real synapse's own restoring force and not a numerical convenience.  Eligibility is a
`tau_elig = 100 ms` trace of the parallel-fibre input, which is the measured PF-before-CF
window.  The two rates fix an equilibrium climbing-fibre rate `a_ltp / (a_ltp + a_ltd)`:
learning stops when the error rate reaches the set point, rather than running to a rail.

**The sign of the loop, stated plainly, because it is easy to get backwards and the gate
depends on it.**  Purkinje cells INHIBIT the nucleus, so LTD at PF -> Purkinje lowers the
Purkinje rate and RAISES the nucleus output.  A climbing fibre that fired when the nucleus
OVERSHOT its target would therefore drive the output further up -- positive feedback, and
the error would grow.  The teaching signal here fires on the UNDERSHOOT, and the
potentiation that happens in its absence supplies the other direction.  `gate_cerebellum.py`
C3 runs the overshoot convention as a third arm and reports what it does, because a sign
that is asserted is worth less than a sign that was measured.

**3. The loop has a conduction budget.**  Every delay below is read off `ibm/rhythms.py`:
the midpoint of the declared range for the corresponding edge, with the range in the
comment.  They are implemented as history ring buffers, so what arrives at a station is
what the upstream station emitted some milliseconds ago, and `delays_in_steps(dt)` reports
the REQUESTED and the REALISED lag separately -- a dt too coarse to resolve a delay is
clamped to one step and says so, rather than silently collapsing it to an instantaneous
handshake.

WHAT MAKES THE FAST RHYTHM, and what does not
---------------------------------------------
`cb_local` declares two 0.5-1.5 ms edges: parallel fibre onto interneuron, and basket cell
onto the Purkinje soma.  Those two by themselves are a feed-FORWARD path from the granule
layer, not a loop, so they cannot ring on their own.  What closes the loop is the molecular
layer's own recurrent inhibition -- basket and stellate cells inhibit each other on the same
0.5-1.5 ms scale -- which is the standard ING generator, with the Purkinje axon collateral
back onto the interneurons as the second return.  At 1 ms of delay and ~1 ms membrane and
synaptic constants the phase budget closes near 200 Hz, and the Purkinje cells inherit the
rhythm through `w_IP`.  Gate C5 measures where it lands and reports the prominence beside
the frequency; if it is not there it is recorded FAILED and diagnosed, not tuned into place.

WHAT MAKES THE OLIVARY CLOCK
----------------------------
The inferior olive is a relaxation oscillator, which is Llinas & Yarom's measurement: a
low-threshold calcium current that regenerates as the cell rises (`g_ca`), against a
calcium-activated potassium current that follows it slowly and puts it out (`g_k`,
`tau_z`).  `tau_z` is the constant that sets the period, and `io_drive` is what places the
cell on the unstable branch of its own nullcline -- which is what "subthreshold
oscillation" means as a statement about a dynamical system.

`ibm/thalamus.py` warns that a window current able to trigger itself is a PACEMAKER rather
than a rebound, and records what that cost when it was not wanted: a limit cycle at about
8 Hz whatever else was happening.  Here it is wanted, and it is declared on purpose.  A bug
in one structure is a mechanism in another; what is not allowed is being vague about which
one you meant.

**The complex spike is not the oscillation.**  `O` is the membrane state and oscillates
continuously; what travels down the climbing fibre is `sigmoid(O; beta_cs, theta_cs)`, a
sharp threshold near the top of the swing.  At rest the peaks stay under it and the
Purkinje cells see no climbing fibre at all, which is what lets them be tonically active;
an error drive lifts the whole oscillation so the peaks cross, and the complex-spike RATE
then carries the error while the clock keeps the TIMING.  Conflating the two would have
made the Purkinje cells pause eight times a second for no reason -- which is what the first
version of this module did, and it is why the resting Purkinje rate came out at 17 Hz.

Gap-junction coupling `w_gap` pulls the heterogeneous cells together, which is what makes
the POPULATION mean oscillate at all -- uncoupled cells with jittered drive average out.
The nucleo-olivary brake `w_NO` (the `olivo` loop's third edge) closes the loop: as the
nucleus output rises the climbing-fibre rate falls.

BOUNDEDNESS
-----------
Every state variable is advanced by an exponential-Euler step -- a convex combination of the
old value and a target that is itself inside [0, 1] -- so every rate and every gating
variable stays in [0, 1] for any dt, any parameters and any input.  Divergence cannot be
represented.  The two exceptions are stated rather than hidden: `eta` is an
Ornstein-Uhlenbeck background and is not a rate, and `W_pf` is a weight clamped to
[0, `w_pf_max`].

UNITS: seconds; rates dimensionless in [0, 1], with `R_MAX = 100 Hz` for readouts (right for
Purkinje simple spikes, generous for granule cells); delays in seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields as dataclass_fields

import torch
import torch.nn as nn

R_MAX = 100.0


def sigmoid(x, beta, theta):
    return torch.sigmoid(beta * (x - theta))


# --------------------------------------------------------------------------------------
# declared priors.  every value carries its reason.  the conduction times are the
# catalogue's (`ibm/rhythms.py`), taken at the midpoint of the declared range.
# --------------------------------------------------------------------------------------
@dataclass
class CerebellarPriors:
    # ---------------------------------------------------------------- mossy fibres
    # a relay, not a computation: the terminal low-passes whatever the pons delivers.
    # 3 ms is a synaptic terminal, fast enough not to be the thing that sets any band.
    tau_M: float = 0.003

    # ---------------------------------------------------------------- granule layer
    tau_G: float = 0.006           # granule cells are small and fast
    w_MG: float = 1.0              # mossy -> granule, per-dendrite gain
    # THE COINCIDENCE THRESHOLD.  a granule cell takes 4 mossy fibres and needs most of
    # them together; `theta_G` is expressed on the MEAN of its 4 inputs, so 0.62 is
    # "about two and a half of four".  this is the sparsifier, and without it the
    # expansion is a random linear map that preserves correlation exactly (gate C2).
    theta_G: float = 0.28
    # STEEP on purpose, and this is the constant the decorrelation lives in.  a soft
    # threshold returns a graded version of the input and a graded version of a
    # correlated input is still correlated: measured, an input pair at r = 0.898 came back
    # at r = 0.886 with beta_G = 14 and at r = 0.751 with beta_G = 40.  the sparsity that
    # comes with it is the point, not a side effect.
    beta_G: float = 40.0
    # per-granule spread of the threshold, drawn once from the supplied generator.
    # SMALL, and the reason is a measurement: a fixed per-cell bias is the same bias for
    # both patterns, so it is common structure that INFLATES the very correlation C2 is
    # about -- at jitter 0.13 the granule correlation came back 0.791 where the same
    # model at 0.04 gave 0.751.  it is kept non-zero because identical thresholds make
    # four hundred granule cells one granule cell.
    theta_G_jitter: float = 0.05
    w_GoG: float = 0.60            # Golgi -> granule feedback inhibition
    # Golgi cells: driven by the granule layer they inhibit, so the layer's total activity
    # is held near a set point whatever the input's overall level is.  that is what makes
    # the code a code rather than a brightness reading.
    tau_Go: float = 0.012
    beta_Go: float = 8.0
    theta_Go: float = 0.18
    w_GGo: float = 2.2             # granule -> Golgi
    w_MGo: float = 0.5             # mossy -> Golgi, the feed-forward arm

    # ---------------------------------------------------------- molecular layer (MLI)
    # basket and stellate cells.  the time constants here are FAST on purpose: this is
    # the population that has to ring at 200 Hz, and 200 Hz needs ~1 ms of everything.
    # in vivo these are high-conductance fast-spiking cells whose effective membrane
    # constant is a small fraction of the slice value.  C5 measures whether the band
    # comes out; C7 says whether these constants are what put it there.
    tau_I: float = 0.0010
    # `beta_I` and `w_II` together are the LOOP GAIN of the molecular layer's recurrent
    # inhibition, and the loop only rings if that gain survives its own lags: at 200 Hz the
    # membrane and the GABA-A synapse between them attenuate by about 0.35, so a gain of
    # 1.4 -- which is what the first version had -- is a damped loop and measures a
    # population standard deviation of exactly 0.0000.  `i_tonic` is set to put the
    # interneurons near I = 0.35 where the sigmoid's own slope, I(1-I), is largest; the
    # gain is beta_I * w_II * I(1-I) and the working point is half of it.
    beta_I: float = 12.0
    theta_I: float = 0.30
    i_tonic: float = 0.98          # resting drive: the molecular layer is not silent
    i_jitter: float = 0.05
    w_GI: float = 3.00             # parallel fibre -> interneuron  (`cb_local` edge 1).
                                   # large because the parallel-fibre input is the
                                   # granule layer's MEAN, which a sparse code holds at
                                   # a few percent; the gain carries the units
    w_II: float = 2.0              # interneuron -> interneuron: THE ING GENERATOR.  the
                                   # loop `cb_local` declares is feed-forward; this is the
                                   # recurrent inhibition that actually closes it.
    tau_gaba_I: float = 0.0012     # GABA-A in the molecular layer
    w_PI: float = 0.25             # Purkinje axon collateral back onto the interneurons,
                                   # inhibitory -- the second return of `cb_local`

    # ---------------------------------------------------------------- Purkinje cells
    tau_P: float = 0.0015
    beta_P: float = 8.0
    theta_P: float = 0.35
    # TONIC ACTIVITY IS THE RESTING STATE, not an accident of the parameters.  a Purkinje
    # cell fires 40-100 simple spikes a second with no input at all, and everything this
    # module does downstream -- disinhibition of the nucleus, the pause, the sign of the
    # learning rule -- is a modulation of that.  `p_tonic` is set so `resting_state()`
    # comes back near 0.6, i.e. ~60 Hz.
    p_tonic: float = 0.740
    p_jitter: float = 0.04
    w_GP: float = 5.0              # parallel fibre -> Purkinje, through the LEARNED W_pf
    w_IP: float = 0.95             # basket -> Purkinje soma    (`cb_local` edge 2)
    tau_bk: float = 0.0012
    # the complex spike and the pause it leaves behind.  a climbing fibre depolarises the
    # cell enormously for a few milliseconds and then SILENCES it for 10-30 ms.  the pause
    # is the mechanism gate C4 is about: a climbing fibre that only nudged the rate would
    # not disinhibit the nucleus, and a rate change without a pause is not this.
    g_cs: float = 0.55             # the complex spike's own depolarisation
    g_pause: float = 1.2           # the post-complex-spike silence
    tau_pause_up: float = 0.0020
    # measured on a 3 ms complex spike: this pair takes the Purkinje population from
    # 0.757 to 0.055 and holds it under 90% of baseline for 37 ms.  at 0.022 the same
    # probe gave a 57 ms silence, which is longer than a complex-spike pause is.
    tau_pause_dn: float = 0.0120

    # ---------------------------------------------------------------- deep nucleus
    tau_N: float = 0.010
    beta_N: float = 7.0
    theta_N: float = 0.40
    n_drive: float = 1.00          # the nucleus is tonically driven and tonically held
                                   # down; its output is a RELEASE, not an excitation
    w_MN: float = 0.45             # mossy collateral -> nucleus, the feed-forward arm
    w_PN: float = 1.25             # Purkinje -> nucleus, inhibitory (`ctc` edge 3)
    tau_gaba_N: float = 0.018      # slower than the molecular layer's: the nucleus
                                   # integrates the Purkinje barrage rather than tracking it

    # ---------------------------------------------------------------- inferior olive
    # A RELAXATION OSCILLATOR, which is Llinas & Yarom's measurement: a low-threshold
    # calcium conductance (fast, regenerative) against a calcium-activated potassium
    # conductance (slow, opposing).  `O` is the MEMBRANE state, not a firing rate -- the
    # subthreshold oscillation is continuous, and it is what `olivary_clock` names.
    tau_O: float = 0.008
    beta_O: float = 3.0            # SHALLOW on purpose: a steep gain saturates the
    theta_O: float = 0.55          # swing at 0.98 whatever the drive, and then the
                                   # complex-spike threshold below cannot be graded by
                                   # error at all.  measured: at beta_O = 9 the peak of
                                   # the oscillation moved 0.942 -> 0.970 over a drive
                                   # range that moved it 0.697 -> 0.767 at beta_O = 3.
    # the operating point.  this is the one constant here that is placed rather than
    # measured: it puts the cell on the UNSTABLE branch of its own nullcline, which is
    # what a subthreshold oscillator is.  too low and the cell sits quiet at a stable
    # fixed point; too high and it sits saturated at another.  C7 sweeps it.
    io_drive: float = 0.700
    io_jitter: float = 0.05
    g_ca: float = 0.85             # the low-threshold calcium current's strength
    ca_theta: float = 0.50         # its activation, placed mid-range so the current
    ca_beta: float = 13.0          # regenerates only once the cell is already rising
    g_k: float = 1.10              # the calcium-activated potassium current: the slow
    tau_z: float = 0.040           # opposition, and the pair that sets the PERIOD.
                                   # measured on the whole microzone, on the LATE window
                                   # after the start transient: (0.040, 1.10) -> 7.96 Hz
                                   # sustained; (0.050, 0.90) -> 4.38 Hz; and at an
                                   # io_drive of 0.575 the oscillation DECAYS to a
                                   # standard deviation of 0.0000 while its start
                                   # transient still puts a +2.8 decade peak in a spectrum
                                   # taken over the whole run.  that is why C6 measures
                                   # late and reports the sustained amplitude beside the
                                   # prominence
    w_gap: float = 0.55            # gap junctions.  without them the jittered cells
                                   # average away and the POPULATION does not oscillate
    w_NO: float = 0.70             # nucleo-olivary brake (`olivo` edge 3)
    tau_gaba_O: float = 0.040      # the brake is slow and GABAergic
    # THE COMPLEX SPIKE IS NOT THE OSCILLATION.  an olivary cell oscillates continuously
    # and fires about once a second: the spike happens when a peak of the subthreshold
    # oscillation crosses threshold, so error raises the RATE of complex spikes while the
    # clock keeps the TIMING.  separating the two is what lets the Purkinje cells be
    # tonically active at rest (no climbing fibre) while the olivary clock still runs.
    # placed so the RESTING peak of the oscillation sits just under it: at 0.72 the
    # resting complex-spike duty is 1.8% and an olivary drive of 0.08/0.16/0.30 takes it
    # to 6.2/10.8/19.8%.  at 0.66 the resting duty is already 8.7% and the Purkinje
    # cells are paused several times a second for no reason.
    theta_cs: float = 0.72
    beta_cs: float = 60.0

    # ---------------------------------------------------------------- plasticity
    tau_elig: float = 0.100        # PF eligibility: the measured PF-before-CF window
    a_ltd: float = 2.4             # depression on conjunction
    a_ltp: float = 0.10            # potentiation on PF alone.  the ratio fixes the
                                   # equilibrium climbing-fibre rate at
                                   # a_ltp/(a_ltp+a_ltd) ~ 4%: learning stops when the
                                   # error rate reaches a set point, not at a rail
    w_pf_max: float = 2.0
    w_pf_init: float = 1.0

    # ---------------------------------------------------------------- background
    sigma: float = 0.02
    tau_eta: float = 0.004

    # ------------------------------------------- conduction, from `ibm/rhythms.py`
    # midpoints of the catalogue's declared ranges, which are in the comments.
    d_mf_gr_s: float = 0.0055      # `ctc`   pons -> cb_ctx, mossy fibres      3-8 ms
    d_pf_s: float = 0.0010         # `cb_local` parallel fibre onto interneuron 0.5-1.5 ms
    d_bk_s: float = 0.0010         # `cb_local` basket onto the Purkinje soma   0.5-1.5 ms
    d_ii_s: float = 0.0010         # molecular-layer recurrent inhibition, the same scale
    d_coll_s: float = 0.0010       # Purkinje axon collateral, the same scale
    d_pk_n_s: float = 0.0035       # `ctc`/`olivo` cb_ctx -> dentate_n          2-5 ms
    d_n_io_s: float = 0.0085       # `olivo` dentate_n -> io, the brake         5-12 ms
    d_io_pk_s: float = 0.0050      # `olivo` io -> cb_ctx, climbing fibres      3-7 ms
    d_n_thal_s: float = 0.0075     # `ctc`   dentate_n -> vl                    5-10 ms
    d_thal_ctx_s: float = 0.0030   # `ctc`   vl -> m1                           2-4 ms
    d_ctx_pons_s: float = 0.0100   # `ctc`   m1 -> pons                         5-15 ms


# which priors get a per-unit learned residual, and over which population.  the declared
# loop may be BENT by learning, it may not be replaced -- the residual passes through a
# tanh and is capped at `residual_frac`, exactly as `ThalamicField` and `CorticalField` do.
_RESIDUAL_POP = {
    "theta_G": "G", "w_GoG": "G",
    "w_GI": "I", "w_II": "I", "w_PI": "I",
    "w_GP": "P", "w_IP": "P", "g_pause": "P", "p_tonic": "P",
    "w_MN": "N", "w_PN": "N",
    "g_ca": "O", "w_gap": "O", "w_NO": "O",
}


def _read(ring: torch.Tensor, t: int, lag: int) -> torch.Tensor:
    """what this station emitted `lag` steps ago.  a HISTORY ring: written at `t % L`,
    read at `(t - lag) % L`, so one ring serves any number of different lags."""
    return ring[:, (t - lag) % ring.shape[1]]


def _write(ring: torch.Tensor, t: int, v: torch.Tensor) -> torch.Tensor:
    """out of place on purpose.  `step` must be a FUNCTION of its arguments -- called
    twice on the same state it must return the same thing -- and a ring mutated in place
    makes the second call see the first call's write.  CLAUDE.md's row 23, general form."""
    r = ring.clone()
    r[:, t % r.shape[1]] = v
    return r


class CerebellarMicrozone(nn.Module):
    """one microzone: a granule expansion, a Purkinje/interneuron molecular layer, a deep
    nucleus and the olivary cells that teach it.

    A microzone is the cerebellum's functional unit -- a strip of Purkinje cells sharing a
    climbing-fibre source and projecting to the same patch of nucleus -- and it is the
    right object here for the same reason a thalamic unit is nucleus-sized in
    `ibm/thalamus.py`: pretending to individual cells would be a precision nobody can
    address.

    `gen` is REQUIRED.  The mossy -> granule projection and every heterogeneity below are
    drawn from it, once, at construction.  See the module docstring.
    """

    def __init__(self, gen: torch.Generator, n_mossy: int = 24, n_granule: int = 400,
                 n_mli: int = 16, n_purkinje: int = 8, n_nucleus: int = 4,
                 priors: CerebellarPriors | None = None, learn: bool = True,
                 device="cpu"):
        super().__init__()
        assert isinstance(gen, torch.Generator), (
            "CerebellarMicrozone needs an explicit generator; a callee that can draw its "
            "own eventually draws its own (CLAUDE.md, Randomness)")
        pr = self.pr = priors or CerebellarPriors()
        assert n_granule > n_mossy, (
            "the granule layer is an EXPANSION; n_granule must exceed n_mossy "
            f"(got {n_granule} <= {n_mossy}).  the real ratio is ~200:1")
        self.n_mossy, self.n_granule = int(n_mossy), int(n_granule)
        self.n_mli, self.n_purkinje = int(n_mli), int(n_purkinje)
        # ONE CLIMBING FIBRE PER PURKINJE CELL.  that is not a simplification, it is the
        # anatomy, and it is why `n_olive` is not a free size.
        self.n_olive = int(n_purkinje)
        self.n_nucleus = int(n_nucleus)
        self.n_dendrite = 4        # the granule cell's four dendrites, the real number
        self.learn = learn
        self.residual_frac = 0.35

        # -------------------------------------------------- the fixed sparse projection
        # drawn ONCE, from the supplied generator, on the cpu, and registered as a buffer
        # so it is saved with the model.  a projection redrawn per call would make the
        # granule code a different code on every evaluation -- the exact shape of the
        # ablation that read as chance while preserving 99.7% of the variance.
        idx = torch.randint(0, self.n_mossy, (self.n_granule, self.n_dendrite),
                            generator=gen)
        self.register_buffer("mf_idx", idx.to(device).long())
        # per-granule threshold offsets, per-unit tonic jitter, per-olive drive jitter:
        # all from the same single generator, all once.
        gj = (torch.rand(self.n_granule, generator=gen) - 0.5) * 2.0 * pr.theta_G_jitter
        ij = (torch.rand(self.n_mli, generator=gen) - 0.5) * 2.0 * pr.i_jitter
        pj = (torch.rand(self.n_purkinje, generator=gen) - 0.5) * 2.0 * pr.p_jitter
        oj = (torch.rand(self.n_olive, generator=gen) - 0.5) * 2.0 * pr.io_jitter
        self.register_buffer("g_jit", gj.to(device))
        self.register_buffer("i_jit", ij.to(device))
        self.register_buffer("p_jit", pj.to(device))
        self.register_buffer("o_jit", oj.to(device))

        # -------------------------------------------------- EXPERIENCE, not gradient
        # the parallel-fibre -> Purkinje weights.  a persistent BUFFER, following
        # `ibm/substrate.py`'s `P`: gradient descent does not touch it, `learn_step` does,
        # and it is saved with the model because it is part of what the model has become.
        self.register_buffer("W_pf", torch.full((self.n_purkinje, self.n_granule),
                                                float(pr.w_pf_init), device=device))

        # -------------------------------------------------- bounded learned residuals
        sizes = {"G": self.n_granule, "I": self.n_mli, "P": self.n_purkinje,
                 "N": self.n_nucleus, "O": self.n_olive}
        self._res_pop = dict(_RESIDUAL_POP)
        for name, pop in _RESIDUAL_POP.items():
            base = torch.zeros(sizes[pop], device=device) + float(getattr(pr, name))
            if name == "theta_G":
                base = base + self.g_jit          # the jitter rides on the prior itself
            self.register_buffer(f"prior_{name}", base)
            self.register_parameter(f"res_{name}", nn.Parameter(
                torch.zeros(sizes[pop], device=device), requires_grad=learn))

    # ------------------------------------------------------------------ parameters
    def site(self, name: str) -> torch.Tensor:
        p = getattr(self, f"prior_{name}")
        r = getattr(self, f"res_{name}")
        return p * (1.0 + self.residual_frac * torch.tanh(r))

    # ------------------------------------------------------------------ conduction
    def delays_in_steps(self, dt: float) -> dict:
        """the conduction budget at this dt.

        Reports the REQUESTED lag and the REALISED one separately.  A dt too coarse to
        resolve a delay is clamped to one step -- never to zero, which would turn a
        conduction time into an instantaneous handshake -- and the two columns differing
        is how a caller finds out.
        """
        pr = self.pr
        out = {}
        for f in dataclass_fields(pr):
            if not f.name.startswith("d_"):
                continue
            sec = float(getattr(pr, f.name))
            req = int(round(sec / dt))
            out[f.name] = {"seconds": sec, "requested_steps": req,
                           "used_steps": max(1, req), "resolved": req >= 1}
        return out

    def _lags(self, dt: float) -> dict:
        return {k: v["used_steps"] for k, v in self.delays_in_steps(dt).items()}

    # ------------------------------------------------------------------ state
    def init_state(self, b: int, dt: float, device=None) -> dict:
        """rest.  The rings are part of the state, so a state is a complete description of
        the microzone including what is still in flight on its axons."""
        device = device or self.W_pf.device
        lags = self._lags(dt)
        z = lambda n: torch.zeros(b, n, device=device)          # noqa: E731
        ring = lambda n, L: torch.zeros(b, L, n, device=device)  # noqa: E731
        st = {
            # `Go` is one Golgi pool per microzone: the feedback is a normalisation over
            # the whole granule layer, and a per-granule Golgi cell would be a precision
            # the rest of this module does not have.
            "M": z(self.n_mossy), "G": z(self.n_granule), "Go": z(1),
            "I": z(self.n_mli), "P": z(self.n_purkinje), "N": z(self.n_nucleus),
            "O": z(self.n_olive), "z": z(self.n_olive),
            "sII": z(self.n_mli), "sPI": z(self.n_mli), "sIP": z(self.n_purkinje),
            "sPN": z(self.n_nucleus), "sNO": z(self.n_olive),
            "sC": z(self.n_purkinje), "cf": z(self.n_purkinje),
            "gtrace": z(self.n_granule),
            "eta_I": z(self.n_mli), "eta_P": z(self.n_purkinje), "eta_O": z(self.n_olive),
            "ring_M": ring(self.n_mossy, lags["d_mf_gr_s"] + 1),
            "ring_G": ring(self.n_granule, lags["d_pf_s"] + 1),
            "ring_I": ring(self.n_mli, max(lags["d_ii_s"], lags["d_bk_s"]) + 1),
            "ring_P": ring(self.n_purkinje,
                           max(lags["d_pk_n_s"], lags["d_coll_s"]) + 1),
            "ring_N": ring(self.n_nucleus, lags["d_n_io_s"] + 1),
            "ring_O": ring(self.n_olive, lags["d_io_pk_s"] + 1),
            "t": 0, "lags": lags, "dt": float(dt),
        }
        return st

    @staticmethod
    def detach(state):
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in state.items()}

    def noise_shapes(self) -> dict:
        return {"I": self.n_mli, "P": self.n_purkinje, "O": self.n_olive}

    def draw_noise(self, gen: torch.Generator, b: int, device=None) -> dict:
        """standard-normal draws for one step, from the generator the CALLER owns.

        This exists so that `step` never has to; `step` takes the numbers and cannot
        produce them.  An instrument that cannot draw cannot quietly redraw.
        """
        assert isinstance(gen, torch.Generator), "draw_noise needs an explicit generator"
        device = device or self.W_pf.device
        return {k: torch.randn(b, n, generator=gen, device="cpu").to(device)
                for k, n in self.noise_shapes().items()}

    # ------------------------------------------------------------------ the step
    def step(self, state: dict, mossy=None, drive_io=None, cf_in=None,
             drive_nucleus=None, noise: dict | None = None) -> dict:
        """one exponential-Euler step.  `dt` and the conduction lags come from the state.

        `mossy`       (b, n_mossy) in [0, 1] -- what the pons delivers, ALREADY delayed by
                      the caller if the caller owns that delay; the mossy -> granule
                      conduction time is applied here and applying it twice is the bug
                      `ThalamoCortical.step` warns about.
        `drive_io`    (b, n_olive) added to the olive's input: the error the olive reports.
        `cf_in`       (b, n_olive) a teacher's climbing fibre, injected at the olive's AXON
                      so that it still arrives at the Purkinje cells after `d_io_pk_s`.
        `drive_nucleus` (b, n_nucleus) an external drive on the nucleus.
        `noise`       a dict of standard-normal draws from `draw_noise`.  Never drawn here.
        """
        pr = self.pr
        dt = state["dt"]
        t = state["t"]
        lg = state["lags"]

        M, G, Go = state["M"], state["G"], state["Go"]
        I, P, N, O, z = state["I"], state["P"], state["N"], state["O"], state["z"]
        sII, sPI, sIP = state["sII"], state["sPI"], state["sIP"]
        sPN, sNO, sC = state["sPN"], state["sNO"], state["sC"]
        gtr = state["gtrace"]
        eI, eP, eO = state["eta_I"], state["eta_P"], state["eta_O"]

        # ---- what arrives now: read every ring BEFORE anything is written into one
        # A synaptic variable lives on its TARGET population, so a projection between two
        # populations of different sizes is pooled here, explicitly, with a mean.  Within
        # one microzone every basket cell reaches every Purkinje cell in its strip and
        # every Purkinje cell in the strip converges on the same nuclear patch, so the
        # mean is the anatomy and not a shortcut; the moment a microzone needs internal
        # topography it needs a projection matrix, and this is where it would go.
        M_del = _read(state["ring_M"], t, lg["d_mf_gr_s"])
        G_del = _read(state["ring_G"], t, lg["d_pf_s"])
        I_ii = _read(state["ring_I"], t, lg["d_ii_s"]).mean(-1, keepdim=True)
        I_bk = _read(state["ring_I"], t, lg["d_bk_s"]).mean(-1, keepdim=True)
        P_n = _read(state["ring_P"], t, lg["d_pk_n_s"]).mean(-1, keepdim=True)
        P_coll = _read(state["ring_P"], t, lg["d_coll_s"]).mean(-1, keepdim=True)
        N_io = _read(state["ring_N"], t, lg["d_n_io_s"]).mean(-1, keepdim=True)
        # the climbing fibre is the ONE projection that is not pooled: one olivary cell
        # per Purkinje cell, which is the anatomy.
        cf = _read(state["ring_O"], t, lg["d_io_pk_s"])

        # ---- background: Ornstein-Uhlenbeck, advanced exactly, never drawn here
        rho = math.exp(-dt / pr.tau_eta)
        s_noise = pr.sigma * math.sqrt(1.0 - rho * rho)
        eI, eP, eO = eI * rho, eP * rho, eO * rho
        if noise is not None:
            eI = eI + s_noise * noise["I"]
            eP = eP + s_noise * noise["P"]
            eO = eO + s_noise * noise["O"]

        # ---- mossy terminals: a relay
        mf = mossy if mossy is not None else torch.zeros_like(M)
        M_inf = torch.clamp(mf, 0.0, 1.0)

        # ---- granule layer: four mossy fibres, a coincidence threshold, Golgi feedback
        dend = M_del[:, self.mf_idx]                              # (b, n_gr, n_dendrite)
        mf_in = pr.w_MG * dend.mean(-1)                           # (b, n_gr)
        u_G = mf_in - self.site("w_GoG") * Go
        G_inf = torch.sigmoid(pr.beta_G * (u_G - self.site("theta_G")))
        # Golgi: driven by the layer it inhibits (feedback) and by the mossy input
        # (feed-forward).  the feedback arm is what holds the code sparse when the whole
        # input gets brighter, which is what makes the code a CODE.
        u_Go = pr.w_GGo * G.mean(-1, keepdim=True) + pr.w_MGo * M_del.mean(-1, keepdim=True)
        Go_inf = sigmoid(u_Go, pr.beta_Go, pr.theta_Go)

        # ---- molecular layer: the parallel fibre in, the recurrent inhibition that rings
        pf = G_del.mean(-1, keepdim=True)
        u_I = (self.site("w_GI") * pf + pr.i_tonic + self.i_jit
               - self.site("w_II") * sII - self.site("w_PI") * sPI + eI)
        I_inf = sigmoid(u_I, pr.beta_I, pr.theta_I)

        # ---- Purkinje: tonically active, driven through the LEARNED weights, silenced by
        #      the basket cells and, after a complex spike, by its own pause
        pfW = (G_del @ self.W_pf.t()) / float(self.n_granule)     # (b, n_pk)
        u_P = (self.site("p_tonic") + self.p_jit + self.site("w_GP") * pfW
               - self.site("w_IP") * sIP + pr.g_cs * cf - self.site("g_pause") * sC + eP)
        P_inf = sigmoid(u_P, pr.beta_P, pr.theta_P)

        # ---- deep nucleus: tonically driven, tonically held down.  its output is a
        #      RELEASE.  a Purkinje pause is what lets it through.
        mn = M_del.mean(-1, keepdim=True)
        u_N = self.site("w_MN") * mn + pr.n_drive - self.site("w_PN") * sPN
        if drive_nucleus is not None:
            u_N = u_N + drive_nucleus
        N_inf = sigmoid(u_N, pr.beta_N, pr.theta_N)

        # ---- inferior olive: a relaxation oscillator, gap-coupled, braked by the nucleus.
        # the calcium current reads the MEMBRANE state `O`, so it is regenerative, and the
        # potassium variable `z` follows `O` slowly and puts it out.  `ibm/thalamus.py`
        # records that a window current which can trigger itself is a PACEMAKER rather
        # than a rebound -- there that was the bug, here it is the specification.
        Obar = O.mean(-1, keepdim=True)
        ca = self.site("g_ca") * sigmoid(O, pr.ca_beta, pr.ca_theta)
        u_O = (pr.io_drive + self.o_jit + ca - pr.g_k * z
               + self.site("w_gap") * (Obar - O) - self.site("w_NO") * sNO + eO)
        if drive_io is not None:
            u_O = u_O + drive_io
        O_inf = sigmoid(u_O, pr.beta_O, pr.theta_O)
        # the potassium current is gated by the CALCIUM that entered, not by the membrane:
        # `z_inf` is the same activation term the calcium current uses.  that is the
        # biology, and it is also what makes the cell oscillate rather than sit still --
        # with `z_inf = O` the nullclines cross on the stable lower branch for every
        # `io_drive`, and the first version of this module measured exactly that (a
        # population standard deviation of 0.0000 at four drive levels).
        z_inf = ca / self.site("g_ca").clamp_min(1e-6)

        # ---- exponential-Euler coefficients.  every target above is inside [0, 1].
        cM = 1.0 - math.exp(-dt / pr.tau_M)
        cG = 1.0 - math.exp(-dt / pr.tau_G)
        cGo = 1.0 - math.exp(-dt / pr.tau_Go)
        cI = 1.0 - math.exp(-dt / pr.tau_I)
        cP = 1.0 - math.exp(-dt / pr.tau_P)
        cN = 1.0 - math.exp(-dt / pr.tau_N)
        cO = 1.0 - math.exp(-dt / pr.tau_O)
        cz = 1.0 - math.exp(-dt / pr.tau_z)
        cgI = 1.0 - math.exp(-dt / pr.tau_gaba_I)
        cbk = 1.0 - math.exp(-dt / pr.tau_bk)
        cgN = 1.0 - math.exp(-dt / pr.tau_gaba_N)
        cgO = 1.0 - math.exp(-dt / pr.tau_gaba_O)
        cel = 1.0 - math.exp(-dt / pr.tau_elig)
        c_up = 1.0 - math.exp(-dt / pr.tau_pause_up)
        c_dn = 1.0 - math.exp(-dt / pr.tau_pause_dn)

        Mn = M + cM * (M_inf - M)
        Gn = G + cG * (G_inf - G)
        Gon = Go + cGo * (Go_inf - Go)
        In = I + cI * (I_inf - I)
        Pn = P + cP * (P_inf - P)
        Nn = N + cN * (N_inf - N)
        On = O + cO * (O_inf - O)
        zn = z + cz * (z_inf - z)
        # the pause: fast to engage, slow to release -- that asymmetry IS the pause
        cpz = torch.where(cf > sC, torch.full_like(sC, c_up), torch.full_like(sC, c_dn))
        sCn = sC + cpz * (cf - sC)

        out = {
            "M": Mn, "G": Gn, "Go": Gon, "I": In, "P": Pn, "N": Nn, "O": On, "z": zn,
            "sII": sII + cgI * (I_ii - sII),
            "sPI": sPI + cgI * (P_coll - sPI),
            "sIP": sIP + cbk * (I_bk - sIP),
            "sPN": sPN + cgN * (P_n - sPN),
            "sNO": sNO + cgO * (N_io - sNO),
            "sC": sCn, "cf": cf, "cs": sigmoid(On, pr.beta_cs, pr.theta_cs),
            "gtrace": gtr + cel * (G_del - gtr),
            "eta_I": eI, "eta_P": eP, "eta_O": eO,
            "t": t + 1, "lags": lg, "dt": dt,
        }
        # ---- post the new values into the rings for their arrival times
        # what leaves the olive is the COMPLEX SPIKE, not the membrane oscillation; a
        # teacher's climbing fibre is injected at the same point, so it too arrives at the
        # Purkinje cells only after the catalogue's climbing-fibre conduction time.
        cs = sigmoid(On, pr.beta_cs, pr.theta_cs)
        emit_O = cs if cf_in is None else torch.clamp(cs + cf_in, 0.0, 1.0)
        out["ring_M"] = _write(state["ring_M"], t, Mn)
        out["ring_G"] = _write(state["ring_G"], t, Gn)
        out["ring_I"] = _write(state["ring_I"], t, In)
        out["ring_P"] = _write(state["ring_P"], t, Pn)
        out["ring_N"] = _write(state["ring_N"], t, Nn)
        out["ring_O"] = _write(state["ring_O"], t, emit_O)
        return out

    # ------------------------------------------------------------------ plasticity
    #
    # THE RULE.  a parallel fibre that was active shortly before a climbing-fibre event is
    # depressed; a parallel fibre active without one is potentiated.
    #
    #     dW_ij/dt = gtrace_j * ( a_ltp * (1 - c_i)  -  a_ltd * c_i )
    #
    # `gtrace` is the eligibility: a 100 ms trace of the parallel-fibre input, which is the
    # measured PF-before-CF window and is why the teaching signal may arrive long after the
    # activity it is about.  `c` is the climbing fibre that ARRIVED at the Purkinje cell
    # this step, i.e. after the catalogue's 3-7 ms climbing-fibre delay.
    #
    # it is a buffer, updated under `no_grad`, following `ibm/substrate.py`'s `P`.  the
    # equilibrium climbing-fibre rate is a_ltp/(a_ltp+a_ltd), so the rule has a set point
    # rather than a rail.
    @torch.no_grad()
    def learn_step(self, state: dict, eta: float = 1.0) -> None:
        dt = state["dt"]
        g = state["gtrace"]                                  # (b, n_gr)
        c = state["cf"]                                      # (b, n_pk)
        drive = (self.pr.a_ltp * (1.0 - c) - self.pr.a_ltd * c).unsqueeze(-1)
        dW = (drive * g.unsqueeze(1)).mean(0)                # (b,n_pk,n_gr) -> (n_pk,n_gr)
        self.W_pf += float(dt) * float(eta) * dW
        self.W_pf.clamp_(0.0, self.pr.w_pf_max)

    # ------------------------------------------------------------------ rollout
    def rollout(self, steps: int, dt: float, state=None, mossy=None, drive_io=None,
                cf_in=None, drive_nucleus=None, noise_gen=None, b: int = 1,
                record=("P", "I", "N", "O", "G")):
        """the microzone alone.  Returns (dict of (B, steps, n) traces, final state).

        `mossy` etc may be None, a (B, n) tensor held constant, or a (B, steps, n) tensor.
        """
        state = state or self.init_state(b, dt, device=self.W_pf.device)
        pick = lambda x, i: (x[:, i] if torch.is_tensor(x) and x.dim() == 3 else x)  # noqa: E731
        out = {k: [] for k in record}
        for i in range(steps):
            z = self.draw_noise(noise_gen, b) if noise_gen is not None else None
            state = self.step(state, mossy=pick(mossy, i), drive_io=pick(drive_io, i),
                              cf_in=pick(cf_in, i),
                              drive_nucleus=pick(drive_nucleus, i), noise=z)
            for k in out:
                out[k].append(state[k])
        return {k: torch.stack(v, 1) for k, v in out.items()}, state

    # ------------------------------------------------------------------ resting state
    @torch.no_grad()
    def resting_state(self, dt: float = 0.0002, seconds: float = 1.0, b: int = 1) -> dict:
        """what the microzone does with NO input at all, run to steady state.

        This is here because "Purkinje cells are tonically active at rest" is a claim about
        the model that can be checked, and a resting state that has to be described in
        prose instead of printed is not a resting state anyone can rely on.
        """
        tr, st = self.rollout(int(seconds / dt), dt, b=b,
                              record=("P", "I", "N", "O", "G", "Go"))
        # averaged over the last HALF SECOND, not the last few steps: the olive cycles
        # several times a second and a short tail reports a phase of that cycle rather
        # than a resting state.
        tail = {k: v[:, -int(min(0.5, seconds * 0.5) / dt):] for k, v in tr.items()}
        return {k: float(v.mean()) for k, v in tail.items()} | {
            "P_hz": float(tail["P"].mean()) * R_MAX,
            "N_hz": float(tail["N"].mean()) * R_MAX,
            "granule_active_fraction": float((tail["G"] > 0.1).float().mean()),
        }

    def rate_hz(self, state, key: str = "P"):
        return state[key] * R_MAX


class CerebelloThalamoCortical(nn.Module):
    """the loop out and back: `ctc`, with every edge's conduction time from the catalogue.

    m1 -> pons -> granule layer -> Purkinje -> nucleus -> VL -> m1.  The cortical and
    thalamic stages here are deliberately THIN -- a leaky integrator each, standing in for
    `ibm/substrate.py`'s cortical field and `ibm/thalamus.py`'s VL relay, which are the real
    ones.  The point of this class is not to model motor cortex; it is to close the loop so
    the conduction budget is a round trip that can be reported in steps and in milliseconds,
    and so the whole thing can be shown to run bounded.  Anything that wants the real cortex
    should compose `CerebellarMicrozone` with those modules instead.
    """

    def __init__(self, mz: CerebellarMicrozone, tau_ctx: float = 0.020,
                 tau_vl: float = 0.012, w_n_vl: float = 1.0, w_vl_ctx: float = 0.8,
                 w_ctx_mf: float = 1.0):
        super().__init__()
        self.mz = mz
        self.tau_ctx, self.tau_vl = float(tau_ctx), float(tau_vl)
        self.w_n_vl, self.w_vl_ctx, self.w_ctx_mf = w_n_vl, w_vl_ctx, w_ctx_mf

    def round_trip(self, dt: float) -> dict:
        """the `ctc` round trip, in seconds and in steps, edge by edge."""
        d = self.mz.delays_in_steps(dt)
        edges = ["d_ctx_pons_s", "d_mf_gr_s", "d_pk_n_s", "d_n_thal_s", "d_thal_ctx_s"]
        total_s = sum(d[e]["seconds"] for e in edges)
        total_steps = sum(d[e]["used_steps"] for e in edges)
        return {"edges": {e: d[e] for e in edges}, "round_trip_s": total_s,
                "round_trip_ms": total_s * 1000.0, "round_trip_steps": total_steps,
                "loop_bound_hz": 1000.0 / (2.0 * total_s * 1000.0)}

    def rollout(self, steps: int, dt: float, command=None, noise_gen=None, b: int = 1):
        """run the closed loop.  Returns (traces, state, info)."""
        mz = self.mz
        dev = mz.W_pf.device
        lg = mz._lags(dt)
        st = mz.init_state(b, dt, device=dev)
        L_c = lg["d_ctx_pons_s"] + 1
        L_v = lg["d_n_thal_s"] + 1
        L_t = lg["d_thal_ctx_s"] + 1
        ring_c = torch.zeros(b, L_c, mz.n_mossy, device=dev)     # m1 -> pons -> mossy
        ring_v = torch.zeros(b, L_v, mz.n_nucleus, device=dev)   # nucleus -> vl
        ring_t = torch.zeros(b, L_t, 1, device=dev)              # vl -> m1
        ctx = torch.zeros(b, 1, device=dev)
        vl = torch.zeros(b, 1, device=dev)
        cc = 1.0 - math.exp(-dt / self.tau_ctx)
        cv = 1.0 - math.exp(-dt / self.tau_vl)
        traces = {"ctx": [], "vl": [], "P": [], "N": []}
        for i in range(steps):
            t = st["t"]
            mf = ring_c[:, (t - lg["d_ctx_pons_s"]) % L_c]
            vl_in = ring_v[:, (t - lg["d_n_thal_s"]) % L_v].mean(-1, keepdim=True)
            ctx_in = ring_t[:, (t - lg["d_thal_ctx_s"]) % L_t]
            cmd = command[:, i] if torch.is_tensor(command) and command.dim() == 3 else command
            drive = self.w_vl_ctx * ctx_in + (cmd if cmd is not None else 0.0)
            ctx = ctx + cc * (torch.clamp(drive, 0.0, 1.0) - ctx)
            vl = vl + cv * (torch.clamp(self.w_n_vl * vl_in, 0.0, 1.0) - vl)
            z = mz.draw_noise(noise_gen, b) if noise_gen is not None else None
            st = mz.step(st, mossy=mf, noise=z)
            ring_c = ring_c.clone()
            ring_v = ring_v.clone()
            ring_t = ring_t.clone()
            ring_c[:, t % L_c] = torch.clamp(self.w_ctx_mf * ctx, 0.0, 1.0).expand(
                b, mz.n_mossy)
            ring_v[:, t % L_v] = st["N"]
            ring_t[:, t % L_t] = vl
            traces["ctx"].append(ctx)
            traces["vl"].append(vl)
            traces["P"].append(st["P"])
            traces["N"].append(st["N"])
        return ({k: torch.stack(v, 1) for k, v in traces.items()}, st,
                self.round_trip(dt))


def swept_constants() -> list:
    """the declared constants a sensitivity sweep should move.

    Everything in `CerebellarPriors`, including the conduction times: a delay that can be
    halved and doubled without moving the fast rhythm is a delay that is not in the loop,
    and that is exactly the thing this programme keeps discovering the hard way.
    """
    return [f.name for f in dataclass_fields(CerebellarPriors)]
