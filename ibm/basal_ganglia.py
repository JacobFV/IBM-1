"""the basal ganglia: three pathways, a diffuse subthalamic nucleus, and N candidate actions.

WHY THIS EXISTS
---------------
`ibm/rhythms.py` declares four basal-ganglia loops -- `cbgtc` (the direct arm),
`bg_indirect`, `bg_hyperdirect` and `stn_gpe` -- and four rows that only those loops can
produce: `beta_bg` (13-30 Hz in the subthalamo-pallidal pair), `beta_bursts` (that beta
arrives in bursts of 100-500 ms and NOT as a tone), `action_selection` ("one proposal
released, the rest suppressed") and `stop_signal` (cancelling a movement already under
way).  Every one of them carries `substrate: needs-basal-ganglia`, which is to say: the
cortical sheet cannot express them, the thalamus cannot express them, and scoring the
13-30 Hz band in cortex alone would produce a number that moves for the wrong reason.

`action_selection` is the row that says what this module is FOR, and its own note says it
plainly: "a model that reproduced every band in this file while selecting nothing would
have missed the point."  So the gates here are a selection gate and a stop gate first and
a spectrum second.

THE MODEL, and why each piece is here
-------------------------------------
N parallel CHANNELS.  A channel is a candidate action -- one cortical proposal and the
loop that carries it -- and the whole point of the architecture is that the channels
compete.  Per channel:

    D1    striatal direct-pathway medium spiny neurons     in [0, 1]
    D2    striatal indirect-pathway medium spiny neurons   in [0, 1]
    GPe   external pallidum                                in [0, 1]
    GPi   internal pallidum / SNr, the OUTPUT nucleus      in [0, 1]
    STN   subthalamic nucleus                              in [0, 1]
    Thal  the pallidal-recipient thalamic relay (VA/VL)    in [0, 1]

plus seven synaptic activations (one per projection, decaying with its transmitter's own
time constant) and one Ornstein-Uhlenbeck background current per population.

    direct       ctx -> D1 -| GPi -| Thal        a proposal REMOVES inhibition
    indirect     ctx -> D2 -| GPe -| STN -> GPi  a proposal ADDS inhibition
    hyperdirect  ctx -> STN -> GPi               straight past the striatum
    pacemaker    STN -> GPe -| STN               the reciprocal pair that rings

**GPi FIRES HIGH AT REST AND SELECTION IS A PAUSE IN IT.**  This is the fact the whole
module is arranged around, and it is why `init_state` does not start at zero.  GPe, GPi
and STN are autonomously active: ~60, ~70 and ~25 Hz in the awake primate (DeLong 1971;
Wichmann & DeLong 1996), and the thalamus sits underneath that tonic inhibition nearly
silent.  A movement is released by the output nucleus going QUIET on one channel, not by
anything switching on.  `rest_gpe`, `rest_gpi`, `rest_stn` and `rest_thal` are therefore
declared as rates, and the tonic bias current each population needs to sit there is
SOLVED from them (`_tonic`), per channel, including the learned residuals.  That way a
swept weight moves the dynamics and leaves the resting state exactly where it was
declared -- which is the only way a sensitivity sweep says anything, since a weight that
also slid the operating point would confound the two.

**THE STN IS THE DIFFUSE ONE, AND THAT ASYMMETRY IS THE HYPERDIRECT PATHWAY'S MEANING.**
A subthalamic axon arborises across the pallidum rather than staying in its channel
(Parent & Hazrati 1995), while the striatopallidal and pallidosubthalamic projections are
topographic.  So here:

  * the STN's OUTPUT to both pallidal segments is `stn_diffuse` of the channel mean plus
    `1 - stn_diffuse` of the channel's own rate -- one constant, applied to both efferents
    because it is one axon arbor;
  * the STN's cortical (hyperdirect) input is the channel MEAN of the cortical drive,
    fully diffuse, because a stop signal is not addressed to a channel;
  * the STN's pallidal input is per channel, because that projection is topographic.

That is what makes the hyperdirect pathway a GLOBAL BRAKE: a pulse into the STN raises
GPi on every channel at once and cancels whatever was selected, whereas the direct
pathway's release is confined to the channel that earned it.  Set `stn_diffuse` to 0 and
the brake becomes per-channel and stops being a brake; gate B7 sweeps it.

**DOPAMINE IS A PARAMETER, NOT A FLAG.**  `dopamine` in [0, 1] scales the cortical drive
onto the two striatal populations in opposite directions about `da_tonic`:

    g_D1 = 1 + k_da_d1 * (DA - da_tonic)        D1 receptors are excitatory-modulatory
    g_D2 = 1 - k_da_d2 * (DA - da_tonic)        D2 receptors are inhibitory-modulatory

Nothing else in the module reads it.  Lowering it therefore weakens the direct arm and
strengthens the indirect arm through the SAME equations that run at normal dopamine --
there is no Parkinsonian branch anywhere in this file.  The prediction that falls out,
which gate B6 declares before it measures it, is that low dopamine both raises beta (more
D2 drive -> less GPe -> a released STN, and the pair moves toward its Hopf point) and
weakens selection (less D1 drive -> a shallower GPi pause).

**THE BETA COMES FROM THE PAIR'S TIME CONSTANTS, NOT FROM ITS DELAY.**  `beta_bg`'s own
note says so: the STN-GPe ring is 5-9 ms net inhibitory, a ceiling of 56-100 Hz, so the
delay is nowhere near binding and "the synaptic time constants do" set the band.  The
phase condition for a delayed negative feedback is

    2*pi*f*D + atan(2*pi*f*tau_stn) + atan(2*pi*f*tau_ampa_stn)
             + atan(2*pi*f*tau_gpe) + atan(2*pi*f*tau_gaba_pal)  =  pi

and with the declared values (D = 7 ms, taus 5, 4, 6, 9 ms) that crosses at ~18 Hz, in
band.  `tau_gaba_pal` and `tau_ampa_stn` are the constants this module CLAIMS set the
frequency, and gate B7 requires each of them to move it by more than 1 Hz.

**AND THE PAIR SITS JUST BELOW ITS BIFURCATION, ON PURPOSE.**  `beta_bursts` is the row
that stops a band-power target from being satisfied the wrong way: a constant 13-30 Hz
tone scores the same band power as real bursting and is not the same object.  A loop gain
above 1 gives a limit cycle -- a tone.  A loop gain just below 1 gives a DAMPED resonance,
which is silent on its own and rings for a few cycles whenever the background current
knocks it, and that is a burst of 100-500 ms.  So the weights are chosen to put the open
loop near unity gain at the resonance rather than comfortably above it, and the bursts are
a property of the operating point rather than an envelope imposed on a tone.

**BOUNDED, like the cortical field and the thalamus.**  Every rate and every synaptic
activation is advanced by an exponential-Euler step, `x <- x + (1 - exp(-dt/tau)) * (f - x)`
with `f` in [0, 1], which is a convex combination of two points in the box.  They stay in
[0, 1] for any dt, any weights and any input; divergence cannot be represented.  The OU
background currents are NOT rates and are not in the box -- they are currents, and gate B0
says so rather than quietly testing something else.

**Noise is passed in, never drawn here** (CLAUDE.md, "Randomness"): `step` takes a
`(B, len(NOISE_POPS), N)` standard-normal draw or None, and `rollout` draws it once per
step from a generator the caller owns and asserts it was given one.

UNITS: seconds, rates dimensionless in [0, 1] with `R_MAX` = 100 Hz for readouts, delays
in seconds (the catalogue declares them in milliseconds, as intervals; the midpoint of
each declared interval is what is used here and the interval is quoted beside it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

R_MAX = 100.0

#: the populations that receive an independent background current, in the order `step`
#: expects along axis 1 of its `noise` argument.
NOISE_POPS = ("D1", "D2", "GPe", "GPi", "STN", "Thal")


# --------------------------------------------------------------------------------------
# declared priors.  every value carries its reason.  the four groups are: rates that a
# recording fixes, time constants that a transmitter fixes, weights that the architecture
# fixes, and conduction delays that `ibm/rhythms.py` fixes.
# --------------------------------------------------------------------------------------
@dataclass
class BasalGangliaPriors:
    # ---- the resting state, declared as RATES because that is what is measured --------
    # Awake primate single units: GPe ~60 Hz, GPi/SNr ~70 Hz, STN ~25 Hz, and the
    # pallidal-recipient thalamus held far below its own ceiling by that tonic inhibition
    # (DeLong 1971; Wichmann & DeLong 1996; Nambu 2000).  The tonic bias current that puts
    # each population here is SOLVED from these four numbers rather than declared, so a
    # change to any weight leaves the resting state where the measurement put it.
    rest_gpe: float = 0.60
    rest_gpi: float = 0.70
    rest_stn: float = 0.25
    rest_thal: float = 0.05

    # ---- membrane time constants -----------------------------------------------------
    # Striatal MSNs are the slow ones (a deep down-state and a large membrane time
    # constant); the pallidal and subthalamic cells are fast autonomous pacemakers.
    tau_msn: float = 0.010
    tau_gpe: float = 0.006
    tau_gpi: float = 0.006
    tau_stn: float = 0.005
    tau_thal: float = 0.010

    # ---- synaptic time constants -----------------------------------------------------
    # One constant per TRANSMITTER-AND-TARGET, not one per edge: the two STN efferents are
    # the same glutamatergic arbor and the two GPe efferents are the same GABAergic one,
    # so giving them separate constants would invent a distinction the anatomy does not
    # make.  These two are what set the beta frequency (see the module docstring) and gate
    # B7 requires each of them to move it.
    tau_gaba_str: float = 0.010    # striatum -> pallidum, slow GABA-A
    tau_ampa_stn: float = 0.004    # STN -> GPe and STN -> GPi, AMPA
    tau_gaba_pal: float = 0.009    # GPe -> STN and GPe -> GPi, GABA-A
    tau_gaba_gpi: float = 0.008    # GPi -> thalamus, the pallidothalamic GABA-A

    # ---- transfer functions ----------------------------------------------------------
    # MSNs have a HIGH threshold on purpose: a striatal projection neuron sits in a
    # down-state and is essentially silent until a coherent cortical input arrives, which
    # is why the striatum can act as a detector of proposals rather than a relay of them.
    beta_msn: float = 10.0
    theta_msn: float = 0.55
    beta_gpe: float = 6.0
    theta_gpe: float = 0.30
    beta_gpi: float = 6.0
    theta_gpi: float = 0.30
    beta_stn: float = 6.0
    theta_stn: float = 0.30
    beta_thal: float = 8.0
    theta_thal: float = 0.35

    # ---- weights ---------------------------------------------------------------------
    w_ctx_d1: float = 1.20         # corticostriatal, onto the direct-pathway MSNs
    w_ctx_d2: float = 1.20         # and onto the indirect-pathway MSNs, equal at rest:
                                   # the ASYMMETRY between the arms is dopamine's job, not
                                   # a wiring difference, or `dopamine` would be a flag.
    # Competition between channels.  This is NOT the weak MSN-to-MSN collateral, which is
    # far too sparse to implement a winner-take-all; it is the FEEDFORWARD inhibition of
    # the fast-spiking interneuron pool, which is powerful, divergent and is what actually
    # makes the striatum competitive (Gittis & Kreitzer 2012; Tepper 2008).  It acts on
    # the MEAN of the OTHER channels, so a channel never inhibits itself.
    w_msn_lat: float = 3.00
    w_d1_gpi: float = 1.60         # the direct arm: the inhibition whose withdrawal IS
                                   # the selection
    w_d2_gpe: float = 1.40         # the indirect arm's first inhibitory step
    # The pair's loop gain, and the one pair of constants in this file that a MEASUREMENT
    # sets rather than the literature.  `gate_basal_ganglia.py --sweep` walks them from
    # 1.20 to 2.00 and records the whole curve.  What it shows:
    #
    #   w     prominence   envelope CV   median "burst"
    #   1.20    +0.55         0.554          47 ms
    #   1.40    +1.07         0.455          54 ms
    #   1.48    +1.50         0.235          55 ms
    #   1.60    +1.76         0.142          53 ms
    #   2.00    +2.20         0.065          45 ms
    #
    # The bifurcation is between 1.44 and 1.48: above it the envelope's coefficient of
    # variation collapses and the pair is a TONE.  The declared value is the largest that
    # keeps the envelope clearly noise-driven (CV 0.455 against the Rayleigh 0.523 a
    # linear resonator would give), which is the sharpest resonance that is still a
    # resonance.  It is NOT chosen to satisfy the burst-duration gate, because no value
    # does -- see `g_adapt` below.
    w_gpe_stn: float = 1.40        # pallidosubthalamic, topographic
    w_stn_gpe: float = 1.40        # subthalamopallidal, diffuse
    w_stn_gpi: float = 2.20        # the brake's grip on the output nucleus.  LARGER than
                                   # w_d1_gpi, and that inequality is what makes a stop
                                   # able to cancel a selection already granted rather
                                   # than merely dent it.
    w_gpe_gpi: float = 0.50        # the pallidopallidal projection.  Anatomically real
                                   # (Smith 1998) and NOT declared in the catalogue, which
                                   # has no gpe->gpi edge; flagged here rather than
                                   # smuggled in, and its delay is marked the same way.
    w_gpe_gpe: float = 0.30        # GPe onto itself, the shell's own inhibition
    # TWO corticosubthalamic weights, because there are two cortical sources and they do
    # different jobs.  The motor proposal's own hyperdirect collateral (`w_ctx_stn`) is
    # the WEAKER one, and its weakness is not a convenience: a proposal that drove the
    # diffuse STN as hard as a stop signal does would brake itself, and measured at
    # w_ctx_stn = 1.5 it does exactly that -- the winner's GPi never leaves its resting
    # rate (70.9 Hz against a 70 Hz rest) because the global brake its own proposal
    # applied outweighs the direct arm's release.  The stop signal (`w_stop_stn`) comes
    # from the right inferior frontal / preSMA stopping network, a different cortical
    # population projecting down the same hyperdirect tract, and it is the strong one --
    # a brake that cannot reach the top of the STN's range cannot cancel a selection that
    # has already been granted.
    w_ctx_stn: float = 0.90
    w_stop_stn: float = 2.00
    w_gpi_thal: float = 1.20       # the pallidothalamic inhibition the relay lives under
    w_ctx_thal: float = 0.50       # the corticothalamic input to the same relay

    # ---- the STN's diffuseness -------------------------------------------------------
    # fraction of the STN's efferent that is the CHANNEL MEAN rather than the channel's
    # own rate.  0.8, not 1.0: the arbor is wide but it is not uniform, and a completely
    # shared STN would make the indirect pathway global too, which would destroy the
    # selection it is supposed to sharpen.  See the module docstring.
    stn_diffuse: float = 0.80

    # ---- the STN's slow adaptation, and why `beta_bursts` cannot be had without it ----
    # A slow, activity-dependent hyperpolarising current in the subthalamic cell (the
    # calcium-dependent potassium conductance; Hallworth 2003), tracking the STN's own
    # rate with a few hundred milliseconds' lag.
    #
    # It is here for a reason worth stating, because the first version of this module did
    # not have it and the burst gate could not be satisfied at ANY loop gain.  With the
    # pair below its bifurcation the beta is broadband-driven, and the envelope of a
    # broadband signal filtered to 13-30 Hz has a correlation time of about 1/(17 Hz) --
    # so the measured "burst duration" came back at 45-57 ms across a 2x sweep of the loop
    # gain.  That number was the ANALYSIS BAND's, not the circuit's.  Pushing the gain up
    # instead gives a limit cycle, and the envelope's coefficient of variation collapses
    # from 0.55 to 0.14 by w = 1.6: a tone, which is precisely the object the catalogue's
    # `beta_bursts` row exists to exclude.  Neither end of that sweep is bursting.
    #
    # A burst of 100-500 ms needs a slow variable of its own.  The pair is put ABOVE its
    # bifurcation so it wants to ring; the adaptation then builds up, moves the STN off
    # the operating point where the sigmoid's slope is steepest, and quenches it; the
    # current decays and it rings again.  `tau_adapt` is therefore the constant this
    # module CLAIMS sets the burst duration, and gate B7 requires it to move it.
    g_adapt: float = 1.00
    tau_adapt: float = 0.250

    # ---- dopamine --------------------------------------------------------------------
    da_tonic: float = 0.50         # the level the declared weights are stated at, and the
                                   # level the tonic bias currents are SOLVED at -- so
                                   # moving dopamine genuinely moves the network off its
                                   # declared resting state instead of being re-absorbed.
    # Dopamine acts twice on each striatal population, because it does two things.
    # `k_*` is the GAIN on the corticostriatal synapse (D1 receptors facilitate it, D2
    # receptors depress it).  `e_*` is the MSN's own EXCITABILITY, as a shift of its input
    # current, which is the term that matters at rest: with the gain term alone, dopamine
    # multiplies a cortical drive that is zero during quiet rest and therefore changes
    # NOTHING there -- measured, and the resting beta spectrum was bit-identical at
    # dopamine 0.5 and 0.2 (peak 17.61 Hz, prominence +0.535, both).  That is CLAUDE.md's
    # "a parameter that changes nothing" in its usual form: the mechanism was real but it
    # was not wired in at the operating point being measured.  The excitability term is
    # the one that makes a depleted striatum tonically over-drive the indirect arm.
    k_da_d1: float = 1.20
    k_da_d2: float = 1.20
    e_da_d1: float = 1.20
    e_da_d2: float = 1.20

    # ---- background ------------------------------------------------------------------
    # The OU current that knocks the subcritical STN-GPe pair into ringing.  tau_eta is
    # 5 ms so the drive is broadband well past beta: a background whose corner sat inside
    # the band would be manufacturing the rhythm it is supposed to reveal.
    sigma: float = 0.05
    tau_eta: float = 0.005

    # ---- conduction, from ibm/rhythms.py ---------------------------------------------
    # Each is the MIDPOINT of the catalogue's declared (lo, hi) interval, quoted beside it.
    # `rollout` reports every one in STEPS, so a dt too coarse to resolve one is visible
    # rather than silent.
    d_ctx_str_s: float = 0.0055    # cbgtc / bg_indirect  pmc->striatum      (3, 8) ms
    d_ctx_stn_s: float = 0.0035    # bg_hyperdirect       pmc->stn           (2, 5) ms
    d_str_gpi_s: float = 0.0045    # cbgtc                striatum->gpi      (3, 6) ms
    d_str_gpe_s: float = 0.0045    # bg_indirect          striatum->gpe      (3, 6) ms
    d_gpe_stn_s: float = 0.0040    # stn_gpe / bg_indirect gpe->stn          (3, 5) ms
    d_stn_gpe_s: float = 0.0030    # stn_gpe              stn->gpe           (2, 4) ms
    d_stn_gpi_s: float = 0.0030    # bg_indirect / hyper  stn->gpi           (2, 4) ms
    d_gpe_gpi_s: float = 0.0030    # NOT IN THE CATALOGUE.  The catalogue has no gpe->gpi
                                   # edge; taken as the intrapallidal distance, shorter
                                   # than gpe->stn.  Declared here so the sweep can see it.
    d_gpi_thal_s: float = 0.0035   # cbgtc                gpi->va_vl_bg      (2, 5) ms
    d_thal_ctx_s: float = 0.0030   # cbgtc                va_vl_bg->pmc      (2, 4) ms
                                   # the return leg.  Cortex is `ibm/substrate.py`'s, not
                                   # this module's, so this delay is applied to the
                                   # EXPORTED thalamic trace (`ctx_return`) and closes no
                                   # loop inside this file.  B7 will therefore find it
                                   # inert for beta and for selection, correctly.


def sigmoid(x, beta, theta):
    return torch.sigmoid(beta * (x - theta))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class BasalGanglia(nn.Module):
    """`n` competing channels through the three pathways, the STN, and the thalamic relay.

    A CHANNEL is a candidate action: one cortical proposal, its two striatal populations,
    its sector of each pallidal segment, and the thalamic relay cell that returns it.  The
    subthalamic nucleus is the exception and it is deliberate -- see `stn_diffuse` and the
    module docstring.  `n = 8` by default because the row this module exists to satisfy
    (`action_selection`) is about several proposals arriving at once, and one or two
    channels cannot distinguish "selected the strongest" from "responded to the input".
    """

    #: the constants that carry a per-channel learned residual.  The declared loop may be
    #: bent, it may not be replaced: `site()` bounds every residual to +-35% of its prior,
    #: exactly as `ThalamicField` does.  Time constants and delays are NOT on this list --
    #: those are transmitter kinetics and axon lengths, not things a gradient may edit.
    RESIDUAL = ("w_ctx_d1", "w_ctx_d2", "w_d1_gpi", "w_d2_gpe", "w_gpe_stn",
                "w_stn_gpe", "w_stn_gpi", "w_ctx_stn", "w_gpi_thal")

    def __init__(self, n: int = 8, priors: BasalGangliaPriors | None = None,
                 learn: bool = True, device="cpu"):
        super().__init__()
        self.pr = priors or BasalGangliaPriors()
        self.n = int(n)
        self.learn = learn
        z = torch.zeros(self.n, device=device)
        for name in self.RESIDUAL:
            self.register_buffer(f"prior_{name}", z + float(getattr(self.pr, name)))
            self.register_parameter(f"res_{name}", nn.Parameter(
                torch.zeros(self.n, device=device), requires_grad=learn))
        self.residual_frac = 0.35

    # ---- parameters ------------------------------------------------------------------
    def site(self, name: str) -> torch.Tensor:
        """the per-channel value of a declared constant, bounded to +-35% of its prior."""
        p = getattr(self, f"prior_{name}")
        r = getattr(self, f"res_{name}")
        return p * (1.0 + self.residual_frac * torch.tanh(r))

    def _msn_rest(self) -> float:
        """the striatal resting rate: the fixed point with no cortical drive at all.

        Solved rather than declared, because it is not a free number -- with zero drive an
        MSN's rate is whatever its threshold and its own lateral inhibition leave it, and
        it enters the tonic solve below.  Three fixed-point iterations is plenty: the
        value is ~0.004 and the lateral term is a hundredth of the threshold.
        """
        pr = self.pr
        r = 1.0 / (1.0 + math.exp(-pr.beta_msn * (0.0 - pr.theta_msn)))
        for _ in range(3):
            u = -pr.w_msn_lat * r
            r = 1.0 / (1.0 + math.exp(-pr.beta_msn * (u - pr.theta_msn)))
        return r

    def params(self) -> dict:
        """every per-channel weight and every tonic current, read ONCE.

        `rollout` calls this once and hands the result to every `step`, rather than
        rebuilding nine bounded residuals and four solved bias currents on each of sixty
        thousand steps.  Nothing here depends on the state or on t, so computing it once
        changes no number and keeps the autograd graph intact -- the tensors are the same
        tensors, used at every step.
        """
        p = {k: self.site(k) for k in self.RESIDUAL}
        p["i_stn"], p["i_gpe"], p["i_gpi"], p["i_thal"] = self._tonic(p)
        return p

    def _tonic(self, p=None):
        """the bias current each population needs to sit at its DECLARED resting rate.

        This is the piece that makes "GPi fires high at rest" a property of the code and
        not a comment.  Given the resting rates, invert each sigmoid for the membrane
        drive it implies and subtract everything the other populations contribute at rest;
        what is left is the autonomous pacemaker current.  It is recomputed per step
        because the weights carry per-channel residuals, so the resting state stays where
        it was declared for every channel and every residual.
        """
        pr = self.pr
        p = p or {k: self.site(k) for k in self.RESIDUAL}
        msn = self._msn_rest()
        u_stn = _logit(pr.rest_stn) / pr.beta_stn + pr.theta_stn
        u_gpe = _logit(pr.rest_gpe) / pr.beta_gpe + pr.theta_gpe
        u_gpi = _logit(pr.rest_gpi) / pr.beta_gpi + pr.theta_gpi
        u_thal = _logit(pr.rest_thal) / pr.beta_thal + pr.theta_thal
        i_stn = u_stn + p["w_gpe_stn"] * pr.rest_gpe + pr.g_adapt * pr.rest_stn
        i_gpe = (u_gpe + p["w_d2_gpe"] * msn
                 - p["w_stn_gpe"] * pr.rest_stn + pr.w_gpe_gpe * pr.rest_gpe)
        i_gpi = (u_gpi + p["w_d1_gpi"] * msn
                 - p["w_stn_gpi"] * pr.rest_stn + pr.w_gpe_gpi * pr.rest_gpe)
        i_thal = u_thal + p["w_gpi_thal"] * pr.rest_gpi
        return i_stn, i_gpe, i_gpi, i_thal

    # ---- state -----------------------------------------------------------------------
    def init_state(self, b: int = 1, device=None):
        """the RESTING state, which is not zero.

        GPi at 70 Hz, GPe at 60, STN at 25, the thalamus nearly silent under them and the
        striatum in its down-state; every synaptic activation equal to the rate driving
        it, which is the fixed point of `tau ds/dt = -s + pre`.  Starting a selection run
        from zeros would begin with the output nucleus silent, i.e. with every channel
        already released, and the first hundred milliseconds of every trace would be the
        pallidum switching on rather than the circuit doing its job.
        """
        device = device or self.prior_w_d1_gpi.device
        pr = self.pr
        one = torch.ones(b, self.n, device=device)
        msn = self._msn_rest()
        st = {"D1": one * msn, "D2": one * msn, "GPe": one * pr.rest_gpe,
              "GPi": one * pr.rest_gpi, "STN": one * pr.rest_stn,
              "Thal": one * pr.rest_thal,
              "sD1": one * msn, "sD2": one * msn,
              "sSG": one * pr.rest_stn, "sSI": one * pr.rest_stn,
              "sGS": one * pr.rest_gpe, "sGG": one * pr.rest_gpe,
              "sGT": one * pr.rest_gpi, "aSTN": one * pr.rest_stn,
              # one (B, K, N) background current, in NOISE_POPS order.  It is a CURRENT,
              # not a rate: it is not in [0, 1] and gate B0 says so rather than quietly
              # testing something it does not bound.
              "eta": torch.zeros(b, len(NOISE_POPS), self.n, device=device)}
        return st

    @staticmethod
    def detach(state):
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in state.items()}

    def rest_rates(self) -> dict:
        """the declared resting rates in Hz, for a gate to compare a measurement against."""
        pr = self.pr
        return {"GPe": pr.rest_gpe * R_MAX, "GPi": pr.rest_gpi * R_MAX,
                "STN": pr.rest_stn * R_MAX, "Thal": pr.rest_thal * R_MAX,
                "D1": self._msn_rest() * R_MAX, "D2": self._msn_rest() * R_MAX}

    # ---- the diffuse efferent ---------------------------------------------------------
    def _diffuse(self, x: torch.Tensor) -> torch.Tensor:
        """the STN's efferent: mostly the channel mean, a little of the channel itself.

        This one line is the whole asymmetry between the hyperdirect pathway and the other
        two.  The direct and indirect arms stay inside their channel at every step; the
        STN's axon does not, so what the pallidum hears from it is dominated by what the
        WHOLE nucleus is doing.  A brake is global because its wiring is.
        """
        f = float(self.pr.stn_diffuse)
        return f * x.mean(dim=-1, keepdim=True).expand_as(x) + (1.0 - f) * x

    def _others_mean(self, x: torch.Tensor) -> torch.Tensor:
        """mean of the OTHER channels: (sum - self) / (n - 1), and 0 when n == 1.

        Written this way rather than as a plain mean because a channel that inhibits
        itself through the competition term is not competing, it is self-limiting, and the
        two are easy to confuse when n is large and the difference is 1/n.
        """
        if self.n < 2:
            return torch.zeros_like(x)
        return (x.sum(dim=-1, keepdim=True) - x) / (self.n - 1)

    # ---- one step ---------------------------------------------------------------------
    def step(self, state, dt: float, ctx=None, stop=None, dopamine: float = 0.5,
             delayed=None, noise: torch.Tensor | None = None, params=None):
        """one exponential-Euler step.

        `ctx`      (B, N) cortical proposal per channel, or a scalar, or None.
        `stop`     (B, N) or scalar: an EXTRA cortical drive that reaches the STN only.
                   It is the hyperdirect pulse, and it is separate from `ctx` because a
                   stop signal is not a proposal -- it never touches the striatum.
        `dopamine` in [0, 1]; `da_tonic` is the level the weights are declared at.
        `delayed`  a dict of the presynaptic rates ALREADY delayed by the caller, keyed
                   ctx_str, ctx_stn, d1, d2, gpe_stn, gpe_gpi, stn_gpe, stn_gpi, gpi.
                   The delays belong to the loop and `rollout` owns them; applying them
                   here as well would apply them twice.  None means zero delay, which is
                   what a boundedness check wants and not what a rhythm wants.
        `noise`    (B, len(NOISE_POPS), N) standard normal, supplied by the caller and
                   NEVER drawn here (CLAUDE.md, "Randomness").
        """
        pr = self.pr
        D1, D2 = state["D1"], state["D2"]
        GPe, GPi, STN, Thal = state["GPe"], state["GPi"], state["STN"], state["Thal"]
        sD1, sD2 = state["sD1"], state["sD2"]
        sSG, sSI, sGS, sGG, sGT = (state["sSG"], state["sSI"], state["sGS"],
                                   state["sGG"], state["sGT"])
        aSTN = state["aSTN"]
        p = params if params is not None else self.params()
        w_ctx_d1, w_ctx_d2 = p["w_ctx_d1"], p["w_ctx_d2"]
        w_d1_gpi, w_d2_gpe = p["w_d1_gpi"], p["w_d2_gpe"]
        w_gpe_stn, w_stn_gpe = p["w_gpe_stn"], p["w_stn_gpe"]
        w_stn_gpi, w_ctx_stn = p["w_stn_gpi"], p["w_ctx_stn"]
        w_gpi_thal = p["w_gpi_thal"]
        i_stn, i_gpe, i_gpi, i_thal = p["i_stn"], p["i_gpe"], p["i_gpi"], p["i_thal"]

        d = delayed or {}
        c_str = d.get("ctx_str", ctx if ctx is not None else 0.0)
        c_stn = d.get("ctx_stn", ctx if ctx is not None else 0.0)
        s_stn = d.get("stop_stn", stop if stop is not None else 0.0)
        p_d1 = d.get("d1", D1)
        p_d2 = d.get("d2", D2)
        p_gpe_stn = d.get("gpe_stn", GPe)
        p_gpe_gpi = d.get("gpe_gpi", GPe)
        p_stn_gpe = d.get("stn_gpe", STN)
        p_stn_gpi = d.get("stn_gpi", STN)
        p_gpi = d.get("gpi", GPi)

        # the background currents, exactly advanced.  One (B, K, N) tensor in NOISE_POPS
        # order, so the six are one multiply and not six.
        rho = math.exp(-dt / pr.tau_eta)
        e = state["eta"] * rho
        if noise is not None:
            e = e + (pr.sigma * math.sqrt(1.0 - rho * rho)) * noise
        eta = {name: e[:, k] for k, name in enumerate(NOISE_POPS)}

        # dopamine: one scalar, two opposite gains, and nothing else in the file reads it
        da = float(dopamine) - pr.da_tonic
        g_d1 = 1.0 + pr.k_da_d1 * da
        g_d2 = 1.0 - pr.k_da_d2 * da

        # ---- striatum: two populations, one competition ------------------------------
        lat1 = self._others_mean(D1)
        lat2 = self._others_mean(D2)
        u_D1 = w_ctx_d1 * g_d1 * c_str - pr.w_msn_lat * lat1 + pr.e_da_d1 * da + eta["D1"]
        u_D2 = w_ctx_d2 * g_d2 * c_str - pr.w_msn_lat * lat2 - pr.e_da_d2 * da + eta["D2"]

        # ---- the STN.  cortical input is the channel MEAN: a stop is not addressed ----
        # Both cortical sources are averaged over channels before they reach the STN.
        # That is the diffuseness on the INPUT side, and it is why the brake cannot be
        # aimed: a pulse into one channel's cortex raises the whole nucleus.
        def _global(x):
            return x.mean(dim=-1, keepdim=True).expand_as(STN) if torch.is_tensor(x) else x
        # `- g_adapt * aSTN` is the slow adaptation: it reads the STN's OWN rate, lagged,
        # so it is a feedback the cell exerts on itself and not a filter of its input.
        # It is what turns a ringing pair into a BURSTING one -- see `tau_adapt`.
        u_STN = (i_stn - w_gpe_stn * sGS + w_ctx_stn * _global(c_stn)
                 + pr.w_stop_stn * _global(s_stn) - pr.g_adapt * aSTN + eta["STN"])

        # ---- the pallidum.  what it hears from the STN is the DIFFUSE efferent --------
        u_GPe = (i_gpe - w_d2_gpe * sD2 + w_stn_gpe * sSG
                 - pr.w_gpe_gpe * GPe + eta["GPe"])
        u_GPi = (i_gpi - w_d1_gpi * sD1 + w_stn_gpi * sSI
                 - pr.w_gpe_gpi * sGG + eta["GPi"])

        # ---- the relay the loop returns through --------------------------------------
        # the corticothalamic drive is applied UNDELAYED, and that is a gap rather than a
        # choice: the catalogue declares no pmc -> va_vl_bg edge, so there is no number to
        # use, and inventing one would put a constant in this file that `ibm/rhythms.py`
        # does not own.  Everything on the pallidothalamic side IS delayed (`sGT` reads
        # `gpi`, which the ring holds for 2-5 ms), which is the edge the loop's timing
        # actually runs through.
        u_Thal = i_thal + pr.w_ctx_thal * (ctx if ctx is not None else 0.0) \
            - w_gpi_thal * sGT + eta["Thal"]

        fD1 = sigmoid(u_D1, pr.beta_msn, pr.theta_msn)
        fD2 = sigmoid(u_D2, pr.beta_msn, pr.theta_msn)
        fGPe = sigmoid(u_GPe, pr.beta_gpe, pr.theta_gpe)
        fGPi = sigmoid(u_GPi, pr.beta_gpi, pr.theta_gpi)
        fSTN = sigmoid(u_STN, pr.beta_stn, pr.theta_stn)
        fThal = sigmoid(u_Thal, pr.beta_thal, pr.theta_thal)

        c_msn = 1.0 - math.exp(-dt / pr.tau_msn)
        c_gpe = 1.0 - math.exp(-dt / pr.tau_gpe)
        c_gpi = 1.0 - math.exp(-dt / pr.tau_gpi)
        c_stn = 1.0 - math.exp(-dt / pr.tau_stn)
        c_thal = 1.0 - math.exp(-dt / pr.tau_thal)
        c_str_syn = 1.0 - math.exp(-dt / pr.tau_gaba_str)
        c_ampa = 1.0 - math.exp(-dt / pr.tau_ampa_stn)
        c_pal = 1.0 - math.exp(-dt / pr.tau_gaba_pal)
        c_out = 1.0 - math.exp(-dt / pr.tau_gaba_gpi)
        c_ad = 1.0 - math.exp(-dt / pr.tau_adapt)

        out = {"D1": D1 + c_msn * (fD1 - D1), "D2": D2 + c_msn * (fD2 - D2),
               "GPe": GPe + c_gpe * (fGPe - GPe), "GPi": GPi + c_gpi * (fGPi - GPi),
               "STN": STN + c_stn * (fSTN - STN), "Thal": Thal + c_thal * (fThal - Thal),
               "sD1": sD1 + c_str_syn * (p_d1 - sD1),
               "sD2": sD2 + c_str_syn * (p_d2 - sD2),
               "sSG": sSG + c_ampa * (self._diffuse(p_stn_gpe) - sSG),
               "sSI": sSI + c_ampa * (self._diffuse(p_stn_gpi) - sSI),
               "sGS": sGS + c_pal * (p_gpe_stn - sGS),
               "sGG": sGG + c_pal * (p_gpe_gpi - sGG),
               "sGT": sGT + c_out * (p_gpi - sGT),
               "aSTN": aSTN + c_ad * (STN - aSTN), "eta": e}
        return out

    # ---- the loop ---------------------------------------------------------------------
    def delays_in_steps(self, dt: float) -> dict:
        """every declared conduction delay in STEPS at this dt.

        Reported, never assumed.  A dt too coarse to resolve a delay collapses it to zero
        and the loop silently loses its conduction budget; this is how that becomes
        visible (`ThalamoCortical.rollout` does the same thing for its two).
        """
        pr = self.pr
        return {k[2:-2]: int(round(float(getattr(pr, k)) / dt))
                for k in pr.__dataclass_fields__ if k.startswith("d_")}

    def rollout(self, steps: int, dt: float, ctx=None, stop=None, dopamine: float = 0.5,
                state=None, noise_gen: torch.Generator | None = None, b: int = 1,
                record=("GPi", "STN", "GPe", "Thal", "D1", "D2"), burn: int = 0):
        """run the three pathways with the catalogue's delays as ring buffers.

        `ctx` and `stop` may be a scalar, a (B, N) tensor held constant, or a
        (B, steps, N) tensor.  Returns `(traces, state, info)` with `traces` a dict of
        (B, steps, N) and `info` carrying `delays_in_steps`.

        `burn` steps are run and discarded.  The rings are carried across the boundary, so
        it is a run-up and not a discarded restart.

        The noise is drawn ONCE per step from the caller's generator, here, outside the
        step -- the callee asserts it was given a draw rather than making its own
        (CLAUDE.md, "Randomness").  `noise_gen=None` means a noiseless run, which is a
        different experiment and not a defaulted one.
        """
        assert noise_gen is None or isinstance(noise_gen, torch.Generator), \
            "pass a torch.Generator or None; this module never touches the global RNG"
        dev = self.prior_w_d1_gpi.device
        state = state or self.init_state(b, device=dev)
        pr = self.pr
        lag = self.delays_in_steps(dt)
        # one ring per EDGE, not per population: two edges out of the same nucleus have
        # different axon lengths and a shared ring would quietly give them the shorter one
        rings, routes = {}, {
            "ctx_str": ("ctx", "ctx_str_s"), "ctx_stn": ("ctx", "ctx_stn_s"),
            "stop_stn": ("ctx", "ctx_stn_s"),
            "d1": ("D1", "str_gpi_s"), "d2": ("D2", "str_gpe_s"),
            "gpe_stn": ("GPe", "gpe_stn_s"), "gpe_gpi": ("GPe", "gpe_gpi_s"),
            "stn_gpe": ("STN", "stn_gpe_s"), "stn_gpi": ("STN", "stn_gpi_s"),
            "gpi": ("GPi", "gpi_thal_s"), "ctx_return": ("Thal", "thal_ctx_s"),
        }
        lag_of = {k: lag[v[1][:-2]] for k, v in routes.items()}
        for k, (src, _f) in routes.items():
            fill = 0.0 if src == "ctx" else float(
                {"D1": self._msn_rest(), "D2": self._msn_rest(), "GPe": pr.rest_gpe,
                 "GPi": pr.rest_gpi, "STN": pr.rest_stn, "Thal": pr.rest_thal}[src])
            rings[k] = torch.full((b, max(1, lag_of[k] + 1), self.n), fill, device=dev)

        def slice_t(x, t):
            if torch.is_tensor(x) and x.dim() == 3:
                return x[:, min(max(0, t - burn), x.shape[1] - 1)]
            return x

        out = {k: [] for k in record}
        out["ctx_return"] = []
        pars = self.params()
        for t in range(steps + burn):
            z = None
            if noise_gen is not None:
                z = torch.randn(b, len(NOISE_POPS), self.n, generator=noise_gen,
                                device="cpu").to(dev)
            delayed = {k: rings[k][:, t % rings[k].shape[1]].clone() for k in routes}
            ctx_return = delayed.pop("ctx_return")
            c_t, s_t = slice_t(ctx, t), slice_t(stop, t)
            # the cortical rings carry the drive itself, so its routes -- into the striatum
            # at 3-8 ms and into the STN at 2-5 ms -- arrive at their own declared times.
            # The STOP gets its OWN ring on the same 2-5 ms hyperdirect tract rather than
            # riding the proposal's, because the two are multiplied by different weights
            # (`w_stop_stn`, `w_ctx_stn`) inside `step`.  `step`'s own `stop` argument is
            # for a direct single-step call, and rollout passes None there so the pulse is
            # never applied twice, once delayed and once not.
            for k, src in (("ctx_str", c_t), ("ctx_stn", c_t), ("stop_stn", s_t)):
                v = src if torch.is_tensor(src) else torch.full(
                    (b, self.n), 0.0 if src is None else float(src), device=dev)
                if v.dim() == 1:
                    v = v.unsqueeze(0).expand(b, self.n)
                rings[k][:, (t + lag_of[k]) % rings[k].shape[1]] = v
            state = self.step(state, dt, ctx=c_t, stop=None, dopamine=dopamine,
                              delayed=delayed, noise=z, params=pars)
            for k, (src, _f) in routes.items():
                if src == "ctx":
                    continue
                rings[k][:, (t + lag_of[k]) % rings[k].shape[1]] = state[src]
            if t >= burn:
                for k in record:
                    out[k].append(state[k])
                out["ctx_return"].append(ctx_return)
        traces = {k: torch.stack(v, 1) for k, v in out.items() if v}
        return traces, state, {"delays_in_steps": lag, "lag_of_edge": lag_of,
                               "dopamine": float(dopamine), "n_channels": self.n}

    def rate_hz(self, x):
        return x * R_MAX
