"""basal ganglia: D1/D2 striatum, GPe, GPi/SNr and STN, in four parallel channels.

`docs/BRAIN_SPEC.md`, "Basal ganglia": *D1 and D2 striatum, GPe, GPi/SNr, STN, SNc, in
parallel channels: motor, oculomotor, associative, limbic.  Direct, indirect and
hyperdirect pathways, each with its own delay.  Tonically active output nuclei that pause
to release a channel.  Dopamine from SNc/VTA sets the D1/D2 balance and is produced by the
valuation system, not passed in.*

WHAT IS DIFFERENT FROM `ibm/basal_ganglia.py` (v2), WHICH IS THE KNOWN ANSWER
----------------------------------------------------------------------------
v2 passes eight gates and every number here is meant to be read beside its number.  Three
things change, and each is a consequence of the v3 contract rather than a preference:

1. **Four named channels, not eight abstract ones.**  v2's `n = 8` channels were
   interchangeable slots.  The four here are the loops the anatomy actually runs, and each
   one names the cortical area it comes from and the thalamic nucleus it returns through.
   A channel is no longer a number, it is a pathway with endpoints -- which is also what
   makes the corticostriatal edges declarable at all.

2. **There is no `dopamine` float.**  v2 took `dopamine` as an argument to `step`.  Here
   dopamine arrives as `Mod` edges from `nm.snc` (dorsal striatum: motor, oculomotor,
   associative) and `nm.vta` (ventral striatum: limbic), and those nuclei belong to
   `ibm/brain/neuromodulators.py`.  This module cannot set its own dopamine level and has
   no branch on one.  That is the entire point of the package -- `ibm/brain/__init__.py`'s
   `NM` table exists so there is one substantia nigra in the brain and not one per module
   that wanted a knob.

3. **The synaptic time constants are gone, folded into the membrane ones, and the beta
   band is worse for it.**  v2 carried seven synaptic activation variables with their own
   transmitter kinetics (`tau_ampa_stn`, `tau_gaba_pal`, ...) and its docstring shows the
   phase condition those produce: ~18 Hz.  `ibm/circuit.py` has no synaptic state -- a
   `Proj` delivers its source's rate through a pure conduction delay and nothing else --
   so the only low-pass filters in the STN-GPe ring are the two membrane time constants.
   With the literature's membrane values alone (5 and 6 ms) against the catalogue's 7 ms
   round trip the phase condition

       2*pi*f*D + atan(2*pi*f*tau_stn) + atan(2*pi*f*tau_gpe) = pi

   crosses at **33 Hz**, out of band.  So `TAU_STN` and `TAU_GPE` here are declared as
   EFFECTIVE constants, membrane plus the dominant synapse each cell receives inside the
   ring -- 5 + 9 ms of GABA-A onto the STN, 6 + 4 ms of AMPA onto the GPe -- and the split
   is written into the constant's note so nobody reads 14 ms as a measured subthalamic
   membrane time constant.  It is a lumping, it is visible, and the engine feature that
   would remove it (a synaptic time constant per projection) is reported rather than added.

THE THREE PATHWAYS, AND WHY EACH DELAY IS WHERE IT IS
-----------------------------------------------------
    direct       ctx -> D1 -| GPi                  a proposal REMOVES inhibition
    indirect     ctx -> D2 -| GPe -| STN -> GPi    a proposal ADDS it, one synapse later
    hyperdirect  ctx -> STN -> GPi                 straight past the striatum

Every delay is the midpoint of the interval `ibm/rhythms.py` declares for that edge, and
the interval is quoted beside it.  Counted out of cortex: hyperdirect reaches GPi in
3.5 + 3.0 = **6.5 ms**, direct in 5.5 + 4.5 = **10.0 ms**, indirect in 5.5 + 4.5 + 4.0 +
3.0 = **17.0 ms**.  That ordering is the hyperdirect pathway's whole meaning and it is a
property of the declared conduction budget, not of a weight.

**OUTPUT NUCLEI ARE TONICALLY ACTIVE AND SELECTION IS A PAUSE IN THEM.**  GPi/SNr rests at
70 Hz, GPe at 60, STN at 25 (DeLong 1971; Wichmann & DeLong 1996), declared as `Pop.r_rest`
and held there by a tonic drive that `drives()` SOLVES from the declared rates rather than
guessing: at rest every projection into a nucleus is cancelled by that current, so the net
input is zero and `r_rest` is where the population sits.  Change any weight and the resting
state stays where the recording put it, which is the only way gate B7's sweep says anything.

A CONSEQUENCE WORTH STATING, because v2 records it as an inert-by-construction finding:
`theta` for GPe, GPi and STN is **derived**, `-logit(r_rest)/beta`, not declared.  That
choice makes `sig(0) = r_rest`, so the sigmoid's whole [0, 1] range is available around the
resting rate -- without it the engine's `r_rest` branch gives a population resting at 70 Hz
a floor of `r_rest - sig(0)`, and a GPi that can only fall to 56 Hz cannot pause.  It also
removes three constants that v2's sweep found inert to five decimals for exactly this
reason: they were a reparametrisation of the tonic solve.  `theta_msn` is a real lever and
stays declared, because the striatum's resting rate is solved FROM it.

**THE STN IS THE DIFFUSE ONE.**  A subthalamic axon arborises across the pallidum rather
than staying in its channel (Parent & Hazrati 1995), while the striatopallidal and
pallidosubthalamic projections are topographic.  So each channel's STN projects to EVERY
channel's GPe and GPi: `stn_diffuse` of the efferent is spread evenly over the four and
`1 - stn_diffuse` stays at home.  Two consequences fall out of that one constant:

  * the uniform mode of the STN-GPe ring keeps the full loop gain (the weights over the
    four targets sum to one) while every differential mode is reduced by `1 - stn_diffuse`,
    so the beta the pair carries is a SHARED nucleus rhythm and the channels stay separable;
  * anything that raises one STN raises every GPi, which is what makes the hyperdirect
    pathway a **global brake**.  Set `stn_diffuse` to 0 and it becomes per-channel and
    stops being a brake; gate B7 sweeps it and B6 is what it would break.

The stop signal comes down the same tract from a different cortical population --
`ctx.parsopercularis.E`, right inferior frontal gyrus, the stopping network (Aron &
Poldrack 2006) -- and it reaches ALL FOUR channels' STN, because a stop is not addressed to
a channel.  A channel's own proposal also has a hyperdirect collateral, onto its own STN
and much weaker: a proposal that drove the STN as hard as a stop does would brake itself,
which v2 measured directly (at `w_ctx_stn = 1.5` the winner's GPi never leaves its resting
rate).

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No dopaminergic populations: `nm.snc` and `nm.vta` are `ibm/brain/neuromodulators.py`'s and
this module only names them.  No cortical populations and no cortical return leg: the
corticostriatal, corticosubthalamic and pallidothalamic edges are declared here (the
pathway is named for the basal ganglia), and the thalamocortical return belongs to
`ibm/brain/thalamus.py`.  No striatal fast-spiking interneuron population -- the structure
list in the specification does not have one, so the competition between channels is written
as direct cross-channel inhibition standing for that pool, exactly as v2 wrote it, and the
note on `w_str_lat` says what the difference is.  No declared `Pop.sparsity` anywhere: a
pallidal segment is not a sparse code, it is a tonic pacemaker whose LEVEL is the signal,
and the engine's divisive normaliser would fight the very pause this module exists to make.

UNITS: seconds, rates dimensionless in [0, 1] with `R_MAX` = 100 Hz for readouts, delays in
seconds (the catalogue declares milliseconds).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

from ibm.brain import NM, Mod, Pop, Proj  # noqa: F401

STRUCTURE = "bg"

#: the four loops the anatomy runs.  Population ids are `bg.<nucleus>.<channel>`: the third
#: field is the CHANNEL here rather than a cell class, because in this structure the
#: nucleus name already carries the class (`d1` and `d2` are the two striatal cell types)
#: and the channel is the thing there are four of.  Said here because the contract's
#: example is `bg.str.d1` and this is a deliberate departure from it.
CHANNELS = ("motor", "oculomotor", "associative", "limbic")

#: where each channel's proposal comes from.  `ctx.<area>.E` are `ibm/brain/cortex.py`'s
#: ids; all four areas are in its `AREAS` tuple, so these edges are kept at assembly rather
#: than orphaned.  caudalmiddlefrontal carries the frontal eye field, rostralmiddlefrontal
#: the dorsolateral prefrontal cortex, medialorbitofrontal the limbic/valuation frontal.
CTX_SOURCE = {
    "motor": "ctx.precentral.E",
    "oculomotor": "ctx.caudalmiddlefrontal.E",
    "associative": "ctx.rostralmiddlefrontal.E",
    "limbic": "ctx.medialorbitofrontal.E",
}

#: the stopping network's hyperdirect source.  A different cortical population from any
#: proposal's, projecting down the same tract, and it reaches every channel.
CTX_STOP = "ctx.parsopercularis.E"

#: the pallidal-recipient thalamic relay each channel returns through.  The pallidothalamic
#: synapse is onto the RELAY cells, so the id is `thal.<nucleus>.relay` and not the nucleus
#: on its own: VL for the motor loop, VA for the other three.  If `ibm/brain/thalamus.py`
#: is missing, or names its relays something else, `collect()` reports these as orphans --
#: which is the right outcome, because an edge that quietly disappears is a loop that
#: quietly stops existing.
THAL_TARGET = {
    "motor": "thal.vl.relay",
    "oculomotor": "thal.va.relay",
    "associative": "thal.va.relay",
    "limbic": "thal.va.relay",
}

#: which dopaminergic cell group reaches each channel's striatum.  Dorsal striatum (motor,
#: oculomotor, associative) from the substantia nigra pars compacta; ventral striatum
#: (limbic) from the ventral tegmental area.  Both are `NM` ids and neither is declared
#: here.
DA_SOURCE = {
    "motor": NM["da_snc"],
    "oculomotor": NM["da_snc"],
    "associative": NM["da_snc"],
    "limbic": NM["da_vta"],
}

N_UNITS = 16       # units per population.  Every projection here is population-to-
                   # population (`dense`, i.e. the source's mean), so the units inside a
                   # channel differ only by their own background current: `n` buys
                   # independent noise, not independent computation.  Said plainly because
                   # a reader could otherwise take 16 units for 16 neurons doing 16 things.

R_MAX = 100.0


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# ======================================================================================
# the declared constants.  every one carries its reason, and gate B7 sweeps all of them.
# ======================================================================================
@dataclass(frozen=True)
class Constants:
    # ---- the resting state, declared as RATES because that is what is recorded ---------
    # Awake primate single units (DeLong 1971; Wichmann & DeLong 1996; Nambu 2000).  The
    # tonic current that holds each population here is SOLVED from these in `drives()`.
    rest_gpe: float = 0.60
    rest_gpi: float = 0.70
    rest_stn: float = 0.25

    # ---- transfer functions -----------------------------------------------------------
    # `theta` is DERIVED for the three tonically active nuclei (see the module docstring);
    # only the slopes are declared for them, and they are real levers because they set the
    # GAIN at the fixed point.  `theta_msn` IS declared, because the striatum's resting
    # rate is solved from it rather than declared beside it.
    beta_msn: float = 10.0
    theta_msn: float = 0.55        # MSNs sit in a deep down-state and are silent until a
                                   # coherent cortical input arrives; that is what lets the
                                   # striatum detect proposals instead of relaying them
    beta_gpe: float = 6.0
    beta_gpi: float = 6.0
    beta_stn: float = 6.0

    # ---- time constants.  EFFECTIVE: membrane + the dominant synapse in the ring -------
    # `ibm/circuit.py` has no synaptic state variable, so a transmitter's kinetics can only
    # be carried by the membrane constant of the cell that receives it.  The split is
    # quoted for each and the module docstring says what is lost by lumping them.
    tau_msn: float = 0.015         # MSN membrane 10 ms + corticostriatal AMPA ~5 ms
    tau_gpe: float = 0.010         # GPe membrane 6 ms + subthalamopallidal AMPA 4 ms
    tau_gpi: float = 0.012         # GPi membrane 6 ms + the GABA-A it lives under ~6 ms
    tau_stn: float = 0.014         # STN membrane 5 ms + pallidosubthalamic GABA-A 9 ms.
                                   # This one and tau_gpe are the pair's clock: with the
                                   # catalogue's 7 ms round trip the phase condition crosses
                                   # at ~23.5 Hz, in band.  B7 requires each to move it.

    # ---- weights ----------------------------------------------------------------------
    w_ctx_d1: float = 1.20         # corticostriatal onto the direct-pathway MSNs
    w_ctx_d2: float = 1.20         # and onto the indirect-pathway MSNs, EQUAL at tonic
                                   # dopamine: the asymmetry between the arms is dopamine's
                                   # job, not a wiring difference, or dopamine would be a
                                   # flag with the wiring doing its work
    # Competition between channels.  It stands for the striatal fast-spiking interneuron
    # pool, which is powerful and divergent and is what actually makes the striatum
    # competitive (Gittis & Kreitzer 2012; Tepper 2008), rather than for the weak MSN-MSN
    # collateral, which is far too sparse for a winner-take-all.  Be precise about what
    # the code does: there is no interneuron population here, so the term is written as
    # inhibition from the OTHER channels' MSNs of the same pathway, which is RECURRENT and
    # not feedforward.  v2's note on the same term works out the difference -- a recurrent
    # term also attenuates the striatum's uniform mode, a feedforward one would not, and a
    # feedforward version gives a far weaker winner.  The total is `w_str_lat`, split
    # evenly over the three other channels.
    w_str_lat: float = 3.00
    w_d1_gpi: float = 1.60         # the direct arm: the inhibition whose withdrawal IS the
                                   # selection
    w_d2_gpe: float = 1.40         # the indirect arm's first inhibitory step
    # The STN-GPe pair's loop gain.  The one pair of constants here that a MEASUREMENT
    # sets rather than the literature, and it is set to put the ring JUST BELOW its Hopf
    # bifurcation: above it the pair is a limit cycle, i.e. a constant beta tone, which is
    # the object `ibm/rhythms.py`'s `beta_bursts` row exists to exclude; just below it the
    # pair is a damped resonance that rings for a few cycles whenever the background
    # current knocks it.  The linear estimate is |L| = w^2 * g_stn * g_gpe / |1 + i w t|^2
    # at the phase crossing, which gives w ~ 1.55 for |L| = 0.95.  Measured
    # (`scripts/gate_brain_basal_ganglia.py --sweep`, 4 s at rest, batch of 16 background
    # realisations, channel-averaged STN), with the whole curve recorded including the part
    # past the bifurcation:
    #
    #   w     peak Hz   prominence   envelope CV   STN rate
    #   1.10   18.54      +0.688        0.511      25.31 +- 0.44 Hz
    #   1.25   20.12      +0.830        0.492      25.28 +- 0.48
    #   1.40   21.74      +1.011        0.483      25.25 +- 0.56
    #   1.55   23.04      +1.284        0.528      25.24 +- 0.73      <- declared
    #   1.70   23.98      +1.966        0.672      25.26 +- 1.57
    #   1.85   23.93      +3.217        0.066      26.25 +- 6.59
    #   2.00   23.61      +3.473        0.029      27.55 +- 9.48
    #
    # The bifurcation is between 1.70 and 1.85: above it the envelope's coefficient of
    # variation collapses by a factor of ten and the STN's standard deviation goes from
    # under a hertz to six, which is a LIMIT CYCLE -- a constant beta tone, the object
    # `beta_bursts` exists to exclude.  1.55 is declared rather than 1.70 because 1.70's
    # envelope CV has already started to climb away from the 0.523 a linear resonator driven
    # by noise gives, i.e. it is on its way into the cycle; 1.55 measures 0.528, which is
    # that value.  The linear estimate and the measurement agree to within one step of the
    # sweep, which is the part worth keeping: the constant was derived and then checked, not
    # searched for.
    w_gpe_stn: float = 1.55        # pallidosubthalamic, topographic
    w_stn_gpe: float = 1.55        # subthalamopallidal, diffuse
    w_stn_gpi: float = 2.20        # the brake's grip on the output nucleus.  LARGER than
                                   # w_d1_gpi, and that inequality is what lets a stop
                                   # cancel a selection already granted rather than dent it
    w_gpe_gpi: float = 0.50        # the pallidopallidal projection.  Anatomically real
                                   # (Smith 1998) and NOT in the catalogue, which has no
                                   # gpe->gpi edge; flagged here rather than smuggled in,
                                   # and its delay is flagged the same way
    w_gpe_gpe: float = 0.30        # GPe onto itself
    w_ctx_stn: float = 0.90        # a proposal's own hyperdirect collateral: the WEAK one
    w_stop_stn: float = 2.00       # the stopping network's, onto every channel: the strong
                                   # one.  A brake that cannot reach the top of the STN's
                                   # range cannot cancel a selection already granted
    w_gpi_thal: float = 1.20       # the pallidothalamic inhibition the relay lives under

    # ---- the STN's diffuseness ---------------------------------------------------------
    # fraction of each STN channel's efferent that is spread evenly over all four channels
    # rather than staying at home.  0.8, not 1.0: the arbor is wide but not uniform, and a
    # completely shared STN would make the INDIRECT pathway global too, which would destroy
    # the selection it is supposed to sharpen.
    stn_diffuse: float = 0.80

    # ---- dopamine, as it arrives -------------------------------------------------------
    # There is no dopamine level here.  These are the properties of the two `Mod` edges per
    # striatal population: the rate of `nm.snc`/`nm.vta` at which the declared weights are
    # stated (`da_baseline`), and how hard that rate acts on the corticostriatal GAIN and
    # on the MSN's own EXCITABILITY.  BOTH terms are needed and v2 measured why: with the
    # gain term alone dopamine multiplies a cortical drive that is zero during quiet rest
    # and therefore changes nothing there -- its resting beta spectrum was bit-identical at
    # dopamine 0.5 and 0.2.  The excitability term (a threshold shift) is the one that
    # makes a depleted striatum tonically over-drive the indirect arm at rest.
    da_baseline: float = 0.50
    k_da_gain: float = 1.20        # on `gain_in`: + for D1, - for D2.  v2's `k_da_d1` and
                                   # `k_da_d2`, which are the same number for the same
                                   # reason, carried over unchanged
    k_da_theta: float = 2.18       # on `theta`: - for D1 (more dopamine, more excitable),
                                   # + for D2.  NOT a free number: v2 declares the same
                                   # excitability as a CURRENT shift, `e_da_d2 = 1.20`, so
                                   # at a dopamine of 0.2 it moves the MSN's operating point
                                   # by 1.20 * 0.3 = 0.36.  A `Mod` edge can only multiply a
                                   # parameter, so the equivalent threshold change is
                                   # 0.55 -> 0.55 - 0.36 = 0.19, a factor 0.345, which needs
                                   # gain (1 - 0.345)/0.3 = 2.18.  Measured at 0.35 first --
                                   # the same mechanism, a sixth of the size -- and it moved
                                   # the resting beta prominence by +0.008 decades, which is
                                   # the "parameter that changes nothing" shape one step
                                   # before it bites: real mechanism, wrong magnitude, and
                                   # it would have been read as a weak Parkinsonian effect.
                                   # `Mod`'s lo/hi clamp keeps the factor in [0.25, 4.0], so
                                   # a nigra at zero silences D1 rather than inverting it

    # ---- background --------------------------------------------------------------------
    # The OU current that knocks the subcritical STN-GPe pair into ringing.  `tau_eta` is
    # 5 ms so the drive is broadband well past beta: a background whose corner sat inside
    # the band would be manufacturing the rhythm it is supposed to reveal.
    sigma: float = 0.05
    tau_eta: float = 0.005

    # ---- conduction, from ibm/rhythms.py ----------------------------------------------
    # Each is the MIDPOINT of the catalogue's declared (lo, hi) interval, quoted beside it.
    d_ctx_str: float = 0.0055      # cbgtc / bg_indirect   pmc->striatum    (3, 8) ms
    d_ctx_stn: float = 0.0035      # bg_hyperdirect        pmc->stn         (2, 5) ms
    d_str_gpi: float = 0.0045      # cbgtc                 striatum->gpi    (3, 6) ms
    d_str_gpe: float = 0.0045      # bg_indirect           striatum->gpe    (3, 6) ms
    d_gpe_stn: float = 0.0040      # stn_gpe / bg_indirect gpe->stn         (3, 5) ms
    d_stn_gpe: float = 0.0030      # stn_gpe               stn->gpe         (2, 4) ms
    d_stn_gpi: float = 0.0030      # bg_indirect / hyper   stn->gpi         (2, 4) ms
    d_gpi_thal: float = 0.0035     # cbgtc                 gpi->va_vl_bg    (2, 5) ms
    d_gpe_gpi: float = 0.0030      # NOT IN THE CATALOGUE. Taken as the intrapallidal
                                   # distance, shorter than gpe->stn.  Declared so the
                                   # sweep can see it.
    d_str_lat: float = 0.0010      # NOT IN THE CATALOGUE. Local striatal inhibition.
    d_gpe_gpe: float = 0.0010      # NOT IN THE CATALOGUE. Local pallidal inhibition.


C = Constants()


def constant_names() -> list:
    """every declared constant, for a sensitivity sweep to walk."""
    return [f.name for f in fields(Constants)]


# ======================================================================================
# derived quantities: the things that are a consequence of the constants, not choices
# ======================================================================================
def theta_of(rest: float, beta: float) -> float:
    """the threshold that makes `sig(0)` equal the resting rate.

    `ibm/circuit.py` gives a population with `r_rest` the target
    `r_rest + sig(u) - sig(0)`, so the range below rest is exactly `sig(0)` wide.  With a
    declared threshold of 0.30 and a slope of 6, `sig(0)` is 0.142 -- a GPi resting at
    70 Hz could then fall no lower than 56 Hz, and selection is a PAUSE, so that is not a
    detail.  Setting `theta = -logit(r_rest)/beta` makes `sig(0) = r_rest`, the target
    becomes `sig(u)` exactly, and the whole [0, 1] range is available around rest.
    """
    return -_logit(rest) / beta


def msn_rest(c: Constants = None) -> float:
    """the striatal resting rate: the fixed point with no cortical drive at all.

    Solved rather than declared -- with zero drive an MSN's rate is whatever its threshold
    and the lateral inhibition from the other channels leave it -- because it enters the
    tonic solve below.  Four iterations is plenty: the value is ~0.004 and the lateral term
    is a hundredth of the threshold.
    """
    c = c or C
    r = 1.0 / (1.0 + math.exp(-c.beta_msn * (0.0 - c.theta_msn)))
    for _ in range(4):
        r = 1.0 / (1.0 + math.exp(-c.beta_msn * (-c.w_str_lat * r - c.theta_msn)))
    return r


def stn_split(c: Constants = None) -> tuple:
    """(own, cross): how one STN channel's efferent divides over the four pallidal sectors.

    `own + (K - 1) * cross == 1` by construction, which is the property that keeps the
    ring's UNIFORM mode at full loop gain while damping every differential mode by
    `1 - stn_diffuse`.  A diffuse efferent must redistribute an axon, not multiply it.
    """
    c = c or C
    k = len(CHANNELS)
    f = float(c.stn_diffuse)
    return (1.0 - f) + f / k, f / k


def rest_rates(c: Constants = None) -> dict:
    """the declared resting rates in Hz, for a gate to compare a measurement against."""
    c = c or C
    m = msn_rest(c)
    out = {}
    for ch in CHANNELS:
        out[f"{STRUCTURE}.gpe.{ch}"] = c.rest_gpe * R_MAX
        out[f"{STRUCTURE}.gpi.{ch}"] = c.rest_gpi * R_MAX
        out[f"{STRUCTURE}.stn.{ch}"] = c.rest_stn * R_MAX
        out[f"{STRUCTURE}.d1.{ch}"] = m * R_MAX
        out[f"{STRUCTURE}.d2.{ch}"] = m * R_MAX
    return out


# ======================================================================================
# the contract
# ======================================================================================
def pops(c: Constants = None):
    c = c or C
    out = []
    for ch in CHANNELS:
        # the striatum.  No r_rest: an MSN's silence is a consequence of its threshold, so
        # `theta_msn` is a lever and the rest rate is solved from it.
        for part, note in (("d1", "direct-pathway MSNs; their withdrawal of inhibition "
                                  "from GPi IS the selection"),
                           ("d2", "indirect-pathway MSNs; they inhibit GPe and so release "
                                  "the STN")):
            out.append(Pop(
                id=f"{STRUCTURE}.{part}.{ch}", n=N_UNITS, kind="I",
                tau=c.tau_msn, beta=c.beta_msn, theta=c.theta_msn,
                sigma=c.sigma, tau_eta=c.tau_eta,
                note=f"{ch} channel: {note}"))
        out.append(Pop(
            id=f"{STRUCTURE}.gpe.{ch}", n=N_UNITS, kind="I",
            tau=c.tau_gpe, beta=c.beta_gpe, theta=theta_of(c.rest_gpe, c.beta_gpe),
            r_rest=c.rest_gpe, sigma=c.sigma, tau_eta=c.tau_eta,
            note=f"{ch} channel: external pallidum, autonomous at 60 Hz"))
        out.append(Pop(
            id=f"{STRUCTURE}.gpi.{ch}", n=N_UNITS, kind="I",
            tau=c.tau_gpi, beta=c.beta_gpi, theta=theta_of(c.rest_gpi, c.beta_gpi),
            r_rest=c.rest_gpi, sigma=c.sigma, tau_eta=c.tau_eta,
            note=f"{ch} channel: the OUTPUT nucleus, autonomous at 70 Hz.  A movement is "
                 f"released when this PAUSES, never when something switches on"))
        out.append(Pop(
            id=f"{STRUCTURE}.stn.{ch}", n=N_UNITS, kind="E",
            tau=c.tau_stn, beta=c.beta_stn, theta=theta_of(c.rest_stn, c.beta_stn),
            r_rest=c.rest_stn, sigma=c.sigma, tau_eta=c.tau_eta,
            note=f"{ch} sector of the subthalamic nucleus, autonomous at 25 Hz; its "
                 f"efferent is the one thing here that leaves its channel"))
    return out


def internal(c: Constants = None):
    c = c or C
    out = []
    own, cross = stn_split(c)
    k = len(CHANNELS)
    for ch in CHANNELS:
        d1, d2 = f"{STRUCTURE}.d1.{ch}", f"{STRUCTURE}.d2.{ch}"
        gpe, gpi = f"{STRUCTURE}.gpe.{ch}", f"{STRUCTURE}.gpi.{ch}"
        stn = f"{STRUCTURE}.stn.{ch}"
        # --- direct: one synapse from striatum to the output nucleus
        out.append(Proj(d1, gpi, weight=c.w_d1_gpi, sign=-1, delay_s=c.d_str_gpi,
                        note="direct arm; catalogue striatum->gpi (3, 6) ms"))
        # --- indirect: three synapses, and it is slower for that reason and no other
        out.append(Proj(d2, gpe, weight=c.w_d2_gpe, sign=-1, delay_s=c.d_str_gpe,
                        note="indirect arm, first step; catalogue striatum->gpe (3, 6) ms"))
        out.append(Proj(gpe, stn, weight=c.w_gpe_stn, sign=-1, delay_s=c.d_gpe_stn,
                        note="pallidosubthalamic, topographic; releasing the STN is the "
                             "point of this arm.  catalogue gpe->stn (3, 5) ms"))
        # --- the pallidum's own inhibition
        out.append(Proj(gpe, gpi, weight=c.w_gpe_gpi, sign=-1, delay_s=c.d_gpe_gpi,
                        note="pallidopallidal (Smith 1998); NOT IN THE CATALOGUE, delay "
                             "taken as the intrapallidal distance"))
        out.append(Proj(gpe, gpe, weight=c.w_gpe_gpe, sign=-1, delay_s=c.d_gpe_gpe,
                        note="the external pallidum's own inhibition; not in the catalogue"))
        # --- the striatal competition, standing for the fast-spiking interneuron pool
        for other in CHANNELS:
            if other == ch:
                continue
            for part in ("d1", "d2"):
                out.append(Proj(f"{STRUCTURE}.{part}.{ch}", f"{STRUCTURE}.{part}.{other}",
                                weight=c.w_str_lat / (k - 1), sign=-1,
                                delay_s=c.d_str_lat,
                                note="striatal competition between channels; stands for "
                                     "the FSI pool, not the MSN-MSN collateral"))
        # --- the diffuse subthalamic efferent: to EVERY channel's pallidum
        for other in CHANNELS:
            w = own if other == ch else cross
            out.append(Proj(stn, f"{STRUCTURE}.gpe.{other}", weight=c.w_stn_gpe * w,
                            delay_s=c.d_stn_gpe,
                            note=f"subthalamopallidal, {'own' if other == ch else 'diffuse'}"
                                 f"; the other half of the beta ring.  catalogue (2, 4) ms"))
            out.append(Proj(stn, f"{STRUCTURE}.gpi.{other}", weight=c.w_stn_gpi * w,
                            delay_s=c.d_stn_gpi,
                            note=f"subthalamo-nigral/pallidal, "
                                 f"{'own' if other == ch else 'diffuse'}; this is what "
                                 f"makes the brake global.  catalogue (2, 4) ms"))
    return out


def external(c: Constants = None):
    """the corticostriatal, corticosubthalamic and pallidothalamic projections.

    Declared here because the pathway is named for the basal ganglia (`ibm/brain/
    __init__.py`'s contract).  The thalamocortical return leg that closes `cbgtc` is
    `ibm/brain/thalamus.py`'s and is deliberately not here; if that module is missing,
    these pallidothalamic edges come back as orphans, which is the correct report.
    """
    c = c or C
    out = []
    for ch in CHANNELS:
        src = CTX_SOURCE[ch]
        out.append(Proj(src, f"{STRUCTURE}.d1.{ch}", weight=c.w_ctx_d1,
                        delay_s=c.d_ctx_str,
                        note="corticostriatal onto direct-pathway MSNs; catalogue "
                             "pmc->striatum (3, 8) ms"))
        out.append(Proj(src, f"{STRUCTURE}.d2.{ch}", weight=c.w_ctx_d2,
                        delay_s=c.d_ctx_str,
                        note="corticostriatal onto indirect-pathway MSNs; same tract"))
        out.append(Proj(src, f"{STRUCTURE}.stn.{ch}", weight=c.w_ctx_stn,
                        delay_s=c.d_ctx_stn,
                        note="the proposal's own hyperdirect collateral, onto its own "
                             "channel and WEAK; catalogue pmc->stn (2, 5) ms"))
        # the stopping network, onto every channel: a stop is not addressed to a channel
        out.append(Proj(CTX_STOP, f"{STRUCTURE}.stn.{ch}", weight=c.w_stop_stn,
                        delay_s=c.d_ctx_stn,
                        note="right inferior frontal stopping network down the same "
                             "hyperdirect tract, onto EVERY channel (Aron & Poldrack 2006)"))
        out.append(Proj(f"{STRUCTURE}.gpi.{ch}", THAL_TARGET[ch], weight=c.w_gpi_thal,
                        sign=-1, delay_s=c.d_gpi_thal,
                        note="pallidothalamic; disinhibition is the release.  catalogue "
                             "gpi->va_vl_bg (2, 5) ms"))
    return out


def mods(c: Constants = None):
    """dopamine, arriving from the cell groups that produce it.

    Two edges per striatal population, in opposite directions on the two pathways, and
    nothing in this module reads a dopamine level.  `gain_in` is the corticostriatal GAIN
    (D1 receptors facilitate the synapse, D2 receptors depress it); `theta` is the MSN's
    own EXCITABILITY, and it is the term that acts at rest, where the cortical drive the
    gain multiplies is zero.  v2 measured what happens without the second one: its resting
    beta spectrum was bit-identical at dopamine 0.5 and 0.2, which is CLAUDE.md's "a
    parameter that changes nothing" in its usual form -- the mechanism was real and was not
    wired in at the operating point being measured.

    `Mod` applies `declared * (1 + gain * (rate_src - baseline))`, so at `da_baseline`
    every factor is exactly 1 and the declared weights are the ones that run.
    """
    c = c or C
    out = []
    for ch in CHANNELS:
        da = DA_SOURCE[ch]
        out.append(Mod(da, f"{STRUCTURE}.d1.{ch}", "gain_in", gain=+c.k_da_gain,
                       baseline=c.da_baseline,
                       note="D1 receptors facilitate the corticostriatal synapse"))
        out.append(Mod(da, f"{STRUCTURE}.d1.{ch}", "theta", gain=-c.k_da_theta,
                       baseline=c.da_baseline,
                       note="and raise the direct-pathway MSN's excitability: more "
                            "dopamine, lower threshold"))
        out.append(Mod(da, f"{STRUCTURE}.d2.{ch}", "gain_in", gain=-c.k_da_gain,
                       baseline=c.da_baseline,
                       note="D2 receptors depress the corticostriatal synapse"))
        out.append(Mod(da, f"{STRUCTURE}.d2.{ch}", "theta", gain=+c.k_da_theta,
                       baseline=c.da_baseline,
                       note="and lower the indirect-pathway MSN's excitability, so a "
                            "DEPLETED striatum over-drives the indirect arm at rest"))
    return out


def drives(c: Constants = None):
    """the autonomous pacemaker current each nucleus needs to sit at its declared rate.

    This is the piece that makes "the output nuclei are tonically active" a property of the
    code and not a comment, and it is SOLVED, not declared: `ibm/circuit.py` puts a
    population with `r_rest` at exactly `r_rest` when its net input is zero, so the current
    is whatever cancels every projection into it at the resting state.  Change any weight
    and the resting rates stay where the recording put them -- which is the only way gate
    B7's sweep says anything, since a weight that also slid the operating point would
    confound the two.

    ONE ASSUMPTION, and it is the gap: the solve is taken at ZERO cortical input, so in the
    assembled brain the cortex's own tonic rate will shift the striatum and the STN off
    these by `w * mean(ctx)`.  It is small (the corticostriatal term at cortex's 6-12%
    sparsity is a few hundredths against a threshold of 0.55) but it is not nothing, and it
    is a solve this module cannot do alone because it does not know what cortex will run at.
    """
    c = c or C
    m = msn_rest(c)
    out = {}
    for ch in CHANNELS:
        # STN: cancel the pallidosubthalamic inhibition it lives under
        out[f"{STRUCTURE}.stn.{ch}"] = c.w_gpe_stn * c.rest_gpe
        # GPe: cancel the striatal inhibition, the subthalamic excitation (own + diffuse,
        # which sum to one efferent) and its own recurrent inhibition
        out[f"{STRUCTURE}.gpe.{ch}"] = (c.w_d2_gpe * m - c.w_stn_gpe * c.rest_stn
                                        + c.w_gpe_gpe * c.rest_gpe)
        # GPi: the same three, with the direct arm and the pallidopallidal projection
        out[f"{STRUCTURE}.gpi.{ch}"] = (c.w_d1_gpi * m - c.w_stn_gpi * c.rest_stn
                                        + c.w_gpe_gpi * c.rest_gpe)
        # the striatum's rest is solved FROM theta_msn, so it needs no current, and saying
        # 0.0 here is a statement rather than an omission
        out[f"{STRUCTURE}.d1.{ch}"] = 0.0
        out[f"{STRUCTURE}.d2.{ch}"] = 0.0
    return out


def targets(c: Constants = None):
    c = c or C
    m = msn_rest(c)
    return {
        # Not a sparse code and deliberately so: these are the fractions of units expected
        # ABOVE the engine's 0.2 activity threshold at rest, which for a tonic pacemaker is
        # everything and for a striatum in its down-state is nothing.  No `Pop.sparsity` is
        # declared anywhere in this module -- the divisive normaliser would fight the pause
        # this structure exists to make.
        "sparsity": {
            **{f"{STRUCTURE}.gpi.{ch}": 1.0 for ch in CHANNELS},
            **{f"{STRUCTURE}.gpe.{ch}": 1.0 for ch in CHANNELS},
            **{f"{STRUCTURE}.stn.{ch}": 1.0 for ch in CHANNELS},
            **{f"{STRUCTURE}.d1.{ch}": 0.0 for ch in CHANNELS},
            **{f"{STRUCTURE}.d2.{ch}": 0.0 for ch in CHANNELS},
        },
        "bands": {
            "beta_bg": {
                "pops": ([f"{STRUCTURE}.stn.{ch}" for ch in CHANNELS]
                         + [f"{STRUCTURE}.gpe.{ch}" for ch in CHANNELS]),
                "hz": (13.0, 30.0),
                "note": "`beta_bg` in ibm/rhythms.py, declared peak 20 Hz, stations "
                        "(stn, gpe).  Here it is the STN-GPe ring's damped resonance, and "
                        "its frequency is set by the two effective membrane constants "
                        "against the catalogue's 7 ms round trip.  Report the prominence "
                        "beside the frequency, always: peak_frequency is a soft-argmax and "
                        "returns the band centre on a flat spectrum",
            },
        },
        "rest_hz": rest_rates(c),
        "known_answers": {
            "selection_is_a_pause": {
                "claim": "with different proposals on the four channels, EXACTLY ONE "
                         "channel's GPi falls well below its resting rate and every other "
                         "channel's stays at or above it",
                "measure": "GPi rate as a fraction of its own measured resting rate, per "
                           "channel",
                "expect": "one channel < 0.50, the rest >= 0.98",
                "note": "v2 measured the winner 70.5 -> 12.3 Hz with the runner-up RISING "
                        "to 95.6.  A model where selection is a rise somewhere has the "
                        "sign of the basal ganglia backwards",
            },
            "hyperdirect_is_the_fastest_way_out_of_cortex": {
                "claim": "counted from the declared conduction delays, cortex reaches GPi "
                         "in 6.5 ms hyperdirect, 10.0 ms direct, 17.0 ms indirect",
                "measure": "sum of the declared delays along each path",
                "expect": "hyperdirect < direct < indirect",
                "note": "a property of ibm/rhythms.py's conduction budget, not of any "
                        "weight here; it is why a stop can cancel a movement already "
                        "under way",
            },
            "stop_is_global": {
                "claim": "a pulse into ctx.parsopercularis.E re-raises EVERY channel's GPi "
                         "to at least its resting rate, not only the selected one",
                "measure": "latency from pulse onset until all four are at or above "
                           "0.98 of rest",
                "expect": "<= 200 ms, the upper end of `stop_signal`'s declared 120-200 ms",
                "note": "the catalogue's interval is a behavioural stop-signal reaction "
                        "time and includes the cortical detection this module does not "
                        "contain, so a SHORTER neural latency is a finding and not a "
                        "failure.  v2 measured 22 ms and said so",
            },
            "channels_are_parallel": {
                "claim": "driving one channel alone does not release any other",
                "measure": "the largest GPi drop on an undriven channel",
                "expect": "<= 0.0, i.e. undriven channels RISE",
                "note": "the only thing crossing channels is the subthalamic efferent, "
                        "and it raises GPi everywhere; there is no path by which one "
                        "channel's proposal can lower another's output nucleus",
            },
            "dopamine_direction": {
                "claim": "lowering the rate of nm.snc/nm.vta both RAISES beta and WEAKENS "
                         "selection",
                "measure": "beta prominence at rest, and the winner's GPi drop",
                "expect": "prominence up, drop down",
                "note": "the Parkinsonian direction, out of one change and with no branch "
                        "on dopamine anywhere in this file.  v2: prominence +2.03 -> +2.35 "
                        "and selection abolished entirely",
            },
            "resting_rates": {
                "claim": "with no input the nuclei sit at the rates a recording fixed",
                "measure": "mean rate of each population over a quiet run",
                "expect": f"GPi {c.rest_gpi * R_MAX:.0f} Hz, GPe {c.rest_gpe * R_MAX:.0f}, "
                          f"STN {c.rest_stn * R_MAX:.0f}, striatum "
                          f"{m * R_MAX:.2f}",
                "note": "held there by the solved currents in `drives()`; if a gate "
                        "measures something else, the solve is wrong and every number "
                        "downstream of it is about a different circuit",
            },
        },
    }
