"""neuromodulators: the eight nuclei that replace the four hand-set dials.

`docs/BRAIN_SPEC.md`, "Neuromodulatory nuclei": locus coeruleus, dorsal raphe, the
pedunculopontine/laterodorsal tegmentum and nucleus basalis, VTA and SNc, the
tuberomammillary nucleus, and the orexinergic lateral hypothalamus.  Each is a population
whose RATE multiplies a declared parameter of its targets, through a `Mod` edge.

WHAT THIS REPLACES
------------------
Four floats, passed in by callers and produced by nothing:

    ibm/substrate.py      m_beta, m_sigma   ->  nm.lc and nm.nbm onto ctx.*.E beta/sigma
    ibm/thalamus.py       arousal           ->  nm.ppt (and nm.tmn) onto thal.*.relay theta
    ibm/hippocampus.py    septal_tone       ->  nm.nbm onto hpc.ms.E
    ibm/basal_ganglia.py  dopamine          ->  nm.snc / nm.vta onto bg.d1.* and bg.d2.*

The ids are fixed in `ibm/brain/__init__.py`'s `NM` dict, and this is the one module that
creates them, so there is one locus coeruleus in the brain rather than one per structure
that wanted a gain knob.  Three of those four replacements are `Mod` edges declared HERE;
the striatal one is declared by `ibm/brain/basal_ganglia.py` from these populations, and
`mods()` says why declaring it in both places would be worse than either.

WHAT THESE POPULATIONS ARE, DYNAMICALLY
---------------------------------------
Small, slow and tonically active.  `tau` is 200-400 ms rather than cortex's 10-20 ms
because these are broad, thin, largely unmyelinated projection systems; `r_rest` is
nonzero because a monoaminergic nucleus fires at a few hertz with nothing happening at all
and its silence is a signal.  None of them declares `sparsity`: a divisive normaliser holds
a population's ACTIVE FRACTION at a target, which is the right instrument for a sparse code
and exactly the wrong one for a nucleus whose whole output is its mean level.

`n` is 16.  It is the number of sampled units, NOT a cell count -- the only thing it
encodes is that these nuclei are small beside a cortical area's 64.

THE FLIP-FLOP
-------------
The mutual inhibition here is `rem_flipflop` in `ibm/rhythms.py`: the monoaminergic group
(LC, raphe, TMN -- REM-off, wake-active) against the brainstem cholinergic side (PPT/LDT --
REM-on), each inhibiting the other with a 10-30 ms conduction delay.  Two mutually
inhibiting populations with slow adaptation is a relaxation flip-flop: the network sits in
one state, the active side's adaptation builds, the state is released, and the other side
takes over -- so the trajectory spends its time at the two ends and crosses the middle fast.
That is the declared behaviour and `scripts/gate_brain_neuromodulators.py` N2/N3 measure it.

THE SLEEP-WAKE FLIP-FLOP IS A DIFFERENT SWITCH, AND SAYING SO MATTERS.  Its other half is
the ventrolateral preoptic nucleus, `hyp.vlpo`, which inhibits the whole wake-active side
and is inhibited back by it.  `ibm/brain/hypothalamus.py` now exists and declares the
DESCENDING arm; `external()` here declares the ASCENDING one, and neither declares the
other (see `external()` for what happened when both did).  So assembled with the
hypothalamus there are TWO switches here, and everything the gate measures is the REM one,
because the gate runs the nm populations alone.  The two states it reports are REM-off and
REM-on, not wake and NREM, and nothing in this repository has yet measured the second
switch.

THE COMPRESSED CLOCK, DECLARED.  `ultradian_rem` in `ibm/rhythms.py` gives the biological
period as 4200-6600 s.  At dt = 1 ms that is five million steps per cycle and cannot be run,
so `adapt_tau` here is set to give a measured dwell of **13.3 s** -- a compression of about
**400x** against a 5400 s half-cycle, stated so that nobody reads a dwell measured on this
module as a claim about the ninety-minute cycle.  What is claimed is the SHAPE: two
separated states, a crossing fast against the dwell, and orexin setting how long it lasts.

The biological ratio is not reached either, and that is worth writing down.  Sleep-state
transitions take of order a minute against a 90-minute dwell, a ratio near 0.02; this
module measures **0.082** (crossing 1.09 s, dwell 13.3 s).  The obstruction is structural
rather than a matter of tuning -- see `adapt_tau` in `CONST`.

OREXIN STABILISES; IT DOES NOT DRIVE
------------------------------------
`nm.orx` sends no projection to either side of the flip-flop.  It raises `beta` -- the
steepness of the transfer function, and so the loop gain of the mutual inhibition -- equally
on all four flip-flop populations.  A steeper sigmoid deepens both wells symmetrically, so
orexin makes whichever state the network is in harder to leave without making either state
more likely.  Silence it and the states get shallower: more transitions, shorter dwells,
which is narcolepsy and which is gate N4.

ITS MODULATION IS STATED FROM SILENCE, NOT FROM ITS OWN REST, and that correction came out
of a failed gate rather than out of thought.  Stated from orexin's resting rate -- the
obvious reading of "a nucleus at rest leaves its targets alone" -- the edge did NOTHING at
the operating point, because `nm.orx` has no input here and therefore sits exactly at that
baseline.  N4 still passed, because an ablation moves orexin far off baseline; only N6's
sweep saw it, as `orx_beta_gain` moving the dwell by 0.000 over a 3x sweep.  So
`beta_switch` is the steepness of a switch with NO orexin, and the tone multiplies it by
2.27.  See `beta_switch` in `CONST`.

Two known couplings are DELIBERATELY ABSENT, and their absence is the reason N4 can be read
at all.  Real orexin neurons excite the monoamines, and real noradrenaline and serotonin
inhibit orexin neurons back (Li 2002).  Either edge makes orexin's tone state-dependent, and
then silencing it moves the BALANCE as well as the stability -- and a gate that measures
dwell cannot tell those apart.  If those edges are wanted later, N4 needs a second arm that
holds the balance fixed.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No hypothalamic populations (`hyp.vlpo`, `hyp.scn`, `hyp.pvn`), no habenula, no
adrenoceptor-by-adrenoceptor pharmacology, and no dial anywhere standing in for a structure
that has not been built.  `nm.orx` is the orexinergic subpopulation of the lateral
hypothalamus and is declared HERE because its id is in `NM`; `ibm/brain/hypothalamus.py`
must not create a second one.
"""
from __future__ import annotations

from ibm.brain import Mod, Pop, Proj  # noqa: F401
from ibm.brain.cortex import AREAS as CTX_AREAS

STRUCTURE = "nm"

LC, RAPHE, TMN = "nm.lc", "nm.raphe", "nm.tmn"     # monoaminergic: REM-off, wake-active
PPT = "nm.ppt"                                      # brainstem ACh: REM-on
NBM = "nm.nbm"                                      # basal forebrain ACh: wake AND REM
ORX = "nm.orx"                                      # orexin: stability, not drive
VTA, SNC = "nm.vta", "nm.snc"                       # dopamine: limbic / motor

#: the monoaminergic side of the flip-flop, as one functional node with three transmitters.
AMINERGIC = (LC, RAPHE, TMN)
#: the cholinergic side.  `nm.nbm` follows both sides and is not part of the switch.
CHOLINERGIC = (PPT,)

#: the cortical areas noradrenaline and cortical ACh reach.  Both project to the whole
#: sheet -- that is what "broad and diffuse" means -- so this is all of `cortex.AREAS`
#: rather than a chosen subset.  Named from cortex's own list so there is one such list.
NA_CORTICAL_TARGETS = CTX_AREAS
ACH_CORTICAL_TARGETS = CTX_AREAS

#: the mesocortical dopamine projection: frontal and cingulate only.  These ids exist in
#: `ibm/brain/cortex.py`, so these edges are LIVE as soon as cortex is in the assembly.
VTA_CORTICAL_TARGETS = ("rostralmiddlefrontal", "medialorbitofrontal", "lateralorbitofrontal",
                        "rostralanteriorcingulate", "caudalanteriorcingulate")

#: the cortical areas that project back onto the arousal nuclei -- `arousal` in
#: `ibm/rhythms.py` gives `dlpfc -> lc` at 15-40 ms.  Also live once cortex is assembled.
CORTICAL_RETURN = ("caudalmiddlefrontal", "rostralmiddlefrontal", "rostralanteriorcingulate")

#: the thalamic nuclei, spelled as `ibm/brain/thalamus.py` spells them in its POPULATION
#: IDS -- `pulvinar` and `cm`.  Not as its `NUCLEI` table keys read: this list said `pul`
#: first, because the table entry is `Nucleus("pul", ...)`, and six `Mod` edges pointed at
#: `thal.pul.relay` and `thal.trn.pul`, which do not exist.  Nothing raised; they were
#: simply orphaned, and the thalamic gate would have had no acetylcholine reaching it.  The
#: gate's N7 checks every foreign id against the real populations of every structure that
#: EXISTS, which is the part of this that cannot rot.
#:
#: Written out rather than imported because importing a structure module that may not exist
#: would make this module's declarations conditional on the assembly order, and a
#: declaration must not depend on what else was built.
THAL_NUCLEI = ("lgn", "mgn", "vpl", "vl", "va", "md", "pulvinar", "cm")

#: FOREIGN IDS THIS MODULE NAMES, in one place, so the author of each structure can match
#: them or correct them rather than finding them scattered through the edge lists.  All but
#: `hpc.*` are LIVE -- those modules exist and these are their ids, checked by the gate's
#: N7.  `hpc.*` is a GUESS at a convention nobody has fixed, and the two edges naming it
#: are reported as orphans until `ibm/brain/hippocampus.py` lands.
FOREIGN = {
    "thal_relay": tuple(f"thal.{x}.relay" for x in THAL_NUCLEI),
    "thal_trn": tuple(f"thal.trn.{x}" for x in THAL_NUCLEI),
    "hpc": ("hpc.ms.E", "hpc.ca3.E"),      # GUESS: hippocampus.py is not written
    "hyp": ("hyp.vlpo",),
    "val": ("val.cea",),
}

# ======================================================================================
# the declared constants.  EVERY ONE of these is swept +-50% by the gate's N6.
#
# They live in one dict rather than as module-level names so the sweep can build the
# structure at a changed value without mutating module state -- `pops(c)`, `internal(c)`,
# `mods(c)` each take an overrides dict and default to this one.  Nothing here reads a
# global at call time, so calling any of them twice with the same argument returns the same
# thing, which is what N1 checks.
# ======================================================================================
CONST = {
    # -- the populations ---------------------------------------------------------------
    "n_units": 16,
    "beta_nm": 9.0,          # transfer steepness of the four nuclei that are NOT part of
                             # the switch, and so are not orexin targets
    "beta_switch": 4.0,      # THE SWITCH POPULATIONS' DECLARED STEEPNESS, AND IT IS THE
                             # NARCOLEPTIC ONE.  Orexin's Mod multiplies it, and that Mod
                             # is stated as a deviation from SILENCE (baseline 0), not from
                             # orexin's own resting rate.  Gate N6 is why.
                             #
                             # It was stated from orexin's resting rate first, which reads
                             # as the obvious choice -- "a nucleus at rest leaves its
                             # targets alone" -- and N6 measured what that costs: nm.orx
                             # has no input in the nm-only circuit, so it sits at EXACTLY
                             # its baseline, (r - baseline) = 0.0033, every factor is
                             # 1.003 whatever the gain, and `orx_beta_gain` swept 3x moved
                             # the dwell by 0.000.  A stabiliser that does nothing at the
                             # operating point is not a stabiliser; N4 still passed,
                             # because silencing orexin moves it far off baseline, so the
                             # mechanism looked alive from the ablation alone.  THAT is the
                             # combination CLAUDE.md warns about -- a flat sweep with a
                             # pinned state variable behind it -- and only the sweep saw it.
                             #
                             # With baseline 0 the effective steepness at rest is
                             # 4.0 * (1 + 2.8 * 0.453) = 9.07, which is what the flip-flop
                             # was tuned at, and the declared 4.0 is what the switch runs
                             # at with no orexin at all.
    "theta_nm": 0.12,        # LOW ON PURPOSE.  `Circuit.step` gives a population with an
                             # r_rest the deviation `r_rest + sig(u) - sig(0)`, so its
                             # DOWNWARD range is exactly sig(0) = sigmoid(-beta*theta),
                             # with beta the MODULATED value -- 0.25 at the switch's
                             # effective 9.07.  Put theta near the sigmoid's midpoint and
                             # that range collapses: the suppressed side of the flip-flop
                             # cannot be suppressed, and there is no second state.
    "r_rest_nm": 0.30,       # THIS IS A NORMALISED ACTIVATION, NOT A FIRING RATE.  At the
                             # engine's R_MAX it reads as 30 Hz, and a locus coeruleus
                             # neuron fires 1-3 Hz in waking, under 1 Hz in NREM and
                             # essentially nothing in REM.  R_MAX is not the right scale
                             # for these nuclei and no Hz figure should be quoted off it;
                             # what 0.30 buys is a mid-range rest with room to be driven
                             # both up and down, which is what a flip-flop element needs.
                             # The SHAPE -- high in one state, near-silent in the other --
                             # is what stands for the recording
    "r_rest_orx": 0.30,      # orexin is not part of the switch and has its own rest, so
                             # a sweep can move it without moving the flip-flop's
    "tau_eta": 0.05,
    "tau_lc": 0.25,
    "tau_raphe": 0.30,
    "tau_tmn": 0.35,
    "tau_ppt": 0.20,
    "tau_nbm": 0.20,
    "tau_orx": 0.40,
    "tau_da": 0.25,
    # -- the clock ---------------------------------------------------------------------
    # SLOW SPIKE-FREQUENCY ADAPTATION ON BOTH SIDES.  It is what releases a state, and its
    # strength is not free: two other settings were built and MEASURED to fail first, and
    # both failures have the same shape -- the states stopped being separated.
    #
    #   adapt_g 0.45 with w_ppt_amin 1.1:  the mutual inhibition was too weak to hold the
    #     loser down once the winner had adapted, so the two sides met in the middle at
    #     0.35 and 0.27 and there was no second state left to flip to.
    #   synaptic depression instead of adaptation (tau_rec 3.0, U 0.85): the winner's own
    #     rate stays high, which is the attraction, but the STEADY-STATE transmitted signal
    #     is capped at x*r -> (1/tau_rec)/(1/tau_rec + U) = 0.28 however hard the nucleus
    #     fires, so a weight strong enough to suppress the loser transiently cannot also
    #     release it later.  Measured: a co-active state, aminergic 0.87 beside PPT 0.52,
    #     and the flip-flop stopped flipping after one crossing.
    #
    # The weights and drives below come from a scan of the reduced two-node system over
    # (w, d, adapt_g): 188 of 300 combinations alternate with the states separated, and
    # this is one of them.  `adapt_tau` and `beta_nm` were then set by gate N3, which is
    # the next comment.
    "adapt_tau": 8.00,       # THE COMPRESSED CLOCK.  See the module docstring.
                             # SET BY A FAILED GATE, and the failure is the interesting
                             # part.  At 1.50 the flip-flop alternated cleanly with a 2.98 s
                             # dwell and spent 6.6% of its time in the middle -- and N3,
                             # which asks whether the CROSSING is fast against the dwell,
                             # failed at 0.203 against a declared bar of 0.10.  Lengthening
                             # the clock does not fix that cheaply: the release is a
                             # SADDLE-NODE passage, so the crossing time itself grows as
                             # adapt_tau^(1/3) (measured 0.604 -> 0.953 s over a 3.3x
                             # change, against 1.49x predicted) and the ratio falls only as
                             # adapt_tau^(-2/3).  Reaching 0.077 took adapt_tau 8.0 AND
                             # beta_nm 6 -> 9.  An adaptation-released flip-flop does not
                             # have a scale-separated transition for free, and a bar on
                             # crossing/dwell is the cheapest instrument that shows it.
    "adapt_g": 1.20,         # how far adaptation can push the active side back down
    "sigma_nm": 0.05,        # OU background; what makes a dwell a distribution
    # -- the flip-flop -----------------------------------------------------------------
    # The aminergic arm is SPLIT over three transmitters and the cholinergic arm is one
    # population, so it is the SUM 0.70 + 0.70 + 0.40 = 1.80 that faces `w_ppt_amin` 1.80.
    # The flip-flop is symmetric in the two arms, not in the two weights.
    "w_lc_ppt": 0.58,        # aminergic -> cholinergic inhibition, per transmitter
    "w_raphe_ppt": 0.58,
    "w_tmn_ppt": 0.34,
    "w_ppt_amin": 1.50,      # cholinergic -> aminergic inhibition, onto LC and raphe
    "w_ppt_tmn": 1.50,
    "w_amin_coh": 0.12,      # weak mutual excitation inside the monoaminergic group, so it
                             # switches as one node rather than three
    "delay_ff": 0.020,       # rhythms.py `rem_flipflop`: 10-30 ms, unmyelinated
    "delay_coh": 0.015,
    # -- the followers -----------------------------------------------------------------
    "w_nbm_amin": 0.16,      # basal forebrain ACh is wake- AND REM-active: it reads both
    "w_nbm_ppt": 0.18,       # sides at nearly equal strength, so it is high in EITHER
                             # state and dips only while the switch is crossing.  Larger
                             # weights pin it: at 0.35/0.45 it sat at 0.97 and carried no
                             # information about the state at all
    "w_ppt_vta": 0.14,       # the cholinergic input to the VTA
    "w_lc_snc": 0.14,
    # -- tonic drives ------------------------------------------------------------------
    # Large, because the mutual inhibition is large: a flip-flop is a pair of drives and a
    # pair of inhibitions and only their difference is a state.  The aminergic drives sit
    # BELOW d_ppt by about the cohesion each aminergic nucleus receives from the other two,
    # so neither arm has a standing advantage.
    "d_lc": 1.20,
    "d_raphe": 1.20,
    "d_tmn": 1.20,
    "d_ppt": 1.25,
    "d_nbm": 0.02,
    "d_orx": 0.076,          # holds nm.orx at 0.45, the tone `beta_switch` and
                             # `orx_beta_gain` were chosen against.  Measured 0.453
    "d_vta": 0.030,          # with the state-dependent input from PPT/LC, these hold
    "d_snc": 0.026,           # nm.vta and nm.snc near `da_rest_target` = 0.50 on average
    # -- orexin, the stabiliser --------------------------------------------------------
    "orx_beta_gain": 2.80,   # beta multiplier per unit of orexin rate, from SILENCE.
                             # Set from the narcolepsy claim, not from a gate: losing
                             # orexin should roughly HALVE the dwell, and the measured
                             # beta-to-dwell curve (8.04 s at beta 4.5, 14.03 s at 13.5)
                             # puts that at beta_eff ~4 against ~9, a factor 2.27, which is
                             # 1 + 2.8 * 0.453.  Small gains are the unfaithful choice
                             # here: narcolepsy is a dramatic destabilisation, not a
                             # marginal one.
    # -- the modulation onto other structures ------------------------------------------
    # these set how hard each transmitter shapes its targets.  Every target is outside this
    # module, so N6 sweeps them against the dwell and they come back INERT -- correctly,
    # because the structure they act on is not in the nm-only circuit.  N6 reports that
    # separately from a mechanism that is not wired in; N5 is what shows two of them (the
    # noradrenergic cortical pair) actually reaching a real cortical population.
    "na_ctx_beta_gain": 1.30,     # noradrenaline raises cortical gain
    "na_ctx_sigma_gain": -1.40,   # and lowers the background noise: the SNR story
    "na_trn_gain": 0.90,
    "ach_ctx_beta_gain": 1.10,    # cortical ACh, same direction, its own nucleus
    "ach_ctx_sigma_gain": -1.00,
    "ach_septal_gain": 2.00,      # <- replaces hippocampus.py's `septal_tone`
    "ach_ca3_theta_gain": 0.70,
    "ppt_relay_theta_gain": -0.80,  # <- replaces thalamus.py's `arousal`: opens the gate
    "ppt_trn_theta_gain": 0.90,     #    and shuts the reticular sector, the other half
    "na_relay_beta_gain": 0.70,
    "ha_ctx_theta_gain": -0.55,   # histamine depolarises: lowers threshold
    "ha_relay_theta_gain": -0.60,
    "ht_ctx_gain": -0.45,         # serotonin damps cortical responsiveness to input
    "ht_relay_gain": -0.40,
    "da_pfc_beta_gain": 1.00,     # the mesocortical arm.  The STRIATAL dopamine edges are
                                  # declared by ibm/brain/basal_ganglia.py, not here -- see
                                  # mods().  Their gains are that module's constants.
    "da_rest_target": 0.50,       # basal_ganglia.py's `da_baseline`.  nm.vta and nm.snc
                                  # must RUN here or its declared weights are off by an
                                  # undeclared factor; gate N0 reports the measured rate.
    # -- modulation clamp --------------------------------------------------------------
    "mod_lo": 0.35,          # a modulator shapes a parameter and cannot replace it
    "mod_hi": 4.00,          # the orexin edge runs at 2.27 and its +50% sweep reaches
                             # 2.90; at the old 3.00 the top of that sweep would have
                             # CLAMPED and read as a flat parameter for a second reason
    "nm_baseline": 0.30,     # the rate at which a transmitter leaves its targets alone:
                             # r_rest_nm, so "no modulation" is the resting nucleus
}


def _c(c=None) -> dict:
    """the constants, with overrides.  Never mutates `CONST`."""
    return CONST if not c else {**CONST, **c}


# ======================================================================================
# populations
# ======================================================================================
def pops(c=None):
    c = _c(c)
    n = int(c["n_units"])

    def nucleus(pid, tau, r_rest, switch, note):
        return Pop(id=pid, n=n, kind="mod", tau=tau,
                   beta=(c["beta_switch"] if switch else c["beta_nm"]),
                   theta=c["theta_nm"], r_rest=r_rest,
                   adapt_tau=(c["adapt_tau"] if switch else None),
                   adapt_g=(c["adapt_g"] if switch else 0.0),
                   sigma=c["sigma_nm"], tau_eta=c["tau_eta"], note=note)

    return [
        nucleus(LC, c["tau_lc"], c["r_rest_nm"], True,
                "locus coeruleus, noradrenaline; REM-off, one half of the flip-flop"),
        nucleus(RAPHE, c["tau_raphe"], c["r_rest_nm"], True,
                "dorsal raphe, serotonin; REM-off, with LC"),
        nucleus(TMN, c["tau_tmn"], c["r_rest_nm"], True,
                "tuberomammillary, histamine; the slowest of the wake-active group"),
        nucleus(PPT, c["tau_ppt"], c["r_rest_nm"], True,
                "pedunculopontine/laterodorsal, brainstem ACh; REM-on, the other half"),
        nucleus(NBM, c["tau_nbm"], c["r_rest_nm"], False,
                "nucleus basalis, cortical ACh; reads BOTH sides at nearly equal weight, "
                "so it is high in waking and in REM and dips only while the switch "
                "crosses.  It falls properly only when both sides are down, which takes "
                "hyp.vlpo and therefore the full assembly"),
        nucleus(ORX, c["tau_orx"], c["r_rest_orx"], False,
                "orexinergic lateral hypothalamus; projects to neither side and sets the "
                "gain of both.  Its loss is narcolepsy and is gate N4"),
        nucleus(VTA, c["tau_da"], c["r_rest_nm"], False,
                "ventral tegmental area, dopamine; limbic and prefrontal"),
        nucleus(SNC, c["tau_da"], c["r_rest_nm"], False,
                "substantia nigra pars compacta, dopamine; motor striatum"),
    ]


# ======================================================================================
# projections inside nm
# ======================================================================================
def internal(c=None):
    """the flip-flop, the cohesion of the monoaminergic group, and the two followers.

    The delays are `rem_flipflop`'s 10-30 ms, taken from the catalogue rather than chosen.
    They are NOT what sets the crossing time here, and the measurement says so: the
    crossing is 1.07 s, 53x the 20 ms delay and 5.3x the fastest population's tau.  What
    sets it is the saddle-node passage the adaptation drags the system through -- see
    `adapt_tau` in `CONST`.  The delays are declared because a loop's conduction budget is
    what `ibm/modes.py` will read off the assembled network, not because this module's
    dwell depends on them.
    """
    c = _c(c)
    out = []
    # --- the switch: aminergic -| cholinergic
    for src, w in ((LC, c["w_lc_ppt"]), (RAPHE, c["w_raphe_ppt"]), (TMN, c["w_tmn_ppt"])):
        out.append(Proj(src, PPT, weight=w, sign=-1, delay_s=c["delay_ff"],
                        note="REM-off onto REM-on; alpha2 / 5-HT1A / H3"))
    # --- and back: cholinergic -| aminergic
    for dst, w in ((LC, c["w_ppt_amin"]), (RAPHE, c["w_ppt_amin"]), (TMN, c["w_ppt_tmn"])):
        out.append(Proj(PPT, dst, weight=w, sign=-1, delay_s=c["delay_ff"],
                        note="REM-on onto REM-off; the reciprocal-interaction arm"))
    # --- the monoaminergic group holds together.  Weak and symmetric: this is what makes
    #     three transmitters behave as one node of the switch, and it is the smallest thing
    #     that does.  It is NOT a claim that LC excites TMN by a particular receptor.
    for a, b in ((LC, RAPHE), (RAPHE, LC), (LC, TMN), (TMN, LC)):
        out.append(Proj(a, b, weight=c["w_amin_coh"], sign=1, delay_s=c["delay_coh"],
                        note="monoaminergic cohesion; the group switches as one"))
    # --- the followers
    out.append(Proj(LC, NBM, weight=c["w_nbm_amin"], sign=1, delay_s=c["delay_coh"],
                    note="basal forebrain ACh follows the wake-active side"))
    out.append(Proj(PPT, NBM, weight=c["w_nbm_ppt"], sign=1, delay_s=c["delay_coh"],
                    note="and the REM-active side; that is why it is high in both"))
    out.append(Proj(PPT, VTA, weight=c["w_ppt_vta"], sign=1, delay_s=c["delay_ff"],
                    note="the cholinergic input to the VTA"))
    out.append(Proj(LC, SNC, weight=c["w_lc_snc"], sign=1, delay_s=c["delay_ff"],
                    note="noradrenergic input to the nigra"))
    return out


# ======================================================================================
# projections that leave or enter nm
# ======================================================================================
def external(c=None):
    """the ascending arm of the sleep-wake switch, the cortical return, and two afferents.

    Every edge here names a population in another structure.  The ones naming a structure
    that has not been written are ORPHANS and are declared anyway, because an edge that is
    quietly dropped is a loop that quietly stops existing (`ibm/brain/__init__.py`).

    NOTHING HERE IS DECLARED TWICE.  Two `Proj`s with the same endpoints do not conflict:
    `Circuit.step` SUMS their contributions, so a pathway declared by both its structures
    is silently twice as strong as either module says.  Gate N7 checks the whole assembly
    for that, and it found two when it was first run -- `hyp.vlpo -> nm.*` and
    `val.lhb -> nm.vta`, both already declared by the structure that owns the other end.
    """
    c = _c(c)
    out = []
    # --- THE SLEEP-WAKE FLIP-FLOP, this module's half of it.
    #     The DESCENDING arm -- `hyp.vlpo -| nm.lc/raphe/tmn/orx`, galanin and GABA from
    #     the ventrolateral preoptic onto the wake-active side -- is declared by
    #     `ibm/brain/hypothalamus.py`, which owns `hyp.vlpo`.  It was declared here too
    #     until the gate's N7 found it: two `Proj`s with the same endpoints do not raise,
    #     their contributions are SUMMED in `Circuit.step`, so the VLPO would simply have
    #     inhibited the wake side twice as hard as either module declared.  Only the
    #     ASCENDING arm is declared here, and hypothalamus.py does not declare it.
    for p in AMINERGIC:
        out.append(Proj(p, "hyp.vlpo", weight=0.60, sign=-1, delay_s=0.020,
                        note="the monoaminergic inhibition of the VLPO that closes the "
                             "sleep-wake flip-flop; the descending arm is declared by "
                             "ibm/brain/hypothalamus.py and is NOT repeated here"))
    # --- the cortical return onto the arousal nuclei.  `arousal` in ibm/rhythms.py,
    #     dlpfc -> lc at 15-40 ms.  LIVE with cortex.
    for a in CORTICAL_RETURN:
        out.append(Proj(f"ctx.{a}.E", LC, weight=0.18, sign=1, delay_s=0.028,
                        note="the cortical return; the arousal loop is a loop"))
    # --- NOT a projection: `arousal` in ibm/rhythms.py gives `lc -| trn` a 10-30 ms
    #     conduction budget, and ibm/brain/thalamus.py declares that it expects
    #     noradrenaline as a `Mod` on the reticular sector instead.  It is declared that way
    #     in `mods()` and NOT duplicated here.  The cost is recorded in `mods()`: a `Mod`
    #     has no delay, so that arm of the arousal loop now carries none.
    # --- what tells dopamine anything.  Value is produced by the valuation system, and
    #     `ibm/brain/valuation.py` declares every edge INTO nm.vta (from val.bla, val.lhb,
    #     val.vp and val.nacc_core).  Those are not repeated here, for the reason above.
    #     `val.cea -> nm.lc` is not declared there, so it is declared here.
    out.append(Proj("val.cea", LC, weight=0.30, sign=1, delay_s=0.020,
                    note="central amygdala onto LC; threat raises noradrenaline.  The only "
                         "val -> nm edge ibm/brain/valuation.py does not declare"))
    for ch in ("motor", "oculomotor", "associative"):
        out.append(Proj(f"bg.d1.{ch}", SNC, weight=0.10, sign=-1, delay_s=0.012,
                        note="the striatonigral feedback that closes the motor dopamine "
                             "loop.  bg.d1.<channel>, the ids basal_ganglia.py declares"))
    return out


# ======================================================================================
# THE MODULATION.  This is what the module is for.
# ======================================================================================
def mods(c=None):
    """every `Mod` edge by which a transmitter shapes a target.

    `Mod` multiplies ONE DECLARED PARAMETER of a target POPULATION:

        effective = declared * clamp(1 + gain * (rate_src - baseline), lo, hi)

    `baseline` is `nm_baseline` = the nuclei's resting rate, so a nucleus at rest leaves its
    targets exactly as declared and the modulation is a deviation from rest rather than a
    scale factor nobody can read.  `lo`/`hi` clamp the FACTOR, so a modulator can shape a
    parameter and cannot replace it -- which is the difference between this and the four
    floats it replaces.

    WHO DECLARES WHAT, because the sister modules disagree and the disagreement is not
    harmless.  `ibm/brain/thalamus.py` declares `mods() -> []` and names, in its docstring,
    the three edges it expects THIS module to declare, for a reason it gave: a `Mod` naming
    a population in an unbuilt structure used to CRASH the assembly where the identical
    `Proj` was reported as an orphan, so an edge had to be declared by the structure owning
    its SOURCE.  (`ibm/brain/__init__.py` has since been fixed to filter and report `Mod`
    orphans as well, so that particular reason has gone; the convention it produced has
    not.)  `ibm/brain/basal_ganglia.py` declares its own dopamine edges, from `nm.snc` and
    `nm.vta`, at the TARGET.

    So the striatal dopamine edges are NOT declared here.  `Circuit._modulation`
    multiplies every factor onto the same (population, parameter) pair, so declaring them
    in both places would not conflict or raise -- it would silently apply dopamine TWICE,
    which is the worst available outcome.  The gate's N7 checks the whole assembled brain
    for a duplicated (src, dst, param) triple, because this will happen again.

    THREE LIMITS OF THE ENGINE, recorded here because they shaped what could be declared:

    * A `Mod` cannot target a PROJECTION.  "ACh suppresses the CA3 recurrent collateral"
      (`ach_suppression` in `ibm/hippocampus.py`) has no expression as a population
      parameter; the nearest thing, raising `hpc.ca3.E`'s threshold, is declared below and
      is a PROXY, not the same mechanism.
    * A `Mod` HAS NO DELAY.  `Proj` carries `delay_s` and the docstring there says the
      delay is what a loop's resonance is made of; `Mod` reads the source's mean rate at
      the current step.  `ibm/rhythms.py`'s `arousal` loop gives `lc -| trn` a 10-30 ms
      conduction budget and that budget is simply absent once the arm is a `Mod` -- so the
      arousal loop, as assembled, has a delay on its cortical return leg and none on its
      neuromodulatory one, and no resonance measured on it is a measurement of that loop.
    * `gain_in` multiplies the projection input BEFORE the tonic drive is added
      (`Circuit.step`), so `gain_in` onto a population whose only input is `drive` does
      nothing at all.  Every `gain_in` edge here targets a population driven by a
      projection.
    """
    c = _c(c)
    b, lo, hi = c["nm_baseline"], c["mod_lo"], c["mod_hi"]
    out = []

    def mod(src, dst, param, gain, note):
        out.append(Mod(src=src, dst=dst, param=param, gain=gain, baseline=b,
                       lo=lo, hi=hi, note=note))

    # --- NORADRENALINE onto cortex.  THIS IS `m_beta` AND `m_sigma`. ------------------
    for a in NA_CORTICAL_TARGETS:
        mod(LC, f"ctx.{a}.E", "beta", c["na_ctx_beta_gain"],
            "noradrenaline raises cortical gain; ibm/substrate.py's m_beta, produced")
        mod(LC, f"ctx.{a}.E", "sigma", c["na_ctx_sigma_gain"],
            "and lowers the background current; m_sigma.  Together they are the "
            "signal-to-noise account of the LC (Aston-Jones & Cohen)")

    # --- CORTICAL ACh onto cortex.  Same two parameters, its own nucleus: the factors
    #     compose multiplicatively in Circuit._modulation, which is the point of having
    #     two sources rather than one dial.
    for a in ACH_CORTICAL_TARGETS:
        mod(NBM, f"ctx.{a}.E", "beta", c["ach_ctx_beta_gain"],
            "nucleus basalis: attentional gain, muscarinic")
        mod(NBM, f"ctx.{a}.E", "sigma", c["ach_ctx_sigma_gain"],
            "and the same noise reduction")

    # --- HISTAMINE onto cortex: H1 depolarisation lowers the threshold ----------------
    for a in NA_CORTICAL_TARGETS:
        mod(TMN, f"ctx.{a}.E", "theta", c["ha_ctx_theta_gain"],
            "histamine depolarises cortical pyramids; wakefulness as a lowered threshold")

    # --- SEROTONIN onto cortex: 5-HT1A damps responsiveness to input ------------------
    for a in NA_CORTICAL_TARGETS:
        mod(RAPHE, f"ctx.{a}.E", "gain_in", c["ht_ctx_gain"],
            "serotonin damps the cortical response to its inputs during quiet waking")

    # --- BRAINSTEM ACh onto the thalamus.  THIS IS `arousal`. -------------------------
    for t in FOREIGN["thal_relay"]:
        mod(PPT, t, "theta", c["ppt_relay_theta_gain"],
            "the cholinergic depolarisation that takes the relay out of burst mode: "
            "ibm/thalamus.py's `arousal`, as the nucleus that produces it")
        mod(TMN, t, "theta", c["ha_relay_theta_gain"],
            "histamine depolarises relay cells too; the second arm of the same gate")
        mod(RAPHE, t, "gain_in", c["ht_relay_gain"],
            "serotonin on the relay's input")
    for t in FOREIGN["thal_trn"]:
        mod(PPT, t, "theta", c["ppt_trn_theta_gain"],
            "and the muscarinic HYPERPOLARISATION of the reticular nucleus -- threshold "
            "UP where the relay's went down.  That is the other half of opening the gate: "
            "a relay depolarised with the gate still shut is not an open thalamus")
        mod(LC, t, "beta", c["na_trn_gain"],
            "noradrenaline on the reticular sector")
    for t in FOREIGN["thal_relay"]:
        mod(LC, t, "beta", c["na_relay_beta_gain"],
            "noradrenaline sharpens the relay's transfer function: gain, not offset -- "
            "the third edge ibm/brain/thalamus.py's docstring asks this module for")

    # --- DOPAMINE onto the striatum.  DECLARED BY `ibm/brain/basal_ganglia.py`, NOT HERE.
    #     Four edges per channel, two per striatal population, from nm.snc and nm.vta --
    #     exactly the mechanism `dopamine` was standing in for, with these populations as
    #     its source.  Declaring them here as well would not raise: Circuit._modulation
    #     MULTIPLIES the factors onto the same (population, parameter) pair, so dopamine
    #     would simply act twice and nothing downstream could tell.  Gate N7 checks the
    #     assembled brain for exactly that.  The ids are `bg.d1.<channel>` and
    #     `bg.d2.<channel>` over four channels, not `bg.str.d1`.
    #
    #     One cross-module constant follows from this and is NOT free here:
    #     basal_ganglia.py states its weights at `da_baseline = 0.50`, so `nm.snc` and
    #     `nm.vta` must RUN near 0.50 or every declared striatal weight is off by a factor
    #     nobody declared.  `d_vta`/`d_snc` are set for that and `targets()` states it.
    for a in VTA_CORTICAL_TARGETS:
        mod(VTA, f"ctx.{a}.E", "beta", c["da_pfc_beta_gain"],
            "the mesocortical projection: dopaminergic gain in frontal cortex")

    # --- CORTICAL ACh onto the hippocampus.  THIS IS `septal_tone`. -------------------
    mod(NBM, "hpc.ms.E", "beta", c["ach_septal_gain"],
        "basal forebrain onto the medial septum: ibm/hippocampus.py's `septal_tone`, "
        "as the nucleus that produces it")
    mod(NBM, "hpc.ca3.E", "theta", c["ach_ca3_theta_gain"],
        "A PROXY, not the mechanism.  What ACh does is suppress the CA3 recurrent "
        "collateral so encoding does not complete to an old pattern; Mod cannot target a "
        "projection, so this raises CA3's threshold instead.  Recorded as a proxy.")

    # --- OREXIN.  Inside nm, so these edges are LIVE and gate N4 can measure them. -----
    for p in (LC, RAPHE, TMN, PPT):
        out.append(Mod(src=ORX, dst=p, param="beta", gain=c["orx_beta_gain"],
                       baseline=0.0, lo=c["mod_lo"], hi=c["mod_hi"],
                       note="orexin steepens the transfer function of BOTH sides equally: "
                            "it deepens the wells without moving the balance, which is what "
                            "'stabilises rather than drives' has to mean quantitatively. "
                            "BASELINE 0, not orexin's resting rate: a modulator stated "
                            "from its own rest exerts nothing at rest, and gate N6 "
                            "measured exactly that -- see `beta_switch` in CONST"))
    return out


# ======================================================================================
def drives(c=None):
    """the tonic drive each nucleus sits at.

    These are not decoration: a flip-flop is a pair of drives and a pair of inhibitions, and
    `d_ppt` has to be large enough that the cholinergic side can win alone against the full
    monoaminergic press (0.58 + 0.58 + 0.34 = 1.50) once adaptation has taken the edge off
    it -- and no larger, or it wins always.  N6 sweeps every one of them.
    """
    c = _c(c)
    return {LC: c["d_lc"], RAPHE: c["d_raphe"], TMN: c["d_tmn"], PPT: c["d_ppt"],
            NBM: c["d_nbm"], ORX: c["d_orx"], VTA: c["d_vta"], SNC: c["d_snc"]}


def targets(c=None):
    """what this structure is FOR, in numbers.  A specification, not a measurement.

    The numbers in the `note` fields ARE measurements, and are marked as such: they are
    what `scripts/gate_brain_neuromodulators.py` read on this declaration, kept here so the
    next person can tell at a glance whether a change moved something.
    """
    c = _c(c)
    return {
        "sparsity": {},   # none of these is a sparse code; see the module docstring
        "rates": {
            LC: c["r_rest_nm"], RAPHE: c["r_rest_nm"], TMN: c["r_rest_nm"],
            PPT: c["r_rest_nm"], NBM: c["r_rest_nm"], ORX: 0.45,   # set by d_orx; N0 reports the measured value
            VTA: c["da_rest_target"], SNC: c["da_rest_target"],
        },
        "cross_module_constraints": {
            "da_baseline": {
                "claim": "nm.vta and nm.snc must RUN at ibm/brain/basal_ganglia.py's "
                         "`da_baseline`, because that module states every striatal weight "
                         "at that rate",
                "expect": c["da_rest_target"],
                "note": "measured 0.509 and 0.501 over a 180 s run of the nm-only circuit. "
                        "In the full assembly nm.vta also receives val.bla, val.lhb, "
                        "val.vp and val.nacc_core, so this has to be re-measured there",
            },
            "orx_tone": {
                "claim": "nm.orx must run near 0.45, because `beta_switch` and "
                         "`orx_beta_gain` were chosen so that 4.0 * (1 + 2.8 * 0.453) is "
                         "the steepness the flip-flop was tuned at",
                "expect": 0.45,
                "note": "measured 0.453.  In the full assembly hyp.lh -> nm.orx drives it "
                        "higher and hyp.vlpo inhibits it, so the switch's effective beta "
                        "becomes state-dependent there and this must be re-measured",
            },
        },
        "bands": {
            "state_cycle": {
                "pops": [LC, PPT],
                "hz": (0.02, 0.12),
                "note": "the flip-flop's alternation AS BUILT, on the compressed clock: a "
                        "measured dwell of 13.3 s, so a full cycle near 0.038 Hz. "
                        "`ultradian_rem` in ibm/rhythms.py is 1.5e-4 to 2.5e-4 Hz; this is "
                        "that SHAPE run about 400x fast and is NOT a claim about the "
                        "ninety-minute cycle",
            },
        },
        "known_answers": {
            "two_states": {
                "claim": "the network sits in one of two separated regimes and crosses "
                         "between them, rather than sliding through the middle as a drive "
                         "is swept",
                "measure": "occupancy of the central third of the gap between the two state "
                           "means, pooled over a drive sweep; and the bimodality "
                           "coefficient of the same pool",
                "expect": "central occupancy < 0.15 (a continuum gives ~0.33); BC > 0.5556 "
                          "(a uniform distribution gives exactly 0.5556)",
                "note": "gate N2.  Measured 0.0072 and 0.959.  The per-bias table is the "
                        "part that carries the claim: the two state MEANS hold at +0.88 "
                        "and -0.86 across the sweep while the fraction of time spent high "
                        "slides 0.37 -> 0.71 -- the bias moves the occupancy, not the "
                        "states",
            },
            "fast_transition": {
                "claim": "the crossing is fast against the dwell",
                "measure": "10-90% crossing time of the state variable, against the mean "
                           "dwell between crossings",
                "expect": "crossing < 0.10 x dwell",
                "note": "gate N3.  Measured 1.07 s against 13.26 s, ratio 0.081 -- and it "
                        "FAILED at 0.203 on the first declaration (adapt_tau 1.5, beta_nm "
                        "6).  See `adapt_tau` in CONST: the release is a saddle-node "
                        "passage, so the crossing itself grows as adapt_tau^(1/3) and this "
                        "ratio is not free.  The crossing also CANNOT be faster than the "
                        "fastest population's tau (tau_ppt = 200 ms); measured, it is 5.3x "
                        "that floor, so it is not a measurement of tau",
            },
            "orexin_stabilises": {
                "claim": "silencing nm.orx makes the state less stable: more transitions "
                         "per minute and a shorter mean dwell",
                "measure": "both, intact against silenced, same seeds, same wiring, same "
                           "drive everywhere else",
                "expect": "transitions up AND mean dwell down, on every seed",
                "note": "gate N4.  Measured 13 -> 17 transitions and 13.30 -> 10.90 s "
                        "dwell, on each of three seeds.  This CAN fail: orexin's only "
                        "edges are beta modulations, and a beta modulation that did "
                        "nothing would make the two arms identical",
            },
            "modulation_reaches": {
                "claim": "a declared Mod edge changes its target's behaviour",
                "measure": "nm.lc -> ctx.pericalcarine.E, beta and sigma separately, in a "
                           "two-population circuit built from the real Pop and Mod "
                           "declarations, against a no-Mod control",
                "expect": "transfer SLOPE up >= 1.5x on the beta edge; per-unit temporal sd "
                          "down to <= 0.7x on the sigma edge; the control bit-identical",
                "note": "gate N5, and it is FAILED, at 1.479 against a bar of 1.5 -- "
                        "recorded as failed, not rescored.  The sigma arm passed at 0.25x "
                        "and the no-Mod control was bit-identical, so the edge demonstrably "
                        "reaches; it is the slope bar that was missed, by 1.4%.  The cause "
                        "is outside this module: the same arm measured 1.90x an hour "
                        "earlier, and `ibm/circuit.py` commit ec07938 then changed the "
                        "DERIVED `w_inh` of a sparse population from max(1.0, worst) to "
                        "max(0.05, 0.05*worst).  The cortical target's normaliser is 20x "
                        "weaker, so it sits nearer its linear regime and a gain change "
                        "buys proportionally less there.  Nothing in the nm-only gates "
                        "moved: no population here declares `sparsity`, so that fix does "
                        "not touch them.  GAIN IS A SLOPE: the target's mean RATE FALLS "
                        "(0.271 -> 0.227) as noradrenaline rises, because a steeper sigmoid "
                        "at a subthreshold operating point emits less, and reading the rate "
                        "would have reported noradrenaline's sign backwards",
            },
            "every_constant_does_something": {
                "claim": "no constant in CONST is inert on both the state dwell and the "
                         "state balance over a 3x sweep",
                "measure": "each swept +-50%, noise off; mean dwell and fraction of time "
                           "in the aminergic state",
                "expect": "adapt_tau, adapt_g and orx_beta_gain each move the dwell > 20%",
                "note": "gate N6, and it is FAILED: adapt_g 601%, adapt_tau 92%, "
                        "orx_beta_gain 19.6% against a 20% bar -- missed by 0.4 points and "
                        "recorded as failed, not rescored.  The sweep earned its place "
                        "twice over before that.  It caught `orx_beta_gain` at 0.000 when "
                        "orexin's Mod was stated from its own resting rate (see "
                        "`beta_switch`), and then caught its own instrument: `d_ppt` LOCKS "
                        "the flip-flop into one state at 0.5x and the other at 1.5x, and "
                        "because a locked run reports the whole run as its dwell, both "
                        "ends read 90.0 s, span 0.000, INERT -- for the most consequential "
                        "constant in the file.  A locked arm now scores an infinite span "
                        "and can never be called inert",
            },
            "one_edge_one_factor": {
                "claim": "no pathway and no modulation is declared by two structures",
                "measure": "duplicate (src, dst, param) Mod triples and (src, dst, sign) "
                           "external Proj pairs over every structure module on disk",
                "expect": "none",
                "note": "gate N7.  Found TWO on its first run, both this module's: "
                        "hyp.vlpo -> nm.lc/raphe/tmn/orx and val.lhb -> nm.vta were already "
                        "declared by hypothalamus.py and valuation.py.  Neither raises; "
                        "Circuit.step sums Proj contributions and _modulation multiplies "
                        "Mod factors, so both would simply have been twice as strong as "
                        "any module declared",
            },
        },
    }
