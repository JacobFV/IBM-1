"""valuation and affect: the system that makes some states better than others.

`docs/BRAIN_SPEC.md`, "Valuation and affect".  Without this a body can learn to avoid and
cannot learn to seek, and every "reward" in this programme is otherwise a number handed in
from outside a training loop.  `docs/DEVELOPMENTAL_COMPONENTS.md`'s note of 18 September
2026 says this structure had no implementation of any kind for nine days while five
rhythm-bearing structures were built to a high standard, because the build order was chosen
by a catalogue of oscillations and this system does not have a famous band.  This module is
the answer to that note.

WHAT THIS IS FOR
----------------
Four things the specification asks for, and where each one lives:

1. **Positive and negative value are different pathways, not one signed number.**  The
   basolateral amygdala (`val.bla`) scores what arrived and excites the dopaminergic VTA
   directly; the central amygdala (`val.cea`) is the defensive arm; the lateral habenula
   (`val.lhb`) is the negative-value path and it *inhibits* dopamine; the accumbens
   (`val.nacc_core`, `val.nacc_shell`) turns value into a pull.  Those are separate
   populations with separate inputs, so approach and avoidance are separable states rather
   than one scalar with a sign -- which is what V3 measures.

2. **Prediction error, not reward.**  The drive this structure delivers onto `nm.vta` is

       D(t) = + w_actual * bla(t-d)          what arrived
              - w_lhb_vta * lhb(t-d)         what was expected, inverted twice on the way
              - w_vp_vta  * vp(t-d)
              - w_core_vta * core(t-d)       the accumbo-tegmental edge `ibm/rhythms.py`
                                             declares in the `val` loop

   and the subtraction happens AT THE VTA, between the amygdala's excitation and the
   habenula's inhibition.  `vta_inputs()` is that list, in one place, so a gate reads the
   declaration rather than reinventing it.

3. **WHERE THE EXPECTATION LIVES, stated plainly**: in the membrane state of
   `val.nacc_core`, whose `tau` is seconds rather than milliseconds.  It is a
   recency-weighted trace of delivered value -- a Rescorla-Wagner cache with a fixed rate --
   and it reaches the VTA inverted twice, through the GABAergic accumbo-pallidal projection
   (`nacc_core -| vp`) and the GABAergic pallido-habenular one (`vp -| lhb`).  So a standing
   expectation *raises* the habenula, and a raised habenula *lowers* dopamine.

   **This is a state variable standing in for a synaptic weight, and that is a declared
   compromise, not an accident.**  In a brain the expectation is stored in weights and
   `ibm/circuit.py` has no plasticity rule that runs during a rollout (`store()` writes a
   matrix between rollouts, which is not the same thing).  Two consequences follow and both
   are reported rather than hidden:
     - the expectation is TEMPORAL, not cue-specific: this circuit predicts "reward is
       arriving at about this rate", not "this cue predicts reward".  A cue-specific
       expectation needs a three-factor plasticity rule the engine does not have.
     - the dip on an omitted reward is therefore a SUSTAINED depression across the window
       where the reward was expected, not a phasic 100 ms pause.  A phasic, time-locked
       pause needs a temporal prediction (an eligibility trace), which is the same missing
       engine feature.

4. **The habenula must be able to push dopamine below baseline.**  `val.lhb` and `val.vp`
   are declared with `r_rest=0.5` and `theta=0.0`, which is the one configuration of
   `ibm/circuit.py`'s tonic branch that gives a population an equal excursion in both
   directions: the engine computes `r_rest + sig(u) - sig(0)`, so a population that rests
   high with a large `theta` can be driven UP by 1 - sig(0) and DOWN by only sig(0).  A
   tonically active nucleus declared the obvious way (`r_rest=0.55, theta=0.35`) can fall by
   0.057 and no further, which is a disappointment system that cannot be disappointed.

THE COMPARISON, AND WHY IT IS WHERE IT IS
-----------------------------------------
The outcome does NOT enter the ventral pallidum.  `val.bla -> val.vp` (the glutamatergic
amygdalo-pallidal projection) is real and is deliberately not declared here: routing what
arrived into the pallidum as well would cancel it against the expectation *before* the
comparison, leaving the habenula with nothing to carry and the dopamine drive computing its
prediction error somewhere no gate can see it.  The pallidum here carries the EXPECTED value
(Tian & Uchida 2016) and nothing else, so the habenula is load-bearing -- which is exactly
what V4 checks, by silencing it and requiring the dip to go.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
- **No second VTA.**  `nm.vta` is the dopaminergic population and it lives in
  `ibm/brain/neuromodulators.py`.  Every edge to and from it is declared in `external()`,
  and `collect()` reports any that does not resolve rather than dropping it.  This module was
  written while that one did not exist and both ends now do, so the `val` loop can be closed
  for the first time -- but `reward_theta` (2-8 Hz), `amygdala_theta` and the loop's
  resonance are properties of the ASSEMBLED circuit, they belong to `ibm/modes.py`, and no
  gate here claims one.  A0 in `scripts/gate_brain_valuation.py` also records two conflicts
  that appeared with those siblings and that this module is not allowed to fix:
  `ibm/brain/neuromodulators.py` declares `val.lhb->nm.vta` as well (so the pathway would be
  applied TWICE, at the sum of the two weights, because `Circuit` keys its matrices by
  `Proj.key` and does not raise on a duplicate), and it and `ibm/brain/hypothalamus.py`
  name the accumbens `val.nac.core` / `val.nac.shell` against the `val.nacc_core` /
  `val.nacc_shell` declared here, so four of their edges can never resolve.
- **No RMTg.**  `val.lhb -| nm.vta` is drawn as one edge; the rostromedial tegmental nucleus
  that actually inverts it is a brainstem population and belongs to `ibm/brain/brainstem.py`.
  When that exists this edge should be split in two rather than reweighted.
- **No negative-valence BLA ensemble.**  `val.bla` is declared as the BLA->NAcc,
  positive-valence ensemble (Namburi 2015 showed the valence of a BLA neuron is a property
  of its projection target).  The fear-encoding BLA->CeA ensemble is not separately
  represented because `Proj` cannot address a SUBSET OF UNITS within a population -- the
  topologies are dense, sparse, one-to-one and diffuse, none of which is a declared mask.
  That is the one engine feature this structure wanted and did not get; it is reported, not
  added.  `w_bla_cea` is small in consequence, standing for the shared salience drive only,
  and the aversive outcome enters at `val.cea` directly (the parabrachial nociceptive
  pathway, Han 2015, whose source population lives in the brainstem).
- **Almost no efferents, and one that had to be added.**  V6 in
  `scripts/gate_brain_valuation.py` measured `w_bla_shell` and `tau_shell` inert over a 3x
  sweep, and the diagnosis -- read off the declared graph, not guessed -- was that
  `val.nacc_shell` had NO OUTGOING EDGE AT ALL: the population that carries the whole
  appetitive score sent it nowhere.  `val.nacc_shell -| hyp.lh` was added for that reason.
  Its constants are still inert in the isolated preparation, and correctly so, because the
  pull is an OUTPUT of this structure and not a term in the prediction error -- V6 partitions
  them on a reachability test rather than on that sentence.  The remaining efferents this
  structure will want (the pallido-thalamic arm onto `thal.md.relay`, and the accumbens onto
  the limbic channel of `ibm/brain/basal_ganglia.py`) are not declared here because those
  pathways are named for the pallidum and the striatum and belong to those modules.
- **No interneurons, and therefore no accumbens gamma.**  `nacc_gamma` (50-100 Hz,
  `ibm/rhythms.py`) needs a fast local inhibitory partner in the accumbens.  Nothing in this
  structure's specification requires one, so none is declared and the band is listed in
  `targets()` as not producible rather than quietly expected.
- **No declared sparsity, on any population here.**  Sparsity in `ibm/circuit.py` is
  divisive inhibition, and it is there so a subset can ignite without the sheet igniting --
  a property of a cortex.  What these nuclei carry is a RATE: how good, how expected, how
  bad.  Declaring a sparsity would be declaring an operating point nothing has measured, and
  normalising a rate-coded difference by the population's own mean is the one operation that
  would destroy the difference the structure exists to compute.  `targets()` says so, and
  says what would change it.

UNITS follow `ibm/circuit.py`: rates dimensionless in [0, 1], delays in seconds.  The
conduction delays of the four edges the `val` loop in `ibm/rhythms.py` names -- amygdala to
vmPFC 8-18 ms, vmPFC to accumbens 6-14 ms, accumbens to VTA 5-12 ms, VTA back to amygdala
10-25 ms -- are taken at their interval midpoints and marked where they are used.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from ibm.brain import NM, Mod, Pop, Proj  # noqa: F401

STRUCTURE = "val"

BLA = "val.bla"                 # basolateral amygdala: what arrived
CEA = "val.cea"                 # central amygdala: the defensive output
CORE = "val.nacc_core"          # accumbens core: WHERE THE EXPECTATION LIVES
SHELL = "val.nacc_shell"        # accumbens shell: the pull
VP = "val.vp"                   # ventral pallidum: expected value, tonically active
LHB = "val.lhb"                 # lateral habenula: negative value, tonically active

VTA = NM["da_vta"]                                  # nm.vta -- one VTA in the brain
MOFC = "ctx.medialorbitofrontal.E"                  # these two resolve: ibm/brain/cortex.py
RACC = "ctx.rostralanteriorcingulate.E"             # carries both in AREAS
LH = "hyp.lh"                                       # ibm/brain/hypothalamus.py; the shell's
                                                    # efferent -- see `external()`


@dataclass(frozen=True)
class ValParams:
    """every constant this structure declares, in one place so a sweep can reach them all.

    `scripts/gate_brain_valuation.py`'s V6 sweeps every float field here +-50% and re-measures
    the three prediction-error numbers.  A constant that moves none of them over a 3x sweep is
    listed as inert -- which in this repository has five times meant a mechanism that was not
    wired in, not a model that is robust (CLAUDE.md, "A parameter that changes nothing").
    """
    # ---- sizes (structural; not swept) -------------------------------------------------
    n_bla: int = 48
    n_cea: int = 32
    n_core: int = 48
    n_shell: int = 48
    n_vp: int = 32
    n_lhb: int = 32

    # ---- time constants ----------------------------------------------------------------
    tau_bla: float = 0.020
    tau_cea: float = 0.030
    tau_core: float = 1.5      # SECONDS.  the expectation's memory; see the docstring on
                               # why a membrane state is standing in for a synaptic weight
    tau_shell: float = 0.030
    tau_vp: float = 0.020
    tau_lhb: float = 0.025

    # ---- transfer ----------------------------------------------------------------------
    beta_e: float = 10.0       # bla, cea, shell
    theta_e: float = 0.30
    beta_core: float = 6.0     # shallower: the expectation must GRADE, not switch
    theta_core: float = 0.35
    beta_tonic: float = 6.0    # vp, lhb, about r_rest with theta = 0
    r_rest: float = 0.50       # the tonic level vp and lhb sit at with no input
    sigma: float = 0.010       # OU background on the four phasic populations

    # ---- tonic drive -------------------------------------------------------------------
    d_tonic: float = 0.05      # bla, cea, core, shell: the amygdala is not silent
    # the resting rates of the two populations that tonically inhibit the pallidum.  These
    # are MEASUREMENTS of the circuit at rest, and `drives()` derives the pallidum's tonic
    # drive from them so that sweeping an inhibitory weight moves the GAIN and not the
    # operating point.  V0 checks them against what the circuit actually rests at, so a
    # drift is reported rather than silently moving the resting state.
    r_core_rest: float = 0.206
    r_cea_rest: float = 0.087

    # ---- internal weights --------------------------------------------------------------
    w_bla_cea: float = 0.20    # small: see "No negative-valence BLA ensemble" above
    w_bla_core: float = 1.00   # drives the expectation trace
    w_bla_shell: float = 1.00  # the appetitive pull
    w_cea_vp: float = 0.80     # GABAergic amygdalo-pallidal: THE AVERSIVE ARM
    w_core_vp: float = 1.20    # GABAergic accumbo-pallidal: THE EXPECTATION
    w_vp_lhb: float = 1.40     # GABAergic pallido-habenular (Faget 2018)

    # ---- the drive onto nm.vta ----------------------------------------------------------
    w_actual: float = 1.00     # glutamatergic BLA -> VTA: WHAT ARRIVED
    w_lhb_vta: float = 1.25    # FITTED, by `gate_brain_valuation.py --fit`: the one constant
                               # here set by a measurement rather than declared.  It is the
                               # gain that makes a fully predicted reward silent, which is a
                               # weight-matching condition a brain learns and this one cannot.
    w_vp_vta: float = 0.15     # GABAergic VP -> VTA; opposes the dip, and is reported doing so
    w_core_vta: float = 0.10   # the `val` loop's accumbo-tegmental edge, 5-12 ms.  Small on
                               # purpose: its job here is to close the loop that carries the
                               # reward rhythm, not to carry the subtraction -- if it carried
                               # the subtraction the habenula would not be load-bearing and
                               # V4 could not fail.
    w_vta_bla: float = 0.30
    w_vta_core: float = 0.25
    w_vta_shell: float = 0.35

    # ---- cortical revision of the score -------------------------------------------------
    w_bla_mofc: float = 0.30
    w_bla_racc: float = 0.25
    w_mofc_core: float = 0.30
    w_mofc_shell: float = 0.25
    w_mofc_bla: float = 0.20
    w_racc_core: float = 0.20
    w_racc_cea: float = 0.15

    # ---- the pull, leaving the structure -------------------------------------------------
    w_shell_lh: float = 0.60   # GABAergic accumbens-shell -> lateral hypothalamus.  Added
                               # after V6 measured w_bla_shell and tau_shell INERT and the
                               # diagnosis was that val.nacc_shell had no outgoing edge at
                               # all: a population that receives the whole appetitive score
                               # and sends it nowhere.  See the docstring.

    # ---- conduction delays, seconds -----------------------------------------------------
    d_bla_cea: float = 0.004
    d_bla_nacc: float = 0.006
    d_cea_vp: float = 0.005
    d_core_vp: float = 0.005
    d_vp_lhb: float = 0.006
    d_bla_vta: float = 0.010
    d_lhb_vta: float = 0.012   # includes the RMTg relay that is not declared; see docstring
    d_vp_vta: float = 0.008
    d_core_vta: float = 0.0085  # rhythms.py `val`: nacc -> vta, 5-12 ms
    d_vta_bla: float = 0.0175   # rhythms.py `val`: vta -> amygdala, 10-25 ms
    d_vta_nacc: float = 0.015
    d_bla_ctx: float = 0.013    # rhythms.py `val`: amygdala -> vmpfc, 8-18 ms
    d_ctx_val: float = 0.010    # rhythms.py `val`: vmpfc -> nacc, 6-14 ms
    d_shell_hyp: float = 0.009

    def swept(self):
        """the float fields, which is what V6 sweeps.  Sizes are structural and excluded."""
        return [f.name for f in fields(self) if f.type == "float"]


PARAMS = ValParams()

#: the dt the declared delays are meant to resolve at.  The shortest is 4 ms, so 2 ms leaves
#: every delay at two steps or more; `Circuit.vanishing_delays(dt)` is the check and the gate
#: prints it.
DT = 0.002


def pops(p: ValParams = PARAMS):
    return [
        Pop(id=BLA, n=p.n_bla, kind="E", tau=p.tau_bla, beta=p.beta_e, theta=p.theta_e,
            sigma=p.sigma,
            note="basolateral amygdala, the BLA->NAcc positive-valence ensemble: it scores "
                 "what arrived.  Excites nm.vta directly -- the 'actual' side of the "
                 "prediction error"),
        Pop(id=CEA, n=p.n_cea, kind="I", tau=p.tau_cea, beta=p.beta_e, theta=p.theta_e,
            sigma=p.sigma,
            note="central amygdala, GABAergic output nucleus: the defensive arm, and the "
                 "port the aversive outcome arrives at (parabrachial, Han 2015)"),
        Pop(id=CORE, n=p.n_core, kind="I", tau=p.tau_core, beta=p.beta_core,
            theta=p.theta_core, sigma=p.sigma,
            note="accumbens core.  THE EXPECTATION LIVES IN THIS POPULATION'S MEMBRANE "
                 "STATE: tau is seconds, so its rate is a recency-weighted trace of "
                 "delivered value.  A state standing in for a weight, deliberately and "
                 "with the consequences declared in the module docstring"),
        Pop(id=SHELL, n=p.n_shell, kind="I", tau=p.tau_shell, beta=p.beta_e,
            theta=p.theta_e, sigma=p.sigma,
            note="accumbens shell: the pull.  Fast, consummatory, and it does NOT project "
                 "to the pallido-habenular arm -- core and shell have different pallidal "
                 "targets, and a shell that fed the habenula would make every reward its "
                 "own suppressor"),
        Pop(id=VP, n=p.n_vp, kind="I", tau=p.tau_vp, beta=p.beta_tonic, theta=0.0,
            r_rest=p.r_rest,
            note="ventral pallidum, tonically active.  Carries EXPECTED value and nothing "
                 "else.  theta=0 with r_rest=0.5 is the tonic configuration that can be "
                 "driven equally far in both directions"),
        Pop(id=LHB, n=p.n_lhb, kind="E", tau=p.tau_lhb, beta=p.beta_tonic, theta=0.0,
            r_rest=p.r_rest,
            note="lateral habenula, tonically active and glutamatergic: the negative-value "
                 "path.  Disinhibited by a fall in the pallidum, so a standing expectation "
                 "with no outcome RAISES it, and a raised habenula pushes dopamine below "
                 "baseline.  This is the population V4 silences"),
    ]


def internal(p: ValParams = PARAMS):
    return [
        Proj(BLA, CEA, weight=p.w_bla_cea, delay_s=p.d_bla_cea,
             note="shared salience only; the fear-encoding BLA ensemble is not declared"),
        Proj(BLA, CORE, weight=p.w_bla_core, delay_s=p.d_bla_nacc,
             note="what arrived, charging the expectation trace"),
        Proj(BLA, SHELL, weight=p.w_bla_shell, delay_s=p.d_bla_nacc,
             note="what arrived, as a pull"),
        Proj(CEA, VP, weight=p.w_cea_vp, sign=-1, delay_s=p.d_cea_vp,
             note="GABAergic amygdalo-pallidal: the aversive arm reaches the habenula here"),
        Proj(CORE, VP, weight=p.w_core_vp, sign=-1, delay_s=p.d_core_vp,
             note="GABAergic accumbo-pallidal: THE EXPECTATION, inverted once"),
        Proj(VP, LHB, weight=p.w_vp_lhb, sign=-1, delay_s=p.d_vp_lhb,
             note="GABAergic pallido-habenular: the expectation, inverted twice"),
    ]


def external(p: ValParams = PARAMS):
    """every edge that leaves or enters this structure.

    The four onto `nm.vta` are the dopamine drive and are listed again by `vta_inputs()`.
    The six touching cortex resolve against `ibm/brain/cortex.py`; the seven touching
    `nm.vta` are orphans until `ibm/brain/neuromodulators.py` exists, and `collect()` reports
    them as such rather than dropping them.
    """
    return [
        # ---- the dopamine drive: actual minus expected -----------------------------------
        Proj(BLA, VTA, weight=p.w_actual, delay_s=p.d_bla_vta,
             note="glutamatergic BLA -> VTA: WHAT ARRIVED, the positive term"),
        Proj(LHB, VTA, weight=p.w_lhb_vta, sign=-1, delay_s=p.d_lhb_vta,
             note="the negative-value path.  Drawn as one edge; the RMTg that inverts it is "
                  "a brainstem population and is not declared here"),
        Proj(VP, VTA, weight=p.w_vp_vta, sign=-1, delay_s=p.d_vp_vta,
             note="GABAergic VP -> VTA.  Opposes the omission dip; reported, not removed"),
        Proj(CORE, VTA, weight=p.w_core_vta, sign=-1, delay_s=p.d_core_vta,
             note="rhythms.py `val` loop, nacc -> vta, 5-12 ms.  Closes the loop that "
                  "carries reward theta; small, so it is not the subtraction"),
        # ---- the dopaminergic return -----------------------------------------------------
        Proj(VTA, BLA, weight=p.w_vta_bla, delay_s=p.d_vta_bla,
             note="rhythms.py `val` loop, vta -> amygdala, 10-25 ms, slow unmyelinated"),
        Proj(VTA, CORE, weight=p.w_vta_core, delay_s=p.d_vta_nacc,
             note="mesolimbic dopamine onto the core.  Its PARAMETRIC action (D1/D2 gain) "
                  "is a Mod edge for ibm/brain/neuromodulators.py to declare, not this one"),
        Proj(VTA, SHELL, weight=p.w_vta_shell, delay_s=p.d_vta_nacc,
             note="mesolimbic dopamine onto the shell: value becomes a pull"),
        # ---- cortex revises the score against context ------------------------------------
        Proj(BLA, MOFC, weight=p.w_bla_mofc, delay_s=p.d_bla_ctx,
             note="rhythms.py `val` loop, amygdala -> vmpfc, 8-18 ms"),
        Proj(BLA, RACC, weight=p.w_bla_racc, delay_s=p.d_bla_ctx),
        Proj(MOFC, CORE, weight=p.w_mofc_core, delay_s=p.d_ctx_val,
             note="rhythms.py `val` loop, vmpfc -> nacc, 6-14 ms: context revising the "
                  "expectation"),
        Proj(MOFC, SHELL, weight=p.w_mofc_shell, delay_s=p.d_ctx_val),
        Proj(MOFC, BLA, weight=p.w_mofc_bla, delay_s=p.d_ctx_val,
             note="top-down appraisal onto the score itself"),
        Proj(RACC, CORE, weight=p.w_racc_core, delay_s=p.d_ctx_val),
        Proj(RACC, CEA, weight=p.w_racc_cea, delay_s=p.d_ctx_val,
             note="cingulate onto the defensive arm"),
        # ---- the pull leaves here --------------------------------------------------------
        Proj(SHELL, LH, weight=p.w_shell_lh, sign=-1, delay_s=p.d_shell_hyp,
             note="GABAergic accumbens-shell -> lateral hypothalamus: the one efferent this "
                  "structure has, and what 'value becomes a pull' means in wiring.  Named "
                  "for the accumbens, so it is declared here; ibm/brain/hypothalamus.py "
                  "declares the three afferents in the other direction"),
    ]


def mods(p: ValParams = PARAMS):
    """valuation originates no neuromodulation.  It PRODUCES the signal that drives one.

    The dopaminergic population is `nm.vta` and the `Mod` edges by which its rate multiplies
    parameters of its targets belong to `ibm/brain/neuromodulators.py`.  Declaring one here
    would be the second VTA this module exists not to create.
    """
    return []


def drives(p: ValParams = PARAMS):
    """the tonic drive each population sits at.

    The two pallidal and habenular entries are DERIVED, not declared: a tonically active
    nucleus rests where its resting inhibition puts it, so its tonic drive is a consequence
    of the weights that inhibit it.  Deriving it means sweeping `w_core_vp` moves the GAIN of
    the expectation and not the operating point of the pallidum, which is the difference
    between a sensitivity measurement and a confound.
    """
    return {
        BLA: p.d_tonic,
        CEA: p.d_tonic,
        CORE: p.d_tonic,
        SHELL: p.d_tonic,
        VP: p.w_core_vp * p.r_core_rest + p.w_cea_vp * p.r_cea_rest,
        LHB: p.w_vp_lhb * p.r_rest,
    }


def inputs(p: ValParams = PARAMS):
    """the ports an outcome arrives at, declared so a gate reads them rather than choosing.

    Both are stand-ins for afferents whose source populations live in structures that are not
    written yet, and both are named here so that when those structures arrive the edge is
    added rather than the port being quietly repurposed.
    """
    return {
        "appetitive_outcome": {
            "pop": BLA,
            "stands_for": "gustatory and visceral reward afferents reaching the BLA, via the "
                          "insula and the parabrachial nucleus"},
        "aversive_outcome": {
            "pop": CEA,
            "stands_for": "the parabrachial nociceptive projection onto the central amygdala "
                          "(Han 2015); a brainstem population, not declared here"},
    }


def vta_inputs(p: ValParams = PARAMS):
    """(src, sign, weight, delay_s) for every declared edge onto `nm.vta`.

    ONE source of truth for the dopamine drive.  Until `ibm/brain/neuromodulators.py` exists
    these edges are orphans, so a gate cannot read the drive off a population and must
    compute it; it computes it from THIS, so the thing measured is the thing declared.

    For a dense, row-normalised projection `ibm/circuit.py` delivers `sign * weight *
    mean(r_src)` to every unit of the target, so the drive is that sum, each term read at its
    own conduction delay.
    """
    return [(e.src, e.sign, e.weight, e.delay_s) for e in external(p) if e.dst == VTA]


def targets(p: ValParams = PARAMS):
    return {
        "sparsity": {},
        "sparsity_note":
            "NO population here declares one, deliberately.  Sparsity in ibm/circuit.py is "
            "divisive inhibition driven by the population's own mean rate, and it exists so "
            "a subset can ignite without the sheet igniting.  What these nuclei carry is a "
            "RATE -- how good, how expected, how bad -- and normalising a rate-coded "
            "difference by the population's own mean is the one operation that destroys it. "
            "This changes the day val.bla is asked to carry WHICH stimulus rather than how "
            "good it was: a BLA engram is sparse (~10-20% of neurons) and at that point the "
            "population needs a declared sparsity and a calibration pass.",
        "dt_s": DT,
        "inputs": inputs(p),
        "bands": {
            "reward_theta": {
                "pops": [MOFC, RACC], "hz": (2.0, 8.0),
                "status": "NOT PRODUCIBLE YET",
                "note": "ibm/rhythms.py `reward_theta`, filed under the `val` loop.  The "
                        "loop's return arm is nm.vta -> val.bla and nm.vta does not exist, "
                        "so the loop is open and no gate here claims this band."},
            "amygdala_theta": {
                "pops": [BLA, MOFC], "hz": (4.0, 8.0),
                "status": "NOT PRODUCIBLE YET",
                "note": "ibm/rhythms.py `amygdala_theta`: a coherence between val.bla and "
                        "cortex.  Both ends exist once cortex is assembled with this "
                        "structure, but the 8-18/6-14 ms arms alone do not make a "
                        "resonance -- that is what ibm/modes.py is for, on the assembled "
                        "brain, not here."},
            "nacc_gamma": {
                "pops": [CORE, SHELL], "hz": (50.0, 100.0),
                "status": "NOT PRODUCIBLE, BY CONSTRUCTION",
                "note": "ibm/rhythms.py `nacc_gamma`.  Gamma is an E-I loop and this "
                        "structure declares no interneuron partner for the accumbens, "
                        "because nothing in its specification requires one.  Listed so it "
                        "is a declared absence rather than a quiet expectation."},
        },
        "known_answers": {
            "three_way_prediction_error": {
                "claim": "the drive onto nm.vta responds to the DIFFERENCE between what "
                         "arrived and what was expected: an unexpected reward raises it, a "
                         "fully predicted one returns it to baseline, an omitted expected "
                         "one pushes it below baseline",
                "measure": "vta_inputs() summed at their delays, in a window at the outcome, "
                           "against the quiescent naive baseline",
                "expect": "unexpected > +margin, |predicted| < 0.25 x unexpected, "
                          "omitted < -margin",
                "note": "this is the specification of a prediction error; V2 in "
                        "scripts/gate_brain_valuation.py is the gate"},
            "habenula_load_bearing": {
                "claim": "silencing the pallido-habenular edge removes the dip",
                "measure": "w_vp_lhb = 0, which also zeroes the habenula's derived tonic "
                           "drive so it freezes at r_rest, and re-measure the omission",
                "expect": "the omission response no longer clears the dip threshold",
                "note": "a control that can fail: if the accumbo-tegmental edge alone "
                        "produced the dip this would pass for the wrong reason, which is "
                        "why w_core_vta is small"},
            "separable_valence": {
                "claim": "a positive and a negative outcome of the same magnitude are not "
                         "one state with opposite sign",
                "measure": "cosine between the two six-population response vectors",
                "expect": "|cos| <= 0.5; a one-signed-scalar model gives exactly -1.0",
                "note": "the -1.0 is the known answer the metric is checked against"},
            "extinction": {
                "claim": "repeated omission decays the expectation, so the dip shrinks",
                "measure": "the omission response on the first and last of a run of "
                           "omissions, and val.nacc_core's rate at each",
                "expect": "|last| < 0.5 x |first|, and the core's rate falls"},
        },
        "engine_features_wanted_and_not_added": [
            "a Proj topology that addresses a declared SUBSET of the source population's "
            "units, so the positive- and negative-valence BLA ensembles could be one "
            "population with two output channels instead of two populations",
            "a three-factor plasticity rule that runs DURING a rollout, so the expectation "
            "could live in a weight (which is where it lives in a brain) instead of in a "
            "slow membrane state, and could be cue-specific rather than temporal",
            "an eligibility trace, without which the omission response is a sustained "
            "depression over the expected window rather than a phasic time-locked pause",
        ],
    }
