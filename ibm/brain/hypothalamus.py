"""hypothalamus: where the body's needs become things the brain wants.

`docs/BRAIN_SPEC.md`, "Hypothalamus and the drives": the suprachiasmatic nucleus carries
the clock, the paraventricular the stress axis, the ventrolateral preoptic sleep, the
lateral arousal and feeding, the arcuate energy state.  "These carry the body's needs into
the brain as signals that change what the brain wants, which is where value comes from in a
body rather than from a scalar."

Shape and voice follow `ibm/brain/cortex.py`, the reference implementation: `pops`,
`internal`, `external`, `mods`, `drives`, `targets`, and nothing else.

WHAT A DRIVE IS HERE
--------------------
**A drive is a slow variable with a set point, not an input.**  Four of them are carried,
each as a population whose RATE is the drive and whose time constant is the drive's own:

    hunger              `hyp.arc`    tau  3 h      relieved by vagal nutrient afference
    thirst              `hyp.mnpo`   tau  2 h      relieved by gastric pre-absorptive signal
    thermal (warm)      `hyp.poa`    tau 20 min    driven by parabrachial thermoafference
    sleep pressure      `hyp.som`    tau 15 h      built by waking, cleared by sleeping

Each sits at a fixed point where its tonic accumulation balances the relief it receives;
that fixed point IS the set point, and nothing in the module reads a clock or a flag to
find it.  `hyp.arc` and `hyp.mnpo` are named for the arcuate and the median preoptic
nucleus; `hyp.som` is the somnogen -- Borbely's process S, the adenosine-like quantity that
rises with time awake -- and it is a population here because a drive that lived in a scalar
would be exactly the `arousal` dial this package exists to delete.

`hyp.mnpo` and `hyp.som` are NOT in the five nuclei `docs/BRAIN_SPEC.md` names.  They are
here because the specification this module is gated against asks for four drives and the
five named nuclei have nowhere to put thirst or sleep pressure: the median preoptic nucleus
is the osmoreceptive nucleus and it is not the thermoregulatory preoptic area, and process S
is not the ventrolateral preoptic nucleus -- it is what DRIVES it.  Folding either into a
neighbour would have been a population doing two jobs with one time constant.

THE CLOCK IS AN OSCILLATOR, NOT A READING OF `time`
---------------------------------------------------
`hyp.scn` is a relaxation oscillator: self-excitation through a dense recurrent projection,
with the engine's slow adaptation variable standing for the PER/CRY protein that accumulates
and represses its own transcription.  That mapping is exact rather than decorative -- the
engine's adaptation is a slow variable driven by the population's own rate and subtracted
from its input, which is what the transcriptional negative-feedback loop is -- so the period
comes from `PER_TAU` and `PER_G`, from time constants, which is what `ibm/rhythms.py`'s
`circadian` row declares (`set_by="time constants"`).  It is NOT set by the `circ` loop's
conduction delay.  `ibm/rhythms.py`'s round-trip bound is a CEILING and never a prediction,
and here it is not remotely binding: the `circ` ring is 40-120 ms, so it bounds the loop at
8-25 Hz, and the rhythm is at 1.16e-5 Hz -- six orders of magnitude below its own ceiling.

`THETA_SCN` is NEGATIVE.  That is the one unusual-looking constant in the file and it says
something true: the suprachiasmatic nucleus is autonomously active, so with zero input its
fast subsystem sits ON and only its own accumulated repression turns it off.  The
alternative -- a positive threshold plus a tonic drive of the same size -- is the same
circuit written with two constants instead of one, and it would make the free-running gate
depend on a drive handed in from outside, which is the thing a free-run gate is for.

A NOTE ON THE `val.` AND `nm.` IDS, ADDED 18 SEPTEMBER 2026
-----------------------------------------------------------
The three edges into the valuation system were first written to `val.nac.shell`.
`ibm/brain/valuation.py` calls that population `val.nacc_shell`, and the two sat in the
orphan list looking exactly like a structure nobody had built.  A cross-structure edge is
written by one author and its target is owned by another, so a naming disagreement and a
missing module produce the same report; `ibm/brain/__init__.py` now suggests the nearest
existing id for precisely this reason.  The ids here are checked against the modules that
exist: `val.nacc_shell` (valuation), `thal.lgn.relay` (thalamus), and every `nm.*` taken
from `NM` rather than spelled out.  `bs.nts` and `bs.pb` are still genuinely missing.

WHERE THE LIGHT COMES IN
------------------------
`thal.lgn.relay -> hyp.scn`, declared in `external()`, weight `W_LIGHT_SCN`, excitatory,
20 ms.  **This is a named substitution, not the anatomy.**  The retinohypothalamic tract
leaves intrinsically photosensitive retinal ganglion cells, and there is no retina in
`ibm.brain.STRUCTURES`; the geniculate relay and the RHT share the optic nerve, so the LGN
relay's rate is the nearest declared thing to retinal irradiance in this package.  When a
retinal population exists the edge should move to it and this paragraph should go.  Until
`thalamus` exists the edge is an orphan and the clock free-runs, which is the correct
behaviour of a blinded clock.

WHAT `ibm/interoception.py` GIVES THESE DRIVES, AND WHAT IT DOES NOT
--------------------------------------------------------------------
It gives fifteen visceral afferent channels on four trunks, each bound to a receptor
component and carrying a MEASURED conduction delay from IHM-1's route lengths: gastric and
intestinal distension on vagal A-beta (8 ms), gastric and intestinal nutrient and
hepatoportal glucose on vagal C (448 ms), GI absorption on A-delta, aortic baroreceptor and
the two chemoreceptor channels, and five high-threshold splanchnic channels on C fibres
(279 ms) bound to `transduction.visceral_nociceptor`.  Its first relay is the nucleus of the
solitary tract, then parabrachial, then VPL, then insula.

So the relief arm of hunger and of thirst is real and is declared: `bs.nts -> hyp.arc` and
`bs.nts -> hyp.mnpo`, because the nutrient and distension channels are exactly what the NTS
is relaying.  The pre-absorptive quench of thirst by gastric distension is a real and fast
signal and it is the one the thirst drive reads.

**It does not give a single endocrine or thermal channel.**  There is no cortisol, no
leptin, no ghrelin, no insulin, no plasma osmolality and no core temperature anywhere in
`PORTS`, and `ONTOLOGY_GAPS` records that the splanchnic rows have no transduction law
either.  Four consequences, all of them stated rather than papered over:

  * osmotic thirst has no osmoreceptor.  `hyp.mnpo`'s accumulation is the tonic `D_MNPO`
    (obligatory water loss), and its only measured relief is the gastric one.  The volaemic
    arm could be built today from `vagus/aortic_baroreceptor`; it is not, because a
    baroreceptor rate is not an osmolality and joining them needs a law the body does not
    declare.
  * thermal load has no thermometer.  `bs.pb -> hyp.poa` carries cutaneous thermoafference
    through the parabrachial relay, which is the anatomy, but no channel in `PORTS` reports
    a temperature, so that edge has nothing to carry yet.
  * the HPA axis cannot close.  `hpa` in `ibm/rhythms.py` runs pvn -> pituitary -> adrenal
    -> pvn, and the pituitary and the adrenal cortex are IHM-1's organs, not populations.
    `hyp.pvn` therefore carries slow adaptation (`GC_TAU`, `GC_G`) as a SURROGATE for
    glucocorticoid negative feedback: same sign, same timescale, not the loop.  The ultradian
    cortisol pulse (`cortisol_ultradian`, 2.0e-4 - 4.0e-4 Hz) is a property of the closed
    loop and is **not claimed** by this module -- see `targets()["not_claimed"]`.
  * `hyp.arc` reads a nutrient arrival, not an energy store.  A leptin signal is what makes
    the arcuate a long-horizon integrator; without one, `TAU_ARC` is doing that job on its
    own and the module says so.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
**No second orexin population.**  Orexinergic lateral hypothalamus is `nm.orx` in
`ibm/brain/__init__.py`'s `NM` table and it belongs to `ibm/brain/neuromodulators.py`.
`hyp.lh` is the lateral hypothalamus as a nucleus -- wake-active, feeding-related, the other
half of the sleep switch -- and it projects ONTO `nm.orx` rather than containing it.

**No sparsity on any population.**  A drive is a LEVEL, and `Pop`'s own docstring says a
declared sparsity is "right for a relay and wrong for a cortex"; divisive normalisation
holds a population's mean bounded, which is precisely what would destroy a signal whose
whole content is its mean.  `calibrate_sparsity` therefore has nothing to do here, and
`targets()["sparsity"]` is empty on purpose rather than by omission.

**No cold-defence population.**  `hyp.poa` is the warm-sensitive preoptic class only, so
thermal discomfort here is one-sided.  A two-sided drive needs two populations and the
second one is not written.

A TRAP FOR WHOEVER WRITES `brainstem.py`
----------------------------------------
This module declares the three INCOMING visceral edges (`bs.nts -> hyp.arc`,
`bs.nts -> hyp.mnpo`, `bs.pb -> hyp.poa`) even though the contract says an external edge is
declared by the structure the pathway is named for, and an ascending visceral projection is
arguably the brainstem's.  They are here because without them a drive has no relief and the
specification above is not expressed anywhere.  **`Circuit` does not check for duplicate
projections**, so if `brainstem.py` declares the same edges the assembled circuit will carry
both and the relief will be twice as strong with nothing saying so.  Whoever writes it must
not re-declare these three.

UNITS: seconds throughout, as `ibm/circuit.py` requires.  The consequence is that this
module's time constants run from 60 s to 68000 s while the rest of the brain's run from
1 ms to 1 s, and that is a fact about the hypothalamus rather than a problem to fix: the
exponential-Euler step is a convex combination at any dt, so a 5-hour time constant is
bounded and correct at dt = 1 ms (it barely moves) and a 60 ms conduction delay is bounded
and correct at dt = 60 s (it vanishes, and `Circuit.vanishing_delays` says which).
`scripts/gate_brain_hypothalamus.py` states the timestep it measures at and what it cost.
"""
from __future__ import annotations

from ibm.brain import NM, Mod, Pop, Proj  # noqa: F401
from ibm.rhythms import RHYTHM            # the one declared rhythm catalogue

STRUCTURE = "hyp"

N = 16              # units per nucleus.  These carry a LEVEL, not a code: the units are
                    # deliberately interchangeable and no population here is sparse, so
                    # n is a numerical convenience and not a claim about cell counts.

# ---------------------------------------------------------------- the clock
# a relaxation oscillator.  the fast subsystem (TAU_SCN, BETA_SCN, THETA_SCN, W_SCN_REC) is
# bistable; the slow variable (PER_TAU, PER_G) sweeps it across both knees.  PER_G is what
# decides whether it oscillates AT ALL -- below about 1.2 the upper knee is never reached
# and the clock stops -- and PER_TAU is what sets the period, near-linearly.
TAU_SCN = 900.0         # 15 min: the nucleus's own membrane/synaptic integration
BETA_SCN = 8.0
THETA_SCN = -0.20       # NEGATIVE: autonomously active.  see the docstring.
W_SCN_REC = 1.2         # recurrent self-excitation; BETA_SCN * W_SCN_REC / 4 = 2.4 > 1,
                        # which is the condition for the fast subsystem to be bistable
PER_TAU = 68000.0       # 18.9 h: the repressor's accumulation and turnover time
PER_G = 1.8             # how hard the accumulated repressor presses back

# ---------------------------------------------------------------- sleep pressure
TAU_SOM = 54000.0       # 15 h awake.  ONE time constant, because `Pop` has one; the
                        # asleep value is this multiplied by SOM_CLEAR_GAIN's factor below
BETA_SOM = 10.0
THETA_SOM = 0.30
W_LH_SOM = 1.0          # waking builds the somnogen
SOM_CLEAR_GAIN = -0.60  # a `Mod` on `tau`: at hyp.vlpo ~= 0.9 the factor is 1 - 0.54, so
                        # the somnogen clears about 2.2x faster asleep than it builds
                        # awake.  Human process-S fits give ~18 h rise against ~4 h decay
                        # (4.3x); this is the same shape and a SMALLER ratio, which the
                        # gate reports rather than hides, and it is a state-dependent
                        # PARAMETER rather than two hand-written branches.
SOM_CLEAR_LO = 0.25     # the `Mod` clamp: tau can be shortened 4x and no further
SOM_CLEAR_HI = 1.0      # and never lengthened -- sleep does not slow the clearance

# ---------------------------------------------------------------- the sleep switch
TAU_VLPO = 60.0
BETA_VLPO = 10.0
THETA_VLPO = 0.55
W_SOM_VLPO = 1.2        # sleep pressure onto the sleep-active population
W_POA_VLPO = 0.35       # warm-sensitive preoptic onto it too: a warm body is a sleepy one
W_LH_VLPO = 0.7         # the wake side pressing back -- half of the flip-flop

TAU_LH = 60.0
BETA_LH = 10.0
THETA_LH = 0.30
W_VLPO_LH = 0.9         # the other half.  vlpo <-> lh mutual inhibition IS the flip-flop
W_ARC_LH = 0.30         # hunger drives the lateral hypothalamus
W_MNPO_LH = 0.15        # and so does thirst: both are seeking states
D_LH = 0.45             # tonic wake tone.  the ascending reticular drive's stand-in until
                        # `brainstem.py` exists; it is in `drives()` so it is visible

# process C: the clock gates the sleep switch by RAISING its threshold in subjective day.
# Borbely's two-process model, written as the engine's own mechanism -- a modulatory edge
# multiplying a declared parameter -- rather than as a schedule.
PROCESS_C_GAIN = 1.00
PROCESS_C_BASE = 0.45   # hyp.scn's own long-run mean rate, so the factor is ~1 at the
                        # middle of the cycle and swings either side of it
PROCESS_C_LO = 0.55
PROCESS_C_HI = 1.60

# ---------------------------------------------------------------- the drives
TAU_ARC = 10800.0       # 3 h.  doing the job a leptin signal would do; see the docstring
BETA_ARC = 6.0
THETA_ARC = 0.25
D_ARC = 0.55            # metabolic depletion: the body is always burning fuel
W_NTS_ARC = 0.9         # vagal nutrient and distension afference, relieving it

TAU_MNPO = 7200.0       # 2 h
BETA_MNPO = 6.0
THETA_MNPO = 0.25
D_MNPO = 0.50           # obligatory water loss
W_NTS_MNPO = 0.8        # gastric pre-absorptive quench: fast, and before any osmolality

TAU_POA = 1200.0        # 20 min
BETA_POA = 6.0
THETA_POA = 0.30
D_POA = 0.35            # ambient thermal load
W_PB_POA = 0.8          # parabrachial cutaneous thermoafference -- the anatomy is declared,
                        # the channel does not exist yet (see the docstring)
W_SCN_POA = 0.25        # the circadian body-temperature rhythm, driven from the clock

TAU_PVN = 600.0
BETA_PVN = 8.0
THETA_PVN = 0.35
D_PVN = 0.30
W_SCN_PVN = 0.50        # the circadian drive on the stress axis
GC_TAU = 3600.0         # glucocorticoid negative feedback, SURROGATE: the sign and the
GC_G = 0.90             # timescale of a loop whose other two stations are IHM-1's organs

# ---------------------------------------------------------------- what leaves
W_LIGHT_SCN = 1.0       # the retinohypothalamic tract's stand-in; see the docstring
W_VLPO_NM = 0.9         # the sleep switch's inhibitory arm onto the wake-active nuclei
W_LH_ORX = 0.8          # the lateral hypothalamus onto its own orexin cells
W_DRIVE_VAL = 0.7       # hunger, thirst and lateral-hypothalamic tone onto the valuation
                        # system.  THIS is "a hungry animal values food more", declared as
                        # an edge rather than asserted in prose
W_LH_VTA = 0.6
VAL_GAIN = 0.80         # the `Mod` on val.nacc_shell's `gain_in`.  at hyp.arc = 0.86
VAL_GAIN_BASE = 0.30    # (starving, no body attached) the factor is 1.45; at 0.03 (fed)
VAL_GAIN_LO = 0.50      # it is 0.78.  a drive SHAPES the valuation and cannot replace it,
VAL_GAIN_HI = 2.50      # which is what the lo/hi clamp is for
W_SCN_LC = 0.35         # the `circ` loop in ibm/rhythms.py, both arms, with its delays
W_LC_SCN = 0.35
D_LIGHT_SCN = 0.020     # 20 ms, inside the circ loop's declared 20-60 ms
D_CIRC = 0.040
D_VISCERAL = 0.030      # NTS -> hypothalamus and PB -> POA are CENTRAL hops and short.
D_PB_POA = 0.020        # the visceral latency that matters is on the nerve, before these:
                        # ibm/interoception.py measures 448 ms on vagal C and 279 ms on
                        # the greater splanchnic from IHM-1's route lengths, and those are
                        # the afferent relay's to carry, not this edge's
D_SLEEP_SWITCH = 0.015

#: the wake-active neuromodulatory nuclei the sleep-active population opposes.  Named from
#: `NM` rather than spelled out, so there is one locus coeruleus in the brain.
WAKE_ACTIVE = (NM["ha"], NM["na"], NM["5ht"], NM["orx"])


def pops():
    return [
        Pop(id="hyp.scn", n=N, kind="E",
            tau=TAU_SCN, beta=BETA_SCN, theta=THETA_SCN,
            adapt_tau=PER_TAU, adapt_g=PER_G,
            note="the circadian clock.  a relaxation oscillator whose slow variable is the "
                 "PER/CRY repressor; free-runs near 24 h with no input, entrainable by "
                 "light.  NOT a reading of the wall clock."),
        Pop(id="hyp.som", n=N, kind="E",
            tau=TAU_SOM, beta=BETA_SOM, theta=THETA_SOM,
            note="the somnogen: Borbely's process S.  built by waking (hyp.lh), cleared "
                 "faster when asleep through the tau modulation in mods()."),
        Pop(id="hyp.vlpo", n=N, kind="I",
            tau=TAU_VLPO, beta=BETA_VLPO, theta=THETA_VLPO,
            note="ventrolateral preoptic, sleep-active and GABAergic.  half of the "
                 "flip-flop; the other half is hyp.lh here and the wake-active "
                 "neuromodulatory nuclei once neuromodulators.py exists."),
        Pop(id="hyp.lh", n=N, kind="E",
            tau=TAU_LH, beta=BETA_LH, theta=THETA_LH,
            note="lateral hypothalamus: wake-active, and the arousal arm of feeding and "
                 "drinking.  the OREXIN cells are nm.orx and are not duplicated here."),
        Pop(id="hyp.arc", n=N, kind="E",
            tau=TAU_ARC, beta=BETA_ARC, theta=THETA_ARC,
            note="arcuate, energy state.  high = hungry.  a 3 h slow variable whose set "
                 "point is where D_ARC balances the vagal nutrient report."),
        Pop(id="hyp.mnpo", n=N, kind="E",
            tau=TAU_MNPO, beta=BETA_MNPO, theta=THETA_MNPO,
            note="median preoptic, thirst.  no osmoreceptor exists in the body's afferent "
                 "port, so this accumulates tonically and is quenched pre-absorptively."),
        Pop(id="hyp.poa", n=N, kind="E",
            tau=TAU_POA, beta=BETA_POA, theta=THETA_POA,
            note="preoptic thermoregulation, WARM-sensitive class only.  no cold-defence "
                 "population is declared, so the drive is one-sided."),
        Pop(id="hyp.pvn", n=N, kind="E",
            tau=TAU_PVN, beta=BETA_PVN, theta=THETA_PVN,
            adapt_tau=GC_TAU, adapt_g=GC_G,
            note="paraventricular, stress and the HPA axis.  the adaptation is a SURROGATE "
                 "for glucocorticoid feedback: the pituitary and the adrenal are IHM-1's "
                 "organs and the hpa loop cannot close inside ibm/brain/."),
    ]


def internal():
    """the clock's own loop, the sleep switch, and the drives' routes into arousal."""
    return [
        # -- the clock ------------------------------------------------------------
        Proj("hyp.scn", "hyp.scn", weight=W_SCN_REC, topology="dense", delay_s=0.0,
             note="recurrent self-excitation, repressed by the population's own slow "
                  "adaptation: the transcription-translation feedback loop, as dynamics"),

        # -- process S ------------------------------------------------------------
        Proj("hyp.lh", "hyp.som", weight=W_LH_SOM, topology="dense", delay_s=0.0,
             note="waking builds the somnogen.  the sleep pressure is a consequence of the "
                  "wake state, not a timer running beside it"),

        # -- the flip-flop --------------------------------------------------------
        Proj("hyp.som", "hyp.vlpo", weight=W_SOM_VLPO, topology="dense",
             delay_s=D_SLEEP_SWITCH, note="sleep pressure pushes the sleep-active side"),
        Proj("hyp.poa", "hyp.vlpo", weight=W_POA_VLPO, topology="dense",
             delay_s=D_SLEEP_SWITCH, note="warm-sensitive preoptic; a warm body is sleepy"),
        Proj("hyp.lh", "hyp.vlpo", weight=W_LH_VLPO, sign=-1, topology="dense",
             delay_s=D_SLEEP_SWITCH, note="the wake side pressing back"),
        Proj("hyp.vlpo", "hyp.lh", weight=W_VLPO_LH, sign=-1, topology="dense",
             delay_s=D_SLEEP_SWITCH,
             note="and the sleep side pressing back.  these two edges ARE the switch"),

        # -- drives into arousal --------------------------------------------------
        Proj("hyp.arc", "hyp.lh", weight=W_ARC_LH, topology="dense", delay_s=0.005,
             note="hunger raises lateral-hypothalamic tone"),
        Proj("hyp.mnpo", "hyp.lh", weight=W_MNPO_LH, topology="dense", delay_s=0.005,
             note="and so does thirst"),

        # -- the clock's reach inside the hypothalamus ----------------------------
        Proj("hyp.scn", "hyp.pvn", weight=W_SCN_PVN, topology="dense", delay_s=0.005,
             note="the circadian drive on the stress axis; the cortisol rhythm's origin"),
        Proj("hyp.scn", "hyp.poa", weight=W_SCN_POA, topology="dense", delay_s=0.005,
             note="the circadian body-temperature rhythm"),
    ]


def external():
    """light in, the sleep switch out, and the drives' edges into the valuation system.

    Measured 18 September 2026 against the modules that exist: every edge here resolves
    except the four naming `bs.nts` and `bs.pb`, because `brainstem` is the one structure
    they need and it is not written.  `collect()` lists those four with the structure that
    declared them rather than dropping them -- an edge that quietly disappears is a loop
    that quietly stops existing -- and the consequence is exact and worth stating: NO DRIVE
    IN THIS MODULE IS EVER RELIEVED until the brainstem exists.  A hypothalamus with no
    body attached starves, and `hyp.arc` sits at 0.86 saying so.

    (`collect()` reports those four as "id does not exist in a structure that IS built".
    That is its own prefix test comparing `bs` against the module name `brainstem`; every
    structure in this package abbreviates -- `bg`, `nm`, `thal` -- so the label is wrong
    and the edge is right.  `ibm/brain/__init__.py` is out of scope for this module.)
    """
    out = [
        # -- light in: the retinohypothalamic tract's STAND-IN.  see the docstring.
        Proj("thal.lgn.relay", "hyp.scn", weight=W_LIGHT_SCN, topology="dense",
             delay_s=D_LIGHT_SCN,
             note="RETINOHYPOTHALAMIC TRACT, substituted onto the geniculate relay because "
                  "ibm.brain.STRUCTURES has no retina.  the light signal enters hyp.scn "
                  "directly and additively, which is what entrains the oscillator"),

        # -- the circ loop in ibm/rhythms.py, both arms, at its declared delay
        Proj("hyp.scn", NM["na"], weight=W_SCN_LC, topology="dense", delay_s=D_CIRC,
             note="`circ` loop, forward arm: the clock's reach into arousal"),
        Proj(NM["na"], "hyp.scn", weight=W_LC_SCN, topology="dense", delay_s=D_CIRC,
             note="`circ` loop, return arm.  40 ms each way, so the round-trip bound puts "
                  "this ring's CEILING near 12 Hz -- six orders of magnitude above the "
                  "rhythm, which is why the 24 h period is NOT set by it"),

        # -- the sleep switch's inhibitory arm onto the wake-active nuclei
        *[Proj("hyp.vlpo", t, weight=W_VLPO_NM, sign=-1, topology="dense",
               delay_s=D_SLEEP_SWITCH,
               note="the sleep-active population opposing a wake-active nucleus.  the "
                    "return arm is neuromodulators.py's to declare; where it is absent "
                    "the flip-flop closes only through hyp.lh, which is why hyp.lh "
                    "carries the wake side inside this module")
          for t in WAKE_ACTIVE],
        Proj("hyp.lh", NM["orx"], weight=W_LH_ORX, topology="dense", delay_s=0.005,
             note="the lateral hypothalamus onto its own orexin cells.  nm.orx is the "
                  "orexin population and this module does not make a second one"),

        # -- the drives reaching the valuation system.  spec item 4, as edges.
        Proj("hyp.arc", "val.nacc_shell", weight=W_DRIVE_VAL, topology="dense",
             delay_s=0.010,
             note="A HUNGRY ANIMAL VALUES FOOD MORE.  arcuate energy state onto the "
                  "incentive-salience population: the same cue is worth more when the "
                  "drive behind it is high"),
        Proj("hyp.mnpo", "val.nacc_shell", weight=W_DRIVE_VAL, topology="dense",
             delay_s=0.010, note="and a thirsty one values water more"),
        Proj("hyp.lh", "val.nacc_shell", weight=W_DRIVE_VAL, topology="dense",
             delay_s=0.010, note="lateral-hypothalamic tone, the common arousal arm"),
        Proj("hyp.lh", NM["da_vta"], weight=W_LH_VTA, topology="dense", delay_s=0.010,
             note="orexinergic drive onto the dopamine cells: the canonical route by which "
                  "a need raises the gain of the whole valuation system"),

        # -- the body's report in.  DECLARED HERE, and see the trap in the docstring.
        Proj("bs.nts", "hyp.arc", weight=W_NTS_ARC, sign=-1, topology="dense",
             delay_s=D_VISCERAL,
             note="vagal nutrient and distension afference relieving hunger.  "
                  "ibm/interoception.py carries gastric_nutrient and hepatoportal_glucose "
                  "on vagal C at 448 ms and gastric_distension on A-beta at 8 ms; that "
                  "spread is the physiology and the NTS is where it lands"),
        Proj("bs.nts", "hyp.mnpo", weight=W_NTS_MNPO, sign=-1, topology="dense",
             delay_s=D_VISCERAL,
             note="gastric distension quenching thirst pre-absorptively -- before any "
                  "osmolality could have changed, and there is no osmolality channel"),
        Proj("bs.pb", "hyp.poa", weight=W_PB_POA, topology="dense", delay_s=D_PB_POA,
             note="parabrachial cutaneous thermoafference.  the anatomy is declared and "
                  "the channel is not: no port in ibm/interoception.py reports a "
                  "temperature"),
        Proj("hyp.pvn", "bs.nts", weight=0.4, sign=-1, topology="dense", delay_s=0.010,
             note="the descending autonomic arm: the hypothalamus answering the viscera "
                  "it just read"),
    ]
    return out


def mods():
    """two modulatory edges, both inside the hypothalamus, both of them mechanisms.

    THE THIRD EDGE EXISTS BECAUSE AN ENGINE GAP WAS CLOSED WHILE THIS FILE WAS BEING
    WRITTEN, and the history is worth one paragraph because it explains why the module says
    the same thing twice in two different ways.  `collect()` used to pass `mods()` to
    `Circuit` UNFILTERED while filtering `external()` into an orphan report, and
    `Circuit.__init__` asserted a `Mod`'s endpoints exist -- so a modulatory edge into a
    structure nobody had written yet crashed the assembly of the WHOLE BRAIN, while the
    identical `Proj` was merely reported.  Measured on 18 September 2026, before the fix:
    `ibm.brain.assemble()` over everything on disk raised `nm.nbm->hpc.ms.E: unknown` from
    `neuromodulators.py`, and with that module removed, `nm.snc->bg.d1.motor: unknown` from
    `basal_ganglia.py`.  Three modules including this one had to route around it.  It is
    fixed now -- `collect()` filters `mods` and reports the dropped ones as orphans -- so
    the gain modulation this module always wanted can be declared.

    It is declared BESIDE the projection rather than instead of it, because the two say
    different things and the physiology has both: `hyp.arc -> val.nacc_shell` ADDS drive
    (a hungry animal's accumbens is more active with no cue at all), and this `Mod`
    MULTIPLIES the gain (the same cue is worth more).  `scripts/gate_brain_hypothalamus.py`
    severs both together, because a control that cuts one arm of a two-arm pathway is not
    a control.
    """
    return [
        Mod(src="hyp.vlpo", dst="hyp.som", param="tau",
            gain=SOM_CLEAR_GAIN, baseline=0.0, lo=SOM_CLEAR_LO, hi=SOM_CLEAR_HI,
            note="sleep clears the somnogen faster than waking builds it.  `Pop` has one "
                 "tau, so the asymmetry process S needs is a state-dependent PARAMETER, "
                 "which is exactly what a Mod edge is for"),
        Mod(src="hyp.scn", dst="hyp.vlpo", param="theta",
            gain=PROCESS_C_GAIN, baseline=PROCESS_C_BASE,
            lo=PROCESS_C_LO, hi=PROCESS_C_HI,
            note="process C: the clock raises the sleep gate's threshold in subjective day "
                 "and lowers it at night, so the same sleep pressure buys sleep at one "
                 "phase and not at another"),
        Mod(src="hyp.arc", dst="val.nacc_shell", param="gain_in",
            gain=VAL_GAIN, baseline=VAL_GAIN_BASE, lo=VAL_GAIN_LO, hi=VAL_GAIN_HI,
            note="A HUNGRY ANIMAL VALUES FOOD MORE, in its strongest form: the arcuate's "
                 "energy state MULTIPLIES the incentive-salience population's input gain, "
                 "so the same cue is worth more.  The projection beside it adds drive; "
                 "this changes what a cue is worth.  An orphan until valuation is in the "
                 "assembly, and `collect()` reports it as one rather than crashing"),
    ]


def drives():
    """the tonic accumulations, which are what make a drive a drive.

    `D_ARC`, `D_MNPO` and `D_POA` are the body running down: fuel burnt, water lost, heat
    gained.  `D_LH` is the ascending reticular wake tone, which is the brainstem's once
    `brainstem.py` exists and is here until then -- visible, in this dict, rather than as a
    float inside a step function.  `D_PVN` is basal corticotroph tone.

    `hyp.scn` gets NOTHING, and that is the point: the clock is autonomous, so the free-run
    gate runs at literally zero input.  `hyp.som` and `hyp.vlpo` get nothing either -- both
    are driven by the circuit and by no one else.
    """
    return {"hyp.lh": D_LH, "hyp.arc": D_ARC, "hyp.mnpo": D_MNPO,
            "hyp.poa": D_POA, "hyp.pvn": D_PVN}


def targets():
    circ = RHYTHM["circadian"]
    return {
        # empty ON PURPOSE.  a drive is a level; divisive normalisation would hold the mean
        # bounded, and the mean is the whole signal.  see the docstring.
        "sparsity": {},
        "bands": {
            "circadian": {"pops": ["hyp.scn"],
                          "hz": circ.band,
                          "period_s": circ.measure["period_s"],
                          "note": "`circadian` in ibm/rhythms.py, loop `circ`, set_by "
                                  "'time constants'.  measured with peak_prominence beside "
                                  "the frequency, never alone"},
        },
        "known_answers": {
            "free_run": {
                "claim": "with NO input at all, hyp.scn oscillates with a period near 24 h",
                "measure": "mean interval between upward mean-crossings of hyp.scn, and "
                           "peak_prominence of the circadian band quoted beside it",
                "expect": "period in [22, 26] h and prominence > 0",
                "note": "a frequency without a prominence is not evidence (CLAUDE.md)"},
            "entrainment": {
                "claim": "a light signal offset from the free-running period pulls the "
                         "clock onto it, over a bounded range and not beyond it",
                "measure": "period of hyp.scn under a 12 h-on square light input at the "
                           "declared test periods, the drive being W_LIGHT_SCN times the "
                           "retinal rate the edge would carry",
                "expect": "locks for |T_light - T_free| <= ~2 h; FAILS to lock at 20 h "
                          "and at 32 h",
                "note": "an oscillator that entrains to everything is not an oscillator, "
                        "it is a filter.  the failure half of this is the half that says "
                        "the clock has a period of its own"},
            "sleep_pressure": {
                "claim": "hyp.som rises while hyp.lh is the winning side and falls while "
                         "hyp.vlpo is, and it clears faster than it builds",
                "measure": "10-90% rise time under forced wake and fall time under forced "
                           "sleep, and the same quantity over a spontaneous cycle",
                "expect": "rise tau = TAU_SOM, fall tau = TAU_SOM * clamp(1 + "
                          "SOM_CLEAR_GAIN * mean(hyp.vlpo)), and rise > fall",
                "note": "human process-S fits give ~18 h rise against ~4 h decay"},
            "drive_changes_valuation": {
                "claim": "the same cue produces a larger response in val.nacc_shell when "
                         "hyp.arc is high than when it is low, and severing the declared "
                         "hyp -> val edges abolishes the difference",
                "measure": "response of val.nacc_shell to a fixed cue, hungry vs sated, "
                           "with an edge-severed control arm",
                "expect": "hungry > sated, and severed difference ~= 0",
                "note": "gate on an ablation, never on a magnitude (CLAUDE.md).  until "
                        "ibm/brain/valuation.py exists this is measured against a stand-in "
                        "population built inside the gate script and nowhere else"},
        },
        "not_claimed": {
            "cortisol_ultradian": {
                "why": "`hpa` runs pvn -> pituitary -> adrenal -> pvn and two of those "
                       "three stations are IHM-1's organs, not populations.  hyp.pvn's "
                       "adaptation is a surrogate for the feedback's sign and timescale "
                       "and cannot produce the loop's ~1 h pulse.  the row stays "
                       "unclaimed rather than being scored on a circuit that is not the "
                       "circuit it describes",
                "rhythm": "cortisol_ultradian, 2.0e-4 - 4.0e-4 Hz"},
            "cold_defence": {
                "why": "hyp.poa is the warm-sensitive class only; no cold-sensitive "
                       "population is declared, so thermal discomfort here is one-sided"},
            "osmotic_thirst": {
                "why": "ibm/interoception.py declares no plasma-osmolality channel, so "
                       "hyp.mnpo's accumulation is tonic and its only measured relief is "
                       "the gastric pre-absorptive one"},
        },
    }
