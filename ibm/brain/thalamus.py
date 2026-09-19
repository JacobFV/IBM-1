"""thalamus: eight relay nuclei, each with its own reticular sector and its own cortex.

Follows `ibm/brain/cortex.py`, the reference implementation of this package's contract:
`pops`, `internal`, `external`, `mods`, `drives`, `targets`, and nothing else.

WHAT THIS IS FOR
----------------
`docs/BRAIN_SPEC.md`, "Thalamus": the relay nuclei as SEPARATE populations -- LGN, MGN,
VPL, VL, VA, MD, pulvinar, centromedian -- each with its own cortical partner, plus the
reticular nucleus as a SECTOR PER NUCLEUS.  The sectors are the point.  A single global
reticular population cannot gate one modality without gating all of them, and selective
gating is what a thalamus is for; `ibm/thalamus.py`'s five lumped units could not express
it and its gating gate failed twice (docs/LOG.md, 18 Sep).

Two kinds of nucleus, and they are different objects:

  FIRST ORDER  (LGN, MGN, VPL, VL, VA)  a subcortical DRIVER -- retina, inferior
      colliculus, medial lemniscus, cerebellar nuclei, pallidum -- reaches cortex through
      the relay.  The cortical input to these nuclei is layer 6, which MODULATES.
  HIGHER ORDER (pulvinar, MD)  the driver is CORTEX itself, layer 5.  Area A drives the
      relay, the relay drives area B, and that is how two cortical areas communicate
      without a direct fibre.  It is visible in `external()` as a cortex -> relay edge
      carrying `role="driver"` beside a relay -> cortex edge into a DIFFERENT area, and
      the pairs are chosen so that `ibm/brain/cortex.py` declares no direct projection
      between them (asserted in `higher_order_pairs()`, checked by T5).

  CENTROMEDIAN is the intralaminar exception: diffuse and weak to cortex, strong to the
      striatum, which is the one thalamic efferent this module declares to a structure
      other than cortex.

THE REBOUND, AND WHAT THE ENGINE CAN AND CANNOT SAY
---------------------------------------------------
`ibm/thalamus.py`'s first version inverted the T-current's activation and became a
pacemaker at ~8 Hz whatever else was happening (docs/LOG.md).  The current is a WINDOW:
availability, which rises with hyperpolarisation, TIMES activation, which rises with
depolarisation -- so an armed cell still needs a push to fire.

`ibm/circuit.py` has no T-current and this module does not add one to the engine.  The
window is built from two primitives the engine does have, and both halves are real
mechanisms rather than stand-ins:

  * `adapt_tau` / `adapt_g` on the relay population.  `a` relaxes toward `adapt_g * r`, so
    hyperpolarisation DE-ACTIVATES it over `adapt_tau` and releasing the hyperpolarisation
    leaves the cell with `u` larger than its own steady state -- a peak that arrives after
    the release and decays as `a` rebuilds.  `adapt_tau` is the de-inactivation time; it is
    what `tau_h_up` was in v2.
  * `depress` on the relay population.  The engine multiplies what a depressing population
    SENDS by its resource `x`, which recovers during silence, so the transmitted burst
    would be literally availability x activation.  Thalamocortical synapses depress
    strongly, which is why a burst is detected by cortex better than the same number of
    tonic spikes.  **DECLARED AND MEASURED INERT -- see the next paragraph.**

At the membrane the two terms are summed inside the sigmoid rather than multiplied.  Below
threshold a sigmoid's foot is exponential, so a sum there IS a product, and the property
that matters is testable either way: **an armed cell with no push must not fire.**  T2's
second arm measures exactly that, and T2's third arm confirms that with the reticular
sector silenced the relay does not oscillate -- the two symptoms the inverted version had.

THE SYNAPTIC HALF DOES NOT ENGAGE, AND THE REASON IS IN THE ENGINE
------------------------------------------------------------------
`tau_rec` and `u_tc` came back INERT in T6 (0.09 Hz each over a 3x sweep), so the state
variable was printed rather than the sweep repeated, and the depression resource `x` sits
at **0.9913 with a range of 0.0002 over six seconds** -- it is pinned.

`ibm/circuit.py` line 446 advances it with `kx = 1/tau_rec + use * r * R_MAX / 100.0`.
`R_MAX` is 100, so the depletion rate is `U * r`, at most 0.25 per second against a
recovery of `1/0.150` = 6.67 -- a synapse that cannot be depressed by any rate in [0, 1].
The facilitation branch eight lines below uses `p.U * r * R_MAX`, a hundred times larger,
and the engine's own comment there says that writing this term with the normalised rate
instead of the spike rate makes it "~100x too small, which cost an afternoon in
ibm/substrate.py".  The depression branch is in exactly that state.  It is not fixed here
because this module may not edit the engine; it is REPORTED, and `depress=True` is left
declared because thalamocortical depression is real and the declaration is right.  Every
`depress=True` population in `ibm/brain/` -- which is every cortical excitatory population
-- is in the same position.

WHAT SETS THE SPINDLE CLOCK
---------------------------
v2's sensitivity sweep found the clock is the GABA-A decay and the relay membrane, not the
T-current's de-inactivation (`tau_h_up` moved the peak 0.84 Hz over a 6.5x sweep).  So the
GABA-A and GABA-B synapses are declared HERE as populations whose `tau` IS the decay:

    thal.<nuc>.gabaa   tau = 40 ms   the fast inhibition the spindle is paced by
    thal.<nuc>.gabab   tau = 150 ms  slow, and it takes a hard TRN drive to recruit

This is the one modelling decision forced by the engine rather than by the anatomy.  A
`Proj` carries a conduction delay and no synaptic decay, and a rate population's `tau` is
its membrane, so a 40 ms IPSC has nowhere else to live.  It is reported as a missing engine
feature; it is not a population anyone would draw on a diagram.

AROUSAL IS NOT A PARAMETER HERE
-------------------------------
v2 had an `arousal` float.  This module has no dial and no equivalent of one.  The relay
cells' polarisation is set by what reaches them: the tonic drive in `drives()`, the layer-6
corticothalamic input, and brainstem neuromodulation.

`mods()` returns [] -- **the thalamus originates no neuromodulation**.  It EXPECTS, from
`ibm/brain/neuromodulators.py`, exactly these edges, which that module must declare because
it owns the populations at their source:

    Mod(NM["ach_bs"] = "nm.ppt",  dst=f"thal.{x}.relay", param="theta", gain<0)
        brainstem acetylcholine depolarises relay cells: the threshold falls, the cell
        leaves the burst window, transmission becomes tonic.  This is waking and REM.
    Mod(NM["ach_bs"] = "nm.ppt",  dst=f"thal.trn.{x}",  param="theta", gain>0)
        ACh HYPERPOLARISES reticular cells, which is the other half of the same switch.
    Mod(NM["na"] = "nm.lc",       dst=f"thal.{x}.relay", param="beta",  gain>0)
        noradrenaline sharpens the transfer function -- gain, not offset.

They are not declared here for a reason worth writing down: `ibm/brain/__init__.py`'s
`collect()` filters a `Proj` whose other end is missing into `report["orphans"]`, but it
does NOT filter `Mod` edges, and `Circuit.__init__` asserts every `Mod`'s endpoints exist.
A `Mod` naming `nm.ppt` while `neuromodulators.py` is unwritten therefore CRASHES the
assembly of {cortex, thalamus} instead of being reported.  An edge that is declared by the
structure that owns its source cannot do that.

UNITS: seconds, rates in [0, 1], delays in seconds, and the delays are the catalogue's
(`ibm/rhythms.py`: relay->cortex 1-3 ms, cortex->reticular 3-6 ms, reticular->relay
1-2 ms).
"""
from __future__ import annotations

from dataclasses import dataclass

from ibm.brain import Mod, Pop, Proj  # noqa: F401

STRUCTURE = "thal"

CTX = "ctx"          # the structure whose population ids the thalamocortical edges name


# ======================================================================================
# the declared constants.  ONE object, so that `scripts/gate_brain_thalamus.py`'s T6 can
# sweep every one of them and list the ones that change nothing -- which is this
# programme's cheapest detector of a mechanism that is not wired in (CLAUDE.md).
# ======================================================================================
@dataclass(frozen=True)
class Constants:
    # --- membranes.  Relay cells 10-20 ms, reticular cells faster.  With the GABA-A decay
    # below these are the spindle clock: v2 measured tau_R and tau_A moving the peak from
    # 18.7 to 13.1 Hz while the T-current's de-inactivation moved it 0.84 Hz over 6.5x.
    tau_relay: float = 0.016
    tau_trn: float = 0.008
    # --- the two inhibitory species, as first-order filters (see the module docstring for
    # why they are populations).  GABA-A: 20-40 ms in thalamocortical cells, slower than a
    # cortical IPSC.  GABA-B: hundreds of ms, and it needs a HARD reticular drive, which is
    # `theta_gabab` -- a burst of TRN firing, not a trickle.
    tau_gabaa: float = 0.040
    tau_gabab: float = 0.150
    theta_gabaa: float = 0.45
    theta_gabab: float = 0.60
    beta_syn: float = 10.0
    # --- the T-current, as availability x activation (module docstring).
    t_deinact_s: float = 0.060      # de-inactivation time: v2's tau_h_up
    g_t: float = 0.55               # the burst's size: v2's g_T
    # --- thalamocortical synaptic depression: the other half of the window, and the reason
    # a burst is worth more to cortex than the same rate delivered tonically.  MEASURED
    # INERT (T6: 0.09 Hz each over a 3x sweep; the resource sits at 0.9913 +- 0.0002),
    # because the engine's depletion term is ~100x too small -- module docstring.
    tau_rec: float = 0.150
    u_tc: float = 0.25
    # --- transfer functions
    beta_relay: float = 9.0
    theta_relay: float = 0.35
    beta_trn: float = 8.0
    theta_trn: float = 0.30
    # --- intrathalamic weights
    w_relay_trn: float = 1.10       # relay -> its own reticular sector; the shell is driven
                                    # by the relay it gates
    w_gabaa: float = 1.30           # gabaa pool -> relay
    w_gabab: float = 0.35           # gabab pool -> relay
    w_trn_trn: float = 0.25         # the sector's own inhibition
    # --- corticothalamic, layer 6: MODULATORY, and it reaches the shell as well as the
    # relay, which is how cortex shuts its own input off.
    w_ct_relay: float = 0.35
    w_ct_trn: float = 0.55
    # --- thalamocortical
    w_tc_e: float = 0.55            # relay -> cortical excitatory
    w_tc_i: float = 0.25            # relay -> cortical interneurons (feedforward inhibition)
    # --- higher order: the layer-5 DRIVER is strong, which is what makes it a driver and
    # not a modulator.  It is the one cortical input to a thalamic relay that is not layer 6.
    w_ho_driver: float = 0.95
    # --- intralaminar
    w_cm_ctx: float = 0.10          # diffuse and weak
    w_cm_str: float = 0.60          # and strong to the striatum
    # --- background
    sigma_relay: float = 0.030
    sigma_trn: float = 0.020
    # --- conduction, from `ibm/rhythms.py`'s tct, rgs, smu and aud loops
    d_relay_cortex: float = 0.002   # 1-3 ms
    d_cortex_trn: float = 0.005     # 3-6 ms
    d_trn_relay: float = 0.0015     # 1-2 ms
    d_relay_trn: float = 0.0015     # 1-2 ms (rgs: lgn -> trn)
    d_cortex_relay: float = 0.006   # layer 6 onto the relay (rgs: v1 -> lgn, 5-12 ms)


#: the live constants.  Rebound to sweep them; every function below reads this object at
#: call time rather than closing over its fields, so a sweep is `TH.C = replace(TH.C, ...)`
#: and not an edit.
C = Constants()

N_RELAY = 32
N_TRN = 16
N_SYN = 8          # the two synaptic pools; they carry a scalar conductance, not a code


# ======================================================================================
# the nuclei
# ======================================================================================
@dataclass(frozen=True)
class Nucleus:
    id: str
    order: str                  # "first" | "higher" | "intralaminar"
    modality: str
    driver: str                 # what drives it, in words.  For a higher-order nucleus
                                # this is a cortical area id.
    to_cortex: tuple            # the cortical areas its relay reaches
    l6_from: tuple              # the areas whose layer 6 feeds back onto it and its sector
    note: str = ""


#: LGN, MGN, VPL, VL, VA, MD, pulvinar, centromedian -- `docs/BRAIN_SPEC.md`'s list, in its
#: order.  VPL carries VPM's trigeminal territory too: the two have one cortical partner in
#: this atlas (`postcentral`) and splitting them would claim a resolution the atlas does not
#: have, so they are one population and this line says so.
NUCLEI = (
    Nucleus("lgn", "first", "visual", "retina (retinogeniculate)",
            ("pericalcarine",), ("pericalcarine",),
            note="the driver is the retina, and there is no retinal structure in "
                 "ibm/brain's STRUCTURES -- see `missing_afferents()`"),
    Nucleus("mgn", "first", "auditory", "inferior colliculus (lateral lemniscus)",
            ("transversetemporal",), ("transversetemporal", "superiortemporal")),
    Nucleus("vpl", "first", "somatosensory", "medial lemniscus and trigeminal lemniscus",
            ("postcentral",), ("postcentral",),
            note="VPL and VPM together: one cortical partner in this atlas"),
    Nucleus("vl", "first", "motor", "cerebellar deep nuclei",
            ("precentral",), ("precentral",),
            note="cerebellar-recipient; the `ctc` loop's thalamic station"),
    Nucleus("va", "first", "premotor", "internal pallidum and SNr",
            ("caudalmiddlefrontal", "paracentral"), ("caudalmiddlefrontal",),
            note="pallidal-recipient; the `cbgtc` loop's thalamic station.  Its driver is "
                 "TONICALLY ACTIVE and inhibitory, so this relay is released by a pause"),
    Nucleus("md", "higher", "prefrontal", "ctx.rostralmiddlefrontal.E",
            ("lateralorbitofrontal",), ("lateralorbitofrontal",),
            note="its layer-6 feedback comes from the area it PROJECTS to and not from the "
                 "area that drives it: listing the driving area here as well declared "
                 "ctx.rostralmiddlefrontal.E -> thal.md.relay twice, once as a layer-5 "
                 "driver and once as layer-6 modulation, and the engine would have run the "
                 "pathway at the sum of the two weights.  A higher-order nucleus's driver "
                 "and its modulator are different areas by definition"),
    Nucleus("pulvinar", "higher", "visual", "ctx.pericalcarine.E",
            ("inferiorparietal",), ("inferiorparietal",)),
    Nucleus("cm", "intralaminar", "diffuse", "brainstem reticular formation",
            ("precentral", "caudalmiddlefrontal", "superiorfrontal"), ("precentral",),
            note="centromedian/parafascicular: weak and diffuse to cortex, strong to the "
                 "striatum, and the one thalamic efferent here that is not cortical"),
)

BY_ID = {n.id: n for n in NUCLEI}

#: the higher-order pairs, written out: (nucleus, driving area, driven area).  A second
#: pulvinar pair is declared so the nucleus is not a single wire.  Every pair is chosen so
#: that `ibm/brain/cortex.py` declares NO direct projection between the two areas -- that is
#: the claim the nucleus exists to make, and `higher_order_pairs()` asserts it.
HIGHER_ORDER = (
    ("pulvinar", "pericalcarine", "inferiorparietal"),
    ("pulvinar", "lateraloccipital", "superiortemporal"),
    ("md", "rostralmiddlefrontal", "lateralorbitofrontal"),
)

#: where the intralaminar efferent lands.  Centromedian reaches the MOTOR striatum and
#: parafascicular the ASSOCIATIVE one; both cell types, because the thalamostriatal input is
#: not a dopamine-like bias.  The ids follow `ibm/brain/basal_ganglia.py`'s own convention
#: (`bg.d1.<channel>`), which departs from the contract docstring's `bg.str.d1` on purpose;
#: if that module renames them these edges become orphans in the assembly report, which is
#: the whole reason the report exists.
STRIATAL_TARGETS = ("bg.d1.motor", "bg.d2.motor", "bg.d1.associative", "bg.d2.associative")


#: THE TWO POLARISATIONS, declared here so a gate reads them rather than inventing them.
#:
#: These are INPUTS -- what reaches the relay cells and the reticular sectors from the
#: brainstem -- and not parameters of the dynamics.  There is no dial: with
#: `ibm/brain/neuromodulators.py` assembled, `nm.ppt` produces the waking row by lowering
#: the relay's threshold and raising the sector's, and these numbers are what that edge is
#: doing when it is absent.  A gate run on the thalamus alone supplies them as `drive`,
#: which is standing in for the brainstem, not switching a mode.
#:
#: `nrem2` is the DEFAULT (`drives()`): a thalamus with no cholinergic tone is a sleeping
#: thalamus.  `wake` depolarises the relay and takes the sector down with it -- both halves,
#: because a relay depolarised with the gate still shut is not an open thalamus.
POLARISATION = {
    "nrem2": {"relay": 0.70, "trn": 0.05},
    "wake": {"relay": 1.00, "trn": -0.35},
}


def relay_id(x: str) -> str:
    return f"{STRUCTURE}.{x}.relay"


def trn_id(x: str) -> str:
    """the reticular SECTOR for nucleus x.  `thal.trn.lgn`, not `thal.trn`."""
    return f"{STRUCTURE}.trn.{x}"


def gabaa_id(x: str) -> str:
    return f"{STRUCTURE}.{x}.gabaa"


def gabab_id(x: str) -> str:
    return f"{STRUCTURE}.{x}.gabab"


def higher_order_pairs(check=True) -> tuple:
    """the (nucleus, from_area, to_area) triples, with the no-direct-fibre claim checked.

    The check imports `ibm/brain/cortex.py` and looks for a projection between the two
    areas in either direction.  If cortex's connection rule changes so that a pair becomes
    directly connected, this raises rather than leaving T5 measuring a cortical fibre and
    calling it a pulvinar.
    """
    if check:
        from ibm.brain import cortex as CX
        direct = {(p.src, p.dst) for p in CX.external()}
        for nuc, a, b in HIGHER_ORDER:
            for u, v in ((a, b), (b, a)):
                key = (f"{CTX}.{u}.E", f"{CTX}.{v}.E")
                assert key not in direct, (
                    f"{nuc}: {a} and {b} are directly connected in ibm/brain/cortex.py, so "
                    f"a higher-order relay between them proves nothing")
    return HIGHER_ORDER


def missing_afferents() -> dict:
    """the subcortical drivers of the first-order nuclei, and who owns each edge.

    NOT declared here.  `ibm/brain/__init__.py`'s rule is that an edge belongs to the
    structure the pathway is named for -- corticostriatal to the basal ganglia,
    thalamocortical to the thalamus -- so the cerebellothalamic, pallidothalamic and
    lemniscal projections belong to the cerebellum, the basal ganglia and the cord.  Until
    those modules exist a first-order relay has no driver and must be given one as `drive`,
    and this function is how a gate or an assembly report says so out loud instead of the
    nuclei quietly relaying nothing.
    """
    return {
        "lgn": {"from": "retina", "owner": "NO STRUCTURE -- there is no retina in "
                                            "ibm/brain's STRUCTURES", "declared": False},
        "mgn": {"from": "inferior colliculus", "owner": "brainstem", "declared": False},
        "vpl": {"from": "medial lemniscus / dorsal column nuclei", "owner": "cord or "
                "brainstem", "declared": False},
        "vl": {"from": "cerebellar deep nuclei", "owner": "cerebellum", "declared": False},
        "va": {"from": "GPi / SNr", "owner": "basal_ganglia", "declared": False},
        "cm": {"from": "brainstem reticular formation", "owner": "brainstem",
               "declared": False},
    }


# ======================================================================================
# the contract
# ======================================================================================
def pops():
    out = []
    for nuc in NUCLEI:
        x = nuc.id
        out.append(Pop(
            id=relay_id(x), n=N_RELAY, kind="relay",
            tau=C.tau_relay, beta=C.beta_relay, theta=C.theta_relay,
            # the T-current as availability: de-activated by hyperpolarisation over
            # `t_deinact_s`, so releasing the hyperpolarisation leaves a peak behind
            adapt_tau=C.t_deinact_s, adapt_g=C.g_t,
            # and as a multiplicative gate on what the burst DELIVERS -- declared, and
            # measured inert: the engine cannot deplete this resource (module docstring)
            depress=True, tau_rec=C.tau_rec, U=C.u_tc,
            sigma=C.sigma_relay,
            # NO `sparsity`: a relay's level is set by its input, which is what a relay is.
            # (ibm/circuit.py's Pop docstring: right for a relay, wrong for a cortex.)
            note=f"{nuc.order}-order {nuc.modality} relay; driver: {nuc.driver}"))
        out.append(Pop(
            id=trn_id(x), n=N_TRN, kind="I",
            tau=C.tau_trn, beta=C.beta_trn, theta=C.theta_trn, sigma=C.sigma_trn,
            note=f"reticular sector for {x}: the gate on this nucleus and no other"))
        out.append(Pop(
            id=gabaa_id(x), n=N_SYN, kind="I",
            tau=C.tau_gabaa, beta=C.beta_syn, theta=C.theta_gabaa,
            note="GABA-A conductance from the sector, as a population because a Proj has "
                 "no synaptic decay; its tau IS the 40 ms IPSC and it is the spindle clock"))
        out.append(Pop(
            id=gabab_id(x), n=N_SYN, kind="I",
            tau=C.tau_gabab, beta=C.beta_syn, theta=C.theta_gabab,
            note="GABA-B: slow, and its high threshold is the requirement that the sector "
                 "fire hard before it is recruited at all"))
    return out


def internal():
    """the relay-reticular loop, once per nucleus, and nothing crossing between sectors.

    There is deliberately no reticular-to-reticular projection BETWEEN sectors.  It exists
    in the anatomy and it would make the sectors compete; leaving it out is the cleanest
    statement of the claim this structure is built to make -- that gating one modality
    leaves the others where they were.  T4 is that claim, and a cross-sector edge would
    make it pass for a different reason.
    """
    out = []
    for nuc in NUCLEI:
        x = nuc.id
        out.append(Proj(relay_id(x), trn_id(x), weight=C.w_relay_trn,
                        delay_s=C.d_relay_trn, topology="dense",
                        note="the relay drives its own sector"))
        out.append(Proj(trn_id(x), gabaa_id(x), weight=1.0, delay_s=C.d_trn_relay,
                        topology="dense", note="conduction; the decay is the pool's tau"))
        out.append(Proj(trn_id(x), gabab_id(x), weight=1.0, delay_s=C.d_trn_relay,
                        topology="dense"))
        out.append(Proj(gabaa_id(x), relay_id(x), weight=C.w_gabaa, sign=-1,
                        topology="dense",
                        note="the fast inhibition the spindle is paced by"))
        out.append(Proj(gabab_id(x), relay_id(x), weight=C.w_gabab, sign=-1,
                        topology="dense",
                        note="the slow one; it deepens, it does not pace"))
        out.append(Proj(trn_id(x), trn_id(x), weight=C.w_trn_trn, sign=-1,
                        delay_s=0.001, topology="dense",
                        note="the sector's own inhibition"))
    return out


def external():
    """thalamocortical, corticothalamic and thalamostriatal.

    The cortex <-> X convention of `ibm/brain/__init__.py`: the non-cortical structure
    declares both directions, so these are the thalamus's.  The subcortical DRIVERS of the
    first-order nuclei are not here -- see `missing_afferents()` for who owns each and why
    that matters.
    """
    out = []
    for nuc in NUCLEI:
        x = nuc.id
        w_e = C.w_cm_ctx if nuc.order == "intralaminar" else C.w_tc_e
        for area in nuc.to_cortex:
            out.append(Proj(relay_id(x), f"{CTX}.{area}.E", weight=w_e,
                            topology="sparse", p=0.25, delay_s=C.d_relay_cortex,
                            note=f"thalamocortical, {nuc.order} order"))
            if nuc.order != "intralaminar":
                out.append(Proj(relay_id(x), f"{CTX}.{area}.I", weight=C.w_tc_i,
                                topology="sparse", p=0.25, delay_s=C.d_relay_cortex,
                                note="feedforward inhibition: the relay contacts "
                                     "interneurons too, which is how a thalamic input "
                                     "can sharpen rather than only add"))
        for area in nuc.l6_from:
            out.append(Proj(f"{CTX}.{area}.E", trn_id(x), weight=C.w_ct_trn,
                            topology="sparse", p=0.25, delay_s=C.d_cortex_trn,
                            note="layer 6 collateral into the shell -- it arrives BEFORE "
                                 "the same axon's branch onto the relay, which is how "
                                 "cortex can shut its own input off"))
            out.append(Proj(f"{CTX}.{area}.E", relay_id(x), weight=C.w_ct_relay,
                            topology="sparse", p=0.25, delay_s=C.d_cortex_relay,
                            note="layer 6 corticothalamic: MODULATORY"))
    # the higher-order drivers: layer 5, strong, and into a nucleus whose output goes
    # somewhere else entirely.  This pair of edges is the cortico-thalamo-cortical route.
    for nuc_id, a, b in HIGHER_ORDER:
        out.append(Proj(f"{CTX}.{a}.E", relay_id(nuc_id), weight=C.w_ho_driver,
                        topology="sparse", p=0.30, delay_s=C.d_cortex_relay,
                        note=f"LAYER 5 DRIVER: {a} drives {nuc_id}, which drives {b}.  "
                             f"Not layer 6, not modulatory -- this is the transthalamic "
                             f"route between two areas with no direct fibre"))
        if b not in BY_ID[nuc_id].to_cortex:
            out.append(Proj(relay_id(nuc_id), f"{CTX}.{b}.E", weight=C.w_tc_e,
                            topology="sparse", p=0.25, delay_s=C.d_relay_cortex,
                            note=f"the second limb of the {a} -> {nuc_id} -> {b} route"))
    # thalamostriatal: named for the thalamus, so declared here.  `bg` is unwritten, so
    # these are reported as orphans and the intralaminar nucleus's main job is on hold.
    for tgt in STRIATAL_TARGETS:
        out.append(Proj(relay_id("cm"), tgt, weight=C.w_cm_str, topology="sparse", p=0.20,
                        delay_s=0.004,
                        note="thalamostriatal, centromedian/parafascicular -- an orphan "
                             "whenever ibm/brain/basal_ganglia.py is not assembled"))
    # no pathway declared twice.  `Circuit` now raises on a duplicate because two entries
    # run the pathway at the SUM of their weights with nothing reporting it, and this
    # module produced one: MD's layer-5 driver and a layer-6 edge from the same area.
    keys = [q.key for q in out]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    assert not dups, f"thalamus declares these pathways twice: {dups}"
    return out


def mods():
    """the thalamus originates no neuromodulation.

    It receives brainstem acetylcholine (`nm.ppt`) and noradrenaline (`nm.lc`), and those
    edges are declared by `ibm/brain/neuromodulators.py`, which owns their source
    populations.  The module docstring names the three of them, their parameters and their
    signs, and says why declaring them from this side would crash the assembly rather than
    be reported as an orphan.

    There is no `arousal` float here and no substitute for one.
    """
    return []


def drives():
    """the tonic drive each population sits at.

    The relay's drive is its POLARISATION, and that is the whole of what v2's `arousal`
    dial was.  It is an INPUT, not a parameter: `nm.ppt`'s cholinergic edge moves it (by
    the relay's threshold), the layer-6 corticothalamic projection moves it, and nothing in
    this module reads a state flag.

    The value declared here is the polarisation of a thalamus with NO brainstem tone, and
    that state is NREM: hand the relay-reticular loop nothing and it spindles.  Waking is
    what `nm.ppt` does to this, not a different default -- see `POLARISATION`.

    A HIGHER-ORDER nucleus gets a much smaller one, and the asymmetry is the point.  A
    first-order relay's driver -- retina, lemniscus, cerebellum, pallidum -- is not declared
    anywhere yet (`missing_afferents()`), so its tonic drive is standing in for a real
    input.  The pulvinar's and MD's driver IS declared: it is the layer-5 cortical
    projection in `external()`.  Giving them the same tonic drive would count that input
    twice and saturate them, which is measurable -- at the first-order level the pulvinar
    relay runs at 100 Hz and its transfer slope to the cortex it drives is zero.
    """
    d = {}
    for nuc in NUCLEI:
        d[relay_id(nuc.id)] = (0.10 if nuc.order == "higher"
                               else POLARISATION["nrem2"]["relay"])
        d[trn_id(nuc.id)] = POLARISATION["nrem2"]["trn"]
    return d


def targets():
    return {
        # a relay is not sparse-coded: its level is its input's.  Declared empty on
        # purpose, so `calibrate_sparsity` has nothing to do here and cannot silently
        # invent an operating point for a population that should not have one.
        "sparsity": {},
        "polarisation": POLARISATION,
        "bands": {
            "spindles": {
                "pops": [relay_id(n.id) for n in NUCLEI] +
                        [trn_id(n.id) for n in NUCLEI],
                "hz": (11.0, 16.0),
                "note": "the reticular-relay loop at sleep-like polarisation.  13.45 Hz "
                        "measured on held-out sleepers (scripts/fit_sleep_resonance.py); "
                        "13.626 Hz measured on ibm/thalamus.py, this module's known answer"},
            "delta": {
                "pops": [relay_id(n.id) for n in NUCLEI],
                "hz": (1.0, 4.0),
                "note": "NOT claimed.  v2's T7 failed at -1.21 decades and the diagnosis "
                        "was structural, not a parameter; this module inherits the failure "
                        "and does not assert otherwise"},
        },
        "known_answers": {
            "rebound_after_release": {
                "claim": "a hyperpolarising pulse then released gives a relay peak AFTER "
                         "the release, larger than the value during the pulse",
                "measure": "argmax of the relay rate after the pulse onset",
                "expect": "peak step > release step",
                "note": "v2 measured 12 ms after release, 0.535 against 0.052 during"},
            "armed_cell_needs_a_push": {
                "claim": "an armed relay with no depolarising drive does not fire, and "
                         "with its reticular sector silenced does not oscillate",
                "measure": "peak rate after release with zero tonic drive; spindle-band "
                           "prominence with the sector's weight at zero",
                "expect": "no peak, and prominence <= 0",
                "note": "v2's first version inverted the activation term and became an "
                        "~8 Hz pacemaker that no other parameter could move"},
            "spindle_frequency": {
                "claim": "the isolated reticular-relay loop peaks in 11-16 Hz with a "
                         "positive prominence over its own 1/f background",
                "measure": "ibm/spectral.py peak_frequency with peak_prominence beside it",
                "expect": "11 <= f <= 16 and prominence > 0",
                "note": "v2: 13.626 Hz at +0.487 decades"},
            "selective_gating": {
                "claim": "driving one reticular sector cuts that nucleus's transfer slope "
                         "and leaves the other nucleus's where it was",
                "measure": "transfer slope from a sensory drive to the relay rate, per "
                           "nucleus, gated against ungated",
                "expect": "gated <= 0.5x its own ungated value; ungated within +-25%",
                "note": "a SLOPE.  v2's gating gate failed twice -- once on a correlation, "
                        "which is scale-free and cannot see a gain change at all, and once "
                        "because five lumped units had no sector to gate"},
            "higher_order_route": {
                "claim": "two cortical areas with no direct projection influence each "
                         "other through the pulvinar, and cutting the pulvinar removes it",
                "measure": "transfer slope from a drive on area A to area B's rate, with "
                           "the pulvinar edges present and absent",
                "expect": "present >> absent, absent ~ 0",
                "note": "the pairs are asserted un-connected in higher_order_pairs()"},
        },
    }
