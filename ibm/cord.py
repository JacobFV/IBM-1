"""the spinal cord for one limb pair: half-centres, motoneuron pools, muscle, spindles.

WHY THIS EXISTS
---------------
The programme's target is a real anatomical human body crawling in a 3D world, driven
through nerve fibres.  Between `ibm/substrate.py`'s cortical sheet and IHM-1's muscles
there is nothing at all today, and three of `ibm/rhythms.py`'s loops -- `cpg`,
`stretch_reflex`, `cms` -- are declared with `substrate: needs-cord`.  Their rhythms
(`locomotor_cpg` 0.5-3 Hz peak 1.2, `flexor_extensor_antiphase` 150-210 deg,
`stretch_reflex_resonance` 6-12 Hz, `corticomuscular_beta` 15-30 Hz) cannot be scored
on a cortical sheet, because the structure that produces them is not in it.

This module is that structure.  It is the thing between the brain and the muscles.

WHAT A CENTRAL PATTERN GENERATOR HAS TO BE, and what it must not be
-------------------------------------------------------------------
The catalogue's `locomotor_cpg` note states the specification in one sentence:

    "The frequency must be a MONOTONIC function of the descending drive with the phase
     relationship unchanged -- that is the actual specification, not the band."

Three separable claims follow, and each is a way the thing can be fake:

  1. It GENERATES.  With a constant descending drive -- no rhythmic input anywhere in
     the model -- there is still a rhythm.  A filter that rings when you shake it is
     not a pattern generator, and a band alone cannot tell the two apart.
  2. The drive sets the FREQUENCY and nothing else.  In the code below the descending
     signal enters every half-centre through the SAME weight `w_mlr` and carries no
     phase information whatever; the phase is a property of the coupling.  A model in
     which the drive is delivered in antiphase to flexor and extensor would reproduce
     every frequency measurement and would be a puppet.
  3. It survives DEAFFERENTATION.  Cut the spindle afferents and the rhythm stands
     (fictive locomotion, Grillner 2006).  If it dies, what was built is a reflex loop
     wearing a CPG's name.  `scripts/gate_cord.py` S6 is that cut, and it is the gate
     in this file that can actually fail.

THE MODEL
---------
Four half-centres, indexed in the order `POOL_NAMES`:

    0 left flexor   1 left extensor   2 right flexor   3 right extensor

and, per half-centre, one alpha motoneuron pool, one muscle and one spindle.  Two
joints, one per side, each driven by its own flexor/extensor pair.  State, all in
[0, 1] except the noise:

    V    half-centre activity
    h    inactivation of the persistent inward current       the RELEASE variable
    sR   reciprocal inhibition received (ipsilateral flexor <-> extensor)
    sC   commissural inhibition received (left <-> right, homologous)
    M    alpha motoneuron pool
    F    muscle force (first-order activation-to-force, `tau_force`)
    th   joint angle, one per side; 0.5 is neutral
    S    spindle afferent firing
    Lm   muscle length at the previous step, so the spindle has a velocity term
    eta  background current, Ornstein-Uhlenbeck, exactly advanced

**The alternation is `h`, and it is in the code, not in a comment.**  Each half-centre
carries a slow depolarising current `g_nap * h` -- the persistent inward current of
lumbar CPG interneurons, its slow inactivation gate `h` recovering during
hyperpolarisation.  `theta_hc = 1.05` sits ABOVE the descending drive's whole declared
range, so a half-centre CANNOT fire on the brainstem command alone: it fires when `h`
has recovered enough to carry it over.  Then:

    active    u = drive + g_nap * h              (partner silent, so no inhibition)
              h inactivates with `tau_h_dn`, u falls, the burst drains
    silent    u = drive - w_recip*sR - w_comm*sC + g_nap * h
              h recovers with `tau_h_up`, and the centre ESCAPES when

                  g_nap * h  >  theta_hc - drive + w_recip + w_comm

and the moment it does its rise drives `sR`/`sC` onto the first centre, which -- its own
`h` now inactivated -- cannot answer, and the pair swaps.  That inequality is also why
the frequency follows the drive: a larger `drive` lowers the `h` the silent centre has
to reach, so the silent phase shortens.  Nothing in this file contains a clock, a phase
variable or a sinusoid.

TWO MECHANISMS THAT WERE BUILT HERE AND MEASURED WRONG, kept because they cost a day
-------------------------------------------------------------------------------------
Both produced a clean 180 deg alternation in the locomotor band.  A band and a phase
are not enough to tell a pattern generator from a puppet, and only the drive sweep
separated them:

  * **Subtractive spike-frequency adaptation** (`-g_adapt*a`, `a` tracking the centre's
    own activity).  The burst then ends when the centre exhausts ITSELF, so a bigger
    drive means a longer climb before exhaustion: measured 0.91 Hz at drive 0.4 falling
    to 0.49 Hz at 1.2 -- monotone, and BACKWARDS.  This is release mode (Skinner, Kopell
    & Marder 1994) and it fails the catalogue's stated specification while passing every
    band and phase check.
  * **Presynaptic depression of the reciprocal inhibition.**  Right direction (0.60 ->
    1.44 Hz) and dead above drive 0.8.  It cannot be repaired by any choice of constants,
    and the algebra says why in one line: escape needs the depressed inhibition to be
    too weak to hold the partner down, and NOT co-activating needs it to be strong enough
    to hold the partner down.  The same inequality, both ways.  When two requirements
    reduce to `x < c` and `x > c`, stop sweeping.

A third near-miss is worth the same line.  The escape mechanism above LATCHES -- one
centre on, one off, for ever -- whenever `h_theta` sits below `theta_hc`, because the
depolarisation `h` produces then shuts `h`'s own recovery off before the centre reaches
firing threshold.  `ibm/thalamus.py` records the identical trap in its `h_H_theta`
comment.  `h_theta` is declared ABOVE `theta_hc` here for that reason and S7 is what
would catch it moving back.

DELAYS ARE RING BUFFERS, from the catalogue, in the state
----------------------------------------------------------
Four conduction delays, each the midpoint of the interval `ibm/rhythms.py` declares:

    d_mlr_cord_s     0.014   `cpg`            mlr -> cord_lumbar        (8-20 ms)
    d_cord_mn_s      0.0035  `cpg`            cord_lumbar -> mn_pool    (2-5 ms)
    d_mn_muscle_s    0.006   `stretch_reflex` mn_pool -> muscle         (4-8 ms)
    d_spindle_mn_s   0.014   `stretch_reflex` spindle_afferent->mn_pool (10-18 ms)

They live in `state` as rings, so `step` is a pure function of `(state, inputs)` and the
delay cannot be applied twice by accident.  `delays_in_steps(dt)` reports them in STEPS,
and `step` asserts the rings it was handed match the `dt` it was called with -- a dt too
coarse to resolve a delay collapses it, and that must be reported rather than hidden.

The catalogue's fifth edge, `muscle -> spindle_afferent` at 1-3 ms, is NOT a ring here:
it is intrafusal transduction inside the muscle rather than conduction between stations,
and it is implemented as `tau_spindle = 0.002`.  Said out loud because a delay that has
quietly become a time constant is exactly the kind of substitution that passes a units
audit (CLAUDE.md, "a length in the right frame between the wrong two points").

THE CORTICAL PORT
-----------------
`step(..., drive_cortex=...)` is where a cortical model writes, so `cms` can be closed
later.  Nothing here imports the cortex.

    shape     (B, 4), one value per motoneuron pool, POOL_NAMES order
    units     DIMENSIONLESS SYNAPTIC DRIVE on the pool's input, added to `u_mn`
              alongside `w_hc_mn * V` and `w_reflex * S`.  It is on the same scale as
              those terms: the pool's sigmoid has `theta_mn = 0.45` and `beta_mn = 6`,
              so +0.5 alone roughly half-saturates a silent pool and -0.5 silences one
              the CPG is driving.  It is NOT a firing rate and NOT in [0, 1]; it may be
              negative, and a corticospinal command that is meant to be a rate should be
              scaled by the caller.
    delay     NOT applied here.  `ibm/rhythms.py`'s `cms` declares m1 -> mn_pool at
              10-16 ms and that delay belongs to the caller's ring, exactly as
              `ThalamicField.step`'s `drive_cortex` is already-delayed.  Applying it in
              both places would apply it twice.

BOUNDEDNESS
-----------
Every state variable is advanced by an exponential-Euler step toward a target that is
itself in [0, 1] -- a sigmoid, another bounded state, or a clamp -- so the update is a
convex combination of two points in the box and nothing can leave it for any `dt`, any
parameters and any input.  Divergence is not representable.  `eta` is the one exception
and is excluded by name from the bounded gate, as in `ibm/thalamus.py`.

UNITS: seconds; rates dimensionless in [0, 1] (`MN_MAX` = 50 Hz for a motoneuron-pool
readout, `HC_MAX` = 40 Hz for an interneuron one); delays in seconds; joint angle
dimensionless with 0.5 neutral, 1.0 fully flexed; muscle length dimensionless with 0.5
at rest.  Noise is passed IN and never drawn here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

MN_MAX = 50.0
HC_MAX = 40.0
L_REST = 0.5
TH_NEUTRAL = 0.5

POOL_NAMES = ("L_flexor", "L_extensor", "R_flexor", "R_extensor")
# who inhibits whom.  `RECIP` is the ipsilateral flexor/extensor pair -- the half-centre
# inhibition the catalogue's `cpg` loop declares as its one negative edge.  `COMM` is the
# commissural pair, homologous muscle to homologous muscle, which is what makes left and
# right alternate rather than hop.  Both are permutations, so each half-centre receives
# exactly one of each and the inhibitory synapse variables stay in [0, 1] by construction.
RECIP = (1, 0, 3, 2)
COMM = (2, 3, 0, 1)
# which muscles act on which joint, and in which direction.  +1 flexes.
FLEXORS = (0, 2)
EXTENSORS = (1, 3)


@dataclass
class CordPriors:
    """every value carries its reason.  none is fitted except `tau_a_up`, which is set by
    the sweep in `scripts/gate_cord.py --sweep` against the catalogue's declared 1.2 Hz.
    """

    # ---- half-centre membrane ----------------------------------------------------
    # Interneuron populations are fast; the locomotor period is hundreds of
    # milliseconds, so the membrane must not be what sets it.  30 ms is a population
    # time constant, not a cell's, and it is an order below the period on purpose:
    # if a sweep of THIS constant moved the frequency, the rhythm would be a membrane
    # filter rather than a current cycling, and S7 would say so.
    tau_hc: float = 0.030
    beta_hc: float = 9.0           # recruitment slope of the half-centre population
    theta_hc: float = 1.05         # firing threshold.  ABOVE the descending drive's
                                   # entire declared range, which is the model's claim
                                   # that the brainstem command alone does not make a
                                   # half-centre fire -- the inward current does.
    # ---- the persistent inward current: the mechanism that releases ---------------
    # g_nap * h.  `h` is its slow inactivation: 1 is fully recovered (the centre can
    # fire), 0 is fully inactivated (it cannot).  The half-period is roughly the time
    # `h` takes to recover far enough to clear the inhibition, so `tau_h_up` is the
    # clock -- and `scripts/gate_cord.py --sweep` is where it is SET, against the
    # catalogue's declared 1.2 Hz peak, with the whole curve recorded rather than one
    # nudged value.
    g_nap: float = 2.40
    tau_h_up: float = 0.370        # recovering, while the centre is inhibited.  THE
                                   # CLOCK.  It must be several times `tau_hc`, or `h`
                                   # tracks the membrane instead of pacing it and the
                                   # whole thing collapses onto a stable co-active
                                   # fixed point -- measured, at tau_h_up = 0.12 s.
    tau_h_dn: float = 0.093        # inactivating, while it is firing.  Faster than the
                                   # recovery, which is what makes a burst a burst.
    h_beta: float = 4.0            # SHALLOW on purpose.  A steep h_inf makes `h` a fast
                                   # follower of the membrane with a large instantaneous
                                   # gain, and that gain stabilises exactly the symmetric
                                   # co-active state the alternation has to break out of.
    h_theta: float = 1.40          # recovers below this.  ABOVE `theta_hc` by 0.35, so
                                   # `h` keeps recovering all the way through the firing
                                   # threshold.  Put it below `theta_hc` and the pair
                                   # latches (see the module docstring).
    # ---- reciprocal and commissural inhibition -----------------------------------
    tau_inh: float = 0.020         # glycinergic IPSC in the cord; fast, so the swap is
                                   # a swap and not a fade
    w_recip: float = 1.05          # ipsilateral flexor <-> extensor.  Strong enough
                                   # that the active centre genuinely silences its
                                   # partner: this is what makes the phase 180 deg
                                   # rather than merely "not in phase".
    w_comm: float = 0.45           # commissural, homologous.  WEAKER than `w_recip`:
                                   # left-right coordination is looser than the
                                   # ipsilateral one, which is why a human can trot,
                                   # hop or limp while flexor/extensor alternation is
                                   # never optional.
                                   # `w_recip`, `w_comm` and `g_nap` were raised 1.5x
                                   # TOGETHER from the first working set: the frequency
                                   # span a drive sweep can reach before it latches at
                                   # one end or co-activates at the other went 2.7x ->
                                   # 4.0x.  The ratio between them is what sets the
                                   # phase; their common scale is what sets how much
                                   # range the drive has to work in.
    # ---- descending drive --------------------------------------------------------
    # The MLR's one weight.  It is a SINGLE scalar applied identically to all four
    # half-centres, and that identity is the model's statement that the brainstem sets
    # frequency and nothing else.  Anything that made this a per-centre vector would
    # let the drive dictate phase.
    w_mlr: float = 1.00
    # ---- alpha motoneuron pool ---------------------------------------------------
    tau_mn: float = 0.015          # motoneuron pools are fast relative to the muscle
    beta_mn: float = 6.0
    theta_mn: float = 0.45
    w_hc_mn: float = 0.80          # half-centre -> pool, the locomotor command
    w_reflex: float = 0.55         # Ia -> pool, MONOSYNAPTIC.  The one edge in this
                                   # file that carries the catalogue's 10-18 ms as a
                                   # conduction delay rather than a time constant.
    w_aff_hc: float = 0.15         # Ia -> half-centre.  Afferents reach the rhythm
                                   # generator too, not only the pool; without this
                                   # edge S6 (deafferentation) would be a gate that
                                   # cannot fail, because there would be nothing to
                                   # cut that the rhythm could ever have depended on.
    # ---- muscle ------------------------------------------------------------------
    tau_force: float = 0.055       # activation-to-force.  Excitation-contraction
                                   # coupling plus series elasticity: tens of ms for a
                                   # fast limb muscle, and the reason a 20 Hz motoneuron
                                   # ripple does not appear as a 20 Hz force ripple.
    # ---- joint -------------------------------------------------------------------
    tau_joint: float = 0.080       # limb inertia as a first-order lag.  A real joint is
                                   # second order; this is a rate model of the CORD and
                                   # the honest place for the real thing is IHM-1's
                                   # solver, which this module is meant to drive.
    joint_gain: float = 0.60       # excursion per unit force imbalance
    # ---- spindle -----------------------------------------------------------------
    tau_spindle: float = 0.002     # intrafusal transduction; stands for the catalogue's
                                   # muscle -> spindle_afferent 1-3 ms edge
    spindle_beta: float = 6.0
    spindle_theta: float = 0.10    # stretch above rest for half activation.  Positive,
                                   # so a muscle at rest length still discharges (~0.35)
                                   # -- spindles are not silent at rest, and a silent one
                                   # would make the reflex a threshold detector.
    spindle_dyn: float = 0.030     # seconds.  Weight on dL/dt: the Ia DYNAMIC response.
                                   # In units of seconds because it multiplies a rate to
                                   # give a length, so a 1/s stretch reads as 0.03 of
                                   # extra length.
    # ---- background --------------------------------------------------------------
    sigma: float = 0.02
    tau_eta: float = 0.005
    # ---- conduction, from ibm/rhythms.py.  Midpoints of the declared intervals. ----
    d_mlr_cord_s: float = 0.014    # `cpg`: mlr -> cord_lumbar, 8-20 ms
    d_cord_mn_s: float = 0.0035    # `cpg`: cord_lumbar -> mn_pool, 2-5 ms
    d_mn_muscle_s: float = 0.006   # `stretch_reflex`: mn_pool -> muscle, 4-8 ms
    d_spindle_mn_s: float = 0.014  # `stretch_reflex`: spindle_afferent -> mn_pool,
                                   # 10-18 ms.  The monosynaptic latency S5 measures.


def sigmoid(x, beta, theta):
    return torch.sigmoid(beta * (x - theta))


def _roll_in(ring: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """advance a delay line by one step, posting `new` at the far end.

    Index 0 is what arrives NOW.  A ring of length `lag + 1` therefore delivers a value
    posted at step t to the reader at step t + lag, which is what `delays_in_steps`
    reports.  Written as a concatenation rather than an in-place write into a modular
    index because the whole point is that `step` be a function of its state: an
    in-place ring needs a cursor, a cursor is state that lives outside the dict, and a
    cursor that gets out of step with the tensor is invisible.
    """
    return torch.cat([ring[:, 1:], new.unsqueeze(1)], dim=1)


class SpinalCord(nn.Module):
    """the cord for ONE limb pair: four half-centres, four pools, four muscles, two joints.

    One limb pair, not one limb, because left-right alternation is not an optional
    refinement -- `flexor_extensor_antiphase` declares it in the same row as the
    ipsilateral phase, and a cord built for a single limb would have to have it bolted
    on afterwards through a coupling that was never part of the mechanism.
    """

    def __init__(self, priors: CordPriors | None = None, learn: bool = True,
                 device="cpu"):
        super().__init__()
        self.pr = priors or CordPriors()
        self.n = 4
        self.learn = learn
        z = torch.zeros(self.n, device=device)
        # learned residuals, bounded to +-35% exactly as `ThalamicField` bounds its own:
        # the declared loop may be BENT by learning, it may not be replaced.  Per-unit,
        # so a left/right or flexor/extensor asymmetry can be learned -- real gait is
        # not symmetric -- without any of it being able to invent a new edge.
        for name in ("tau_h_up", "g_nap", "w_recip", "w_comm", "w_mlr", "w_hc_mn",
                     "w_reflex", "w_aff_hc", "tau_force"):
            self.register_buffer(f"prior_{name}", z + float(getattr(self.pr, name)))
            self.register_parameter(f"res_{name}", nn.Parameter(
                torch.zeros(self.n, device=device), requires_grad=learn))
        self.residual_frac = 0.35
        self.register_buffer("recip_idx", torch.tensor(RECIP, dtype=torch.long,
                                                       device=device))
        self.register_buffer("comm_idx", torch.tensor(COMM, dtype=torch.long,
                                                      device=device))
        self.register_buffer("flex_idx", torch.tensor(FLEXORS, dtype=torch.long,
                                                      device=device))
        self.register_buffer("ext_idx", torch.tensor(EXTENSORS, dtype=torch.long,
                                                     device=device))

    # ---------------------------------------------------------------- declared sites
    def site(self, name: str) -> torch.Tensor:
        p = getattr(self, f"prior_{name}")
        r = getattr(self, f"res_{name}")
        return p * (1.0 + self.residual_frac * torch.tanh(r))

    # ---------------------------------------------------------------- delays
    def delays_in_steps(self, dt: float) -> dict:
        """the four conduction delays in STEPS at this dt.

        Reported, never assumed.  A dt too coarse to resolve a delay rounds it to zero
        and the loop silently becomes an instantaneous handshake; the gate prints this
        table so that cannot happen without being on the record.
        """
        pr = self.pr
        return {"mlr_to_cord": int(round(pr.d_mlr_cord_s / dt)),
                "cord_to_mn": int(round(pr.d_cord_mn_s / dt)),
                "mn_to_muscle": int(round(pr.d_mn_muscle_s / dt)),
                "spindle_to_mn": int(round(pr.d_spindle_mn_s / dt))}

    # ---------------------------------------------------------------- state
    def init_state(self, b: int, dt: float, device=None, asym: float = 0.20):
        """`dt` is required because the delay lines live in the state.

        `asym` breaks the symmetry.  With all four half-centres identical and a tonic
        drive the symmetric point is an equilibrium, so SOMETHING has to start the
        pair apart; the honest choices are a declared initial condition or noise.  This
        is the declared one: left flexor and right extensor start at `asym`, everything
        else at zero -- a diagonal, which is the gait's own symmetry.

        It is a starting point, not the answer.  `asym` is small, the gates measure
        after a burn-in of several cycles, and `gate_cord.py` S2 re-measures the phase
        from a DIFFERENT initial condition: if the antiphase were the initial condition
        persisting rather than an attractor, the two would disagree.
        """
        device = device or self.prior_w_mlr.device
        lg = self.delays_in_steps(dt)
        z = torch.zeros(b, self.n, device=device)
        V0 = z.clone()
        V0[:, 0] = asym
        V0[:, 3] = asym
        th = torch.zeros(b, 2, device=device) + TH_NEUTRAL
        L0 = self._lengths(th)
        return {"V": V0, "h": z.clone() + 0.30, "sR": z.clone(), "sC": z.clone(),
                "M": z.clone(), "F": z.clone(), "th": th, "S": z.clone(),
                "Lm": L0, "eta": z.clone(),
                "ring_drive": torch.zeros(b, lg["mlr_to_cord"] + 1, self.n, device=device),
                "ring_V": torch.zeros(b, lg["cord_to_mn"] + 1, self.n, device=device),
                "ring_M": torch.zeros(b, lg["mn_to_muscle"] + 1, self.n, device=device),
                "ring_S": torch.zeros(b, lg["spindle_to_mn"] + 1, self.n, device=device)}

    @staticmethod
    def detach(state):
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in state.items()}

    def _lengths(self, th: torch.Tensor) -> torch.Tensor:
        """muscle lengths (B, 4) from joint angles (B, 2), POOL_NAMES order.

        Flexion shortens the flexor and lengthens the extensor; both are in [0, 1]
        because `th` is, and 0.5 is rest for both.
        """
        L = torch.zeros(th.shape[0], self.n, device=th.device, dtype=th.dtype)
        L[:, FLEXORS[0]] = 1.0 - th[:, 0]
        L[:, EXTENSORS[0]] = th[:, 0]
        L[:, FLEXORS[1]] = 1.0 - th[:, 1]
        L[:, EXTENSORS[1]] = th[:, 1]
        return L

    # ---------------------------------------------------------------- one step
    def step(self, state, dt: float, drive=None, drive_cortex=None, perturb=None,
             afferent_gain: float = 1.0, noise: torch.Tensor | None = None):
        """one exponential-Euler step.

        drive          descending MLR command.  Scalar or (B, 4).  It is applied through
                       the SINGLE weight `w_mlr` and carries no phase; see the module
                       docstring.  Delayed here by `d_mlr_cord_s`, because unlike the
                       cortical port this signal's delay belongs to the cord's own
                       afferent limb and there is no caller-side ring for it.
        drive_cortex   THE CORTICAL PORT.  (B, 4) dimensionless synaptic drive on the
                       motoneuron pools, already delayed by the caller.  See the module
                       docstring for units.
        perturb        (B, 2) externally imposed joint displacement, added to the joint's
                       target before the clamp.  This is how an experimenter stretches a
                       muscle (S5) and how a world would push the limb about.
        afferent_gain  1.0 intact, 0.0 a DORSAL RHIZOTOMY -- both the monosynaptic arc
                       and the afferent input to the half-centres are cut.  S6 is this
                       argument at 0.0 and is the gate that can fail.
        noise          standard-normal draw supplied by the caller, never drawn here.
        """
        pr = self.pr
        V, h, sR, sC = state["V"], state["h"], state["sR"], state["sC"]
        M, F, th, S, Lm, eta = (state["M"], state["F"], state["th"], state["S"],
                                state["Lm"], state["eta"])
        rd, rV, rM, rS = (state["ring_drive"], state["ring_V"], state["ring_M"],
                          state["ring_S"])
        lg = self.delays_in_steps(dt)
        # the rings encode dt.  A state built at one dt and stepped at another would
        # silently apply the wrong conduction delay -- assert rather than discover.
        assert rd.shape[1] == lg["mlr_to_cord"] + 1 and rV.shape[1] == lg["cord_to_mn"] + 1 \
            and rM.shape[1] == lg["mn_to_muscle"] + 1 \
            and rS.shape[1] == lg["spindle_to_mn"] + 1, \
            "delay rings were built for a different dt than this step was called with"

        tau_h_up = self.site("tau_h_up")
        g_nap, w_recip, w_comm = (self.site("g_nap"), self.site("w_recip"),
                                  self.site("w_comm"))
        w_mlr, w_hc_mn = self.site("w_mlr"), self.site("w_hc_mn")
        w_reflex, w_aff_hc = self.site("w_reflex"), self.site("w_aff_hc")
        tau_force = self.site("tau_force")

        # background, exactly advanced
        rho = math.exp(-dt / pr.tau_eta)
        eta = eta * rho
        if noise is not None:
            eta = eta + pr.sigma * math.sqrt(1.0 - rho * rho) * noise

        # what arrives NOW, having left its station one conduction delay ago
        drive_arr = rd[:, 0]
        V_arr = rV[:, 0]
        M_arr = rM[:, 0]
        S_arr = rS[:, 0] * float(afferent_gain)

        # ---- the half-centres.  The whole alternation is `+ g_nap * h` pulling against
        # `- w_recip * sR - w_comm * sC`, with a drive that is identical for all four
        # and below threshold on its own: the active centre's `h` drains, the silent
        # centre's recovers, and the pair swaps when the silent one clears theta_hc.
        u_hc = (w_mlr * drive_arr - w_recip * sR - w_comm * sC + g_nap * h
                + w_aff_hc * S_arr + eta)
        V_inf = sigmoid(u_hc, pr.beta_hc, pr.theta_hc)
        # The inactivation gate reads the FULL membrane drive `u_hc` -- the inward
        # current it itself carries included -- because it is voltage-gated and the
        # voltage is whatever every current has made it.  Reading the synaptic input
        # alone would leave the current without the feedback that terminates it, which
        # is the failure `ibm/thalamus.py`'s docstring records twice.
        # h_inf is 1 when hyperpolarised (recovered, ready to fire), 0 when depolarised.
        h_inf = 1.0 - sigmoid(u_hc, pr.h_beta, pr.h_theta)
        # asymmetric: slow to recover, faster to inactivate -- a burst, not a sinusoid.
        tau_h = torch.where(h_inf > h, tau_h_up, torch.full_like(tau_h_up, pr.tau_h_dn))
        # the two inhibitory species.  Each reads exactly one partner, so each target is
        # a single value in [0, 1] and the synapse cannot leave the box however many
        # partners are shouting.
        sR_inf = V.index_select(1, self.recip_idx)
        sC_inf = V.index_select(1, self.comm_idx)

        # ---- alpha motoneuron pools: the locomotor command, the monosynaptic reflex,
        # and the cortical port, summed on one dendrite.
        u_mn = w_hc_mn * V_arr + w_reflex * S_arr
        if drive_cortex is not None:
            u_mn = u_mn + drive_cortex
        M_inf = sigmoid(u_mn, pr.beta_mn, pr.theta_mn)

        # ---- muscle: first order, activation to force, driven by the pool's output as
        # it arrives at the endplate one `d_mn_muscle_s` later.
        F_inf = M_arr

        # ---- joint: the force imbalance about each side, as a first-order lag.
        flex = F.index_select(1, self.flex_idx)
        ext = F.index_select(1, self.ext_idx)
        th_target = TH_NEUTRAL + pr.joint_gain * (flex - ext)
        if perturb is not None:
            th_target = th_target + perturb
        th_inf = torch.clamp(th_target, 0.0, 1.0)

        # ---- spindles: static stretch plus the Ia dynamic (velocity) term.
        L = self._lengths(th)
        dL = (L - Lm) / dt
        S_inf = sigmoid((L - L_REST) + pr.spindle_dyn * dL, pr.spindle_beta,
                        pr.spindle_theta)

        cV = 1.0 - math.exp(-dt / pr.tau_hc)
        ch = 1.0 - torch.exp(-dt / tau_h)
        cI = 1.0 - math.exp(-dt / pr.tau_inh)
        cM = 1.0 - math.exp(-dt / pr.tau_mn)
        cF = 1.0 - torch.exp(-dt / tau_force)
        cth = 1.0 - math.exp(-dt / pr.tau_joint)
        cS = 1.0 - math.exp(-dt / pr.tau_spindle)

        V_new = V + cV * (V_inf - V)
        M_new = M + cM * (M_inf - M)
        S_new = S + cS * (S_inf - S)
        d_in = drive if torch.is_tensor(drive) else torch.zeros_like(V) + float(
            drive if drive is not None else 0.0)
        return {"V": V_new,
                "h": h + ch * (h_inf - h),
                "sR": sR + cI * (sR_inf - sR),
                "sC": sC + cI * (sC_inf - sC),
                "M": M_new,
                "F": F + cF * (F_inf - F),
                "th": th + cth * (th_inf - th),
                "S": S_new,
                "Lm": L,
                "eta": eta,
                "ring_drive": _roll_in(rd, d_in),
                "ring_V": _roll_in(rV, V_new),
                "ring_M": _roll_in(rM, M_new),
                "ring_S": _roll_in(rS, S_new)}

    # ---------------------------------------------------------------- rollout
    def rollout(self, steps: int, dt: float, state=None, drive=0.6, drive_cortex=None,
                perturb=None, afferent_gain: float = 1.0, noise_gen=None, b: int = 1,
                record=("V", "M", "F", "S", "th"), burn: int = 0):
        """run the cord.  Returns (traces dict of (B, steps, *), final state, info).

        `drive`, `drive_cortex` and `perturb` may be a scalar, a (B, ...) tensor held
        constant, or a (B, steps, ...) tensor.  `burn` steps are run and discarded: a
        cord started from rest spends its first few cycles in a transient and the
        spectrum of a transient is the transient's.
        """
        dev = self.prior_w_mlr.device
        state = state or self.init_state(b, dt, device=dev)
        out = {k: [] for k in record}
        for t in range(steps + burn):
            i = max(0, t - burn)

            def _slice(x, idx=i):
                if torch.is_tensor(x) and x.dim() == 3:
                    return x[:, min(idx, x.shape[1] - 1)]
                return x

            z = None
            if noise_gen is not None:
                z = torch.randn(b, self.n, generator=noise_gen, device="cpu").to(dev)
            state = self.step(state, dt, drive=_slice(drive),
                              drive_cortex=_slice(drive_cortex),
                              perturb=_slice(perturb), afferent_gain=afferent_gain,
                              noise=z)
            if t >= burn:
                for k in record:
                    out[k].append(state[k])
        info = {"delays_in_steps": self.delays_in_steps(dt), "dt": dt,
                "pools": list(POOL_NAMES), "afferent_gain": float(afferent_gain)}
        return {k: torch.stack(v, 1) for k, v in out.items()}, state, info

    # ---------------------------------------------------------------- readouts
    def pool_rate_hz(self, state):
        return state["M"] * MN_MAX

    def halfcentre_rate_hz(self, state):
        return state["V"] * HC_MAX
