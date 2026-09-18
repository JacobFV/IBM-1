"""the thalamus: relay cells, the reticular shell, and the loop between them and cortex.

WHY THIS EXISTS
---------------
`docs/RHYTHMS.md` declares 53 rhythms and `docs/DISCONNECTS.md` §11 counts how many the
cortical field can express: **9**.  Fourteen of the rest are blocked on one structure.  The
slow oscillation's thalamic partner, delta, sleep spindles, occipital alpha, its reactivity
to eye opening, the alpha travelling wave, sensorimotor mu, the 40 Hz auditory steady-state
response, PGO waves, arousal gating -- every one of them runs through the thalamus, and the
single frequency this programme has ever MEASURED on held-out people (13.45 Hz spindles,
`scripts/fit_sleep_resonance.py`) is produced by the reticular-to-relay loop.

So this is not an addition to a model that works.  It is the structure without which most
of the specification cannot even be scored.

THE MODEL, and why each piece is here
-------------------------------------
Two populations per thalamic unit, plus the two synaptic species that set the timing:

    R   relay (thalamocortical) cells      in [0, 1]
    T   reticular (TRN) cells              in [0, 1]
    h   T-current de-inactivation          in [0, 1]   the slow variable that makes a burst
    H   I_h (sag) activation                in [0, 1]   slower still, and depolarising
    Ca  calcium, and the slow K current it opens   in [0, 1]   the post-burst refractoriness
    sA  GABA-A activation from TRN         in [0, 1]   fast  (~10 ms)
    sB  GABA-B activation from TRN         in [0, 1]   slow  (~150 ms)
    eta background current, Ornstein-Uhlenbeck, exactly advanced

    u_R = drive_sense + w_CT_R * E_ctx(t - d_ct) - wA * sA - wB * sB + g_T * h * burst(u)
          - arousal_offset + eta
    u_T = w_RT * R(t - d_rt) + w_CT_T * E_ctx(t - d_ct) - w_TT * T

    tau_R dR/dt = -R + F(u_R; beta_R, theta_R)
    tau_T dT/dt = -T + F(u_T; beta_T, theta_T)
    tau_h dh/dt = -h + h_inf(u_R)          tau_h differs by direction (see below)
    tau_A dsA/dt = -sA + T ;  tau_B dsB/dt = -sB + T

**The rebound burst is the whole mechanism.**  A relay cell that is inhibited de-inactivates
its T-type calcium current (`h` rises), and when the inhibition decays the current fires it
back -- a burst that is not a response to its input but a response to having been silenced.
That burst drives the reticular cells, which inhibit the relay cells again, and the cycle is
a spindle.  Nothing about it is a filter of the input, which is precisely why v1's
`CorticalDynamics` could never have had one.

**The frequency is set by `tau_h`, and `tau_h` is set by the measurement.**  The period of
the loop is roughly the de-inactivation time plus the decay of the inhibition that caused
it.  The catalogue declares the spindle peak at **13.45 Hz** because that is what was fitted
on held-out subjects, so `tau_h_up` is declared at the value that puts the isolated loop
there, and `gate_thalamus.py` measures where the loop actually lands rather than assuming
it.  That is the direction this programme is supposed to run in: a measured number
constrains a declared parameter, instead of a fitted parameter chasing a measurement.

**One structure, several rhythms, selected by polarisation.**  `arousal` in [0, 1] sets the
relay cells' resting offset, and the same machinery then produces different things:

    arousal ~ 1.0   tonic relay, no oscillation, input passes through        waking
    arousal ~ 0.5   the reticular-relay loop rings in the spindle band       NREM2
    arousal ~ 0.1   deep hyperpolarisation, the slow h-current dominates     NREM3 delta

This is the catalogue's `arousal` row as a mechanism rather than a flag: the state is where
the substrate sits, not a mode it is told to be in.  `gate_thalamus.py` sweeps arousal and
checks that the bands come out in the declared order.

**Bounded, like the cortical field.**  Every variable is advanced by an exponential-Euler
step, a convex combination of the old value and a target inside the box, so R, T, h, sA and
sB stay in [0, 1] for any dt, any parameters and any input.  Divergence cannot be
represented.

UNITS: seconds, rates dimensionless in [0, 1] (`R_MAX` = 100 Hz for readouts), delays in
seconds, and the cortical trace this couples to is `CorticalField`'s `E`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

R_MAX = 100.0


# --------------------------------------------------------------------------------------
# declared priors.  every value carries its reason; none is fitted except where the
# comment says a MEASURED number set it.
# --------------------------------------------------------------------------------------
@dataclass
class ThalamicPriors:
    # population time constants.  relay and reticular cells are fast; the rhythms come
    # from the synapses and the T-current, not from the membranes.
    # These two, with tau_A below, are what actually set the spindle frequency: the
    # relay-reticular-relay loop's period is roughly twice the sum of the membrane and
    # synaptic constants around it.  Measured, not assumed -- a sweep of tau_h_up over
    # 6.5x moved the frequency 0.84 Hz, while tau_A and tau_R move it from 18.7 to 13.1 Hz
    # (docs/LOG.md, 18 Sep).  The values below put the isolated loop at the 13.45 Hz
    # measured on held-out sleepers.
    tau_R: float = 0.012
    tau_T: float = 0.006
    # T-current de-inactivation.  ASYMMETRIC on purpose: de-inactivation during
    # hyperpolarisation is slow, inactivation once the cell fires is fast, and the
    # asymmetry is what makes a burst a burst rather than a sinusoid.  tau_h_up is the
    # parameter the 13.45 Hz measurement constrains (see the module docstring).
    tau_h_up: float = 0.060        # de-inactivating, cell mildly hyperpolarised
    tau_h_dn: float = 0.020        # inactivating, cell depolarised
    # De-inactivation is itself VOLTAGE-DEPENDENT, and this is the one line that makes the
    # module produce two bands from one mechanism.  In a real relay cell tau_h runs tens of
    # milliseconds near rest and several HUNDRED at deep hyperpolarisation, so the
    # inter-burst interval stretches as the cell is taken down: mild hyperpolarisation
    # gives the spindle band, deep hyperpolarisation gives delta.  Without it the module
    # has a single limit cycle whose frequency barely moves, which is what four sweeps
    # found before this was added (docs/LOG.md).
    tau_h_deep: float = 0.420      # de-inactivating, cell deeply hyperpolarised
    deep_theta: float = -0.35      # below this the slow branch takes over
    deep_beta: float = 10.0
    h_theta: float = 0.20          # u_R below this de-inactivates
    h_beta: float = 12.0
    g_T: float = 0.55              # T-current strength; the burst's size
    burst_theta: float = -0.12     # the T-window's activation foot: a small depolarisation
                                   # from rest is enough to fire an armed cell
    burst_beta: float = 14.0
    # the SAG current, I_h.  Hyperpolarisation-activated and DEPOLARISING, so it is a
    # negative feedback on the cell's own polarisation with a time constant of a few
    # hundred milliseconds -- which is a period in the delta band.  Delta is I_h and the
    # T-current taking turns: the sag slowly lifts a hyperpolarised cell until the
    # de-inactivated T-current fires, the burst deactivates the sag, the cell falls back,
    # and it starts again (Destexhe & Sejnowski 2003, and McCormick & Pape 1990's original
    # measurement of the current itself).
    #
    # It is here because the first version of this module DESCRIBED this mechanism in its
    # docstring and did not implement it, and the T4 sweep showed delta prominence negative
    # at every arousal level while the prose claimed three regimes (docs/LOG.md).
    # Calcium, and the slow potassium current it opens.  A burst floods the cell with
    # calcium through the same T-channels that produced it, and the calcium-activated
    # potassium conductance then holds the cell down for a few hundred milliseconds.  This
    # is the refractoriness the first two versions lacked: without it the inter-burst
    # interval is set by the fast loop and the cell cannot produce anything slower than the
    # spindle band, which is exactly what gate T7 measured (delta prominence -1.21).
    #
    # Declared as a MECHANISM, on the diagnosis, not tuned into place: the gate T7 is
    # unchanged and either it now passes or the diagnosis was wrong.
    # DEFAULT ZERO, and the reason is the finding, not a preference.  The mechanism is
    # implemented and the gate T7 still fails: at the polarisation where this refractoriness
    # would pace delta, the relay does not fire AT ALL (0.00 bursts/s at arousal 0.05), so
    # there is nothing for it to pace, and all it does at NREM2 is cost burst rate (7.5/s
    # down to 0.4/s at 0.55) and spindle amplitude.  The missing piece is an operating point
    # at which a deeply hyperpolarised relay still fires -- the arousal offset, the sag's
    # strength and the T-window's position have to be designed together -- and that is a
    # design step, not a constant.  Left in place at zero rather than deleted, because the
    # diagnosis that called for it is still the diagnosis.
    g_KCa: float = 0.0
    tau_Ca: float = 0.380          # calcium clearance: the inter-burst interval it sets
    ca_gain: float = 1.6           # influx per unit burst
    g_H: float = 0.85
    tau_H_up: float = 0.320        # activating, cell hyperpolarised: sets the delta period
    tau_H_dn: float = 0.140        # deactivating once the cell depolarises
    h_H_theta: float = 0.02        # activates below this -- ABOVE the T-window's foot, so
                                   # the sag can actually push the cell into the window
    h_H_beta: float = 16.0
    # inhibition from TRN.  GABA-A is fast and carries the spindle; GABA-B is slow and
    # is what deepens into delta when the cells sit hyperpolarised (Destexhe 1996).
    tau_A: float = 0.026           # TRN -> relay GABA-A decay.  Slower than a cortical
                                   # IPSC, which is measured (20-40 ms in TC cells) and is
                                   # the constant the spindle frequency is set by here.
    tau_B: float = 0.150
    w_A: float = 0.85
    w_B: float = 0.35
    # the reticular shell
    w_RT: float = 1.10             # relay -> TRN, strong: the shell is driven by its relay
    w_TT: float = 0.25             # TRN -> TRN, the shell's own inhibition
    beta_R: float = 9.0
    theta_R: float = 0.35
    beta_T: float = 8.0
    theta_T: float = 0.30
    # cortical drive.  layer 6 reaches both, and reaches TRN FIRST -- the corticothalamic
    # axon's collateral into the shell is why cortex can shut its own input off.
    w_CT_R: float = 0.35
    w_CT_T: float = 0.55
    # arousal sets the relay cells' resting offset: 1 depolarised (tonic), 0 hyperpolarised
    arousal_span: float = 0.55
    # background
    sigma: float = 0.03
    tau_eta: float = 0.005
    # conduction, from the catalogue's `tct` and `rgs` loops (ibm/rhythms.py)
    d_relay_cortex_s: float = 0.002
    d_cortex_trn_s: float = 0.005
    d_trn_relay_s: float = 0.0015


def sigmoid(x, beta, theta):
    return torch.sigmoid(beta * (x - theta))


class ThalamicField(nn.Module):
    """`n` thalamic units, each a relay population with its reticular sector.

    A unit is a NUCLEUS-SIZED object, not a cell: one per cortical region group is the
    resolution the atlas supports (`docs/DISCONNECTS.md` §11 -- LGN, MGN, VL and the
    reticular shell are all inside one `thalamus` label), and pretending to more would be
    a precision nobody can address.  `region_of[i]` says which cortical region unit i
    relays for, so the coupling is anatomical rather than all-to-all.
    """

    def __init__(self, n: int, priors: ThalamicPriors | None = None,
                 learn: bool = True, device="cpu", seed: int = 0):
        super().__init__()
        self.pr = priors or ThalamicPriors()
        self.n = int(n)
        self.learn = learn
        z = torch.zeros(self.n, device=device)
        # learned residuals, bounded the same way the cortical field bounds its own: the
        # declared loop may be bent, it may not be replaced.
        for name in ("tau_h_up", "g_T", "w_A", "w_B", "w_RT", "g_H", "tau_H_up",
                     "g_KCa", "tau_Ca"):
            self.register_buffer(f"prior_{name}", z + float(getattr(self.pr, name)))
            self.register_parameter(f"res_{name}", nn.Parameter(
                torch.zeros(self.n, device=device), requires_grad=learn))
        self.residual_frac = 0.35

    def site(self, name: str) -> torch.Tensor:
        p = getattr(self, f"prior_{name}")
        r = getattr(self, f"res_{name}")
        return p * (1.0 + self.residual_frac * torch.tanh(r))

    def init_state(self, b: int, device=None):
        device = device or self.prior_g_T.device
        z = torch.zeros(b, self.n, device=device)
        return {"R": z.clone(), "T": z.clone(), "h": z.clone() + 0.5,
                "H": z.clone(), "Ca": z.clone(), "sA": z.clone(), "sB": z.clone(),
                "eta": z.clone()}

    @staticmethod
    def detach(state):
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in state.items()}

    def step(self, state, dt: float, drive_sense=None, drive_cortex=None,
             arousal: float = 1.0, noise: torch.Tensor | None = None):
        """one exponential-Euler step.

        `drive_cortex` is the cortical trace that has ALREADY been delayed by the caller --
        the delay belongs to the loop, and putting it here as well would apply it twice.
        `noise` is a standard-normal draw supplied by the caller, never drawn here.
        """
        pr = self.pr
        R, T, h, sA, sB = state["R"], state["T"], state["h"], state["sA"], state["sB"]
        H = state.get("H")
        Ca = state.get("Ca")
        eta = state.get("eta")
        if eta is None:
            eta = torch.zeros_like(R)
        if H is None:
            H = torch.zeros_like(R)
        if Ca is None:
            Ca = torch.zeros_like(R)
        g_T, w_A, w_B, w_RT = (self.site("g_T"), self.site("w_A"),
                               self.site("w_B"), self.site("w_RT"))
        tau_h_up, g_H, tau_H_up = (self.site("tau_h_up"), self.site("g_H"),
                                   self.site("tau_H_up"))
        g_KCa, tau_Ca = self.site("g_KCa"), self.site("tau_Ca")

        rho = math.exp(-dt / pr.tau_eta)
        eta = eta * rho
        if noise is not None:
            eta = eta + pr.sigma * math.sqrt(1.0 - rho * rho) * noise

        dsense = drive_sense if drive_sense is not None else 0.0
        dctx = drive_cortex if drive_cortex is not None else 0.0
        offset = pr.arousal_span * (1.0 - float(arousal))

        # the input to the relay cell WITHOUT its own rebound, which is what decides
        # whether the T-current is de-inactivating
        u_syn = dsense + pr.w_CT_R * dctx - w_A * sA - w_B * sB - offset + eta
        # the cell's polarisation INCLUDING its own sag: both the T-current's
        # de-inactivation and the sag's activation read this, not the synaptic input alone,
        # because the currents respond to the membrane and not to what is driving it
        u_mem = u_syn + g_H * H - g_KCa * Ca
        # The T-current as a WINDOW current: availability (h, which rises with
        # hyperpolarisation) times activation (which rises with depolarisation).  The
        # product is non-zero only in the window between them, and the cell therefore needs
        # a small depolarising push -- from the sag, from a synapse, from the release of
        # inhibition -- to fire.
        #
        # The first version had the activation term INVERTED, so the burst grew the more
        # hyperpolarised the cell was.  That makes it self-triggering: h re-arms in 60 ms
        # and fires itself again, a limit cycle at ~8 Hz whatever else is happening.  It is
        # why the sag could be swept over two-and-a-half times its strength and move the
        # output by 0.1 Hz, and why the reticular shell could be silenced to 0.45 Hz with
        # the relay still ringing at 7 Hz.  A current that does not need a trigger is a
        # pacemaker, not a rebound.
        burst = g_T * h * sigmoid(u_mem, pr.burst_beta, pr.burst_theta)
        u_R = u_mem + burst
        u_base = u_mem
        u_T = w_RT * R + pr.w_CT_T * dctx - pr.w_TT * T

        fR = sigmoid(u_R, pr.beta_R, pr.theta_R)
        fT = sigmoid(u_T, pr.beta_T, pr.theta_T)
        # Both gating variables read the FULL membrane drive `u_R` -- synaptic input, sag
        # and burst together -- because both currents are voltage-gated and the voltage is
        # whatever every current has made it.  The first version read only the synaptic
        # input, which left each of them without the feedback that terminates it: the sag
        # sat pinned at 1 and became a tonic depolarisation, and a sweep over its threshold
        # and its time constant moved the output by less than 0.1 Hz (docs/LOG.md). A
        # parameter that changes nothing is the symptom; a missing feedback loop is the
        # disease.
        # h_inf is 1 when hyperpolarised (de-inactivated, ready to burst), 0 when not
        h_inf = 1.0 - sigmoid(u_R, pr.h_beta, pr.h_theta)
        # asymmetric: slow to arm, fast to disarm -- and the arming time itself stretches
        # the deeper the cell is held (see `tau_h_deep`)
        deep = 1.0 - sigmoid(u_R, pr.deep_beta, pr.deep_theta)
        tau_arm = tau_h_up + (pr.tau_h_deep - tau_h_up) * deep
        tau_h = torch.where(h_inf > h, tau_arm, torch.full_like(tau_arm, pr.tau_h_dn))
        # the sag: activated by hyperpolarisation, slower than the T-current, and
        # asymmetric in the same direction
        H_inf = 1.0 - sigmoid(u_R, pr.h_H_beta, pr.h_H_theta)
        tau_H = torch.where(H_inf > H, tau_H_up, torch.full_like(tau_H_up, pr.tau_H_dn))

        cR = 1.0 - math.exp(-dt / pr.tau_R)
        cT = 1.0 - math.exp(-dt / pr.tau_T)
        ch = 1.0 - torch.exp(-dt / tau_h)
        cA = 1.0 - math.exp(-dt / pr.tau_A)
        cB = 1.0 - math.exp(-dt / pr.tau_B)
        cH = 1.0 - torch.exp(-dt / tau_H)
        # calcium: driven by the BURST (the T-channels are the influx path), cleared with
        # its own time constant.  Clamped into the box like every other variable.
        Ca_inf = torch.clamp(pr.ca_gain * burst, 0.0, 1.0)
        cCa = 1.0 - torch.exp(-dt / tau_Ca)
        return {"R": R + cR * (fR - R), "T": T + cT * (fT - T),
                "h": h + ch * (h_inf - h), "H": H + cH * (H_inf - H),
                "Ca": Ca + cCa * (Ca_inf - Ca),
                "sA": sA + cA * (T - sA), "sB": sB + cB * (T - sB), "eta": eta}

    def rollout(self, steps: int, dt: float, state=None, drive_sense=None,
                drive_cortex=None, arousal: float = 1.0, noise_gen=None, b: int = 1,
                record: str = "R"):
        """the thalamus alone, no cortex: what an isolated nucleus does.

        `drive_sense` and `drive_cortex` may be a scalar, a (B, N) tensor held constant, or
        a (B, steps, N) tensor.  Returns (trace (B, steps, N), final state).
        """
        state = state or self.init_state(b, device=self.prior_g_T.device)
        out = []
        for t in range(steps):
            ds = drive_sense[:, t] if torch.is_tensor(drive_sense) and drive_sense.dim() == 3 \
                else drive_sense
            dc = drive_cortex[:, t] if torch.is_tensor(drive_cortex) and drive_cortex.dim() == 3 \
                else drive_cortex
            z = None
            if noise_gen is not None:
                z = torch.randn(b, self.n, generator=noise_gen, device="cpu").to(
                    self.prior_g_T.device)
            state = self.step(state, dt, drive_sense=ds, drive_cortex=dc,
                              arousal=arousal, noise=z)
            out.append(state[record])
        return torch.stack(out, 1), state

    def rate_hz(self, state):
        return state["R"] * R_MAX


class ThalamoCortical(nn.Module):
    """the loop: a `CorticalField` and a `ThalamicField`, with the declared delays between.

    The delays are not decoration.  `ibm/rhythms.py`'s `tct` loop declares relay to cortex
    at 1-3 ms and cortex to the reticular shell at 3-6 ms, and the round-trip bound in that
    file is computed from them.  Here they are implemented as ring buffers, so what arrives
    at cortex is what the thalamus emitted a few milliseconds ago, and the loop has a
    conduction budget rather than an instantaneous handshake.

    `unit_of_site[i]` maps each cortical site to its thalamic unit, and `sites_of_unit` is
    the reverse as a normalised (n_units, n_sites) matrix -- built once, because building
    it per step was what `CorticalField.edge_weights` was written to avoid.
    """

    def __init__(self, cortex, thalamus: ThalamicField, unit_of_site: torch.Tensor):
        super().__init__()
        self.cortex = cortex
        self.thal = thalamus
        assert unit_of_site.numel() == cortex.n, "one thalamic unit per cortical site"
        self.register_buffer("unit_of_site", unit_of_site.long())
        m = torch.zeros(thalamus.n, cortex.n, device=unit_of_site.device)
        m[self.unit_of_site, torch.arange(cortex.n, device=unit_of_site.device)] = 1.0
        # TWO matrices, because the two directions are different operations and sharing
        # one would silently divide the thalamus's output by the size of its territory:
        # `pool` AVERAGES cortex into a unit, `spread` BROADCASTS a unit back to its sites.
        self.register_buffer("pool", m / m.sum(1, keepdim=True).clamp_min(1.0))
        self.register_buffer("spread", m)

    def _ring(self, b, n, lag, device, dtype):
        return torch.zeros(b, max(1, lag + 1), n, device=device, dtype=dtype)

    def rollout(self, steps: int, dt: float, drive_sense=None, cortical_drive=None,
                arousal: float = 1.0, m_beta: float = 1.0, m_sigma: float = 1.0,
                noise_gen=None, b: int = 1):
        """run the closed loop.  Returns (cortical trace, thalamic trace, states).

        Both directions are delayed by the catalogue's declared conduction times, rounded
        to the nearest step.  A dt too coarse to resolve a delay collapses it to zero, and
        that is reported by `delays_in_steps` rather than hidden.
        """
        pr = self.thal.pr
        dev = self.pool.device
        lag_tc = int(round(pr.d_relay_cortex_s / dt))
        lag_ct = int(round(pr.d_cortex_trn_s / dt))
        cs = self.cortex.init_state(b, device=dev)
        ts = self.thal.init_state(b, device=dev)
        ring_tc = self._ring(b, self.cortex.n, lag_tc, dev, torch.float32)
        ring_ct = self._ring(b, self.thal.n, lag_ct, dev, torch.float32)
        W = self.cortex.edge_weights()
        ctx_out, thal_out = [], []
        for t in range(steps):
            z_c = z_t = None
            if noise_gen is not None:
                z_c = torch.randn(b, self.cortex.n, generator=noise_gen,
                                  device="cpu").to(dev)
                z_t = torch.randn(b, self.thal.n, generator=noise_gen, device="cpu").to(dev)
            # what the cortex hears from the thalamus: emitted lag_tc steps ago
            relay_in = ring_tc[:, t % ring_tc.shape[1]].clone()
            drive = relay_in
            if cortical_drive is not None:
                drive = drive + (cortical_drive[:, t] if cortical_drive.dim() == 3
                                 else cortical_drive)
            cs = self.cortex.step(cs, drive, dt, W=W, noise=z_c,
                                  m_beta=m_beta, m_sigma=m_sigma)
            # what the thalamus hears from the cortex: emitted lag_ct steps ago
            ctx_in = ring_ct[:, t % ring_ct.shape[1]].clone()
            ds = drive_sense[:, t] if torch.is_tensor(drive_sense) and drive_sense.dim() == 3 \
                else drive_sense
            ts = self.thal.step(ts, dt, drive_sense=ds, drive_cortex=ctx_in,
                                arousal=arousal, noise=z_t)
            # post into the rings for their arrival times
            ring_tc = ring_tc.clone() if torch.is_grad_enabled() else ring_tc
            ring_ct = ring_ct.clone() if torch.is_grad_enabled() else ring_ct
            ring_tc[:, (t + lag_tc) % ring_tc.shape[1]] = ts["R"] @ self.spread
            ring_ct[:, (t + lag_ct) % ring_ct.shape[1]] = cs["E"] @ self.pool.transpose(0, 1)
            ctx_out.append(cs["E"])
            thal_out.append(ts["R"])
        return (torch.stack(ctx_out, 1), torch.stack(thal_out, 1),
                {"cortex": cs, "thalamus": ts,
                 "delays_in_steps": {"relay_to_cortex": lag_tc, "cortex_to_trn": lag_ct}})


def units_from_regions(region_names, groups=None):
    """(unit_of_site, unit_names): one thalamic unit per cortical region group.

    The default grouping is the catalogue's own relay nuclei -- visual, auditory,
    somatosensory, motor and a single associative unit for everything else -- because those
    are the ones its loops name (LGN, MGN, VPL, VL, MD/pulvinar).  A finer grouping is an
    argument, not an edit.
    """
    groups = groups or {
        "lgn": ("pericalcarine", "cuneus", "lingual", "lateraloccipital", "fusiform"),
        "mgn": ("transversetemporal", "superiortemporal", "bankssts"),
        "vpl": ("postcentral", "supramarginal", "paracentral"),
        "vl": ("precentral", "caudalmiddlefrontal"),
    }
    names = list(groups) + ["assoc"]
    lookup = {}
    for u, (g, labs) in enumerate(groups.items()):
        for lab in labs:
            lookup[lab] = u
    assoc = len(groups)
    out = [lookup.get(str(n).split(".", 1)[-1], assoc) for n in region_names]
    return torch.tensor(out, dtype=torch.long), names
