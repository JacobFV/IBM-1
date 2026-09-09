"""the thalamo-cortico-thalamic loop, closed and running in the time domain.

`thalamocortical_coupling` (ibm/processes/neural.py) declares three
implementations.  two of them -- `corticothalamic_loop_lti` and
`alpha_resonator` -- are LTI transfer functions: they can be *fitted* to a
spectrum but they cannot be *run* alongside `CorticalDynamics`, which is a
nonlinear rate model in the time domain.  the third, `burst_relay_rate`, is
declared with `form=Form.RATE` and a full parameter block, and has never had an
implementation.  this module is that implementation.

why it has to be the burst form.  the process docstring says it plainly:

    relay neurons have a t-type calcium current, so they fire tonically when
    depolarized and in bursts when hyperpolarized, and burst mode is a
    different regime rather than a large-signal version of the same one.

a linear relay is a filter.  driven by nothing it outputs nothing, so a linear
TCT loop bolted onto a cortex with no stimulus is silent, and "the loop is
continuously active" would be a claim with no mechanism behind it.  the T
current supplies the mechanism: it is de-inactivated by hyperpolarization, so a
relay cell that TRN has just inhibited rebounds with a burst, that burst excites
TRN, and TRN inhibits it again.  that is a relaxation oscillator and it needs no
input to run.

the state and the equations
---------------------------

relay population, per thalamic unit:

    v_t   membrane potential                 tau_relay_s
    h     T-current de-inactivation          t_deinactivation_s
    r_t   firing rate                        5 ms

    m_inf(v) = sig((v - V_MT) / K_MT)        activation, instantaneous
    h_inf(v) = sig(-(v - v_deinact) / K_H)   inactivation, slow
    I_T      = g_T * m_inf^2 * h * (E_CA - v)

    dv_t = [ -(v_t - E_L) + I_T + I_bg + 20*(w_ct_relay * r_c(t - ct) / r_max) ] / tau_relay
           - (v_t - E_GABA_A) * g_a / tau_relay
           - (v_t - E_GABA_B) * g_b / tau_relay

TRN population, per thalamic unit:

    dv_n = [ -(v_n - E_L_TRN)
             + 20*(w_relay_trn * r_t(t - 1ms) / r_max
                   + w_ct_trn * r_c(t - ct) / r_max) ] / tau_trn_s
    dg_a = (w_trn_gaba_a * r_n / r_max - g_a) / tau_gaba_a_s
    dg_b = (w_trn_gaba_b * r_n / r_max - g_b) / tau_trn_gaba_b_s

and the ascending limb, delivered into `CorticalDynamics.step`'s own `drive`
argument so that the cortical model is not modified at all:

    drive_c = 20 * w_tc * r_t(t - tc) / r_max

the three severable limbs are `w_tc` (thalamus -> cortex), `w_ct_relay` /
`w_ct_trn` (cortex -> thalamus) and `w_trn_gaba_a` / `w_trn_gaba_b` (TRN
inhibition).  `sever=` takes any of them by name, and that is what the ablation
in `scripts/measure_tct.py` uses: an ablation that removes a limb of the running
loop, not a re-parameterization that re-runs a different model.

the delays are real delays
--------------------------

ring buffers, not first-order lags.  `tc_delay_s` and `ct_delay_s` default to the
values `corticothalamic_loop_lti` declares -- 5 ms up, 8 ms down, a 13 ms round
trip, inside the 5-20 ms physiological range -- and the buffer length is
`round(delay / dt)`, so the integration timestep has to be small enough to
resolve them.  `reset()` asserts it: at `dt` = 5e-3 (the default of
`pretrain_video_loop.py`) the 5 ms limb is a single step and the 8 ms limb rounds
to two, which is not a delay line, it is a rounding error.  `reset` raises rather
than silently discretizing the mechanism away.

parameter provenance
--------------------

every parameter the declaration names is read from the declaration's prior
median and nothing here re-derives one:

    t_deinactivation_s   lognormal(0.1, 2.0)     -> 0.1
    v_deinactivation_mv  normal(-75.0, 5.0)      -> -75.0
    burst_spikes         uniform(2.0, 8.0)       -> 5.0
    t_current_gain       speculative(1.0, 20.0)  -> see below
    trn_gain             weak(1.0, 8.0)
    tc_delay_s           lognormal(5e-3, 1.5)    -> 5e-3
    ct_delay_s           lognormal(8e-3, 1.5)    -> 8e-3
    tau_relay_s          lognormal(0.012, 1.5)   -> 0.012
    tau_trn_gaba_b_s     lognormal(0.15, 1.6)    -> 0.15

`t_current_gain` is declared in nA over a `speculative(1.0, 20.0)` range and this
model has no capacitance, so it cannot be carried across in its declared units.
it enters here as a dimensionless conductance relative to the leak, which is the
same quantity the leak term `-(v - E_L)` is written in.  that conversion is a
choice this module makes and the declaration does not license, so it is named
`g_t` and not `t_current_gain`, and no result below should be read as
constraining the declared parameter.

the channel kinetics -- V_MT, K_MT, K_H, E_CA, E_GABA_A, E_GABA_B -- are NOT in
the declaration.  they are the standard low-threshold-calcium parameterization
and they are stated as module constants so that they are visible as an
assumption imported from outside ibm-1 rather than buried in an expression.

what this does not have
-----------------------

TRN cells have a T current of their own and it matters for spindle waxing; there
is none here.  there is no laminar structure, so "L6 corticothalamic" is the mean
cortical rate over a thalamic unit's cortical field and not a layer.  the
cortico-thalamic map is a fixed topographic partition, not a learned or
tractographic one.  there is no nucleus identity: `n_thal` units are one
homogeneous relay population, where the process declaration is explicit that a
lumped thalamus "cannot oscillate for the right reason".  what is here is the
loop's topology and its burst mechanism, which is what is needed to ask whether
the loop runs; it is not a thalamus.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# low-threshold calcium channel kinetics.  NOT declared in ibm.processes.neural;
# imported from the standard single-cell parameterization and stated here so the
# import is visible.  (Destexhe & Sejnowski 2003, the source the declaration
# already cites, in the reduced two-variable form.)
# ---------------------------------------------------------------------------
V_MT = -59.0       #: half-activation of the T current, mV
K_MT = 6.2         #: activation slope, mV
K_H = 4.0          #: inactivation slope, mV
E_CA = 120.0       #: calcium reversal, mV
E_GABA_A = -80.0   #: TRN -> relay GABA-A reversal, mV
E_GABA_B = -95.0   #: TRN -> relay GABA-B reversal (K+), mV

#: the severable limbs.  `ThalamoCortical(sever=...)` accepts any subset.
LIMBS = ("tc", "ct", "trn", "trn_gaba_a", "trn_gaba_b", "t_current")


class ThalamoCortical(nn.Module):
    """a thalamic relay + TRN population closed around a cortical sheet.

    holds no cortical state.  `step` is given the cortical rate the sheet
    produced on the previous step and returns the drive the sheet should receive
    on this one, so it wraps `CorticalDynamics` without touching it::

        tct = ThalamoCortical(n_sites=dyn.n, device=dev)
        tct.reset(batch, dt, device=dev)
        s, w = dyn.init_state(batch, dev), dyn.edge_weights()
        for _ in range(n):
            drive = tct.step(s[1], dt)      # s[1] is the cortical rate
            s = dyn.step(s, drive, dt, w)
    """

    def __init__(
        self,
        n_sites: int,
        n_thal: int = 16,
        device=None,
        # --- delays, from `corticothalamic_loop_lti` -----------------------
        tc_delay_s: float = 5e-3,
        ct_delay_s: float = 8e-3,
        # --- membrane and synaptic kinetics --------------------------------
        tau_relay_s: float = 0.012,
        tau_trn_s: float = 0.010,
        tau_rate_s: float = 5e-3,
        tau_gaba_a_s: float = 0.010,
        tau_trn_gaba_b_s: float = 0.15,
        # --- T current, from `burst_relay_rate` ----------------------------
        g_t: float = 0.30,
        t_deinactivation_s: float = 0.10,
        v_deinactivation_mv: float = -75.0,
        burst_spikes: float = 5.0,
        # --- loop gains ----------------------------------------------------
        w_tc: float = 0.60,
        w_ct_relay: float = 0.35,
        w_ct_trn: float = 0.50,
        w_relay_trn: float = 1.20,
        w_trn_gaba_a: float = 1.10,
        w_trn_gaba_b: float = 0.15,
        # --- resting state -------------------------------------------------
        i_bg_mv: float = 5.0,
        e_l_relay: float = -70.0,
        e_l_trn: float = -65.0,
        v_half: float = -55.0,
        slope: float = 4.0,
        r_max: float = 100.0,
        sever: tuple[str, ...] = (),
        seed: int = 0,
    ):
        super().__init__()
        for s in sever:
            if s not in LIMBS:
                raise ValueError(f"unknown limb {s!r}; known: {LIMBS}")
        self.n_sites, self.n_thal = n_sites, n_thal
        self.tc_delay_s, self.ct_delay_s = tc_delay_s, ct_delay_s
        self.tau_relay_s, self.tau_trn_s = tau_relay_s, tau_trn_s
        self.tau_rate_s = tau_rate_s
        self.tau_gaba_a_s, self.tau_gaba_b_s = tau_gaba_a_s, tau_trn_gaba_b_s
        self.g_t, self.tau_h = g_t, t_deinactivation_s
        self.v_deinact, self.burst_spikes = v_deinactivation_mv, burst_spikes
        self.i_bg = i_bg_mv
        self.e_l_relay, self.e_l_trn = e_l_relay, e_l_trn
        self.v_half, self.slope, self.r_max = v_half, slope, r_max
        self.sever = tuple(sever)

        # severing a limb ZEROES ITS GAIN and changes nothing else.  it is the
        # same integration of the same equations with one term removed, which is
        # what makes the ablation a control on this model rather than a
        # comparison against a different one.
        cut = set(sever)
        self.w_tc = 0.0 if "tc" in cut else w_tc
        self.w_ct_relay = 0.0 if "ct" in cut else w_ct_relay
        self.w_ct_trn = 0.0 if "ct" in cut else w_ct_trn
        self.w_relay_trn = w_relay_trn
        self.w_trn_gaba_a = 0.0 if {"trn", "trn_gaba_a"} & cut else w_trn_gaba_a
        self.w_trn_gaba_b = 0.0 if {"trn", "trn_gaba_b"} & cut else w_trn_gaba_b
        if "t_current" in cut:
            self.g_t = 0.0

        # topographic cortico-thalamic map: contiguous blocks of the site index.
        #
        # WHAT A CONTIGUOUS BLOCK ACTUALLY IS depends on the sheet, and this
        # comment used to say "`cortical_sites` lays sites out on a fibonacci
        # spiral, so a contiguous block is a latitude band".  That was never
        # true: the spherical proxy drew sites from a seeded RNG, so a block was
        # an arbitrary subset of a random point cloud -- the same "the name
        # asserted an anatomy the index did not have" that cost the video branch
        # a factor of 112 (docs/LOG.md ledger 25).
        #
        # on the fsaverage sheet a block IS something: sites are sorted by
        # vertex index, fsaverage vertex order is the icosahedral subdivision
        # order, and lh precedes rh -- so the first half of the blocks are left
        # hemisphere and the second half right, while WITHIN a hemisphere a
        # block is spread over the whole sheet rather than being a patch.  so it
        # is a hemisphere-respecting shuffle, not a topography.  a real
        # topographic map wants `cortical_regions` and is not built here.
        #
        # the ascending and descending limbs share it, which is the reciprocity
        # the process declaration asserts ("back onto the same relay cells"),
        # and that part is unaffected by what the blocks mean.
        grp = (torch.arange(n_sites, device=device) * n_thal) // n_sites
        self.register_buffer("group", grp.clamp_max(n_thal - 1).long())
        cnt = torch.zeros(n_thal, device=device).index_add_(
            0, self.group, torch.ones(n_sites, device=device))
        self.register_buffer("group_count", cnt.clamp_min(1.0))
        self._seed = seed
        self._ready = False

    # -- helpers ------------------------------------------------------------

    def rate(self, v):
        return self.r_max * torch.sigmoid((v - self.v_half) / self.slope)

    def m_inf(self, v):
        return torch.sigmoid((v - V_MT) / K_MT)

    def h_inf(self, v):
        return torch.sigmoid(-(v - self.v_deinact) / K_H)

    def reset(self, batch: int, dt: float, device=None, jitter: float = 2.0):
        """allocate state and the two delay lines.

        `jitter` puts a small deterministic spread on the initial relay
        potentials.  a perfectly homogeneous population started at its fixed
        point stays there in exact arithmetic, so a run started homogeneous
        measures how fast floating point noise breaks the symmetry rather than
        the loop's dynamics.  it is a seeded initial condition, not an input:
        it is applied once at t=0 and nothing is injected afterwards, which is
        what makes the spectra below spectra of an autonomous system.
        """
        dev = device
        n_tc = int(round(self.tc_delay_s / dt))
        n_ct = int(round(self.ct_delay_s / dt))
        if n_tc < 5 or n_ct < 5:
            raise ValueError(
                f"dt={dt:g}s resolves the {self.tc_delay_s*1e3:.0f}/"
                f"{self.ct_delay_s*1e3:.0f} ms loop delays as {n_tc}/{n_ct} steps. "
                "a delay line under ~5 steps is a rounding error, not a delay: "
                f"use dt <= {min(self.tc_delay_s, self.ct_delay_s)/5:.1e}s.")
        g = torch.Generator(device="cpu").manual_seed(self._seed)
        v0 = self.e_l_relay + jitter * torch.randn(
            batch, self.n_thal, generator=g).to(dev)
        self.v_t = v0
        self.h = self.h_inf(v0).clone()
        self.r_t = self.rate(v0).clone()
        self.v_n = torch.full((batch, self.n_thal), self.e_l_trn, device=dev)
        self.r_n = self.rate(self.v_n).clone()
        self.g_a = torch.zeros(batch, self.n_thal, device=dev)
        self.g_b = torch.zeros(batch, self.n_thal, device=dev)
        # ring buffers hold the SIGNAL, initialized at its resting value, so the
        # first delay-line-length of the run is not a step from zero.
        self.buf_tc = self.r_t[None].repeat(n_tc, 1, 1)
        self.buf_ct = torch.zeros(n_ct, batch, self.n_thal, device=dev)
        self.i_tc = self.i_ct = 0
        self.dt, self._ready = dt, True
        return self

    def ensure(self, batch: int, dt: float, device=None):
        """reset only if the shape or the timestep changed.

        this is what lets a thalamus stay attached to `CorticalDynamics` across a
        training loop: the loop calls `init_state` every forward, and the
        thalamus must NOT be reset every forward or it has no continuity between
        them.  `ThalamoCortical` carries its own state across forwards and is
        re-initialized only when the batch shape or the timestep actually
        changes.
        """
        if (not self._ready or self.dt != dt
                or self.v_t.shape[0] != batch or self.v_t.device != torch.device(
                    device if device is not None else self.v_t.device)):
            self.reset(batch, dt, device=device)
        return self

    def pool(self, r_cortex):
        """cortical rate (B, N) -> mean rate per thalamic unit (B, M)."""
        out = torch.zeros(r_cortex.shape[0], self.n_thal, device=r_cortex.device,
                          dtype=r_cortex.dtype)
        out.index_add_(1, self.group, r_cortex)
        return out / self.group_count

    # -- the loop -----------------------------------------------------------

    def step(self, r_cortex, dt: float):
        """advance the thalamus one step; return the cortical drive, (B, N) mV.

        `r_cortex` is the cortical rate at the END of the previous cortical
        step, so cortex and thalamus advance in lockstep with one step of
        implicit lag on top of the explicit delay lines.  at dt = 1e-4 s that
        lag is 0.1 ms against a 13 ms round trip.
        """
        if not self._ready:
            raise RuntimeError("call reset(batch, dt) before step()")

        # ---- descending limb: cortex -> thalamus, delayed by ct_delay_s ----
        self.buf_ct[self.i_ct] = self.pool(r_cortex)
        self.i_ct = (self.i_ct + 1) % self.buf_ct.shape[0]
        r_c_del = self.buf_ct[self.i_ct]                  # oldest = most delayed

        # ---- relay cells --------------------------------------------------
        m = self.m_inf(self.v_t)
        i_t = self.g_t * m * m * self.h * (E_CA - self.v_t)
        i_ct = 20.0 * self.w_ct_relay * r_c_del / self.r_max
        dv = (-(self.v_t - self.e_l_relay) + i_t + self.i_bg + i_ct) / self.tau_relay_s
        dv = dv - (self.v_t - E_GABA_A) * self.g_a.clamp_min(0.0) / self.tau_relay_s
        dv = dv - (self.v_t - E_GABA_B) * self.g_b.clamp_min(0.0) / self.tau_relay_s
        dh = (self.h_inf(self.v_t) - self.h) / self.tau_h
        # tonic rate plus the burst.  `burst_spikes` is declared as the number of
        # spikes a burst carries, and it enters as an ADDITIVE rate proportional
        # to the T-current's open fraction: a burst is extra spikes on top of
        # whatever the membrane potential alone would produce, which is the
        # distinction between burst and tonic mode the declaration insists on.
        r_inf = self.rate(self.v_t) + self.burst_spikes * self.r_max * (
            m * m * self.h) / 8.0

        # ---- TRN ----------------------------------------------------------
        drive_n = 20.0 * (self.w_relay_trn * self.r_t / self.r_max
                          + self.w_ct_trn * r_c_del / self.r_max)
        dvn = (-(self.v_n - self.e_l_trn) + drive_n) / self.tau_trn_s
        rn_inf = self.rate(self.v_n)
        dga = (self.w_trn_gaba_a * self.r_n / self.r_max - self.g_a) / self.tau_gaba_a_s
        dgb = (self.w_trn_gaba_b * self.r_n / self.r_max - self.g_b) / self.tau_gaba_b_s

        # ---- integrate (forward euler, as CorticalDynamics.step does) ------
        self.v_t = self.v_t + dt * dv
        self.h = (self.h + dt * dh).clamp(0.0, 1.0)
        self.r_t = self.r_t + dt * (r_inf - self.r_t) / self.tau_rate_s
        self.v_n = self.v_n + dt * dvn
        self.r_n = self.r_n + dt * (rn_inf - self.r_n) / self.tau_rate_s
        self.g_a = self.g_a + dt * dga
        self.g_b = self.g_b + dt * dgb

        # ---- ascending limb: thalamus -> cortex, delayed by tc_delay_s -----
        self.buf_tc[self.i_tc] = self.r_t
        self.i_tc = (self.i_tc + 1) % self.buf_tc.shape[0]
        r_t_del = self.buf_tc[self.i_tc]
        drive = 20.0 * self.w_tc * r_t_del / self.r_max
        return drive[:, self.group]                       # (B, M) -> (B, N)

    # -- observables --------------------------------------------------------

    def observe(self) -> dict:
        """the thalamic state, for a recording that is not the cortical one."""
        return {"v_relay": self.v_t, "r_relay": self.r_t, "h": self.h,
                "v_trn": self.v_n, "r_trn": self.r_n,
                "g_gaba_a": self.g_a, "g_gaba_b": self.g_b}


def run_closed_loop(dyn, tct: ThalamoCortical, n_steps: int, dt: float,
                    batch: int = 1, device=None, drive_fn=None,
                    burn_in: int = 0):
    """integrate cortex and thalamus together and record the mean rates.

    `drive_fn(i)` supplies external (sensory) drive if there is any.  the whole
    point of the measurement is the case where it is None: nothing enters the
    system after t=0, so anything in the recorded spectrum was generated by the
    system itself.
    """
    with torch.no_grad():
        s = dyn.init_state(batch, device)
        w = dyn.edge_weights()
        tct.reset(batch, dt, device=device)
        n_keep = n_steps - burn_in
        rec_c = torch.empty(n_keep, batch)
        rec_t = torch.empty(n_keep, batch)
        rec_v = torch.empty(n_keep, batch)
        for i in range(n_steps):
            drive = tct.step(s[1], dt)
            if drive_fn is not None:
                drive = drive + drive_fn(i)
            s = dyn.step(s, drive, dt, w)
            if i >= burn_in:
                j = i - burn_in
                rec_c[j] = s[1].mean(1).cpu()
                rec_t[j] = tct.r_t.mean(1).cpu()
                rec_v[j] = s[0].mean(1).cpu()
    return {"cortex_rate": rec_c.numpy(), "thal_rate": rec_t.numpy(),
            "cortex_v": rec_v.numpy()}


__all__ = ["ThalamoCortical", "run_closed_loop", "LIMBS"]
