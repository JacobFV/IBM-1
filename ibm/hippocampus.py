"""the hippocampal formation: separation in the dentate, completion in CA3, theta from a loop.

WHY THIS EXISTS
---------------
`docs/RHYTHMS.md` files **seven** rhythms under structures this module supplies, and they are
not a list of bands -- they are the machinery the programme's episodic claims rest on:
hippocampal theta, the gamma nested inside it, CA3's slow gamma and the entorhinal fast
gamma, sharp-wave ripples, the ripple-spindle-slow-oscillation nesting, and phase precession.
`ibm/substrate.py` addresses one `hippocampus` blob, so none of them could be expressed, let
alone scored.

And one more reason, which is why this was built before the basal ganglia or the cerebellum.
Today's operating-point measurement (docs/LOG.md, 18 Sep 2026) found the cortical sheet has no
regime between silent and saturated, and that the inhibition which gives it a firing range
costs it its attractors -- 8 invariant sets down to 1. **CA3 is multistability built a
different way.** Its attractors come from a recurrent weight matrix holding stored patterns,
not from each site sitting on a bistable sigmoid, so it is the one structure in the
specification that can hold a discrete state without every unit having to be bistable on its
own. Whether that is the answer to the cortical problem is an open question; having the
mechanism in the repo is the precondition for asking it.

THE MODEL
---------
Five populations, each a vector of rates in [0, 1], plus one inhibitory scalar per region and
a septal pacemaker:

    EC   entorhinal input           the cue, and the direct path to CA1
    DG   dentate granule cells      sparse, strongly inhibited: SEPARATION
    CA3  pyramidal cells            recurrent, pattern-storing: COMPLETION
    CA1  pyramidal cells            CA3 via Schaffer + EC direct: the comparator
    SUB  subiculum                  the output stage
    MS   medial septum              an E/I pair with adaptation: the theta PACEMAKER

    tau_E dE/dt = -E + F(u_E)
    u_E = W_rec E  +  W_ff (upstream)  -  w_I I_region  -  a  +  drive  +  eta
    tau_I dI/dt = -I + F(w_IE mean(E) - theta_I - septal_disinhibition(t))
    tau_a da/dt = -a + g_a E

**Sparseness is inhibition, not a mask.** DG and CA3 are held sparse by a global inhibitory
population whose input is the region's own mean rate -- so the fraction active is a *result*
of the dynamics at every step, and a pattern that tries to activate too much shuts itself
down. Nothing thresholds a k-largest set by hand; a hand-picked k is a decision the model
should be making.

**Theta is a loop, not a clock.** The septum's E/I pair with adaptation oscillates in the
theta band, its output disinhibits the hippocampal interneurons, and CA3's mean rate feeds
back to the septum. Cutting that feedback is a gate (H4b): if theta survives untouched, it
was an imposed drive and this module would be lying about where it comes from.

**Ripples are the same circuit in the other state.** With the septal drive withdrawn, CA3's
recurrent excitation is free to detonate, and CA1's fast interneurons ring at 140-200 Hz.
One circuit, two regimes, selected by septal tone -- the same shape as the thalamus's
arousal.

**Storage is experience, not gradient descent.** `store()` writes patterns into CA3's
recurrent matrix by a covariance rule, into a BUFFER the optimiser never touches, exactly as
`ibm/substrate.py` handles its `P`. Weights drawn at construction come from an explicit
generator passed in and never from the global RNG (CLAUDE.md's standing trap).

UNITS: seconds, rates in [0, 1], `R_MAX` = 100 Hz for readouts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

R_MAX = 100.0


@dataclass
class HippocampalPriors:
    """every constant declared with its reason; none fitted."""
    # membranes.  CA1's inhibition is the fastest thing here because it is what has to ring
    # at 200 Hz during a ripple: a 5 ms interneuron cannot.
    tau_E: float = 0.010
    tau_I: float = 0.004
    tau_I_ca1: float = 0.0022
    tau_a: float = 0.180                 # adaptation, the slow variable that releases a state
    g_a: float = 0.55
    # sparseness.  Measured target: ~1-3% of granule cells active, ~5-10% of CA3.  These are
    # the inhibitory gains that produce it, not a k to select.
    w_I_dg: float = 6.5
    w_I_ca3: float = 3.2
    w_I_ca1: float = 2.6
    w_IE: float = 14.0                   # region mean rate -> its inhibitory population
    theta_I: float = 0.055
    beta_E: float = 12.0
    beta_I: float = 10.0
    theta_E_dg: float = 0.42
    theta_E_ca3: float = 0.34
    theta_E_ca1: float = 0.30
    # pathways.  The mossy fibre is the "detonator": few, strong, and it is what makes DG's
    # sparse code able to set CA3's state at all.
    w_pp: float = 0.55                   # perforant path, EC -> DG
    w_mf: float = 2.40                   # mossy fibre, DG -> CA3
    w_rec: float = 1.35                  # CA3 recurrent gain on the stored matrix
    w_sc: float = 1.10                   # Schaffer collateral, CA3 -> CA1
    w_ec_ca1: float = 0.45               # EC direct to CA1: the comparison path
    w_ca1_sub: float = 1.20
    # the septum.  An E/I pair with adaptation; these put it in the theta band, and the gate
    # measures where it actually lands rather than trusting the arithmetic.
    tau_ms_E: float = 0.020
    tau_ms_I: float = 0.012
    tau_ms_a: float = 0.075
    g_ms_a: float = 1.30
    w_ms_EE: float = 2.20
    w_ms_IE: float = 2.00
    w_ms_EI: float = 1.80
    theta_ms: float = 0.32
    beta_ms: float = 9.0
    w_ms_out: float = 0.42               # septal output -> disinhibition of hippocampal I
    w_hpc_ms: float = 0.85               # CA3 mean rate -> septum: the return limb
    # conduction, from ibm/rhythms.py's `hpc` and `septohpc` loops
    d_pp_s: float = 0.005
    d_mf_s: float = 0.003
    d_sc_s: float = 0.003
    d_ms_s: float = 0.007
    # background
    sigma: float = 0.035
    tau_eta: float = 0.005


def F(u, beta, theta):
    return torch.sigmoid(beta * (u - theta))


class Hippocampus(nn.Module):
    """one hippocampal formation: EC -> DG -> CA3 -> CA1 -> subiculum, with a septum.

    `n_ec`, `n_dg`, `n_ca3`, `n_ca1` are unit counts, not cells: each is a small population.
    The dentate is the largest because the expansion is the mechanism -- more granule cells
    than entorhinal inputs is what lets two similar inputs land on different sparse codes.
    """

    def __init__(self, n_ec: int = 120, n_dg: int = 480, n_ca3: int = 180, n_ca1: int = 180,
                 priors: HippocampalPriors | None = None, seed: int = 0, device="cpu",
                 sparsity_pp: float = 0.12, sparsity_mf: float = 0.06):
        super().__init__()
        self.pr = priors or HippocampalPriors()
        self.n_ec, self.n_dg, self.n_ca3, self.n_ca1 = n_ec, n_dg, n_ca3, n_ca1
        # every fixed projection is drawn ONCE from a dedicated generator.  A graph drawn
        # from the global RNG makes the topology a function of the caller's weight seed,
        # which is the trap CLAUDE.md records twice.
        g = torch.Generator(device="cpu").manual_seed(seed)
        pp = (torch.rand(n_dg, n_ec, generator=g) < sparsity_pp).float()
        mf = (torch.rand(n_ca3, n_dg, generator=g) < sparsity_mf).float()
        sc = (torch.rand(n_ca1, n_ca3, generator=g) < 0.25).float()
        ec1 = (torch.rand(n_ca1, n_ec, generator=g) < 0.20).float()
        # row-normalised so a gain is a gain and not a fan-in count
        self.register_buffer("W_pp", (pp / pp.sum(1, keepdim=True).clamp_min(1)).to(device))
        self.register_buffer("W_mf", (mf / mf.sum(1, keepdim=True).clamp_min(1)).to(device))
        self.register_buffer("W_sc", (sc / sc.sum(1, keepdim=True).clamp_min(1)).to(device))
        self.register_buffer("W_ec1", (ec1 / ec1.sum(1, keepdim=True).clamp_min(1)).to(device))
        # CA3's recurrent matrix is EXPERIENCE: written by `store`, never by an optimiser.
        self.register_buffer("W_rec", torch.zeros(n_ca3, n_ca3, device=device))
        self.register_buffer("stored", torch.zeros(0, n_ca3, device=device))
        self.device_ = device

    # ------------------------------------------------------------------ storage
    def store(self, patterns: torch.Tensor, normalise: bool = True):
        """write patterns into CA3 by a covariance rule.  (P, n_ca3) in [0, 1].

        Covariance and not plain Hebb: a plain outer product of sparse patterns is
        all-positive and every pattern then excites every other, which shows up as a model
        that "completes" anything you give it, including a pattern it never stored.  The
        mean-subtracted rule is what makes the stored set discriminable, and the gate that
        cues with an UNSTORED pattern is what would catch it if this were wrong.
        """
        p = patterns.to(self.W_rec.device).float()
        a = p.mean()
        dev = p - a
        W = dev.t() @ dev
        W.fill_diagonal_(0.0)
        if normalise:
            W = W / max(1, p.shape[0]) / max(1e-6, a * (1 - a))
        self.W_rec += W
        self.stored = torch.cat([self.stored, p], 0)
        return self.W_rec

    def overlap(self, ca3: torch.Tensor) -> torch.Tensor:
        """normalised overlap of a CA3 state with every stored pattern.  (B, P)."""
        if self.stored.numel() == 0:
            return torch.zeros(ca3.shape[0], 0, device=ca3.device)
        x = ca3 - ca3.mean(-1, keepdim=True)
        s = self.stored - self.stored.mean(-1, keepdim=True)
        num = x @ s.t()
        den = x.norm(dim=-1, keepdim=True) * s.norm(dim=-1)[None, :] + 1e-9
        return num / den

    # ------------------------------------------------------------------ state
    def init_state(self, b: int = 1, device=None):
        d = device or self.W_rec.device
        z = lambda n: torch.zeros(b, n, device=d)  # noqa: E731
        return {"dg": z(self.n_dg), "ca3": z(self.n_ca3), "ca1": z(self.n_ca1),
                "sub": z(self.n_ca1), "a3": z(self.n_ca3),
                "i_dg": torch.zeros(b, 1, device=d), "i_ca3": torch.zeros(b, 1, device=d),
                "i_ca1": torch.zeros(b, 1, device=d),
                "ms_E": torch.zeros(b, 1, device=d), "ms_I": torch.zeros(b, 1, device=d),
                "ms_a": torch.zeros(b, 1, device=d),
                "eta": z(self.n_ca3), "hist_mf": None, "hist_sc": None, "ptr": 0}

    @staticmethod
    def detach(state):
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in state.items()}

    # ------------------------------------------------------------------ dynamics
    def step(self, state, dt: float, ec=None, septal_tone: float = 1.0,
             noise: torch.Tensor | None = None, ms_feedback: bool = True,
             ca3_clamp: torch.Tensor | None = None):
        """one exponential-Euler step.

        `septal_tone` in [0, 1] is how strongly the septum drives the hippocampus: 1 is
        exploration (theta), 0 is quiet rest (the state ripples live in).  It is the same
        kind of object as the thalamus's `arousal` -- a context the circuit is put in.

        `ms_feedback=False` cuts the hippocampal return limb to the septum, which gate H4b
        uses: if theta is unchanged without it, theta was never a loop.
        """
        pr = self.pr
        dg, ca3, ca1 = state["dg"], state["ca3"], state["ca1"]
        a3 = state["a3"]
        eta = state.get("eta")
        if eta is None:
            eta = torch.zeros_like(ca3)
        ec = ec if ec is not None else torch.zeros(ca3.shape[0], self.n_ec, device=ca3.device)

        rho = math.exp(-dt / pr.tau_eta)
        eta = eta * rho
        if noise is not None:
            eta = eta + pr.sigma * math.sqrt(1.0 - rho * rho) * noise

        # ---- the septum: an E/I pair with adaptation, which is what makes it oscillate
        msE, msI, msa = state["ms_E"], state["ms_I"], state["ms_a"]
        back = pr.w_hpc_ms * ca3.mean(-1, keepdim=True) if ms_feedback else torch.zeros_like(msE)
        u_msE = pr.w_ms_EE * msE - pr.w_ms_EI * msI - msa + back + 0.30 * septal_tone
        u_msI = pr.w_ms_IE * msE
        fE = F(u_msE, pr.beta_ms, pr.theta_ms)
        fI = F(u_msI, pr.beta_ms, pr.theta_ms)
        msE2 = msE + (1 - math.exp(-dt / pr.tau_ms_E)) * (fE - msE)
        msI2 = msI + (1 - math.exp(-dt / pr.tau_ms_I)) * (fI - msI)
        msa2 = msa + (1 - math.exp(-dt / pr.tau_ms_a)) * (pr.g_ms_a * msE - msa)
        # the septal output DISINHIBITS: it lands on interneurons, so hippocampal inhibition
        # is rhythmically lifted rather than the pyramids being rhythmically driven.  That
        # is the anatomy, and it is also what makes theta a window rather than a push.
        disinh = pr.w_ms_out * septal_tone * msE2

        # ---- inhibitory populations, one per region, driven by the region's own mean rate
        i_dg, i_ca3, i_ca1 = state["i_dg"], state["i_ca3"], state["i_ca1"]
        i_dg2 = i_dg + (1 - math.exp(-dt / pr.tau_I)) * (
            F(pr.w_IE * dg.mean(-1, keepdim=True) - disinh, pr.beta_I, pr.theta_I) - i_dg)
        i_ca32 = i_ca3 + (1 - math.exp(-dt / pr.tau_I)) * (
            F(pr.w_IE * ca3.mean(-1, keepdim=True) - disinh, pr.beta_I, pr.theta_I) - i_ca3)
        i_ca12 = i_ca1 + (1 - math.exp(-dt / pr.tau_I_ca1)) * (
            F(pr.w_IE * ca1.mean(-1, keepdim=True) - disinh, pr.beta_I, pr.theta_I) - i_ca1)

        # ---- the three stages
        u_dg = pr.w_pp * (ec @ self.W_pp.t()) - pr.w_I_dg * i_dg
        dg2 = dg + (1 - math.exp(-dt / pr.tau_E)) * (
            F(u_dg, pr.beta_E, pr.theta_E_dg) - dg)

        rec = (ca3 @ self.W_rec.t()) / max(1, self.n_ca3)
        u_ca3 = (pr.w_mf * (dg @ self.W_mf.t()) + pr.w_rec * rec
                 - pr.w_I_ca3 * i_ca3 - a3 + eta)
        ca3_2 = ca3 + (1 - math.exp(-dt / pr.tau_E)) * (
            F(u_ca3, pr.beta_E, pr.theta_E_ca3) - ca3)
        if ca3_clamp is not None:
            ca3_2 = torch.where(torch.isnan(ca3_clamp), ca3_2, ca3_clamp)
        a3_2 = a3 + (1 - math.exp(-dt / pr.tau_a)) * (pr.g_a * ca3 - a3)

        u_ca1 = (pr.w_sc * (ca3 @ self.W_sc.t()) + pr.w_ec_ca1 * (ec @ self.W_ec1.t())
                 - pr.w_I_ca1 * i_ca1)
        ca1_2 = ca1 + (1 - math.exp(-dt / pr.tau_E)) * (
            F(u_ca1, pr.beta_E, pr.theta_E_ca1) - ca1)
        sub2 = state["sub"] + (1 - math.exp(-dt / pr.tau_E)) * (
            F(pr.w_ca1_sub * ca1, pr.beta_E, pr.theta_E_ca1) - state["sub"])

        return {"dg": dg2, "ca3": ca3_2, "ca1": ca1_2, "sub": sub2, "a3": a3_2,
                "i_dg": i_dg2, "i_ca3": i_ca32, "i_ca1": i_ca12,
                "ms_E": msE2, "ms_I": msI2, "ms_a": msa2, "eta": eta,
                "hist_mf": state.get("hist_mf"), "hist_sc": state.get("hist_sc"),
                "ptr": state.get("ptr", 0)}

    def rollout(self, steps: int, dt: float, state=None, ec=None, septal_tone: float = 1.0,
                noise_gen=None, b: int = 1, ms_feedback: bool = True, record=("ca1", "ca3")):
        """returns ({name: (B, T, n)}, final state).  `ec` may be (B, n_ec) held constant
        or (B, steps, n_ec).  Noise comes from the generator passed in; none is drawn here."""
        state = state or self.init_state(b)
        out = {k: [] for k in record}
        for t in range(steps):
            e = ec[:, t] if (torch.is_tensor(ec) and ec.dim() == 3) else ec
            z = None
            if noise_gen is not None:
                z = torch.randn(b, self.n_ca3, generator=noise_gen,
                                device="cpu").to(self.W_rec.device)
            state = self.step(state, dt, ec=e, septal_tone=septal_tone, noise=z,
                              ms_feedback=ms_feedback)
            for k in record:
                out[k].append(state[k])
        return {k: torch.stack(v, 1) for k, v in out.items()}, state

    # ------------------------------------------------------------------ helpers
    def sparsity(self, x: torch.Tensor, thr: float = 0.2) -> float:
        """fraction of units above `thr` -- what the inhibition is supposed to be setting."""
        return float((x > thr).float().mean())

    def make_patterns(self, p: int, active: float = 0.08, generator=None) -> torch.Tensor:
        """p random sparse CA3 patterns.  A generator must be PASSED: two arms of a gate
        that drew their own would not be comparing the same patterns."""
        assert generator is not None, "pass an explicit generator"
        x = (torch.rand(p, self.n_ca3, generator=generator) < active).float()
        return x.to(self.W_rec.device)
