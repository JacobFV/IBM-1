"""v3: one circuit engine.  Declared populations, projections with delays, modulation.

WHY THIS EXISTS
---------------
`docs/V3_ARCHITECTURE.md` has the argument; this is the object it decided on.  In short,
three measurements forced it: every v2-era module carries a hand-set dial where a
neuromodulatory nucleus should be (`m_beta`, `arousal`, `septal_tone`, `dopamine` -- four
floats produced by nothing); the build order was chosen by a metric computed from a
catalogue of oscillations, which is blind to every structure whose job is not rhythmic; and
**resonance modes cannot be computed at all** while each structure hand-rolls its own
populations and interfaces, because nothing assembles the loops.

So: one network.  A structure is a set of `Pop`s.  A pathway is a `Proj` with a conduction
delay.  A neuromodulator is a `Pop` whose rate MULTIPLIES a declared parameter of its
targets through a `Mod` edge -- which is what replaces the four dials.  The loops in
`ibm/rhythms.py` become projections, and the resonance of a loop becomes something to
measure on the assembled thing rather than to bound from its conduction budget.

THE THREE PROPERTIES THAT ARE STRUCTURAL
----------------------------------------
Each is here because a measurement said so, and each is recorded in the architecture doc
with the number that said it.

1. **Sparseness is inhibition, not a mask.**  A `Pop` with a declared `sparsity` carries an
   inhibitory partner driven by that population's OWN mean rate, with its gain derived so
   the partner is half-activated exactly at the target.  Nothing thresholds a k-largest set:
   the active fraction is a result of the dynamics at every step, which is what makes it
   able to fail.  `ibm/hippocampus.py` is the existence proof (4.4% / 8.9% / 11.1% measured
   across three populations); the v2 cortical field has no equivalent, and both the
   operating-point search (0 of 18) and the facilitation negative (selectivity 0.652) trace
   back to that absence.

2. **Attractors live in weights.**  Populations here are graded.  A recurrent `Proj` can
   carry a stored matrix (`store()`), which is how CA3 holds 6/6 retrieval at 6% active with
   no bistable unit anywhere.  The operating-point search showed a single sigmoidal rate per
   site cannot be both an attractor and a graded transfer function, because one loop gain
   sets both; so the memory does not live in the rates.

3. **Boundedness is a property, not a hope.**  Every variable is advanced by an
   exponential-Euler step -- a convex combination of the old value and a target inside the
   box -- so every rate stays in [0, 1] for any dt, any weights and any input.

UNITS: seconds, rates dimensionless in [0, 1], `R_MAX` = 100 Hz for readouts, delays in
seconds.  Every draw comes from a generator passed in; nothing here touches the global RNG.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

R_MAX = 100.0


# ======================================================================================
# declarations
# ======================================================================================
@dataclass(frozen=True)
class Pop:
    """a named rate population.

    `sparsity` is the fraction of units this population is DECLARED to run at.  Setting it
    gives the population an inhibitory partner; leaving it None means the population's level
    is set by its inputs alone, which is right for a relay and wrong for a cortex.
    """
    id: str
    n: int = 64
    kind: str = "E"                  # E | I | relay | mod | sensor | motor
    tau: float = 0.010
    beta: float = 10.0
    theta: float = 0.30
    r_rest: float = 0.0              # tonic level with no input (GPi and Purkinje are high)
    sparsity: float | None = None
    tau_inh: float = 0.005
    w_inh: float | None = None       # how hard the partner presses back.  None means
                                     # DERIVED: see Circuit.__init__ -- the inhibition must
                                     # be able to balance the strongest excitation this
                                     # population can receive, and that is a property of the
                                     # wiring, not a number to guess per population
    beta_inh: float = 8.0
    theta_inh: float = 0.25
    adapt_tau: float | None = None   # spike-frequency adaptation, when declared
    adapt_g: float = 0.0
    depress: bool = False            # Tsodyks-Markram resource
    tau_rec: float = 0.50
    U: float = 0.20
    facilitate: bool = False         # and its utilisation partner
    tau_f: float = 1.50
    sigma: float = 0.0               # OU background current, stationary std
    tau_eta: float = 0.005
    note: str = ""

    @property
    def structure(self) -> str:
        """'ctx' from 'ctx.v1.E' -- the structure a population belongs to."""
        return self.id.split(".", 1)[0]


@dataclass(frozen=True)
class Proj:
    """a pathway.  `delay_s` is conduction plus synaptic, and it is the thing the resonance
    of a loop is made of, so it is never optional and never silently zero: a delay too short
    for the dt in use is reported by `delays_in_steps`, not rounded away."""
    src: str
    dst: str
    weight: float = 1.0
    sign: int = 1                    # +1 excitatory, -1 inhibitory
    delay_s: float = 0.0
    topology: str = "dense"          # dense | sparse | one_to_one | diffuse
    p: float = 0.15                  # connection probability, for `sparse`
    stored: bool = False             # carries a pattern matrix written by store()
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.src}->{self.dst}"


@dataclass(frozen=True)
class Mod:
    """a neuromodulatory edge: `src`'s mean rate multiplies `param` on `dst`.

    This is the thing that replaces `m_beta`, `arousal`, `septal_tone` and `dopamine`.  A
    state -- waking, NREM, alert -- becomes where the network sits rather than a number a
    caller passes, which is what `ibm/rhythms.py`'s `arousal` row has always declared and
    never had.

        effective = declared * (1 + gain * (rate_src - baseline))

    clamped to `lo`..`hi` of the declared value, so a modulator can shape a parameter and
    cannot replace it.
    """
    src: str
    dst: str
    param: str                       # beta | theta | tau | sigma | w_inh | gain_in
    gain: float = 1.0
    baseline: float = 0.0
    lo: float = 0.25
    hi: float = 4.0
    note: str = ""


def sig(x, beta, theta):
    return torch.sigmoid(beta * (x - theta))


# ======================================================================================
# the network
# ======================================================================================
class Circuit(nn.Module):
    """populations, projections and modulation, stepped together.

    State is a dict of dicts: `state[pop_id]` holds `r` and whichever of `a`, `x`, `u`,
    `eta`, `inh` that population declared.  Ring buffers for delayed projections live in
    `state["_rings"]`, keyed by the projection, and `state["_t"]` is the step counter they
    are indexed by -- carried in the state rather than on the module, so two rollouts from
    the same state cannot interfere.
    """

    def __init__(self, pops, projs, mods=(), seed: int = 0, device="cpu"):
        super().__init__()
        self.pops = {p.id: p for p in pops}
        assert len(self.pops) == len(list(pops)), "duplicate population id"
        self.projs = list(projs)
        self.mods = list(mods)
        for pr in self.projs:
            assert pr.src in self.pops, f"{pr.key}: no such source"
            assert pr.dst in self.pops, f"{pr.key}: no such target"
        bad_mods = [f"{m.src}->{m.dst}" for m in self.mods
                    if m.src not in self.pops or m.dst not in self.pops]
        if bad_mods:
            raise ValueError(
                f"modulatory edges naming populations that do not exist: {bad_mods}. "
                f"`ibm.brain.collect` filters these into its orphan report before building "
                f"a Circuit; reaching this error means a caller assembled by hand.")
        self.device_ = device
        # every fixed connectivity draw comes from ONE dedicated generator, so the wiring is
        # a pure function of `seed` and not of whatever the caller drew before
        # A DUPLICATE PROJECTION IS A SILENT DOUBLING.  Weights are keyed by `Proj.key`, so
        # two declarations of the same src->dst overwrite one entry while BOTH still deliver
        # a contribution every step -- the pathway then runs at the sum of the two weights
        # and nothing says so.  Found by the valuation gate's assembly check, which noticed
        # `ctx.rostralmiddlefrontal.E->thal.md.relay` declared twice.  If two pathways
        # between the same populations are really wanted (a fast and a slow arm, say), they
        # are one `Proj` with the combined weight or two populations; they are never two
        # entries hoping the engine adds them.
        keys = [pr.key for pr in self.projs]
        dups = sorted({k for k in keys if keys.count(k) > 1})
        if dups:
            raise ValueError(
                f"duplicate projections: {dups}. Two declarations of the same src->dst "
                f"would run the pathway at the sum of their weights with nothing reporting "
                f"it. Declare one Proj, or two populations.")
        g = torch.Generator(device="cpu").manual_seed(seed)
        self._W = {}
        for pr in self.projs:
            self._W[pr.key] = self._make_weights(pr, g).to(device)
            if pr.stored:
                self.register_buffer(f"stored__{pr.key.replace('.', '_').replace('->', '__')}",
                                     torch.zeros(self.pops[pr.dst].n, self.pops[pr.src].n,
                                                 device=device))
        # the inhibitory gain of a sparse population is DERIVED, not declared: the partner
        # must be half-activated exactly at the target sparsity, or it switches instead of
        # grading.  ibm/hippocampus.py paid for that lesson (CA3 went to 0.000 active after
        # cue release with one shared gain).
        self.w_drive_inh = {}
        self.w_inh = {}
        for p in self.pops.values():
            if not p.sparsity:
                continue
            # the linear gain from the population's mean rate to its normalising
            # signal: 1.0 at the declared sparsity with active units near 0.5
            self.w_drive_inh[p.id] = 1.0 / max(p.sparsity * 0.5, 1e-6)
            if p.w_inh is not None:
                self.w_inh[p.id] = p.w_inh
                continue
            # DERIVED.  At saturation an incoming excitatory projection delivers
            # weight / src_sparsity (the row normalisation is by the source's declared
            # operating point), so the most this population can receive is the sum of those.
            # Inhibition that cannot reach it cannot prevent a runaway -- which is exactly
            # what the first smoke run did: a recurrent projection able to deliver 6.0
            # against a declared w_inh of 2.0, and every population pinned at 100 Hz.
            worst = 0.0
            for q in self.projs:
                if q.dst == p.id and q.sign > 0:
                    worst += q.weight / max(self.pops[q.src].sparsity or 1.0, 1e-6)
            # scaled DOWN from the saturation-balance figure, because that is the wrong end
            # of the range: measured on a mid-hierarchy cortical area, the useful band is
            # w_inh 0.05-0.10 (7% active, transfer slope 0.19) and the saturation-balance
            # estimate was 10.25.  Starting an order of magnitude above the answer costs the
            # multiplicative search four passes before it even enters the range where the
            # measurement means anything.
            worst *= 0.05
            # A STARTING POINT ONLY.  Balancing the saturation-level excitation is far too
            # strong at the operating point -- measured: ctx.E fell to 0.41 Hz with 1% active
            # against a declared 8%.  The gain that actually holds a population at its
            # declared sparsity is not an algebraic consequence of the wiring, it is a
            # property of the whole circuit at a drive, so it is MEASURED by
            # `calibrate_sparsity()` and this value is only where that search starts.
            self.w_inh[p.id] = max(0.05, worst)

    # ------------------------------------------------------------------ wiring
    def _make_weights(self, pr: Proj, g) -> torch.Tensor:
        n_src, n_dst = self.pops[pr.src].n, self.pops[pr.dst].n
        if pr.topology == "one_to_one":
            assert n_src == n_dst, f"{pr.key}: one_to_one needs equal sizes"
            m = torch.eye(n_dst)
        elif pr.topology == "diffuse":
            m = torch.ones(n_dst, n_src)
        elif pr.topology == "sparse":
            m = (torch.rand(n_dst, n_src, generator=g) < pr.p).float()
        elif pr.topology == "dense":
            m = torch.ones(n_dst, n_src)
        else:
            raise ValueError(f"{pr.key}: unknown topology {pr.topology!r}")
        # row-normalised, then divided by the sparsity the SOURCE is declared to run at, so
        # a weight is a gain on a population at its operating point rather than on a mean
        # rate that depends on how sparse the sender happens to be.  ibm/hippocampus.py ran
        # at 0.0005 Hz for exactly this reason before it was corrected.
        m = m / m.sum(1, keepdim=True).clamp_min(1.0)
        src_rate = self.pops[pr.src].sparsity or 1.0
        return m / max(src_rate, 1e-6)

    def store(self, proj_key: str, patterns: torch.Tensor, a: float | None = None):
        """write patterns into a `stored` projection by the Treves-Rolls rule.

        Presynaptic term mean-subtracted, postsynaptic not: a covariance matrix beside a
        global inhibitory population subtracts the mean twice, and the measured consequence
        in `ibm/hippocampus.py` was retrieval into the COMPLEMENT of the cue.
        """
        name = f"stored__{proj_key.replace('.', '_').replace('->', '__')}"
        W = getattr(self, name)
        p = patterns.to(W.device).float()
        a = float(p.mean()) if a is None else a
        add = p.t() @ (p - a)
        add.fill_diagonal_(0.0)
        W += add / max(1, p.shape[0]) / max(1e-6, a * (1 - a))
        return W

    def delays_in_steps(self, dt: float) -> dict:
        """what each declared delay becomes at this dt, and which ones vanish.

        A delay that rounds to zero is not an approximation, it is a different circuit --
        the loop it belongs to loses its conduction budget -- so the zeros are returned for
        a caller to look at rather than silently applied.
        """
        out = {}
        for pr in self.projs:
            out[pr.key] = int(round(pr.delay_s / dt))
        return out

    def vanishing_delays(self, dt: float) -> list:
        return [k for k, v in self.delays_in_steps(dt).items()
                if v == 0 and self.projs[[p.key for p in self.projs].index(k)].delay_s > 0]

    # ------------------------------------------------------------------ state
    def init_state(self, b: int = 1, dt: float = 1e-3, device=None) -> dict:
        dev = device or self.device_
        st = {"_t": 0, "_rings": {}}
        for p in self.pops.values():
            s = {"r": torch.full((b, p.n), float(p.r_rest), device=dev)}
            if p.sparsity:
                s["inh"] = torch.zeros(b, 1, device=dev)
            if p.adapt_tau:
                s["a"] = torch.zeros(b, p.n, device=dev)
            if p.depress:
                s["x"] = torch.ones(b, p.n, device=dev)
            if p.facilitate:
                s["u"] = torch.full((b, p.n), float(p.U), device=dev)
            if p.sigma:
                s["eta"] = torch.zeros(b, p.n, device=dev)
            st[p.id] = s
        for pr in self.projs:
            lag = int(round(pr.delay_s / dt))
            if lag > 0:
                st["_rings"][pr.key] = torch.zeros(b, lag + 1, self.pops[pr.src].n,
                                                   device=dev)
        return st

    @staticmethod
    def detach(state: dict) -> dict:
        out = {}
        for k, v in state.items():
            if k == "_rings":
                out[k] = {kk: vv.detach() for kk, vv in v.items()}
            elif isinstance(v, dict):
                out[k] = {kk: (vv.detach() if torch.is_tensor(vv) else vv)
                          for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------ modulation
    def _modulation(self, state) -> dict:
        """{pop_id: {param: factor}} from the modulatory populations' current rates."""
        out = {}
        for m in self.mods:
            r = state[m.src]["r"].mean()
            f = 1.0 + m.gain * (r - m.baseline)
            f = torch.clamp(f, m.lo, m.hi) if torch.is_tensor(f) else min(max(f, m.lo), m.hi)
            d = out.setdefault(m.dst, {})
            d[m.param] = d.get(m.param, 1.0) * f
        return out

    # ------------------------------------------------------------------ dynamics
    def step(self, state: dict, dt: float, drive=None, noise=None) -> dict:
        """one exponential-Euler step of the whole circuit.

        `drive` is {pop_id: tensor (B, n) or scalar}; `noise` is {pop_id: standard-normal
        (B, n)} supplied by the caller and never drawn here, so two passes can be made to
        see the same draw.
        """
        drive = drive or {}
        noise = noise or {}
        t = state["_t"]
        mod = self._modulation(state)
        rings = dict(state["_rings"])
        new = {"_t": t + 1, "_rings": rings}

        # what each projection delivers this step, read at its own lag.  EXCITATORY and
        # INHIBITORY sums are kept APART, because only the excitatory one is normalised.
        # Dividing the net input would divide the inhibition too, which disinhibits a
        # population exactly when it is busiest -- and that is a positive feedback wearing
        # the costume of a normaliser.
        exc = {pid: None for pid in self.pops}
        inh_in = {pid: None for pid in self.pops}
        for pr in self.projs:
            src = state[pr.src]
            s = src["r"]
            if self.pops[pr.src].depress:
                s = s * src["x"]
            if self.pops[pr.src].facilitate:
                s = s * (src["u"] / max(self.pops[pr.src].U, 1e-6))
            lag = int(round(pr.delay_s / dt))
            if lag > 0:
                ring = rings[pr.key]
                ring = ring.clone() if torch.is_grad_enabled() else ring
                ring[:, t % ring.shape[1]] = s
                rings[pr.key] = ring
                s_eff = ring[:, (t - lag) % ring.shape[1]]
            else:
                s_eff = s
            W = self._W[pr.key]
            if pr.stored:
                name = f"stored__{pr.key.replace('.', '_').replace('->', '__')}"
                contrib = (s_eff @ getattr(self, name).t()) / max(
                    self.pops[pr.src].n * (self.pops[pr.src].sparsity or 1.0), 1.0)
            else:
                contrib = s_eff @ W.t()
            contrib = pr.weight * contrib
            bag = exc if pr.sign > 0 else inh_in
            bag[pr.dst] = contrib if bag[pr.dst] is None else bag[pr.dst] + contrib

        for p in self.pops.values():
            s = state[p.id]
            r = s["r"]
            m = mod.get(p.id, {})
            beta = p.beta * m.get("beta", 1.0)
            theta = p.theta * m.get("theta", 1.0)
            tau = p.tau * m.get("tau", 1.0)
            sigma = p.sigma * m.get("sigma", 1.0)
            w_inh = (self.w_inh.get(p.id, 0.0)) * m.get("w_inh", 1.0)
            gain_in = m.get("gain_in", 1.0)

            e_in = exc[p.id] if exc[p.id] is not None else torch.zeros_like(r)
            i_in = inh_in[p.id] if inh_in[p.id] is not None else torch.zeros_like(r)
            d = drive.get(p.id)
            if d is not None:
                dd = d if torch.is_tensor(d) else torch.full_like(r, float(d))
                # a tonic drive is excitatory input and is normalised with the rest of it;
                # a hyperpolarising pulse is not, so the two signs are split here too
                e_in = e_in + dd.clamp_min(0.0)
                i_in = i_in - dd.clamp_max(0.0)
            e_in = e_in * gain_in

            out = {}
            eta_add = None
            if p.sigma:
                eta = s.get("eta", torch.zeros_like(r))
                rho = math.exp(-dt / p.tau_eta)
                eta = eta * rho
                z = noise.get(p.id)
                if z is not None:
                    eta = eta + sigma * math.sqrt(1.0 - rho * rho) * z
                out["eta"] = eta
                eta_add = eta
            adapt = None
            if p.adapt_tau:
                adapt = s["a"]
                out["a"] = s["a"] + (1 - math.exp(-dt / p.adapt_tau)) * (
                    p.adapt_g * r - s["a"])
            if p.sparsity:
                # DIVISIVE, not subtractive, and this is the one design decision in the file
                # that came from a failed measurement rather than from the literature first.
                #
                # A subtractive inhibitory population saturates: its rate is bounded by 1,
                # so the most it can ever subtract is `w_inh`, and past that the excitation
                # grows and nothing answers.  Measured on the first v3 smoke run: calibrated
                # to 7.6% active at its drive, the same population went to 100% SATURATED at
                # twice that drive -- the same ignition that made v2's operating-point search
                # return 0 of 18.  Inhibition that cannot scale cannot normalise.
                #
                # Divisive inhibition is what cortex is thought to do (Carandini & Heeger's
                # normalisation) and it has the property the measurement demands: double the
                # input and both the drive and the inhibition double, so the active fraction
                # is roughly preserved.  `inh` therefore tracks the population's own mean
                # rate LINEARLY -- no sigmoid, because a sigmoid would reintroduce exactly
                # the ceiling this replaces.
                inh = s["inh"]
                e_in = e_in / (1.0 + w_inh * inh)
                out["inh"] = inh + (1 - math.exp(-dt / p.tau_inh)) * (
                    self.w_drive_inh[p.id] * r.mean(-1, keepdim=True) - inh)

            u = e_in - i_in
            if eta_add is not None:
                u = u + eta_add
            if adapt is not None:
                u = u - adapt

            target = sig(u, beta, theta)
            if p.r_rest:
                # a tonically active population (GPi, Purkinje) rests HIGH and is driven
                # down; the sigmoid gives its deviation from that rest, not its level
                target = torch.clamp(torch.as_tensor(p.r_rest) + (target - sig(
                    torch.zeros_like(u), beta, theta)), 0.0, 1.0)
            out["r"] = r + (1 - math.exp(-dt / tau)) * (target - r)

            if p.depress:
                x = s["x"]
                use = s["u"] if p.facilitate else torch.full_like(r, p.U)
                # `r` is a NORMALISED rate in [0, 1]; depletion is per spike, so the rate
                # has to be in spikes per second -- r * R_MAX.  The first version wrote
                # `use * r * R_MAX / 100.0`, and R_MAX is 100, so the two cancelled and the
                # depletion rate was U*r <= 0.25/s against a recovery of 1/tau_rec = 6.67/s.
                # Measured by the thalamus gate: a relay population's resource sat at 0.9913
                # with a range of 0.0013 over five seconds.  Synaptic depression has never
                # depleted anything in this engine, in any population declaring it -- and
                # the facilitation branch eight lines below carried a comment about exactly
                # this units error, written by the same hand that made it here.
                kx = 1.0 / p.tau_rec + use * r * R_MAX
                x_inf = (1.0 / p.tau_rec) / kx
                out["x"] = x_inf + (x - x_inf) * torch.exp(-dt * kx)
            if p.facilitate:
                uu = s["u"]
                # the facilitation rate constant is per SPIKE, so the drive is r * R_MAX --
                # written with the normalised rate it is ~100x too small, which cost an
                # afternoon in ibm/substrate.py (docs/LOG.md)
                ku = 1.0 / p.tau_f + p.U * r * R_MAX
                u_inf = (p.U / p.tau_f + p.U * r * R_MAX) / ku
                out["u"] = (u_inf + (uu - u_inf) * torch.exp(-dt * ku)).clamp(0.0, 1.0)
            new[p.id] = out
        return new

    def rollout(self, steps: int, dt: float, state=None, drive=None, noise_gen=None,
                b: int = 1, record=None, burn: int = 0):
        """returns ({pop_id: (B, T, n)}, final state).  `record` defaults to every
        population.  `burn` steps run first and are not recorded; under `no_grad` when the
        caller is building a graph, so a burn-in does not cost memory it will not use."""
        state = state or self.init_state(b, dt)
        record = list(record or self.pops)
        out = {k: [] for k in record}
        for i in range(steps + burn):
            z = None
            if noise_gen is not None:
                z = {p.id: torch.randn(b, p.n, generator=noise_gen).to(self.device_)
                     for p in self.pops.values() if p.sigma}
            d = drive(i - burn) if callable(drive) else drive
            if i < burn:
                with torch.no_grad():
                    state = self.step(state, dt, drive=d, noise=z)
                if i == burn - 1:
                    state = self.detach(state)
            else:
                state = self.step(state, dt, drive=d, noise=z)
                for k in record:
                    out[k].append(state[k]["r"])
        return {k: torch.stack(v, 1) for k, v in out.items()}, state

    # ------------------------------------------------------------------ calibration
    def calibrate_sparsity(self, drive, dt: float = 1e-3, seconds: float = 1.5,
                           passes: int = 12, seed: int = 0, alpha: float = 0.7,
                           step_clip: float = 2.0, verbose: bool = False):
        """bring every sparse population to its declared sparsity, all at once.

        A MEASUREMENT, not a guess: the gain that holds a population at 8% active depends on
        every projection into it, on the drive, and on what the populations upstream are
        doing, so it cannot be read off the declarations.  `ibm/hippocampus.py` needed
        exactly this and got it by hand -- three gains swept until the sparsities matched --
        and that is not a step anyone should repeat per structure.

        All populations are updated TOGETHER by a multiplicative fixed point,

            w <- w * clip((measured / target) ** alpha, 1/step_clip, step_clip)

        rather than bisected one at a time.  Bisecting 30 cortical populations would need
        about 840 rollouts of the whole brain; this needs `passes`.  The populations are
        coupled -- quietening the dentate changes what CA3 receives -- so a simultaneous
        update is also the honest shape: it finds a joint operating point rather than a
        sequence of local ones that each invalidate the last.

        Returns what it measured per population, including any that did not reach target,
        which is reported rather than left silently wrong.
        """
        sparse_pops = [p for p in self.pops.values() if p.sparsity]
        if not sparse_pops:
            return {}
        keep = int(min(0.5, seconds * 0.5) / dt)
        history = []
        for it in range(passes):
            g = torch.Generator().manual_seed(seed)
            # the burn-in has to outlast the IGNITION TRANSIENT, not just the membrane
            # time constants.  At 0.5 s it read 0% active at a setting that ignites a second
            # later, and drove the gain the wrong way -- which is how the cortical
            # calibration ended up oscillating instead of converging.
            tr, _ = self.rollout(int(seconds / dt), dt, state=self.init_state(1, dt),
                                 drive=drive, noise_gen=g, burn=int(max(1.5, seconds) / dt),
                                 record=[p.id for p in sparse_pops])
            row = {}
            for p in sparse_pops:
                act = float((tr[p.id][:, -keep:] > 0.2).float().mean())
                row[p.id] = act
                ratio = (max(act, 1e-4) / p.sparsity) ** alpha
                ratio = min(max(ratio, 1.0 / step_clip), step_clip)
                self.w_inh[p.id] = max(1e-3, self.w_inh[p.id] * ratio)
            history.append(row)
            if verbose:
                err = max(abs(row[p.id] - p.sparsity) / p.sparsity for p in sparse_pops)
                print(f"    pass {it:2d}  worst relative error {err:6.2%}", flush=True)
        g = torch.Generator().manual_seed(seed)
        tr, _ = self.rollout(int(seconds / dt), dt, state=self.init_state(1, dt),
                             drive=drive, noise_gen=g, burn=int(max(1.5, seconds) / dt),
                             record=[p.id for p in sparse_pops])
        out = {}
        for p in sparse_pops:
            act = float((tr[p.id][:, -keep:] > 0.2).float().mean())
            out[p.id] = {"w_inh": float(self.w_inh[p.id]), "active": act,
                         "target": p.sparsity,
                         "on_target": abs(act - p.sparsity) <= max(0.3 * p.sparsity, 0.01)}
        self.calibration = out
        return out

    # ------------------------------------------------------------------ readouts
    def rate_hz(self, state, pop_id: str) -> float:
        return float(state[pop_id]["r"].mean() * R_MAX)

    def active_fraction(self, state, pop_id: str, thr: float = 0.2) -> float:
        return float((state[pop_id]["r"] > thr).float().mean())

    def summary(self, state) -> dict:
        return {pid: {"rate_hz": self.rate_hz(state, pid),
                      "active": self.active_fraction(state, pid)}
                for pid in self.pops}
