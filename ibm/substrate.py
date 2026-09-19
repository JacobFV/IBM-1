"""the cortical field, v2: nonlinear, bounded, heterogeneous, stateful, metastable.

why a second substrate, and not an edit to `CorticalDynamics`
-------------------------------------------------------------
five independent measurements (docs/LOG.md, 2026-09-12, "the triad closes") say the
v1 sheet contributes nothing on the one task where the objective works: trained,
frozen-at-random and ABSENT all score 4.2-4.4x chance.  reading `CorticalDynamics`
says why, and it is architectural rather than a tuning failure:

  1. **it has no memory across inputs.**  every forward calls `init_state`, and a
     forward is 4 x 4 substeps of 5 ms -- 80 ms from rest, against its own 300 ms
     adaptation constant.  the sheet is a short transient from a fixed point, i.e.
     a smooth feedforward filter of the current drive, and anything a filter of the
     current drive can do, a readout of the drive can do.  "memory beyond the input
     window" is impossible for it BY CONSTRUCTION.
  2. **it is homogeneous.**  tau_m, tau_a, the threshold, the slope, w_ee, w_ei are
     one scalar each, shared by every site.  every site is the same dynamical
     system, so the sheet can only express a spatial pattern, never a spatial
     pattern of TIMESCALES -- and the cortical timescale hierarchy (sensory ~tens of
     ms, association ~hundreds) is one of the best-measured facts about cortex.
  3. **its multistability is an accident of parameters.**  STATE.md measured
     bistable cells in ~0.05% of a swept parameter cube, gone at prior medians, and
     found inhibition LINEARISES the only bistability it had.  a substrate whose
     attractor landscape exists only on a sliver of parameter space the optimiser
     never visits has no landscape.
  4. **its voltage is unbounded.**  the forward-Euler conductance model diverged to
     7.5e15 mV once, and "bounded" has been a thing to check rather than a property.

v1 stays exactly as it is -- every checkpoint on disk loads into it and must keep
loading -- and this module is the replacement the next experiments are built on.

the model
---------
per cortical site i, five state variables, all bounded by construction:

    E_i  excitatory population rate           in [0, 1]   (x R_MAX for Hz)
    I_i  inhibitory population rate           in [0, 1]
    a_i  spike-frequency adaptation           in [0, g_a,i]
    x_i  synaptic resource (Tsodyks-Markram)  in [0, 1]
    u_i  synaptic utilisation (facilitation)  in [0, 1]   OPTIONAL; the memory, when on
    eta_i background current, Ornstein-Uhlenbeck, stationary std sigma * m_sigma

    u_E = w_EE,i x_i E_i  +  G_L sum_j W+_ij x_j E_j(t - d_ij)
          - w_EI,i I_i  -  a_i  +  drive_i  +  eta_i(t)
    u_I = w_IE,i E_i  +  G_F sum_j W-_ij E_j(t - d_ij)  -  w_II I_i  +  drive_I,i

    tau_E,i dE/dt = -E + F(u_E; beta_E,i * m_beta, theta_E,i)
    tau_I   dI/dt = -I + F(u_I; beta_I, theta_I)
    tau_a,i da/dt = -a + g_a,i E
            dx/dt = (1 - x) / tau_rec,i  -  U x E / tau_use

    F(u; beta, theta) = sigmoid(beta (u - theta))

**bounded.**  every variable is advanced by an EXPONENTIAL-Euler step,
`y <- y + (1 - exp(-dt/tau)) (target - y)`, which is a convex combination of the old
value and a target inside the box -- so E, I, x stay in [0, 1] and a in [0, g_a] for
ANY dt, any parameters and any input.  divergence is not something to test for any
more; it cannot be represented.  (the gate script still checks it, because a claim
of "cannot happen" is exactly the kind of thing to measure once.)

**multistable where the cortex is.**  a column with recurrent gain
`beta w_EE / 4 > 1` has a pitchfork: a down state and an up state.  the prior puts
association cortex ABOVE that threshold and primary sensory cortex below it -- the
arrangement Wang (2001) and Chaudhuri et al. (2015) use to get persistent activity
in prefrontal cortex and fast, input-following responses in V1 from one
architecture.  it is the prior, not a fitted value; the landscape is where the
parameters start, not a region the optimiser has to discover.

**quasi-multistable, not frozen.**  an up state recruits adaptation (tau_a) and
depletes its own synaptic resource (tau_rec), and both erode the recurrent gain that
holds it, so an attractor is held for a dwell time and then released.  with noise,
that is itinerancy between metastable states -- the object DYNAMICS.md §1 says a
percept, a concept or an intention IS.  a fixed-point attractor network holds its
state forever and cannot move on; this one cannot help moving on, at a pace set per
site.

**heterogeneous.**  tau_E, tau_a, tau_rec, w_EE and the threshold are PER SITE, each a
declared function of the site's place on the sensory-association hierarchy plus a
learned, bounded residual.  see HIERARCHY below: it is a declared ordinal prior and
is marked for replacement by the Sydnor 2021 S-A map when that corpus is bound.

**long-range coupling excites and inhibits, and the sign is a learned property of
the pair.**  association fibres are glutamatergic, so they cannot carry a negative
weight onto a pyramidal cell; what they CAN do is land on interneurons -- feedforward
inhibition.  each edge therefore has an E-target weight W+ and an I-target weight W-,
both >= 0, driven in opposite directions by the same learned pair similarity: similar
columns excite each other, dissimilar ones reach each other's interneurons.  the
kernel can subtract (which v1 needed tanh for, see its `edge_weights` comment) without
ever giving an excitatory axon a negative synapse.

**stateful.**  `rollout` carries the state from one window to the next and returns
it; callers detach it at window boundaries (truncated BPTT).  resetting to rest every
forward is what made v1 memoryless, and it is not done here unless asked for.

**modulated.**  `m_beta` and `m_sigma` are global neuromodulatory gains (arousal): the
same landscape run at higher gain and lower noise is more stable, at lower gain and
higher noise more itinerant.  DYNAMICS.md mechanism 15, as the smallest thing that
can express a regime change.

units: time in seconds, rates dimensionless in [0, 1] (`rate_hz` multiplies by
R_MAX), drive in the same units as u.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

R_MAX = 100.0          # Hz, for readouts that want v1's units

# --------------------------------------------------------------------------------------
# HIERARCHY.  a site's place on the sensory-association axis, 0 = primary, 1 = transmodal.
#
# DECLARED, ORDINAL, and to be REPLACED.  the right object is a measured map -- the
# Sydnor et al. (2021) sensorimotor-association axis, or the Murray et al. (2014) /
# Raut et al. (2020) intrinsic-timescale maps -- and all three have source cards in
# data/sources (`sydnor2021-hierarchy`, `intrinsic-timescale-maps`,
# `margulies2016-gradients`) whose raw/ holds no bytes.  until one is bound, this table
# is the ordering those papers agree on, per Desikan-Killiany gyrus, rounded to tenths
# on purpose: it carries RANK, and a third decimal would claim a precision nobody
# measured.  a label that is not here gets 0.5 and is reported by `hierarchy_of`.
# --------------------------------------------------------------------------------------
HIERARCHY = {
    # primary sensory and motor
    "pericalcarine": 0.0, "precentral": 0.05, "postcentral": 0.05,
    "transversetemporal": 0.05, "paracentral": 0.1, "cuneus": 0.1,
    # unimodal association
    "lingual": 0.15, "lateraloccipital": 0.2, "superiortemporal": 0.25,
    "fusiform": 0.35, "superiorparietal": 0.4, "supramarginal": 0.45,
    "bankssts": 0.45, "inferiortemporal": 0.5, "middletemporal": 0.5,
    # heteromodal / transmodal
    "caudalmiddlefrontal": 0.55, "parsopercularis": 0.6, "insula": 0.6,
    "inferiorparietal": 0.6, "parstriangularis": 0.7, "superiorfrontal": 0.75,
    "parsorbitalis": 0.75, "rostralmiddlefrontal": 0.8, "lateralorbitofrontal": 0.8,
    "caudalanteriorcingulate": 0.8, "precuneus": 0.8, "parahippocampal": 0.8,
    "posteriorcingulate": 0.85, "isthmuscingulate": 0.85, "entorhinal": 0.85,
    "medialorbitofrontal": 0.9, "frontalpole": 0.9, "rostralanteriorcingulate": 0.9,
    "temporalpole": 0.9,
}


def hierarchy_of(names) -> tuple[torch.Tensor, list[str]]:
    """h in [0, 1] per site from DK labels ('lh.insula' or 'insula'), and the labels
    that were not in the table (they get 0.5).  returned, not warned: a caller that
    sees a long unknown list is on the wrong atlas, and should be told so."""
    h, unknown = [], []
    for n in names:
        g = str(n).split(".", 1)[-1]
        if g in HIERARCHY:
            h.append(HIERARCHY[g])
        else:
            h.append(0.5)
            unknown.append(str(n))
    return torch.tensor(h, dtype=torch.float32), sorted(set(unknown))


# --------------------------------------------------------------------------------------
# the declared priors, each a function of h: value(h) = lo + (hi - lo) * h.
# --------------------------------------------------------------------------------------
@dataclass
class Priors:
    """each (lo, hi) pair is the value at h = 0 (primary) and h = 1 (transmodal).

    the numbers are declared, with their reasons; none is fitted.  they are chosen so
    that at h = 0 a column is MONOSTABLE and input-following and at h = 1 it is
    BISTABLE and self-sustaining, which is the qualitative content of the hierarchy
    and is checked by `scripts/gate_substrate_v2.py` rather than asserted.
    """
    # membrane/population time constant: Murray 2014's intrinsic timescales run from
    # ~50-80 ms in sensory cortex to ~200-350 ms in prefrontal; the POPULATION time
    # constant here is the fast part of that, the slow part is adaptation below.
    tau_E: tuple = (0.010, 0.020)
    # Inhibition's time constant, PER SITE.  It was a single global scalar until 18 Sep
    # 2026, which made it the last homogeneous thing in a model whose whole argument against
    # v1 was homogeneity -- and it is the constant that sets how fast the E/I loop can ring,
    # so a sheet with one tau_I can have only one gamma.  Fast-spiking interneuron kinetics
    # and GABA-A decay are both faster in primary sensory cortex than in association cortex,
    # which is the same direction as every other prior here.
    tau_I: tuple = (0.003, 0.008)
    # adaptation: the slow variable that releases an attractor.  0.3 s at the sensory
    # end is STATE.md 4.10's value (puts the slow oscillation in band); 1.2 s at the
    # transmodal end lets an association up-state outlast a working-memory delay.
    tau_a: tuple = (0.30, 1.20)
    g_a: tuple = (0.9, 0.30)             # adaptation strength, falls up the hierarchy
    # synaptic depression recovery (Tsodyks & Markram 1997: ~0.5-1 s)
    tau_rec: tuple = (0.45, 0.80)
    U: float = 0.05                      # utilisation per unit rate
    # resource use per unit rate.  U = 0.2, tau_use = 0.05 (the first draft) drained an
    # up state's resource to x ~ 0.2 within ~0.1 s, and NO column held an up state at
    # all -- depression, not adaptation, was setting every dwell time.  measured on
    # isolated columns before choosing (see the dwell table in the class docstring).
    tau_use: float = 0.10
    # SYNAPTIC FACILITATION, off by default.  Declared 18 Sep 2026 after the
    # operating-point search returned 0 of 18: no prior gives this model class both a
    # graded firing range and an attractor landscape, because the recurrent gain that holds
    # a pattern is the gain that runs away when drive rises (docs/LOG.md).  The escape is
    # to stop asking the RATES to hold the memory.  With facilitation the released
    # transmitter is u*x*E rather than x*E, and `u` -- which rises with use and decays over
    # a second or more -- is a memory that persists while the rates return to baseline
    # (Mongillo, Barak & Tsodyks 2008).
    #
    # OFF by default because every gate result on record was measured without it, and a
    # default that silently changes the model would invalidate them without saying so.  It
    # is switched on by a declared prior and judged by its own pre-registered search.
    facilitation: bool = False
    U_f: float = 0.15                    # utilisation increment per unit rate
    tau_f: float = 1.50                  # facilitation decay: the memory's timescale
    u_max: float = 1.0
    # recurrent excitation.  beta * w_EE / 4 = 1 is the rule of thumb for an isolated
    # E population, but the E -> I -> E loop LINEARISES a column (STATE.md found the
    # same thing in v1) and the first draft, (0.40, 1.30) with w_EI = 0.70, had NO
    # bistable column anywhere on the hierarchy.  (0.40, 2.50) with w_EI = 0.50 was
    # chosen from four candidates on isolated columns; what it produces is tabulated
    # in the class docstring and re-measured by the gate.
    w_EE: tuple = (0.40, 2.50)
    beta_E: float = 8.0
    theta_E: tuple = (0.30, 0.60)        # higher threshold where recurrence is stronger
    # INHIBITION-STABILISED, and these three were measured into place on 18 Sep 2026.
    # With the first values (w_EI 0.50, w_IE 0.90, beta_I 6) the SHEET had no operating point
    # between silent and saturated: swept over tonic drive it jumped from 4.60 Hz with zero
    # saturation to 23.95 Hz with 19% saturated between drives of 0.064 and 0.066, and no
    # drive produced anything in between.  Cortex lives in that middle.  Stronger, steeper
    # inhibition stabilises the recurrent excitation instead of letting it ignite: the sheet
    # now answers drive smoothly from 5.16 to 17.03 Hz with saturation under 1.5% across a
    # fourfold range of drive.
    #
    # The cost is measured too, and it is real: raising w_EI erodes the isolated column's
    # bistability, which is the timescale hierarchy this model exists for.  At w_EI 1.2 only
    # h = 0.90 is still bistable; at 0.9 the bistable range is h >= 0.75, against h >= 0.50
    # with the old values.  The association end keeps its attractors and the middle of the
    # hierarchy loses them.  That trade is the declaration; docs/LOG.md holds both sweeps.
    # NOT ADOPTED, and the measurement is why.  The inhibition-stabilised values
    # (w_EI 0.90, w_IE 1.40, beta_I 12) give the sheet the operating range it lacks -- a
    # smooth 5.16 to 17.03 Hz across a fourfold drive range with under 1.5% saturation,
    # against the declared values' jump from 4.60 Hz to 23.95 Hz with 19% saturated between
    # two adjacent drives.  Run against the gates, they cost the substrate its reason for
    # existing: G2 invariant sets fell from 8 attractors to **1**, and G3 metastability went
    # to **zero transitions, one macrostate, and no region ever entering an up state at all**.
    # G0, G1, G4 and G5 still passed.
    #
    # An attractor landscape is what DYNAMICS.md says a percept or an intention IS, so a
    # graded firing range bought with it is not a trade worth taking blind.  The declared
    # values stay.  The open question -- whether some prior has BOTH, judged on G2, G3 and
    # the graded-regime criterion together rather than one after the other -- is in
    # docs/LOG.md, and it needs one search, not a sequence of retries.
    w_EI: float = 0.50                   # I -> E
    w_IE: float = 0.90                   # E -> I
    w_II: float = 0.30                   # I -> I
    beta_I: float = 6.0
    theta_I: float = 0.35
    # long range.  G_L on E targets, G_F on I targets (feedforward inhibition); both
    # multiply a row-normalised kernel, so they are the gains of a WHOLE row.
    G_L: float = 0.45
    G_F: float = 0.35
    # background input: an Ornstein-Uhlenbeck current on u_E with this STATIONARY std
    # and correlation time -- synaptic bombardment, not white noise on a rate.  an OU
    # process is advanced exactly, so sigma means the same thing at every dt.
    sigma: float = 0.04
    tau_eta: float = 0.005
    # how far a learned residual may move a prior, as a fraction of its own value.
    # bounded so that training cannot silently push the whole cortex below the
    # bistability threshold and turn the substrate back into v1.
    residual_frac: float = 0.35

    def at(self, name: str, h: torch.Tensor) -> torch.Tensor:
        v = getattr(self, name)
        if isinstance(v, tuple):
            return v[0] + (v[1] - v[0]) * h
        return torch.full_like(h, float(v))


HETERO = ("tau_E", "tau_I", "tau_a", "g_a", "tau_rec", "w_EE", "theta_E")


class CorticalField(nn.Module):
    """the v2 substrate on a fixed graph.

    what the declared priors do to an ISOLATED column -- kicked up by a 200 ms pulse,
    then left alone, no noise -- measured before the priors were chosen (four
    candidates; this is the one kept).  B = bistable fast subsystem (adaptation and
    resource frozen), m = monostable; dwell = how long the up state holds:

        h      0.00  0.25  0.50   0.60   0.75   0.80   0.90
        fast   m     m     B      B      B      B      B
        dwell  --    --    0.06s  0.24s  0.69s  0.94s  2.08s

    primary cortex follows its input and holds nothing; association cortex holds an
    up state for a time that rises monotonically along the hierarchy, and NOTHING holds
    forever -- every attractor is eventually released by its own slow variables.  that
    is quasi-multistability with a timescale hierarchy, and it is the prior, before any
    training.  (Murray et al. 2014 measure the same ORDERING of intrinsic timescales;
    the absolute range here is longer at the top because a working-memory up state is
    measured in seconds.)

    pos (N,3) mm, idx (N,k) partner indices, dist (N,k) mm, region names (N) -- the
    same graph objects v1 builds, so `from_v1(dyn)` can put this model on EXACTLY the
    wiring a v1 checkpoint was trained on, which is what an ablation needs.
    """

    def __init__(self, pos: torch.Tensor, idx: torch.Tensor, dist: torch.Tensor,
                 regions, embed_dim: int = 64, length_scale_mm: float = 40.0,
                 n_far: int = 0, delay_s: torch.Tensor | None = None,
                 priors: Priors | None = None, learn_hetero: bool = True,
                 seed: int = 0):
        super().__init__()
        self.pr = priors or Priors()
        n, k = idx.shape
        self.n, self.k, self.n_far = int(n), int(k), int(n_far)
        dev = idx.device
        self.register_buffer("pos", pos.float())
        self.register_buffer("idx", idx.long())
        h, self.unknown_regions = hierarchy_of(regions)
        self.register_buffer("h", h.to(dev))
        # the atlas label per site, kept so an analysis can ask for region means without
        # re-deriving labels from positions (and possibly from a different atlas)
        self.region_names = [str(r) for r in regions]
        uniq = sorted(set(self.region_names))
        self.region_list = uniq
        self.register_buffer("region_id", torch.tensor(
            [uniq.index(r) for r in self.region_names], device=dev))
        # geometric prior on the local edges, flat on the long-range ones -- v1's
        # division of labour, kept for the same measured reason (its comment on
        # `long_range`: a distance-decayed long-range prior makes the edges inert).
        geo = torch.exp(-dist.float() / length_scale_mm)
        if n_far:
            n_loc = k - n_far
            geo[:, n_loc:] = torch.exp(-dist[:, :n_loc].float().median() / length_scale_mm)
        geo = geo / geo.sum(1, keepdim=True)          # row-normalised: G_L, G_F are row gains
        self.register_buffer("geo", geo)
        if delay_s is not None:
            self.register_buffer("delay_s", delay_s.float())
        else:
            self.delay_s = None
        self._lag_cache: dict = {}

        # the learned pair factor.  seeded from a DEDICATED generator: CLAUDE.md's
        # "shared generator" trap -- an init drawn from the global RNG made v1's
        # topology a function of the caller's weight seed.
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.embed = nn.Parameter(torch.randn(n, embed_dim, generator=g).to(dev) * 0.02)
        self.pair_slope = nn.Parameter(torch.tensor(3.0))
        # EXPERIENCE.  a per-edge term added to the pair drive, written by the covariance
        # rule in `plasticity_*` below.  a persistent buffer, not a parameter: gradient
        # descent does not touch it, experience does -- and like every other thing that
        # selects or weights an edge it is saved with the model (CLAUDE.md: `read_idx`).
        self.register_buffer("P", torch.zeros(n, k, device=dev))

        # heterogeneous parameters: declared prior(h) x (1 + bounded residual).
        # the residual passes through tanh, so |change| <= residual_frac of the prior
        # whatever the optimiser does -- the landscape cannot be trained away.
        self.learn_hetero = learn_hetero
        for name in HETERO:
            self.register_buffer(f"prior_{name}", self.pr.at(name, self.h))
            self.register_parameter(f"res_{name}", nn.Parameter(
                torch.zeros(n, device=dev), requires_grad=learn_hetero))

    # ------------------------------------------------------------------ parameters
    def site(self, name: str) -> torch.Tensor:
        p = getattr(self, f"prior_{name}")
        r = getattr(self, f"res_{name}")
        return p * (1.0 + self.pr.residual_frac * torch.tanh(r))

    @classmethod
    def from_v1(cls, dyn, **kw):
        """the v2 model on v1's graph: same positions, same partners, same delays."""
        import ibm.cortical_sheet as CS
        reg = CS.regions_at(dyn.pos.detach().cpu().numpy())
        names = [CS.REGIONS[i] for i in reg]
        dist = (dyn.pos[dyn.idx] - dyn.pos[:, None, :]).norm(dim=-1)
        return cls(dyn.pos, dyn.idx, dist, names, n_far=getattr(dyn, "n_far", 0),
                   delay_s=getattr(dyn, "delay_s", None) if getattr(dyn, "tract_delays", False) else None,
                   **kw)

    def edge_weights(self):
        """(W+, W-): E-target and I-target weights, each (N, k), each >= 0.

        both come from ONE learned similarity per pair, pushed in opposite
        directions: similar columns excite each other, dissimilar ones reach each
        other's interneurons.  computed once per rollout -- v1's comment on
        `edge_weights` records what rebuilding it per step cost.
        """
        e = F.normalize(self.embed, dim=-1)
        sim = (e.unsqueeze(1) * e[self.idx]).sum(-1)
        s = self.pair_slope * sim + self.P
        wp = self.geo * torch.sigmoid(s) * 2.0      # x2: at sim = 0 the row gain is 1
        wm = self.geo * torch.sigmoid(-s) * 2.0
        return wp, wm

    # ------------------------------------------------------------------ state
    def init_state(self, b: int, device=None, generator: torch.Generator | None = None,
                   random: bool = False):
        """rest (all down), or a uniformly random point in the box when `random`.

        a random start takes the generator it is GIVEN and never draws its own --
        the gate compares two runs from the same starts, and CLAUDE.md's rule is
        that a callee handed a shared draw must not re-draw it.
        """
        device = device or self.idx.device
        z = torch.zeros(b, self.n, device=device)
        if random:
            assert generator is not None, "a random start needs an explicit generator"
            E = torch.rand(b, self.n, generator=generator, device="cpu").to(device)
            x = torch.rand(b, self.n, generator=generator, device="cpu").to(device)
            return {"E": E, "I": z.clone(), "a": z.clone(), "x": x,
                    "u": z.clone() + self.pr.U_f, "eta": z.clone(),
                    "hist": None, "ptr": 0}
        return {"E": z, "I": z.clone(), "a": z.clone(), "x": torch.ones_like(z),
                "u": z.clone() + self.pr.U_f, "eta": z.clone(), "hist": None, "ptr": 0}

    @staticmethod
    def detach(state):
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in state.items()}

    # ------------------------------------------------------------------ dynamics
    def _lags(self, dt: float):
        key = round(float(dt), 12)
        got = self._lag_cache.get(key)
        if got is None:
            lag = torch.round(self.delay_s / float(dt)).long().clamp_min(0)
            got = (lag, int(lag.max().item()) + 1)
            self._lag_cache[key] = got
        return got

    def _long(self, src, W, hist, ptr, dt):
        """sum_j W_ij src_j(t - d_ij).  local columns undelayed (a u-fibre over a 4 mm
        k-NN edge is ~0.5 ms, below any dt used here); long-range columns delayed
        when delays are carried and `dt` resolves them."""
        if hist is None or self.delay_s is None or not self.n_far:
            return (src[:, self.idx] * W).sum(-1)
        n_loc = self.k - self.n_far
        out = (src[:, self.idx[:, :n_loc]] * W[:, :n_loc]).sum(-1)
        lag, depth = self._lags(dt)
        slot = torch.remainder(ptr - lag, depth)
        flat = (slot * self.n + self.idx[:, n_loc:]).reshape(-1)
        far = hist.reshape(hist.shape[0], depth * self.n)[:, flat].view(src.shape[0], self.n, self.n_far)
        return out + (far * W[:, n_loc:]).sum(-1)

    def step(self, state, drive, dt: float, W=None, drive_I=None,
             noise: torch.Tensor | None = None, m_beta: float = 1.0, m_sigma: float = 1.0):
        """one exponential-Euler step.  `noise` is a standard-normal (B, N) draw
        supplied by the caller (or None for a noiseless step) -- never drawn here, so
        that two passes can be made to see the same noise."""
        pr = self.pr
        W = W if W is not None else self.edge_weights()
        wp, wm = W
        E, I, a, x = state["E"], state["I"], state["a"], state["x"]
        hist, ptr = state.get("hist"), state.get("ptr", 0)
        if self.delay_s is not None and self.n_far:
            _, depth = self._lags(dt)
            if hist is None:
                hist = torch.zeros(E.shape[0], depth, self.n, device=E.device, dtype=E.dtype)
            assert hist.shape[1] == depth, "dt changed inside a trajectory"
            hist = hist.clone() if torch.is_grad_enabled() else hist
            hist[:, ptr] = E * x            # what arrives is the RELEASED transmitter
        u = state.get("u")
        if u is None:
            u = torch.full_like(E, pr.U_f)
        # what a spike releases: the resource present times the fraction used.  Without
        # facilitation `u` is the constant U_f and this is the old `E * x` up to that
        # scale; with it, `u` carries the trace of recent use.
        src = E * x * (u / pr.U_f if pr.facilitation else 1.0)
        tau_E, tau_I = self.site("tau_E"), self.site("tau_I")
        tau_a, g_a = self.site("tau_a"), self.site("g_a")
        tau_rec, w_EE, th_E = self.site("tau_rec"), self.site("w_EE"), self.site("theta_E")

        # the background current, advanced EXACTLY: an OU process with stationary std
        # sigma*m_sigma.  with no draw supplied it only decays, so a noiseless run is
        # deterministic and a noisy one is reproducible from the generator it was given.
        eta = state.get("eta")
        if eta is None:
            eta = torch.zeros_like(E)
        rho = math.exp(-dt / pr.tau_eta)
        eta = eta * rho
        if noise is not None:
            eta = eta + pr.sigma * m_sigma * math.sqrt(1.0 - rho * rho) * noise
        u_E = (w_EE * x * E + pr.G_L * self._long(src, wp, hist, ptr, dt)
               - pr.w_EI * I - a + drive + eta)
        u_I = (pr.w_IE * E + pr.G_F * self._long(src, wm, hist, ptr, dt) - pr.w_II * I
               + (drive_I if drive_I is not None else 0.0))
        fE = torch.sigmoid(pr.beta_E * m_beta * (u_E - th_E))
        fI = torch.sigmoid(pr.beta_I * (u_I - pr.theta_I))

        cE = 1.0 - torch.exp(-dt / tau_E)
        cI = 1.0 - torch.exp(-dt / tau_I)
        ca = 1.0 - torch.exp(-dt / tau_a)
        E2 = E + cE * (fE - E)
        I2 = I + cI * (fI - I)
        a2 = a + ca * (g_a * E - a)
        # facilitation, exactly integrated with E held: du/dt = (U_f - u)/tau_f + U_f(1-u)E.
        # Both terms are linear in u, so the step is again a convex combination toward a
        # target inside [0, 1] -- boundedness is preserved for any dt, as everywhere else
        # in this file.
        if pr.facilitation:
            # E is a NORMALISED rate in [0, 1]; the facilitation rate constant is per SPIKE,
            # so the drive term needs spikes per second -- E * R_MAX.  Written with E
            # directly (the first version) the increment was U_f*(1-u)*E ~ 0.06/s against a
            # decay of 1/tau_f = 0.67/s, so `u` moved by 8% during a cue and 0.8% two
            # seconds later: a memory variable that cannot remember, for the same reason a
            # constant is not portable across a change of units.
            r_hz = E * R_MAX
            ku = 1.0 / pr.tau_f + pr.U_f * r_hz
            u_inf = (pr.U_f / pr.tau_f + pr.U_f * r_hz) / ku
            u2 = u_inf + (u - u_inf) * torch.exp(-dt * ku)
            u2 = u2.clamp(0.0, pr.u_max)
            use = u2                                   # depression is driven by what is used
        else:
            u2 = u
            use = torch.full_like(E, pr.U)
        # depression, exactly integrated over the step with E held:
        #   dx/dt = (1-x)/tau_rec - (use E / tau_use) x
        kx = 1.0 / tau_rec + use * E / pr.tau_use
        x_inf = (1.0 / tau_rec) / kx
        x2 = x_inf + (x - x_inf) * torch.exp(-dt * kx)
        out = {"E": E2, "I": I2, "a": a2, "x": x2, "u": u2, "eta": eta, "hist": hist,
               "ptr": (ptr + 1) % hist.shape[1] if hist is not None else 0}
        return out

    def rollout(self, drive, state, dt: float, substeps: int = 1, noise_gen=None,
                drive_I=None, m_beta: float = 1.0, m_sigma: float = 1.0, record: str = "E"):
        """drive (B, T, N): one row per frame, held for `substeps` steps of `dt`.

        returns (trace (B, T, N) of `record` at each frame's END, final state).  the
        state is carried IN and OUT -- the caller decides where to detach.  noise, when
        wanted, comes from the generator passed in; none is drawn otherwise.
        """
        W = self.edge_weights()
        B, T, N = drive.shape
        trace = []
        for t in range(T):
            d = drive[:, t]
            dI = drive_I[:, t] if drive_I is not None else None
            for _ in range(substeps):
                z = None
                if noise_gen is not None:
                    z = torch.randn(B, N, generator=noise_gen, device="cpu").to(d.device)
                state = self.step(state, d, dt, W=W, drive_I=dI, noise=z,
                                  m_beta=m_beta, m_sigma=m_sigma)
            trace.append(state[record])
        return torch.stack(trace, 1), state

    def rate_hz(self, state):
        return state["E"] * R_MAX

    # ------------------------------------------------------------------ plasticity
    #
    # WHY.  G3 failed and its diagnostics (docs/LOG.md 2026-09-18) found every region
    # metastable but switching INDEPENDENTLY, on random and connectome wiring alike: at
    # initialisation W+ ~ W- on every edge, so a partner being up excites and inhibits in
    # about equal measure.  a connectome says who is connected; an assembly needs a SIGN
    # pattern, and that is written by experience.
    #
    # THE RULE: covariance Hebbian with decay, on the pair drive of each edge,
    #
    #     dP_ij/dt = eta * ( <(E_i - Ebar_i)(E_j - Ebar_j)> / var_ref  -  lam * P_ij )
    #
    # co-active pairs move toward W+ (mutual excitation), anti-active pairs toward W-
    # (feedforward inhibition).  COVARIANCE, not plain Hebb: plain Hebb on rates that are
    # never negative can only grow, and a rule that can only grow saturates every edge --
    # the same flat kernel again, one sign higher.  centring on a slow running mean Ebar
    # makes independent sites produce zero drift in expectation, which is the known answer
    # the plasticity experiment's control is built on.  |P| is capped at P_MAX so no edge
    # can leave sigmoid's useful range.
    P_MAX = 4.0

    def plasticity_init(self, b: int, device=None):
        device = device or self.idx.device
        return {"Ebar": torch.zeros(b, self.n, device=device), "acc": torch.zeros(self.n, self.k, device=device),
                "count": 0}

    @torch.no_grad()
    def plasticity_accumulate(self, pst, state, dt: float, tau_bar: float = 10.0,
                              competitive: bool = False):
        E = state["E"]
        rho = math.exp(-dt / tau_bar)
        pst["Ebar"] = pst["Ebar"] * rho + (1.0 - rho) * E
        dE = E - pst["Ebar"]
        if competitive:
            # remove the COMMON MODE: the fluctuation every site shares at this instant.
            # the first (non-competitive) run learned exactly that -- a sheet whose rate
            # rises and falls as one makes every pair's covariance positive whatever the
            # input is (docs/LOG.md 2026-09-18, the plasticity RESULT).
            dE = dE - dE.mean(1, keepdim=True)
        # batch-mean product over edges: (B, N, 1) x (B, N, k) -> (N, k)
        pst["acc"] += (dE.unsqueeze(-1) * dE[:, self.idx]).mean(0)
        pst["count"] += 1

    @torch.no_grad()
    def plasticity_apply(self, pst, dt: float, eta: float, lam: float, var_ref: float = 0.01,
                         competitive: bool = False):
        """apply the accumulated covariance as one update over `count` steps, then reset.

        `competitive` makes the update ROW-ZERO-SUM, separately over a site's local and
        long-range edges: a site's total incoming drive is conserved (subtractive synaptic
        scaling, DYNAMICS.md mechanism 11), so learning can only REDISTRIBUTE weight among
        a site's partners, never inflate all of it -- which is what drove the first run
        from 15 Hz to 46 Hz."""
        if pst["count"] == 0:
            return
        C = pst["acc"] / pst["count"]
        T = pst["count"] * dt
        dP = eta * T * (C / var_ref - lam * self.P)
        if competitive:
            n_loc = self.k - self.n_far
            dP[:, :n_loc] -= dP[:, :n_loc].mean(1, keepdim=True)
            if self.n_far:
                dP[:, n_loc:] -= dP[:, n_loc:].mean(1, keepdim=True)
        self.P += dP
        self.P.clamp_(-self.P_MAX, self.P_MAX)
        pst["acc"].zero_(); pst["count"] = 0

    # ------------------------------------------------------------------ analysis
    @torch.no_grad()
    def column_bistable(self) -> torch.Tensor:
        """per site: is an ISOLATED column (no coupling, no adaptation, full resource)
        bistable at its current parameters?  solved numerically, not by the
        beta*w/4 rule of thumb: count the fixed points of E = F(w_EE E - w_EI I*(E) - theta)
        with I at its own fixed point, on a fine grid of E."""
        pr = self.pr
        Eg = torch.linspace(0, 1, 2001, device=self.h.device)[None, :]
        w = self.site("w_EE")[:, None]
        th = self.site("theta_E")[:, None]
        # I*(E): solve I = sigmoid(beta_I (w_IE E - w_II I - theta_I)) by fixed-point iteration
        I = torch.zeros_like(Eg).expand(self.n, -1).clone()
        for _ in range(60):
            I = torch.sigmoid(pr.beta_I * (pr.w_IE * Eg - pr.w_II * I - pr.theta_I))
        g = torch.sigmoid(pr.beta_E * (w * Eg - pr.w_EI * I - th)) - Eg
        crossings = ((g[:, 1:] * g[:, :-1]) < 0).sum(1)
        return crossings >= 3          # down, unstable middle, up


def build_sheet(n: int = 1024, k: int = 32, long_frac: float = 0.25, seed: int = 0,
                device="cpu", long_topology: str = "random", tract_threshold: float = 0.5,
                tract_delays: bool = True, **kw) -> CorticalField:
    """a v2 field on a fresh fsaverage sheet: n area-weighted white-surface sites,
    k-1-n_far nearest neighbours plus n_far uniform long-range partners.

    every draw is a pure function of `seed` and runs on the CPU, so the graph is the
    same on every device -- the property v1's device-generator draw lacked (CLAUDE.md,
    "a shared generator makes two things vary").
    """
    import numpy as np
    import ibm.cortical_sheet as CS
    xyz, reg = CS.sample_sites(n, seed=seed)
    pos = torch.from_numpy(xyz)
    n_far = int(round(k * long_frac))
    d = torch.cdist(pos, pos)
    d.fill_diagonal_(float("inf"))
    loc = d.topk(k - n_far, largest=False).indices
    delay_s = None
    if n_far and long_topology == "tract":
        # the long-range partners from the HCP group connectome (braingraph, 1064
        # subjects), as v1's `long_topology="tract"` draws them: a site in parcel a
        # draws among the sites of the parcels the consensus joins a to, with the
        # parcel pair's conduction delay.  G3's diagnostic (docs/LOG.md 2026-09-18)
        # found regions metastable but switching INDEPENDENTLY on random wiring; the
        # connectome is the first thing that could couple them.
        import ibm.cortical_tracts as CT
        far_np, dly, _len, _note = CT.draw_partners(np.asarray(reg), n_far, seed=seed,
                                                     threshold=tract_threshold)
        idx = torch.cat([loc, torch.from_numpy(far_np)], 1)
        if tract_delays:
            delay_s = torch.cat([torch.zeros(n, k - n_far), torch.from_numpy(dly)], 1)[:, k - n_far:]
    elif n_far and long_topology != "random":
        raise ValueError(f"long_topology must be 'random' or 'tract', not {long_topology!r}")
    elif n_far:
        rng = np.random.default_rng(seed + 7919)
        far = torch.from_numpy(rng.integers(0, n, size=(n, n_far)))
        # a self-edge or a duplicate of a local partner is re-drawn, not kept: either
        # would quietly add local recurrence to what is supposed to be association
        for i in range(n):
            bad = (far[i] == i) | torch.isin(far[i], loc[i])
            while bad.any():
                far[i, bad] = torch.from_numpy(rng.integers(0, n, size=int(bad.sum())))
                bad = (far[i] == i) | torch.isin(far[i], loc[i])
        idx = torch.cat([loc, far], 1)
    else:
        idx = loc if not (n_far and long_topology == "tract") else idx
    dist = (pos[idx] - pos[:, None, :]).norm(dim=-1)
    names = [CS.REGIONS[i] for i in reg]
    return CorticalField(pos.to(device), idx.to(device), dist.to(device), names,
                         n_far=n_far, seed=seed,
                         delay_s=delay_s.to(device) if delay_s is not None else None,
                         **kw).to(device)
