"""reusable learned building blocks: the f a Form.LEARNED implementation uses.

nothing here is a process, a component or a declaration.  these are the pieces a
learned implementation's f is assembled from, and they live in the processes
package because the choice of piece is a modelling choice about what structure
the dynamics have, not an engineering choice about how to compute them.

three commitments shape everything below.

*framework-agnostic.*  each block is a frozen dataclass describing the module's
shape, plus a numpy reference implementation of its forward pass.  no torch, no
jax, no autodiff.  the dataclass is the declaration -- it is what a materialization
records, what a prior is placed over, and what a fitted checkpoint is checked
against -- and the numpy path exists so that the reference semantics are
executable and unambiguous rather than described in prose.  a real training run
will reimplement the forward pass in whatever framework it uses; if it disagrees
with the numpy here, the numpy is right.

*every block reports its parameter count as a function of tying policy.*  this is
the reason the file is organized around `param_count` rather than around layers.
`Tying` (ibm.vocabulary) determines how many *effective* parameters exist, and
therefore what data could possibly move them off their prior.  a per-site
message-passing weight over 10^5 positions is a perfectly legitimate
declaration -- the architecture says so explicitly -- but it is a declaration that
10^5 numbers exist which almost no dataset can distinguish, so the posterior
stays near the prior and the fitted structure comes out smooth.  that is the
intended behaviour and not a failure, and the only way to reason about it is to
be able to count.

*topology carries geometry; the learned part carries content.*  this is the
architecture's own division and it is enforced structurally in the first block
below rather than left as a convention.  a topology knows distance, tract length,
orientation and contact area -- descriptive facts about which state variables may
interact.  how strongly a particular pair actually interacts is the process's
business, and factorizing an edge weight into a geometric term the topology
supplies and a learned term over content embeddings is what keeps the two from
being confounded during fitting.

a note on what these blocks are *for*, because it is easy to reach for a learned
module reflexively.  every one of them is designed to be a **correction** on an
analytic form, not a replacement for one.  `ResidualOverPhysics` makes that
explicit and the others assume it: their priors are centred so that a
data-starved posterior degrades to the physics rather than to noise.  a learned f
that replaces an analytic one throws away the only thing in the model that
generalizes outside the training distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ibm.vocabulary import Tying

_EPS = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    """numerically stable logistic.

    written out rather than imported because the naive form overflows for large
    negative x, and an edge weight that silently becomes nan during fitting is
    an unpleasant thing to debug.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), _EPS)


# ---------------------------------------------------------------------------
# 1. distance-modulated message passing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DistanceMessagePassing:
    """message passing whose edge weight is geometry times learned content.

        w_ij = exp(-d_ij / l) * sigmoid(<e_i, e_j> / sqrt(dim))

    the architecture's own example, and the block that most directly expresses
    what a topology is for.  the first factor is supplied by the topology: it is
    a fact about the tissue -- a distance along a surface, a tract length, a
    contact area -- and it is not fitted, or rather only its one length scale is.
    the second factor is learned, over per-position embeddings, and carries
    everything about *what* is at those two positions that makes them couple
    strongly or weakly.

    the reason to factorize rather than to learn a free weight per edge is
    identifiability, and it is worth being concrete.  a free per-edge weight over
    a cortical surface topology is 10^7 or more parameters, essentially none of
    which any dataset distinguishes; what a fit does with them is memorize the
    training set's noise and produce a connectivity map that looks structured and
    is not.  the factorized form has a few thousand, and the geometric factor
    means that with no data at all the block reduces to a distance kernel --
    which is a defensible prior about cortical connectivity rather than an
    arbitrary one.

    the exponential decay is not arbitrary either: lateral cortical connection
    probability falls close to exponentially with distance over the range where
    it has been measured, with a length constant of a few millimetres, and the
    prior on `length_scale_mm` should reflect that rather than being flat.

    where this is the wrong block: anywhere the coupling is genuinely
    long-range and content-specific in a way distance does not predict, which is
    to say for tract-mediated connectivity.  there the geometric factor should be
    tract conduction delay and existence rather than euclidean distance, and the
    topology supplies a different edge feature -- but the factorization is the
    same and that is the point of writing the block over a supplied edge feature
    rather than over coordinates.
    """

    #: input and output channels per position.  usually the number of spectral
    #: components a band-limited coupling retains, or a small latent width.
    channels_in: int
    channels_out: int
    #: dimensionality of the content embedding whose inner product sets the
    #: learned factor.  small on purpose: this is a similarity, not a feature bank.
    embed_dim: int = 8
    #: which topology edge feature supplies the geometric factor.  named rather
    #: than assumed, because the same block over `distance_mm` and over
    #: `tract_length_mm` are different models.
    geometric_feature: str = "distance_mm"
    #: initial length constant of the geometric decay.
    length_scale_mm: float = 3.0
    #: whether the length scale is fitted.  usually yes, and it is one parameter.
    learn_length_scale: bool = True
    #: how embeddings are shared across positions.
    tying: Tying = Tying.EMBEDDING
    symmetric: bool = False

    def param_count(self, n_sites: int, n_partitions: int = 1,
                    n_edges: int = 0) -> int:
        """free parameters, as a function of the tying policy.

        the four regimes differ by orders of magnitude and the difference is the
        whole reason `Tying` exists as a declared property rather than as a
        training detail:

        - ``GLOBAL``: no embeddings at all.  the content factor collapses to one
          learned scalar shared by every edge, so the block is a distance kernel
          with a gain.  ``channels_in * channels_out + 2``.  identifiable from
          almost any dataset.
        - ``PER_PARTITION``: one embedding per anatomical partition, positions
          inheriting theirs by soft membership.  ``+ n_partitions * embed_dim``.
          with a few hundred cortical areas this is thousands of parameters and
          is what a multi-subject dataset can actually constrain.
        - ``EMBEDDING``: the same table, factorized -- partitions map into a
          low-rank basis which maps to the embedding -- so several partitions
          share structure.  the same count as PER_PARTITION here because the
          factorization's rank is `embed_dim`; the difference is in the *prior*,
          which is smooth across partitions rather than independent.
        - ``PER_SITE``: one embedding per materialized position.
          ``+ n_sites * embed_dim``.  at 10^5 positions and dim 8 that is 8*10^5
          numbers, which no available dataset distinguishes.  legitimate, and the
          posterior will stay near the prior -- which is exactly the behaviour the
          architecture describes and not a bug.

        `n_edges` is unused by the factorized forms and is accepted so that the
        signature can also price the *unfactorized* alternative for comparison:
        a free weight per edge is `n_edges` parameters, and quoting that number
        next to these is usually the fastest way to end an argument about whether
        the factorization is worth it.
        """
        base = self.channels_in * self.channels_out
        base += 1 if self.learn_length_scale else 0
        if self.tying is Tying.GLOBAL:
            return base + 1
        if self.tying in (Tying.PER_PARTITION, Tying.EMBEDDING):
            return base + n_partitions * self.embed_dim
        return base + n_sites * self.embed_dim

    def edge_weights(self, geometric: np.ndarray, embed_src: np.ndarray,
                     embed_dst: np.ndarray) -> np.ndarray:
        """w_ij for a batch of edges.

        `geometric` is the topology's edge feature in millimetres; the embeddings
        are gathered per edge by the caller, because gathering is where a real
        implementation's memory layout decisions live and pretending otherwise in
        a reference implementation is misleading.
        """
        g = np.exp(-np.asarray(geometric, float) / max(self.length_scale_mm, _EPS))
        s = np.sum(np.asarray(embed_src, float) * np.asarray(embed_dst, float), axis=-1)
        return g * sigmoid(s / np.sqrt(max(self.embed_dim, 1)))

    def apply(self, x: np.ndarray, edges: tuple[np.ndarray, np.ndarray],
              geometric: np.ndarray, embeddings: np.ndarray,
              weight: np.ndarray) -> np.ndarray:
        """one round of message passing.  ``x`` is (n_sites, channels_in``).

        the accumulation is a scatter-add, which is the operation that makes this
        cheap: the cost is linear in the number of edges rather than quadratic in
        positions, which is why a topology is a sparse incidence structure and
        not a dense matrix.
        """
        src, dst = edges
        w = self.edge_weights(geometric, embeddings[src], embeddings[dst])
        msg = (x[src] @ np.asarray(weight, float)) * w[:, None]
        out = np.zeros((x.shape[0], self.channels_out), dtype=float)
        np.add.at(out, dst, msg)
        if self.symmetric:
            np.add.at(out, src, (x[dst] @ np.asarray(weight, float)) * w[:, None])
        return out


# ---------------------------------------------------------------------------
# 2. per-partition embedding table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionEmbedding:
    """a parameter table indexed by anatomical partition, read through soft
    membership.

    this is `Tying.PER_PARTITION` and `Tying.EMBEDDING` made into an object, and
    it is the most useful block in the file because almost every learned
    implementation in the inventory needs exactly this and nothing more: a
    per-region gain, a per-region time constant, a per-region correction.

    the softness is the design decision worth defending.  ARCHITECTURE.md §2 says
    a partitioning system maps position to *memberships* in [0,1], not to a
    label, and that a partition boundary is a gradient rather than a wall.  so
    the lookup here is a weighted sum over the table rather than an index, and a
    position on the boundary between two areas gets a blend of their parameters.
    that has a real consequence during fitting: gradients flow to both areas
    whose boundary a site straddles, so a mislocalized boundary degrades
    gracefully instead of assigning a site's evidence entirely to the wrong
    region.

    the difference between the two tying policies it implements is a difference
    of prior, not of shape.  PER_PARTITION gives each partition an independent
    row: with 180 cortical areas that is 180 independent parameter vectors, and
    an area with no data in the training set learns nothing.  EMBEDDING
    factorizes the table through a low-rank basis, so partitions share structure
    and evidence from one informs its neighbours in the learned basis -- which is
    what you want when the parcellation is finer than the data, which it usually
    is.

    a fitted table is also the most interpretable object a learned implementation
    can produce: it is a map, in the same coordinates as the atlas, of how one
    parameter varies across the brain, and it can be looked at.
    """

    n_partitions: int
    dim: int
    #: PER_PARTITION for an independent row per partition, EMBEDDING for the
    #: low-rank factorized table.
    tying: Tying = Tying.PER_PARTITION
    #: rank of the factorization when tying is EMBEDDING.  ignored otherwise.
    rank: int = 8
    #: prior mean of every row.  usually the analytic value the learned table is
    #: correcting, so that an untouched row reproduces the physics.
    init: float = 0.0

    def param_count(self, n_sites: int = 0) -> int:
        """free parameters.

        - ``PER_PARTITION``: ``n_partitions * dim``.  independent rows.
        - ``EMBEDDING``: ``(n_partitions + dim) * rank``, which is smaller as soon
          as ``rank < n_partitions * dim / (n_partitions + dim)`` -- for 180 areas,
          dim 16 and rank 8 that is 1568 against 2880, and the real saving is not
          the count but the smoothness the factorization imposes.
        - ``GLOBAL``: ``dim``.  one row for the whole brain, which is the right
          choice far more often than it is chosen.
        - ``PER_SITE``: ``n_sites * dim``, ignoring the partitioning entirely.

        `n_sites` is only consulted for PER_SITE.
        """
        if self.tying is Tying.GLOBAL:
            return self.dim
        if self.tying is Tying.PER_SITE:
            return n_sites * self.dim
        if self.tying is Tying.EMBEDDING:
            return (self.n_partitions + self.dim) * self.rank
        return self.n_partitions * self.dim

    def table(self, params: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """materialize the (n_partitions, dim) table from its parameters.

        for EMBEDDING the parameters arrive as a pair (C, B) with shapes
        (n_partitions, rank) and (rank, dim); the table is their product, and it
        is never stored -- which matters when the partitioning is fine.
        """
        if self.tying is Tying.EMBEDDING:
            c, b = params
            return np.asarray(c, float) @ np.asarray(b, float)
        p = np.asarray(params, float)
        if self.tying is Tying.GLOBAL:
            return np.broadcast_to(p.reshape(1, self.dim), (self.n_partitions, self.dim))
        return p.reshape(self.n_partitions, self.dim)

    def lookup(self, membership: np.ndarray,
               params: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """soft read: ``a(q) @ table``, with ``membership`` of shape (n_sites, n_partitions).

        memberships are used directly as weights and are deliberately not
        normalized or thresholded here.  a probabilistic atlas that assigns a
        voxel 0.3 to one area and 0.2 to another and nothing to the rest is
        saying something true about that voxel, and renormalizing to 1 would
        discard the fact that it is not confidently in either.
        """
        return np.asarray(membership, float) @ self.table(params)


# ---------------------------------------------------------------------------
# 3. low-rank spatio-temporal factorization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LowRankSpatioTemporal:
    """state over positions and spectral components, factorized as a sum of
    separable modes.

        X[n, k] ~ sum_r  u_r[n] * v_r[k]

    the workhorse compression for anything defined jointly over space and
    frequency, and the reason to have it as a declared block rather than as an
    implementation detail is that the rank is a *modelling* statement.  saying a
    coupling is rank 20 across 10^5 positions and 500 spectral components is
    saying that the process has twenty spatial patterns, each with its own
    spectrum, and that no other combination occurs.  that is a strong,
    falsifiable claim and it is usually roughly true: cortical activity is
    massively low-dimensional at the scales this model materializes, and the
    resting-state literature's twenty-odd reproducible components are the
    empirical version of the same statement.

    it maps directly onto the `factor` field of `SpectralGaussian`, which carries
    an optional low-rank term ``F F^H`` across frequencies -- so a fitted
    factorization here is not merely compressed storage, it is the object the
    uncertainty representation already knows how to propagate.  that is the
    reason to prefer this specific compression over any other.

    the separability assumption is where it breaks, and it breaks in a way worth
    knowing.  a rank-r sum of separable terms cannot represent a spatial pattern
    whose spectrum changes continuously across space -- a travelling wave, most
    obviously, which has a phase gradient across positions at one frequency.  a
    travelling wave needs complex-valued modes to be represented at all, which is
    why `complex_modes` defaults to true here even though it doubles the count:
    cortical travelling waves are real and a real-valued factorization turns them
    into a pair of standing modes with an arbitrary phase relationship.
    """

    n_sites: int
    n_components: int
    rank: int = 20
    #: complex spatial and spectral factors.  doubles the parameter count and is
    #: usually worth it; see the docstring on travelling waves.
    complex_modes: bool = True
    #: whether the spectral factors are shared across a partition rather than
    #: fitted per mode.  rarely useful, and offered because a materialization
    #: that has already committed to a band structure may want it.
    tying: Tying = Tying.GLOBAL

    def param_count(self, n_partitions: int = 1) -> int:
        """free parameters: ``rank * (n_sites + n_components)``, doubled if complex.

        the comparison that matters is against the dense alternative,
        ``n_sites * n_components``.  at 10^5 positions and 500 components dense
        is 5*10^7 and rank 20 is 4*10^6 real numbers complex -- an order of
        magnitude, which is the difference between a materialization that fits in
        memory and one that does not.

        under ``PER_PARTITION`` the spatial factors are replaced by per-partition
        ones, giving ``rank * (n_partitions + n_components)``: with 180 areas that
        is a few tens of thousands rather than a few million, and it is the right
        choice whenever the parcellation is believed rather than merely available.
        """
        n_spatial = n_partitions if self.tying is Tying.PER_PARTITION else self.n_sites
        c = 2 if self.complex_modes else 1
        return c * self.rank * (n_spatial + self.n_components)

    def reconstruct(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """X = U V, with U (n_spatial, rank) and V (rank, n_components)."""
        return np.asarray(u) @ np.asarray(v)

    def project(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """least-squares factorization of an observed X, by truncated svd.

        offered as the reference initializer rather than as a fitting method: the
        svd gives the optimal rank-r approximation in the frobenius norm, which is
        the right starting point and the wrong objective -- the fitted objective is
        a likelihood over state, not a reconstruction error, and the two differ
        wherever the uncertainty is not isotropic, which is everywhere.
        """
        x = np.asarray(x)
        u, s, vh = np.linalg.svd(x, full_matrices=False)
        r = min(self.rank, s.size)
        return u[:, :r] * s[:r], vh[:r]


# ---------------------------------------------------------------------------
# 4. spectral gated unit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralGatedUnit:
    """a per-band complex gain whose magnitude is gated by band power.

    the block that exists because a fixed transfer function cannot express
    something the physiology plainly does: the gain of a coupling depends on how
    much power is where.  alpha-band power gates the gain of everything faster;
    a high-conductance state shortens the effective membrane time constant and
    moves the whole transfer function; attention changes the gain of a pathway
    without changing its latency.  all three are the same computational shape --
    a gain modulated by the state's own spectral content -- and none of them is
    an LTI filter.

    the design respects band structure rather than fighting it, and that is the
    reason to prefer this over a dense complex-valued layer over all k
    components.  a dense layer has k^2 parameters, mixes DC with gamma, and
    learns cross-frequency couplings that are mostly artefacts of the training
    window.  this block instead partitions the retained components into bands,
    applies one complex gain per band -- which is exactly a piecewise-constant
    transfer function, so it *is* an LTI filter for any fixed gate value -- and
    lets a small gate network modulate those gains from the band powers.

    the consequence is that cross-frequency interaction is present but
    constrained to one channel: band powers in, band gains out.  phase-amplitude
    coupling is representable (slow band power sets fast band gain) and arbitrary
    cross-frequency mixing is not, which is the correct asymmetry -- the first is
    well documented and the second is mostly what overfitting looks like in this
    domain.

    the phase of each gain is a delay, and letting the gate modulate magnitude
    only is deliberate: a state-dependent *latency* is a much stronger claim than
    a state-dependent gain and there is far less evidence for it.
    """

    #: band edges in Hz, ascending, length ``n_bands + 1``.  the bands should be
    #: the ones the process actually distinguishes, not a fixed octave grid.
    band_edges_hz: tuple[float, ...] = (0.0, 4.0, 8.0, 13.0, 30.0, 90.0, 300.0)
    #: hidden width of the gating network.  small: it maps a handful of band
    #: powers to a handful of gains and does not need capacity.
    gate_hidden: int = 16
    #: whether the gate reads band powers at all.  false makes the block a plain
    #: piecewise-constant LTI filter, which is a useful ablation and a useful
    #: prior mean.
    gated: bool = True
    tying: Tying = Tying.PER_PARTITION

    @property
    def n_bands(self) -> int:
        return len(self.band_edges_hz) - 1

    def param_count(self, n_sites: int = 0, n_partitions: int = 1) -> int:
        """free parameters.

        the per-band gains are complex, so ``2 * n_bands`` real numbers, times
        whatever the tying multiplies them by: 1 for GLOBAL, ``n_partitions`` for
        PER_PARTITION, ``n_sites`` for PER_SITE.  EMBEDDING is priced as
        PER_PARTITION here, since the factorization applies to the gain table and
        not to the gate.

        the gate network is shared across sites in every policy and costs
        ``n_bands * h + h + h * n_bands + n_bands``.  sharing it is a real
        assumption: it says the *rule* by which power gates gain is the same
        everywhere while the gains themselves are not.  with seven bands and a
        hidden width of 16 that is about 250 parameters, against 14 per site for
        the gains -- so under PER_SITE at 10^5 positions the gains dominate by
        three orders of magnitude and the gate is free.
        """
        b = self.n_bands
        if self.tying is Tying.GLOBAL:
            mult = 1
        elif self.tying is Tying.PER_SITE:
            mult = max(n_sites, 1)
        else:
            mult = max(n_partitions, 1)
        n = 2 * b * mult
        if self.gated:
            h = self.gate_hidden
            n += b * h + h + h * b + b
        return n

    def band_masks(self, freqs_hz: np.ndarray) -> np.ndarray:
        """(n_bands, k) boolean masks over the retained components.

        computed from the basis's own frequencies rather than from indices,
        because the same declared band covers a different number of components at
        every window length and hard-coding indices is how a block silently
        changes meaning when the runtime changes its window.
        """
        f = np.asarray(freqs_hz, float)
        e = self.band_edges_hz
        return np.stack([(f >= e[i]) & (f < e[i + 1]) for i in range(self.n_bands)])

    def band_power(self, z: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """total power in each band: (..., n_bands)."""
        p = np.abs(np.asarray(z)) ** 2
        return np.stack([p[..., m].sum(-1) for m in masks], axis=-1)

    def gate(self, power: np.ndarray, w1: np.ndarray, b1: np.ndarray,
             w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
        """band powers -> per-band multiplicative gain in (0, 2).

        log power in, because power spans orders of magnitude and a linear layer
        over raw power is dominated by whichever band happens to be largest.
        twice a sigmoid out, so the gate can suppress a band or double it and the
        identity sits at the middle of its range -- which is what makes an
        untrained gate harmless.
        """
        h = np.tanh(np.log(np.asarray(power, float) + _EPS) @ w1 + b1)
        return 2.0 * sigmoid(h @ w2 + b2)

    def apply(self, z: np.ndarray, freqs_hz: np.ndarray, gains: np.ndarray,
              gate_params: Sequence[np.ndarray] | None = None) -> np.ndarray:
        """apply the block to spectral coefficients ``z`` of shape (..., k).

        exact and diagonal, like every LTI operation in the model: the result is
        `z` multiplied elementwise by a per-component complex number, so it
        composes with `SpectralGaussian.apply_transfer` and costs nothing beyond
        the multiply.  the only non-LTI part is that the multiplier depended on
        the state, which is why an implementation using this block must set
        `state_dependent_weights`.
        """
        masks = self.band_masks(freqs_hz)
        g = np.asarray(gains, dtype=np.complex128)
        if self.gated and gate_params is not None:
            g = g * self.gate(self.band_power(z, masks), *gate_params)
        h = np.zeros(np.asarray(freqs_hz).shape, dtype=np.complex128)
        for i, m in enumerate(masks):
            h[m] = g[..., i] if g.ndim == 1 else g[..., i].item()
        return np.asarray(z) * h


# ---------------------------------------------------------------------------
# 5. residual over physics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualOverPhysics:
    """wrap a learned module so that it is a correction to an analytic f rather
    than a replacement for it.

        y = f_analytic(x, theta_a) + alpha * f_learned(x, theta_l)

    every Form.LEARNED implementation in the inventory is parameterized this way,
    and this block is where the pattern is written down once so that the reason
    survives.

    the reason is not modesty about neural networks.  it is that the analytic
    forms are the only part of the model with any claim to extrapolate.  a
    balloon model fitted to one dataset makes a specific, physically grounded
    prediction about a hypercapnic challenge it has never seen; a learned map
    fitted to the same data makes no prediction at all outside its support and
    will produce something confident and wrong.  parameterizing the learned part
    as a residual with a small prior on `alpha` means the model's behaviour
    degrades towards the physics exactly where the data runs out, which is where
    it should.

    it also makes the fitted residual *interpretable as a diagnostic*.  a large
    fitted alpha in one region is a statement that the analytic form is wrong
    there, and that is a finding rather than a nuisance -- it says where the
    physics needs work.  a monolithic learned model cannot make that statement
    because it has no reference to be wrong against.

    the honest cost: the decomposition is not identifiable in general.  if
    `f_learned` can represent `f_analytic`, the split between them is arbitrary
    and only the sum is determined.  the small prior on alpha is what breaks the
    tie, and it breaks it by fiat rather than by evidence.  `gate_by_confidence`
    is offered as a sharper alternative -- scaling the residual down where the
    analytic form's own validity conditions are violated -- but it requires the
    materialization to supply that signal, which not all do.
    """

    #: the analytic implementation this corrects, by ``process:name`` key.  named
    #: rather than held as a callable so that the wrapper is a declaration a
    #: materialization can record and check, not a closure.
    analytic: str
    #: the learned block being wrapped.
    learned: object
    #: prior mean of the residual gain.  small by construction: with no data the
    #: process must reduce to the analytic form.
    alpha_init: float = 0.1
    #: whether alpha is per-partition rather than global.  usually yes -- the
    #: analytic form is wrong by different amounts in different places, and that
    #: variation is the diagnostic.
    tying: Tying = Tying.PER_PARTITION
    #: scale the residual down where the analytic form's `Validity` is violated.
    #: requires the materializer to supply a per-site confidence.
    gate_by_confidence: bool = False

    def param_count(self, n_sites: int = 0, n_partitions: int = 1,
                    inner: int = 0) -> int:
        """free parameters: the wrapped block's, plus the residual gains.

        one alpha under GLOBAL, `n_partitions` under PER_PARTITION or EMBEDDING,
        `n_sites` under PER_SITE.  the gain is one number and it is the most
        informative number in the whole fitted model, because it is the answer to
        "how much of this coupling does the physics fail to explain".

        `inner` is the wrapped block's own count and is passed in rather than
        computed, because the wrapped block's `param_count` needs arguments this
        wrapper does not have and guessing them would produce a plausible wrong
        number.
        """
        if self.tying is Tying.GLOBAL:
            return inner + 1
        if self.tying is Tying.PER_SITE:
            return inner + max(n_sites, 1)
        return inner + max(n_partitions, 1)

    def combine(self, analytic_out: np.ndarray, learned_out: np.ndarray,
                alpha: np.ndarray | float, confidence: np.ndarray | None = None
                ) -> np.ndarray:
        """the sum.  trivially simple, and written out because the whole point of
        the block is that this specific arithmetic happens rather than a
        concatenation or a gate.

        when `gate_by_confidence` is set, `confidence` in [0, 1] multiplies the
        residual: 1 where the analytic form is being used inside its declared
        validity and falling towards 0 as it is pushed outside.  the effect is
        that the learned correction is trusted where the physics is trusted --
        which is backwards from the intuition that a learned term should take
        over where physics fails, and deliberately so.  outside the analytic
        form's validity there is usually no training data either, so the learned
        term is extrapolating too, and the honest response is to widen the
        posterior rather than to lean on either.
        """
        a = np.asarray(alpha, float)
        r = np.asarray(learned_out, float)
        if self.gate_by_confidence and confidence is not None:
            r = r * np.asarray(confidence, float)
        return np.asarray(analytic_out, float) + a * r


__all__ = [
    "sigmoid", "softmax",
    "DistanceMessagePassing",
    "PartitionEmbedding",
    "LowRankSpatioTemporal",
    "SpectralGatedUnit",
    "ResidualOverPhysics",
]
