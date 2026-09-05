"""shared helpers for declaring processes and their implementations.

CALLING CONVENTION -- fixed, and load-bearing across every process module.

an implementation's `transfer` is always::

    transfer(basis, **theta) -> ndarray of shape (..., basis.k)

it takes a `TemporalBasis` and returns H(omega) sampled on that basis's retained
components -- never a bare omega array, and never a curried callable.  the
runtime holds the basis (it owns the window and the retained bandwidth) and the
process does not get to choose either, so the basis is what crosses the
boundary.  a transfer needing raw frequencies takes `omega = basis.omega` in its
first line.

the helpers below follow the same shape: `low_pass(basis, tau)` returns an
array, and `series`/`parallel`/`feedback` compose arrays rather than callables.
two conventions briefly coexisted here and the mismatch was invisible until a
transfer was actually evaluated, which is why this is written down.
"""


from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ibm.fields.uncertainty.spectral import TemporalBasis
from ibm.registry import REGISTRY, Form, Implementation, Process
from ibm.vocabulary import Prior, Provenance, Sel, Tying, Validity

_EPS = 1e-30


# ---------------------------------------------------------------------------
# declaration wrappers
# ---------------------------------------------------------------------------


def process(
    id: str,
    doc: str,
    inputs: Iterable[Sel],
    outputs: Iterable[Sel],
    topology: str,
    **kw: Any,
) -> Process:
    """declare P = (I, O, T) and hand it to the one registry.

    (I, O, T) is ontology and is meant to be stable; (f, theta) is what we know
    about the dynamics and is meant to move.  keeping the two calls separate in
    the source is the whole reason a process can begin analytic and become
    learned without any declaration changing.
    """
    return REGISTRY.process(
        Process(id=id, doc=doc, inputs=tuple(inputs), outputs=tuple(outputs),
                topology=topology, **kw)
    )


def implementation(
    name: str,
    process: str,
    doc: str,
    form: Form,
    params: Mapping[str, Prior] | None = None,
    **kw: Any,
) -> Implementation:
    """declare one candidate f, and the prior over its theta.

    `params` is p(theta).  it is the honest place: a literature value goes in
    with `lognormal(...)` and `Provenance.LITERATURE`, and everything else goes
    in with `weak()` or `speculative()`.  inventing precision the science has not
    provided is not a shortcut, it is a wrong posterior later.
    """
    return REGISTRY.implementation(
        Implementation(name=name, process=process, doc=doc, form=form,
                       params=dict(params or {}), **kw)
    )


# ---------------------------------------------------------------------------
# the standard LTI library
# ---------------------------------------------------------------------------


def constant_gain(basis: TemporalBasis, gain: float = 1.0) -> np.ndarray:
    """H(omega) = g.  frequency-flat, and it is worth having a name for it.

    the quasi-static lead field is this: an instantaneous algebraic relation is
    an LTI coupling whose transfer function has no frequency dependence at all,
    which is exactly why it can be treated as the stiff limit of pressure rather
    than as a different kind of thing.
    """
    return np.full(basis.k, complex(gain))


def low_pass(basis: TemporalBasis, tau_s: float, gain: float = 1.0) -> np.ndarray:
    """first-order leak, H = g / (1 + i omega tau).

    the membrane, in one line.  a passive patch of membrane with capacitance c
    and leak conductance g_l integrates its input current with tau = c / g_l,
    and every population-level "activity relaxes towards its input" rate law is
    the same filter with a different name for tau.

    where it breaks: tau is treated as a constant, but the membrane time
    constant of a cortical neuron in a high-conductance state is several times
    shorter than at rest, because tau depends on the synaptic conductance the
    process is being used to compute.  this form is the small-perturbation limit
    around a fixed background rate, and its tau should be understood as the
    effective tau at that background, not the resting one.
    """
    return gain / (1.0 + 1j * basis.omega * tau_s)


def alpha_synapse(basis: TemporalBasis, tau_s: float, gain: float = 1.0) -> np.ndarray:
    """critically damped second-order synapse, H = g / (1 + i omega tau)^2.

    the alpha function t/tau exp(-t/tau) is what a synaptic conductance actually
    looks like when rise and decay are comparable, and the double pole is why a
    synapse rolls off at 40 dB/decade while a membrane rolls off at 20.  peak at
    t = tau, unit area for g = 1.
    """
    return gain / (1.0 + 1j * basis.omega * tau_s) ** 2


def double_exponential(basis: TemporalBasis, tau_rise_s: float, tau_decay_s: float,
                       gain: float = 1.0) -> np.ndarray:
    """two-pole synapse with distinct rise and decay.

    the honest form for receptors whose two timescales differ by an order of
    magnitude and where collapsing them onto one pole misplaces the phase: NMDA
    (~2 ms rise, ~100 ms decay) and GABA-B, whose rise is a G-protein cascade
    rather than a channel opening.
    """
    return gain / ((1.0 + 1j * basis.omega * tau_rise_s)
                   * (1.0 + 1j * basis.omega * tau_decay_s))


def resonator(basis: TemporalBasis, f0_hz: float, q: float = 3.0,
              gain: float = 1.0) -> np.ndarray:
    """second-order band-pass with unit DC gain.

        H = g w0^2 / (w0^2 - w^2 + i w w0 / q)

    a delayed negative feedback loop of two populations -- excitatory driving
    inhibitory driving excitatory -- is a damped oscillator, and the alpha and
    spindle rhythms of the thalamocortical loop are the clearest examples in the
    inventory.  q is the reciprocal of the damping: q below 1/2 is overdamped
    and has no peak, q above ~10 is a ringing loop that any real tissue would
    have to be pathological to sustain.

    the reduction is only valid while the loop stays near a fixed point.  a
    seizure is precisely the case where it does not, and this form will
    cheerfully report a large but finite response where the real system has left
    the linear regime entirely.
    """
    w0 = 2.0 * np.pi * f0_hz
    w = basis.omega
    return gain * w0 ** 2 / (w0 ** 2 - w ** 2 + 1j * w * w0 / max(q, _EPS))


def leaky_integrator(basis: TemporalBasis, tau_s: float, gain: float = 1.0) -> np.ndarray:
    """H = g tau / (1 + i omega tau): integration with a finite memory.

    the same pole as `low_pass`, and deliberately a separate name because the DC
    gain is different and the difference is the whole point.  a perfect
    integrator has infinite DC gain and unbounded drift; a leak makes the DC
    gain tau, which is the accumulation time.  extracellular potassium
    accumulating against glial uptake, and adaptation accumulating against its
    own decay, are both this.
    """
    return gain * tau_s / (1.0 + 1j * basis.omega * tau_s)


def pure_delay(basis: TemporalBasis, tau_s: float) -> np.ndarray:
    """H = exp(-i omega tau): conduction delay as a phase ramp.

    exact, at any window length, with no history buffer and no delay-bounded
    timestep.  a conventional simulator carries one ring buffer per edge and
    caps dt below the shortest delay in the graph; here a heterogeneous
    tractogram of delays is one elementwise multiply.

    the finite window is the catch: a delay comparable to the window duration
    wraps, because the laplacian basis is cyclic.  keep tau well under
    basis.duration_s or overlap the windows.
    """
    return np.exp(-1j * basis.omega * tau_s)


def delay_dispersion(basis: TemporalBasis, mean_s: float, sd_s: float) -> np.ndarray:
    """delay drawn from a distribution rather than fixed, H = exp(-i w mu - w^2 s^2 / 2).

    a tract is a bundle of fibres with a two-order-of-magnitude spread of
    diameters, so its delay is a distribution and its transfer function is that
    distribution's characteristic function: a phase ramp at the mean delay times
    a real low-pass from the jitter.  a bundle with 5 ms mean delay and 2 ms
    spread is already 3 dB down at ~55 Hz purely from dispersion, before any
    synaptic filtering -- which is why long-range coupling in the gamma band is
    weak for reasons that have nothing to do with synapses.

    gaussian is wrong in the tail: conduction velocity distributions are closer
    to lognormal and the true kernel is skewed.  the second moment is right,
    which is what the low-pass corner depends on.
    """
    w = basis.omega
    return np.exp(-1j * w * mean_s - 0.5 * (w * sd_s) ** 2)


def diffusion_mode(basis: TemporalBasis, d_eff_m2_s: float, wavelength_mm: float,
                   uptake_hz: float = 0.0) -> np.ndarray:
    """response of one spatial eigenmode of a diffusion operator.

        H = 1 / (D k^2 + u + i omega),      k = 2 pi / wavelength

    diffusion is not diagonal in the temporal basis by itself -- it couples
    positions -- but on the eigenbasis of the interstitial laplacian each mode
    decays independently, and then it is.  a materialization that has the
    topology's eigenvalues in hand supplies them here as k^2; one that does not
    supplies the wavelength of the structure it cares about.

    the uptake term matters more than it looks.  potassium released into the
    interstitium is cleared by glial Na/K-ATPase far faster than it diffuses, so
    for k+ the corner is set by u, and the spatial scale over which a hot spot
    spreads is sqrt(D/u) -- tens of micrometres, not millimetres.
    """
    k = 2.0 * np.pi / max(wavelength_mm * 1e-3, _EPS)
    rate = d_eff_m2_s * k ** 2 + uptake_hz
    return 1.0 / (rate + 1j * basis.omega)


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def series(*H: np.ndarray) -> np.ndarray:
    """cascade: the product.  a delay line into a synapse into a membrane."""
    out = np.ones_like(np.asarray(H[0], dtype=np.complex128))
    for h in H:
        out = out * np.asarray(h, dtype=np.complex128)
    return out


def parallel(*H: np.ndarray) -> np.ndarray:
    """parallel paths onto the same target: the sum.

    this is the same summation rule as `dx = sum_p f_p(x)`.  two processes
    writing one component and two branches of one process are the same
    arithmetic; the only reason to prefer one declaration over the other is
    whether the branches have separately identifiable parameters.
    """
    out = np.zeros_like(np.asarray(H[0], dtype=np.complex128))
    for h in H:
        out = out + np.asarray(h, dtype=np.complex128)
    return out


def feedback(forward: np.ndarray, loop: np.ndarray, negative: bool = True) -> np.ndarray:
    """close a loop: H = G / (1 +- G L).

    every recurrent circuit in the inventory reduces to this -- local E/I
    balance, the thalamocortical loop, the corticothalamic loop through TRN.
    negative feedback with delay in L is where resonance comes from: the loop
    gain crosses -1 in phase at the frequency where the round trip is half a
    cycle, and the denominator gets small there.

    the sign convention is explicit because getting it wrong turns a stabilizing
    circuit into a runaway one and the result still looks like a plausible
    spectrum.
    """
    f = np.asarray(forward, dtype=np.complex128)
    l = np.asarray(loop, dtype=np.complex128)
    den = 1.0 + f * l if negative else 1.0 - f * l
    return f / np.where(np.abs(den) < _EPS, _EPS, den)


def stability_margin(loop: np.ndarray) -> float:
    """min |1 + L(omega)| over the retained band.

    the distance from the nyquist curve to the -1 point, i.e. the reciprocal of
    the peak sensitivity.  1.0 is a comfortably damped loop, below ~0.5 is a
    loop that will ring, and 0 is a pole on the imaginary axis -- a circuit that
    oscillates without being driven.

    two honest caveats.  it is evaluated only on the components the basis
    retains, so a truncated band can hide an instability above kmax: a model
    materialized to 30 Hz cannot see a gamma-band runaway.  and it is a
    linearized statement about a fixed point, so a small margin is a warning
    that the linearization is about to stop being the right f, not a prediction
    of what the nonlinear system does instead.
    """
    return float(np.min(np.abs(1.0 + np.asarray(loop, dtype=np.complex128))))


__all__ = [
    "process", "implementation",
    "constant_gain", "low_pass", "alpha_synapse", "double_exponential",
    "resonator", "leaky_integrator", "pure_delay", "delay_dispersion",
    "diffusion_mode", "series", "parallel", "feedback", "stability_margin",
    "TemporalBasis", "Form", "Tying", "Provenance", "Validity",
]
