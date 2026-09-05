"""named prior builders: what a component's belief looks like before evidence.

a component declares `prior="neural_population"` rather than carrying numbers,
for the same reason processes declare implementations rather than carrying code.
the shape of a resting belief is an empirical claim about the variable -- a 1/f
background with an alpha bump is a claim, and a flat one is a different claim --
so it belongs somewhere it can be argued about, cited, and replaced, not inlined
in fourteen field modules.

for spectral components a builder returns a `SpectralGaussian` over a supplied
`TemporalBasis`.  this is where the architecture's remark that "aperiodic 1/f^beta
background is a one-line prior" is cashed: the laplacian eigenvalues are monotone
in frequency, so a psd *is* the diagonal of the belief and nothing has to be
fitted to express it.  a resting rhythm is an isotropic gaussian on each
(cos, sin) eigenplane -- known amplitude, uniform phase -- so every prior here
leaves `relation=None`.  phase preference is something evidence and evoked
dynamics create; it is never a prior.

for scalar components a builder returns a `ScalarGaussian` centred on a
physiological baseline with a standard deviation covering the between-subject and
between-region spread the literature actually reports.  where it reports none,
the builder says `WEAK` and widens rather than inventing a figure.

the priors are deliberately *not* tight.  a materialization that never sees data
for a region should come out smooth and uncertain there, which is the machinery
working (ARCHITECTURE.md §1), and a prior narrow enough to look confident would
hide exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ibm.fields.uncertainty.scalar import ScalarGaussian
from ibm.fields.uncertainty.spectral import SpectralGaussian, TemporalBasis
from ibm.vocabulary import Provenance


@dataclass(frozen=True)
class PriorBuilder:
    """a named, citable resting belief for one class of component.

    `form` must match the component's declared uncertainty form; a scalar
    component pointing at a spectral builder is a declaration error and is
    checked when the field modules are loaded, not when someone tries to run.
    """

    name: str
    doc: str
    form: str                        # "scalar" | "spectral"
    fn: Callable[..., Any]
    provenance: Provenance = Provenance.WEAK
    source: str = ""

    def __call__(self, *a, **kw) -> Any:
        return self.fn(*a, **kw)


PRIORS: dict[str, PriorBuilder] = {}


def register(name: str, doc: str, form: str,
             provenance: Provenance = Provenance.WEAK, source: str = ""):
    """decorator.  duplicate names are refused, as everywhere else in ibm-1."""
    if form not in ("scalar", "spectral"):
        raise ValueError(f"prior {name!r}: form must be scalar or spectral, not {form!r}")

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in PRIORS:
            raise ValueError(f"prior builder {name!r} already registered")
        PRIORS[name] = PriorBuilder(name, doc, form, fn, provenance, source)
        return fn
    return deco


def get(name: str) -> PriorBuilder:
    try:
        return PRIORS[name]
    except KeyError:
        raise KeyError(f"no registered prior builder {name!r}; have "
                       f"{', '.join(sorted(PRIORS))}") from None


def build(name: str, *a, **kw) -> Any:
    return get(name).fn(*a, **kw)


def names() -> list[str]:
    return sorted(PRIORS)


def check_components(registry) -> list[str]:
    """every component's `prior` names a builder of its own uncertainty form.

    called from `ibm.fields.__init__` after the field modules import, so a typo
    in a prior name is a load-time error rather than a materialization-time one.
    """
    bad: list[str] = []
    for c in registry.components.values():
        if c.prior is None:
            continue
        p = PRIORS.get(c.prior)
        if p is None:
            bad.append(f"{c.id}: unknown prior builder {c.prior!r}")
        elif p.form != c.uncertainty:
            bad.append(f"{c.id}: {c.uncertainty} component points at {p.form} prior {c.prior!r}")
    return bad


# ---------------------------------------------------------------------------
# spectral shape primitives
# ---------------------------------------------------------------------------


def _f(basis: TemporalBasis) -> np.ndarray:
    """frequencies with DC replaced by half the fundamental.

    DC is a real degree of freedom, not a frequency, and a power law evaluated
    there is infinite.  the lowest frequency the window can distinguish from DC
    is the honest substitute, and it makes the finite window visible in the
    prior instead of hiding it behind a clipped bin.
    """
    f = basis.freqs_hz.copy()
    lowest = 0.5 / max(basis.duration_s, 1e-9)
    f[f <= 0.0] = lowest
    return f


def aperiodic_psd(basis: TemporalBasis, exponent: float = 1.0,
                  knee_hz: float = 0.0, at_1hz: float = 1.0) -> np.ndarray:
    """P(f) = k / (knee^chi + f^chi) -- the lorentzian-knee aperiodic form.

    the knee is not cosmetic.  a pure power law asserts scale-free dynamics all
    the way to DC, which no measured neural spectrum shows; the knee is the
    timescale at which the process decorrelates, and putting it in the prior is
    the difference between a background that can be fitted and one that forces
    every low-frequency residual into the periodic components.
    """
    # evaluated as a log-sum-exp rather than as written, because the denominator
    # overflows for exactly the parameters an optimizer probes on its way past a
    # steep region -- a knee of a few hertz raised to an exponent of a few hundred
    # is a python OverflowError, not an inf, and it aborts a fit rather than
    # scoring badly.  the identity is exact; only the arithmetic differs.  a knee
    # of zero is the pure power law and comes through as log(0) = -inf, which
    # logaddexp absorbs correctly.
    f = _f(basis)
    with np.errstate(divide="ignore"):
        lk = exponent * np.log(np.float64(max(knee_hz, 0.0)))
        lf = exponent * np.log(f)
    return at_1hz * np.exp(-np.logaddexp(lk, lf))


def bump_psd(basis: TemporalBasis, center_hz: float, sd_hz: float,
             height: float) -> np.ndarray:
    """a gaussian bump on the psd: an ongoing rhythm of known band and power.

    added to power, never to the mean.  a bump in power with a zero mean and no
    relation term is precisely "there is a rhythm here and I do not know its
    phase", which is the correct resting belief about alpha, and it becomes an
    evoked response only when something anisotropic arrives on the eigenplane.
    """
    return height * np.exp(-0.5 * ((basis.freqs_hz - center_hz) / sd_hz) ** 2)


def lowpass_psd(basis: TemporalBasis, tau_s: float, gain: float = 1.0,
                order: int = 1) -> np.ndarray:
    """|H(w)|^2 for a cascade of first-order poles at 1/tau.

    the spectrum of anything driven by broadband input through a leaky
    integrator: a synaptic conductance, a receptor potential, an ion cleared by
    a pump.  writing it as a transfer magnitude rather than a fitted curve keeps
    the parameter a time constant the literature reports.
    """
    return gain / (1.0 + (basis.omega * tau_s) ** 2) ** order


def _spectral(basis: TemporalBasis, psd: np.ndarray, shape: tuple[int, ...] = (),
              dc_mean: float = 0.0) -> SpectralGaussian:
    """assemble a zero-phase-preference belief with an optional DC offset.

    a constant c analyzes, under the orthonormal transform, to c*sqrt(n) at DC,
    which is where a baseline membrane potential or a resting firing rate lives.
    everything else is power without phase.
    """
    p = np.broadcast_to(np.asarray(psd, float), shape + (basis.k,)).copy()
    mean = np.zeros(shape + (basis.k,), dtype=np.complex128)
    if dc_mean:
        mean[..., 0] = dc_mean * np.sqrt(basis.n)
    return SpectralGaussian(basis, mean, p)


# ---------------------------------------------------------------------------
# spectral priors
# ---------------------------------------------------------------------------


@register("aperiodic", "scale-free 1/f^beta background with a knee; the substrate every "
          "electrophysiological spectrum sits on and the default when nothing more "
          "specific is known about a spectral component", "spectral",
          Provenance.LITERATURE,
          source="donoghue et al. 2020 nat neurosci; he 2014 trends cogn sci")
def _aperiodic(basis: TemporalBasis, shape: tuple[int, ...] = (), exponent: float = 1.5,
               knee_hz: float = 1.0, at_1hz: float = 1.0) -> SpectralGaussian:
    return _spectral(basis, aperiodic_psd(basis, exponent, knee_hz, at_1hz), shape)


@register("neural_population", "aperiodic background plus resting theta, alpha and beta "
          "rhythms, all with uniform phase.  the canonical prior for population membrane "
          "and activity state, and the reason those components are spectral at all: the "
          "bumps and the background are simultaneously present in one variable", "spectral",
          Provenance.LITERATURE,
          source="buzsaki & draguhn 2004 science; donoghue et al. 2020 nat neurosci")
def _neural_population(basis: TemporalBasis, shape: tuple[int, ...] = (),
                       exponent: float = 2.0, knee_hz: float = 2.0, at_1hz: float = 1.0,
                       alpha_gain: float = 0.6, beta_gain: float = 0.15,
                       theta_gain: float = 0.3, alpha_hz: float = 10.0,
                       dc_mean: float = 0.0) -> SpectralGaussian:
    # alpha_hz is a parameter and the other two centres are not, because it is the
    # one this prior makes a falsifiable claim about: individual alpha peak
    # frequency varies from about 8 to 13 Hz between people and moves with state,
    # and a prior that pinned it at 10 Hz would absorb that variation into the
    # amplitude of a bump sitting in the wrong place.  theta and beta are wide,
    # low and poorly separated from the background in scalp recordings, so a free
    # centre for either buys a parameter the data cannot pin.
    p = aperiodic_psd(basis, exponent, knee_hz, at_1hz)
    total = p.sum() or 1.0
    p = p + bump_psd(basis, 6.0, 1.5, theta_gain * total / basis.k)
    p = p + bump_psd(basis, alpha_hz, 2.0, alpha_gain * total / basis.k)
    p = p + bump_psd(basis, 20.0, 5.0, beta_gain * total / basis.k)
    return _spectral(basis, p, shape, dc_mean)


@register("neural_spiking", "a poisson-like flat floor above a 1/f rate drift, offset by a "
          "positive baseline rate.  population firing is not the same spectrum as population "
          "voltage: the counting noise is white, and the structure is all at low frequency "
          "and in the rhythms that modulate it", "spectral", Provenance.LITERATURE,
          source="softky & koch 1993 j neurosci; mochol et al. 2015 pnas")
def _neural_spiking(basis: TemporalBasis, shape: tuple[int, ...] = (),
                    baseline_hz: float = 3.0, drift_exponent: float = 1.0,
                    floor: float = 1.0) -> SpectralGaussian:
    p = floor + aperiodic_psd(basis, drift_exponent, knee_hz=0.5, at_1hz=4.0 * floor)
    p = p + bump_psd(basis, 10.0, 2.5, 0.5 * floor) + bump_psd(basis, 45.0, 12.0, 0.4 * floor)
    return _spectral(basis, p, shape, baseline_hz)


@register("synaptic_conductance", "broadband presynaptic drive through the receptor's own "
          "low-pass.  the receptor time constant is the whole prior, which is why ampa, "
          "nmda, gaba-a and gaba-b are four components rather than one with a parameter",
          "spectral", Provenance.LITERATURE,
          source="destexhe et al. 1998; jahr & stevens 1990 j neurosci")
def _synaptic_conductance(basis: TemporalBasis, shape: tuple[int, ...] = (),
                          tau_s: float = 5e-3, gain: float = 1.0,
                          baseline: float = 0.0) -> SpectralGaussian:
    return _spectral(basis, lowpass_psd(basis, tau_s, gain), shape, baseline)


@register("adaptation_slow", "heavily low-passed activity history: spike-frequency "
          "adaptation, calcium-dependent potassium current, short-term depression.  its "
          "power lives below a few hertz by construction and its interest is entirely in "
          "how it interacts with the faster components of the same population", "spectral",
          Provenance.LITERATURE, source="benda & herz 2003 neural comput")
def _adaptation_slow(basis: TemporalBasis, shape: tuple[int, ...] = (),
                     tau_s: float = 0.5, gain: float = 1.0,
                     baseline: float = 0.0) -> SpectralGaussian:
    return _spectral(basis, lowpass_psd(basis, tau_s, gain, order=2), shape, baseline)


@register("transmembrane_current", "the current-source spectrum that generates the "
          "electromagnetic field: a shallower power law than the population voltage, "
          "extending into the spike band because action currents contribute there even "
          "where the extracellular filter suppresses them", "spectral", Provenance.LITERATURE,
          source="buzsaki, anastassiou & koch 2012 nat rev neurosci; einevoll et al. 2013")
def _transmembrane_current(basis: TemporalBasis, shape: tuple[int, ...] = (),
                           exponent: float = 1.0, knee_hz: float = 5.0,
                           at_1hz: float = 1.0) -> SpectralGaussian:
    p = aperiodic_psd(basis, exponent, knee_hz, at_1hz)
    return _spectral(basis, p + bump_psd(basis, 600.0, 300.0, 0.05 * at_1hz), shape)


@register("em_field", "quasi-static head-volume field spectrum: the source spectrum passed "
          "through a conduction path that is resistive to good approximation, so the shape "
          "is inherited rather than filtered.  wide, because the same component must hold "
          "both spontaneous microvolts and a stimulator's kilohertz transient", "spectral",
          Provenance.PHYSICS, source="plonsey & heppner 1967; nunez & srinivasan 2006")
def _em_field(basis: TemporalBasis, shape: tuple[int, ...] = (), exponent: float = 1.5,
              knee_hz: float = 2.0, at_1hz: float = 1.0) -> SpectralGaussian:
    p = aperiodic_psd(basis, exponent, knee_hz, at_1hz)
    return _spectral(basis, p + bump_psd(basis, 10.0, 2.0, 0.4 * at_1hz), shape)


@register("ionic_transient", "two poles in one variable: a fast pole at the release and "
          "diffusion timescale and a slow pole at glial and pump clearance.  this is the "
          "concrete reason extracellular ions are spectral and the slower solutes are not "
          "-- the shape cannot be written with a single time constant", "spectral",
          Provenance.LITERATURE,
          source="somjen 2002 neuroscientist; chever et al. 2010 j neurosci")
def _ionic_transient(basis: TemporalBasis, shape: tuple[int, ...] = (),
                     fast_tau_s: float = 0.05, slow_tau_s: float = 3.0,
                     fast_gain: float = 1.0, slow_gain: float = 4.0,
                     baseline: float = 0.0) -> SpectralGaussian:
    p = lowpass_psd(basis, fast_tau_s, fast_gain) + lowpass_psd(basis, slow_tau_s, slow_gain)
    return _spectral(basis, p, shape, baseline)


@register("receptor_transduction", "band-pass: the receptor's transduction low-pass against "
          "its adaptation high-pass.  a receptor that did not high-pass would saturate on "
          "the mean of its stimulus, and every sensory epithelium in the inventory has a "
          "mechanism that prevents that", "spectral", Provenance.LITERATURE,
          source="fain et al. 2001 physiol rev; hudspeth 2014 nat rev neurosci")
def _receptor_transduction(basis: TemporalBasis, shape: tuple[int, ...] = (),
                           tau_s: float = 0.02, adapt_tau_s: float = 1.0,
                           gain: float = 1.0, baseline: float = 0.0) -> SpectralGaussian:
    w = basis.omega
    hp = (w * adapt_tau_s) ** 2 / (1.0 + (w * adapt_tau_s) ** 2)
    return _spectral(basis, lowpass_psd(basis, tau_s, gain) * hp, shape, baseline)


@register("motor_drive", "the spectrum of common drive to a motor-unit pool: 1/f below "
          "5 Hz, physiological tremor near 10 Hz, a beta band that is corticospinal in "
          "origin, and a piper component near 40 Hz.  four bands that a scalar belief "
          "would fold into one number", "spectral", Provenance.LITERATURE,
          source="de luca & erim 1994; farina & negro 2015 exerc sport sci rev")
def _motor_drive(basis: TemporalBasis, shape: tuple[int, ...] = (), gain: float = 1.0,
                 baseline: float = 0.0) -> SpectralGaussian:
    p = aperiodic_psd(basis, 1.5, knee_hz=1.0, at_1hz=gain)
    p = p + bump_psd(basis, 10.0, 2.0, 0.5 * gain)
    p = p + bump_psd(basis, 21.0, 5.0, 0.3 * gain)
    p = p + bump_psd(basis, 40.0, 8.0, 0.1 * gain)
    return _spectral(basis, p, shape, baseline)


@register("device_broadband", "instrument noise: a white johnson-nyquist floor plus the "
          "1/f drift of an electrode double layer.  a device component's prior is a noise "
          "model, and stating it here is what lets an observation contribute honest "
          "precision instead of a fitted fudge factor", "spectral", Provenance.PHYSICS,
          source="nyquist 1928 phys rev; hassibi et al. 2004 j appl phys")
def _device_broadband(basis: TemporalBasis, shape: tuple[int, ...] = (),
                      floor: float = 1.0, drift_at_1hz: float = 10.0,
                      baseline: float = 0.0) -> SpectralGaussian:
    return _spectral(basis, floor + aperiodic_psd(basis, 1.0, 0.0, drift_at_1hz), shape, baseline)


@register("device_drive", "an exogenous drive waveform before anyone says what it is: flat "
          "across the component's band and wide.  the honest prior for a display, a speaker "
          "or a coil, because an experiment will clamp it as an intervention and the prior "
          "should not fight that", "spectral", Provenance.WEAK)
def _device_drive(basis: TemporalBasis, shape: tuple[int, ...] = (), scale: float = 1.0,
                  baseline: float = 0.0) -> SpectralGaussian:
    return _spectral(basis, np.full(basis.k, float(scale) ** 2), shape, baseline)


# ---------------------------------------------------------------------------
# scalar priors
# ---------------------------------------------------------------------------


@register("hemodynamic", "a perfusion-related variable at rest: a physiological baseline "
          "with the between-region spread that grey and white matter actually differ by. "
          "scalar because blood has essentially no structure above ~0.5 Hz and a spectrum "
          "would be budget spent on zeros", "scalar", Provenance.LITERATURE,
          source="ito et al. 2004 j cereb blood flow metab; buxton 2009")
def _hemodynamic(shape: tuple[int, ...] = (), mean: float = 50.0,
                 sd: float = 15.0) -> ScalarGaussian:
    return ScalarGaussian.prior(shape, mean, sd)


@register("metabolic_pool", "a tissue substrate or product pool at steady state.  buffered "
          "and slow: atp is held within a few percent of 2 mM by creatine kinase even while "
          "consumption doubles, so the prior is tight in the mean and wide in what can move "
          "it", "scalar", Provenance.LITERATURE,
          source="attwell & laughlin 2001 j cereb blood flow metab; erecinska & silver 1989")
def _metabolic_pool(shape: tuple[int, ...] = (), mean: float = 1.0,
                    sd: float = 0.3) -> ScalarGaussian:
    return ScalarGaussian.prior(shape, mean, sd)


@register("concentration", "a positive solute concentration whose relevant timescale is its "
          "clearance time.  the mean is a literature baseline where one exists and the "
          "spread is generous, because interstitial concentrations are measured by methods "
          "that disagree with each other by factors, not percentages", "scalar",
          Provenance.LITERATURE, source="sykova & nicholson 2008 physiol rev")
def _concentration(shape: tuple[int, ...] = (), mean: float = 1.0,
                   sd: float = 0.5) -> ScalarGaussian:
    return ScalarGaussian.prior(shape, mean, sd)


@register("structural_dc", "tissue structure over an experiment: constant.  the band is "
          "below a millihertz, so the prior is a belief about a number rather than about a "
          "trajectory, and plasticity moves it over days without ever making it fast",
          "scalar", Provenance.ATLAS, source="atlas- and histology-derived; see data/sources")
def _structural_dc(shape: tuple[int, ...] = (), mean: float = 0.0,
                   sd: float = 1.0) -> ScalarGaussian:
    return ScalarGaussian.prior(shape, mean, sd)


@register("material_constant", "an exogenous tissue property.  the spread is the real one: "
          "reported head-tissue conductivities differ across studies by a factor of three "
          "and skull by more, and a narrow prior here is the single most common way a "
          "forward model becomes confidently wrong", "scalar", Provenance.LITERATURE,
          source="gabriel et al. 1996 phys med biol; mccann et al. 2019 plos one")
def _material_constant(shape: tuple[int, ...] = (), mean: float = 1.0,
                       sd: float = 0.5) -> ScalarGaussian:
    return ScalarGaussian.prior(shape, mean, sd)


@register("thermal_dc", "brain temperature at rest: a tight prior, because it is one of the "
          "most tightly regulated quantities in the body, offset above core temperature by "
          "the few tenths of a degree that local metabolism adds", "scalar",
          Provenance.LITERATURE, source="wang et al. 2014 j neurosci methods")
def _thermal_dc(shape: tuple[int, ...] = (), mean: float = 37.2,
                sd: float = 0.5) -> ScalarGaussian:
    return ScalarGaussian.prior(shape, mean, sd)


@register("mechanical_quasistatic", "bulk tissue motion at rest: zero-mean, with a spread "
          "set by cardiac and respiratory pulsation rather than by anything the model "
          "drives.  the prior is centred at zero because displacement and strain are "
          "measured from a reference configuration, not from an origin", "scalar",
          Provenance.LITERATURE, source="greitz et al. 1992; soellinger et al. 2009 mrm")
def _mechanical_quasistatic(shape: tuple[int, ...] = (), mean: float = 0.0,
                            sd: float = 1.0) -> ScalarGaussian:
    return ScalarGaussian.prior(shape, mean, sd)


__all__ = ["PriorBuilder", "PRIORS", "register", "get", "build", "names",
           "check_components", "aperiodic_psd", "bump_psd", "lowpass_psd"]
