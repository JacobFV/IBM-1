"""mechanics: the brain as a very soft, nearly incompressible, fluid-saturated
solid sitting in a closed box.

this file spans two decades of timescale and the split matters, because the two
regimes are different physics wearing the same variables.  the fast one is
elastic wave propagation: shear modulus around 2 kPa against a density around
1040 kg/m^3 gives a shear wave speed near 1.4 m/s, so a shear wave crosses a
centimetre in about seven milliseconds and mr elastography drives tissue at
50 Hz to watch it.  the slow one is bulk displacement: the brain moves a hundred
micrometres or so with every heartbeat, breathes with respiration, and settles
over minutes when posture changes.  `timescale_s` is set to the fast number
because that is what bounds a materialization's step, and the docstrings below
say where the slow regime takes over.

the constraint that organizes everything here is that the box is closed.  brain
tissue is nearly incompressible -- poisson's ratio is about 0.4999, the bulk
modulus is six orders of magnitude above the shear modulus -- so it can be
sheared almost freely and compressed essentially not at all.  every volume the
arteries take up must be given back by csf or venous blood leaving the cranium,
which is the monro-kellie doctrine and the reason `csf.pressure` and
`blood.volume` are inputs to a mechanical process rather than decorations on it.
it is also why the interesting failure of this file is a stiffness problem: an
incompressible solid has an infinitely fast pressure mode, and any explicit
formulation of it either takes absurdly small steps or lies.  the constraint
implementation is the honest way out, per ARCHITECTURE.md §4 -- the
incompressibility is the stiff limit of pressure, not a separate kind of thing.

everything in this file is scalar-form throughout, so no uncertainty conversion
is inserted.  that is not a virtue, it is a statement about what is being
modelled: mechanical state is being carried as an amplitude with one timescale,
which is adequate for the pulsatile and quasi-static regimes and is already a
poor description of a propagating wave field, whose phase is the whole content.
a materialization that cares about wave phase should be reading this process's
output through a spectral variable, and the ontology as it stands does not offer
one.
"""

from __future__ import annotations

import numpy as np

from ibm.processes.base import implementation, process
from ibm.registry import Form
from ibm.vocabulary import (
    Band,
    OnSupport,
    Provenance,
    STRUCTURAL,
    Tying,
    Validity,
    lognormal,
    normal,
    speculative,
    weak,
    within,
)

#: the band mechanical state is declared over.  the top end is set by mr
#: elastography practice -- tissue is driven between 20 and 100 Hz and the
#: viscoelastic parameters reported in the literature are the values at those
#: frequencies -- with headroom to 200 Hz.  it deliberately does *not* reach the
#: megahertz band of therapeutic ultrasound: at those frequencies the medium is
#: a fluid with a compressional wave in it and the shear physics below is
#: irrelevant, so a tus materialization needs a different f, not a wider band.
ELASTIC = Band(0.0, 200.0)

#: the sub-band where bulk displacement lives: cardiac at ~1 Hz, respiratory at
#: ~0.3 Hz, and the postural and vasomotor drifts below both.  read separately
#: from the wave band because the quasi-static implementation is only valid here.
PULSATILE = Band(0.0, 5.0)


# ---------------------------------------------------------------------------
# transfer functions
# ---------------------------------------------------------------------------


def viscoelastic_mode_transfer(basis, shear_modulus_pa: float = 2000.0,
                               density_kg_m3: float = 1040.0,
                               viscosity_pa_s: float = 3.0,
                               wavelength_mm: float = 10.0):
    """one spatial eigenmode of a kelvin-voigt viscoelastic solid.

        rho u'' = -(G + eta d/dt) q^2 u + f,   q = 2 pi / L

        H = 1 / (G q^2 + i omega eta q^2 - omega^2 rho)

    a damped harmonic oscillator per mode, which is what a linear elastic solid
    is on the eigenbasis of its laplacian.  the resonance sits at
    ``sqrt(G/rho) q / 2 pi`` -- about 22 Hz for a centimetre mode at 2 kPa -- and
    the damping term is what stops it ringing.  this is the arithmetic mr
    elastography inverts, so the parameters below are measured in exactly the
    regime this expression describes, which is rarer than it sounds.

    two things it does not contain.  it is the shear branch only: the
    compressional branch travels at 1540 m/s and is, at these frequencies,
    instantaneous and therefore properly a constraint rather than a wave.  and
    kelvin-voigt damping is linear in frequency, whereas measured brain tissue
    damping is closer to a power law with a fractional exponent -- the springpot
    or fractional-derivative models fit better across a decade of frequency, and
    this form is a two-parameter fit to a three-parameter reality.
    """
    omega = basis.omega
    q = 2.0 * np.pi / max(wavelength_mm * 1e-3, 1e-30)
    stiffness = shear_modulus_pa * q ** 2
    damping = viscosity_pa_s * q ** 2
    return 1.0 / (stiffness + 1j * omega * damping - omega ** 2 * density_kg_m3)


def pulsatile_displacement_transfer(basis, tau_s: float = 0.2,
                                    compliance_gain: float = 1.0):
    """intracranial volume change into tissue displacement: a low pass.

    at a heartbeat's frequency the tissue is far below its own shear resonance,
    so it moves quasi-statically: displacement follows the volume the arteries
    took up, filtered by how fast the csf and venous compartments can accommodate
    it.  one pole with a short time constant, and a dc gain that is entirely
    determined by the compliance of the compartments that have to give way.

    the measured amplitude is the check on it: brain tissue displaces of order a
    hundred micrometres per cardiac cycle, more near the brainstem and less in
    the convexity, and any parameterization that produces millimetres or
    nanometres is wrong for a reason worth finding.
    """
    omega = basis.omega
    return compliance_gain / (1.0 + 1j * omega * tau_s)


# ---------------------------------------------------------------------------
# mechanical propagation
# ---------------------------------------------------------------------------

MECHANICAL_PROPAGATION = process(
    id="mechanical_propagation",
    doc="""displacement, strain, stress and pressure propagating through brain
    tissue and the fluids saturating it.

    declared over the mechanical topology, which is the volumetric contact
    relation: mechanical coupling is transmitted between materially adjacent
    points, and unlike the vascular or tractometric cases euclidean adjacency is
    the right relation for once.  the fluid compartments enter through their
    pressures rather than through the topology, because what couples them to the
    solid is a boundary condition, not an edge.

    the drives are all read rather than assumed.  `blood.pressure` and
    `blood.volume` carry the cardiac and vasomotor volume pulse that makes the
    brain move at all; `csf.pressure` carries the compartment that has to yield
    when it does; `material.stiffness`, `mass_density` and `porosity` carry the
    tissue an atlas or an elastography acquisition supplies.  external mechanical
    drive -- a transducer, an impact, a head movement -- arrives as an
    intervention clamping the mechanical variables at a boundary, which is why
    there is no device input in this declaration.

    where it breaks, in order of how badly.  linear viscoelasticity is a
    small-strain theory and traumatic loading is not small strain; at the strains
    where injury happens the tissue is strongly nonlinear, rate-dependent and
    damaging, and none of that is here.  the tissue is treated as isotropic when
    white matter is measurably stiffer along its fibres than across them, which
    is a 10-30% anisotropy that matters for any question about tract-level
    strain.  and the brain-skull interface -- the arachnoid trabeculae and the
    csf film that let the brain slide -- is the single most important boundary
    condition for impact biomechanics and appears here only as whatever the
    topology's edge set does at the surface.""",
    inputs=(
        within("mechanical", "displacement", "velocity", region=OnSupport("head_volume"),
               band=ELASTIC),
        within("mechanical", "stress", "strain", "pressure",
               region=OnSupport("head_volume"), band=ELASTIC),
        within("csf", "pressure", band=PULSATILE),
        within("blood", "pressure", "volume", band=PULSATILE),
        within("material", "stiffness", "mass_density", "porosity", band=STRUCTURAL),
        within("structural", "fiber_orientation", band=STRUCTURAL),
    ),
    outputs=(
        within("mechanical", "displacement", "velocity", region=OnSupport("head_volume"),
               band=ELASTIC),
        within("mechanical", "stress", "strain", "pressure",
               region=OnSupport("head_volume"), band=ELASTIC),
    ),
    topology="mechanical",
    timescale_s=0.01,
    validity=Validity(
        min_spacing_mm=0.5, max_spacing_mm=20.0, band=ELASTIC,
        note="a continuum description of a material whose microstructure is cells and "
             "extracellular matrix.  below ~0.5 mm the averaging cell is small enough that "
             "a single modulus is not a property of it, and the vasculature inside it stops "
             "being a homogenizable inclusion; above ~20 mm grey matter, white matter and "
             "csf are averaged into one material whose moduli differ by orders of magnitude, "
             "and a shear wave speed computed from that average is meaningless."),
    provenance=Provenance.PHYSICS,
    tags=frozenset({"mechanical", "physical"}),
)

implementation(
    name="viscoelastic_wave_lti",
    process="mechanical_propagation",
    doc="""kelvin-voigt shear wave propagation on the eigenmodes of the
    mechanical topology.

    the form mr elastography measures and therefore the one whose parameters are
    real.  linear, so exact at any timestep, with delays and dispersion carried
    as phase rather than as a history buffer -- which for a wave process is the
    whole cost argument, since an explicit wave solver's step is bounded by the
    fastest wave speed on the finest cell.

    valid for small strains at the frequencies elastography actually uses.  it
    fails in three known ways: kelvin-voigt damping rises linearly with frequency
    where measured tissue damping follows a fractional power law, so fitting one
    frequency and predicting another is a systematic error; it is isotropic, so
    it cannot express the fibre-aligned stiffness of white matter; and it
    contains only the shear branch, so any question involving compression needs
    the constraint implementation alongside it rather than instead of it.""",
    form=Form.LTI,
    transfer=viscoelastic_mode_transfer,
    params={
        "shear_modulus_pa": lognormal(2000.0, 1.7, units="Pa",
                                      provenance=Provenance.LITERATURE,
                                      source="mr elastography of human brain at 50-60 Hz; 1-3 kPa",
                                      note="frequency dependent by roughly a factor of two across "
                                           "the elastography band, and the reported value is always "
                                           "the value at the drive frequency; grey and white matter "
                                           "differ by 10-30% and the sign of that difference is "
                                           "still argued about"),
        "viscosity_pa_s": lognormal(3.0, 3.0, units="Pa.s",
                                    provenance=Provenance.LITERATURE,
                                    source="loss modulus roughly a quarter of the storage modulus",
                                    note="the kelvin-voigt viscosity that reproduces a damping ratio "
                                         "near 0.25 at 50 Hz; it is a fit, not a material constant"),
        "density_kg_m3": normal(1040.0, 20.0, units="kg/m^3",
                                provenance=Provenance.LITERATURE,
                                note="essentially water, which is why the shear wave speed is set "
                                     "almost entirely by the modulus"),
        "damping_ratio": normal(0.25, 0.10, units="dimensionless",
                                provenance=Provenance.LITERATURE,
                                source="mre damping ratio of human brain, 0.2-0.3",
                                note="carried alongside the viscosity because it is what elastography "
                                     "reports and the two are not independently identifiable"),
        "shear_wave_speed_m_s": lognormal(1.4, 1.4, units="m/s",
                                          provenance=Provenance.LITERATURE,
                                          source="sqrt(G/rho) at G = 2 kPa",
                                          note="about a millionth of the compressional speed; the "
                                               "ratio is the numerical difficulty of this file"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.LITERATURE,
    source="mr elastography literature; sack et al; hiscox et al 2016 review",
)

implementation(
    name="poroelastic_biot",
    process="mechanical_propagation",
    doc="""biot poroelasticity: a solid skeleton and a pore fluid that can move
    through it.

    the form to select when the fluid and the solid are the same question --
    hydrocephalus, oedema, infusion, the slow deformation that follows a shunt.
    it is the only implementation here in which the tissue can change volume by
    losing water, which is the mechanism the clinical questions turn on, and it
    couples directly to the interstitial transport processes because the pore
    fluid it moves is the same interstitial fluid they carry.

    the price is severe.  the hydraulic conductivity of brain parenchyma is
    measured indirectly and reported across two orders of magnitude, so the
    consolidation timescale this implementation predicts is uncertain by the same
    factor -- and that timescale is the answer to most of the questions it would
    be selected for.  it is nonlinear, it does not preserve gaussianity, and it
    adds a pressure variable whose stiffness is the reason the constraint form
    exists.""",
    form=Form.RATE,
    params={
        "hydraulic_conductivity": speculative(1.0e-11, 30.0, units="m^4/(N.s)",
                                              note="reported over two orders of magnitude from "
                                                   "infusion experiments and back-fitted models; "
                                                   "the consolidation time scales linearly with it"),
        "biot_coefficient": normal(0.9, 0.1, units="dimensionless",
                                   provenance=Provenance.LITERATURE,
                                   note="near 1 because the solid skeleton is far more compressible "
                                        "than the water filling it"),
        "porosity": normal(0.20, 0.03, units="dimensionless",
                           provenance=Provenance.LITERATURE,
                           source="extracellular volume fraction, the same alpha the interstitial "
                                  "transport process uses",
                           note="tied to interstitial_transport's volume fraction in any "
                                "materialization that selects both; they are one quantity"),
        "drained_shear_modulus_pa": lognormal(2000.0, 2.0, units="Pa",
                                              provenance=Provenance.LITERATURE),
        "consolidation_time_s": speculative(1000.0, 30.0, units="s",
                                            note="the derived quantity, carried explicitly so its "
                                                 "uncertainty is visible rather than hidden inside "
                                                 "the conductivity"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.SPECULATIVE,
    source="biot 1941; poroelastic hydrocephalus models",
)

implementation(
    name="monro_kellie_constraint",
    process="mechanical_propagation",
    doc="""incompressible volume balance: brain plus blood plus csf is constant,
    solved algebraically.

    the stiff limit of the pressure mode, per ARCHITECTURE.md §4.  brain tissue's
    bulk modulus is about 2.1 GPa against a shear modulus of 2 kPa, a ratio of a
    million, so the compressional response is instantaneous on every timescale
    this ontology cares about.  representing it as a wave would force a
    microsecond step for a mode that carries no information; representing it as a
    constraint costs a linear solve and is exactly right.

    what it gives is the coupling that makes intracranial pressure a global
    variable: a volume added anywhere raises pressure everywhere, which is why
    icp is one number in clinical practice and why a mass lesion in one lobe
    presents with symptoms from another.  what it cannot give is any dynamics at
    all -- select it *with* one of the other implementations, not instead of one,
    unless the only question is pressure.""",
    form=Form.CONSTRAINT,
    params={
        "bulk_modulus_pa": normal(2.1e9, 3.0e8, units="Pa",
                                  provenance=Provenance.LITERATURE,
                                  source="brain tissue bulk modulus, essentially water's",
                                  note="a million times the shear modulus; the entire justification "
                                       "for treating compression as a constraint"),
        "poisson_ratio": normal(0.4999, 0.0002, units="dimensionless",
                                provenance=Provenance.LITERATURE,
                                note="the same statement in the other parameterization; the digits "
                                     "matter, since 0.499 and 0.4999 differ by a factor of ten in "
                                     "the volumetric stiffness they imply"),
        "craniospinal_compliance_ml_mmhg": lognormal(0.6, 2.0, units="mL/mmHg",
                                                     provenance=Provenance.LITERATURE,
                                                     source="from a pressure-volume index of ~25 mL",
                                                     note="the same compliance csf_flow carries; a "
                                                          "materialization selecting both should tie "
                                                          "them, since there is one craniospinal "
                                                          "space and not two"),
        "baseline_icp_mmhg": normal(10.0, 3.0, units="mmHg",
                                    provenance=Provenance.LITERATURE,
                                    source="supine adult intracranial pressure 7-15 mmHg"),
        "cardiac_displacement_um": lognormal(150.0, 2.0, units="um",
                                             provenance=Provenance.LITERATURE,
                                             source="phase-contrast and dense mri of brain tissue "
                                                    "motion over the cardiac cycle",
                                             note="largest near the brainstem and diencephalon, "
                                                  "smallest at the convexity; the observable that "
                                                  "most directly checks this implementation"),
    },
    tying=Tying.GLOBAL,
    provenance=Provenance.PHYSICS,
)

implementation(
    name="pulsatile_quasi_static_lti",
    process="mechanical_propagation",
    doc="""bulk displacement following the intracranial volume pulse, as one lag.

    the cheap form for the regime almost every materialization actually lives in:
    below a few hertz the tissue is far from its shear resonance, so it simply
    follows the volume the vasculature imposes and the wave machinery earns
    nothing.  select it when brain motion is a confound to be modelled -- image
    registration, electrode drift, the pulsatility artefact in a recording --
    rather than a subject of study.

    it is structurally incapable of propagation.  there is no wave speed in it,
    so a disturbance appears everywhere at once, which above about 10 Hz is
    visibly wrong.""",
    form=Form.LTI,
    transfer=pulsatile_displacement_transfer,
    params={
        "tau_s": lognormal(0.2, 3.0, units="s", provenance=Provenance.WEAK,
                           note="how fast the csf and venous compartments accommodate a volume "
                                "change; inferred from waveform shape rather than measured"),
        "compliance_gain": weak(1.0, 5.0, units="um per mL of volume pulse",
                                note="geometry-dominated and strongly position-dependent within the "
                                     "cranium"),
        "respiratory_fraction": normal(0.4, 0.2, units="dimensionless",
                                       provenance=Provenance.LITERATURE,
                                       note="respiration contributes a comparable share of brain and "
                                            "csf motion to the cardiac cycle, and a larger share of "
                                            "net transport"),
    },
    tying=Tying.PER_PARTITION,
    provenance=Provenance.WEAK,
)

__all__ = ["MECHANICAL_PROPAGATION"]
