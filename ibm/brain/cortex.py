"""cortex: excitatory and inhibitory populations per area, on the sensory-association axis.

THE REFERENCE IMPLEMENTATION of `ibm/brain/`'s contract.  Every other structure module
follows this shape: `pops`, `internal`, `external`, `mods`, `drives`, `targets`, and nothing
else.

WHAT THIS IS FOR
----------------
`docs/BRAIN_SPEC.md`: fast, input-following, weakly recurrent at the sensory end; slow,
strongly recurrent, long intrinsic timescales at the transmodal end.  Every population
sparse and divisively normalised, which is what lets a subset ignite without the sheet
igniting -- the property `ibm/substrate.py` could not have at any of the 18 parameter
settings searched on 18 September, because its inhibition was subtractive and saturated.

The hierarchy values come from `ibm/substrate.py`'s declared DK table -- imported rather
than copied, so there is one such table in the repository and not two.  That table is
declared and ordinal and is marked there for replacement by the Sydnor 2021 map when that
corpus is bound; this module inherits both the table and that caveat.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No thalamic populations, no neuromodulatory ones, and no dials standing in for them.  The
cortex receives its relay input and its gain modulation from structures that exist in
`ibm/brain/`, and if one is missing the assembly reports the edge as an orphan rather than
substituting a number.  That is the whole point of the package.
"""
from __future__ import annotations

from ibm.brain import Mod, Pop, Proj  # noqa: F401
from ibm.substrate import HIERARCHY   # the one declared DK hierarchy table

STRUCTURE = "ctx"

#: the areas carried, spanning the hierarchy from primary sensory to transmodal.  A subset
#: of Desikan-Killiany rather than all 34 per hemisphere: every one here has a job in a
#: declared loop in `ibm/rhythms.py`, and an area nothing projects to is a population that
#: would sit at its sparsity doing nothing.
AREAS = (
    "pericalcarine", "lateraloccipital", "lingual", "fusiform",         # visual
    "transversetemporal", "superiortemporal",                           # auditory
    "postcentral", "precentral", "paracentral",                         # sensorimotor
    "superiorparietal", "inferiorparietal", "supramarginal",            # parietal
    "caudalmiddlefrontal", "rostralmiddlefrontal", "superiorfrontal",   # frontal
    "parsopercularis", "medialorbitofrontal", "lateralorbitofrontal",   # ventral frontal
    "rostralanteriorcingulate", "caudalanteriorcingulate",              # cingulate
    "posteriorcingulate", "precuneus", "isthmuscingulate",              # posteromedial
    "insula", "entorhinal", "parahippocampal", "temporalpole",          # limbic / MTL
    "inferiortemporal", "middletemporal", "bankssts",                   # temporal
)

N_E = 64          # units per excitatory population
N_I = 16          # and its local interneuron partner

# declared as a function of the hierarchy h in [0, 1]: value(h) = lo + (hi - lo) * h.
# Same shape, and the same reasons, as `ibm/substrate.py`'s priors -- Murray 2014's
# intrinsic-timescale gradient and the Chaudhuri 2015 recurrence gradient -- with one
# addition this model class needs and that one did not have: SPARSITY, declared per area.
TAU_E = (0.010, 0.020)
TAU_I = (0.003, 0.008)
W_REC = (0.30, 1.10)          # local recurrent excitation, weak at V1, strong at PFC
SPARSITY = (0.06, 0.12)       # transmodal cortex runs a little denser than primary
THETA_E = (0.28, 0.34)
ADAPT_TAU = (0.30, 1.20)      # the slow variable that releases a state
ADAPT_G = (0.60, 0.25)
INPUT_BUDGET = (1.30, 1.00)   # primary areas are driven harder from outside than transmodal


def h_of(area: str) -> float:
    return HIERARCHY.get(area, 0.5)


def _lerp(pair, h):
    return pair[0] + (pair[1] - pair[0]) * h


def pops():
    out = []
    for a in AREAS:
        h = h_of(a)
        out.append(Pop(
            id=f"{STRUCTURE}.{a}.E", n=N_E, kind="E",
            tau=_lerp(TAU_E, h), beta=10.0, theta=_lerp(THETA_E, h),
            sparsity=_lerp(SPARSITY, h),
            # the total excitatory input an area should receive at its operating point.
            # Without it an area receives a median of 28x this, and 19x more than its
            # neighbour, because each of its 12-25 incoming projections is normalised as
            # though it were the only one.
            input_budget=_lerp(INPUT_BUDGET, h),
            adapt_tau=_lerp(ADAPT_TAU, h), adapt_g=_lerp(ADAPT_G, h),
            depress=True, tau_rec=0.5, U=0.2, sigma=0.02,
            note=f"h={h:.2f} on the sensory-association axis"))
        out.append(Pop(
            id=f"{STRUCTURE}.{a}.I", n=N_I, kind="I",
            tau=_lerp(TAU_I, h), beta=12.0, theta=0.25,
            note="fast local interneurons; the E/I loop that carries cortical gamma"))
    return out


def internal():
    out = []
    for a in AREAS:
        h = h_of(a)
        e, i = f"{STRUCTURE}.{a}.E", f"{STRUCTURE}.{a}.I"
        # local recurrence: the gradient that makes association cortex hold and V1 follow.
        # Sparse rather than dense, because a dense recurrent matrix over a sparse code
        # gives every unit the same input and nothing can differentiate.
        out.append(Proj(e, e, weight=_lerp(W_REC, h), topology="sparse", p=0.15,
                        delay_s=0.001, note="local recurrent excitation"))
        out.append(Proj(e, i, weight=1.0, delay_s=0.001, note="drive onto interneurons"))
        out.append(Proj(i, e, weight=0.8, sign=-1, delay_s=0.002,
                        note="the E-I-E loop; its delay is what sets the gamma band"))
    return out


def external():
    """cortico-cortical long range, with conduction delays that scale with hierarchy
    distance.

    The `ctx_ctx` loop in `ibm/rhythms.py` declares feedforward at 10-25 ms and feedback at
    15-35 ms, and feedback is slower because those axons are thinner and more diffuse.  The
    delay is a real part of the computation here, not decoration: it is what the resonance
    measurement in `ibm/modes.py` will read.
    """
    out = []
    for a in AREAS:
        for b in AREAS:
            if a == b:
                continue
            ha, hb = h_of(a), h_of(b)
            d = abs(ha - hb)
            if d < 0.14 or d > 0.55:
                continue                      # neighbours on the axis, not everything
            up = hb > ha
            delay = (0.010 + 0.015 * d) if up else (0.015 + 0.020 * d)
            w = 0.35 if up else 0.20          # feedback is weaker and more diffuse
            out.append(Proj(f"{STRUCTURE}.{a}.E", f"{STRUCTURE}.{b}.E",
                            weight=w, topology="sparse", p=0.10, delay_s=delay,
                            note="feedforward" if up else "feedback"))
            if not up:
                # feedback also lands on interneurons: that is how a higher area can
                # suppress a lower one rather than only drive it
                out.append(Proj(f"{STRUCTURE}.{a}.E", f"{STRUCTURE}.{b}.I",
                                weight=0.15, topology="sparse", p=0.08, delay_s=delay,
                                note="feedback onto interneurons"))
    return out


def mods():
    """cortex originates no neuromodulation.  It receives it."""
    return []


def drives():
    """a small tonic drive so cortex is not silent with no input.

    Deliberately small: the sensory areas are meant to be driven by the thalamus, and a
    cortex that runs on its own tonic drive would be a cortex whose activity says nothing
    about the world.
    """
    return {f"{STRUCTURE}.{a}.E": 0.06 for a in AREAS}


def targets():
    return {
        "sparsity": {f"{STRUCTURE}.{a}.E": _lerp(SPARSITY, h_of(a)) for a in AREAS},
        "bands": {
            "gamma": {"pops": [f"{STRUCTURE}.{a}.E" for a in
                               ("pericalcarine", "lateraloccipital", "transversetemporal")],
                      "hz": (30.0, 80.0),
                      "note": "the local E-I-E loop; `gamma_visual` in ibm/rhythms.py"},
            "beta_posterior": {"pops": [f"{STRUCTURE}.{a}.E" for a in
                                        ("superiorparietal", "lateraloccipital")],
                               "hz": (13.0, 30.0),
                               "note": "measured at the scalp on 119 subjects, "
                                       "`beta_posterior_scalp`"},
        },
        "known_answers": {
            "timescale_gradient": {
                "claim": "the autocorrelation timescale of an area rises with its place on "
                         "the hierarchy",
                "measure": "rank correlation of measured tau against h over AREAS",
                "expect": "> 0.5",
                "note": "v2's G4 measured 0.85 on its own sheet with a permuted control at "
                        "0.03; the same must hold here or the gradient is decoration"},
            "no_ignition": {
                "claim": "no population saturates at any drive over a 32-fold range",
                "measure": "fraction of (unit, time) above 0.9",
                "expect": "0.0",
                "note": "this is the property ibm/substrate.py could not have; it is why "
                        "the populations are divisively normalised"},
        },
    }
