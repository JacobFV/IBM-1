"""the process inventory.

importing this package populates the registry with every process and every
candidate implementation in ibm-1.  `ibm.load_all()` imports it and then seals,
which is when dangling selectors, unregistered topologies, band violations and
missing uncertainty conversions become errors rather than latent bugs.

every module import below is guarded, and the guard is not defensiveness for its
own sake.  the inventory is written by several people in parallel against
`docs/CONTRACT.md`, so at any moment some of these modules exist and some do
not, and a partially written inventory should still import: a materialization
targeting a haemodynamic question has no need of the mechanical module and
should not fail because nobody has written it yet.  the registry's `check()` is
the mechanism that notices what is missing -- a process reading a component
nothing writes is reported there, with the component named -- and it reports it
far more usefully than an ImportError does.

only ImportError is caught.  a module that exists and raises anything else --
a duplicate id, a malformed selector, a collision refused by the controlled
vocabulary -- is a real defect in a real file and propagates, which is the
behaviour the contract's opening paragraph asks for: the collision is the
symptom, the contract is the fix, and swallowing it helps nobody.
"""

from __future__ import annotations

# shared declaration wrappers and the standard LTI transfer library.  imported
# first and unguarded: every other module in this package imports from it, so if
# this one is broken there is nothing to degrade gracefully to.
from ibm.processes import base  # noqa: F401

# -- neural signal traffic ---------------------------------------------------

try:
    # local excitation and inhibition, laminar, lateral, tract and
    # thalamocortical propagation.
    from ibm.processes import neural  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    neural = None  # type: ignore[assignment]

try:
    # ionic exchange and interstitial diffusion.
    from ibm.processes import ionic  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    ionic = None  # type: ignore[assignment]

try:
    # release, clearance and receptor occupancy for fast and volume transmitters.
    from ibm.processes import transmitter  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    transmitter = None  # type: ignore[assignment]

try:
    # ascending modulatory systems; the one process that writes parameters
    # rather than state, so a missing module also removes every target's
    # modulatory pressure and not just its own.
    from ibm.processes import neuromodulation  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    neuromodulation = None  # type: ignore[assignment]

try:
    # the quasi-static forward model and the field's effect back on membranes.
    from ibm.processes import electromagnetic  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    electromagnetic = None  # type: ignore[assignment]

# -- physiological milieu ----------------------------------------------------

try:
    # neurovascular coupling, vascular flow, tissue exchange, bold formation.
    from ibm.processes import vascular  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    vascular = None  # type: ignore[assignment]

try:
    # substrate consumption, energy state and the heat that falls out of it.
    from ibm.processes import metabolic  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    metabolic = None  # type: ignore[assignment]

try:
    # csf flow and csf/interstitial exchange.
    from ibm.processes import csf  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    csf = None  # type: ignore[assignment]

try:
    # thermal diffusion, which reads metabolic heat and blood flow.
    from ibm.processes import thermal  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    thermal = None  # type: ignore[assignment]

try:
    # mechanical propagation through the head as an elastic body.
    from ibm.processes import mechanical  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    mechanical = None  # type: ignore[assignment]

# -- structural change -------------------------------------------------------

try:
    # the other process that writes parameters: activity history into structural
    # state and into the weights of the propagation processes.
    from ibm.processes import plasticity  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    plasticity = None  # type: ignore[assignment]

# -- the periphery -----------------------------------------------------------

try:
    # physical field state at a receptor surface into afferent drive.
    from ibm.processes import transduction  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    transduction = None  # type: ignore[assignment]

try:
    # motor-unit drive into activation, force and body mechanics.
    from ibm.processes import effector  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    effector = None  # type: ignore[assignment]

try:
    # brain, body or device field state into device element state, and back.
    from ibm.processes import device  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    device = None  # type: ignore[assignment]

# -- evidence and control ----------------------------------------------------

try:
    # observations are not a primitive (ARCHITECTURE.md §6); the likelihoods
    # attached to ordinary processes are declared here.
    from ibm.processes import observation  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    observation = None  # type: ignore[assignment]

try:
    # interventions, likewise not a primitive: externally clamped state and
    # process inputs.
    from ibm.processes import intervention  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    intervention = None  # type: ignore[assignment]

try:
    # driving early sensory state from a continuous naturalistic stimulus.
    # imported after `transduction` and `intervention` because everything it
    # declares is a second f for the former and a chain through the latter; it
    # introduces no process and no component of its own, which is the property
    # that keeps a stimulus encoder out of the state graph.
    from ibm.processes import forcing  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    forcing = None  # type: ignore[assignment]

try:
    # learned modules that implement processes declared elsewhere.  imported
    # last because a learned f is a different p(theta) over an existing
    # declaration, so it has nothing to register until the declaration exists.
    from ibm.processes import nn  # noqa: F401
except ImportError:  # pragma: no cover - written concurrently
    nn = None  # type: ignore[assignment]


__all__ = [
    "base",
    "neural",
    "ionic",
    "transmitter",
    "neuromodulation",
    "electromagnetic",
    "vascular",
    "metabolic",
    "csf",
    "thermal",
    "mechanical",
    "plasticity",
    "transduction",
    "effector",
    "device",
    "observation",
    "intervention",
    "forcing",
    "nn",
]
