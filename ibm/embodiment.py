"""the I/O contract between ibm-1 and a body simulator.

the question this answers is "if I drop this brain into a body, what exactly do I
connect, and in what units".  everything it returns is derived from the
declarations -- components, anatomy, the nerve topology -- so it cannot drift out
of sync with the model the way a hand-written interface document would.

the contract is deliberately narrow and it runs one way at each port.  ibm-1 owns
neural state and receptor transduction.  **the simulator owns the plant**: the
skeleton, the joints, the segment inertias, the contact model, the world.  the
wires are exactly the places where those two meet, and there are four kinds:

*motor out* -- per muscle, an alpha drive and a gamma drive.  the gamma wire is
the one people forget, and omitting it is not a simplification: without fusimotor
drive the spindle unloads whenever the muscle shortens, so a simulator that
ignores gamma gets a brain whose proprioception switches off during movement.

*plant in* -- per muscle, length, velocity and force.  these are what the
receptors transduce.

*sensor in* -- per receptor surface, the physical quantity at it.

*afferent out* -- not a wire the simulator connects, but the fibre-class traffic a
recording would see, listed because it is the observable that says whether the
loop is working.

every port carries its conduction delay, and the delays differ by fibre class on
the same nerve by up to two orders of magnitude -- so a simulator that applies one
latency per muscle has already lost the thing the periphery is for.
"""

from __future__ import annotations

from dataclasses import dataclass

from ibm.anatomy.muscles import INNERVATION
from ibm.topologies.nerve import TRUNK_COMPOSITION, TRUNK_LENGTH_MM, fibre_delays_s


@dataclass(frozen=True)
class Port:
    """one wire: a name, a direction, a component, units, and a latency."""
    name: str
    direction: str            # "to_body" | "from_body"
    component: str
    units: str
    bounds: tuple[float, float]
    delay_s: float
    nerve: str = ""
    roots: tuple[str, ...] = ()

    def __str__(self) -> str:
        arrow = "-->" if self.direction == "to_body" else "<--"
        return (f"{self.name:46s} {arrow} {self.component:28s} "
                f"{self.units:8s} {self.delay_s * 1000:6.1f} ms  {self.nerve}")


def motor_ports() -> list[Port]:
    """two per muscle: alpha to the contractile machinery, gamma to the spindle."""
    out: list[Port] = []
    for muscle, (nerve, roots, spindle_density) in sorted(INNERVATION.items()):
        d = fibre_delays_s(nerve)
        # a cranial or cutaneous trunk may not carry motor classes; fall back to the
        # trunk's own length at alpha velocity rather than inventing a delay.
        d_alpha = d.get("alpha", (TRUNK_LENGTH_MM.get(nerve, 200.0) * 1e-3) / 100.0)
        d_gamma = d.get("gamma", (TRUNK_LENGTH_MM.get(nerve, 200.0) * 1e-3) / 25.0)
        out.append(Port(f"{muscle}.alpha_drive", "to_body", "neural.efferent.alpha",
                        "Hz", (0.0, 200.0), d_alpha, nerve, roots))
        if spindle_density > 0.0:
            out.append(Port(f"{muscle}.gamma_drive", "to_body", "neural.efferent.gamma",
                            "Hz", (0.0, 200.0), d_gamma, nerve, roots))
    return out


def plant_ports() -> list[Port]:
    """three per muscle: what the simulator must report back for proprioception."""
    out: list[Port] = []
    for muscle, (nerve, roots, spindle_density) in sorted(INNERVATION.items()):
        d = fibre_delays_s(nerve)
        d_ia = d.get("ia", (TRUNK_LENGTH_MM.get(nerve, 200.0) * 1e-3) / 100.0)
        out.append(Port(f"{muscle}.length", "from_body", "effector.length",
                        "L0", (0.0, 2.0), d_ia, nerve, roots))
        out.append(Port(f"{muscle}.velocity", "from_body", "effector.velocity",
                        "L0/s", (-20.0, 20.0), d_ia, nerve, roots))
        out.append(Port(f"{muscle}.force", "from_body", "effector.force",
                        "N", (0.0, 5000.0), d_ia, nerve, roots))
    return out


#: the receptor surfaces and the physical quantity each one reads.  these are the
#: non-muscular sensory ports; the muscular ones are `plant_ports`.
SENSOR_PORTS: tuple[tuple[str, str, str, tuple[float, float]], ...] = (
    ("retina.luminance",          "device.display_luminance", "cd/m^2", (0.0, 1e4)),
    ("cochlea.pressure",          "device.speaker_pressure",  "Pa",     (-100.0, 100.0)),
    ("skin.pressure",             "mechanical.pressure",      "Pa",     (0.0, 1e6)),
    ("skin.displacement",         "mechanical.displacement",  "m",      (-0.05, 0.05)),
    ("skin.temperature",          "thermal.temperature",      "degC",   (0.0, 50.0)),
    ("vestibular.acceleration",   "mechanical.velocity",      "m/s",    (-20.0, 20.0)),
    ("epithelium.concentration",  "extracellular.ph",         "pH",     (6.5, 8.0)),
    ("viscera.blood_pressure",    "blood.pressure",           "mmHg",   (0.0, 250.0)),
    ("viscera.oxygenation",       "blood.oxygenation",        "frac",   (0.0, 1.0)),
)


def sensor_ports() -> list[Port]:
    out = []
    for name, comp, units, bounds in SENSOR_PORTS:
        surface = name.split(".")[0]
        # receptor-to-cord/brainstem latency is the afferent stage, not a trunk
        delay = {"retina": 0.0067, "cochlea": 0.00125, "skin": 0.012,
                 "vestibular": 0.0004, "epithelium": 0.02, "viscera": 0.06}[surface]
        out.append(Port(name, "from_body", comp, units, bounds, delay))
    return out


def visceral_ports() -> list[Port]:
    """the interoceptive port group: one wire per visceral afferent channel.

    kept apart from `sensor_ports` because these do not share its shape.  a
    sensor port carries one latency per receptor SURFACE -- 12 ms for skin,
    0.4 ms for the vestibular organ -- and the whole content of this group is
    that a single visceral latency is wrong by 55x: the vagal A-beta channel
    reporting gastric volume arrives in 9 ms and the vagal C channel reporting
    the same meal's nutrient content in 508 ms, over the route IHM measured.
    Collapsing them into one `viscera` entry in the surface table is exactly the
    lumping `ibm/topologies/nerve.py` exists to refuse, so the delay here comes
    from the (trunk, fibre class) pair.

    before this there were two visceral wires, `viscera.blood_pressure` and
    `viscera.oxygenation`, both at a flat 60 ms.
    """
    from ibm.interoception import PORTS, group_delays_s
    d = group_delays_s()
    out: list[Port] = []
    for p in PORTS:
        out.append(Port(f"viscera.{p.channel}", "from_body", p.receptor,
                        "Hz", (0.0, 100.0), d[(p.trunk, p.fibre)], p.trunk))
    return out


def manifest() -> dict[str, list[Port]]:
    return {"motor_out": motor_ports(),
            "plant_in": plant_ports(),
            "sensor_in": sensor_ports(),
            "visceral_in": visceral_ports()}


def describe() -> str:
    m = manifest()
    lines = ["ibm-1 <-> body simulator: the wires", ""]
    total = 0
    for group, ports in m.items():
        lines.append(f"== {group}  ({len(ports)} ports) ==")
        total += len(ports)
        for p in ports[:4]:
            lines.append("  " + str(p))
        if len(ports) > 4:
            lines.append(f"  ... {len(ports) - 4} more")
        lines.append("")
    lines.append(f"{total} ports over {len(INNERVATION)} named muscles and "
                 f"{len(SENSOR_PORTS)} receptor surfaces.")
    lines.append("")
    lines.append("the simulator owns the plant -- skeleton, joints, inertia, contact,")
    lines.append("world.  ibm-1 owns neural state and transduction.  these are the")
    lines.append("places they meet, and every one carries its own conduction delay.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
