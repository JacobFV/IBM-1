"""the visceral afferent port: what the body sends, on which trunk, at what speed.

`ibm/embodiment.py` declares the wires to a body simulator and had four kinds --
motor out, plant in, sensor in, afferent out -- of which the visceral share was
two ports, `viscera.blood_pressure` and `viscera.oxygenation`.  A brain wired that
way has vision, hearing and a somatic strip and is blind to its own gut.  This
module is the missing port group, and it is a JOIN rather than a new declaration:
IHM-1 computes the rates, `ibm/topologies/nerve.py` owns the trunks and the
fibre-class velocities, `ibm/topologies/ihm_bridge.py` owns the measured route
lengths, and `ibm/fields/transduction.py` already owns the receptor state.  What
was missing was the row that says which channel is which.

### the delay is the point, and it is large

Interoception is genuinely late.  Over IHM's measured 508 mm vagal route from
gastric wall to the solitary nucleus, the myelinated A-beta channel that reports
gastric volume arrives in 9 ms and the unmyelinated C channel that reports the
same meal's nutrient content arrives in 508 ms.  The splanchnic report of the
same stomach -- the high-threshold, nociceptive arm -- arrives at 168 ms, between
them.  That half-second spread is not an artifact to be smoothed away; it is why
a gut feeling is slow and a touch is not, and a model given one visceral latency
has asserted the opposite.  `ibm/topologies/nerve.py`'s whole argument applies
here more strongly than anywhere else in the body, because the ratio of fastest
to slowest on the vagus is 55x on one trunk.

### which classes an AFFERENT loop may use

`TRUNK_COMPOSITION["vagus"]` is
`("abeta", "adelta", "c", "b_preganglionic", "c_postganglionic")` and the
splanchnic trunks are `("b_preganglionic", "c_postganglionic", "c")`.  The last
two of each are EFFERENT classes -- preganglionic autonomic outflow and its
unmyelinated postganglionic continuation -- and an afferent materialization that
encoded a stimulus onto them would be driving the cortex through the wire the
cortex drives the gut with.  `AFFERENT_CLASSES` filters them out.

There is a second reason, worth stating because it is the kind of thing that
otherwise turns up as a confusing exception: `c` and `c_postganglionic` are
declared at the same conduction velocity, so a loop that carried both would find
two classes arriving on the same step and could not tell whether that was
physiology or a collapsed timestep.

### the cortical target, and what is honestly available

**The insula is not separable at the resolution the training loops run at, and
this module does not pretend otherwise.**  Interoceptive afference reaches
posterior insula first and anterior insula and anterior cingulate after; the
DK parcellation this repo declares in `ibm/anatomy/systems.py` does contain
`insula`, `rostralanteriorcingulate` and `caudalanteriorcingulate`, so a real
materialization through `ibm/materialize/build.py` has the target.  The
pretraining loops do not run on that: they run on `cortical_regions`, a
six-label geometric convention over a spherical proxy -- occipital, temporal,
parietal, frontal, precentral, postcentral -- and a sphere has no lateral sulcus
for the insula to be buried in.

So the port is the `frontal` label, which is the declared region containing the
anterior insula's and the anterior cingulate's nearest territory, and this
substitution is a named constant rather than a comment so that it appears in
every report rather than only in this file.  It is a substitution, not a target:
what it buys is that the drive lands somewhere consistent and disjoint from the
optic (occipital) and cochlear (temporal) ports, so the interoceptive term is
not competing for the same sites as a term whose result is already published.

### the honest limits of the receptor binding

Each channel names the IBM component that holds its receptor state.  The
mechanoreceptive channels bind to `transduction.baroreceptor`, which is declared
on the `viscera` support and whose own docstring says it covers "the visceral
mechanoreceptors that report gut and bladder distension" -- so that is a use of
an existing declaration and not a parallel one.

**The gap, stated rather than papered over: there is no viscera-supported
nociceptor component.**  `transduction.nociceptor` is declared on the field's
default support, which is `body_surface`.  The splanchnic channels here are
nociceptive by threshold and by fibre class -- high-threshold, unmyelinated,
silent until the organ is genuinely loaded -- and they have nowhere in the
ontology to put their receptor state that says so.  They are bound to
`transduction.baroreceptor` like the vagal ones and tagged `nociceptive` in the
row.  Adding `transduction.visceral_nociceptor` would be the fix; it is not made
here because a component added to carry one training term, with no process
reading it, is how an ontology sprawls.  It is recorded as `ONTOLOGY_GAPS`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from ibm.topologies.nerve import FIBRE_VELOCITY_M_S, TRUNK_COMPOSITION

#: the fibre classes on a trunk that carry AFFERENT traffic.  the autonomic
#: trunks declare mostly efferent classes and a materialization that drove them
#: would be sending a stimulus out along the outflow.
EFFERENT_CLASSES = frozenset({"alpha", "gamma", "b_preganglionic",
                              "c_postganglionic"})


def afferent_classes(trunk: str) -> tuple[str, ...]:
    """the classes of `trunk` an afferent loop may encode onto."""
    if trunk not in TRUNK_COMPOSITION:
        raise KeyError(f"no trunk {trunk!r}")
    got = tuple(c for c in TRUNK_COMPOSITION[trunk]
                if c not in EFFERENT_CLASSES)
    if not got:
        raise ValueError(f"trunk {trunk!r} declares no afferent fibre class; "
                         f"it carries {TRUNK_COMPOSITION[trunk]}")
    return got


#: the cortical label the interoceptive drive enters, on the spherical proxy
#: `scripts/pretrain_video_loop.py:cortical_regions` provides.  a named constant
#: so that it is printed, logged and reported rather than living in a comment.
PORT_LOBE = "frontal"
PORT_SUBSTITUTION = (
    "insula and anterior cingulate are NOT separable on the six-label spherical "
    "proxy `cortical_regions` provides (occipital, temporal, parietal, frontal, "
    "precentral, postcentral); a sphere has no lateral sulcus.  the drive enters "
    "the `frontal` label, the declared region nearest anterior insula and ACC.  "
    "the DK parcellation in ibm/anatomy/systems.py DOES declare insula, "
    "rostralanteriorcingulate and caudalanteriorcingulate, so a materialization "
    "built through ibm/materialize/build.py has the real target and this "
    "substitution ends there.")

ONTOLOGY_GAPS = (
    "no viscera-supported nociceptor component: transduction.nociceptor is on "
    "the transduction field's default support (body_surface), so the splanchnic "
    "channels -- high-threshold, unmyelinated, nociceptive by construction -- "
    "bind to transduction.baroreceptor and are tagged nociceptive in the row "
    "rather than by their component.",
    "no gastric-volume component: the quantity is derived in IHM from stomach "
    "water plus macronutrient mass at nominal densities, because BioGears "
    "exposes no gastric volume port and this repo declares no visceral "
    "mechanical.volume.",
)


@dataclass(frozen=True)
class VisceralPort:
    """one interoceptive wire, joined across both repos."""
    channel: str              # IHM channel name
    trunk: str                # an IBM declared trunk, and an IHM measured route
    fibre: str                # an IBM declared fibre class carried by that trunk
    reads: str                # the IBM component the physical quantity is
    receptor: str             # the IBM component holding the receptor state
    units: str
    nociceptive: bool = False

    @property
    def key(self) -> str:
        """the key IHM emits, `<trunk>/<channel>`."""
        return f"{self.trunk}/{self.channel}"


#: the join table.  `channel` must match `ihm.assembly.interoception.CHANNELS`;
#: `check()` asserts that rather than trusting it, because the two tables live in
#: two repositories and the failure mode of a silent mismatch is a channel that
#: is present, plausible and wired to the wrong nerve.
PORTS: tuple[VisceralPort, ...] = (
    # -- vagus: the main gut-to-brain route, ~80% afferent -----------------
    VisceralPort("gastric_distension", "vagus", "abeta",
                 "mechanical.strain", "transduction.baroreceptor", "mL"),
    VisceralPort("gastric_nutrient", "vagus", "c",
                 "metabolic.glucose", "transduction.baroreceptor", "g"),
    VisceralPort("intestinal_distension", "vagus", "abeta",
                 "mechanical.strain", "transduction.baroreceptor", "mL"),
    VisceralPort("intestinal_nutrient", "vagus", "c",
                 "metabolic.glucose", "transduction.baroreceptor", "mg/dL"),
    VisceralPort("gi_absorption", "vagus", "adelta",
                 "blood.flow", "transduction.baroreceptor", "mL/min"),
    VisceralPort("hepatoportal_glucose", "vagus", "c",
                 "metabolic.glucose", "transduction.baroreceptor", "mg/dL"),
    VisceralPort("pulmonary_stretch", "vagus", "abeta",
                 "mechanical.strain", "transduction.baroreceptor", "mL"),
    VisceralPort("aortic_baroreceptor", "vagus", "abeta",
                 "blood.pressure", "transduction.baroreceptor", "mmHg"),
    VisceralPort("aortic_chemo_hypoxia", "vagus", "c",
                 "blood.oxygenation", "transduction.baroreceptor", "mmHg"),
    VisceralPort("aortic_chemo_hypercapnia", "vagus", "c",
                 "blood.oxygenation", "transduction.baroreceptor", "mmHg"),
    # -- splanchnic: the high-threshold arm, unmyelinated -------------------
    VisceralPort("foregut_mechano", "greater_splanchnic", "c",
                 "mechanical.strain", "transduction.baroreceptor", "mL",
                 nociceptive=True),
    VisceralPort("foregut_ischaemia", "greater_splanchnic", "c",
                 "metabolic.lactate", "transduction.baroreceptor", "mg/dL",
                 nociceptive=True),
    VisceralPort("midgut_distension", "lesser_splanchnic", "c",
                 "mechanical.strain", "transduction.baroreceptor", "mL",
                 nociceptive=True),
    VisceralPort("renal_afferent", "least_splanchnic", "c",
                 "blood.flow", "transduction.baroreceptor", "mL/min",
                 nociceptive=True),
    VisceralPort("bladder_distension", "pelvic_splanchnic", "c",
                 "mechanical.strain", "transduction.baroreceptor", "mL",
                 nociceptive=True),
)

TRUNKS = tuple(dict.fromkeys(p.trunk for p in PORTS))


def route_lengths_m() -> dict[str, float]:
    """route length per visceral trunk, from IHM's measurement where it exists.

    `ihm_bridge.visceral_routes` raises rather than falling back to the typed
    trunk table, because IHM's own route contract says
    `missing_length_policy: error; never silently substitute a trunk length`
    and because the substitution is worth 158 ms on the vagus -- larger than
    most of the latencies elsewhere in this model.
    """
    from ibm.topologies.ihm_bridge import visceral_routes
    vr = visceral_routes()
    missing = [t for t in TRUNKS if t not in vr]
    if missing:
        raise KeyError(f"IHM declares no visceral route for {missing}; "
                       f"cannot compute a conduction delay without inventing a "
                       f"length.  is data/derived/canonical/peripheral.json "
                       f"present in IHM-1?")
    return {t: vr[t]["path_length_m"] for t in TRUNKS}


def group_delays_s(lengths_m: dict[str, float] | None = None) -> dict[tuple[str, str], float]:
    """conduction delay per (trunk, fibre class) actually used by the ports.

    the key is the pair because that is what an arrival step is a property of:
    the vagal A-beta group and the vagal C group are one nerve and two arrivals
    half a second apart, and the greater splanchnic C group is a third arrival
    in between.
    """
    L = route_lengths_m() if lengths_m is None else lengths_m
    out = {}
    for p in PORTS:
        if p.fibre not in afferent_classes(p.trunk):
            raise ValueError(f"{p.channel}: {p.fibre!r} is not an afferent class "
                             f"of {p.trunk!r} ({afferent_classes(p.trunk)})")
        out[(p.trunk, p.fibre)] = L[p.trunk] / FIBRE_VELOCITY_M_S[p.fibre][1]
    return out


def groups() -> dict[tuple[str, str], list[str]]:
    """the port keys belonging to each (trunk, fibre class) group, in port order."""
    out: dict[tuple[str, str], list[str]] = {}
    for p in PORTS:
        out.setdefault((p.trunk, p.fibre), []).append(p.key)
    return out


IHM_CORPUS = "data/derived/intero-corpus"


def check(corpus_meta: dict | None = None) -> dict:
    """assert the join holds, against the registry and against IHM's own table.

    the three things that can silently go wrong here, all of which have
    equivalents in this repo's corrections ledger:

    *a component id that does not exist* -- the row would read as a binding and
    bind to nothing.  every `reads` and `receptor` is looked up in the registry.

    *a fibre class the trunk does not carry* -- the delay would still compute,
    from a velocity table that has an entry for every class regardless of which
    trunk carries it, and the number would look fine.

    *a channel set that has drifted from IHM's* -- the two tables are in two
    repositories.  a channel present here and renamed there produces a port
    wired to a rate that is never sent, and the encoder still runs.
    """
    import ibm.fields  # noqa: F401  -- populates the registry
    from ibm.registry import REGISTRY

    problems = []
    for p in PORTS:
        for cid in (p.reads, p.receptor):
            if cid not in REGISTRY.components:
                problems.append(f"{p.channel}: no component {cid!r}")
        if p.fibre not in TRUNK_COMPOSITION.get(p.trunk, ()):
            problems.append(f"{p.channel}: {p.trunk!r} does not carry {p.fibre!r}")
        if p.fibre in EFFERENT_CLASSES:
            problems.append(f"{p.channel}: {p.fibre!r} is an efferent class")
    if corpus_meta is not None:
        ours = {p.key for p in PORTS}
        theirs = set(corpus_meta["channels"])
        if ours != theirs:
            problems.append(f"channel set differs from the corpus: "
                            f"only here {sorted(ours - theirs)}; "
                            f"only there {sorted(theirs - ours)}")
    if problems:
        raise ValueError("interoceptive join is broken:\n  " +
                         "\n  ".join(problems))
    return {"ports": len(PORTS), "trunks": len(TRUNKS),
            "groups": len(groups()), "checked_against_corpus":
            corpus_meta is not None}


def load_corpus_meta(root: str = ".") -> dict | None:
    p = os.path.join(root, IHM_CORPUS, "meta.json")
    return json.load(open(p)) if os.path.exists(p) else None


def describe() -> str:
    meta = load_corpus_meta()
    check(meta)
    d = group_delays_s()
    L = route_lengths_m()
    lines = [f"{len(PORTS)} visceral afferent ports on {len(TRUNKS)} trunks, "
             f"{len(d)} conduction groups", ""]
    lines.append(f"  {'channel':24s} {'trunk':20s} {'fibre':7s} "
                 f"{'delay':>9s}  {'reads':22s} nocic")
    for p in PORTS:
        lines.append(f"  {p.channel:24s} {p.trunk:20s} {p.fibre:7s} "
                     f"{1000*d[(p.trunk, p.fibre)]:8.1f}ms  {p.reads:22s} "
                     f"{'yes' if p.nociceptive else '-'}")
    lines += ["", "route lengths, from IHM's measured centrelines:"]
    for t in TRUNKS:
        lines.append(f"  {t:22s} {1000*L[t]:7.1f} mm")
    span = max(d.values()) / min(d.values())
    lines += ["", f"fastest {1000*min(d.values()):.1f} ms, "
                  f"slowest {1000*max(d.values()):.1f} ms -- {span:.0f}x.  "
                  f"that spread is the physiology, not a rounding budget.",
              "", "cortical port: " + PORT_SUBSTITUTION, ""]
    lines.append("declared ontology gaps:")
    for g in ONTOLOGY_GAPS:
        lines.append("  - " + g)
    if meta:
        lines += ["", f"corpus: {meta['n']} frames, {meta['n_train']} train / "
                      f"{meta['n_test']} test, {meta['n_channels']} channels",
                  "  " + meta["scalar_semantics"]]
    else:
        lines += ["", f"corpus absent at {IHM_CORPUS}; run IHM-1's "
                      f"scripts/collect_interoceptive_corpus.py"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
