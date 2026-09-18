"""the rhythms this brain is built to have, and the circuits that have to produce them.

WHAT THIS IS
------------
A specification, not a measurement.  This programme is a construction: we say what the
brain must be and shape the model toward it (CLAUDE.md, "construction, not search").
A brain is not a box that maps inputs to outputs; it is a set of loops, each with an
anatomy, a conduction budget and a band it lives in.  This module is that set, written
down once, machine-readable, so three things can share it:

  * `scripts/shape_rhythms.py` -- turns an entry into a term in the training objective
    via `ibm/spectral.py`, so the substrate is PULLED toward the spectrum rather than
    inspected afterwards to see whether it happened to land there;
  * the measurement scripts -- an entry says which band, which structures and which
    behavioural state, so reading a rhythm off real EEG and reading it off the model
    are the same code path with a different input;
  * the site -- `scripts/export_rhythms.py` writes `site/data/rhythms.js` from here, so
    the eleven tiles on the page and the table in `docs/RHYTHMS.md` cannot drift from
    what the code trains against.

HONESTY, because most of this is not measured
---------------------------------------------
Every rhythm carries `evidence`:

    measured-here   we fitted it on held-out subjects in THIS repo; the entry names the
                    script and the number.  Two entries qualify today.
    measurable-now  the corpus is on disk and the instrument exists; nobody has run it.
    literature      declared from published physiology; no local measurement.
    target          what we intend the substrate to produce.  Nothing measured, here or
                    necessarily anywhere, at this specificity.

and `substrate`, which says what the MODEL would need before the entry could even be
scored on it.  Most say `needs-*`: `ibm/substrate.py` is cortex only.  A rhythm whose
loop runs through the thalamic reticular nucleus, the subthalamic nucleus or the spinal
cord cannot be expressed by a cortical sheet, and pretending otherwise by scoring the
band in cortex alone would produce a number that moves for the wrong reason.

THE ROUND-TRIP BOUND, the one law this table checks against itself
------------------------------------------------------------------
A loop cannot oscillate faster than it can go round.  With a total round-trip delay D
(conduction plus synaptic, summed over the ring):

    net-inhibitory loop   f <= 1 / (2 D)     half a cycle is one traverse
    net-excitatory loop   f <= 1 / D         a re-entrant wave, one cycle per traverse

This is an upper bound, never a prediction: most of these frequencies are set by
membrane and synaptic time constants (spindles by the T-current and GABA-B, the slow
oscillation by adaptation), with the delay far from binding.  Where it IS nearly
binding the coincidence is worth stating -- the auditory loop's ~12 ms round trip puts
its ceiling at ~42 Hz, right at the 40 Hz steady-state response, and corticomuscular
beta at 20 Hz sits on a ~50 ms re-entrant loop.  `check_catalogue()` enforces the bound
and prints the margin, so a delay or a band that is edited into inconsistency is caught
by running this file rather than by someone noticing.

UNITS.  Frequencies in Hz, delays in milliseconds, always as (lo, hi) intervals where
the literature gives a range -- a declared band IS an interval, and collapsing it to a
point invents precision.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ======================================================================================
# stations.  `labels` are atlas labels that exist in site/data/graph.js (Desikan-Killiany
# gyri plus the aseg subcortical blobs), so a station resolves to real vertices and real
# substrate sites.  `resolution` says, plainly, when the real structure is FINER than
# anything we can address: the thalamic reticular nucleus is inside `thalamus`, CA3 and
# CA1 are inside `hippocampus`, the subthalamic nucleus is inside nothing at all.
# ======================================================================================
@dataclass(frozen=True)
class Station:
    id: str
    name: str
    kind: str                       # cortex | thalamus | basal-ganglia | mtl | cerebellum
                                    # | brainstem | periphery | cord | body
    labels: tuple = ()              # atlas labels that stand for it
    resolution: str = ""            # "" when the label IS the structure
    in_substrate: bool = False      # addressable in ibm/substrate.py today (cortex only)


def _S(id, name, kind, labels=(), resolution="", in_substrate=False):
    return Station(id, name, kind, tuple(labels), resolution, in_substrate)


STATIONS = {s.id: s for s in [
    # ---- cortex: the substrate we have -------------------------------------------
    _S("v1", "primary visual cortex", "cortex", ("pericalcarine",), "", True),
    _S("v_extra", "extrastriate visual cortex", "cortex",
       ("lateraloccipital", "cuneus", "lingual", "fusiform"), "", True),
    _S("a1", "primary auditory cortex", "cortex", ("transversetemporal",),
       "Heschl's gyrus; core and belt are not separated", True),
    _S("stg", "superior temporal cortex", "cortex", ("superiortemporal", "bankssts"), "", True),
    _S("m1", "primary motor cortex", "cortex", ("precentral",), "", True),
    _S("s1", "primary somatosensory cortex", "cortex", ("postcentral",), "", True),
    _S("pmc", "premotor and supplementary motor cortex", "cortex",
       ("caudalmiddlefrontal", "paracentral"), "SMA is inside paracentral/superiorfrontal", True),
    _S("ips", "intraparietal / superior parietal cortex", "cortex",
       ("superiorparietal", "inferiorparietal", "supramarginal"),
       "IPS is a sulcus, not a DK label", True),
    _S("fef", "frontal eye field", "cortex", ("caudalmiddlefrontal", "precentral"),
       "FEF is not a DK label; it sits at this junction", True),
    _S("dlpfc", "dorsolateral prefrontal cortex", "cortex",
       ("rostralmiddlefrontal", "superiorfrontal"), "", True),
    _S("acc", "anterior cingulate cortex", "cortex",
       ("caudalanteriorcingulate", "rostralanteriorcingulate"), "", True),
    _S("vmpfc", "ventromedial prefrontal cortex", "cortex",
       ("medialorbitofrontal", "frontalpole"), "", True),
    _S("ofc", "orbitofrontal cortex", "cortex", ("lateralorbitofrontal", "parsorbitalis"), "", True),
    _S("insula", "insular cortex", "cortex", ("insula",),
       "posterior (interoceptive) and anterior (salience) insula are one label here", True),
    _S("pcc", "posterior cingulate and precuneus", "cortex",
       ("posteriorcingulate", "isthmuscingulate", "precuneus"), "", True),
    _S("ang", "angular gyrus", "cortex", ("inferiorparietal",),
       "angular and supramarginal share inferiorparietal in DK", True),
    _S("mtl_ctx", "parahippocampal and entorhinal cortex", "cortex",
       ("entorhinal", "parahippocampal"), "", True),
    _S("temporal_pole", "temporal pole", "cortex", ("temporalpole",), "", True),
    _S("broca", "inferior frontal gyrus", "cortex",
       ("parsopercularis", "parstriangularis"), "", True),

    # ---- thalamus: one blob, and the loops need four nuclei ------------------------
    _S("trn", "thalamic reticular nucleus", "thalamus", ("thalamus",),
       "TRN is a shell around the thalamus; not separable in this atlas"),
    _S("lgn", "lateral geniculate nucleus", "thalamus", ("thalamus",), "inside `thalamus`"),
    _S("mgn", "medial geniculate nucleus", "thalamus", ("thalamus",), "inside `thalamus`"),
    _S("vl", "ventrolateral thalamus", "thalamus", ("thalamus",),
       "the cerebellar-recipient motor nucleus; inside `thalamus`"),
    _S("va_vl_bg", "ventroanterior thalamus", "thalamus", ("thalamus",),
       "the pallidal-recipient nucleus; inside `thalamus`"),
    _S("md", "mediodorsal thalamus", "thalamus", ("thalamus",), "inside `thalamus`"),
    _S("pulvinar", "pulvinar", "thalamus", ("thalamus",), "inside `thalamus`"),
    _S("vpl", "ventroposterior thalamus", "thalamus", ("thalamus",),
       "somatosensory relay; inside `thalamus`"),

    # ---- basal ganglia --------------------------------------------------------------
    _S("striatum", "striatum", "basal-ganglia", ("caudate", "putamen"), ""),
    _S("gpe", "external pallidum", "basal-ganglia", ("pallidum",),
       "GPe and GPi are one `pallidum` label"),
    _S("gpi", "internal pallidum", "basal-ganglia", ("pallidum",), "see GPe"),
    _S("stn", "subthalamic nucleus", "basal-ganglia", (),
       "NO label at all: the STN is absent from this atlas"),
    _S("snc", "substantia nigra pars compacta", "basal-ganglia", ("brainstem",),
       "inside the `brainstem` blob"),
    _S("nacc", "nucleus accumbens", "basal-ganglia", ("accumbens",), ""),

    # ---- medial temporal lobe -------------------------------------------------------
    _S("dg", "dentate gyrus", "mtl", ("hippocampus",), "inside the `hippocampus` blob"),
    _S("ca3", "CA3", "mtl", ("hippocampus",), "inside the `hippocampus` blob"),
    _S("ca1", "CA1", "mtl", ("hippocampus",), "inside the `hippocampus` blob"),
    _S("sub", "subiculum", "mtl", ("hippocampus",), "inside the `hippocampus` blob"),
    _S("ms", "medial septum / diagonal band", "mtl", (),
       "NO label: the septal theta pacemaker is absent from this atlas"),
    _S("amygdala", "amygdala", "mtl", ("amygdala",), ""),

    # ---- cerebellum and its input/output --------------------------------------------
    _S("cb_ctx", "cerebellar cortex", "cerebellum", ("cerebellum",),
       "one blob: no lobules, no Purkinje/granule layers"),
    _S("dentate_n", "dentate nucleus", "cerebellum", ("cerebellum",), "inside `cerebellum`"),
    _S("pons", "pontine nuclei", "brainstem", ("brainstem",), "inside the `brainstem` blob"),
    _S("io", "inferior olive", "brainstem", ("brainstem",), "inside the `brainstem` blob"),

    # ---- brainstem, hypothalamus, neuromodulators -----------------------------------
    _S("lc", "locus coeruleus", "brainstem", ("brainstem",), "inside the `brainstem` blob"),
    _S("raphe", "dorsal raphe", "brainstem", ("brainstem",), "inside the `brainstem` blob"),
    _S("ppt", "pedunculopontine / laterodorsal tegmentum", "brainstem", ("brainstem",),
       "the cholinergic REM-on group; inside `brainstem`"),
    _S("vta", "ventral tegmental area", "brainstem", ("brainstem",), "inside `brainstem`"),
    _S("nts", "nucleus of the solitary tract", "brainstem", ("brainstem",),
       "first visceral relay; inside `brainstem`"),
    _S("pb", "parabrachial nucleus", "brainstem", ("brainstem",), "inside `brainstem`"),
    _S("sc_coll", "superior colliculus", "brainstem", ("brainstem",), "inside `brainstem`"),
    _S("preBotC", "pre-Botzinger complex", "brainstem", ("brainstem",),
       "the respiratory rhythm generator; inside `brainstem`"),
    _S("scn", "suprachiasmatic nucleus", "brainstem", (),
       "NO label: the circadian clock is absent from this atlas"),
    _S("nbm", "nucleus basalis", "brainstem", (), "NO label; cholinergic basal forebrain"),
    _S("mlr", "mesencephalic locomotor region", "brainstem", ("brainstem",), "inside `brainstem`"),

    # ---- periphery and body ----------------------------------------------------------
    _S("retina", "retina", "periphery", ("retina",), ""),
    _S("cochlea", "cochlea", "periphery", ("cochlea",), ""),
    _S("ob", "olfactory bulb", "periphery", (), "NO label in this atlas"),
    _S("cord_lumbar", "lumbar central pattern generator", "cord", ("spinal_cord", "spinal"),
       "the CPG is a distributed interneuron network, not a segment"),
    _S("cord_cervical", "cervical central pattern generator", "cord", ("spinal_cord", "spinal"), ""),
    _S("mn_pool", "alpha motoneuron pools", "cord", ("spinal_cord", "spinal"), ""),
    _S("muscle", "skeletal muscle (EMG)", "body", (),
       "IHM-1 owns the muscles; the EEG/MEG graph has no body"),
    _S("spindle_afferent", "muscle spindle afferents", "body", (), "IHM-1"),
    _S("gut", "stomach pacemaker (interstitial cells of Cajal)", "body", (), "IHM-1"),
    _S("heart", "heart", "body", (), "IHM-1"),
    _S("lung", "diaphragm and lungs", "body", (), "IHM-1"),
]}


# ======================================================================================
# loops.  `ring` is the ordered circuit -- the thing that resonates -- and `edges` carry
# the conduction budget the round-trip bound is computed from.  `sign` is the sign of
# that edge's net effect on its target population: '+' excitatory, '-' inhibitory.
# ======================================================================================
@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    delay_ms: tuple                 # (lo, hi), conduction + synaptic
    sign: str = "+"
    note: str = ""


@dataclass(frozen=True)
class Loop:
    id: str
    name: str
    summary: str                    # one line, the site's voice: what the loop is FOR
    edges: tuple
    site_tile: str = ""             # the `rz_*` tile in site/resonance.js, when there is one

    @property
    def ring(self) -> tuple:
        return tuple(e.src for e in self.edges)

    @property
    def delay_ms(self) -> tuple:
        lo = sum(e.delay_ms[0] for e in self.edges)
        hi = sum(e.delay_ms[1] for e in self.edges)
        return (lo, hi)

    @property
    def net_sign(self) -> str:
        neg = sum(1 for e in self.edges if e.sign == "-")
        return "-" if neg % 2 else "+"

    def bound_hz(self) -> tuple:
        """(fast bound, slow bound) in Hz from the round trip: a loop cannot oscillate
        faster than it can go round.  The FAST bound uses the shortest declared delay."""
        lo, hi = self.delay_ms
        k = 2.0 if self.net_sign == "-" else 1.0
        return (1000.0 / (k * lo), 1000.0 / (k * hi))


def _L(id, name, summary, edges, site_tile=""):
    return Loop(id, name, summary, tuple(Edge(*e) for e in edges), site_tile)


LOOPS = {l.id: l for l in [
    _L("tct", "thalamo-cortico-thalamic",
       "The loop the whole brain integrates through: cortex drives thalamus, and thalamus "
       "drives cortex back.",
       [("md", "dlpfc", (1.0, 3.0), "+", "relay to layer 4, fast myelinated"),
        ("dlpfc", "trn", (3.0, 6.0), "+", "layer 6 corticothalamic collateral into the shell"),
        ("trn", "md", (1.0, 2.0), "-", "GABAergic; its decay, not its delay, sets the band")],
       "rz_tct"),

    _L("cbgtc", "cortico-basal-ganglia-thalamo-cortical",
       "Motor and premotor cortex propose; the striatum and pallidum release one proposal and "
       "suppress the rest; the thalamus returns the winner.",
       [("pmc", "striatum", (3.0, 8.0), "+"),
        ("striatum", "gpi", (3.0, 6.0), "-", "the direct arm"),
        ("gpi", "va_vl_bg", (2.0, 5.0), "-", "disinhibition is the release"),
        ("va_vl_bg", "pmc", (2.0, 4.0), "+")],
       "rz_bg"),

    _L("stn_gpe", "subthalamo-pallidal pacemaker",
       "The reciprocal STN-GPe pair, the fastest loop in the basal ganglia and the one whose "
       "synchrony is read out as pathological beta.",
       [("stn", "gpe", (2.0, 4.0), "+"),
        ("gpe", "stn", (3.0, 5.0), "-")]),

    _L("hpc", "hippocampal-entorhinal",
       "Entorhinal input rides a slow theta cycle, and each cycle is divided into faster gamma "
       "slots.",
       [("mtl_ctx", "dg", (3.0, 8.0), "+", "perforant path"),
        ("dg", "ca3", (2.0, 5.0), "+", "mossy fibres"),
        ("ca3", "ca1", (2.0, 4.0), "+", "Schaffer collaterals"),
        ("ca1", "sub", (2.0, 4.0), "+"),
        ("sub", "mtl_ctx", (3.0, 8.0), "+", "the return that closes the loop")],
       "rz_hpc"),

    _L("ca3_ca1", "CA3-CA1 local circuit",
       "The local recurrent circuit that a sharp wave detonates; too short to be anything but "
       "fast.",
       [("ca3", "ca1", (1.5, 3.0), "+"),
        ("ca1", "ca3", (1.0, 2.0), "-", "via local interneurons")]),

    _L("dan", "fronto-parietal dorsal attention",
       "Occipito-temporal cortex proposes what is worth looking at, parietal cortex holds the "
       "map it is proposed on, and frontal cortex commits to one place.",
       [("v_extra", "ips", (8.0, 15.0), "+"),
        ("ips", "fef", (8.0, 15.0), "+"),
        ("fef", "v_extra", (10.0, 20.0), "+", "the feedback that biases the map")],
       "rz_att"),

    _L("sal", "salience and medial frontal control",
       "The anterior insula and cingulate mark what matters and hand control the need to "
       "change course.",
       [("insula", "acc", (5.0, 12.0), "+"),
        ("acc", "dlpfc", (6.0, 14.0), "+"),
        ("dlpfc", "insula", (6.0, 14.0), "+")]),

    _L("val", "amygdala-vmPFC-accumbens valuation",
       "The amygdala scores what has just arrived, ventromedial prefrontal cortex and the "
       "anterior cingulate revise that score against context, and the accumbens turns it into a "
       "pull toward or away.",
       [("amygdala", "vmpfc", (8.0, 18.0), "+"),
        ("vmpfc", "nacc", (6.0, 14.0), "+"),
        ("nacc", "vta", (5.0, 12.0), "-"),
        ("vta", "amygdala", (10.0, 25.0), "+", "slow unmyelinated dopaminergic return")],
       "rz_lim"),

    _L("ctc", "cerebello-thalamo-cortical",
       "Motor cortex sends a copy of each command through the pons to the cerebellum, which "
       "returns a correction through the thalamus before the movement has finished.",
       [("m1", "pons", (5.0, 15.0), "+"),
        ("pons", "cb_ctx", (3.0, 8.0), "+", "mossy fibres"),
        ("cb_ctx", "dentate_n", (2.0, 5.0), "-", "Purkinje output is inhibitory"),
        ("dentate_n", "vl", (5.0, 10.0), "+"),
        ("vl", "m1", (2.0, 4.0), "+")],
       "rz_cbl"),

    _L("olivo", "olivo-cerebellar timing",
       "The inferior olive's electrically coupled cells fire together and stamp a clock on the "
       "cerebellar cortex, one complex spike at a time.",
       [("io", "cb_ctx", (3.0, 7.0), "+", "climbing fibres"),
        ("cb_ctx", "dentate_n", (2.0, 5.0), "-"),
        ("dentate_n", "io", (5.0, 12.0), "-", "the nucleo-olivary brake")]),

    _L("dmn", "default-mode network",
       "Posterior cingulate, precuneus, medial prefrontal and angular cortex rise together when "
       "nothing outside demands attention.",
       [("pcc", "vmpfc", (12.0, 25.0), "+"),
        ("vmpfc", "ang", (12.0, 25.0), "+"),
        ("ang", "pcc", (8.0, 18.0), "+")],
       "rz_dmn"),

    _L("aud", "auditory thalamocortical",
       "Cochlea, brainstem, medial geniculate and Heschl's gyrus, each station locking to the "
       "timing of the sound.",
       [("mgn", "a1", (1.0, 3.0), "+"),
        ("a1", "mgn", (3.0, 6.0), "+", "corticothalamic feedback")],
       "rz_aud"),

    _L("aud_afferent", "cochlea to cortex",
       "The ascending path, which is a delay line rather than a loop: it sets when a sound "
       "arrives, not what rhythm it arrives in.",
       [("cochlea", "mgn", (5.0, 12.0), "+", "through cochlear nucleus, olive, lemniscus, IC"),
        ("mgn", "a1", (1.0, 3.0), "+")]),

    _L("arousal", "brainstem-thalamic arousal",
       "The ascending arousal system sets how the thalamus gates the cortex, and so whether the "
       "brain is awake, drowsy or asleep.",
       [("lc", "trn", (10.0, 30.0), "-", "noradrenaline; slow unmyelinated"),
        ("trn", "md", (1.0, 2.0), "-"),
        ("md", "dlpfc", (1.0, 3.0), "+"),
        ("dlpfc", "lc", (15.0, 40.0), "+", "the cortical return onto the arousal nuclei")],
       "rz_aro"),

    _L("rem_flipflop", "REM-on / REM-off flip-flop",
       "Two mutually inhibiting brainstem populations; the brain is in one state or the other "
       "and crosses between them on a ninety-minute cycle.",
       [("ppt", "lc", (10.0, 30.0), "-"),
        ("lc", "ppt", (10.0, 30.0), "-")]),

    _L("rgs", "retino-geniculo-striate",
       "Retina to the lateral geniculate to striate cortex and out into the visual hierarchy, "
       "with the thalamus gating what gets through.",
       [("retina", "lgn", (5.0, 12.0), "+", "optic nerve"),
        ("lgn", "v1", (2.0, 5.0), "+"),
        ("v1", "lgn", (5.0, 12.0), "+", "layer 6 feedback"),
        ("lgn", "trn", (1.0, 2.0), "+"),
        ("trn", "lgn", (1.0, 2.0), "-", "the gate")],
       "rz_vis"),

    _L("cb_local", "Purkinje and molecular-layer interneurons",
       "The cerebellar cortex's own reciprocal circuit: Purkinje cells and the basket and "
       "stellate cells that silence them, a loop short enough to ring at 200 Hz.",
       [("cb_ctx", "cb_ctx", (0.5, 1.5), "+", "parallel fibre onto interneuron"),
        ("cb_ctx", "cb_ctx", (0.5, 1.5), "-", "basket cell onto the Purkinje soma")]),

    _L("ob_local", "mitral-granule dendrodendritic loop",
       "Mitral cells excite granule cells through dendrodendritic synapses and are inhibited "
       "back across the same contact: the olfactory bulb's gamma generator.",
       [("ob", "ob", (1.0, 2.5), "+", "mitral onto granule"),
        ("ob", "ob", (1.0, 2.5), "-", "granule back onto mitral")]),

    _L("v1_local", "local cortical excitation-inhibition",
       "Pyramidal cells excite fast interneurons and the interneurons silence them a few "
       "milliseconds later: the smallest loop in the cortex and the fastest.",
       [("v1", "v1", (1.0, 3.0), "+", "E to I"),
        ("v1", "v1", (2.0, 5.0), "-", "I back onto E; PING")]),

    _L("smu", "sensorimotor mu",
       "Sensorimotor cortex idles in a rhythm that breaks the moment a movement is planned, made "
       "or watched.",
       [("vpl", "s1", (1.0, 3.0), "+"),
        ("s1", "m1", (2.0, 5.0), "+"),
        ("m1", "vpl", (4.0, 10.0), "+"),
        ("vpl", "trn", (1.0, 2.0), "+"),
        ("trn", "vpl", (1.0, 2.0), "-")],
       "rz_mu"),

    _L("cms", "cortico-muscular-spinal",
       "Motor cortex drives the motoneuron pool, the muscle answers, and the spindle afferents "
       "carry the answer back -- the shortest loop that leaves the head.",
       [("m1", "mn_pool", (10.0, 16.0), "+", "corticospinal, to an upper-limb pool"),
        ("mn_pool", "muscle", (4.0, 8.0), "+", "peripheral nerve plus neuromuscular delay"),
        ("muscle", "spindle_afferent", (1.0, 3.0), "+"),
        ("spindle_afferent", "s1", (18.0, 25.0), "+", "dorsal column to cortex"),
        ("s1", "m1", (2.0, 5.0), "+")]),

    _L("stretch_reflex", "monosynaptic stretch reflex",
       "Spindle to motoneuron and straight back out: the loop that does not consult the brain.",
       [("spindle_afferent", "mn_pool", (10.0, 18.0), "+"),
        ("mn_pool", "muscle", (4.0, 8.0), "+"),
        ("muscle", "spindle_afferent", (1.0, 3.0), "+")]),

    _L("cpg", "spinal locomotor central pattern generator",
       "Half-centres in the cord that alternate flexor and extensor drive on their own, with the "
       "brainstem setting only how fast.",
       [("mlr", "cord_lumbar", (8.0, 20.0), "+", "the drive that sets frequency, not phase"),
        ("cord_lumbar", "cord_lumbar", (5.0, 15.0), "-", "reciprocal half-centre inhibition"),
        ("cord_lumbar", "mn_pool", (2.0, 5.0), "+")]),

    _L("intero", "gut-vagus-insula interoception",
       "The stomach's own pacemaker, carried up the vagus, read in the insula: a rhythm the "
       "brain does not generate and cannot ignore.",
       [("gut", "nts", (250.0, 500.0), "+", "unmyelinated vagal C fibres; IHM-1 measures 448 ms"),
        ("nts", "pb", (5.0, 15.0), "+"),
        ("pb", "vpl", (5.0, 15.0), "+"),
        ("vpl", "insula", (2.0, 6.0), "+")]),

    _L("resp", "respiratory-olfactory-limbic",
       "Breathing paces the olfactory bulb, and the bulb paces the structures behind it, so the "
       "body's slowest motor act shows up as a cortical rhythm.",
       [("preBotC", "lung", (20.0, 60.0), "+", "phrenic drive"),
        ("lung", "ob", (30.0, 80.0), "+", "airflow through the nose is the stimulus"),
        ("ob", "mtl_ctx", (10.0, 25.0), "+"),
        ("mtl_ctx", "ca1", (5.0, 12.0), "+")]),

    _L("circ", "suprachiasmatic circadian",
       "A transcriptional clock with a period near a day, which every other rhythm here is "
       "nested inside.",
       [("scn", "lc", (20.0, 60.0), "+"),
        ("lc", "scn", (20.0, 60.0), "+")]),
]}


# ======================================================================================
# the rhythms.
#
# `measure` is a spec `ibm/spectral.py`'s `spectral_loss` understands, and the same spec
# a measurement script runs against real data.  `target` is what we are shaping TOWARD,
# as an interval wherever the literature gives one.
# ======================================================================================
@dataclass(frozen=True)
class Rhythm:
    id: str
    name: str
    loop: str
    band: tuple | None              # (lo, hi) Hz; None for an entry that is not a band
    peak: float | None              # declared centre, when there is one
    states: tuple                   # behavioural states it is defined in
    set_by: str                     # "time constants" | "loop delay" | "external" | "network"
    mechanism: str
    measure: dict
    evidence: str                   # measured-here | measurable-now | literature | target
    substrate: str                  # expressible | needs-<what>
    refs: tuple = ()
    corpus: tuple = ()
    note: str = ""
    bound_exempt: str = ""          # a declared reason this rhythm is NOT its loop's resonance


def _R(id, name, loop, band, peak, states, set_by, mechanism, measure,
       evidence, substrate, refs=(), corpus=(), note="", bound_exempt=""):
    return Rhythm(id, name, loop, tuple(band) if band else None, peak, tuple(states),
                  set_by, mechanism,
                  dict(measure), evidence, substrate, tuple(refs), tuple(corpus), note,
                  bound_exempt)


RHYTHMS = [
    # ---------------------------------------------------------------- thalamo-cortical
    _R("slow_oscillation", "cortical slow oscillation", "tct", (0.5, 1.5), 0.8,
       ("nrem2", "nrem3", "anaesthesia"), "time constants",
       "Adaptation and synaptic depression drain an up state until the cortex falls silent, and "
       "recover until it ignites again. The period is the recovery time, not a delay.",
       dict(kind="peak_prominence", band=(0.5, 1.5), stations=("dlpfc", "pcc", "m1"),
            target=(0.3, 2.0)),
       "literature", "expressible",
       ("Steriade 1993", "Sanchez-Vives & McCormick 2000"), ("ds008037-rest",),
       "The one rhythm the v2 substrate can already produce unaided: tau_a runs 0.30-1.20 s "
       "across the hierarchy, which is the right range for a 0.5-1.5 Hz up/down cycle."),

    _R("delta", "thalamocortical delta", "tct", (1.0, 4.0), 2.0,
       ("nrem3",), "time constants",
       "The relay cell's own T-type calcium current and h-current alternate burst and pause when "
       "it is hyperpolarised; the cortex follows.",
       dict(kind="relative_power", band=(1.0, 4.0), stations=("dlpfc", "pcc"), target=(0.25, 0.55)),
       "literature", "needs-thalamus",
       ("McCormick & Pape 1990", "Steriade 1993")),

    _R("spindles", "sleep spindles", "tct", (11.0, 16.0), 13.45,
       ("nrem2",), "time constants",
       "The reticular nucleus inhibits the relay cells, they rebound through the T-current, and "
       "they drive the reticular nucleus again. The frequency is set by the GABA-B decay and the "
       "de-inactivation time of the T-current.",
       dict(kind="peak_frequency", band=(11.0, 16.0), stations=("dlpfc", "ips"), target=13.45),
       "measured-here", "needs-thalamus",
       ("Steriade 1993", "Luthi 2014"), ("sleep-edf", "ds008037-rest"),
       "MEASURED: scripts/fit_sleep_resonance.py fitted the declared alpha_resonator to N2 "
       "spectra split by subject; the centre moved off its 10 Hz prior to 13.452 Hz on held-out "
       "people, +5503.2 nats/night, p = 7.2e-11. The only fitted frequency in this table."),

    _R("so_spindle_coupling", "spindles nested in the slow oscillation", "tct", (11.0, 16.0), None,
       ("nrem2", "nrem3"), "network",
       "Spindles do not fall anywhere in the slow oscillation: they ride its up-state phase. The "
       "nesting is the mechanism memory consolidation is attributed to, so it is a target in its "
       "own right and not a by-product of getting both bands right.",
       dict(kind="pac", phase_band=(0.5, 1.5), amp_band=(11.0, 16.0),
            stations=("dlpfc", "pcc"), target=(0.02, 0.30)),
       "measurable-now", "needs-thalamus",
       ("Staresina 2015", "Helfrich 2018"), ("sleep-edf",),
       "Two right bands with no coupling between them is the failure mode this entry exists to "
       "catch: a model can score both power terms and still have them independent."),

    _R("k_complex", "K-complex", "tct", (0.5, 1.5), None,
       ("nrem2",), "time constants",
       "A single evoked down state: the slow oscillation's machinery fired once by a stimulus "
       "rather than free-running.",
       dict(kind="evoked_response", band=(0.5, 4.0), stations=("dlpfc",), target=None),
       "literature", "expressible",
       ("Cash 2009",), (),
       "An evoked event, not a band. It is here because a substrate that produces the slow "
       "oscillation but cannot be made to produce ONE on demand has the rhythm without the "
       "mechanism."),

    # ---------------------------------------------------------------- visual
    _R("alpha_occipital", "occipital alpha", "rgs", (8.0, 13.0), 10.0,
       ("wake-eyes-closed",), "loop delay",
       "The geniculo-cortical loop with the reticular gate closed on it. Eyes closed, the loop "
       "rings; eyes open, the retinal drive breaks it.",
       dict(kind="peak_prominence", band=(8.0, 13.0), stations=("v1", "v_extra"),
            target=(0.4, 1.5)),
       "measured-here", "needs-thalamus",
       ("Lopes da Silva 1991", "Klimesch 1999"), ("eegmmidb",),
       "MEASURED as a two-state contrast on 44 held-out subjects: the declared occipital alpha "
       "resonator beats a 1/f background by +201.6 and the eyes-open/eyes-closed contrast by "
       "+200.4; the relabelling control collapses. Moving the centre OFF its 10 Hz prior FAILED "
       "-- the prior is already where the data wants it.  Measured again on ds008037's "
       "119 resting subjects, EYES OPEN, which is the other state: peak 10.000 +/- 0.071 Hz, "
       "so the declared 10 Hz centre survives a second corpus and a second state. Prominence "
       "there is +0.413 +/- 0.036 decades with 46% of subjects inside this row's declared "
       "0.4-1.5 interval -- expected with the eyes open, and not a test of an eyes-closed "
       "declaration."),

    _R("alpha_reactivity", "alpha blocking on eye opening", "rgs", (8.0, 13.0), None,
       ("wake-eyes-open", "wake-eyes-closed"), "external",
       "The same loop in two states. Berger's observation, and the cleanest state contrast in "
       "electrophysiology.",
       dict(kind="state_contrast", band=(8.0, 13.0), stations=("v1", "v_extra"),
            states=("wake-eyes-closed", "wake-eyes-open"), target=(0.5, 5.0)),
       "measured-here", "needs-thalamus",
       ("Berger 1929", "Barry 2007"), ("eegmmidb",),
       "A ratio, and the reason this loop is worth more than a power target: a model can be "
       "given an alpha peak by hand, but it cannot be given one that switches off when the eyes "
       "open unless the gating is real.  The corpus is eegmmidb (R01 eyes-open, R02 "
       "eyes-closed); this row and the one above named ds008037-rest until 18 Sep 2026, and "
       "that corpus has no eyes-closed data at all -- caught by the run that measured it."),

    _R("alpha_travelling", "alpha as a travelling wave", "rgs", (8.0, 13.0), None,
       ("wake-eyes-closed",), "loop delay",
       "Alpha is not a standing hum: its phase sweeps across the cortical sheet, mostly "
       "occipital to frontal, at a few metres per second.",
       dict(kind="phase_gradient", band=(8.0, 13.0), stations=("v1", "v_extra", "ips", "dlpfc"),
            target=(2.0, 15.0), units="m/s"),
       "literature", "needs-thalamus",
       ("Halgren 2019", "Zhang 2018"), ("ds008037-rest",),
       "This is the term that makes the substrate's conduction delays load-bearing: a sheet with "
       "instantaneous coupling can have the band and cannot have the wave."),

    _R("ssvep", "steady-state visual entrainment", "rgs", (4.0, 60.0), None,
       ("wake-eyes-open",), "external",
       "A flickering stimulus drags the visual loop to its own frequency, and the response "
       "appears at that frequency and its harmonics.",
       dict(kind="entrainment", band=(4.0, 60.0), stations=("v1", "v_extra"), target=(0.3, 1.0)),
       "literature", "needs-thalamus",
       ("Norcia 2015",), (),
       "The calibration rhythm of this table. Drive at f, get a peak at f: any instrument or "
       "substrate that fails THIS is broken in a way that has nothing to do with physiology."),

    _R("gamma_visual", "visual gamma", "v1_local", (30.0, 80.0), 50.0,
       ("wake-eyes-open",), "loop delay",
       "Pyramidal cells excite fast interneurons, the interneurons silence them, and the cycle "
       "is the interneuron's decay. Frequency rises with contrast: it is a property of the "
       "drive, not a fixed clock.",
       dict(kind="peak_prominence", band=(30.0, 80.0), stations=("v1", "v_extra"),
            target=(0.2, 1.2)),
       "literature", "expressible",
       ("Buzsaki & Wang 2012", "Ray & Maunsell 2010"), (),
       "The E/I loop that produces it is exactly what `ibm/substrate.py` is, so this is the "
       "second rhythm the cortical sheet could own today -- but tau_I is 5 ms, which puts the "
       "natural PING frequency near 60-80 Hz, and nothing has checked where it actually lands."),

    # ---------------------------------------------------------------- sensorimotor
    _R("mu", "sensorimotor mu", "smu", (8.0, 13.0), 10.0,
       ("wake-rest",), "loop delay",
       "The somatosensory arm of the same thalamocortical machinery that makes occipital alpha, "
       "idling over the central sulcus.",
       dict(kind="peak_prominence", band=(8.0, 13.0), stations=("s1", "m1"), target=(0.2, 1.0)),
       "literature", "needs-thalamus",
       ("Pfurtscheller 1999", "Hari 2006"), ("ds008037-rest",),
       "FAILED once already, and the failure was the DECLARATION, not the data: the declared "
       "resonator was a low-pass form that cannot fit a small peak on a steep background. A "
       "band-pass declaration is owed a fresh pre-registration before it is fitted again.  AND "
       "IT MUST NOT BE PROMOTED ON A SCALP MEASUREMENT: over ds008037's 119 resting subjects "
       "the central 8-13 Hz prominence (+0.476 +/- 0.033) correlates with the occipital one at "
       "r = +0.836 and the two peak frequencies at r = +0.732, and the paired difference runs "
       "the WRONG way -- central exceeds occipital by 0.063 +/- 0.020 decades, which is what "
       "volume conduction with an average reference predicts.  A central alpha peak is not "
       "evidence of a separable mu generator; separating it needs a spatial filter or the "
       "movement contrast `mu_erd` declares."),

    _R("mu_erd", "mu breaking on movement", "smu", (8.0, 13.0), None,
       ("movement", "movement-imagery", "action-observation"), "external",
       "The rhythm desynchronises before the muscle moves, and on watching someone else move.",
       dict(kind="state_contrast", band=(8.0, 13.0), stations=("s1", "m1"),
            states=("wake-rest", "movement"), target=(1.3, 4.0)),
       "literature", "needs-thalamus",
       ("Pfurtscheller & Lopes da Silva 1999",), (),
       "The goal is that preparing to act shows in the loop before any muscle moves -- which "
       "means this entry cannot be scored until the substrate is attached to a body that can be "
       "about to move."),

    _R("beta_rebound", "post-movement beta rebound", "smu", (15.0, 30.0), 20.0,
       ("movement",), "network",
       "Beta collapses during the movement and overshoots its baseline for a second afterwards.",
       dict(kind="state_contrast", band=(15.0, 30.0), stations=("m1", "s1"),
            states=("movement", "post-movement"), target=(1.2, 3.0)),
       "literature", "needs-body", ("Jurkiewicz 2006",)),

    _R("corticomuscular_beta", "corticomuscular coherence", "cms", (15.0, 30.0), 20.0,
       ("steady-contraction",), "loop delay",
       "During a held contraction the motor cortex and the muscle it drives share a beta rhythm, "
       "and the coherence between cortex and EMG is the most direct evidence that the loop is "
       "closed.",
       dict(kind="coherence", band=(15.0, 30.0), stations=("m1", "muscle"), target=(0.05, 0.40)),
       "literature", "needs-body",
       ("Conway 1995", "Salenius 1997", "Baker 2007"), (),
       "The round-trip bound nearly binds here and it is worth saying out loud: the loop is "
       "~35-57 ms and net excitatory, so f <= 18-29 Hz. Beta is where the loop CAN ring."),

    _R("piper", "Piper rhythm", "cms", (30.0, 60.0), 40.0,
       ("strong-contraction",), "loop delay",
       "The same coupling at a higher band during strong contractions.",
       dict(kind="coherence", band=(30.0, 60.0), stations=("m1", "muscle"), target=(0.03, 0.30)),
       "literature", "needs-body", ("Brown 1998",), (),
       "Sits ABOVE the cortico-muscular round-trip bound, so whatever it is, it is not that "
       "loop resonating.",
       bound_exempt="40 Hz needs a round trip under 25 ms and the cortico-muscular loop is "
                    "35-57 ms. Declared as an exception rather than dropped: the coupling is "
                    "measured, so the loop it is assigned to here is wrong, probably a shorter "
                    "subcortical or intraspinal path we have not written down."),

    _R("physiological_tremor", "physiological tremor", "ctc", (8.0, 12.0), 10.0,
       ("posture",), "loop delay",
       "A limb held against gravity oscillates at around 10 Hz, and the olivo-cerebellar clock "
       "is one of the things that sets it.",
       dict(kind="peak_prominence", band=(8.0, 12.0), stations=("muscle",), target=(0.2, 1.5)),
       "literature", "needs-body", ("Elble & Koller 1990", "Llinas 1988")),

    # ---------------------------------------------------------------- basal ganglia
    _R("beta_bg", "basal-ganglia beta", "stn_gpe", (13.0, 30.0), 20.0,
       ("wake-rest", "hold"), "loop delay",
       "The subthalamo-pallidal pair is a delayed inhibitory loop with a natural period in the "
       "beta range, and cortex entrains it. High beta means hold; movement suppresses it.",
       dict(kind="peak_prominence", band=(13.0, 30.0), stations=("stn", "gpe"), target=(0.3, 1.5)),
       "literature", "needs-basal-ganglia",
       ("Brown 2001", "Kuhn 2006", "Jenkinson & Brown 2011"), (),
       "The bound: 5-9 ms net inhibitory gives f <= 56-100 Hz, so the delay does not set beta "
       "here -- the synaptic time constants do. Worth knowing before anyone tunes a delay to get "
       "the band."),

    _R("beta_bursts", "beta arrives in bursts, not as a tone", "stn_gpe", (13.0, 30.0), None,
       ("wake-rest", "hold"), "network",
       "Averaged beta power is an artefact of averaging: the real signal is bursts of 100-500 ms "
       "whose RATE and DURATION carry the state, and long bursts are what correlates with "
       "slowness in Parkinson's disease.",
       dict(kind="burst_statistics", band=(13.0, 30.0), stations=("stn", "m1"),
            target_duration_ms=(100.0, 500.0), target=(0.1, 0.5)),
       "literature", "needs-basal-ganglia",
       ("Tinkhauser 2017", "Feingold 2015"), (),
       "The entry that stops a power target from being satisfied the wrong way: a constant "
       "13-30 Hz tone scores the same band power as real bursting and is not the same object."),

    _R("gamma_striatal", "prokinetic striatal gamma", "cbgtc", (60.0, 90.0), 75.0,
       ("movement",), "network",
       "A fast rhythm that appears in the striatum and subthalamic nucleus at movement onset -- "
       "the opposite polarity to beta.",
       dict(kind="state_contrast", band=(60.0, 90.0), stations=("striatum", "stn"),
            states=("wake-rest", "movement"), target=(1.2, 4.0)),
       "literature", "needs-basal-ganglia", ("Jenkinson 2013",)),

    _R("action_selection", "one proposal released, the rest suppressed", "cbgtc", None, None,
       ("movement",), "network",
       "Not a band at all: the loop's actual job. Several cortical proposals arrive, the "
       "pallidum's tonic inhibition lifts off exactly one, and the thalamus returns it.",
       dict(kind="selection_contrast", stations=("pmc", "striatum", "gpi", "va_vl_bg"),
            target=(0.8, 1.0)),
       "target", "needs-basal-ganglia", ("Redgrave 1999", "Mink 1996"),
       note="Here to keep the table honest: a loop is specified by what it computes, not only "
            "by the frequency it hums at, and a model that reproduced every band in this file "
            "while selecting nothing would have missed the point."),

    # ---------------------------------------------------------------- hippocampus
    _R("theta_hpc", "hippocampal theta", "hpc", (3.0, 8.0), 6.0,
       ("movement", "exploration", "rem"), "network",
       "The septal pacemaker and the entorhinal-hippocampal loop together; in humans it is "
       "slower and burstier than the rodent's continuous 8 Hz.",
       dict(kind="peak_prominence", band=(3.0, 8.0), stations=("ca1", "ca3", "mtl_ctx"),
            target=(0.3, 1.5)),
       "literature", "needs-hippocampal-subfields",
       ("Buzsaki 2002", "Jacobs 2014"), (),
       "The bound is instructive: the entorhinal ring is 12-29 ms excitatory, f <= 34-83 Hz. "
       "Theta is nowhere near it, which is why the septal pacemaker is in the table -- the ring "
       "alone would ring far too fast."),

    _R("theta_gamma", "gamma nested in theta", "hpc", (30.0, 100.0), None,
       ("movement", "encoding"), "network",
       "Each theta cycle is divided into gamma slots, and what fires in which slot is the order "
       "of the episode. This is the entry the whole loop is for: sequence carried by WHEN, not "
       "stored beside the item as an index.",
       dict(kind="pac", phase_band=(3.0, 8.0), amp_band=(30.0, 100.0),
            stations=("ca1", "ca3"), target=(0.02, 0.40)),
       "target", "needs-hippocampal-subfields",
       ("Lisman & Jensen 2013", "Tort 2009", "Colgin 2009")),

    _R("slow_gamma_ca1", "CA3-driven slow gamma", "ca3_ca1", (25.0, 55.0), 40.0,
       ("retrieval",), "loop delay",
       "The slow gamma CA1 shows when CA3 is driving it -- the retrieval mode.",
       dict(kind="peak_prominence", band=(25.0, 55.0), stations=("ca1",), target=(0.2, 1.0)),
       "literature", "needs-hippocampal-subfields", ("Colgin 2009",)),

    _R("fast_gamma_ca1", "entorhinal-driven fast gamma", "hpc", (60.0, 100.0), 80.0,
       ("encoding",), "loop delay",
       "The faster gamma CA1 shows when the entorhinal cortex is driving it -- the encoding "
       "mode. Two gammas, two sources, one region: the pair is the evidence that the region is "
       "switching between inputs rather than running one oscillator.",
       dict(kind="peak_prominence", band=(60.0, 100.0), stations=("ca1",), target=(0.2, 1.0)),
       "literature", "needs-hippocampal-subfields", ("Colgin 2009",)),

    _R("ripples", "sharp-wave ripples", "ca3_ca1", (140.0, 200.0), 170.0,
       ("quiet-wake", "nrem2", "nrem3"), "loop delay",
       "A CA3 population burst detonates CA1 and the local interneurons ring at 150-200 Hz for "
       "50-100 ms. Sequences replay inside one, compressed twentyfold.",
       dict(kind="peak_prominence", band=(140.0, 200.0), stations=("ca1",), target=(0.3, 2.0)),
       "target", "needs-hippocampal-subfields",
       ("Buzsaki 2015", "Csicsvari 1999"), (),
       "The fastest entry in the table, and the bound is the reason it is possible at all: the "
       "CA3-CA1 local loop is 2.5-5 ms net inhibitory, f <= 100-200 Hz. It only just fits, which "
       "is the point -- nothing longer could produce a ripple."),

    _R("ripple_spindle_so", "ripples inside spindles inside slow oscillations", "hpc",
       (140.0, 200.0), None,
       ("nrem2", "nrem3"), "network",
       "The triple nesting: hippocampal ripples fall inside the troughs of thalamic spindles, "
       "which fall on the up state of the cortical slow oscillation. Three loops, three bands, "
       "one hierarchy -- the mechanism consolidation is attributed to.",
       dict(kind="pac_chain", chain=(((0.5, 1.5), (11.0, 16.0)), ((11.0, 16.0), (140.0, 200.0))),
            stations=("dlpfc", "ca1"), target=(0.02, 0.40)),
       "target", "needs-thalamus+hippocampal-subfields",
       ("Staresina 2015", "Latchoumane 2017"), (),
       "The most demanding entry here, and the one that would say most: it cannot be satisfied "
       "by tuning any single loop."),

    _R("phase_precession", "phase precession", "hpc", (3.0, 8.0), None,
       ("movement",), "network",
       "A place cell fires progressively earlier in the theta cycle as the animal crosses its "
       "field, so position is read out from PHASE rather than from rate.",
       dict(kind="phase_position_slope", band=(3.0, 8.0), stations=("ca1",),
            target=(-360.0, -90.0), units="degrees per field"),
       "target", "needs-hippocampal-subfields+body",
       ("O'Keefe & Recce 1993", "Mehta 2002"), (),
       "Needs a body that goes somewhere, which is the whole programme's target, so this entry "
       "is a milestone as much as a rhythm."),

    # ---------------------------------------------------------------- attention, control
    _R("attention_sampling", "attention samples rhythmically", "dan", (4.0, 8.0), 7.0,
       ("wake-task",), "network",
       "Sustained attention is not sustained: behavioural performance at an attended location "
       "waxes and wanes at 4-8 Hz, in antiphase between two objects.",
       dict(kind="behavioural_rhythm", band=(4.0, 8.0), stations=("fef", "ips"),
            target=(0.03, 0.30)),
       "target", "needs-task+body", ("Landau & Fries 2012", "Fiebelkorn 2018")),

    _R("alpha_lateralisation", "alpha takes sides", "dan", (8.0, 13.0), None,
       ("wake-task",), "network",
       "Attend left and alpha rises in the left hemisphere, gating out the ignored side. The "
       "goal is that visual salience is a competition running inside the substrate, not a "
       "saliency map bolted on.",
       dict(kind="lateralisation", band=(8.0, 13.0), stations=("ips", "v_extra"),
            target=(0.05, 0.60)),
       "measurable-now", "needs-thalamus",
       ("Worden 2000", "Thut 2006"), ("ds008037",),
       "ds008037's working-memory task is lateralised, so this is measurable on data already on "
       "disk -- unlike the single-trial decoding the withdrawn step 2 attempted, a lateralisation "
       "index is a within-subject contrast and does not need trial-level stimulus information."),

    _R("frontal_midline_theta", "frontal midline theta", "sal", (4.0, 8.0), 6.0,
       ("wake-task", "working-memory"), "network",
       "Medial frontal cortex runs a theta rhythm that grows with how much is being held and "
       "with how much control is needed; it is the clearest cortical signature of effort.",
       dict(kind="load_slope", band=(4.0, 8.0), stations=("acc", "dlpfc"), target=(0.02, 0.50)),
       "measurable-now", "expressible",
       ("Cavanagh & Frank 2014", "Jensen & Tesche 2002"), ("ds008037",),
       "MEASURABLE ON WHAT WE HOLD: ds008037 has 111 subjects, set sizes 2/4/6, and the extracted "
       "delay-period features. Theta power against set size is a within-subject slope -- exactly "
       "the kind of contrast that survives where trial-level decoding did not."),

    _R("error_theta", "error and feedback theta", "sal", (4.0, 8.0), 6.0,
       ("wake-task",), "network",
       "A burst of medial frontal theta follows an error or a worse-than-expected outcome, "
       "within about 200 ms.",
       dict(kind="evoked_band", band=(4.0, 8.0), stations=("acc",), window_s=(0.0, 0.5),
            target=(0.1, 1.0)),
       "measurable-now", "expressible",
       ("Cavanagh 2012",), ("ds008037",)),

    _R("gamma_attention", "attentional gamma", "dan", (40.0, 70.0), None,
       ("wake-task",), "network",
       "Attending to a stimulus raises gamma coherence between the areas representing it -- "
       "communication through coherence.",
       dict(kind="coherence", band=(40.0, 70.0), stations=("v_extra", "ips"), target=(0.05, 0.40)),
       "target", "expressible", ("Fries 2015", "Bosman 2012")),

    # ---------------------------------------------------------------- valuation, limbic
    _R("reward_theta", "feedback theta and delta", "val", (2.0, 8.0), 5.0,
       ("wake-task",), "network",
       "A reward or its absence is followed by a low-frequency response over medial frontal "
       "cortex whose size tracks how surprising the outcome was.",
       dict(kind="evoked_band", band=(2.0, 8.0), stations=("acc", "vmpfc"), window_s=(0.2, 0.6),
            target=(0.1, 1.0)),
       "literature", "expressible", ("Cavanagh & Frank 2014", "Holroyd & Coles 2002")),

    _R("amygdala_theta", "amygdala-prefrontal theta coherence", "val", (4.0, 8.0), None,
       ("threat", "wake-task"), "network",
       "Under threat, the amygdala and ventromedial prefrontal cortex lock in theta, and which "
       "one leads is the difference between being gripped by a thing and appraising it.",
       dict(kind="coherence", band=(4.0, 8.0), stations=("amygdala", "vmpfc"), target=(0.05, 0.50)),
       "target", "needs-limbic", ("Likhtik 2014",),
       note="The goal is that value is a state of a brain attached to a body, not a scalar "
            "handed in from outside it."),

    _R("nacc_gamma", "accumbens gamma", "val", (50.0, 100.0), None,
       ("reward",), "network",
       "Fast oscillations in the accumbens that shift band with approach and consumption.",
       dict(kind="peak_prominence", band=(50.0, 100.0), stations=("nacc",), target=(0.1, 1.0)),
       "target", "needs-limbic", ("van der Meer & Redish 2009",)),

    # ---------------------------------------------------------------- cerebellum
    _R("purkinje_fast", "Purkinje simple-spike synchrony", "cb_local", (160.0, 250.0), 200.0,
       ("wake",), "network",
       "Neighbouring Purkinje cells fire simple spikes in near-lockstep at 200 Hz, which is how "
       "the cerebellar cortex carries fine timing rather than rate.",
       dict(kind="peak_prominence", band=(160.0, 250.0), stations=("cb_ctx",), target=(0.2, 1.5)),
       "literature", "needs-cerebellum", ("de Solages 2008", "De Zeeuw 2011")),

    _R("olivary_clock", "olivary complex-spike rhythm", "olivo", (5.0, 10.0), 8.0,
       ("wake", "movement"), "time constants",
       "The inferior olive's cells are electrically coupled and oscillate subthreshold near "
       "10 Hz, so complex spikes arrive on a clock rather than at random.",
       dict(kind="peak_prominence", band=(5.0, 10.0), stations=("io",), target=(0.2, 1.5)),
       "literature", "needs-cerebellum", ("Llinas & Yarom 1986", "Welsh 1995")),

    _R("cerebello_cortical", "cerebello-cortical coherence", "ctc", (13.0, 30.0), None,
       ("posture", "movement"), "loop delay",
       "During steady posture the cerebellum and motor cortex share beta, and the correction "
       "arrives before the movement has finished. The goal is timing that comes from the loop, "
       "not from a tuned controller.",
       dict(kind="coherence", band=(13.0, 30.0), stations=("cb_ctx", "m1"), target=(0.05, 0.40)),
       "target", "needs-cerebellum", ("Gross 2002",)),

    # ---------------------------------------------------------------- default mode, arousal
    _R("dmn_infraslow", "default-mode infraslow coupling", "dmn", (0.01, 0.1), 0.05,
       ("wake-rest",), "network",
       "Posterior cingulate, medial prefrontal and angular cortex rise and fall together over "
       "tens of seconds when nothing outside demands attention. The goal is a resting brain with "
       "structure -- recollection and self-reference -- rather than a model that idles.",
       dict(kind="coherence", band=(0.01, 0.1), stations=("pcc", "vmpfc", "ang"),
            target=(0.3, 0.9)),
       "literature", "expressible",
       ("Raichle 2001", "Fox 2005"), (),
       "Expressible in principle and expensive in practice: 0.01 Hz needs runs of several "
       "minutes of simulated time to have any frequency resolution at all."),

    _R("infraslow_arousal", "infraslow arousal fluctuation", "arousal", (0.01, 0.1), None,
       ("wake-rest", "nrem1"), "network",
       "A global fluctuation in excitability, locked to pupil size and to noradrenergic tone.",
       dict(kind="relative_power", band=(0.01, 0.1), stations=("dlpfc", "pcc", "insula"),
            target=(0.05, 0.40)),
       "literature", "needs-neuromodulators", ("Aston-Jones & Cohen 2005", "Raut 2021")),

    _R("ultradian_rem", "the REM/NREM cycle", "rem_flipflop", (1.5e-4, 2.5e-4), 1.85e-4,
       ("sleep",), "network",
       "Two mutually inhibiting brainstem populations hold the brain in REM or in NREM and flip "
       "between them roughly every ninety minutes. The goal is that sleep and waking are states "
       "the substrate moves between, not modes chosen by a flag.",
       dict(kind="state_cycle", period_s=(4200.0, 6600.0), stations=("ppt", "lc"),
            target=(0.5, 1.0)),
       "target", "needs-neuromodulators", ("Saper 2001", "Lu 2006")),

    _R("pgo_waves", "ponto-geniculo-occipital waves", "arousal", (0.5, 2.0), None,
       ("rem",), "network",
       "Bursts that leave the pons, pass through the geniculate and reach occipital cortex "
       "during REM: the visual system driven from inside.",
       dict(kind="evoked_band", band=(0.5, 4.0), stations=("lgn", "v1"), window_s=(0.0, 0.5),
            target=(0.1, 1.0)),
       "target", "needs-thalamus+brainstem", ("Datta 1997",)),

    _R("circadian", "the circadian cycle", "circ", (1.0e-5, 1.3e-5), 1.157e-5,
       ("all",), "time constants",
       "A transcriptional clock with a period near 24 hours, which every other rhythm in this "
       "table is nested inside.",
       dict(kind="state_cycle", period_s=(82800.0, 90000.0), stations=("scn",), target=(0.5, 1.0)),
       "target", "needs-hypothalamus", ("Reppert & Weaver 2002",),
       note="Out of reach of any simulation that runs for seconds, and in the table because a "
            "brain without it has no reason to sleep."),

    # ---------------------------------------------------------------- auditory, speech
    _R("assr_40", "40 Hz auditory steady-state response", "aud", (38.0, 42.0), 40.0,
       ("wake",), "loop delay",
       "Click trains near 40 Hz get a disproportionate response: the thalamocortical auditory "
       "loop's own resonance.",
       dict(kind="entrainment", band=(38.0, 42.0), stations=("a1",), target=(0.3, 1.0)),
       "literature", "needs-thalamus",
       ("Galambos 1981", "Picton 2003"), (),
       "I first wrote this entry claiming the conduction budget PREDICTS 40 Hz, by counting the "
       "ring twice to make the arithmetic come out. It does not: the A1-MGN ring is 4-9 ms, so "
       "the bound is 111-250 Hz and nowhere near binding. The 40 Hz preference is set by "
       "synaptic and rebound time constants, like most of this table. The bound is a ceiling, "
       "never a prediction, and the first draft of this file is what happens when that is "
       "forgotten."),

    _R("speech_envelope", "cortical tracking of the speech envelope", "aud_afferent", (1.0, 8.0),
       None, ("listening",), "external",
       "Auditory cortex follows the slow amplitude envelope of speech, at the rate of syllables. "
       "The goal is that speech is followed by a loop that entrains to it, not by a spectrogram "
       "handed in from outside.",
       dict(kind="coherence", band=(1.0, 8.0), stations=("a1", "stg"), target=(0.05, 0.40)),
       "target", "needs-thalamus", ("Ding & Simon 2014", "Gross 2013")),

    _R("auditory_gamma", "auditory cortical gamma", "aud", (30.0, 80.0), None,
       ("listening",), "loop delay",
       "The local E/I loop in auditory cortex, the same machinery as visual gamma with a "
       "different input.",
       dict(kind="peak_prominence", band=(30.0, 80.0), stations=("a1",), target=(0.1, 1.0)),
       "literature", "expressible", ("Brosch 2002",)),

    # ---------------------------------------------------------------- body, cord, viscera
    _R("locomotor_cpg", "the locomotor rhythm", "cpg", (0.5, 3.0), 1.2,
       ("locomotion",), "network",
       "Half-centres in the cord alternate flexor and extensor drive with no rhythmic input at "
       "all; the brainstem drive sets only how fast. The crawling body this programme is for "
       "moves because of this loop or it does not move.",
       dict(kind="peak_prominence", band=(0.5, 3.0), stations=("cord_lumbar", "mn_pool"),
            target=(0.5, 3.0)),
       "target", "needs-cord",
       ("Grillner 2006", "Kiehn 2016"), (),
       "The frequency must be a MONOTONIC function of the descending drive with the phase "
       "relationship unchanged -- that is the actual specification, not the band."),

    _R("flexor_extensor_antiphase", "flexor and extensor in antiphase", "cpg", (0.5, 3.0), None,
       ("locomotion",), "network",
       "The two half-centres are half a cycle apart, and left and right are too. A pattern "
       "generator that produced the right frequency in phase would be no pattern generator.",
       dict(kind="phase_difference", band=(0.5, 3.0), stations=("cord_lumbar",),
            target=(150.0, 210.0), units="degrees"),
       "target", "needs-cord", ("Kiehn 2016",)),

    _R("gastric", "the gastric rhythm", "intero", (0.045, 0.055), 0.05,
       ("wake-rest",), "external",
       "The stomach's interstitial cells of Cajal fire at three cycles a minute whatever the "
       "brain does, and a network of cortical regions is phase-locked to them.",
       dict(kind="coherence", band=(0.045, 0.055), stations=("insula", "s1"), target=(0.1, 0.6)),
       "target", "needs-body",
       ("Rebollo 2018", "Richter 2017"), (),
       "IBM-1 already declares the vagal path this rides: 448 ms on unmyelinated C fibres, "
       "re-measured after the relay fix. The rhythm is the body's, not the brain's, which is "
       "exactly why it belongs in a brain built to be attached to one."),

    _R("heartbeat_evoked", "the heartbeat-evoked response", "intero", (1.0, 1.5), None,
       ("wake-rest",), "external",
       "Every heartbeat produces a cortical response in insula and somatosensory cortex, whose "
       "size tracks how much attention is on the body.",
       dict(kind="evoked_band", band=(1.0, 20.0), stations=("insula", "s1"),
            window_s=(0.2, 0.5), target=(0.05, 0.5)),
       "target", "needs-body", ("Park 2014", "Al 2020")),

    _R("respiration_entrained", "breathing paces the brain", "resp", (0.15, 0.4), 0.25,
       ("wake-rest", "sleep"), "external",
       "Nasal airflow drives the olfactory bulb and, behind it, entorhinal cortex, hippocampus "
       "and amygdala; memory and reaction time vary with the phase of the breath.",
       dict(kind="coherence", band=(0.15, 0.4), stations=("mtl_ctx", "ca1", "amygdala"),
            target=(0.1, 0.6)),
       "target", "needs-body", ("Zelano 2016", "Herrero 2018")),

    _R("olfactory_gamma", "olfactory bulb gamma", "ob_local", (40.0, 100.0), 70.0,
       ("sniffing",), "loop delay",
       "Mitral cells and granule cells form a reciprocal loop that rings at gamma on each "
       "inhalation.",
       dict(kind="peak_prominence", band=(40.0, 100.0), stations=("ob",), target=(0.2, 1.5)),
       "target", "needs-olfactory-bulb", ("Kay 2009",)),

    _R("beta_posterior_scalp", "posterior resting beta at the scalp", "v1_local",
       (13.0, 30.0), 16.2,
       ("wake-rest", "wake-eyes-open"), "network",
       "A beta peak over posterior cortex at rest, standing above its own 1/f background. "
       "What generates it is not identified, and filing it under the local cortical E/I loop "
       "is a placement, not a claim: this row is a MEASURED SCALP CONSTRAINT that any future "
       "cortical beta claim has to beat.",
       dict(kind="peak_prominence", band=(13.0, 30.0), stations=("v_extra", "ips"),
            target=(0.0, 0.45)),
       "measured-here", "expressible",
       ("this repo, scripts/measure_rhythms_eeg.py",), ("ds008037-rest",),
       "MEASURED on 119 subjects of ds008037 resting EEG, split 59 declaration / 60 held out: "
       "prominence +0.178 +/- 0.020 decades, peak 16.18 +/- 0.19 Hz, relative power "
       "0.122 +/- 0.007. The declared interval is the declaration half's 10-90 range and 70% "
       "of HELD-OUT subjects fall inside it against an expectation of 80 +/- 7.3%, so it is "
       "if anything slightly narrow. The catalogue had no row for scalp beta at all -- its "
       "only 13-30 Hz resting row is `beta_bg` in the subthalamic nucleus, and a scalp "
       "measurement must never be used to confirm that one."),

    _R("stretch_reflex_resonance", "the stretch reflex's own resonance", "stretch_reflex",
       (6.0, 12.0), 9.0,
       ("posture",), "loop delay",
       "The monosynaptic loop from spindle to motoneuron and back through the muscle has a "
       "round trip near 20-30 ms, which is a resonance whether or not anything wants one -- it "
       "is one of the things a limb's tremor is made of.",
       dict(kind="peak_prominence", band=(6.0, 12.0), stations=("muscle",), target=(0.1, 1.0)),
       "literature", "needs-body", ("Lippold 1970", "Matthews 1997")),
]

RHYTHM = {r.id: r for r in RHYTHMS}

# behavioural states a rhythm can be declared in.  a state is a CONTEXT, not a flag the
# model reads: `arousal` and `rem_flipflop` are in the table precisely so that a state is
# something the substrate is in rather than something it is told.
STATES = ("wake", "wake-rest", "wake-eyes-closed", "wake-eyes-open", "wake-task", "working-memory",
          "listening", "sniffing", "movement", "movement-imagery", "action-observation",
          "post-movement", "steady-contraction", "strong-contraction", "posture", "locomotion",
          "hold", "encoding", "retrieval", "reward", "threat", "quiet-wake",
          "nrem1", "nrem2", "nrem3", "rem", "sleep", "anaesthesia", "exploration", "all")


# ======================================================================================
# the background.  not a rhythm, and shaping toward it matters as much as any peak: real
# cortical spectra are a 1/f^x background with a few peaks ON it, and a model that gets
# every band's RELATIVE power right on top of a white background has the wrong spectrum.
# The exponent is also read as an excitation/inhibition ratio (Gao 2017), which makes it
# the one spectral quantity that speaks directly to what `ibm/substrate.py` parameterises.
# ======================================================================================
BACKGROUND = {
    "id": "aperiodic",
    "name": "the 1/f background",
    "fit_band": (1.0, 45.0),
    "exclude": ((8.0, 13.0), (0.5, 1.5)),   # do not let the alpha or SO peak bend the fit
    "exponent_target": (0.73, 1.80),        # MEASURED, see `measured` below
    "evidence": "measured-here",
    "refs": ("Gao 2017", "Donoghue 2020", "this repo, scripts/measure_rhythms_eeg.py"),
    "measured": {
        "corpus": "ds008037-rest, 119 subjects, awake eyes open",
        "split": "59 declaration / 60 held out, one seeded draw",
        "global_exponent": (1.224, 0.037),       # mean, SE over subjects
        "interval_from_declaration_half": (0.729, 1.803),
        "held_out_coverage": 0.817,              # against an expectation of 0.80 +/- 0.073
        "by_group": {"occipital": (1.326, 0.040), "sensorimotor": (1.394, 0.027),
                     "frontal_midline": (1.542, 0.026)},
        "method": "average reference, 4 s Hann segments at 50% overlap, linear-bin OLS over "
                  "1-45 Hz with 8-13 and 45-55 excluded",
        "caveat": "the fit weights linear bins uniformly, so 13-45 Hz supplies 128 of the 176 "
                  "bins and the headline is mostly the high-frequency slope: 1.274 +/- 0.043 "
                  "over 1-20 Hz against 1.332 +/- 0.064 over 20-45 Hz. Not comparable to a "
                  "published exponent fitted on log-spaced bins.",
    },
    "note": "Steeper under anaesthesia and in NREM, shallower with arousal, so the target is "
            "state-dependent and this is the awake resting one. It was (0.8, 2.0) from the "
            "literature until 18 Sep 2026; the measured declaration half supports "
            "(0.73, 1.80), slightly lower and narrower. The untrained cortical field already "
            "sits at 1.5-1.6 at its healthy operating point, inside both.",
}


# ======================================================================================
# queries
# ======================================================================================
def by_state(state: str):
    """every rhythm declared in this behavioural state."""
    return [r for r in RHYTHMS if state in r.states or "all" in r.states]


def by_loop(loop_id: str):
    return [r for r in RHYTHMS if r.loop == loop_id]


def by_evidence(kind: str):
    return [r for r in RHYTHMS if r.evidence == kind]


def expressible(rhythms=None):
    """the rhythms a cortex-only substrate could be scored on today."""
    return [r for r in (rhythms or RHYTHMS) if r.substrate == "expressible"]


def labels_of(station_ids) -> list:
    """every atlas label the given stations resolve to, deduplicated, in order."""
    out = []
    for sid in station_ids:
        for lab in STATIONS[sid].labels:
            if lab not in out:
                out.append(lab)
    return out


def station_sites(station_ids, region_names):
    """indices into a substrate's site list that belong to these stations.

    `region_names` is `CorticalField.region_names` -- one atlas label per site, possibly
    hemisphere-prefixed ('lh.insula').  Returns a plain list of ints, so the caller can
    make whatever tensor it wants on whatever device.
    """
    want = set(labels_of(station_ids))
    return [i for i, n in enumerate(region_names) if str(n).split(".", 1)[-1] in want]


# ======================================================================================
# turning a catalogue row into a training target
#
# The output is deliberately NOT `ibm/spectral.py`'s own dict: it is an intermediate with
# site indices resolved, so that a trainer, an evaluator on real EEG, or a different loss
# can each consume the same rows.  `skipped` is returned, never swallowed -- a target that
# silently disappears because its kind is unsupported is how a run ends up optimising
# three terms while its log claims eight.
# ======================================================================================
SCORABLE_KINDS = ("relative_power", "peak_prominence", "peak_frequency", "pac", "coherence")

# what a substrate has.  A row's `substrate` field is either "expressible" (cortex alone) or
# "needs-a+b", so the requirement is readable rather than a free-text note nothing can act on.
STRUCTURES = ("cortex", "thalamus", "basal-ganglia", "hippocampal-subfields", "cerebellum",
              "limbic", "neuromodulators", "hypothalamus", "olfactory-bulb", "brainstem",
              "cord", "body", "task")


def needs(r) -> set:
    """the structures a row needs beyond the cortical field, as a set."""
    if r.substrate == "expressible":
        return set()
    return {x for x in r.substrate.replace("needs-", "", 1).split("+") if x}


def substrate_ok(r, have) -> bool:
    """can a substrate with these structures score this row at all?"""
    return needs(r) <= set(have)


def rhythm_targets(state: str, region_names, only_expressible: bool = True,
                   min_sites: int = 4, have=(), unit_names=()):
    """(targets, skipped) for every catalogue row that can be scored on this substrate.

    `region_names` is the substrate's per-site atlas label list.  A target carries the
    resolved site indices, so the caller never re-derives them (and cannot re-derive them
    differently).  A row is skipped, with its reason, when:

      * its loop needs a structure the substrate does not have (`only_expressible`);
      * its measure kind is not one this instrument can score -- an evoked response, a
        behavioural rhythm, a phase gradient, a burst statistic all need a protocol or an
        analysis that is not a spectrum;
      * its stations resolve to fewer than `min_sites` sites, which would make the trace a
        handful of columns rather than a region.
    """
    targets, skipped = [], []
    have = set(have)
    # `unit_names` names the non-cortical units a caller can address (thalamic nuclei, say).
    # A station resolves to sites when it is cortical and to a unit index when it is not, and
    # the target says which -- a caller must never have to guess which trace a number came
    # from.
    unit_ix = {n: i for i, n in enumerate(unit_names)}
    for r in by_state(state):
        if only_expressible and not substrate_ok(r, have):
            missing = sorted(needs(r) - have)
            skipped.append((r.id, f"substrate lacks: {', '.join(missing)}"))
            continue
        kind = r.measure.get("kind")
        if kind not in SCORABLE_KINDS:
            skipped.append((r.id, f"kind not scorable as a spectrum: {kind}"))
            continue
        stations = tuple(r.measure.get("stations", ()))
        # a row whose stations are not cortical is scored on the unit trace instead
        non_cortex = [st for st in stations if STATIONS[st].kind != "cortex"]
        if non_cortex and kind != "coherence":
            units = sorted({unit_ix[STATIONS[st].id] for st in non_cortex
                            if STATIONS[st].id in unit_ix}
                           | {unit_ix[k] for st in non_cortex for k in unit_ix
                              if k == STATIONS[st].id})
            if not units:
                # fall back to the nucleus group a caller declared for this station
                units = sorted({unit_ix[k] for st in non_cortex for k in unit_ix
                                if k in STATIONS[st].labels or k == st})
            if not units:
                skipped.append((r.id, "no unit addresses "
                                      + ", ".join(sorted({st for st in non_cortex}))))
                continue
            t = {"id": r.id, "name": r.name, "kind": kind, "units": units,
                 "trace": "units", "target": r.measure.get("target"),
                 "evidence": r.evidence}
            if kind == "pac":
                t["phase_band"] = r.measure["phase_band"]
                t["amp_band"] = r.measure["amp_band"]
            else:
                t["band"] = r.measure["band"]
                if kind == "peak_frequency" and r.peak is not None:
                    t["target"] = r.measure.get("target", r.peak)
            targets.append(t)
            continue
        if kind == "coherence":
            if len(stations) < 2:
                skipped.append((r.id, "coherence needs two stations"))
                continue
            groups = [station_sites((s,), region_names) for s in stations]
            if min(len(g) for g in groups) < min_sites:
                skipped.append((r.id, f"a station resolves to <{min_sites} sites"))
                continue
            targets.append({"id": r.id, "name": r.name, "kind": kind, "groups": groups,
                            "trace": "cortex", "band": r.measure["band"],
                            "target": r.measure.get("target"), "evidence": r.evidence})
            continue
        sites = station_sites(stations, region_names)
        if len(sites) < min_sites:
            skipped.append((r.id, f"{len(sites)} sites < {min_sites}"))
            continue
        t = {"id": r.id, "name": r.name, "kind": kind, "sites": sites,
             "trace": "cortex", "target": r.measure.get("target"), "evidence": r.evidence}
        if kind == "pac":
            t["phase_band"] = r.measure["phase_band"]
            t["amp_band"] = r.measure["amp_band"]
        else:
            t["band"] = r.measure["band"]
            if kind == "peak_frequency" and r.peak is not None:
                t["target"] = r.measure.get("target", r.peak)
        targets.append(t)
    return targets, skipped


# ======================================================================================
# the catalogue checks itself.  run this file to see them.
# ======================================================================================
def _page_tiles():
    """the `rz_*` tile ids site/resonance.js actually registers, or None if unreadable."""
    import os
    import re
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "site", "resonance.js")
    try:
        with open(path) as fh:
            return set(re.findall(r"\b(rz_[a-z0-9_]+)\s*:", fh.read()))
    except OSError:
        return None


def check_catalogue() -> list:
    """every consistency check on the table, as (name, ok, detail) triples.

    These are known answers about the table, not about any model: ids unique, stations
    and loops resolve, bands ordered, a declared peak inside its band, and the
    round-trip bound satisfied.  A table that has drifted out of consistency is a table
    that will quietly train the model toward something nobody wrote down.
    """
    out = []

    def chk(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    ids = [r.id for r in RHYTHMS]
    chk("rhythm ids unique", len(ids) == len(set(ids)),
        f"{len(ids)} rhythms, {len(set(ids))} distinct")

    bad = [r.id for r in RHYTHMS if r.loop not in LOOPS]
    chk("every rhythm's loop exists", not bad, ", ".join(bad))

    bad = [(l.id, e.src if e.src not in STATIONS else e.dst)
           for l in LOOPS.values() for e in l.edges
           if e.src not in STATIONS or e.dst not in STATIONS]
    chk("every loop edge resolves to declared stations", not bad, str(bad))

    bad = [r.id for r in RHYTHMS if r.band is not None and not (r.band[0] < r.band[1])]
    chk("bands are ordered intervals", not bad, ", ".join(bad))

    bad = [f"{r.id} peak {r.peak} outside {r.band}" for r in RHYTHMS
           if r.peak is not None and r.band is not None
           and not (r.band[0] <= r.peak <= r.band[1])]
    chk("a declared peak lies inside its band", not bad, "; ".join(bad))

    bad = [r.id for r in RHYTHMS
           if any(s not in STATIONS for s in r.measure.get("stations", ()))]
    chk("every measured station exists", not bad, ", ".join(bad))

    bad = [f"{r.id}:{s}" for r in RHYTHMS for s in r.states if s not in STATES]
    chk("every declared state is in STATES", not bad, ", ".join(bad))

    bad = [r.id for r in RHYTHMS
           if r.evidence not in ("measured-here", "measurable-now", "literature", "target")]
    chk("evidence tags are from the declared set", not bad, ", ".join(bad))

    # the round-trip bound.  a rhythm whose frequency is set externally (the gastric
    # pacemaker, a flickering screen) is not a resonance of its loop and is exempt --
    # the entry says so in `set_by`, which is why that field exists.
    viol, exempt = [], []
    for r in RHYTHMS:
        if r.peak is None or r.set_by == "external" or r.band is None:
            continue
        loop = LOOPS[r.loop]
        fast, _slow = loop.bound_hz()
        if r.peak > fast * 1.0001:
            line = (f"{r.id}: peak {r.peak:g} Hz > bound {fast:.1f} Hz "
                    f"(round trip {loop.delay_ms[0]:g}-{loop.delay_ms[1]:g} ms, "
                    f"net {loop.net_sign})")
            (exempt if r.bound_exempt else viol).append(line)
    chk("no declared peak is faster than its loop can go round", not viol, "; ".join(viol))
    # an exemption is a claim that the loop assignment is wrong, so it is PRINTED every
    # run rather than quietly skipped: a silent exception is how a table rots.
    chk("declared bound exemptions", True,
        "; ".join(exempt) if exempt else "none")

    bad = [r.id for r in RHYTHMS if r.bound_exempt and len(r.bound_exempt) < 40]
    chk("every exemption carries a reason", not bad, ", ".join(bad))

    # the tile ids are checked against site/resonance.js ITSELF, not merely counted.
    # counting passed while `rz_ars` sat here and the page called that tile `rz_aro`:
    # eleven distinct strings, one of them matching nothing.  A check that cannot see
    # the other side of a correspondence is not checking the correspondence.
    tiles = {l.site_tile for l in LOOPS.values() if l.site_tile}
    page = _page_tiles()
    if page is None:
        chk("the site tiles all have a loop here", len(tiles) == 11,
            f"{len(tiles)} tiles, but site/resonance.js was not readable to compare")
    else:
        missing = sorted(tiles - page)
        unclaimed = sorted(page - tiles)
        chk("every declared tile id exists in site/resonance.js", not missing,
            f"not on the page: {missing}" if missing else f"{len(tiles)} matched")
        chk("every tile on the page has a loop here", not unclaimed,
            f"no loop for: {unclaimed}" if unclaimed else f"{len(page)} tiles")

    return out


def summary() -> dict:
    ev = {}
    for r in RHYTHMS:
        ev[r.evidence] = ev.get(r.evidence, 0) + 1
    sub = {}
    for r in RHYTHMS:
        sub[r.substrate] = sub.get(r.substrate, 0) + 1
    return {"rhythms": len(RHYTHMS), "loops": len(LOOPS), "stations": len(STATIONS),
            "evidence": ev, "substrate": sub}


if __name__ == "__main__":
    import sys
    s = summary()
    print(f"{s['rhythms']} rhythms over {s['loops']} loops and {s['stations']} stations")
    print("  evidence: " + ", ".join(f"{k} {v}" for k, v in sorted(s["evidence"].items())))
    print("  substrate: " + ", ".join(f"{k} {v}" for k, v in sorted(s["substrate"].items())))
    print()
    ok = True
    for name, good, detail in check_catalogue():
        print(f"  [{'PASS' if good else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        ok &= good
    print()
    for lid, l in LOOPS.items():
        lo, hi = l.delay_ms
        fast, slow = l.bound_hz()
        print(f"  {lid:16s} {lo:6.1f}-{hi:6.1f} ms  net {l.net_sign}  "
              f"ceiling {slow:7.1f}-{fast:7.1f} Hz   {l.name}")
    sys.exit(0 if ok else 1)
