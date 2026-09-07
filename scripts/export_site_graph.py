"""export the node graph and materialization annotations for the project site.

    .venv/bin/python scripts/export_site_graph.py

reads the mne `sample` subject's own geometry -- pial surfaces with the aparc
parcellation, the aseg subcortical segmentation, the digitised EEG montage and
the MEG sensor array brought into the surface-RAS frame through the measured
head->MRI transform -- and writes `site/data/graph.js`.

what is real: every cortical, subcortical and sensor position, every aparc /
aseg colour, every mesh edge.  what is synthesized, and marked as such in the
output (`synthetic: true`): a handful of peripheral anchors this subject has no
geometry for (retina, cochlea, a spinal stub) and the placement of devices the
library declares abstractly (a coil, an implanted grid, a dbs lead) at the
centroid of the regions their materialization names.

the per-materialization node sets are read off `m.request` symbolically:
`Anat`, `OnSupport`, `Near` and the device list.  no model is built.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/sources/mne-sample/raw/processed-v6/MNE-sample-data"
SUBJECTS = DATA / "subjects"
OUT = ROOT / "site/data/graph.js"

sys.path.insert(0, str(ROOT))
import ibm  # noqa: E402

ibm.load_all(seal=True, strict=True)
from ibm.materialize.library import MODELS  # noqa: E402
from ibm.vocabulary import (  # noqa: E402
    Anat, Difference, Everywhere, Intersect, Near, OnSupport, Union,
)

import mne  # noqa: E402
import nibabel as nib  # noqa: E402

mne.set_log_level("ERROR")
rng = np.random.default_rng(1)


# ---------------------------------------------------------------- cortex ----
def cortex():
    """oct5 source space on the pial surface: ~1k nodes per hemisphere plus
    the triangulation among them, which is the mesh the site draws."""
    src = mne.setup_source_space("sample", spacing="oct5", surface="pial",
                                 subjects_dir=SUBJECTS, add_dist=False)
    xyz, region, hemi, edges, offset = [], [], [], set(), 0
    for h, s in zip(("lh", "rh"), src):
        labels = mne.read_labels_from_annot("sample", "aparc", h, subjects_dir=SUBJECTS)
        vert_label = {}
        for lab in labels:
            for v in lab.vertices:
                vert_label[int(v)] = lab.name.rsplit("-", 1)[0]
        used = s["vertno"]
        pos = s["rr"][used] * 1000.0            # m -> mm, surface RAS
        idx = {int(v): i + offset for i, v in enumerate(used)}
        for v in used:
            xyz.append(pos[idx[int(v)] - offset])
            region.append(vert_label.get(int(v), "unknown"))
            hemi.append(h)
        for tri in s["use_tris"]:
            a, b, c = (idx[int(t)] for t in tri)
            for e in ((a, b), (b, c), (a, c)):
                edges.add((min(e), max(e)))
        offset += len(used)
    return np.array(xyz), region, hemi, sorted(edges)


def aparc_colors():
    ctab = SUBJECTS / "sample/label/aparc.annot.ctab"
    out = {}
    for line in ctab.read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and p[0].isdigit():
            out[p[1]] = "#%02x%02x%02x" % (int(p[2]), int(p[3]), int(p[4]))
    return out


# ----------------------------------------------------------- subcortex ----
ASEG = {  # FreeSurferColorLUT ids, display name, colour, group, spacing
    (10, 49): ("thalamus", "#00760e", "thalamus", 4.5),
    (11, 50): ("caudate", "#7abadc", "basal_ganglia", 5.0),
    (12, 51): ("putamen", "#ec0db0", "basal_ganglia", 5.0),
    (13, 52): ("pallidum", "#0c30ff", "basal_ganglia", 5.0),
    (26, 58): ("accumbens", "#ff8c1a", "basal_ganglia", 4.0),
    (17, 53): ("hippocampus", "#dcd814", "hippocampus", 5.0),
    (18, 54): ("amygdala", "#67ffff", "amygdala", 4.5),
    (16,): ("brainstem", "#779fb0", "brainstem", 7.0),
    (8, 47): ("cerebellum", "#e69422", "cerebellum", 9.0),
}


def poisson_subsample(pts, spacing):
    order = rng.permutation(len(pts))
    kept = []
    for i in order:
        p = pts[i]
        if all(np.linalg.norm(p - pts[j]) >= spacing for j in kept):
            kept.append(i)
    return pts[kept]


def subcortex():
    img = nib.load(str(SUBJECTS / "sample/mri/aseg.mgz"))
    vol = np.asarray(img.dataobj)
    vox2tkr = img.header.get_vox2ras_tkr()
    nodes = []
    for ids, (name, color, group, spacing) in ASEG.items():
        mask = np.isin(vol, ids)
        ijk = np.argwhere(mask)
        ras = (vox2tkr @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
        ras = ras[rng.choice(len(ras), min(len(ras), 4000), replace=False)]
        for p in poisson_subsample(ras, spacing):
            nodes.append((p, name, color, group))
    return nodes


# ------------------------------------------------------------- sensors ----
def sensors():
    raw = mne.io.read_raw_fif(DATA / "MEG/sample/sample_audvis_raw.fif", preload=False)
    trans = mne.read_trans(DATA / "MEG/sample/sample_audvis_raw-trans.fif")
    head_mri = trans["trans"]
    dev_head = raw.info["dev_head_t"]["trans"]
    eeg, meg = [], []
    for ch in raw.info["chs"]:
        loc = ch["loc"][:3]
        if not np.isfinite(loc).all() or not np.any(loc):
            continue
        if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH:
            p = head_mri @ np.r_[loc, 1.0]
            eeg.append((p[:3] * 1000.0, ch["ch_name"]))
        elif ch["kind"] == mne.io.constants.FIFF.FIFFV_MEG_CH and ch["ch_name"].endswith("1"):
            p = head_mri @ dev_head @ np.r_[loc, 1.0]     # magnetometer of each triplet
            meg.append((p[:3] * 1000.0, ch["ch_name"]))
    return eeg, meg


# ------------------------------------------------ assemble the node table ----
cx_xyz, cx_region, cx_hemi, cx_edges = cortex()
APARC = aparc_colors()
nodes = []   # dicts: p, group, region, color, support
for p, r, h in zip(cx_xyz, cx_region, cx_hemi):
    nodes.append(dict(p=p, group="cortex", region=r, hemi=h,
                      color=APARC.get(r, "#999999"), support="cortical_surface"))
edges = [(a, b) for a, b in cx_edges]

sub_start = len(nodes)
for p, name, color, group in subcortex():
    nodes.append(dict(p=p, group=group, region=name, hemi="l" if p[0] < 0 else "r",
                      color=color, support="tissue"))
# edges inside each subcortical structure: 3 nearest neighbours within 2x spacing
sub_pts = np.array([n["p"] for n in nodes[sub_start:]])
sub_reg = [n["region"] for n in nodes[sub_start:]]
for i, p in enumerate(sub_pts):
    d = np.linalg.norm(sub_pts - p, axis=1)
    d[i] = np.inf
    same = np.array([sub_reg[j] == sub_reg[i] for j in range(len(sub_pts))])
    d[~same] = np.inf
    for j in np.argsort(d)[:3]:
        if np.isfinite(d[j]) and d[j] < 2.2 * ASEG[[k for k, v in ASEG.items() if v[0] == sub_reg[i]][0]][3]:
            edges.append((min(i, j) + sub_start, max(i, j) + sub_start))

# thalamocortical spokes: a few edges from thalamus to nearest cortex
thal = [i for i, n in enumerate(nodes) if n["region"] == "thalamus"]
for i in rng.choice(thal, 12, replace=False):
    d = np.linalg.norm(cx_xyz - nodes[i]["p"], axis=1)
    edges.append((int(np.argsort(d)[rng.integers(0, 40)]), i))

# spinal stub (synthetic): descends from the brainstem's lowest point
bs = np.array([n["p"] for n in nodes if n["region"] == "brainstem"])
base = bs[np.argmin(bs[:, 2])]
spine_start = len(nodes)
for k in range(1, 8):
    p = base + np.array([0.0, -2.5 * k, -9.0 * k]) + rng.normal(0, 0.6, 3)
    nodes.append(dict(p=p, group="spinal", region="spinal_cord", hemi="m",
                      color="#8fa3ad", support="body", synthetic=True))
    prev = spine_start + k - 2 if k > 1 else int(np.argmin(np.linalg.norm(
        np.array([n["p"] for n in nodes[:spine_start]]) - base, axis=1)))
    edges.append((prev, spine_start + k - 1))

# retina and cochlea (synthetic, at anatomically plausible positions)
per_start = len(nodes)
for side in (-1, 1):
    eye = np.array([32.0 * side, 62.0, -28.0])
    for k in range(10):
        u = rng.normal(0, 1, 3); u /= np.linalg.norm(u)
        nodes.append(dict(p=eye + 6.5 * u, group="retina", region="retina", hemi="l" if side < 0 else "r",
                          color="#f2e6a7", support="retina", synthetic=True))
    ear = np.array([66.0 * side, -8.0, -32.0])
    for k in range(7):
        t = k / 7 * 4.0
        p = ear + np.array([0.0, 5.0 * np.cos(t) * (1 - k / 9), 5.0 * np.sin(t) * (1 - k / 9)])
        nodes.append(dict(p=p, group="cochlea", region="cochlea", hemi="l" if side < 0 else "r",
                          color="#c9d6ff", support="cochlea", synthetic=True))
for i in range(per_start, len(nodes) - 1):
    if nodes[i]["region"] == nodes[i + 1]["region"] and nodes[i]["hemi"] == nodes[i + 1]["hemi"]:
        edges.append((i, i + 1))

eeg_pos, meg_pos = sensors()
eeg_start = len(nodes)
for p, name in eeg_pos:
    nodes.append(dict(p=p, group="eeg", region="eeg", hemi="m", color="#ffffff",
                      support="sensor_array", name=name))
meg_start = len(nodes)
for p, name in meg_pos:
    nodes.append(dict(p=p, group="meg", region="meg", hemi="m", color="#9fd8ff",
                      support="sensor_array", name=name))

P = np.array([n["p"] for n in nodes])
center = P[:sub_start].mean(0)


def centroid(idx):
    return P[list(idx)].mean(0) if len(idx) else center


def shell_top(idx):
    """the crown of a sensor array: its centroid lifted to the array's own radius."""
    pts = P[list(idx)]
    c = pts.mean(0)
    r = np.linalg.norm(pts - c, axis=1).mean()
    return c + np.array([0.0, 0.0, r])


def cortex_idx(labels=None):
    return [i for i in range(sub_start) if labels is None or nodes[i]["region"] in labels]


def group_idx(groups):
    return [i for i, n in enumerate(nodes) if n["group"] in groups]


def scalp_point(inside, out_mm=12.0):
    """push a point radially out from the brain centre until it clears the eeg shell."""
    v = np.asarray(inside, float) - center
    v /= np.linalg.norm(v) + 1e-9
    r = np.linalg.norm(P[eeg_start:meg_start] - center, axis=1).max() + out_mm
    return center + v * r


# ----------------------------------------------- symbolic region walking ----
SYSTEM_GROUP = {
    "thalamic_nuclei": ["thalamus"], "brainstem_nuclei": ["brainstem"],
    "bg_territories": ["basal_ganglia"], "striosome_matrix": ["basal_ganglia"],
    "hippocampal_subfields": ["hippocampus"], "amygdalar_nuclei": ["amygdala"],
    "cerebellar_lobules": ["cerebellum"], "cerebellar_microzones": ["cerebellum"],
    "hypothalamic_nuclei": ["brainstem"], "spinal_levels": ["spinal"],
    "myotomes": ["spinal"], "dermatomes": ["spinal"], "peripheral_nerves": ["spinal"],
    "autonomic_ganglia": ["spinal"], "skeletal_muscles": ["spinal"], "cranial_nerves": ["brainstem"],
}


def walk(region, acc):
    """collect (kind, payload) from a Region tree."""
    if isinstance(region, Anat):
        acc.append(("anat", region.system, region.label))
    elif isinstance(region, OnSupport):
        acc.append(("support", region.support, None))
    elif isinstance(region, Near):
        acc.append(("near", region.anchor, region.radius_mm))
    elif isinstance(region, (Union, Intersect)):
        for p in region.parts:
            walk(p, acc)
    elif isinstance(region, Difference):
        walk(region.left, acc)
    elif isinstance(region, Everywhere):
        acc.append(("everywhere", None, None))


OBS_LABEL = {
    "eeg": "scalp EEG", "meg": "MEG", "ecog": "ECoG grid", "ieeg": "intracranial EEG",
    "lfp": "local field potential", "intracortical_spikes": "intracortical spikes",
    "dc_potential": "DC potential", "bold": "BOLD fMRI", "asl_perfusion": "arterial spin labelling",
    "pet": "PET", "structural_mri": "structural MRI", "dwi_microstructure": "diffusion MRI",
    "fnirs": "fNIRS optodes", "emg": "EMG", "mep": "motor evoked potential", "behaviour": "behaviour",
    "eye_tracking": "eye tracking", "polysomnography": "polysomnography", "respiration": "respiration",
    "ecg": "ECG", "ppg": "PPG", "temperature": "tissue temperature", "csf_flow_velocity": "CSF flow",
    "drug_concentration": "drug concentration", "tissue_displacement": "tissue displacement",
    "motion_capture": "motion capture", "produced_audio": "produced audio",
}
INT_LABEL = {
    "visual_stimulus": "visual stimulus", "auditory_stimulus": "auditory stimulus",
    "task_cue": "task cue", "tms": "TMS pulse", "tdcs": "tDCS / tACS", "tacs": "tACS", "trns": "tRNS",
    "tfus": "focused ultrasound", "dbs": "DBS current", "anaesthetic": "anaesthetic",
    "pharmacological": "drug", "respiratory_challenge": "CO₂ challenge",
    "graph_ablation": "graph ablation", "naturalistic_stream": "naturalistic stream",
    "invasive_electrical_stimulation": "cortical stimulation",
}
COMP_LABEL = {
    "neural.exc.activity": "excitatory activity", "neural.inh.activity": "inhibitory activity",
    "neural.exc.potential": "membrane potential", "neural.transmembrane_current": "transmembrane current",
    "neural.exc.nmda": "NMDA conductance", "neural.inh.gaba_a": "GABA-A conductance",
    "electromagnetic.potential": "extracellular potential", "electromagnetic.bfield": "magnetic field",
    "electromagnetic.efield": "electric field", "electromagnetic.current_density": "current density",
    "device.contact_potential": "contact potential", "device.channel_gain": "channel gain",
    "transduction.photoreceptor": "photoreceptor state", "transduction.hair_cell": "hair-cell state",
    "effector.drive": "motor drive", "effector.activation": "muscle activation", "effector.force": "contractile force",
    "blood.flow": "blood flow", "blood.volume": "blood volume", "blood.oxygenation": "oxygenation",
    "blood.deoxyhemoglobin": "deoxyhemoglobin", "blood.oxygen_content": "oxygen content",
    "metabolic.oxygen": "oxygen availability", "metabolic.consumption": "oxygen consumption",
    "metabolic.heat": "metabolic heat", "csf.velocity": "CSF velocity", "csf.pressure": "CSF pressure",
    "csf.solute": "CSF solute", "extracellular.volume_fraction": "interstitial volume",
    "extracellular.osmolarity": "osmolarity", "extracellular.k": "K⁺", "extracellular.na": "Na⁺",
    "extracellular.ca": "Ca²⁺", "extracellular.gaba": "GABA", "extracellular.dopamine": "dopamine",
    "extracellular.serotonin": "serotonin", "extracellular.acetylcholine": "acetylcholine",
    "extracellular.noradrenaline": "noradrenaline", "extracellular.adenosine": "adenosine",
    "thermal.temperature": "temperature", "mechanical.pressure": "acoustic pressure",
    "mechanical.displacement": "displacement", "structural.synaptic_density": "synaptic density",
    "structural.dendritic_density": "dendritic density", "structural.axonal_density": "axonal density",
    "structural.myelination": "myelination", "structural.fiber_orientation": "fibre orientation",
}

SCANNER_ANCHOR = center + np.array([0.0, -5.0, 96.0])
BODY_ANCHOR = None  # filled from the spinal stub


def group_from_file():
    out = {}
    for f in (ROOT / "ibm/materialize/library").glob("*.py"):
        for m in re.finditer(r'id="([a-z_]+)"', f.read_text()):
            out[m.group(1)] = f.stem
    return out


GROUP = group_from_file()
GROUP_TITLE = {
    "decoding": "decoding", "electrophysiology": "electrophysiology", "hemodynamic": "hemodynamic",
    "slow": "slow physiology", "state": "state & disorder", "stimulation": "stimulation", "surrogate": "surrogates",
}

spine_idx = group_idx(["spinal"])
BODY_ANCHOR = P[spine_idx[-1]]
RETINA = group_idx(["retina"])
COCHLEA = group_idx(["cochlea"])
EEG = list(range(eeg_start, meg_start))
MEG = list(range(meg_start, len(nodes)))

materializations = []
for mid, m in MODELS.items():
    acc = []
    for _, r in m.request.regions:
        walk(r, acc)
    for rule in m.request.resolution.rules:
        walk(rule.region, acc)
    for t in m.request.targets:
        walk(t.region, acc)

    cortical_labels = {lab for k, s, lab in acc if k == "anat" and s == "cortical_areas"}
    sub_groups = set()
    for k, s, lab in acc:
        if k == "anat" and s in SYSTEM_GROUP:
            sub_groups.update(SYSTEM_GROUP[s])
    supports = {s for k, s, _ in acc if k == "support"}
    nears = {a: r for k, a, r in acc if k == "near"}

    # focus: what the model names.  everything else is the shared substrate.
    if cortical_labels:
        focus = set(cortex_idx(cortical_labels))
    elif "cortical_surface" in supports or "cortical_depth" in supports or "cortex" in nears \
            or "tissue" in supports or not sub_groups:
        focus = set(cortex_idx())
    else:
        focus = set()
    if "tissue" in supports and not cortical_labels:
        focus |= set(group_idx(["thalamus", "basal_ganglia", "hippocampus", "amygdala", "cerebellum", "brainstem"]))
    if sub_groups:
        focus |= set(group_idx(sub_groups))

    focus_c = centroid(focus) if focus else center

    # devices: real montages where the subject has them, otherwise placed at the focus.
    devices = []
    dev_nodes = set()
    for d in m.request.devices:
        if d.support == "sensor_array" and d.name in ("eeg", "psg", "optodes"):
            idx = EEG if d.name != "psg" else EEG[::10][:6]
            devices.append(dict(name=d.name, support=d.support, role=d.role, n=d.n_elements,
                                anchor=shell_top(EEG).tolist(), nodes=idx, note=d.note))
            dev_nodes |= set(idx)
        elif d.support == "sensor_array" and d.name == "meg":
            devices.append(dict(name=d.name, support=d.support, role=d.role, n=d.n_elements,
                                anchor=shell_top(MEG).tolist(), nodes=MEG, note=d.note))
            dev_nodes |= set(MEG)
        elif d.support == "sensor_array" and d.name == "emg":
            devices.append(dict(name=d.name, support=d.support, role=d.role, n=d.n_elements,
                                anchor=BODY_ANCHOR.tolist(), nodes=spine_idx[-3:], note=d.note))
        elif d.support == "scanner_element":
            devices.append(dict(name=d.name, support=d.support, role=d.role, n=d.n_elements,
                                anchor=SCANNER_ANCHOR.tolist(), nodes=[], note=d.note))
        elif d.support == "display":
            devices.append(dict(name=d.name, support=d.support, role=d.role, n=d.n_elements,
                                anchor=(centroid(RETINA) + np.array([0, 40, 0])).tolist(), nodes=RETINA, note=d.note))
        elif d.support == "stimulator":
            # a coil / pad / transducer sits on the scalp over the focus; a dbs lead is inside
            if d.name == "lead":
                anchor = centroid(group_idx(["basal_ganglia"]) or focus)
                near_idx = [i for i in range(len(nodes)) if np.linalg.norm(P[i] - anchor) < 14]
            else:
                anchor = scalp_point(focus_c if cortical_labels else center + np.array([-35, 10, 60]))
                near_idx = [i for i in cortex_idx() if np.linalg.norm(P[i] - anchor) < 30]
            devices.append(dict(name=d.name, support=d.support, role=d.role, n=d.n_elements,
                                anchor=anchor.tolist(), nodes=[], note=d.note, near=near_idx))
            focus |= set(near_idx)
        elif d.support == "implanted_array":
            radius = nears.get(d.name, 10.0)
            if cortical_labels:
                anchor = focus_c + (focus_c - center) * 0.15
            else:
                anchor = P[cortex_idx(["superiorparietal"])].mean(0)
            near_idx = [i for i in cortex_idx() if np.linalg.norm(P[i] - anchor) < max(radius, 12.0) + 6]
            devices.append(dict(name=d.name, support=d.support, role=d.role, n=d.n_elements,
                                anchor=anchor.tolist(), nodes=[], note=d.note, near=near_idx))
            focus |= set(near_idx)
        else:
            devices.append(dict(name=d.name, support=d.support, role=d.role, n=d.n_elements,
                                anchor=center.tolist(), nodes=[], note=d.note))

    dev_by_support = {}
    for d in devices:
        dev_by_support.setdefault(d["support"], d)

    def dev_anchor(*supports, fallback):
        for s in supports:
            if s in dev_by_support:
                return np.asarray(dev_by_support[s]["anchor"])
        return fallback

    peripheral = set()
    for t in m.request.targets:
        for v in t.vars:
            if v.startswith("transduction.photoreceptor"):
                peripheral |= set(RETINA)
            if v.startswith("transduction.hair_cell"):
                peripheral |= set(COCHLEA)
            if v.startswith("effector."):
                peripheral |= set(spine_idx)
    for o in m.request.interventions:
        if o in ("visual_stimulus",):
            peripheral |= set(RETINA)
        if o in ("auditory_stimulus",):
            peripheral |= set(COCHLEA)
    for o in m.request.observations:
        if o in ("emg", "mep", "behaviour", "motion_capture"):
            peripheral |= set(spine_idx)

    # inputs: observations (likelihoods) and interventions (clamped state)
    inputs = []
    for o in dict.fromkeys(m.request.observations):
        if o == "eeg":
            a, n = shell_top(EEG), EEG
        elif o == "meg":
            a, n = shell_top(MEG), MEG
        elif o in ("bold", "asl_perfusion", "pet", "structural_mri", "dwi_microstructure"):
            a, n = SCANNER_ANCHOR, []
        elif o in ("ecog", "ieeg", "lfp", "intracortical_spikes", "dc_potential"):
            a, n = dev_anchor("implanted_array", "stimulator", fallback=focus_c), []
        elif o in ("emg", "mep", "behaviour", "eye_tracking", "respiration", "motion_capture",
                   "produced_audio", "ecg", "ppg"):
            a, n = BODY_ANCHOR, spine_idx[-2:]
        elif o == "polysomnography":
            a, n = shell_top(EEG), EEG[::10][:6]
        elif o == "fnirs":
            a, n = shell_top(EEG), EEG
        elif o in ("temperature", "tissue_displacement"):
            a, n = dev_anchor("stimulator", fallback=focus_c), []
        elif o == "csf_flow_velocity":
            a, n = centroid(group_idx(["thalamus"])) + np.array([0, 0, 12]), []
        elif o == "drug_concentration":
            a, n = BODY_ANCHOR + np.array([0, 25, 10]), []
        else:
            a, n = focus_c, []
        inputs.append(dict(id=o, kind="observation", label=OBS_LABEL.get(o, o.replace("_", " ")),
                           anchor=np.asarray(a).tolist(), nodes=list(n)))
    for o in dict.fromkeys(m.request.interventions):
        if o in ("visual_stimulus",):
            a, n = centroid(RETINA) + np.array([0, 22, 0]), RETINA
        elif o in ("auditory_stimulus",):
            a, n = centroid(COCHLEA), COCHLEA
        elif o in ("tms", "tdcs", "tacs", "trns", "tfus"):
            a, n = dev_anchor("stimulator", fallback=scalp_point(focus_c)), []
        elif o == "dbs":
            a, n = dev_anchor("stimulator", fallback=centroid(group_idx(["basal_ganglia"]))), []
        elif o in ("anaesthetic", "pharmacological"):
            a, n = BODY_ANCHOR + np.array([0, 25, 10]), []
        elif o in ("task_cue", "respiratory_challenge"):
            a, n = BODY_ANCHOR, []
        elif o == "graph_ablation":
            a, n = focus_c, []
        else:
            a, n = focus_c, []
        inputs.append(dict(id=o, kind="intervention", label=INT_LABEL.get(o, o.replace("_", " ")),
                           anchor=np.asarray(a).tolist(), nodes=list(n)))

    # outputs: target components, anchored where that component lives
    outputs = []
    seen = set()
    for t in m.request.targets:
        for v in t.vars:
            if v in seen:
                continue
            seen.add(v)
            field = v.split(".")[0]
            if v.startswith("transduction.photoreceptor"):
                a, n = centroid(RETINA), RETINA
            elif v.startswith("transduction.hair_cell"):
                a, n = centroid(COCHLEA), COCHLEA
            elif field == "effector":
                a, n = BODY_ANCHOR, spine_idx
            elif field == "device":
                d = devices[0] if devices else None
                a, n = (np.asarray(d["anchor"]), d["nodes"]) if d else (shell_top(EEG), EEG)
            elif field == "electromagnetic":
                d = dev_by_support.get("sensor_array") or dev_by_support.get("stimulator")
                a = np.asarray(d["anchor"]) if d else scalp_point(focus_c)
                n = d["nodes"] if d else []
            elif field in ("blood", "metabolic"):
                a, n = focus_c + np.array([0, 0, 0]), sorted(focus)
            elif field == "csf":
                a, n = centroid(group_idx(["thalamus"])) + np.array([0, 0, 14]), []
            elif field in ("thermal", "mechanical"):
                a, n = dev_anchor("stimulator", fallback=focus_c), sorted(focus)
            else:
                a, n = focus_c, sorted(focus)
            outputs.append(dict(id=v, label=COMP_LABEL.get(v, v), field=field,
                                anchor=np.asarray(a).tolist(), nodes=list(n)))

    hot = set(peripheral) | dev_nodes | {i for x in inputs for i in x["nodes"]}
    if cortical_labels:
        hot |= set(cortex_idx(cortical_labels))
    if sub_groups:
        hot |= set(group_idx(sub_groups))
    for d in devices:
        hot |= set(d.get("near", []))
    involved = sorted(focus | hot)
    hot = sorted(hot)
    doc = m.doc.strip().split("\n\n")[0].replace("\n", " ").strip()
    doc = re.sub(r"\s+", " ", doc)
    materializations.append(dict(
        id=mid, group=GROUP.get(mid, "other"), group_title=GROUP_TITLE.get(GROUP.get(mid, ""), "other"),
        doc=doc, window_dt_s=m.request.window.dt, window_n=m.request.window.n,
        regions=sorted(cortical_labels), systems=sorted(
            {s for k, s, _ in acc if k == "anat"}),
        supports=sorted(supports), observations=list(dict.fromkeys(m.request.observations)),
        interventions=list(dict.fromkeys(m.request.interventions)), targets=[v for t in m.request.targets for v in t.vars],
        devices=[dict(name=d["name"], support=d["support"], role=d["role"], n=d["n"],
                      anchor=d["anchor"], nodes=d["nodes"], note=d["note"]) for d in devices],
        inputs=inputs, outputs=outputs, involved=involved, hot=hot,
        whole=not (cortical_labels or sub_groups or any(d.get("near") for d in devices)), focus_anchor=focus_c.tolist(),
        constrained=list(m.constrained), prior_dominated=list(m.prior_dominated),
    ))


def q(v):
    return [round(float(x), 1) for x in v]


def ranges(idx):
    """run-length encode a sorted index list as [start, stop) pairs."""
    idx = sorted(set(int(i) for i in idx))
    out = []
    for i in idx:
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


out = dict(
    subject="mne sample", frame="surface RAS (mm)", center=q(center),
    surface="pial, oct5 source space", cortex_range=[0, sub_start],
    counts=dict(cortex=sub_start, subcortical=spine_start - sub_start,
                spinal=len(spine_idx), retina=len(RETINA), cochlea=len(COCHLEA),
                eeg=len(EEG), meg=len(MEG), edges=len(edges)),
    nodes=[dict(p=q(n["p"]), g=n["group"], r=n["region"], h=n["hemi"], c=n["color"],
                **({"s": 1} if n.get("synthetic") else {})) for n in nodes],
    edges=[[int(a), int(b)] for a, b in edges],
    regions={r: APARC.get(r, "#999") for r in sorted(set(cx_region))},
    subcortical={v[0]: v[1] for v in ASEG.values()},
    materializations=materializations,
)
for mat in out["materializations"]:
    mat["focus_anchor"] = q(mat["focus_anchor"])
    mat["involved"] = ranges(mat["involved"])
    mat["hot"] = ranges(mat["hot"])
    for x in mat["inputs"] + mat["outputs"] + mat["devices"]:
        x["anchor"] = q(x["anchor"])
        x["nodes"] = ranges(x["nodes"])

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("window.IBM_GRAPH = " + json.dumps(out, separators=(",", ":")) + ";\n")
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
print(out["counts"])
for mat in materializations:
    print(f"{mat['id']:24s} {mat['group']:18s} involved={len(mat['involved']):5d} "
          f"hot={len(mat['hot']):4d} in={[x['id'] for x in mat['inputs']]} out={len(mat['outputs'])}")
