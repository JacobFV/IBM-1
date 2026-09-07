"""the training curriculum as a DAG, with gates that are predicates.

CURRICULUM.md is prose and prose drifts.  this is the same curriculum as a
structure that can be queried: which stages are done, which are runnable now,
which are blocked and by what.  a stage's gate is a function of measured
artifacts, so "stage 1 is passed" is something the repo can answer rather than
something a document asserts.

three things this makes explicit that the prose did not:

*it is a DAG and not a chain.*  the paired materialization and the self-supervised
loops do not depend on each other -- they are two likelihood terms over one theta
(STATE.md 7c) -- so they are siblings that both depend on the substrate being
stable and both feed the joint schedule.  running them in parallel is the plan,
not opportunism.

*gates are predicates, not opinions.*  `|L(0)| < 1` and "effective rank rises
while loss falls" are checkable.  a stage whose gate cannot be written as a
predicate is a stage whose completion nobody can dispute, which is how a
curriculum becomes a story about what was going to happen.

*failure has edges too.*  the ablation is not a milestone on the way to anything;
it is the test that decides whether the whole self-supervised branch was worth
running, and if it fails the graph reroutes rather than continues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(Enum):
    DONE = "done"
    RUNNING = "running"
    READY = "ready"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class Stage:
    id: str
    title: str
    deps: tuple[str, ...]
    trains: str
    data: str
    gate: str
    status: Status
    note: str = ""
    machine: str = ""


STAGES: tuple[Stage, ...] = (
    Stage("s0.gain", "fan-in-aware gain prior", (),
          "nothing -- a declaration change",
          "none",
          "|L(0)| < 1 at prior medians",
          Status.DONE,
          "|L(0)| was 7218 because `gain` is per-edge and a node has 4118 incoming "
          "edges. association_transfer now divides by fan_in; the trainer normalizes "
          "the geometric prior by k. the eeg_forward path still needs the same fix",
          "either"),

    Stage("s0.window", "window follows implementation selection", (),
          "nothing -- a scheduling change",
          "none",
          "multi-window plan not refused with Form.RATE selected",
          Status.DONE,
          "selecting the nonlinearity gives the graph a 1.5 s memory from the "
          "adaptation current, longer than half the 2.048 s window. tau_adaptation_s "
          "= 0.30 s drops it to 0.44 and puts the SO at 0.533 Hz, in band -- one "
          "parameter fixes both. APPLIED and VERIFIED: at tau_a=0.50 the plan is "
          "refused at every overlap from 0.25 to 0.60; at 0.30 it is CLEAR from "
          "0.44 up. eeg_forward now asks for 0.50",
          "either"),

    Stage("s1.regime", "nonlinear regime selection", ("s0.gain", "s0.window"),
          "tau_adaptation_s, w_ee, w_ei, adaptation gain",
          "sleep-edfx N3 spectra",
          ">1 invariant set; SO in 0.5-1.0 Hz; bounded",
          Status.DONE,
          "FITTED against measurement, not argued: the slow-oscillation peak across 8 "
          "scored sleep-edfx N3 recordings is 1.000 +/- 0.296 Hz, and tau_adaptation_s "
          "= 0.12 s puts the model's own SO at exactly 1.000 Hz with a 43.5 mV swing "
          "and zero divergence. prior medians sat at a single 4.77 Hz fixed point",
          "either"),

    Stage("s2.spectra", "per-source spectral fit", ("s1.regime",),
          "process parameters, per source",
          "eegmmidb, sleep-edfx, ds000117, ds004873",
          "posterior beats the literature prior on held-out subjects",
          Status.BLOCKED,
          "the existing values are scalp quantities reported as cortical -- redo, do "
          "not inherit. do NOT re-attempt the global joint fit (falsified, 4.3b)",
          "either"),

    # ---- the two branches: siblings, not a sequence ----------------------
    Stage("s3.paired", "paired stimulus -> measured neural", ("s0.gain",),
          "association embeddings + learned lead field",
          "LibriBrain: 235.6 min of 306ch MEG at 250 Hz + cochleagram",
          "MEG variance explained > 50% AND effective rank > 2",
          Status.RUNNING,
          "explains 91-97% of MEG variance at RANK 1.0-1.2 -- the variance gate "
          "passes and the RANK GATE FAILS, which is the decorative-cortex signature. "
          "the run is not the deliverable; the ablation is",
          "local"),

    Stage("s4.selfsup", "self-supervised AV continuation", ("s0.gain",),
          "association embeddings + encoder/decoder",
          "Koyaanisqatsi: 16,500 frames + aligned cochleagram",
          "effective rank RISES while loss falls",
          Status.DONE,
          "20,000 steps. recon 1.77 -> 0.185, effective rank recovered 1.42 -> 2.97. "
          "gate passes",
          "remote"),

    Stage("s4b.crossmodal", "occipito-temporal association", ("s4.selfsup",),
          "the cross-modal edges of the association kernel",
          "the same AV run",
          "|w| on occ->tmp edges > 2x the random-pair baseline",
          Status.FAILED,
          "measured 0.618 against a 0.139 baseline -- 4.4x, net excitatory, signed. "
          "BUT the ablation shows severing every long-range edge costs +0.1% loss, "
          "so the weights are large and functionally inert. WEIGHT MAGNITUDE IS NOT "
          "FUNCTIONAL CONTRIBUTION, and this gate should never have been magnitude",
          "remote"),

    # ---- the test that decides whether either branch meant anything ------
    Stage("s4c.longrange", "make long-range association actually reach",
          ("s5.ablate",),
          "the geometric prior on long-range edges",
          "none -- a declaration change",
          "severing long-range edges costs > 5% loss",
          Status.RUNNING,
          "the distance prior exp(-d/40mm) is applied to long-range edges too, and a "
          "uniform-random partner on a 127 mm sphere is ~85 mm away, so those edges "
          "carry 7x less geometric weight than local ones before learning starts. the "
          "learned factor cannot overcome it. real association fibres are long-range "
          "AND strong -- they need a patchy, distance-INDEPENDENT prior, which is what "
          "topologies/association.py already argues for and the trainer does not do. "
          "APPLIED: long-range edges now take the median LOCAL geometric weight "
          "instead of exp(-d/40mm), so the learned factor decides which survive. "
          "needs a rerun of the ablation to confirm severing them now costs",
          "either"),

    Stage("s5.ablate", "is the cortex load-bearing?", ("s4.selfsup",),
          "nothing -- an evaluation",
          "the trained checkpoints",
          "loss degrades materially when the kernel is frozen at prior, "
          "and when long-range edges are severed",
          Status.DONE,
          "THE PIVOTAL NODE. both branches can hit their loss targets with a "
          "decorative cortex, because encoder and decoder are both learned and the "
          "lead field is linear. a rank-one state explaining 95% of MEG is exactly "
          "what a bypassed cortex looks like. if this fails, the graph reroutes: "
          "constrain the readout before spending more compute on either branch. "
          "RUN on av_v5: bypass +324%, frozen +180%, no_assoc +180%, local_only "
          "+0.1%. the cortex IS load-bearing and the long-range half of it is inert",
          "either"),

    Stage("s6.scale", "scale data and parameters", ("s5.ablate",),
          "everything, at 0.5 mm / 73M association params",
          "10+ hours of varied video; more MEG/EEG corpora",
          "blob persistence, then 2D motion, on HELD-OUT video",
          Status.BLOCKED,
          "11 minutes of one film is memorization territory. data is the binding "
          "constraint for the first milestone, not compute (~3 GB10-hours optimized)",
          "both"),

    Stage("s7.joint", "one schedule over both likelihood terms", ("s5.ablate",),
          "one theta, scheduled across d",
          "paired + self-supervised together",
          "joint beats each branch on its own held-out set",
          Status.RUNNING,
          "grad log p(theta|D) = grad log p(theta) + sum_d grad log p(D_d|theta). "
          "ibm/forge/fit.py already takes a sequence of Tasks; only the schedule is "
          "missing. BUILT and running: scripts/train_multi_materialization.py holds "
          "one CorticalDynamics and accumulates gradients from an AV head and a "
          "paired-MEG head before a single optimizer step. at 250k sites the shared "
          "substrate is 32.0M and the two heads are 9.9M and 3.0M, so most of the "
          "model is shared -- which is the architecture's own claim, measured",
          "both"),

    Stage("s8.curriculum", "expansion-aware stimulus selection", ("s6.scale",),
          "the order of the data itself",
          "held-out expansion measure",
          "loss falls AND effective rank + invariant-set count rise",
          Status.BLOCKED,
          "predictive loss alone is a capture curriculum (7d). select where error is "
          "high AND declining; hold one expansion measure out of the loss",
          "both"),

    Stage("s9.rl", "cognitive schema, then RL", ("s1.regime", "s7.joint"),
          "reward -> neuromodulator -> plasticity -> theta",
          "task_cue, which is declared and carries no content",
          "a stable multi-attractor schema exists to condition on",
          Status.BLOCKED,
          "blocked twice over: needs s1 for attractors to exist at all, and needs "
          "mechanisms 5/7/8 -- basal ganglia, hippocampal indexing, replay -- whose "
          "anatomy is declared and whose processes are not written (4.9)",
          "both"),
)

BY_ID = {s.id: s for s in STAGES}


def frontier() -> tuple[list[Stage], list[Stage]]:
    """(runnable now, blocked) -- a stage is runnable when every dep is DONE."""
    ready, blocked = [], []
    for s in STAGES:
        if s.status in (Status.DONE, Status.RUNNING):
            continue
        if all(BY_ID[d].status is Status.DONE for d in s.deps):
            ready.append(s)
        else:
            blocked.append(s)
    return ready, blocked


def describe() -> str:
    out = ["the training curriculum, as a DAG", ""]
    sym = {Status.DONE: "[done]", Status.RUNNING: "[running]", Status.READY: "[READY]",
           Status.BLOCKED: "[blocked]", Status.FAILED: "[FAILED]"}
    for s in STAGES:
        dep = ", ".join(s.deps) or "-"
        out.append(f"{sym[s.status]:10s} {s.id:16s} {s.title}")
        out.append(f"{'':10s} deps: {dep}")
        out.append(f"{'':10s} gate: {s.gate}")
        if s.note:
            out.append(f"{'':10s} {s.note}")
        out.append("")
    ready, blocked = frontier()
    out.append(f"runnable now: {', '.join(s.id for s in ready) or 'none'}")
    out.append(f"blocked:      {', '.join(s.id for s in blocked)}")
    return "\n".join(out)


if __name__ == "__main__":
    print(describe())
