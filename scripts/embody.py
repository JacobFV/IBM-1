"""materialize a sensorimotor brain from the implicit model and step it in IHM-1.

this is the handoff point.  everything else in this repo ends at a measurement;
this ends at an action, and IHM-1's `BodyPeripheral` is what receives it.

the protocol is theirs, read from `ihm/assembly/peripheral.py` rather than
guessed:

    peripheral.step(dt_s, stimuli, mechanical_state, brain_state, blocked_nerves)
    brain_state = {"motor_commands": {muscle_id: activation in [0, 1]}}

they validate it strictly -- an unknown muscle id or an activation outside [0, 1]
raises -- which is the right behaviour and the reason this script tests against
their actual validator instead of a mock.  the muscle list comes from their
`peripheral.json`, so the two cannot drift apart silently.

**what this is:** a working wire.  the brain materializes from the fused kernel,
receives afferent rates keyed by brain region, runs the dynamics from postcentral
sites to precentral sites, and emits an activation per muscle that IHM accepts.

**what this is not:** a controller.  there is no motor corpus in this programme
and nothing has trained these weights, so a fresh materialization emits small
activations around 0.5 and means nothing by them.  teaching it to move requires a
closed loop with reward or a demonstration corpus, and that loop runs on IHM's
side because the body is there.  the point of this script is that the loop can
now be closed at all.

`--sever` zeroes the association kernel.  afference enters postcentral and the
command is read from precentral, so the signal has to cross between two disjoint
site populations through the kernel: severing it should collapse the command's
dependence on the input, and if it does not, the kernel is not carrying movement
and the readout is doing it alone.  that is the same ablation this repo runs on
every perceptual claim, applied to the motor path before anyone trusts it.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(HERE, "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
fu = importlib.util.spec_from_file_location(
    "fuse", os.path.join(HERE, "fuse_implicit.py"))
FU = importlib.util.module_from_spec(fu); fu.loader.exec_module(FU)

IHM = os.path.expanduser("~/Documents/IHM-1")
PERIPHERAL = f"{IHM}/data/derived/canonical/peripheral.json"


def load_body():
    if not os.path.exists(PERIPHERAL):
        raise SystemExit(f"IHM-1 peripheral.json not found at {PERIPHERAL}")
    spec = json.load(open(PERIPHERAL))
    sys.path.insert(0, IHM)
    from ihm.assembly.mechanical_peripheral import MechanicalPeripheral
    return spec, MechanicalPeripheral.from_directory(os.path.dirname(PERIPHERAL))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--implicit", default="ckpt/ibm1_implicit.pt")
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--sever", action="store_true",
                    help="zero the association kernel -- the ablation for whether "
                         "the substrate carries the sensorimotor path at all")
    ap.add_argument("--stimulate", action="store_true",
                    help="drive the receptor patches with a varying pressure ramp. "
                         "without it the afferent input is constant and a constant "
                         "command tells you nothing -- it looks identical to a dead "
                         "loop, which is how a wire test gets mistaken for a "
                         "controller")
    ap.add_argument("--no-cord", action="store_true",
                    help="send cortical commands straight to muscle, bypassing the "
                         "segmental cord.  the cord closes the stretch reflex at a "
                         "30 ms loop delay -- an order of magnitude faster than "
                         "anything routed through cortex -- so this is the ablation "
                         "for whether local feedback matters")
    ap.add_argument("--seed", type=int, default=0, help="matched initialization for ablations")
    ap.add_argument("--out", default="out/embody.json")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    spec, body = load_body()
    muscles = sorted(spec["muscle_bindings"], key=lambda b: b["muscle_id"])
    ids = [b["muscle_id"] for b in muscles]
    print(f"IHM-1 body: {len(ids)} muscle bindings, "
          f"{len(spec['receptor_patches'])} receptor patches, "
          f"{len(spec['nerves'])} nerves, {len(spec['relays'])} relays")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dyn = P.CorticalDynamics(a.sites, a.embed, a.k, dev)
    src = "random init"
    if os.path.exists(a.implicit):
        d = torch.load(a.implicit, map_location="cpu")
        e = d["dyn.embed"]
        if e.shape[0] != a.sites:
            e = FU.resample(e, e.shape[0], a.sites)
        dyn.embed.data.copy_(e.to(dev))
        src = f"{os.path.basename(a.implicit)} ({len(d.get('sources', []))} fused sources)"
    # size the afferent encoder from what the body actually sends.  the default
    # 32 silently dropped half of IHM's 64 channels -- the loop still ran and
    # still validated, which is exactly how a truncated input goes unnoticed.
    probe = body.step(a.dt, stimuli={}, brain_state={})
    n_aff = max(1, len(probe.get("afferent_rates_hz", {})
                       or probe.get("receptor_rates_hz", {})))
    spec2, body = load_body()          # fresh body: the probe advanced its clock
    brain = P.SensorimotorLoop(dyn, muscles=ids, afferent_channels=n_aff).to(dev).eval()
    print(f"afferent channels from the body: {n_aff}")
    print(f"brain: kernel from {src}")
    print(f"  afference -> {len(brain.sense_idx):,} postcentral sites")
    # `motor_read` has been None since the readout moved to the whole sheet --
    # `read_idx_full = linspace(0, n-1, read_sites)` -- so this line raised
    # TypeError before it could print anything, and every embody run died at the
    # banner.  It also described the readout as disjoint from the afferent port,
    # which it no longer is; saying so is the point of the line.
    print(f"  command   <- {len(brain.motor_idx):,} precentral sites "
          f"({len(brain.read_idx_full):,} sites read)")
    print(f"  the readout spans the whole sheet, so it is NOT disjoint from the "
          f"afferent port; --sever is what tests whether the kernel carries it\n")

    patches = [p["id"] for p in spec["receptor_patches"]]
    cord = None
    if not a.no_cord:
        sys.path.insert(0, HERE + "/..")
        from ibm.processes.cord import SegmentalCord
        # PASS THE BINDINGS.  IHM's muscle ids are opaque BodyParts3D strings for
        # everything outside the OpenSim subset, so without the catalog records
        # the cord could only map the 72 opensim channels by name and 66 of 249
        # got arcs.  With them the same body maps 192 of 214 contractile
        # channels.  The loop ran either way and never said which.
        cord = SegmentalCord(muscles=ids, dt=a.dt, muscle_bindings=muscles)
        print(f"cord: {cord.describe()}\n")
    log, prev = [], None
    for i in range(a.steps):
        stim = {}
        if a.stimulate:
            # a travelling pressure wave over the patches: every step presents a
            # different afferent pattern, so a command that does not move is a
            # command that is not listening.
            for j, pid in enumerate(patches):
                phase = 2 * np.pi * (i / max(a.steps, 1) + j / max(len(patches), 1))
                stim[pid] = {"pressure_pa": float(2e4 * (1 + np.sin(phase)))}
        out = body.step(a.dt, stimuli=stim, brain_state=(
            brain.act({}, device=dev, sever=a.sever) if prev is None else prev))
        aff = out.get("afferent_rates_hz", {}) or out.get("receptor_rates_hz", {})
        cortical = brain.act(aff, device=dev, sever=a.sever)
        if cord is not None:
            # the descending command is a REQUEST; what the muscle receives is
            # what the cord makes of it once the reflexes have had their say.
            desc = np.array([cortical["motor_commands"][m] for m in ids], np.float32)
            stretch = np.zeros(len(ids), np.float32)
            if set(out['spindle_rates_hz']) != set(ids):
                raise ValueError('Spindle output must contain every bare muscle ID')
            for j, b in enumerate(muscles):
                sid = b.get("muscle_id")
                # IHM keys proprioceptor rates by the BARE muscle id. Reading
                # "proprio:" + id returned 0.0 for every muscle every step, so the
                # stretch reflex never fired and reflex_max sat at exactly 0.0000
                # while the cord still looked alive on Renshaw inhibition alone.
                # Use stretch-only spindle rates: legacy proprioception mixes
                # length and force. Strict indexing rejects a broken join.
                stretch[j] = float(np.clip(
                    out["spindle_rates_hz"][sid]
                    / 100.0, 0.0, 1.0))
            res = cord.step(desc, stretch=stretch)
            prev = {"motor_commands": {m: float(res["alpha"][j])
                                       for j, m in enumerate(ids)}}
            reflex = float(np.abs(res["stretch"]).max())
        else:
            prev = cortical
            reflex = 0.0
        cmds = np.array(list(prev["motor_commands"].values()))
        log.append({"step": i, "n_afferent": len(aff), "reflex_max": reflex,
                    "spindle_max_hz": max(out["spindle_rates_hz"].values()),
                    "proprioceptor_max_hz": max(out["proprioceptor_rates_hz"].values()),
                    "motor_activation_max": max(out["motor_activations"].values(), default=0.),
                    "arc_max": ({name: float(np.abs(res[name]).max()) for name in
                                 ("stretch", "reciprocal", "autogenic", "renshaw")}
                                if cord is not None else {}),
                    "cmd_mean": float(cmds.mean()), "cmd_sd": float(cmds.std()),
                    "cmd_min": float(cmds.min()), "cmd_max": float(cmds.max()),
                    "nerve_activity": len(out.get("nerve_activity_hz", {}))})
        if i < 3 or i == a.steps - 1:
            print(f"  step {i:3d}  afferent channels {len(aff):3d}  "
                  f"command mean {cmds.mean():.4f} sd {cmds.std():.4f} "
                  f"range [{cmds.min():.3f}, {cmds.max():.3f}]", flush=True)

    sd = float(np.std([r["cmd_mean"] for r in log]))
    print(f"\nclosed {a.steps} brain->body->brain steps without a protocol error.")
    print(f"  IHM accepted every command dict ({len(ids)} muscles, all in [0,1])")
    print(f"  command mean varied by sd {sd:.5f} across steps"
          f"{'  -- SEVERED kernel' if a.sever else ''}")
    if sd < 1e-6:
        print("  NOTE: the command did not vary. with no stimuli and untrained "
              "weights that is expected; it is a wire test, not a controller.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"sever": a.sever, "no_cord": a.no_cord, "seed": a.seed,
               "mechanics": "IHM reduced SI BodyMechanics", "dt_s": a.dt,
               "n_muscles": len(ids), "steps": log},
              open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
