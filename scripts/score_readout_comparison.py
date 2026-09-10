"""score the readout-normalisation comparison by the rule fixed BEFORE the runs finished.

the rule is in docs/LOG.md ("readout normalisation: the comparison, fixed BEFORE the runs
finish"), committed while no sheet arm had finished:
  statistic  per arm x seed, the MEAN held-out skill vs persistence over the evaluations at
             step >= 1500 on the 250-step grid (seven evaluations). best-so-far is reported
             and NOT used.
  verdict    compare arms by the mean over their two seeds. a difference smaller than
             max(0.3, the arms' own seed spreads) is NOT DISTINGUISHABLE.
this script only computes that rule from the logs, so the verdict is not arithmetic done by
hand. a run whose log has not reached step 3000 is reported as pending, never scored.

GATE: on the runs already recorded in docs/LOG.md, the late-window means it computes must
equal the recorded ones to the printed precision.
"""
import argparse, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {  # arm -> {seed: log}; the batchnorm logs live on gb10-direct and are copied here
    "no cortex":          {0: "logs/proprio_nocortex_seed0.log", 1: "logs/proprio_nocortex_seed1.log"},
    "sheet, raw readout": {0: "logs/proprio_readout_raw.log",   1: "logs/proprio_readout_raw_seed1.log"},
    "sheet, batchnorm":   {0: "logs/proprio_readout_batchnorm_seed0.remote.log",
                           1: "logs/proprio_readout_batchnorm_seed1.remote.log"},
}
RECORDED = {("no cortex", 0): -3.6442, ("no cortex", 1): -3.6832,
            ("sheet, raw readout", 0): -3.8419, ("sheet, batchnorm", 0): -4.2806}
FLOOR, FROM_STEP, WINDOW = 0.3, 1500, 7

def evals(path):
    pat = re.compile(r"\s+(\d+)\s+train \S+\s+held \S+\s+SKILL vs persistence ([-+]\d+\.\d+)")
    return [(int(m.group(1)), float(m.group(2))) for m in (pat.match(l) for l in open(path)) if m]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gate-only", action="store_true"); a = ap.parse_args()
    score = {}
    print(f"{'arm':20s} {'seed':>4s} {'late mean':>10s} {'best so far':>12s}  status")
    for arm, seeds in RUNS.items():
        for seed, rel in seeds.items():
            p = ROOT / rel
            if not p.exists(): print(f"{arm:20s} {seed:4d} {'':>10s} {'':>12s}  pending (no log)"); continue
            e = evals(p); late = [s for st, s in e if st >= FROM_STEP]
            if not e or e[-1][0] < 3000 or len(late) != WINDOW:
                print(f"{arm:20s} {seed:4d} {'':>10s} {'':>12s}  pending (last step {e[-1][0] if e else '-'})"); continue
            m = sum(late) / WINDOW; score[(arm, seed)] = m
            print(f"{arm:20s} {seed:4d} {m:+10.4f} {max(s for _, s in e):+12.4f}  complete")
    bad = [(k, v, round(score[k], 4)) for k, v in RECORDED.items() if k in score and round(score[k], 4) != v]
    print("\nGATE: computed late means equal the ones recorded in docs/LOG.md: "
          + ("PASS" if not bad and all(k in score for k in RECORDED) else f"FAIL {bad or 'missing runs'}"))
    if bad: sys.exit(1)
    if a.gate_only: return
    arms = {}
    for arm in RUNS:
        s = [score.get((arm, 0)), score.get((arm, 1))]
        if None in s: print(f"\n{arm}: not both seeds complete -- no verdict yet"); continue
        arms[arm] = ((s[0] + s[1]) / 2, abs(s[0] - s[1]))
    if len(arms) < len(RUNS): print("\nVERDICT: pending -- every arm needs both seeds"); return
    thr = max([FLOOR] + [sp for _, sp in arms.values()])
    print(f"\n{'arm':20s} {'two-seed mean':>14s} {'seed spread':>12s}")
    for arm, (m, sp) in sorted(arms.items(), key=lambda kv: -kv[1][0]): print(f"{arm:20s} {m:+14.4f} {sp:12.4f}")
    print(f"\nthreshold = max({FLOOR}, arms' seed spreads) = {thr:.4f}")
    ref = arms["no cortex"][0]
    for arm in ("sheet, raw readout", "sheet, batchnorm"):
        d = arms[arm][0] - ref
        print(f"  {arm} vs no cortex: {d:+.4f} -> " + ("NOT DISTINGUISHABLE" if abs(d) < thr else ("BETTER" if d > 0 else "WORSE")))
    d = arms["sheet, batchnorm"][0] - arms["sheet, raw readout"][0]
    print(f"  batchnorm vs raw readout: {d:+.4f} -> " + ("NOT DISTINGUISHABLE" if abs(d) < thr else ("BETTER" if d > 0 else "WORSE")))

if __name__ == "__main__": main()
