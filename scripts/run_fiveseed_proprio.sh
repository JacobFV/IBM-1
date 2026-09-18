#!/usr/bin/env bash
# the pre-registered five-seed test (docs/LOG.md, 2026-09-13): seeds 2,3,4 of the
# raw-readout sheet and of no-cortex, added to the seeds 0,1 already run.  same corpus,
# split, steps and code as those.
#
# STRICTLY ONE AT A TIME.  a cortex-arm run holds ~65 GB, and on the GB10 the GPU and the
# CPU share one 121 GB pool.  the first relaunch ran three at once: one run died on CUDA
# OOM, and when two of them grew together the KERNEL ran out of memory system-wide and
# started killing the desktop, then the machine went down.  that is very probably what
# silently killed seeds 2 and 3 on 13 Sep too -- their logs stop after step 0 with no
# error, which is what a SIGKILL from the OOM killer leaves behind.
#
# a run whose output already reached the final step is skipped, so this can be relaunched
# after an interruption without redoing finished work.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export IBM_GIT_SHA="$(git rev-parse --short HEAD)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# .venv, not venv: the first launch fell through to a system python3 with no torch.
PY=.venv/bin/python
[ -x "$PY" ] || { echo "no $PY" >&2; exit 2; }
"$PY" -c "import torch" || { echo "torch missing in $PY" >&2; exit 2; }
STEPS=3000
done_() {  # does this output hold an evaluation at the final step?
  "$PY" - "$1" "$STEPS" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    ok = any(a["history"] and a["history"][-1]["step"] >= int(sys.argv[2]) for a in d["arms"].values())
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PYEOF
}
jobs_=(
  "--arms no_cortex --seed 4 --out out/proprioceptive_motor_nocortex_seed4.json|logs/proprio_nocortex_seed4.log"
  "--arms trained --readout-norm none --seed 2 --out out/proprio_readout_raw_seed2.json|logs/proprio_readout_raw_seed2.log"
  "--arms trained --readout-norm none --seed 3 --out out/proprio_readout_raw_seed3.json|logs/proprio_readout_raw_seed3.log"
  "--arms trained --readout-norm none --seed 4 --out out/proprio_readout_raw_seed4.json|logs/proprio_readout_raw_seed4.log"
)
for j in "${jobs_[@]}"; do
  args="${j%%|*}"; log="${j##*|}"; out="${args##*--out }"
  if done_ "$out"; then echo "[$(date -Is)] skip (complete) $args" >> logs/fiveseed_driver.log; continue; fi
  echo "[$(date -Is)] start $args" >> logs/fiveseed_driver.log
  "$PY" -u scripts/train_proprioceptive_motor.py $args --steps $STEPS > "$log" 2>&1
  rc=$?   # captured before anything else runs, or $(date) resets it
  echo "[$(date -Is)] exit $rc $args" >> logs/fiveseed_driver.log
done
echo "[$(date -Is)] ALL DONE" >> logs/fiveseed_driver.log
