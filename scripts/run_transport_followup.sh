#!/usr/bin/env bash
# the recommended operating point, on the real task and on the task the sheet
# already does.  run after scripts/run_transport_experiments.sh.
set -u
cd /home/brandonin/Documents/IBM-1
export PYTHONPATH=.
PY=.venv/bin/python
CFG="base:2.0,1.0,0;aniso1:2.0,8.0,1;aniso4d:2.0,8.0,4,1.0,120"

while pgrep -f "venv/bin/python -u scripts/ablate_disjoint" >/dev/null; do sleep 20; done

echo '=== REGRESSION CHECK: whole-sheet readout, head refitted per arm ==='
echo '# NOT transport -- linspace(0, n-1) samples the driven region.  this asks'
echo '# only whether the kernel change costs the task the sheet already does.'
$PY -u scripts/ablate_disjoint_transport.py --configs "$CFG" \
    --read-region all --modes intact --noise "0,1e-2,1e-1" \
    --out out/disjoint_wholesheet.json 2>&1 \
  | grep --line-buffered -v "UserWarning\|Consider using"

echo
echo '=== THE RECOMMENDED ARM on the disjoint task ==='
$PY -u scripts/ablate_disjoint_transport.py --configs "$CFG" \
    --out out/disjoint_transport_3arm.json 2>&1 \
  | grep --line-buffered -v "UserWarning\|Consider using"
echo '=== FOLLOWUP DONE ==='
