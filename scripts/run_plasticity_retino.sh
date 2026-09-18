#!/usr/bin/env bash
# pre-registered retinotopic-ports runs (docs/LOG.md 2026-09-18): arms A/B x seeds 0/1/2,
# competitive rule, drive 0.6.  CPU only, three at a time (~1 GB each).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="" THREADS=5
D=${LOGDIR:-logs/plasticity_v2_retino}; mkdir -p $D
jobs_=("A 0" "B 0" "A 1" "B 1" "A 2" "B 2")
for j in "${jobs_[@]}"; do
  set -- $j
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 15; done
  ( .venv/bin/python -u scripts/plasticity_v2.py --arm $1 --seed $2 --rule competitive --ports retinotopic --drive 0.6 > $D/${1}_seed$2.log 2>&1
    rc=$?; echo "[$(date -Is)] exit $rc $1 seed $2" >> $D/driver.log ) &
  sleep 2
done
wait; echo "[$(date -Is)] ALL DONE" >> $D/driver.log
