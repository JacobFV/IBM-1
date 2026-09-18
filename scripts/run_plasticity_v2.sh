#!/usr/bin/env bash
# the pre-registered plasticity experiment (docs/LOG.md 2026-09-18): arms A/B/C x seeds
# 0/1/2, three at a time.  CPU ONLY -- the GPU belongs to the five-seed test and on the
# GB10 a second GPU job is a machine OOM (CLAUDE.md).  each run is ~1 GB RSS.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="" THREADS=5
TAG=${RULE:+_$RULE}; mkdir -p out/plasticity_v2$TAG logs/plasticity_v2$TAG
for s in 0 1 2; do
  for arm in ${ARMS:-A B C}; do
    ( .venv/bin/python -u scripts/plasticity_v2.py --arm $arm --seed $s ${RULE:+--rule $RULE} > logs/plasticity_v2$TAG/${arm}_seed$s.log 2>&1
      rc=$?; echo "[$(date -Is)] exit $rc $arm seed $s" >> logs/plasticity_v2$TAG/driver.log ) &
  done
  wait
done
echo "[$(date -Is)] ALL DONE" >> logs/plasticity_v2$TAG/driver.log
