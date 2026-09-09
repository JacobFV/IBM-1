#!/usr/bin/env bash
# the remaining transport measurements, run STRICTLY SEQUENTIALLY.
#
# the machine is shared and the GPU has already OOMed once when two of these ran
# at the same time, so this is a queue rather than a fan-out.  each stage writes
# its own JSON under out/ and prints one line per configuration.
set -u
cd /home/brandonin/Documents/IBM-1
export PYTHONPATH=.
PY=.venv/bin/python
MMC="scripts/measure_multimodal_convergence.py"

# wait for anything already on the GPU from an earlier stage
while pgrep -f "venv/bin/python -u scripts/measure_multimodal" >/dev/null; do sleep 15; done

read_one () {  # $1 = json path, $2 = label
  $PY -c "
import json,sys
d=json.load(open('$1'))
t=max(v['transport'] for v in d['modalities'].values())
print(f\"$2  transport {t:.3e}  superadd {d['superadditivity']:.4f}\")"
}

echo '=== THE ABLATION: real THINGS-EEG2 retrieval read from precentral only ==='
$PY -u scripts/ablate_disjoint_transport.py \
    --configs "base:2.0,1.0,0;aniso:2.0,8.0,1" \
    --out out/disjoint_transport.json 2>&1 | grep -v "UserWarning\|Consider using"


echo "=== (stage 2) superadditivity vs DRIVE AMPLITUDE ==="
echo '# superadditivity at amp 1.0 reports the drive size, not the sheet: the'
echo '# sigmoid curvature scale is slope = 4 mV and a 0.1 mV perturbation gives a'
echo '# second-order term of (0.1/4)^2 = 6e-4.  so sweep the amplitude.'
for A in 1 3 10 30 100; do
  for CFG in "base:0:1" "aniso:1:8"; do
    NAME=${CFG%%:*}; REST=${CFG#*:}; M=${REST%%:*}; G=${REST##*:}
    F=out/aniso/amp${A}_${NAME}.json
    $PY -u $MMC --amp $A --long-topm $M --long-gain $G --out $F >/dev/null 2>&1
    read_one $F "amp=$A ${NAME}"
  done
done

echo
echo '=== L1-NEUTRAL REALLOCATION: move mass from local to long-range ==='
echo '# row L1 -- and so the resting rate and the operating point -- held fixed.'
for PAIR in "1.0:1.0" "0.75:1.75" "0.5:2.50" "0.25:3.25" "0.0:4.00"; do
  LG=${PAIR%%:*}; FG=${PAIR##*:}
  F=out/aniso/realloc_l${LG}_f${FG}.json
  $PY -u $MMC --long-topm 1 --local-gain $LG --long-gain $FG --out $F >/dev/null 2>&1
  read_one $F "local=$LG long=$FG"
done

echo
echo '=== TONIC DRIVE degrades transport (the fixed-point-climbs prediction) ==='
for T in 0 2 4 8 14; do
  $PY -u scripts/measure_hop_transfer.py --drive-region postcentral \
      --read-region precentral --tonic $T --seeds 1 \
      --out out/hop_tonic_$T.json 2>/dev/null \
    | grep -E "resting rate|^     READ" | tr '\n' ' ' | sed "s/^/tonic=$T  /"
  echo
done

echo

echo '=== ALL DONE ==='
