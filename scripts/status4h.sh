#!/usr/bin/env bash
# Combined status for the 4h simulation + GRU v2 search.
T=/project/shakeri-lab/Alireza_timeseries/sim-prep/UVA-T1D-Simulator/TestAID
B=/project/shakeri-lab/Alireza_timeseries/benchmark
NOW=$(date +%s)
echo "===== $(date '+%F %H:%M:%S %Z') ====="

echo "--- QUEUE ---"
squeue -u mez4em -o "%.12i %.9j %.2t %.9M %R" | grep -E "eff4h|bolus4h|gru2hv2" || echo "  (none)"
echo "  running: $(squeue -u mez4em -t R -h -o '%j' | grep -cE 'eff4h|bolus4h')/7 sim, $(squeue -u mez4em -t R -h -o '%j' | grep -c gru2hv2)/8 search"

echo "--- SIM PROGRESS (patients complete / 20) ---"
for d in action_cf_controliq_4h:eff4h action_cf_bolus_controliq_4h:bolus4h; do
  dir=${d%%:*}; tag=${d##*:}
  done_n=$(grep -h -o "patient [0-9]* done" $T/logs/${tag}_*.out 2>/dev/null | sort -u | wc -l)
  npat=$(ls -d $T/$dir/patient_* 2>/dev/null | wc -l)
  nf=$(find $T/$dir -name 'suffix_*.csv' 2>/dev/null | wc -l)
  echo "  $tag: complete=$done_n/20  started=$npat  csv=$nf"
done

echo "--- SIM STALL WATCH (RUNNING tasks only; min since last log write) ---"
squeue -u mez4em -t R -h -o "%i %j" | grep -E "eff4h|bolus4h" | while read jid tag; do
  base=${jid%%_*}; task=${jid##*_}
  f="$T/logs/${tag}_${base}_${task}.out"
  [ -f "$f" ] || { printf "    %-30s (no log yet)\n" "$jid"; continue; }
  age=$(( (NOW - $(stat -c %Y "$f")) / 60 ))
  last=$(tail -1 "$f" | cut -c1-40)
  if [ $age -gt 25 ]; then st="** STALLED **"; else st="ok"; fi
  printf "    %-14s %-9s %4smin  %-13s %s\n" "$jid" "$tag" "$age" "$st" "$last"
done

echo "--- GRU v2 SEARCH ---"
O=$B/outputs/hyperparameters/gru_action_cf_benchmark_controliq_v2
n_done=$(ls $O/*/val_rmse.json 2>/dev/null | wc -l)
n_fail=$(ls $O/*/config_failed.json 2>/dev/null | wc -l)
echo "  scored=$n_done/64  failed=$n_fail"
if [ "$n_done" -gt 0 ]; then
  echo "  best so far:"
  for f in $O/*/val_rmse.json; do
    v=$(python3 -c "import json,sys;print(json.load(open('$f')).get('val_rmse','nan'))" 2>/dev/null)
    echo "$v $(basename $(dirname $f))"
  done | sort -g | head -3 | sed 's/^/    /'
fi
