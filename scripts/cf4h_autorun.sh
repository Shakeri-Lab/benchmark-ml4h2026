#!/usr/bin/env bash
# Wait for BOTH 4h simulator trees to verify complete, then run the downstream chain.
# Completeness is manifest-based, not the "done" log marker: one job died in its
# epilogue after writing a complete patient, so the marker undercounts.
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
export PYTHONNOUSERSITE=1
PY=.conda_envs/bg-benchmark/bin/python
LOG=logs/cf4h_autorun.log
echo "$(date '+%F %T') autorun armed" >> $LOG

while true; do
  read -r NE NB <<< "$($PY - <<'PYX' 2>/dev/null
import sys, io, contextlib
from pathlib import Path
sys.path.insert(0,"scripts")
from cf4h_prepare import verify_raw
T=Path("/project/shakeri-lab/Alireza_timeseries/sim-prep/UVA-T1D-Simulator/TestAID")
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    a=verify_raw(T/"action_cf_controliq_4h","e",False)
    b=verify_raw(T/"action_cf_bolus_controliq_4h","b",False)
print(len(a), len(b))
PYX
)"
  NE=${NE:-0}; NB=${NB:-0}
  if [ "$NE" -ge 20 ] && [ "$NB" -ge 20 ]; then
    echo "$(date '+%F %T') both trees complete (eff=$NE bolus=$NB); starting chain" >> $LOG
    bash scripts/cf4h_run_all.sh >> $LOG 2>&1
    echo "$(date '+%F %T') chain exited with status $?" >> $LOG
    exit 0
  fi
  # if no sim jobs remain but trees are incomplete, something died -- stop and say so
  if ! squeue -u mez4em -h -o "%j" | grep -qE "eff4h|bolus4h"; then
    echo "$(date '+%F %T') SIM QUEUE EMPTY but incomplete (eff=$NE bolus=$NB) -- NOT starting chain" >> $LOG
    exit 1
  fi
  sleep 180
done
