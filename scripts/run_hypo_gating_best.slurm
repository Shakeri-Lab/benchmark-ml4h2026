#!/bin/bash
#SBATCH --job-name=hypo_gating_best
#SBATCH --output=logs/hypo_gating_best_%j.out
#SBATCH --error=logs/hypo_gating_best_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard

set -euo pipefail

mkdir -p logs
source ./scripts/activate_env.sh
POST_BOLUS_THRESHOLD="${POST_BOLUS_THRESHOLD:-}"

echo "[+] Running hypo gating metrics for best* models"
shopt -s nullglob
for d in outputs/results/best*; do
  if [[ -d "$d" ]]; then
    echo "[+] Evaluating $d"
    ARGS=(--models_dir "$d" --split test)
    if [[ -n "$POST_BOLUS_THRESHOLD" ]]; then
      ARGS+=(--post_bolus_threshold "$POST_BOLUS_THRESHOLD")
    fi
    python analysis/evaluate_hypo_gating.py "${ARGS[@]}"
  fi
done
shopt -u nullglob

echo "[+] Aggregating results"
python - <<'PY'
import pandas as pd
from pathlib import Path

rows = []
for p in Path("outputs/results").glob("best*/hypo_gating_metrics.csv"):
    try:
        df = pd.read_csv(p)
        if not df.empty:
            rows.append(df)
    except Exception:
        pass

if rows:
    out = pd.concat(rows, ignore_index=True)
    out.to_csv("outputs/results/hypo_gating_best_summary.csv", index=False)
    print(f"[✓] Wrote summary to outputs/results/hypo_gating_best_summary.csv ({len(out)} rows)")
else:
    print("[!] No per-model hypo_gating_metrics.csv found.")
PY
