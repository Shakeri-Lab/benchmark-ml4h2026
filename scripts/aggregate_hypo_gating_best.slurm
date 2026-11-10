#!/bin/bash
#SBATCH --job-name=hypo_gating_agg
#SBATCH --output=logs/hypo_gating_agg_%j.out
#SBATCH --error=logs/hypo_gating_agg_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail
mkdir -p logs

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
