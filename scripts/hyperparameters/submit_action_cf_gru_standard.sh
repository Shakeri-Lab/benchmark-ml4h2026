#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG_DIR="${ROOT}/configs/hyperparameter/gru_action_cf_benchmark"
JOB_SCRIPT="${ROOT}/scripts/hyperparameters/hyper_search_job_standard.sh"

if [[ ! -d "${CFG_DIR}" ]]; then
  echo "Missing config dir: ${CFG_DIR}" >&2
  exit 1
fi

mkdir -p "${ROOT}/logs"

echo "Submitting GRU hyperparameter search on action_cf_benchmark via standard partition"
sbatch "${JOB_SCRIPT}" "${CFG_DIR}" "gru" "action_cf_benchmark"
