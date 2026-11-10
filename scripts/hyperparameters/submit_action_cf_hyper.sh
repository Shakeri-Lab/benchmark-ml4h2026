#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG_ROOT="${ROOT}/configs/hyperparameter"
JOB_GPU="${ROOT}/scripts/hyperparameters/hyper_search_job.sh"
JOB_CPU="${ROOT}/scripts/hyperparameters/hyper_search_job_cpu.sh"

ARCHES=(attengluco dlinear gru patchmlp itransformer arimax)
DATASET="${1:-action_cf_benchmark}"

for arch in "${ARCHES[@]}"; do
  cfg_dir="${CFG_ROOT}/${arch}_${DATASET}"
  if [[ ! -d "${cfg_dir}" ]]; then
    echo "[SKIP] Missing config dir: ${cfg_dir}"
    continue
  fi
  if [[ "${arch}" == "arimax" ]]; then
    sbatch "${JOB_CPU}" "${cfg_dir}" "${arch}" "${DATASET}"
  else
    sbatch "${JOB_GPU}" "${cfg_dir}" "${arch}" "${DATASET}"
  fi
done
