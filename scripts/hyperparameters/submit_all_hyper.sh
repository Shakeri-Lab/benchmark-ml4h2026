#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG_ROOT="${ROOT}/configs/hyperparameter"
JOB_GPU="${ROOT}/scripts/hyperparameters/hyper_search_job.sh"
JOB_CPU="${ROOT}/scripts/hyperparameters/hyper_search_job_cpu.sh"

ARCHES=(attengluco dlinear gru patchmlp itransformer arimax)
DATASETS=(dclp3 dclp5 pedap shanghait1dm)

for arch in "${ARCHES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    cfg_dir="${CFG_ROOT}/${arch}_${ds}"
    if [[ ! -d "${cfg_dir}" ]]; then
      echo "[SKIP] Missing config dir: ${cfg_dir}"
      continue
    fi
    if [[ "${arch}" == "arimax" ]]; then
      sbatch "${JOB_CPU}" "${cfg_dir}" "${arch}" "${ds}"
    else
      sbatch "${JOB_GPU}" "${cfg_dir}" "${arch}" "${ds}"
    fi
  done
done
