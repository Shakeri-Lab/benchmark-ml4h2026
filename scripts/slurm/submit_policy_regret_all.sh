#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DATASET="${1:-${TARGET_DATASET:-action_cf_benchmark_bolus}}"
CONFIG_ROOT="${CONFIG_ROOT:-${REPO_DIR}/configs/policy_regret_best}"
TRAIN_DIR="${CONFIG_ROOT}/train"
EVAL_DIR="${CONFIG_ROOT}/eval"
RUN_PREFIX="${RUN_PREFIX:-${TARGET_DATASET}_}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/results/policy_regret}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

GPU_JOB="${SCRIPT_DIR}/train_policy_regret_cpu.sh"
CPU_JOB="${SCRIPT_DIR}/train_policy_regret_cpu.sh"
GLIMMER_JOB="${SCRIPT_DIR}/train_policy_regret_glimmer.sh"

mkdir -p "${REPO_DIR}/logs"

for train_cfg in "${TRAIN_DIR}"/*.yaml; do
  [[ -e "${train_cfg}" ]] || continue
  name="$(basename "${train_cfg}" .yaml)"
  eval_cfg="${EVAL_DIR}/${name}.yaml"
  if [[ ! -f "${eval_cfg}" ]]; then
    echo "Skipping ${name}: missing eval config ${eval_cfg}" >&2
    continue
  fi

  suffix="_${TARGET_DATASET}"
  if [[ "${name}" != *"${suffix}" ]]; then
    echo "Skipping ${name}: expected suffix ${suffix}" >&2
    continue
  fi
  arch="${name%${suffix}}"
  run_id="${RUN_PREFIX}${arch}"
  run_dir="${REPO_DIR}/${OUTPUT_ROOT}/${run_id}"
  if [[ "${ALLOW_EXISTING}" != "1" && -e "${run_dir}" ]]; then
    echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
    exit 1
  fi
  job_script="${GPU_JOB}"
  if [[ "${name}" == patchmlp_glimmer_* ]]; then
    job_script="${GLIMMER_JOB}"
  elif [[ "${name}" == arimax_* ]]; then
    job_script="${CPU_JOB}"
  fi

  echo "Submitting ${run_id} via $(basename "${job_script}")"
  sbatch --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}" "${job_script}" "${train_cfg}" "${eval_cfg}" "${run_id}"
done
