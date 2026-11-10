#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET="${1:-${DATASET:-action_cf_benchmark}}"
CONFIG_ROOT="${CONFIG_ROOT:-${REPO_DIR}/configs/counterfactual_best}"
TRAIN_DIR="${CONFIG_ROOT}/train"
EVAL_DIR="${CONFIG_ROOT}/eval"
JOB_SCRIPT="${SCRIPT_DIR}/run_counterfactual_best.slurm"
RUN_PREFIX="${RUN_PREFIX:-action_cf_best_}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/results}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

mkdir -p "${REPO_DIR}/logs"

for train_cfg in "${TRAIN_DIR}"/*.yaml; do
  [[ -e "${train_cfg}" ]] || continue
  name="$(basename "${train_cfg}" .yaml)"
  eval_cfg="${EVAL_DIR}/${name}.yaml"
  if [[ ! -f "${eval_cfg}" ]]; then
    echo "Skipping ${name}: missing eval config ${eval_cfg}" >&2
    continue
  fi

  suffix="_${DATASET}"
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
  partition="gpu"
  time_limit="12:00:00"
  mem="64G"
  gres="--gres=gpu:1"
  env_prefix="${REPO_DIR}/.conda_envs/bg-benchmark"

  if [[ "${arch}" == "arimax" ]]; then
    partition="standard"
    time_limit="12:00:00"
    mem="64G"
    gres=""
  elif [[ "${arch}" == "patchmlp_glimmer" ]]; then
    partition="standard"
    time_limit="48:00:00"
    mem="128G"
    gres=""
  fi

  echo "Submitting ${run_id} (${arch})"
  sbatch \
    --partition="${partition}" \
    --account=shakeri-lab \
    --time="${time_limit}" \
    --mem="${mem}" \
    ${gres} \
    --export=ALL,TRAIN_CFG="${train_cfg}",EVAL_CFG="${eval_cfg}",RUN_ID="${run_id}",ENV_PREFIX="${env_prefix}",OUTPUT_ROOT="${OUTPUT_ROOT}",N_BOOTSTRAP=1000,BOOTSTRAP_SEED=42 \
    "${JOB_SCRIPT}"
done
