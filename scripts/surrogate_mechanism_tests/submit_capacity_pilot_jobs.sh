#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT}/configs/surrogate/mechanism_frontier_tests}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/surrogate/mechanism_frontier_tests/capacity_sweep/pilot}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${ROOT}/scripts/surrogate_mechanism_tests/run_mechanism_training_job.slurm}"
DRY_RUN=0
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) echo "Usage: $0 [--dry-run] [--overwrite]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}" "${ROOT}/outputs/surrogate/mechanism_frontier_tests/slurm_logs"
job_file="${ROOT}/outputs/surrogate/mechanism_frontier_tests/slurm_logs/capacity_pilot_job_ids.txt"
[[ "${DRY_RUN}" == "1" ]] || : > "${job_file}"

for cap in cap_1p0x cap_0p75x cap_0p5x cap_0p25x cap_0p125x; do
  config="${CONFIG_DIR}/capacity_pilot_mse_patchmlp_dclp3_${cap}_seed42.yaml"
  run_id="mse/${cap}/patchmlp_dclp3/seed_42"
  run_dir="${OUTPUT_ROOT}/${run_id}"
  [[ -f "${config}" ]] || { echo "Missing config: ${config}" >&2; exit 1; }
  if [[ "${DRY_RUN}" != "1" && -f "${run_dir}/prediction_pairs_summary.json" && "${OVERWRITE}" != "1" ]]; then
    echo "Completed run exists and --overwrite was not set: ${run_dir}" >&2
    exit 1
  fi
  cmd=(sbatch --job-name="mf-pilot-${cap}" --export="ALL,ROOT=${ROOT},CONFIG=${config},OUTPUT_ROOT=${OUTPUT_ROOT},RUN_ID=${run_id},OVERWRITE=${OVERWRITE}" "${SLURM_SCRIPT}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; printf '\n'
  else
    output="$("${cmd[@]}")"
    echo "${output}"
    job_id="$(awk '{print $NF}' <<< "${output}")"
    printf '%s\t%s\t%s\n' "${job_id}" "${cap}" "${run_id}" >> "${job_file}"
  fi
done
