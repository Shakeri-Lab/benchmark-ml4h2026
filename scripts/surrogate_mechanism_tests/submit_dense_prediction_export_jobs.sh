#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TRAIN_ROOT="${TRAIN_ROOT:-${ROOT}/outputs/surrogate/mechanism_frontier_tests/training}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/surrogate/mechanism_frontier_tests/dense_predictions}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${ROOT}/scripts/surrogate_mechanism_tests/run_dense_prediction_export_job.slurm}"
DRY_RUN=0
OVERWRITE=0
SPLITS="test"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --splits) SPLITS="${2:-test}"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--dry-run] [--overwrite] [--splits test[,val]]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUTPUT_ROOT}" "${ROOT}/outputs/surrogate/mechanism_frontier_tests/slurm_logs"
job_file="${ROOT}/outputs/surrogate/mechanism_frontier_tests/slurm_logs/dense_prediction_export_job_ids.txt"
[[ "${DRY_RUN}" == "1" ]] || : > "${job_file}"

losses=(mse threshold_crossing_kovatchev_clinical_loss wolff_weighted_loss glimmer_fixed_transformer_weights)
datasets=(dclp3 dclp5 pedap azt1d)
for loss in "${losses[@]}"; do
  for dataset in "${datasets[@]}"; do
    run_id="${loss}/patchmlp_${dataset}/seed_42"
    run_dir="${TRAIN_ROOT}/${run_id}"
    out_dir="${OUTPUT_ROOT}/${run_id}"
    [[ -f "${run_dir}/checkpoint.pt" ]] || { echo "Missing checkpoint: ${run_dir}/checkpoint.pt" >&2; exit 1; }
    if [[ "${DRY_RUN}" != "1" && -f "${out_dir}/prediction_pairs_summary.json" && "${OVERWRITE}" != "1" ]]; then
      echo "Completed dense export exists and --overwrite was not set: ${out_dir}" >&2
      exit 1
    fi
    cmd=(sbatch --job-name="mf-dense-${loss:0:6}-${dataset}" --export="ALL,ROOT=${ROOT},RUN_DIR=${run_dir},OUTPUT_ROOT=${OUTPUT_ROOT},RUN_ID=${run_id},OVERWRITE=${OVERWRITE},SPLITS=${SPLITS}" "${SLURM_SCRIPT}")
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '%q ' "${cmd[@]}"; printf '\n'
    else
      output="$("${cmd[@]}")"
      echo "${output}"
      job_id="$(awk '{print $NF}' <<< "${output}")"
      printf '%s\t%s\t%s\t%s\n' "${job_id}" "${loss}" "${dataset}" "${run_id}" >> "${job_file}"
    fi
  done
done
