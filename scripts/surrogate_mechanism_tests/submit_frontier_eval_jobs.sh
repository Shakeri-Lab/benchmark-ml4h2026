#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_DIR="${CONFIG_DIR:-${ROOT}/configs/surrogate/mechanism_frontier_tests}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${ROOT}/scripts/surrogate_mechanism_tests/run_mechanism_eval_job.slurm}"
DRY_RUN=0
ALLOW_SPARSE=0
DEPENDENCY_AFTEROK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --allow-sparse) ALLOW_SPARSE=1; shift ;;
    --dependency-afterok) DEPENDENCY_AFTEROK="${2:-}"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--dry-run] [--allow-sparse] [--dependency-afterok jobid[:jobid...]]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${ROOT}/outputs/surrogate/mechanism_frontier_tests/slurm_logs"
job_file="${ROOT}/outputs/surrogate/mechanism_frontier_tests/slurm_logs/frontier_eval_job_ids.txt"
[[ "${DRY_RUN}" == "1" ]] || : > "${job_file}"

for mode in frontier general; do
  if [[ "${mode}" == "frontier" ]]; then
    config="${CONFIG_DIR}/eval_single_lag_frontier.yaml"
  else
    config="${CONFIG_DIR}/eval_general_metrics.yaml"
  fi
  [[ -f "${config}" ]] || { echo "Missing config: ${config}" >&2; exit 1; }
  if [[ "${ALLOW_SPARSE}" != "1" ]]; then
    for loss in mse threshold_crossing_kovatchev_clinical_loss wolff_weighted_loss glimmer_fixed_transformer_weights; do
      for dataset in dclp3 dclp5 pedap azt1d; do
        dense="${ROOT}/outputs/surrogate/mechanism_frontier_tests/dense_predictions/${loss}/patchmlp_${dataset}/seed_42/prediction_pairs_test.npz"
        [[ -f "${dense}" ]] || { echo "Missing dense stride-1 artifact: ${dense}. Run submit_dense_prediction_export_jobs.sh first, or pass --allow-sparse." >&2; exit 1; }
      done
    done
  fi
  cmd=(sbatch --job-name="mf-eval-${mode}" --export="ALL,ROOT=${ROOT},MODE=${mode},CONFIG=${config}")
  if [[ -n "${DEPENDENCY_AFTEROK}" ]]; then
    cmd+=(--dependency="afterok:${DEPENDENCY_AFTEROK}")
  fi
  cmd+=("${SLURM_SCRIPT}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; printf '\n'
  else
    output="$("${cmd[@]}")"
    echo "${output}"
    job_id="$(awk '{print $NF}' <<< "${output}")"
    printf '%s\t%s\t%s\n' "${job_id}" "${mode}" "${config}" >> "${job_file}"
  fi
done
