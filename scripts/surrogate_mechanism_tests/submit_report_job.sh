#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${ROOT}/scripts/surrogate_mechanism_tests/run_mechanism_report_job.slurm}"
DRY_RUN=0
DEPENDENCY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --dependency) DEPENDENCY="${2:-}"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--dry-run] [--dependency afterok:JOBID[:JOBID...]]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${ROOT}/outputs/surrogate/mechanism_frontier_tests/slurm_logs"
cmd=(sbatch --job-name="mf-report" --export="ALL,ROOT=${ROOT}")
if [[ -n "${DEPENDENCY}" ]]; then cmd+=(--dependency="${DEPENDENCY}"); fi
cmd+=("${SLURM_SCRIPT}")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  output="$("${cmd[@]}")"
  echo "${output}"
  job_id="$(awk '{print $NF}' <<< "${output}")"
  printf '%s\n' "${job_id}" > "${ROOT}/outputs/surrogate/mechanism_frontier_tests/slurm_logs/report_job_id.txt"
fi
