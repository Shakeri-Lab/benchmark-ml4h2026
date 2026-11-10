#!/usr/bin/env bash
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=outputs/surrogate/mechanism_frontier_tests/slurm_logs/%x-%j.out
#SBATCH --error=outputs/surrogate/mechanism_frontier_tests/slurm_logs/%x-%j.err

set -euo pipefail

ROOT="${ROOT:-/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark}"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.conda_envs/bg-benchmark}"
MODE="${MODE:?MODE is required: frontier or general}"
CONFIG="${CONFIG:?CONFIG is required}"

cd "${ROOT}"
mkdir -p outputs/surrogate/mechanism_frontier_tests/slurm_logs
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mechanism-frontier-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

case "${MODE}" in
  frontier) script="scripts/surrogate_mechanism_tests/evaluate_single_lag_frontier.py" ;;
  general) script="scripts/surrogate_mechanism_tests/evaluate_general_metrics.py" ;;
  *) echo "Unknown MODE=${MODE}" >&2; exit 2 ;;
esac

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  "${ENV_PREFIX}/bin/python" "${script}" --config "${CONFIG}"
else
  module load miniforge/24.3.0-py3.11 || true
  conda run -p "${ENV_PREFIX}" python "${script}" --config "${CONFIG}"
fi
