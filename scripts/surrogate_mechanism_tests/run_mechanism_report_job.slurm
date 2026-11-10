#!/usr/bin/env bash
#SBATCH --account=shakeri-lab
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=outputs/surrogate/mechanism_frontier_tests/slurm_logs/%x-%j.out
#SBATCH --error=outputs/surrogate/mechanism_frontier_tests/slurm_logs/%x-%j.err

set -euo pipefail

ROOT="${ROOT:-/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark}"
ENV_PREFIX="${ENV_PREFIX:-${ROOT}/.conda_envs/bg-benchmark}"

cd "${ROOT}"
mkdir -p outputs/surrogate/mechanism_frontier_tests/slurm_logs
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-mechanism-frontier-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  "${ENV_PREFIX}/bin/python" scripts/surrogate_mechanism_tests/generate_mechanism_report.py
else
  module load miniforge/24.3.0-py3.11 || true
  conda run -p "${ENV_PREFIX}" python scripts/surrogate_mechanism_tests/generate_mechanism_report.py
fi
