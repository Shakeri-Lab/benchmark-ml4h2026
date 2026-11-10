#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${REPO_DIR}/outputs/results/policy_regret"
N_BOOTSTRAP=1000
SEED=42
TARGET=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [--run-dir DIR | --run-id ID] [--n-bootstrap N] [--seed S]

Bootstrap policy-regret results locally (no Slurm).

Examples:
  $(basename "$0")
  $(basename "$0") --run-id action_cf_benchmark_bolus_patchmlp
  $(basename "$0") --run-dir outputs/results/policy_regret/action_cf_benchmark_bolus_patchmlp
  $(basename "$0") --n-bootstrap 2000 --seed 7
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      TARGET="$2"
      shift 2
      ;;
    --run-id)
      TARGET="${RUN_ROOT}/$2"
      shift 2
      ;;
    --n-bootstrap)
      N_BOOTSTRAP="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "${REPO_DIR}"

run_one() {
  local run_dir="$1"
  echo "Bootstrapping ${run_dir}"
  python analysis/policy_regret_bootstrap.py \
    --run-dir "${run_dir}" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --seed "${SEED}"
}

if [[ -n "${TARGET}" ]]; then
  if [[ ! -d "${TARGET}" ]]; then
    echo "Run directory not found: ${TARGET}" >&2
    exit 1
  fi
  run_one "${TARGET}"
  exit 0
fi

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "Policy-regret root not found: ${RUN_ROOT}" >&2
  exit 1
fi

found=0
while IFS= read -r run_dir; do
  found=1
  run_one "${run_dir}"
done < <(find "${RUN_ROOT}" -maxdepth 1 -mindepth 1 -type d | sort)

if [[ "${found}" -eq 0 ]]; then
  echo "No policy-regret run directories found under ${RUN_ROOT}" >&2
  exit 1
fi
