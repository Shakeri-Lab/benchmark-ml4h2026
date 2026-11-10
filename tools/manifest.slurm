#!/bin/bash
#SBATCH --job-name=dataset-manifest
#SBATCH --account=cdt_computing
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/manifest-%j.out

# Hash every dataset directory and write its MANIFEST.sha256.json.
#
# This is a batch job rather than a shell one-liner because the tree is ~9 GB
# across datasets_raw and datasets_processed, and reading all of it on a login
# node is exactly the thing that gets a session killed and annoys everyone else.
#
#   sbatch tools/manifest.slurm                 # every dataset
#   sbatch tools/manifest.slurm DCLP3 DCLP5     # named ones only

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module purge
module load miniforge 2>/dev/null || true

if [ "$#" -gt 0 ]; then
    roots=()
    for name in "$@"; do
        for tree in datasets_raw datasets_processed datasets_preprocessed; do
            [ -d "$tree/$name" ] && roots+=("$tree/$name")
        done
    done
else
    mapfile -t roots < <(
        find datasets_raw datasets_processed datasets_preprocessed \
             -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort
    )
fi

if [ "${#roots[@]}" -eq 0 ]; then
    echo "no dataset directories matched" >&2
    exit 2
fi

printf 'hashing %d directories\n' "${#roots[@]}"
python tools/dataset_manifest.py write "${roots[@]}"
