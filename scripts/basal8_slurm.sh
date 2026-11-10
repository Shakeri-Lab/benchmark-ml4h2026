#!/usr/bin/env bash
#SBATCH --job-name=basal8all
#SBATCH --time=08:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --partition=standard --mem=64G -A shakeri-lab
#SBATCH --output=/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark/logs/%x-%j.out
#SBATCH --error=/sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark/logs/%x-%j.err
cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark
bash scripts/basal8_run_all.sh
