# Benchmark Orchestration

This folder will host tools for running suites of experiments (multiple data/train/eval configs) and aggregating results.

Planned contents:
- Config schemas for data/train/eval and run lists (triplets).
- Batch runner to iterate over multiple experiment configs.
- Example run lists for standard benchmarks.

Single-run harness lives in `exp/`; models live in `models/`; analysis/visualization lives in `analysis/`.
