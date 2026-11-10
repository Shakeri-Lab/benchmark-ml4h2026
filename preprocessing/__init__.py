"""
Dataset-specific preprocessing functions live here.

Each module exposes a preprocess_<dataset>() function that takes:
- raw_dir: Path to datasets_raw/<dataset>
- output_dir: Path to write processed artifacts (e.g., parquet/csv/splits)
"""

__all__ = [
    "action_cf_benchmark",
    "action_cf_benchmark_bolus",
    "common",
    "dclp3",
    "dclp5",
    "pedap",
    "shanghai_t1dm",
    "registry",
]
