from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

from preprocessing.dclp3 import preprocess_dclp3
from preprocessing.diax import preprocess_iobp2
from preprocessing.dclp5 import preprocess_dclp5
from preprocessing.pedap import preprocess_pedap
from preprocessing.shanghai_t1dm import preprocess_shanghai_t1dm
from preprocessing.action_cf_benchmark import (
    preprocess_action_cf_benchmark,
    preprocess_action_cf_benchmark_bolus,
    preprocess_action_cf_benchmark_bolus_controliq,
    preprocess_action_cf_benchmark_controliq,
)

Preprocessor = Callable[[Path, Path], None]

DATASET_REGISTRY: Dict[str, Preprocessor] = {
    "DCLP3": preprocess_dclp3,
    "IOBP2": preprocess_iobp2,
    "DCLP5": preprocess_dclp5,
    "PEDAP": preprocess_pedap,
    "Shanghai_T1DM": preprocess_shanghai_t1dm,
    "action_cf_benchmark": preprocess_action_cf_benchmark,
    "action_cf_benchmark_bolus": preprocess_action_cf_benchmark_bolus,
    "action_cf_benchmark_controliq": preprocess_action_cf_benchmark_controliq,
    "action_cf_benchmark_bolus_controliq": preprocess_action_cf_benchmark_bolus_controliq,
}


def get_dataset_names() -> List[str]:
    return sorted(DATASET_REGISTRY.keys())


def get_preprocessor(name: str) -> Preprocessor:
    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"No preprocessor registered for dataset '{name}'") from exc
