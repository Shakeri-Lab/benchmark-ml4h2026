"""Fresh training-only fit of the validation winner, followed by test inference.

Search results and their artifacts are kept intact. Final refits and alternative
tables go into run/selected_refits/. Selection never uses counterfactual metrics.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

import run_isolated as work
from recursive_arx import Specification, fit_moments
from make_tables import make


def arx(out):
    original = out / "recursive_arx"
    winner = json.loads((original / "selection.json").read_text())
    root = out / "selected_refits"
    run = root / "recursive_arx"; run.mkdir(parents=True, exist_ok=False)
    moments = dict(np.load(out / "training_moments.npz"))
    model = fit_moments(Specification(**winner["specification"]), moments)
    checkpoint = dict(specification=asdict(model.spec), coefficients=model.coefficients.tolist(), intercept=model.intercept)
    work.write_json(run / "checkpoint.json", checkpoint)
    old = json.loads((original / "checkpoint.json").read_text())
    error = float(np.max(np.abs(np.asarray(old["coefficients"]) - model.coefficients)))
    np.testing.assert_allclose(old["coefficients"], model.coefficients, rtol=1e-10, atol=1e-10)
    work.write_json(run / "selection.json", dict(**winner, fresh_training_only_refit=True,
                    refit_method="one-step fitting using sufficient statistics of every eligible factual training row",
                    refit_vs_search_max_abs_coefficient_difference=error))
    # Only now construct the counterfactual test inputs and infer both families.
    work.infer_arx(root, model)
    make(root, "recursive_arx")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    arx(a.out)
