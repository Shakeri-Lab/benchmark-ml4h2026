"""Use the original warning evaluator unchanged, with baseline glucose forecasts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

BASE = Path("outputs/review_revision_20261011/iob_forecaster")
REFERENCE = Path("outputs/review_revision_20261009/warning_lead_budget")
sys.path.insert(0, str(REFERENCE/"code"))
import evaluate as evaluator


def configure(cohort):
    bundles = {}
    for part in ("val", "test"):
        groups, slices, support, ref_prov = evaluator.load_part("patchmlp_glimmer", cohort, part)
        private = BASE / "private" / cohort
        prov = json.loads((private/part/"provenance.json").read_text())
        if prov["source_sha256"] != ref_prov["source_sha256"] or prov["split_patients"] != ref_prov["split_patients"]:
            raise RuntimeError("Source or split differs from reference.")
        if prov["checkpoint_sha256"] != evaluator.sha(private/"model.json") or prov["config_sha256"] != evaluator.sha(private/"config.json"):
            raise RuntimeError("Baseline model/config provenance mismatch.")
        with np.load(private/part/"dense_min_predictions.npz") as loaded:
            cached = {k: loaded[k] for k in loaded.files}
        offset = 0
        for group in groups:
            n = len(group["origins"])
            stop = offset+n
            if not np.array_equal(cached["slot"][offset:stop], group["origins"]) or not np.all(cached["pid"][offset:stop] == group["pid"]):
                raise RuntimeError("Baseline timeline differs from common evaluation grid.")
            group["scores"] = cached["min_pred"][offset:stop]
            offset = stop
        if offset != len(cached["min_pred"]) or not np.isfinite(cached["min_pred"]).all():
            raise RuntimeError("Incomplete or invalid baseline predictions.")
        support["n_cached_origins"] = len(cached["min_pred"])
        bundles[part] = groups, slices, support, prov
    def load_part(model, requested_cohort, part):
        if model != "iob_linear" or requested_cohort != cohort:
            raise RuntimeError("Unexpected baseline request.")
        return bundles[part]
    evaluator.BASE = BASE
    evaluator.load_part = load_part


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("dclp3", "dclp5", "pedap", "iobp2"), required=True)
    args = parser.parse_args()
    configure(args.cohort)
    evaluator.run("iob_linear", args.cohort)
    # Import after configuration: the existing module receives only an isolated
    # output root and the replacement forecast loader. No source file is edited.
    import test_rate_matched
    test_rate_matched.run("iob_linear", args.cohort)
