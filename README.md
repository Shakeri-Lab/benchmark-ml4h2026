# Blood Glucose Forecasting: ML4H 2026

Code and aggregate results for **From Prediction to Practice: A Task-Aware
Evaluation Framework for Blood Glucose Forecasting**, by **Alireza Namazi and
Heman Shakeri**, Proceedings Track, Machine Learning for Health (ML4H) 2026,
PMLR volume 346.

We evaluate glucose forecasts for hypoglycemia advance warning across clinical
contexts and for predicting insulin responses and selecting doses in paired
simulations.

## Frozen ML4H Version

Use the tagged version
[`ml4h2026-v1.0.0`](https://github.com/Shakeri-Lab/benchmark-ml4h2026/tree/ml4h2026-v1.0.0)
for the paper. The tag fixes the source and aggregate records at one commit;
subsequent development on `main` is separate. Do not move this tag when making
changes; create a new version instead.

```bash
git clone https://github.com/Shakeri-Lab/benchmark-ml4h2026.git
cd benchmark-ml4h2026
git checkout ml4h2026-v1.0.0
```

The paper-specific sources, configurations, and records are in
[`ml4h2026/`](ml4h2026/). Start with the
[reproduction guide](ml4h2026/REPRODUCE.md). The root `models/`, `data/`,
`exp/`, and `analysis/` directories provide the shared research harness.
The paper's warning-time evaluation and simulator ARIMAX use the entry points
in `ml4h2026/`, rather than the older generic alarm and ARIMAX workflows.

## Download the Data

Download the [four-hour simulator intervention dataset ZIP](dataset/controliq_interventional_4h.zip).
The ZIP contains a README with its format and citation. The original archive is
also available on [OSF](https://osf.io/5rzsg/). For the expected installation
paths, see the [reproduction guide](ml4h2026/REPRODUCE.md).

The clinical datasets (DCLP3, DCLP5, PEDAP, and IOBP2) are available through
the [Jaeb Center for Health Research](https://public.jaeb.org/datasets/diabetes)
under their data use agreements. Processed-data and checkpoint locations in
the archived configurations should be adjusted to the user's installation.
See the reproduction guide for expected paths.

## Paper Evaluation

- **Hypoglycemia warning:** 60-minute glucose forecasts, full-timeline alarm
  matching before grouping hypoglycemia episodes by onset context, and alarm
  cutoffs chosen on validation patients. Recall is evaluated at minimum warning
  times from 0 to 60 minutes and validation targets of 1, 1.5, 2, 2.5, and
  3 false alarms per patient-day. The main table uses 15 minutes and target 2/day;
  achieved test rates are reported separately.
- **Simple IOB forecaster:** latest glucose, its backward derivative, and
  literature-based total insulin on board; coefficients are fitted on training
  data and selected by validation forecasting error. The same alarm protocol
  applies afterward.
- **Insulin responses and dose selection:** paired four-hour simulator episodes,
  subject-level uncertainty, endpoint effects, and trajectory-based dose costs.
  The simulator ARIMAX is the selected one-step autoregression with insulin
  lags, recursively rolled out using predicted glucose and the future insulin
  plan. Its implementation class is named `RecursiveARX`; this is the single
  simulator baseline called ARIMAX in the paper.

Aggregate CSVs and figure PDFs are in `ml4h2026/results/` and
`ml4h2026/figures/`. Exact internal model identifiers and unrounded values are
retained; metrics stored as fractions are displayed as percentages where
specified in the paper. Source hashes are recorded in
`ml4h2026/source_manifest.json`.

## Citation

If you use this code, dataset, or evaluation, please cite the paper:

```bibtex
@inproceedings{namazi2026prediction,
  title = {From Prediction to Practice: A Task-Aware Evaluation Framework for Blood Glucose Forecasting},
  author = {Namazi, Alireza and Shakeri, Heman},
  booktitle = {Proceedings of Machine Learning for Health},
  series = {Proceedings of Machine Learning Research},
  volume = {346},
  year = {2026},
  url = {https://github.com/Shakeri-Lab/benchmark-ml4h2026/tree/ml4h2026-v1.0.0}
}
```

Pagination and the publication URL can be added when PMLR publishes the volume.
Machine-readable citation information is provided in [`CITATION.cff`](CITATION.cff).

## General Benchmark Workflows

Environment setup, preprocessing, model training, and general experiment
commands are documented in [the shared-workflow guide](docs/GENERAL_WORKFLOWS.md).
That guide describes the broader research harness; use the paper-specific
guide above for the frozen ML4H evaluation.

This work was partially supported by the LaunchPad for Diabetes program.
