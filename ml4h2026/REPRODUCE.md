# Frozen ML4H 2026 Evaluation

Run commands from the repository root after checking out `ml4h2026-v1.0.0`.
The source snapshot includes the final warning-time evaluator, the simple IOB
forecaster, and the single selected simulator ARIMAX formulation. Existing
research-harness sources are retained. No archived dataset or saved prediction
is changed by staging these entry points.

## Dependencies and Inputs

Use the environment setup in `docs/GENERAL_WORKFLOWS.md` and the repository
`requirements.txt`. The baseline and metric tests need NumPy; the complete
pipeline also uses pandas, PyYAML, PyTorch, and the model-specific dependencies.

Download and extract the [simulated intervention dataset ZIP](../dataset/controliq_interventional_4h.zip).
Copy the five CSVs from `controliq_interventional_4h/processed/` to
`datasets_processed/action_cf_controliq_4h/`:
`train.csv`, `val.csv`, `test_prefix.csv`, `test_suffix.csv`, and
`test_scenarios.csv`. Clinical records must be obtained separately under the
Jaeb data use agreements and processed with the cohort-specific parsers.

Archived run configurations are under `ml4h2026/configs/`. The clinical and
neural simulator files record the training and evaluation settings. Their `train`
and `eval` sections can be supplied as separate YAML files to the generic
training/evaluation commands in the shared-workflow guide. Adapt filesystem
roots to your installation while retaining the patient split, preprocessing,
history, horizon, and model settings.
`configs/simulator/arimax.yaml` and `configs/simulator_arimax_selection.json`
describe the published one-step simulator baseline. The separately named
`input_layout_reference.yaml` is used to reconstruct the original simulator
dataloader layout; its legacy model fields do not specify the published fit.

The pipeline expects saved clinical runs at
`outputs/results/bestck_<model>_<cohort>_p12/`, and simulator runs at
`outputs/results/cf4h_<model>/`, each with its run configuration and model
checkpoint. Simulator runs also supply `cf_artifacts.npz`. These saved
predictions/checkpoints are inputs to the original evaluation drivers and are
not included in this source release. They can be generated with the training
and inference harness; source data alone does not supply trained weights.

## Stage the Frozen Sources

```bash
python ml4h2026/stage_sources.py
```

This restores the experiment-relative `outputs/.../code` paths used by the
audited scripts. It creates source files only and refuses to overwrite
different existing source. Run it in a fresh clone, rather than over an
existing experiment directory containing other code revisions.

## Hypoglycemia Warning

For each clinical model/cohort, generate dense validation and test forecast
minima from its frozen 60-minute forecaster:

```bash
python ml4h2026/warning/dense_inference.py \
  --run-dir outputs/results/bestck_gru_dclp3_p12 \
  --out outputs/review_revision_20261009/warning_lead_budget/private/caches/gru_dclp3_p12
python ml4h2026/warning/evaluate.py --model gru --cohort dclp3
```

Generate the corresponding Glimmer* cache first: the evaluator checks all
model origin grids against `patchmlp_glimmer` within each cohort. The CLI help
lists the supported inference/device options. Evaluations refuse to overwrite
completed results and check checkpoint, configuration, data, and split hashes.

`warning/evaluate.py` calibrates on validation patients and freezes the alarm
cutoff for test, every context, and every warning requirement. This is the
paper protocol. `warning/test_rate_matched.py` is an additional descriptive
diagnostic that chooses operating points on test; its outputs are not used
for the paper's validation-calibrated tables or curves.

The shared event logic is in `warning/warning_metrics.py`. Matching uses the
full timeline before onset contexts are applied. A complete preceding
60-minute origin window is required for recall, so the denominator remains
fixed as minimum warning time changes.

For the IOB forecaster, after the reference clinical cache is available:

```bash
python ml4h2026/iob/train_and_infer.py --cohort dclp3
python ml4h2026/iob/evaluate_baseline.py --cohort dclp3
```

The latter writes the validation-calibrated result first and also runs the
descriptive test-rate diagnostic. Use only the validation-calibrated records
for the paper. The fitted IOB coefficients are archived in `configs/iob/`.

## Simulator ARIMAX

The published simulator baseline trains one-step glucose predictions using
earlier glucose and insulin at the target timestamp and preceding timestamps.
During rollout, predicted glucose is fed back and future insulin comes from
the supplied plan. There is no moving-average term.

```bash
python ml4h2026/simulator/run_isolated.py prepare --out outputs/ml4h2026_arimax
python ml4h2026/simulator/run_isolated.py arx --out outputs/ml4h2026_arimax
python ml4h2026/simulator/refit_and_evaluate.py --out outputs/ml4h2026_arimax
```

The search evaluates 240 configurations on factual validation trajectories;
the selected specification and final training-only refit are used for both
intervention menus. `recursive_arx.py` and the numerical search, fitting, and
inference functions retain the audited implementation. Unused diagnostic
formulations have been removed from this publication entry point.

The original table driver additionally reads the other models' endpoint
metrics from `outputs/review_revision_20261007/table4/<model>.csv` and the
behavior-policy summaries from `outputs/analysis/behavior_policy_<family>.csv`.
These are produced by `analysis/counterfactual_metrics.py` and
`analysis/behavior_policy_baselines.py` from the saved simulator predictions.
`simulator/make_tables.py` combines them with the selected ARIMAX output.
Its historical CSVs may use `Recursive ARX` as the internal model label; the
paper calls this baseline ARIMAX. No alternate simulator formulation is reported.

The exact BGRI clipping, menu inclusion, equivalent action labels, deterministic
ties, and bootstrap conventions are retained. The retrospective constant is
selected on the test episodes within each menu and objective.

## Aggregate Records and Tests

`results/warning_curves.csv` contains all five alarm budgets and all warning
requirements; `warning_slice_contrasts.csv` contains the paired comparisons.
`table4.csv`, `table5.csv`, `restricted_selection.csv`,
`effect_metrics_all_horizons.csv`, and `action_confusion_counts.csv` contain
the simulator records used in the paper. They are aggregate statistics, not
individual clinical records. Existing figure PDFs are in `figures/`.

```bash
python -m unittest discover -s ml4h2026/warning -p 'test_warning_metrics.py'
python -m unittest discover -s ml4h2026/iob -p 'test_iob_forecaster.py'
python -m unittest discover -s ml4h2026/simulator -p 'test_recursive_arx.py'
python ml4h2026/verify_results.py
```

The independent timeline audit also has synthetic tests in
`warning/test_full_timeline_audit.py`. The input/source manifest records the
original source hashes and the unchanged numerical functions. Administrative
paths in those provenance records identify the original experiment layout;
they are not external runtime dependencies.
