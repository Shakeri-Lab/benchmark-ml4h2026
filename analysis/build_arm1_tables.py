"""Build the arm-1 LaTeX tables from common-window gating CSVs.

Consumes the per-run CSVs written by analysis/evaluate_hypo_gating.py (run with
--min_context_steps 288) and emits:

  * the main table: overall and post-bolus(30 min) Recall/FA-day per cohort,
  * a slice-coverage table, since the post-bolus slice spans a very different
    fraction of samples under different controllers and recall differences must
    not be read as coverage differences,
  * the appendix table at a 60-minute post-bolus lookback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MODEL_ORDER = ["arimax", "dlinear", "gru", "itransformer", "patchmlp",
               "patchmlp_glimmer", "attengluco", "zoh"]
MODEL_LABEL = {"arimax": "ARX", "dlinear": "DLin", "gru": "GRU",
               "itransformer": "iTr", "patchmlp": "PMLP",
               "patchmlp_glimmer": r"Glim", "attengluco": "AttG", "zoh": "ZOH"}
COHORT_ORDER = ["DCLP3", "DCLP5", "PEDAP", "IOBP2"]


REQUIRED_COLS = {"slice", "model_name", "dataset_name", "prediction_horizon_minutes",
                 "event_recall", "false_alarms_per_patient_day",
                 "num_true_events", "num_patient_days_evaluated"}
REQUIRED_SLICES = {"overall", "post_bolus"}


def _is_complete(d: pd.DataFrame) -> tuple[bool, str]:
    """A CSV read while a duplicate job is still writing can be non-empty but
    truncated, so size alone is not a sufficient check."""
    missing = REQUIRED_COLS - set(d.columns)
    if missing:
        return False, f"missing columns {sorted(missing)}"
    have = set(d["slice"].dropna().unique())
    if not REQUIRED_SLICES.issubset(have):
        return False, f"missing slices {sorted(REQUIRED_SLICES - have)}"
    if d[["event_recall", "num_patient_days_evaluated"]].isna().all().any():
        return False, "all-NaN metric column"
    return True, ""


def load(csv_dir: Path) -> pd.DataFrame:
    frames, bad = [], []
    for f in sorted(csv_dir.glob("*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception as e:
            bad.append((f.name, str(e)))
            continue
        if d.empty:
            bad.append((f.name, "empty"))
            continue
        ok, why = _is_complete(d)
        if not ok:
            bad.append((f.name, why))
            continue
        d["source"] = f.stem
        frames.append(d)
    if bad:
        print(f"[!] {len(bad)} CSV(s) skipped as incomplete:")
        for n, why in bad[:10]:
            print(f"      {n}: {why}")
    if not frames:
        raise SystemExit(f"no usable CSVs in {csv_dir}")
    print(f"[+] {len(frames)} complete CSVs loaded from {csv_dir}")
    return pd.concat(frames, ignore_index=True)


def _cell(row) -> str:
    r = row["event_recall"]
    fa = row["false_alarms_per_patient_day"]
    if pd.isna(r) or pd.isna(fa):
        return "--"
    return f"{r:.2f}/{fa:.2f}"


def main_table(df: pd.DataFrame, horizon: int, slices=("overall", "post_bolus")) -> str:
    models = [m for m in MODEL_ORDER if m in set(df["model_name"])]
    cohorts = [c for c in COHORT_ORDER if c in set(df["dataset_name"])]
    head = " & ".join(MODEL_LABEL.get(m, m) for m in models)
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{Hypoglycemia safety gating at the 30-minute alarm horizon.} "
        r"Each cell is \texttt{Recall/FA-day}; higher recall and lower false alarms per "
        r"patient-day are better. All models are scored on an identical window set "
        r"(Section~\ref{sec:selection}).}",
        r"\label{tab:hypo_gating_main}",
        r"\begin{tabular}{ll" + "c" * len(models) + "}", r"\toprule",
        r"\textbf{Slice} & \textbf{Cohort} & " + head + r" \\", r"\midrule",
    ]
    for si, sl in enumerate(slices):
        label = {"overall": "Overall", "post_bolus": r"Post-bolus"}.get(sl, sl)
        sub = df[(df["slice"] == sl) & (df["prediction_horizon_minutes"] == horizon)]
        if sub.empty:
            continue
        if si:
            lines.append(r"\midrule")
        for ci, c in enumerate(cohorts):
            cells = []
            for m in models:
                r = sub[(sub["dataset_name"] == c) & (sub["model_name"] == m)]
                cells.append(_cell(r.iloc[0]) if len(r) else "--")
            first = rf"\multirow{{{len(cohorts)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\small {label}}}}}" if ci == 0 else ""
            lines.append(f"{first} & {c} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def coverage_table(df: pd.DataFrame, horizon: int) -> str:
    """Post-bolus slice coverage: patient-days in slice / patient-days overall."""
    rows = []
    for c in [c for c in COHORT_ORDER if c in set(df["dataset_name"])]:
        ov = df[(df["dataset_name"] == c) & (df["slice"] == "overall")
                & (df["prediction_horizon_minutes"] == horizon)]["num_patient_days_evaluated"]
        pb = df[(df["dataset_name"] == c) & (df["slice"] == "post_bolus")
                & (df["prediction_horizon_minutes"] == horizon)]["num_patient_days_evaluated"]
        ev = df[(df["dataset_name"] == c) & (df["slice"] == "post_bolus")
                & (df["prediction_horizon_minutes"] == horizon)]["num_true_events"]
        if ov.empty or pb.empty:
            continue
        frac = 100.0 * pb.mean() / ov.mean() if ov.mean() else float("nan")
        rows.append((c, ov.mean(), pb.mean(), frac, ev.mean() if len(ev) else float("nan")))
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{\textbf{Post-bolus slice coverage.} The slice spans a very different "
        r"fraction of samples under different controllers, so recall differences across "
        r"cohorts must not be read as differences in difficulty alone.}",
        r"\label{tab:slice-coverage}",
        r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r"\textbf{Cohort} & \textbf{Patient-days} & \textbf{Post-bolus} & \textbf{Events} \\",
        r"\midrule",
    ]
    for c, ov, pb, frac, ev in rows:
        lines.append(f"{c} & {ov:.0f} & {frac:.1f}\\% & {ev:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", type=Path, default=Path("outputs/gating_common"))
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--out-tex", type=Path, default=Path("arm1_tables.tex"))
    return p.parse_args()


def main():
    a = parse_args()
    df = load(a.csv_dir)
    print(f"loaded {len(df)} rows; models={sorted(set(df['model_name']))}; "
          f"cohorts={sorted(set(df['dataset_name']))}; slices={sorted(set(df['slice']))}")
    tex = main_table(df, a.horizon) + "\n" + coverage_table(df, a.horizon)
    a.out_tex.write_text(tex)
    print(f"[+] wrote {a.out_tex}")
    print(tex[:1500])


if __name__ == "__main__":
    main()
