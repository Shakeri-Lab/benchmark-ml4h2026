"""Model-specific, four-cohort slice comparisons from aggregate results only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collect_and_plot import BUDGETS, COHORTS, MODELS, NAMES, save, setup_axes

PRIMARY = ("overall", "post_bolus_30", "nocturnal")
EXTENDED = PRIMARY + ("not_post_bolus_30", "daytime", "post_meal_15", "not_post_meal_15")
STYLE = {
    "overall": ("Overall", "#333333", "-"),
    "post_bolus_30": ("Post-bolus (≤30 min)", "#0072B2", "-"),
    "nocturnal": ("Nocturnal (00:00–06:00)", "#D55E00", "-"),
    "not_post_bolus_30": ("Not post-bolus", "#56B4E9", "--"),
    "daytime": ("Daytime", "#E69F00", "--"),
    "post_meal_15": ("Post-meal (≤15 min; PEDAP)", "#009E73", "-"),
    "not_post_meal_15": ("Not post-meal (PEDAP)", "#CC79A7", "--"),
}


def validate(frame):
    for _, sub in frame.groupby(["model", "cohort", "target_fa_day", "slice"]):
        ordered = sub.sort_values("lead_min")
        if ordered.lead_min.tolist() != list(range(0, 61, 5)):
            raise ValueError("Missing or duplicate warning-time observations.")
        if ordered.n_events.nunique() != 1:
            raise ValueError("Event denominator changed with warning time.")
        recall = ordered.recall.to_numpy(dtype=float)
        if not np.isfinite(recall).all() or (np.diff(recall) > 1e-12).any():
            raise ValueError("Invalid recall curve.")
    # Context comparisons must use the same full-timeline alarm stream.
    for _, sub in frame.groupby(["model", "cohort", "target_fa_day"]):
        if sub.alarm_cutoff_mgdl.nunique() != 1 or sub.test_fa_day.nunique() != 1:
            raise ValueError("Slices unexpectedly have different operating thresholds.")


def plot(frame, directory, model, budget, extended=False, validation=False):
    slices = EXTENDED if extended else PRIMARY
    selected = frame[(frame.model == model) & (frame.target_fa_day == budget)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.0), sharex=True, sharey=True)
    handles = {}
    for ax, cohort in zip(axes.flat, COHORTS):
        sub = selected[selected.cohort == cohort]
        support = []
        for slice_name in slices:
            values = sub[sub.slice == slice_name].sort_values("lead_min")
            if values.empty:
                continue
            label, color, linestyle = STYLE[slice_name]
            x = values.lead_min.to_numpy()
            y = 100 * values.recall.to_numpy(dtype=float)
            line, = ax.plot(x, y, color=color, linestyle=linestyle,
                            linewidth=2.3 if slice_name in PRIMARY else 1.8,
                            label=label, zorder=3)
            ax.fill_between(x, 100 * values.recall_lo.to_numpy(dtype=float),
                            100 * values.recall_hi.to_numpy(dtype=float),
                            color=color, alpha=.08 if not extended else .035,
                            linewidth=0)
            handles.setdefault(slice_name, line)
            if slice_name in PRIMARY:
                short = {"overall": "Overall", "post_bolus_30": "Post-bolus", "nocturnal": "Night"}[slice_name]
                support.append(f"{short}: {int(values.iloc[0].n_events)}")
        setup_axes(ax)
        ax.set_title(cohort.upper(), fontsize=14)
        rate = sub.iloc[0].test_fa_day
        ax.text(.98, .97, f"Test FA/day: {rate:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color="#555555",
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=2))
        ax.text(.98, .87, "Events · " + "; ".join(support), transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#555555",
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=2))
    heading = (f"Validation-calibrated target: {budget:g} false alarms/day" if validation else
               f"Retrospective test-rate matching: {budget:g} false alarms/day target")
    fig.suptitle(f"{NAMES[model]} · recall across event-onset slices\n{heading}", fontsize=15, y=.98)
    fig.supxlabel("Minimum advance warning required (minutes)", fontsize=13, y=.13 if extended else .11)
    fig.supylabel("Event recall (%)", fontsize=13, x=.025)
    fig.legend(handles=[handles[name] for name in slices if name in handles],
               loc="lower center", bbox_to_anchor=(.5, .035),
               ncol=3 if extended else 3, fontsize=10.5 if extended else 11,
               frameon=False)
    note = ("Pointwise 95% patient-bootstrap CIs; cutoff frozen on validation. Test FA/day may differ from the target."
            if validation else
            "Pointwise 95% patient-bootstrap CIs with test-cutoff reselection; descriptive, not deployment-validated.")
    fig.text(.5, .008, note, ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.09, right=.985, top=.86, bottom=.23 if extended else .20,
                        hspace=.25, wspace=.11)
    suffix = "_all_slices" if extended else ""
    save(fig, directory, f"slice_comparison_{model}_target_{budget:g}{suffix}")


def write_index(base, destination, link_base):
    link_destination = link_base / destination.name
    lines = [
        "# Model-specific slice figures",
        "",
        "Each figure overlays overall, post-bolus, and nocturnal recall for one model, with four panels for DCLP3, DCLP5, PEDAP, and IOBP2. PNG and editable SVG versions are supplied.",
        "",
        "The extended 2-FA/day figures also include daytime, non-post-bolus, and the PEDAP-only meal slices. These are overlapping/complementary event contexts, not mutually exclusive cohorts.",
        "",
        "## Interpretation",
        "",
        "- The x-axis is minimum advance warning required, not forecast horizon. Forecasts use the next 60 minutes.",
        "- One full-timeline alarm stream and threshold is shared by all slices of a model/cohort/budget. FA/day is the full-timeline patient-macro burden, not a separately calibrated slice burden.",
        "- Slice membership is determined at event onset. For post-bolus events, warnings ≥30 minutes early occur at or before the bolus; these are not necessarily forecasts made after insulin delivery.",
        "- Post-bolus means onset within 30 minutes after a recorded bolus. Nocturnal means recorded clock 00:00–06:00, not verified sleep. Post-meal means onset within 15 minutes after a recorded meal; only PEDAP has usable meal annotations.",
        "- Recall is patient-macro over patients with slice events. Shading is pointwise 95% patient-bootstrap uncertainty, not simultaneous or multiplicity-adjusted intervals.",
        "",
    ]
    for analysis, title in (("test_rate_matched", "Retrospective test-rate matching"),
                            ("validation_calibrated", "Validation-calibrated, frozen test thresholds")):
        lines.extend([f"## {title}", ""])
        if analysis == "test_rate_matched":
            lines.extend(["Cutoffs are selected using test FA/day, including cutoff reselection within each bootstrap draw. This is a descriptive operating-curve comparison, not deployment-validated performance.", ""])
        else:
            lines.extend(["Cutoffs are selected on validation and frozen for testing. The target is a validation target; the achieved test burden shown in each panel may differ.", ""])
        lines.extend(["| Model | 1 FA/day | 1.5 FA/day | 2 FA/day | 2.5 FA/day | 3 FA/day | All slices at 2 FA/day |",
                      "|---|---|---|---|---|---|---|"])
        for model in MODELS:
            links = []
            for budget in BUDGETS:
                stem = f"slice_comparison_{model}_target_{budget:g}"
                prefix = link_destination / analysis / stem
                links.append(f"[PNG]({prefix}.png) · [SVG]({prefix}.svg)")
            prefix = link_destination / analysis / f"slice_comparison_{model}_target_2_all_slices"
            links.append(f"[PNG]({prefix}.png) · [SVG]({prefix}.svg)")
            lines.append(f"| {NAMES[model]} | " + " | ".join(links) + " |")
        lines.append("")
    lines.extend([f"Aggregate experiment details: [SUMMARY.md]({link_base / 'SUMMARY.md'}) and [PROTOCOL.md]({link_base / 'PROTOCOL.md'}).", ""])
    (destination / "INDEX.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--link-base", type=Path, help="Optional local download path for index links.")
    args = parser.parse_args()
    base = args.base.resolve()
    destination = base / "model_slice_figures"
    for analysis, source, validation in (
        ("test_rate_matched", base / "aggregate" / "test_rate_matched" / "combined" / "curves.json", False),
        ("validation_calibrated", base / "aggregate" / "combined" / "curves.json", True),
    ):
        frame = pd.DataFrame(json.loads(source.read_text()))
        validate(frame)
        directory = destination / analysis
        directory.mkdir(parents=True, exist_ok=True)
        for model in MODELS:
            for budget in BUDGETS:
                plot(frame, directory, model, budget, validation=validation)
            plot(frame, directory, model, 2, extended=True, validation=validation)
        print(f"Created 48 PNG/SVG figure pairs for {analysis}.", flush=True)
    write_index(base, destination, args.link_base or base)
    print(destination / "INDEX.md", flush=True)


if __name__ == "__main__":
    main()
