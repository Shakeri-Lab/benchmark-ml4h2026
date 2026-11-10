"""Check the frozen aggregate records without access to individual records."""
from pathlib import Path
import csv
import json
import statistics


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    base = Path(__file__).resolve().parent / "results"
    rows = read(base / "warning_curves.csv")
    keys = {}
    for row in rows:
        key = (row["model"], row["cohort"], float(row["target_fa_day"]), row["slice"])
        keys.setdefault(key, []).append(row)
    for group in keys.values():
        group.sort(key=lambda row: float(row["lead_min"]))
        assert len({int(row["n_events"]) for row in group}) == 1
        assert all(float(a["recall"]) >= float(b["recall"])-1e-12
                   for a, b in zip(group, group[1:]))
    means = {}
    for context in ["post_bolus_30", "not_post_bolus_30"]:
        selected = [row for row in rows if row["model"] != "zoh"
                    and float(row["target_fa_day"]) == 2
                    and float(row["lead_min"]) == 15
                    and row["slice"] == context]
        assert len(selected) == 32
        means[context] = statistics.mean(float(row["recall"])*100 for row in selected)
    assert round(means["post_bolus_30"], 1) == 49.5
    assert round(means["not_post_bolus_30"], 1) == 67.6
    paired = [row for row in read(base / "warning_slice_contrasts.csv")
              if row["model"] != "zoh" and float(row["target_fa_day"]) == 2
              and float(row["lead_min"]) == 15
              and row["left"] == "post_bolus_30"
              and row["right"] == "not_post_bolus_30"]
    assert len(paired) == 32
    assert all(float(row["difference"]) < 0 for row in paired)
    assert sum(float(row["hi"]) < 0 for row in paired) == 19
    counts = read(base / "action_confusion_counts.csv")
    checks = 0
    for row in read(base / "table5.csv"):
        matrix = [cell for cell in counts if cell["family"] == row["family"]
                  and cell["model"] == row["policy"]]
        if not matrix:
            continue
        total = sum(int(cell["n"]) for cell in matrix)
        assert total == 80
        match = sum(int(cell["n"]) for cell in matrix
                    if cell["oracle"] == cell["selected"]) / total
        assert abs(match-float(row["AMR"])) < 1e-12
        checks += 1
    assert checks == 14
    print(json.dumps({"warning_context_means_percent": means,
                      "negative_paired_comparisons": len(paired),
                      "paired_intervals_below_zero": 19,
                      "action_confusion_checks": checks,
                      "passed": True}, indent=2))


if __name__ == "__main__":
    main()
