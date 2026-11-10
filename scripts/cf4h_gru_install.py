#!/usr/bin/env python3
"""Install the winning GRU config from the v2 2-hour re-search into the 4-hour
training/eval configs, then optionally launch the 4h retrain.

The v2 search exists because the original grid searched hidden_size in {16,32} and
its optimum sat on that ceiling. Selection protocol is otherwise identical to every
other arm-2 model: one 64-cell grid, take the winner by validation RMSE.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path
import yaml

SEARCH_OUT = Path("outputs/hyperparameters/gru_action_cf_benchmark_controliq_v2")
SEARCH_CFG = Path("configs/hyperparameter/gru_action_cf_benchmark_controliq_v2")
TUNED = ["hidden_size", "num_layers", "dropout", "bidirectional"]


def pick_winner():
    rows = []
    for f in SEARCH_OUT.glob("*/val_rmse.json"):
        try:
            v = json.load(open(f)).get("val_rmse")
        except Exception:
            continue
        if v is not None:
            rows.append((float(v), f.parent.name))
    if not rows:
        sys.exit("[gru] no scored runs found")
    rows.sort()
    n_fail = len(list(SEARCH_OUT.glob("*/config_failed.json")))
    print(f"[gru] {len(rows)} scored, {n_fail} failed, of 64 cells")
    print("[gru] top 5:")
    for v, n in rows[:5]:
        print(f"        {n}  val_rmse={v:.4f}")
    return rows[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg-dir", type=Path, default=Path("configs/counterfactual_4h"))
    p.add_argument("--pred-len", type=int, default=48)
    p.add_argument("--run-id", default="cf4hb_gru")
    p.add_argument("--require-complete", action="store_true",
                   help="Refuse to install unless all 64 cells are accounted for.")
    p.add_argument("--train", action="store_true", help="Launch the 4h retrain after installing.")
    a = p.parse_args()

    n_acc = len(list(SEARCH_OUT.glob("*/val_rmse.json"))) + len(list(SEARCH_OUT.glob("*/config_failed.json")))
    if a.require_complete and n_acc < 64:
        sys.exit(f"[gru] only {n_acc}/64 cells accounted for; refusing to pick a winner early")

    best_v, best_id = pick_winner()
    win = yaml.safe_load((SEARCH_CFG / f"{best_id}.yaml").read_text())
    wp = win["model_data"]["model"]["params"]
    wlr = win["training"]["optimizer"]["lr"]
    wbs = win["model_data"]["dataloader"]["batch_size"]
    print(f"\n[gru] WINNER {best_id} val_rmse={best_v:.4f}: "
          + " ".join(f"{k}={wp[k]}" for k in TUNED) + f" lr={wlr} batch_size={wbs}")

    for kind in ("train", "eval"):
        dst = a.cfg_dir / kind / "gru.yaml"
        if not dst.exists():
            sys.exit(f"[gru] missing {dst}")
        shutil.copy(dst, dst.with_suffix(".yaml.bak_oldgrid"))
        c = yaml.safe_load(dst.read_text())
        prm = c["model_data"]["model"]["params"]
        for k in TUNED:
            prm[k] = wp[k]
        dl = c["model_data"]["dataloader"]
        dl["batch_size"] = wbs
        dl["pred_len"] = a.pred_len
        dl["stride"] = a.pred_len
        # arm 2 never sees meals; the search was run under the same restriction.
        dl["feature_cols"] = ["cgm", "basal", "bolus"]
        c.setdefault("training", {}).setdefault("optimizer", {})["lr"] = wlr
        dst.write_text(yaml.safe_dump(c, sort_keys=True))
        print(f"[gru] installed -> {dst} (old grid saved as .yaml.bak_oldgrid)")

    if a.train:
        cmd = [sys.executable, "exp/run_counterfactual_experiment.py", "--train",
               "--train-config", str(a.cfg_dir / "train/gru.yaml"),
               "--eval-config",  str(a.cfg_dir / "eval/gru.yaml"),
               "--output-root", "outputs/results", "--run-id", a.run_id]
        print("\n[gru] launching retrain:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"[gru] retrain FAILED (exit {r.returncode})")
        ck = Path("outputs/results") / a.run_id / "checkpoint.pt"
        print(f"[gru] checkpoint present: {ck.exists()}")


if __name__ == "__main__":
    main()
