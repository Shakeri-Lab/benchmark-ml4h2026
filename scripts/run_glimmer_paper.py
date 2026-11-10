#!/usr/bin/env python
"""
Paper-faithful GLIMMER runner on our datasets.

Differences vs upstream:
- Dataset source is our processed timeseries.csv.
- We create a temporal train/val/test split per patient (since we don't have
  separate train/test CSVs like the original OhioT1DM dataset).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Dataset name (e.g., DCLP3, DCLP5, PEDAP)")
    p.add_argument("--processed-root", default="datasets_processed", help="Root containing datasets_processed")
    p.add_argument("--variant", choices=["transformer", "cnn_lstm"], default="transformer")
    p.add_argument("--train-window", type=int, default=72, help="Context length (time steps)")
    p.add_argument("--pred-len", type=int, default=12, help="Prediction horizon")
    p.add_argument("--ma-window", type=int, default=200, help="Moving average window size")
    p.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of each patient series used for test")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of train+val used for val (temporal)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", required=True, help="Run ID (outputs/results/<run-id>)")
    p.add_argument("--output-root", default="outputs/results", help="Output root for metrics/preds")
    return p.parse_args()


def calculate_moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    window_size = int(window_size)
    if window_size <= 1:
        return data.copy()
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    return np.concatenate((np.zeros(window_size - 1), moving_avg), axis=0)


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps - pred_len + 1):
        X_seq.append(X[i : i + time_steps])
        y_temp = y[i + time_steps : i + time_steps + pred_len]
        y_seq.append(y_temp)
    return np.array(X_seq), np.array(y_seq)


def temporal_split(X: np.ndarray, y: np.ndarray, test_ratio: float, val_ratio: float):
    n = len(X)
    n_test = int(round(n * test_ratio))
    n_test = max(1, min(n - 2, n_test))
    split = n - n_test
    X_train_val, y_train_val = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    n_tv = len(X_train_val)
    n_val = int(round(n_tv * val_ratio))
    n_val = max(1, min(n_tv - 1, n_val))
    split_tv = n_tv - n_val
    X_train, y_train = X_train_val[:split_tv], y_train_val[:split_tv]
    X_val, y_val = X_train_val[split_tv:], y_train_val[split_tv:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_glucose_class(glucose: np.ndarray) -> np.ndarray:
    cls = np.zeros_like(glucose, dtype=np.float32)
    cls[glucose < 70.0] = 0.0
    cls[(glucose >= 70.0) & (glucose <= 180.0)] = 1.0
    cls[glucose > 180.0] = 2.0
    return cls


def load_patient_series(df: pd.DataFrame, ma_window: int) -> Tuple[np.ndarray, np.ndarray]:
    df = df.copy()
    df.replace(-1, np.nan, inplace=True)
    df = df.dropna(subset=["cgm"])
    df.fillna(0, inplace=True)
    df = df.sort_values("date").reset_index(drop=True)

    glucose = df["cgm"].to_numpy(dtype=np.float32)
    ma_200_full = calculate_moving_average(glucose, ma_window)
    keep_mask = ma_200_full != 0.0
    df = df[keep_mask].reset_index(drop=True)
    if df.empty:
        return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)

    basal = df.get("basal", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
    carbs = df.get("meal", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float32)
    bolus_cols = [c for c in df.columns if c in ("bolus", "bolus_short", "bolus_long")]
    if bolus_cols:
        bolus = df[bolus_cols].to_numpy(dtype=np.float32).sum(axis=1)
    else:
        bolus = np.zeros(len(df), dtype=np.float32)
    glucose = df["cgm"].to_numpy(dtype=np.float32)
    ma_200 = ma_200_full[keep_mask]
    glucose_class = compute_glucose_class(glucose)

    features = np.stack([basal, carbs, bolus, glucose, ma_200, glucose_class], axis=-1)
    return features, glucose


def standardize(train: np.ndarray, val: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (val - mean) / std, (test - mean) / std


def main():
    args = parse_args()
    np.random.seed(args.seed)

    root = Path(args.processed_root) / args.dataset / "timeseries.csv"
    if not root.exists():
        raise FileNotFoundError(root)
    df = pd.read_csv(root)
    if "pat_id" not in df.columns:
        if "pid" in df.columns:
            df = df.rename(columns={"pid": "pat_id"})
        else:
            raise KeyError("pat_id (or pid) not found in dataset")

    from models.glimmer import _add_glimmer_to_path, _require_tf

    _add_glimmer_to_path()
    tf = _require_tf()

    glimmer_root = Path(__file__).resolve().parents[1] / "third_party" / "glimmer_upstream"
    if str(glimmer_root) not in sys.path:
        sys.path.insert(0, str(glimmer_root))
    models_root = glimmer_root / "models"
    if str(models_root) not in sys.path:
        sys.path.insert(0, str(models_root))
    if args.variant == "transformer":
        from transformer_model import build_model as build_model  # type: ignore
    else:
        from cnn_lstm_model import build_model as build_model  # type: ignore

    # Loss function (same as upstream)
    def loss_fn(y_true, y_pred):
        mae = tf.keras.backend.abs(y_pred - y_true)
        normal_mask = tf.logical_and(y_true >= 70.0, y_true <= 180.0)
        hypo_mask = y_true < 70.0
        hyper_mask = y_true > 180.0
        normal_error = mae * tf.cast(normal_mask, tf.keras.backend.floatx()) * 1.0
        hypo_error = mae * tf.cast(hypo_mask, tf.keras.backend.floatx()) * 4.0
        hyper_error = mae * tf.cast(hyper_mask, tf.keras.backend.floatx()) * 2.5
        return tf.keras.backend.mean(normal_error + hypo_error + hyper_error)

    run_dir = Path(args.output_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rmse_list = []
    mae_list = []

    all_preds = []
    all_gts = []
    all_pat = []
    all_seq = []

    from metrics.clinical import MetricsAggregator
    agg = MetricsAggregator()
    seq_counter = 0

    for pid, df_p in df.groupby("pat_id"):
        features, glucose = load_patient_series(df_p, args.ma_window)
        if len(features) == 0:
            continue
        X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(
            features, glucose, args.test_ratio, args.val_ratio
        )

        X_train, X_val, X_test = standardize(X_train, X_val, X_test)
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, args.train_window, args.pred_len)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, args.train_window, args.pred_len)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, args.train_window, args.pred_len)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            continue

        y_pred = None
        for i in range(args.repeat):
            seed = int(i + 100)
            tf.random.set_seed(seed)
            np.random.seed(seed)

            model = build_model(
                input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                output_dim=args.pred_len,
                config=type(
                    "Cfg",
                    (),
                    {
                        "NUM_BLOCKS": 1,
                        "NUM_HEADS": 8,
                        "FF_DIM": 256,
                        "DROPOUT": 0.1,
                        "KEY_DIM": 16,
                        "ACTIVATION": "relu",
                    },
                )(),
            )

            model.compile(optimizer="adam", loss=loss_fn)
            model.fit(
                X_train_seq,
                y_train_seq,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_data=(X_val_seq, y_val_seq),
                verbose=1,
            )

            y_pred_temp = model.predict(X_test_seq)
            y_pred = y_pred_temp if y_pred is None else y_pred + y_pred_temp

        y_pred = y_pred / float(args.repeat)
        y_true = y_test_seq[:, 0]
        y_pred_1 = y_pred[:, 0]
        rmse = np.sqrt(np.mean((y_true - y_pred_1) ** 2))
        mae = np.mean(np.abs(y_true - y_pred_1))
        rmse_list.append(rmse)
        mae_list.append(mae)
        print(f"[{pid}] RMSE={rmse:.3f} MAE={mae:.3f}")

        # Metrics aggregator expects full horizon arrays
        agg.update(y_test_seq, y_pred, subject_id=[pid] * len(y_test_seq))

        all_preds.append(y_pred)
        all_gts.append(y_test_seq)
        all_pat.extend([pid] * len(y_test_seq))
        all_seq.extend(list(range(seq_counter, seq_counter + len(y_test_seq))))
        seq_counter += len(y_test_seq)

    if rmse_list:
        print(f"Mean RMSE: {np.mean(rmse_list):.3f} ± {np.std(rmse_list):.3f}")
        print(f"Mean MAE:  {np.mean(mae_list):.3f} ± {np.std(mae_list):.3f}")
    else:
        print("No valid patients found.")

    # Save outputs in benchmark format
    results = {
        "metrics": agg.compute(),
        "feature_cols": ["basal", "carbs", "bolus", "glucose", "glucose_MA_200", "glucose_class"],
        "event_conditioned": {
            "post_meal": {"threshold": 3, "metrics": {}, "count": 0},
            "post_bolus": {"threshold": 3, "metrics": {}, "count": 0},
            "nocturnal": {"threshold": 0, "metrics": {}, "count": 0},
        },
    }

    train_cfg = {
        "dataset": args.dataset,
        "variant": args.variant,
        "train_window": args.train_window,
        "pred_len": args.pred_len,
        "ma_window": args.ma_window,
        "test_ratio": args.test_ratio,
        "val_ratio": args.val_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "repeat": args.repeat,
        "seed": args.seed,
    }
    eval_cfg = dict(train_cfg)

    import json
    import yaml

    def _to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    jsonable = _to_jsonable({"train_config": train_cfg, "eval_config": eval_cfg, **results})
    (run_dir / "metrics.json").write_text(json.dumps(jsonable, indent=2))
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump({"train": train_cfg, "eval": eval_cfg, "model": "glimmer_paper"}, f)

    if all_preds:
        preds = np.concatenate(all_preds, axis=0)
        gts = np.concatenate(all_gts, axis=0)
        np.savez(run_dir / "preds.npz", preds=preds, pat_id=np.array(all_pat), seq_id=np.array(all_seq))
        np.savez(run_dir / "GT.npz", gts=gts, pat_id=np.array(all_pat), seq_id=np.array(all_seq))
        np.savez(
            run_dir / "event.npz",
            post_meal=np.full(len(all_pat), -1),
            post_bolus=np.full(len(all_pat), -1),
            nocturnal=np.full(len(all_pat), -1),
            pat_id=np.array(all_pat),
            seq_id=np.array(all_seq),
        )


if __name__ == "__main__":
    main()
