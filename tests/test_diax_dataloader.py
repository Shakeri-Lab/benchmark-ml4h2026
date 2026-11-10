from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data.adapters import DiaxAdapter, load_timeseries
from data.dataloader import TimeSeriesWindowDataset, make_dataloader_from_cfg


def _write_diax_subject(path: Path, subject_id: str = "alpha") -> None:
    times = pd.date_range("2025-01-01 00:00:00", periods=16, freq="5min")

    def timestamp(value: pd.Timestamp) -> str:
        return value.strftime("%Y-%m-%d %H:%M:%S UTC-05:00")

    document = {
        "unique_id": f"TEST_{subject_id}",
        "source_subject_id": subject_id,
        "subject_id": f"TEST_{subject_id}",
        "cgm": {
            "time": [timestamp(value) for value in times],
            "value": list(range(100, 116)),
        },
        "basal_rate": {
            "time": [timestamp(times[0]), timestamp(times[8])],
            "value": [0.8, 0.9],
        },
        "bolus": {
            "time": [timestamp(times[2] - pd.Timedelta(seconds=30))],
            "value": [2.5],
        },
        "carbs": {
            "time": [timestamp(times[2] - pd.Timedelta(seconds=30))],
            "value": [30.0],
        },
        "weight": {"date": [timestamp(times[0])], "value": [70.0]},
    }
    path.write_text(json.dumps(document))


def _write_split(path: Path, subject_id: str = "alpha") -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": "TEST",
                "patients": {
                    "train": [subject_id],
                    "val": [subject_id],
                    "test": [subject_id],
                },
            }
        )
    )


class DiaxDataloaderTests(unittest.TestCase):
    def test_diax_adapter_normalizes_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "diax"
            source.mkdir()
            _write_diax_subject(source / "subject.json")

            frame = DiaxAdapter().read(source)

        self.assertEqual(
            list(frame.columns),
            [
                "pat_id",
                "seq_id",
                "date",
                "cgm",
                "basal",
                "bolus_standard",
                "bolus_extended",
                "weight_kg",
                "meal",
            ],
        )
        self.assertEqual(frame["pat_id"].unique().tolist(), ["alpha"])
        self.assertEqual(len(frame), 16)
        self.assertTrue(np.isclose(frame["bolus_standard"].sum(), 2.5))
        self.assertTrue(np.isclose(frame["meal"].sum(), 30.0))
        self.assertTrue(frame["weight_kg"].eq(70.0).all())

    def test_window_dataset_reads_diax_and_batches_string_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "TEST"
            source.mkdir()
            _write_diax_subject(source / "subject.json")
            split_root = root / "splits"
            split_root.mkdir()
            _write_split(split_root / "test_split.json")

            dataset = TimeSeriesWindowDataset(
                split_name="test_split",
                split_part="train",
                split_root=split_root,
                processed_root=root,
                source_format="auto",
                history_len=4,
                pred_len=2,
                stride=1,
            )

            self.assertEqual(dataset.source_format, "diax")
            self.assertGreater(len(dataset), 0)
            features, target, patient_id, _, date, events, metadata = dataset[0]
            self.assertEqual(features.shape, (4, 6))
            self.assertEqual(target.shape, (2,))
            self.assertFalse(patient_id.dtype.is_floating_point)
            self.assertEqual(dataset.patient_id_map[int(patient_id)], "alpha")
            self.assertTrue(str(date).startswith("2025-01-01"))
            self.assertEqual(len(events), 3)
            self.assertEqual(metadata["feature_cols"][0], "cgm")

            loader = make_dataloader_from_cfg(
                {
                    "split_name": "test_split",
                    "split_root": split_root,
                    "processed_root": root,
                    "source_format": "diax",
                },
                {"history_len": 4, "pred_len": 2, "batch_size": 2},
                split_part="train",
                shuffle=False,
            )
            batch = next(iter(loader))
            self.assertEqual(batch[0].shape, (2, 4, 6))

    def test_csv_adapter_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "timeseries.csv"
            pd.DataFrame(
                {
                    "pat_id": [1],
                    "seq_id": [1],
                    "date": ["2025-01-01 00:00:00"],
                    "cgm": [100.0],
                }
            ).to_csv(path, index=False)

            frame, source_format = load_timeseries(path)

        self.assertEqual(source_format, "csv")
        self.assertEqual(frame.loc[0, "cgm"], 100.0)


if __name__ == "__main__":
    unittest.main()
