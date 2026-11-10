"""Adapters from supported dataset formats to the benchmark's tabular schema."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd


class TimeSeriesAdapter(Protocol):
    """Load a source into the columns consumed by ``TimeSeriesWindowDataset``."""

    def read(
        self,
        path: Path,
        *,
        patient_ids: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame: ...


def _canonical_id(value: Any) -> str:
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


@dataclass(frozen=True)
class CsvAdapter:
    """Read the benchmark's existing unified CSV representation."""

    def read(
        self,
        path: Path,
        *,
        patient_ids: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        path = Path(path)
        if path.is_dir():
            path = path / "timeseries.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        return pd.read_csv(path)


_TIMEZONE_SUFFIX = re.compile(r"\s+(?:UTC[+-]\d{2}:?\d{2}|Z)$", re.IGNORECASE)


def _parse_local_times(values: Iterable[Any]) -> pd.Series:
    """Parse DIAX local times while retaining their wall-clock hour.

    DIAX timestamps may end in ``UTC-05:00`` or ``Z``. The benchmark stores
    timezone-naive local timestamps, and its nocturnal slices use the local hour,
    so the suffix is intentionally removed rather than converted to UTC.
    """

    cleaned = pd.Series(list(values), dtype="object").map(
        lambda value: _TIMEZONE_SUFFIX.sub("", str(value).strip())
        if value is not None
        else value
    )
    return pd.to_datetime(cleaned, errors="coerce", format="mixed")


def _channel_frame(
    document: Mapping[str, Any],
    aliases: Sequence[str],
    value_name: str,
) -> pd.DataFrame:
    for key in aliases:
        channel = document.get(key)
        if not isinstance(channel, Mapping):
            continue
        times = channel.get("time")
        if times is None:
            times = channel.get("date")
        if times is None:
            times = []
        values = channel.get("value")
        if values is None:
            values = channel.get("values")
        if values is None:
            values = []
        if not isinstance(times, (list, tuple)):
            times = [times]
        if not isinstance(values, (list, tuple)):
            values = [values]
        count = min(len(times), len(values))
        if count == 0:
            continue
        frame = pd.DataFrame(
            {
                "date": _parse_local_times(times[:count]),
                value_name: pd.to_numeric(values[:count], errors="coerce"),
            }
        )
        return (
            frame.dropna(subset=["date", value_name])
            .sort_values("date")
            .reset_index(drop=True)
        )
    return pd.DataFrame(columns=["date", value_name])


def _deduplicate_cgm(frame: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.sort_values("date")
    delta = frame["date"].diff().dt.total_seconds()
    group = (delta.isna() | (delta > window_seconds)).cumsum()
    return frame.groupby(group, sort=False).tail(1).reset_index(drop=True)


def _resample_cgm(
    frame: pd.DataFrame,
    frequency_minutes: int,
    max_gap_minutes: int,
) -> list[pd.DataFrame]:
    if frame.empty:
        return []
    frame = frame.sort_values("date").reset_index(drop=True)
    segment = (frame["date"].diff() > pd.Timedelta(minutes=max_gap_minutes)).cumsum()
    output: list[pd.DataFrame] = []
    for _, part in frame.groupby(segment, sort=False):
        sampled = (
            part.set_index("date")[["cgm"]]
            .resample(f"{frequency_minutes}min")
            .mean()
        )
        sampled["cgm"] = sampled["cgm"].interpolate(method="time")
        sampled = sampled.dropna(subset=["cgm"]).reset_index()
        if not sampled.empty:
            output.append(sampled)
    return output


def _latest_values(
    grid: pd.DataFrame,
    observations: pd.DataFrame,
    value_col: str,
    lookback_minutes: int,
) -> np.ndarray:
    output = np.full(len(grid), np.nan, dtype=float)
    if grid.empty or observations.empty:
        return output
    observation_times = (
        observations["date"].to_numpy(dtype="datetime64[ns]").view("int64")
    )
    observation_values = observations[value_col].to_numpy(dtype=float)
    grid_times = grid["date"].to_numpy(dtype="datetime64[ns]").view("int64")
    positions = np.searchsorted(observation_times, grid_times, side="right") - 1
    valid = positions >= 0
    clipped = np.clip(positions, 0, len(observation_times) - 1)
    age = grid_times - observation_times[clipped]
    valid &= age <= pd.Timedelta(minutes=lookback_minutes).value
    output[valid] = observation_values[clipped[valid]]
    return output


def _event_sums(
    grid: pd.DataFrame,
    events: pd.DataFrame,
    value_col: str,
    frequency_minutes: int,
) -> np.ndarray:
    if grid.empty or events.empty:
        return np.zeros(len(grid), dtype=float)
    event_times = events["date"].to_numpy(dtype="datetime64[ns]").view("int64")
    event_values = events[value_col].to_numpy(dtype=float)
    grid_times = grid["date"].to_numpy(dtype="datetime64[ns]").view("int64")
    # Match the established preprocessing convention: each event belongs to one
    # right-closed sampling bin, with a 15-second tolerance after the grid point.
    upper = grid_times + pd.Timedelta(seconds=15).value
    lower = upper - pd.Timedelta(minutes=frequency_minutes).value
    starts = np.searchsorted(event_times, lower, side="right")
    ends = np.searchsorted(event_times, upper, side="right")
    cumulative = np.concatenate(([0.0], np.cumsum(event_values)))
    return cumulative[ends] - cumulative[starts]


def _weight_values(grid: pd.DataFrame, document: Mapping[str, Any]) -> np.ndarray:
    weight = document.get("weight")
    if isinstance(weight, (int, float)) and np.isfinite(weight):
        return np.full(len(grid), float(weight))
    observations = _channel_frame(document, ("weight",), "weight_kg")
    if observations.empty:
        if isinstance(weight, Mapping):
            raw_values = weight.get("value")
            values = pd.to_numeric([] if raw_values is None else raw_values, errors="coerce")
            values = np.atleast_1d(np.asarray(values, dtype=float))
            values = values[np.isfinite(values)]
            if values.size:
                return np.full(len(grid), float(values[-1]))
        return np.full(len(grid), np.nan)
    series = observations.set_index("date")["weight_kg"]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    index = pd.DatetimeIndex(grid["date"])
    expanded = (
        series.reindex(series.index.union(index))
        .interpolate(method="time")
        .ffill()
        .bfill()
    )
    return expanded.reindex(index).to_numpy(dtype=float)


def _subject_id(document: Mapping[str, Any], path: Path) -> str:
    metadata = document.get("metadata")
    subject_metadata = metadata.get("subject") if isinstance(metadata, Mapping) else {}
    if not isinstance(subject_metadata, Mapping):
        subject_metadata = {}
    candidates = (
        document.get("source_subject_id"),
        subject_metadata.get("source_subject_id"),
        document.get("subject_id"),
        document.get("unique_id"),
        path.stem,
    )
    return next(
        _canonical_id(value)
        for value in candidates
        if value is not None and str(value).strip()
    )


@dataclass(frozen=True)
class DiaxAdapter:
    """Normalize DIAX per-subject JSON files without creating an intermediate CSV."""

    frequency_minutes: int = 5
    max_cgm_gap_minutes: int = 30
    dedup_window_seconds: int = 15
    basal_lookback_minutes: int = 180
    cgm_min_mgdl: float = 20.0
    cgm_max_mgdl: float = 450.0
    basal_max_u_per_hr: float = 15.0
    bolus_max_u: float = 50.0

    def read(
        self,
        path: Path,
        *,
        patient_ids: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        path = Path(path)
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(path.glob("*.json"))
        else:
            raise FileNotFoundError(path)
        if not files:
            raise FileNotFoundError(f"No DIAX JSON files found in {path}")

        selected = (
            None
            if patient_ids is None
            else {_canonical_id(value) for value in patient_ids}
        )
        frames: list[pd.DataFrame] = []
        sequence_id = 1
        for json_path in files:
            try:
                document = json.loads(json_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read DIAX JSON {json_path}: {exc}") from exc
            if not isinstance(document, Mapping):
                raise ValueError(f"DIAX document must be a JSON object: {json_path}")

            patient_id = _subject_id(document, json_path)
            if selected is not None and patient_id not in selected:
                continue

            cgm = _channel_frame(document, ("cgm",), "cgm")
            cgm = cgm[cgm["cgm"].between(self.cgm_min_mgdl, self.cgm_max_mgdl)]
            cgm = _deduplicate_cgm(cgm, self.dedup_window_seconds)
            if cgm.empty:
                continue

            basal = _channel_frame(document, ("basal_rate", "basal"), "basal")
            basal = basal[basal["basal"].between(0.0, self.basal_max_u_per_hr)]
            bolus = _channel_frame(document, ("bolus", "bolus_insulin"), "bolus")
            bolus = bolus[bolus["bolus"].between(0.0, self.bolus_max_u)]
            meal = _channel_frame(
                document,
                ("meal", "carbs", "carb", "carbohydrate", "dietary_intake"),
                "meal",
            )

            for sampled in _resample_cgm(
                cgm,
                frequency_minutes=self.frequency_minutes,
                max_gap_minutes=self.max_cgm_gap_minutes,
            ):
                sampled["basal"] = _latest_values(
                    sampled, basal, "basal", self.basal_lookback_minutes
                )
                sampled["bolus_standard"] = _event_sums(
                    sampled, bolus, "bolus", self.frequency_minutes
                )
                sampled["bolus_extended"] = 0.0
                sampled["weight_kg"] = _weight_values(sampled, document)
                sampled["meal"] = _event_sums(
                    sampled, meal, "meal", self.frequency_minutes
                )
                sampled["pat_id"] = patient_id
                sampled["seq_id"] = sequence_id
                sequence_id += 1
                frames.append(sampled)

        if not frames:
            selection = "" if selected is None else f" for patient ids {sorted(selected)}"
            raise ValueError(f"No usable DIAX CGM series found in {path}{selection}")

        output = pd.concat(frames, ignore_index=True)
        output["basal"] = output["basal"].fillna(0.0)
        output = output[
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
            ]
        ].sort_values(["pat_id", "seq_id", "date"]).reset_index(drop=True)
        # Match CSV behavior so PyTorch's default collator receives strings,
        # not numpy.datetime64 objects.
        output["date"] = output["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return output


def infer_source_format(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() == ".csv":
            return "csv"
        if path.suffix.lower() == ".json":
            return "diax"
    if path.is_dir():
        if (path / "timeseries.csv").is_file():
            return "csv"
        if next(path.glob("*.json"), None) is not None:
            return "diax"
    raise ValueError(f"Could not infer dataset format from {path}")


def load_timeseries(
    path: Path,
    *,
    source_format: str = "auto",
    patient_ids: Optional[Sequence[Any]] = None,
    options: Optional[Mapping[str, Any]] = None,
) -> tuple[pd.DataFrame, str]:
    """Load CSV or DIAX data and return ``(dataframe, resolved_format)``."""

    path = Path(path)
    resolved = (
        infer_source_format(path)
        if source_format.lower() == "auto"
        else source_format.lower()
    )
    options = dict(options or {})
    if resolved == "csv":
        if options:
            raise ValueError(f"CSV adapter does not accept options: {sorted(options)}")
        adapter: TimeSeriesAdapter = CsvAdapter()
    elif resolved == "diax":
        adapter = DiaxAdapter(**options)
    else:
        raise ValueError(
            f"Unsupported source_format '{source_format}'; expected auto, csv, or diax"
        )
    return adapter.read(path, patient_ids=patient_ids), resolved


__all__ = [
    "CsvAdapter",
    "DiaxAdapter",
    "TimeSeriesAdapter",
    "infer_source_format",
    "load_timeseries",
]
