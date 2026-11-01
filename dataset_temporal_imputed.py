"""Hourly, within-meter windows. No CUDA or torch dependency for the data audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = "temporal_logmse_trainmedian_v2"
FEATURES = ["meter_reading", "air_temperature", "cloud_coverage", "dew_temperature"]
KEYS = ["building_id", "meter", "timestamp"]
DEFAULT_DATA_DIR = Path.home() / "s4convd-paper" / "pre_preprocessed_data"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TemporalDataset:
    """Duck-typed torch map dataset; default_collate converts returned numpy arrays.

    Each target has 48 immediately preceding hourly observations from the same
    building and meter. All 49 rows must have finite features and nonnegative
    meter readings. The first input is 48 hours before the target.
    """

    def __init__(self, frame, *, input_length=48, fraction=0.1,
                 sampling_seed=42, scaler=None, split="train"):
        if input_length < 1 or not 0 < fraction <= 1:
            raise ValueError("Require input_length >= 1 and 0 < fraction <= 1")
        if scaler is None and split != "train":
            raise ValueError("Validation/test must receive the training scaler")
        frame = frame[KEYS + FEATURES].copy()
        source_rows = len(frame)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        if frame[KEYS].isna().any().any():
            raise ValueError(f"{split}: missing series key or timestamp")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{split}: duplicate building/meter/timestamp keys; audit source first")
        # Retain source CSV row numbers so every selected target is traceable.
        frame["source_row"] = np.arange(source_rows, dtype=np.int64)
        date_min, date_max = frame.timestamp.min(), frame.timestamp.max()
        values = frame[FEATURES].to_numpy(dtype=np.float64, copy=True)
        valid_energy = np.isfinite(values[:, 0]) & (values[:, 0] >= 0)
        weather = values[:, 1:].copy()
        missing_weather = ~np.isfinite(weather)
        imputed_counts = {
            name: int(missing_weather[:, j].sum())
            for j, name in enumerate(FEATURES[1:])
        }

        if scaler is None:
            training_weather = weather[valid_energy].copy()
            training_weather[~np.isfinite(training_weather)] = np.nan
            medians = np.nanmedian(training_weather, axis=0)
        else:
            medians = np.asarray(
                scaler["weather_imputation_medians"], dtype=np.float64
            )

        if medians.shape != (3,) or not np.isfinite(medians).all():
            raise ValueError("Cannot derive finite training weather medians")

        for j in range(3):
            weather[missing_weather[:, j], j] = medians[j]

        frame[FEATURES[1:]] = weather
        valid = valid_energy
        frame = frame.loc[valid].sort_values(KEYS, kind="stable").reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"{split}: no valid rows")
        values = frame[FEATURES].to_numpy(dtype=np.float64, copy=True)
        # Predict log1p(raw energy); use the same transform for the energy input.
        values[:, 0] = np.log1p(values[:, 0])
        self.log_readings = values[:, 0].copy()
        if scaler is None:
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=0)
            std[std == 0] = 1.0
            scaler = {"mean": mean.tolist(), "std": std.tolist(),
                      "features": FEATURES, "meter_transform": "log1p",
                      "weather_imputation_medians": medians.tolist()}
        if scaler["features"] != FEATURES or scaler["meter_transform"] != "log1p":
            raise ValueError("Incompatible scaler")
        mean, std = np.asarray(scaler["mean"]), np.asarray(scaler["std"])
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid training scaler")
        self.scaler = scaler
        self.inputs = np.asarray((values - mean) / std, dtype=np.float32)
        if not np.isfinite(self.inputs).all():
            raise ValueError("Nonfinite normalized inputs")
        self.input_length = input_length
        # A boundary at row i invalidates the edge from i-1 to i.
        times = frame.timestamp.to_numpy(dtype="datetime64[ns]")
        same_series = (frame.building_id.to_numpy()[1:] == frame.building_id.to_numpy()[:-1])
        same_series &= (frame.meter.to_numpy()[1:] == frame.meter.to_numpy()[:-1])
        consecutive = np.diff(times) == np.timedelta64(1, "h")
        boundaries = np.r_[True, ~(same_series & consecutive)]
        cumulative = np.cumsum(boundaries, dtype=np.int64)
        candidates = np.arange(input_length, len(frame), dtype=np.int64)
        candidates = candidates[cumulative[candidates] == cumulative[candidates - input_length]]
        count = len(candidates)
        if not count:
            raise ValueError(f"{split}: no complete hourly windows of length {input_length}+1")
        sample_count = max(1, int(round(count * fraction)))
        # Random selection of complete windows; chronological/group order retained.
        rng = np.random.default_rng(sampling_seed)
        if sample_count < count:
            candidates = np.sort(rng.choice(candidates, sample_count, replace=False))
        self.target_indices = candidates
        self.target_source_rows = frame.source_row.to_numpy()[candidates].copy()
        self.audit = {
            "split": split, "source_rows": source_rows, "valid_rows": len(frame),
            "dropped_invalid_rows": source_rows - len(frame),
            "weather_imputed_cells_by_column": imputed_counts,
            "weather_imputation": "training_feature_median",
            "date_min": str(date_min), "date_max": str(date_max),
            "series_count": len(frame[["building_id", "meter"]].drop_duplicates()),
            "valid_windows": count, "selected_windows": len(candidates),
            "selected_target_rows_sha256": hashlib.sha256(
                self.target_source_rows.astype("<i8").tobytes()).hexdigest(),
        }

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, index):
        i = self.target_indices[index]
        return {"input_sequence": self.inputs[i-self.input_length:i],
                "label_sequence": np.float32(self.log_readings[i])}

    def evaluation_targets(self):
        return self.log_readings[self.target_indices]

    def persistence_predictions(self):
        return self.log_readings[self.target_indices - 1]


def forecast_metrics(predicted_log, target_log, persistence_log):
    """RMSLE on raw nonnegative energy, evaluated stably in log space.

    In energy units prediction = expm1(max(predicted_log, 0)). No clipping is
    applied to predictions during the training MSE loss.
    """
    prediction, target, persistence = [np.asarray(a, dtype=np.float64).reshape(-1)
                                       for a in (predicted_log, target_log, persistence_log)]
    if not len(target) or not (prediction.shape == target.shape == persistence.shape):
        raise ValueError("Metric arrays must have identical nonempty shapes")
    if not all(np.isfinite(a).all() for a in (prediction, target, persistence)):
        raise ValueError("Nonfinite prediction or target; run is invalid")
    if (target < 0).any() or (persistence < 0).any():
        raise ValueError("Targets and persistence must be log1p of nonnegative energy")
    return {
        "test_rmsle": float(np.sqrt(np.mean((np.maximum(prediction, 0) - target)**2))),
        "zero_baseline_rmsle": float(np.sqrt(np.mean(target**2))),
        "persistence_baseline_rmsle": float(np.sqrt(np.mean((persistence-target)**2))),
        "prediction_nonpositive_fraction": float(np.mean(prediction <= 0)),
        "test_samples": len(target),
    }


def build_datasets(data_dir=DEFAULT_DATA_DIR, fraction=0.1, input_length=48):
    datasets, manifest = {}, {"protocol": PROTOCOL, "sampling_seed": 42,
        "sampling_unit": "complete_hourly_window", "fraction": fraction,
        "input_length": input_length, "normalization_fit": "all_valid_training_rows",
        "split_policy": "strictly_disjoint_global_time_ranges; no cross-split history",
        "splits": {}}
    scaler = None
    previous_max = None
    for split in ("train", "val", "test"):
        path = Path(data_dir) / f"{split}.csv"
        print(f"Preparing {split}: {path}", flush=True)
        frame = pd.read_csv(path, usecols=KEYS + FEATURES)
        dataset = TemporalDataset(frame, input_length=input_length, fraction=fraction,
                                  scaler=scaler, split=split)
        current_min = pd.Timestamp(dataset.audit["date_min"])
        current_max = pd.Timestamp(dataset.audit["date_max"])
        if previous_max is not None and current_min <= previous_max:
            raise ValueError(f"{split}: time range overlaps or precedes the previous split")
        previous_max = current_max
        if split == "train":
            scaler = dataset.scaler
        datasets[split] = dataset
        manifest["splits"][split] = {**dataset.audit, "file": str(path.resolve()),
                                      "file_sha256": sha256_file(path)}
        del frame
    manifest["scaler"] = scaler
    return datasets, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16384)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    datasets, manifest = build_datasets(args.data_dir, args.fraction)
    manifest["training_batches_drop_last"] = len(datasets["train"]) // args.batch_size
    for split in ("val", "test"):
        dataset = datasets[split]
        manifest["splits"][split]["baselines"] = forecast_metrics(
            np.zeros(len(dataset)), dataset.evaluation_targets(),
            dataset.persistence_predictions())
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
